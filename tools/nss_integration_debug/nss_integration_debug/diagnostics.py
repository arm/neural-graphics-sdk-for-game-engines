# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import math
import re
import shutil
import struct
import tempfile
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import OpenEXR

from dataset_validator import ProfileError, ValidationError, load_profile, validate_dataset
from dataset_validator.exr_reader import ExrReadError, MAX_IMAGE_PIXELS, exr_dimensions


TOOL_VERSION = "0.3.0"
REPORT_SCHEMA_VERSION = 3
PROFILE_NAME = "nss-replay-v1"
SUPPORTED_PROFILES = (PROFILE_NAME,)
SUPPORTED_DIAGNOSTICS = ("jitter", "histogram", "motion")
AUTO_FRAME_LIMIT = 8
MAX_SELECTED_FRAMES = 256
HISTOGRAM_BINS = 64
FRAME_CACHE_LIMIT = 2
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360
POLARITY_IMPROVEMENT_RATIO = 0.75
FLOAT32_MAX = float(np.finfo(np.float32).max)
MAX_FP16 = 65504.0
SEQUENCE_PATTERN = re.compile(r"[0-9]+")
SCALE_PATTERN = re.compile(r"x[0-9]+(?:_[0-9]+)?")


class DiagnosticError(ValueError):
    pass


@dataclass(frozen=True)
class _FrameState:
    frame: int
    exposure: float
    exposure_source: str
    jitter_source: str
    jitter_pixels: tuple[float, float]
    raw_color: np.ndarray
    processed_color: np.ndarray
    unjittered_color: np.ndarray


@dataclass(frozen=True)
class _MotionCandidate:
    warped: np.ndarray
    error: np.ndarray
    valid: np.ndarray


def _parse_positive_float(value: Any, description: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DiagnosticError(f"{description}: {value!r}")
    try:
        parsed = float(value)
    except (OverflowError, ValueError) as error:
        raise DiagnosticError(f"{description}: {value!r}") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise DiagnosticError(f"{description}: {value!r}")
    return parsed


def _resolve_input_path(
    root: Path,
    path: Path,
    description: str,
    *,
    directory: bool = False,
    require_file: bool = True,
) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise DiagnosticError(f"Cannot resolve {description} {path}: {error}") from error
    if resolved != root and root not in resolved.parents:
        raise DiagnosticError(f"{description.capitalize()} escapes the dataset root: {path}")
    if directory and not resolved.is_dir():
        raise DiagnosticError(f"{description.capitalize()} is not a directory: {resolved}")
    if not directory and require_file and not resolved.is_file():
        raise DiagnosticError(f"{description.capitalize()} is not a file: {resolved}")
    return resolved


class _NssDataset:
    def __init__(
        self,
        root: Path,
        sequence: str,
        scale: str,
        profile: dict[str, Any],
        exposure_override: float | None,
    ):
        self.root = root
        self.sequence = sequence
        self.scale = scale
        self.profile = profile
        self.jitter_keys = tuple(profile["jitter"]["keys"])
        self.exposure_field = str(profile["exposure"]["field"])
        depth_profile = profile["lanes"]["depth"]
        self.depth_channels = tuple(depth_profile["channels"])
        self.depth_channel_aliases = tuple(depth_profile.get("channel_aliases", ()))
        self.metadata_path = _resolve_input_path(
            root,
            root / f"{sequence}.json",
            "sequence metadata",
        )
        self.metadata = self._read_json(self.metadata_path, "sequence metadata")
        self.frames = self._index_frames()
        self.ratio_index = self._ratio_index()
        self.color_directory = self._resolve_lane_directory(
            profile["lanes"]["color"]["paths"],
            "color",
        )
        self.depth_directory = self._resolve_lane_directory(
            profile["lanes"]["depth"]["paths"],
            "depth",
        )
        self.motion_directory = self._resolve_lane_directory(
            profile["lanes"]["motion"]["paths"],
            "motion",
        )
        self.color_files = self._index_exr_files(self.color_directory)
        self.depth_files = self._index_exr_files(self.depth_directory)
        self.motion_files = self._index_exr_files(self.motion_directory)
        self.exposure_override = exposure_override
        self._frame_cache: OrderedDict[int, _FrameState] = OrderedDict()

    @staticmethod
    def _read_json(path: Path, description: str) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise DiagnosticError(f"Missing {description}: {path}") from error
        except UnicodeDecodeError as error:
            raise DiagnosticError(
                f"{description.capitalize()} is not valid UTF-8: {path}: byte {error.start}"
            ) from error
        except OSError as error:
            raise DiagnosticError(f"Cannot read {description} {path}: {error}") from error
        except json.JSONDecodeError as error:
            raise DiagnosticError(f"Invalid JSON in {description} {path}: {error}") from error
        if not isinstance(payload, dict):
            raise DiagnosticError(f"{description.capitalize()} must contain a JSON object: {path}")
        return payload

    def _index_frames(self) -> dict[int, dict[str, Any]]:
        frames = self.metadata.get("Frames")
        if not isinstance(frames, list):
            raise DiagnosticError(f"Frames array is missing from {self.metadata_path}")
        indexed: dict[int, dict[str, Any]] = {}
        for entry in frames:
            if not isinstance(entry, dict):
                continue
            frame = entry.get("Frame", entry.get("FrameNumber"))
            if isinstance(frame, int) and not isinstance(frame, bool):
                indexed[frame] = entry
        if not indexed:
            raise DiagnosticError(f"No indexed frames were found in {self.metadata_path}")
        return indexed

    def _ratio_index(self) -> int:
        indices = self.metadata.get("UpscalingRatiosIndices")
        ratio_key = f"{self.scale}_index"
        if not isinstance(indices, dict) or not isinstance(indices.get(ratio_key), int):
            raise DiagnosticError(f"Missing UpscalingRatiosIndices.{ratio_key} in {self.metadata_path}")
        ratio_index = indices[ratio_key]
        if ratio_index < 0:
            raise DiagnosticError(f"UpscalingRatiosIndices.{ratio_key} must be non-negative")
        return ratio_index

    def _resolve_lane_directory(self, templates: Sequence[str], kind: str) -> Path:
        candidates = [
            self.root / template.format(scale=self.scale, sequence=self.sequence)
            for template in templates
        ]
        invalid_candidates: list[DiagnosticError] = []
        for candidate in candidates:
            if candidate.exists() or candidate.is_symlink():
                try:
                    return _resolve_input_path(
                        self.root,
                        candidate,
                        f"{kind} directory",
                        directory=True,
                    )
                except DiagnosticError as error:
                    invalid_candidates.append(error)
        if invalid_candidates:
            raise invalid_candidates[0]
        expected = ", ".join(str(candidate) for candidate in candidates)
        raise DiagnosticError(f"Missing {kind} directory; expected one of: {expected}")

    def _index_exr_files(self, directory: Path) -> dict[int, Path]:
        indexed: dict[int, Path] = {}
        for path in directory.iterdir():
            if path.suffix.lower() != ".exr" or not path.stem.isdigit():
                continue
            resolved = _resolve_input_path(
                self.root,
                path,
                "OpenEXR input",
                require_file=False,
            )
            if resolved.is_file():
                indexed[int(path.stem)] = resolved
        return indexed

    @staticmethod
    def _read_exposure_override(root: Path) -> float | None:
        candidate = root / "config.json"
        if not candidate.exists() and not candidate.is_symlink():
            return None
        config_path = _resolve_input_path(root, candidate, "dataset config")
        config = _NssDataset._read_json(config_path, "dataset config")
        if "Exposure" not in config:
            return None
        exposure = config["Exposure"]
        return _parse_positive_float(
            exposure,
            "config.json Exposure must be a finite value greater than zero",
        )

    @property
    def available_frames(self) -> list[int]:
        return sorted(self.frames)

    def frame_state(self, frame: int) -> _FrameState:
        if frame in self._frame_cache:
            self._frame_cache.move_to_end(frame)
            return self._frame_cache[frame]
        metadata = self.frames.get(frame)
        if metadata is None:
            raise DiagnosticError(f"Frame {frame} is not present in sequence {self.sequence}")

        color_path = self.color_files.get(frame)
        if color_path is None:
            raise DiagnosticError(f"Color input is missing for frame {frame}")
        raw_color = _read_exr_channels(color_path, ("R", "G", "B"))
        height, width, _ = raw_color.shape
        exposure = self.exposure_override
        if exposure is None:
            exposure_source = "sequence metadata"
            exposure_value = metadata.get(self.exposure_field)
            exposure = _parse_positive_float(
                exposure_value,
                f"Frame {frame} has an invalid Exposure value",
            )
        else:
            exposure_source = "config.json"

        jitter_source, jitter_pixels = self._jitter_pixels(metadata, width, height)
        processed_color = _karis_tonemap(raw_color, exposure)
        unjittered_raw_color = _shift_image(
            raw_color,
            source_offset_x=-jitter_pixels[0],
            source_offset_y=-jitter_pixels[1],
        )
        unjittered_color = _karis_tonemap(unjittered_raw_color, exposure)
        state = _FrameState(
            frame=frame,
            exposure=exposure,
            exposure_source=exposure_source,
            jitter_source=jitter_source,
            jitter_pixels=jitter_pixels,
            raw_color=raw_color,
            processed_color=processed_color,
            unjittered_color=unjittered_color,
        )
        self._frame_cache[frame] = state
        self._frame_cache.move_to_end(frame)
        if len(self._frame_cache) > FRAME_CACHE_LIMIT:
            self._frame_cache.popitem(last=False)
        return state

    def _jitter_pixels(
        self,
        metadata: dict[str, Any],
        width: int,
        height: int,
    ) -> tuple[str, tuple[float, float]]:
        for key in self.jitter_keys:
            values = metadata.get(key)
            if not isinstance(values, list) or self.ratio_index >= len(values):
                continue
            value = values[self.ratio_index]
            if not isinstance(value, dict):
                continue
            x = value.get("X")
            y = value.get("Y")
            if all(
                isinstance(component, (int, float))
                and not isinstance(component, bool)
                and math.isfinite(float(component))
                for component in (x, y)
            ):
                return key, (float(x) * width, float(y) * height)
        raise DiagnosticError(
            f"Frame {metadata.get('Frame', metadata.get('FrameNumber'))} "
            f"has no finite jitter entry for scale {self.scale}"
        )

    def motion(self, frame: int) -> np.ndarray:
        motion_path = self.motion_files.get(frame)
        if motion_path is None:
            raise DiagnosticError(f"Motion input is missing for frame {frame}")
        return _read_exr_channels(motion_path, ("R", "G"))

    def motion_vector_scale(
        self,
        frame: int,
        width: int,
        height: int,
    ) -> tuple[str, tuple[float, float]]:
        metadata = self.frames.get(frame)
        if metadata is None:
            raise DiagnosticError(f"Frame {frame} is not present in sequence {self.sequence}")
        if "MotionVectorScale" not in metadata:
            return "default_negative_dimensions", (-float(width), -float(height))
        values = metadata["MotionVectorScale"]
        if (
            not isinstance(values, list)
            or len(values) != 2
        ):
            raise DiagnosticError(
                f"Frame {frame} has an invalid MotionVectorScale value: {values!r}"
            )
        converted: list[float] = []
        for value in values:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                raise DiagnosticError(
                    f"Frame {frame} has an invalid MotionVectorScale value: {values!r}"
                )
            try:
                converted_value = float(value)
            except (OverflowError, ValueError) as error:
                raise DiagnosticError(
                    f"Frame {frame} has an invalid MotionVectorScale value: {values!r}"
                ) from error
            if not math.isfinite(converted_value) or abs(converted_value) > FLOAT32_MAX:
                raise DiagnosticError(
                    f"Frame {frame} MotionVectorScale values must be finite and representable "
                    f"as float32: {values!r}"
                )
            converted.append(converted_value)
        return "metadata", (converted[0], converted[1])

    def depth(self, frame: int) -> np.ndarray:
        depth_path = self.depth_files.get(frame)
        if depth_path is None:
            raise DiagnosticError(f"Depth input is missing for frame {frame}")
        return _read_exr_channels(
            depth_path,
            self.depth_channels,
            single_channel_aliases=self.depth_channel_aliases,
        )


def _read_exr_channels(
    path: Path,
    channel_names: Sequence[str],
    *,
    single_channel_aliases: Sequence[str] = (),
) -> np.ndarray:
    try:
        with OpenEXR.File(str(path), separate_channels=True) as image:
            try:
                header_width, header_height = exr_dimensions(image)
            except ExrReadError as error:
                raise DiagnosticError(str(error)) from error
            if header_width > MAX_IMAGE_PIXELS // header_height:
                raise DiagnosticError(
                    f"OpenEXR dimensions are unsupported for diagnostics: "
                    f"{header_width}x{header_height}: {path}"
                )
            channels = image.channels()
            selected_names = tuple(channel_names)
            missing = [name for name in selected_names if name not in channels]
            if missing and len(selected_names) == 1:
                selected_alias = next(
                    (
                        name
                        for name in sorted(set(single_channel_aliases))
                        if name in channels
                    ),
                    None,
                )
                if selected_alias is not None:
                    selected_names = (selected_alias,)
                    missing = []
            if missing:
                raise DiagnosticError(f"OpenEXR file {path} is missing channels: {', '.join(missing)}")
            arrays = [np.asarray(channels[name].pixels, dtype=np.float32) for name in selected_names]
    except DiagnosticError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise DiagnosticError(f"Cannot read OpenEXR file {path}: {error}") from error

    shapes = {array.shape for array in arrays}
    if len(shapes) != 1 or any(array.ndim != 2 for array in arrays):
        raise DiagnosticError(f"OpenEXR channels have inconsistent two-dimensional shapes: {path}")
    height, width = arrays[0].shape
    if (width, height) != (header_width, header_height):
        raise DiagnosticError(
            f"OpenEXR channel dimensions do not match the image data window: "
            f"{width}x{height} versus {header_width}x{header_height}: {path}"
        )
    if height <= 0 or width <= 0 or width > MAX_IMAGE_PIXELS // height:
        raise DiagnosticError(f"OpenEXR dimensions are unsupported for diagnostics: {width}x{height}: {path}")
    return np.stack(arrays, axis=-1)


def _karis_tonemap(color: np.ndarray, exposure: float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        half_color = color.astype(np.float16, copy=False)
        half_exposure = np.asarray(
            min(max(float(exposure), 0.0), MAX_FP16),
            dtype=np.float16,
        )
        scaled = half_color * half_exposure
    safe = np.nan_to_num(
        scaled,
        nan=0.0,
        posinf=MAX_FP16,
        neginf=0.0,
    )
    safe = np.clip(safe, 0.0, MAX_FP16).astype(np.float16, copy=False)
    maximum = np.max(safe.astype(np.float32), axis=2, keepdims=True)
    return (
        (safe.astype(np.float32) / (1.0 + maximum))
        .astype(np.float16)
        .astype(np.float32)
    )


def _sample_bilinear(image: np.ndarray, source_x: np.ndarray, source_y: np.ndarray) -> np.ndarray:
    height, width, _ = image.shape
    clipped_x = np.clip(source_x, 0.0, float(width - 1))
    clipped_y = np.clip(source_y, 0.0, float(height - 1))
    x0 = np.floor(clipped_x).astype(np.int64)
    y0 = np.floor(clipped_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    weight_x = (clipped_x - x0)[..., None]
    weight_y = (clipped_y - y0)[..., None]

    top = image[y0, x0] * (1.0 - weight_x) + image[y0, x1] * weight_x
    bottom = image[y1, x0] * (1.0 - weight_x) + image[y1, x1] * weight_x
    return top * (1.0 - weight_y) + bottom * weight_y


def _shift_image(image: np.ndarray, source_offset_x: float, source_offset_y: float) -> np.ndarray:
    height, width, _ = image.shape
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    return _sample_bilinear(
        image,
        grid_x + np.float32(source_offset_x),
        grid_y + np.float32(source_offset_y),
    )


def _to_display_rgb(color: np.ndarray) -> np.ndarray:
    linear = np.clip(color, 0.0, 1.0)
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _preview_image(array: np.ndarray) -> np.ndarray:
    height, width, _ = array.shape
    scale = min(PREVIEW_WIDTH / width, PREVIEW_HEIGHT / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    source_x = np.linspace(0.0, width - 1, resized_width, dtype=np.float32)
    source_y = np.linspace(0.0, height - 1, resized_height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(source_x, source_y)
    resized = _sample_bilinear(array.astype(np.float32) / 255.0, grid_x, grid_y)
    resized_u8 = np.rint(np.clip(resized, 0.0, 1.0) * 255.0).astype(np.uint8)
    preview = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
    offset_x = (PREVIEW_WIDTH - resized_width) // 2
    offset_y = (PREVIEW_HEIGHT - resized_height) // 2
    preview[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized_u8
    return preview


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    payload = chunk_type + data
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", checksum)


def _write_png(path: Path, image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise DiagnosticError("PNG output must be an HxWx3 uint8 array.")
    height, width, _ = image.shape
    scanlines = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _save_panels(panels: Sequence[tuple[str, np.ndarray]], path: Path) -> None:
    images = [_preview_image(panel) for _, panel in panels]
    separator = np.full((PREVIEW_HEIGHT, 4, 3), 32, dtype=np.uint8)
    combined: list[np.ndarray] = []
    for index, image in enumerate(images):
        if index:
            combined.append(separator)
        combined.append(image)
    _write_png(path, np.concatenate(combined, axis=1))


def _histogram_metrics(state: _FrameState) -> tuple[np.ndarray, dict[str, Any]]:
    finite_pixels = np.all(np.isfinite(state.raw_color), axis=2)
    negative_pixels = finite_pixels & np.any(state.raw_color < 0.0, axis=2)
    luminance = np.dot(
        state.unjittered_color,
        np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32),
    )
    histogram, _ = np.histogram(luminance, bins=HISTOGRAM_BINS, range=(0.0, 1.0))
    dark_pixels = int(np.count_nonzero(luminance < 0.1))
    bright_pixels = int(np.count_nonzero(luminance > 0.9))
    percentiles = (
        np.percentile(luminance, (1.0, 50.0, 99.0)).tolist()
        if luminance.size
        else [None, None, None]
    )
    metrics = {
        "sample_count": int(luminance.size),
        "non_finite_pixels": int(finite_pixels.size - finite_pixels.sum()),
        "negative_input_pixels": int(negative_pixels.sum()),
        "dark_pixels": dark_pixels,
        "bright_pixels": bright_pixels,
        "dark_fraction": float(dark_pixels / luminance.size),
        "bright_fraction": float(bright_pixels / luminance.size),
        "tonemapped_out_of_range_pixels": int(
            np.count_nonzero((luminance < 0.0) | (luminance > 1.0))
        ),
        "luminance_percentiles": {
            "p01": percentiles[0],
            "p50": percentiles[1],
            "p99": percentiles[2],
        },
        "bins": histogram.astype(int).tolist(),
        "interpretation": (
            "Karis-tonemapped input approximation before model quantization; "
            "this is not a decoded runtime tensor."
        ),
    }
    return histogram, metrics


def _histogram_chart(histogram: np.ndarray) -> np.ndarray:
    chart = np.full((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), (18, 18, 22), dtype=np.uint8)
    left = 36
    top = 18
    right = PREVIEW_WIDTH - 18
    bottom = PREVIEW_HEIGHT - 18
    chart[top:bottom + 1, left] = (170, 170, 170)
    chart[bottom, left:right + 1] = (170, 170, 170)
    maximum = max(int(histogram.max()), 1)
    chart_height = bottom - top
    bin_width = (right - left) / HISTOGRAM_BINS
    for index, count in enumerate(histogram):
        x0 = int(round(left + index * bin_width))
        x1 = max(x0 + 1, int(round(left + (index + 1) * bin_width)))
        bar_height = int(round(chart_height * int(count) / maximum))
        if bar_height > 0:
            chart[bottom - bar_height:bottom, x0:x1 + 1] = (52, 205, 220)
    return chart


def _save_histogram(
    state: _FrameState,
    histogram: np.ndarray,
    path: Path,
) -> None:
    preview = _preview_image(_to_display_rgb(state.unjittered_color))
    chart = _histogram_chart(histogram)
    separator = np.full((PREVIEW_HEIGHT, 4, 3), 32, dtype=np.uint8)
    _write_png(path, np.concatenate((preview, separator, chart), axis=1))


def _estimate_translation(reference: np.ndarray, current: np.ndarray) -> dict[str, float] | None:
    reference_luma = np.dot(
        reference,
        np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32),
    )
    current_luma = np.dot(
        current,
        np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32),
    )
    reference_luma = reference_luma - float(np.mean(reference_luma))
    current_luma = current_luma - float(np.mean(current_luma))
    if np.linalg.norm(reference_luma) <= 1e-8 or np.linalg.norm(current_luma) <= 1e-8:
        return None

    cross_power = np.fft.fft2(reference_luma) * np.conj(np.fft.fft2(current_luma))
    magnitude = np.abs(cross_power)
    cross_power = np.divide(
        cross_power,
        magnitude,
        out=np.zeros_like(cross_power),
        where=magnitude > 1e-12,
    )
    correlation = np.abs(np.fft.ifft2(cross_power))
    peak_y, peak_x = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    height, width = correlation.shape
    shift_x = float(peak_x - width if peak_x > width // 2 else peak_x)
    shift_y = float(peak_y - height if peak_y > height // 2 else peak_y)
    return {
        "x": shift_x,
        "y": shift_y,
        "magnitude": math.hypot(shift_x, shift_y),
    }


def _motion_coordinates(
    shape: tuple[int, int],
    motion: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    finite_motion = np.all(np.isfinite(motion), axis=2)
    source_x = grid_x + np.float32(scale_x) * motion[..., 0]
    source_y = grid_y + np.float32(scale_y) * motion[..., 1]
    valid = (
        finite_motion
        & (source_x >= 0.0)
        & (source_x <= width - 1)
        & (source_y >= 0.0)
        & (source_y <= height - 1)
    )
    return source_x, source_y, valid


def _motion_candidate(
    current: np.ndarray,
    previous: np.ndarray,
    motion: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> _MotionCandidate:
    height, width, _ = current.shape
    source_x, source_y, valid = _motion_coordinates(
        (height, width),
        motion,
        scale_x,
        scale_y,
    )
    warped = _sample_bilinear(previous, source_x, source_y)
    error = np.mean(np.square(warped - current), axis=2)
    return _MotionCandidate(warped=warped, error=error, valid=valid)


def _mean_masked_error(error: np.ndarray, valid: np.ndarray) -> float | None:
    return float(np.mean(error[valid])) if np.any(valid) else None


def _depth_consistency_mask(
    current_depth: np.ndarray,
    previous_depth: np.ndarray,
    motion: np.ndarray,
    motion_vector_scale: tuple[float, float],
    geometric_valid: np.ndarray,
) -> tuple[np.ndarray, float | None]:
    height, width, _ = current_depth.shape
    source_x, source_y, _ = _motion_coordinates(
        (height, width),
        motion,
        motion_vector_scale[0],
        motion_vector_scale[1],
    )
    warped_previous_depth = _sample_bilinear(previous_depth, source_x, source_y)[..., 0]
    current_values = current_depth[..., 0]
    finite_depth = np.isfinite(current_values) & np.isfinite(warped_previous_depth)
    candidate_mask = geometric_valid & finite_depth
    if not np.any(candidate_mask):
        return candidate_mask, None

    difference = np.abs(warped_previous_depth - current_values)
    candidate_differences = difference[candidate_mask]
    median = float(np.median(candidate_differences))
    median_absolute_deviation = float(
        np.median(np.abs(candidate_differences - median))
    )
    threshold = max(median + 6.0 * median_absolute_deviation, 1e-6)
    return candidate_mask & (difference <= threshold), threshold


def _error_heatmap(error: np.ndarray, valid: np.ndarray) -> np.ndarray:
    valid_error = error[valid]
    scale = float(np.percentile(valid_error, 95.0)) if valid_error.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    normalized = np.clip(error / scale, 0.0, 1.0)
    red = np.clip(normalized * 3.0, 0.0, 1.0)
    green = np.clip(normalized * 3.0 - 1.0, 0.0, 1.0)
    blue = np.clip(normalized * 3.0 - 2.0, 0.0, 1.0)
    heatmap = np.stack((red, green, blue), axis=2)
    heatmap[~valid] = 0.0
    return np.rint(heatmap * 255.0).astype(np.uint8)


def _frame_selector(selector: str, available: Sequence[int]) -> list[int]:
    if not available:
        raise DiagnosticError("The selected sequence contains no frames.")
    if selector == "all":
        if len(available) > MAX_SELECTED_FRAMES:
            raise DiagnosticError(
                f"At most {MAX_SELECTED_FRAMES} frames may be selected per run; "
                "use separate bounded selections for larger captures."
            )
        return list(available)
    if selector == "auto":
        if len(available) <= AUTO_FRAME_LIMIT:
            return list(available)
        indices = np.linspace(0, len(available) - 1, AUTO_FRAME_LIMIT)
        return sorted({available[int(round(index))] for index in indices})

    available_min = min(available)
    available_max = max(available)
    requested: set[int] = set()

    def parse_number(value: str, item: str) -> int:
        try:
            return int(value)
        except (OverflowError, ValueError) as error:
            raise DiagnosticError(
                f"Frame selector item is too large to parse: {item!r}"
            ) from error

    for part in selector.split(","):
        item = part.strip()
        if not item:
            raise DiagnosticError(f"Invalid empty item in --frames {selector!r}")
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                raise DiagnosticError(f"Invalid frame range {item!r}")
            start = parse_number(start_text, item)
            end = parse_number(end_text, item)
            if end < start:
                raise DiagnosticError(f"Frame range must be ascending: {item!r}")
            range_size = end - start + 1
            if range_size > MAX_SELECTED_FRAMES:
                raise DiagnosticError(
                    f"At most {MAX_SELECTED_FRAMES} frames may be selected per run: {item!r}"
                )
            if start < available_min or end > available_max:
                raise DiagnosticError(
                    f"Requested frame range {item!r} is outside the available timeline "
                    f"{available_min}-{available_max}"
                )
            requested.update(range(start, end + 1))
        elif item.isdigit():
            requested.add(parse_number(item, item))
        else:
            raise DiagnosticError(f"Invalid frame selector item {item!r}")
        if len(requested) > MAX_SELECTED_FRAMES:
            raise DiagnosticError(
                f"At most {MAX_SELECTED_FRAMES} frames may be selected per run."
            )

    missing = sorted(requested - set(available))
    if missing:
        raise DiagnosticError(f"Requested frames are absent from the sequence: {missing}")
    if not requested:
        raise DiagnosticError("No frames were selected.")
    return sorted(requested)


def _validate_request(
    dataset_path: Path,
    output_dir: Path,
    profile: str,
    sequence: str,
    scale: str,
    diagnostics: Sequence[str],
) -> tuple[Path, Path, tuple[str, ...]]:
    try:
        dataset_root = dataset_path.expanduser().resolve()
    except (OSError, RuntimeError) as error:
        raise DiagnosticError(f"Cannot resolve dataset path {dataset_path}: {error}") from error
    if not dataset_root.is_dir():
        raise DiagnosticError(f"Dataset path is not a directory: {dataset_root}")
    if profile not in SUPPORTED_PROFILES:
        raise DiagnosticError(
            f"Unsupported profile {profile!r}; supported profiles: {', '.join(SUPPORTED_PROFILES)}"
        )
    if SEQUENCE_PATTERN.fullmatch(sequence) is None:
        raise DiagnosticError("--sequence must be a numeric sequence identifier.")
    if SCALE_PATTERN.fullmatch(scale) is None:
        raise DiagnosticError("--scale must use dataset notation such as x2 or x1_7.")

    try:
        output = output_dir.expanduser().resolve()
    except (OSError, RuntimeError) as error:
        raise DiagnosticError(f"Cannot resolve output directory {output_dir}: {error}") from error
    if output == dataset_root or dataset_root in output.parents:
        raise DiagnosticError("--output-dir must be outside the dataset root.")
    if output.exists():
        raise DiagnosticError(f"--output-dir already exists; choose a new directory: {output}")

    requested = set(diagnostics)
    unsupported = sorted(requested - set(SUPPORTED_DIAGNOSTICS) - {"all"})
    if unsupported:
        raise DiagnosticError(f"Unsupported diagnostics: {unsupported}")
    selected = (
        SUPPORTED_DIAGNOSTICS
        if not requested or "all" in requested
        else tuple(name for name in SUPPORTED_DIAGNOSTICS if name in requested)
    )
    return dataset_root, output, selected


def _write_report_directory(output: Path, report: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        _write_report_files(staging, report)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _metric_comparison(before: float, after: float) -> str:
    if math.isclose(before, after, rel_tol=1e-9, abs_tol=1e-12):
        return "unchanged"
    return "improved" if after < before else "worsened"


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _preflight_interpretation(preflight: dict[str, Any]) -> dict[str, Any]:
    findings = preflight["findings"]
    errors = [finding for finding in findings if finding["severity"] == "error"]
    warnings = [finding for finding in findings if finding["severity"] == "warning"]
    error_count = len(errors)
    warning_count = len(warnings)
    info_count = sum(finding["severity"] == "info" for finding in findings)
    if error_count:
        assessment = "input_errors_detected"
        meaning = (
            "The dataset contract failed. Visual diagnostics were not generated because "
            "invalid inputs would make their interpretation unreliable."
        )
        next_step = "Fix the error findings, then rerun the same command with a new output directory."
    elif warning_count:
        assessment = "passed_with_warnings"
        meaning = (
            "Required inputs passed validation. Warning findings are review items and do not "
            "by themselves make the dataset unusable."
        )
        next_step = "Review warning findings before relying on the visual diagnostics."
    else:
        assessment = "passed"
        meaning = "Required inputs passed the selected dataset contract."
        next_step = "Interpret the requested diagnostics and inspect their referenced images."
    return {
        "assessment": assessment,
        "error_count": error_count,
        "warning_count": warning_count,
        "info_count": info_count,
        "blocking_errors": [
            {
                "check_id": finding["check_id"],
                "path": finding["path"],
                "message": finding["message"],
            }
            for finding in errors
        ],
        "review_warnings": [
            {
                "check_id": finding["check_id"],
                "path": finding["path"],
                "message": finding["message"],
            }
            for finding in warnings
        ],
        "meaning": meaning,
        "next_step": next_step,
    }


def _jitter_interpretation(
    frame_results: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    raw_mse_values: list[float] = []
    compensated_mse_values: list[float] = []
    mse_counts = {"improved": 0, "unchanged": 0, "worsened": 0}
    translation_counts = {"improved": 0, "unchanged": 0, "worsened": 0}

    for frame_result in frame_results:
        jitter = frame_result.get("jitter", {})
        raw_mse = jitter.get("raw_temporal_mse")
        compensated_mse = jitter.get("unjittered_temporal_mse")
        if raw_mse is not None and compensated_mse is not None:
            raw_mse_values.append(raw_mse)
            compensated_mse_values.append(compensated_mse)
            mse_counts[_metric_comparison(raw_mse, compensated_mse)] += 1

        raw_translation = jitter.get("estimated_raw_translation_pixels")
        residual_translation = jitter.get("estimated_residual_translation_pixels")
        if raw_translation is not None and residual_translation is not None:
            translation_counts[
                _metric_comparison(
                    raw_translation["magnitude"],
                    residual_translation["magnitude"],
                )
            ] += 1

    compared_pairs = len(raw_mse_values)
    if compared_pairs == 0:
        assessment = "insufficient_temporal_comparison"
        meaning = (
            "No adjacent frame pair was available, so this run cannot assess whether the "
            "supplied jitter compensation improves temporal similarity."
        )
    elif mse_counts["improved"] == compared_pairs:
        assessment = "metadata_compensation_consistently_improved"
        meaning = (
            "Applying the supplied jitter offsets reduced temporal MSE in every compared "
            "frame pair. This supports the metadata sign and scale for the sampled content."
        )
    elif mse_counts["worsened"] == compared_pairs:
        assessment = "metadata_compensation_consistently_worsened"
        meaning = (
            "Applying the supplied jitter offsets increased temporal MSE in every compared "
            "frame pair. Investigate jitter sign, scale, ratio selection, and frame ordering."
        )
    elif mse_counts["improved"] > mse_counts["worsened"]:
        assessment = "metadata_compensation_generally_improved"
        meaning = (
            "Applying the supplied jitter offsets improved temporal MSE in more sampled "
            "frame pairs than it worsened."
        )
    elif mse_counts["worsened"] > mse_counts["improved"]:
        assessment = "metadata_compensation_generally_worsened"
        meaning = (
            "Applying the supplied jitter offsets worsened temporal MSE in more sampled "
            "frame pairs than it improved."
        )
    else:
        assessment = "mixed_or_unchanged_temporal_evidence"
        meaning = (
            "The sampled frame pairs show mixed or unchanged temporal evidence after applying "
            "the supplied jitter offsets."
        )

    jitter_aggregate = aggregate.get("jitter", {})
    return {
        "assessment": assessment,
        "meaning": meaning,
        "does_not_mean": (
            "Intentional capture jitter is a defect, that residual motion is caused only by "
            "jitter, or that final NSS image quality is proven correct."
        ),
        "evidence": {
            "compared_pairs": compared_pairs,
            "temporal_mse_pair_counts": mse_counts,
            "mean_raw_temporal_mse": _mean_or_none(raw_mse_values),
            "mean_compensated_temporal_mse": _mean_or_none(compensated_mse_values),
            "translation_pair_counts": translation_counts,
            "mean_raw_translation_magnitude_pixels": jitter_aggregate.get(
                "mean_raw_translation_magnitude_pixels"
            ),
            "mean_residual_translation_magnitude_pixels": jitter_aggregate.get(
                "mean_residual_translation_magnitude_pixels"
            ),
        },
        "inspect_next": [
            "Compare static edges in the jitter contact sheets.",
            "Prefer nearly static frame ranges when judging phase-correlation translation.",
            "If compensation worsens consistently, verify sign, scale, ratio index, and frame order.",
        ],
    }


def _histogram_interpretation(aggregate: dict[str, Any]) -> dict[str, Any]:
    histogram = aggregate.get("histogram", {})
    non_finite_pixels = int(histogram.get("non_finite_pixels", 0))
    negative_input_pixels = int(histogram.get("negative_input_pixels", 0))
    if non_finite_pixels or negative_input_pixels:
        assessment = "input_value_anomalies_detected"
        meaning = (
            "The sampled color input contains non-finite or negative values. Review the "
            "affected input before drawing exposure conclusions from the histogram."
        )
    else:
        assessment = "sampled_input_values_finite_and_non_negative"
        meaning = (
            "The sampled raw color values are finite and non-negative. The dark and bright "
            "fractions describe scene content after the public tonemapping approximation."
        )
    return {
        "assessment": assessment,
        "meaning": meaning,
        "does_not_mean": (
            "A concentration of dark or bright pixels proves underexposure, overexposure, "
            "lost detail, or correct final NSS output."
        ),
        "evidence": {
            "sample_count": histogram.get("sample_count"),
            "non_finite_pixels": non_finite_pixels,
            "negative_input_pixels": negative_input_pixels,
            "dark_fraction": histogram.get("dark_fraction"),
            "bright_fraction": histogram.get("bright_fraction"),
        },
        "inspect_next": [
            "Inspect the tonemapped image beside the histogram for visible tonal detail.",
            "Compare suspect frames with nearby frames and their exposure metadata.",
            "Use runtime-specific tooling if exact quantized tensor values are required.",
        ],
    }


def _motion_interpretation(
    frame_results: Sequence[dict[str, Any]],
    warnings: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    generated = [
        frame_result["motion"]
        for frame_result in frame_results
        if frame_result.get("motion", {}).get("status") == "generated"
    ]
    best_polarity_counts: dict[str, int] = {}
    candidate_values: dict[str, list[float]] = {}
    for motion in generated:
        best_polarity = motion.get("best_polarity")
        if best_polarity is not None:
            best_polarity_counts[best_polarity] = best_polarity_counts.get(best_polarity, 0) + 1
        for name, value in motion.get("candidate_mse", {}).items():
            if value is not None:
                candidate_values.setdefault(name, []).append(value)

    candidate_mean_mse = {
        name: float(np.mean(values))
        for name, values in candidate_values.items()
    }
    polarity_warnings = [
        warning for warning in warnings if warning["code"] == "motion.polarity.suspect"
    ]
    generated_count = len(generated)
    documented_best_count = best_polarity_counts.get("documented", 0)
    if generated_count == 0:
        assessment = "insufficient_motion_comparison"
        meaning = "No adjacent frame pair was available for motion reprojection."
    elif len(polarity_warnings) == generated_count:
        assessment = "alternative_polarity_consistently_better"
        meaning = (
            "A substantially lower-MSE alternative polarity was found in every compared "
            "frame. This is a strong lead to verify the capture and integration convention."
        )
    elif polarity_warnings:
        assessment = "alternative_polarity_better_on_some_frames"
        meaning = (
            "A substantially lower-MSE alternative polarity was found on some compared "
            "frames. The evidence is mixed and needs visual and convention checks."
        )
    elif documented_best_count == generated_count:
        assessment = "documented_polarity_supported"
        meaning = (
            "The documented motion-vector polarity produced the lowest photometric MSE in "
            "every compared frame."
        )
    else:
        assessment = "no_substantial_polarity_issue_detected"
        meaning = (
            "No alternative polarity crossed the configured warning threshold, although the "
            "lowest-MSE candidate was not uniform across all frames."
        )
    return {
        "assessment": assessment,
        "meaning": meaning,
        "does_not_mean": (
            "The lowest-MSE polarity should be applied automatically, that the motion vectors "
            "are the root cause, or that the diagnostic mask matches runtime disocclusion."
        ),
        "evidence": {
            "generated_frames": generated_count,
            "polarity_warning_frames": len(polarity_warnings),
            "best_polarity_counts": best_polarity_counts,
            "candidate_mean_mse": candidate_mean_mse,
            "mean_valid_fraction": _mean_or_none(
                [motion["valid_fraction"] for motion in generated]
            ),
        },
        "inspect_next": [
            "Compare current and warped-previous panels around static edges and moving objects.",
            "Check the squared-error heatmap together with the valid reprojection mask.",
            "Verify whether the capture stores forward or backward motion and its X/Y sign convention.",
        ],
    }


def _build_interpretation(
    *,
    status: str,
    selected_diagnostics: Sequence[str],
    preflight: dict[str, Any],
    aggregate: dict[str, Any],
    frame_results: Sequence[dict[str, Any]],
    warnings: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    preflight_result = _preflight_interpretation(preflight)
    if status == "FAIL":
        overall = {
            "assessment": "preflight_failed",
            "meaning": (
                "Dataset input errors prevented reliable visual diagnostics. Read the error "
                "findings before inspecting or comparing image metrics."
            ),
            "next_step": preflight_result["next_step"],
        }
    elif warnings or preflight_result["warning_count"]:
        overall = {
            "assessment": "diagnostics_generated_with_warnings",
            "meaning": (
                "Diagnostics were generated, and one or more preflight or heuristic warnings "
                "require follow-up. Warnings are investigation leads rather than automatic fixes."
            ),
            "next_step": "Read each diagnostic section and inspect the referenced PNG evidence.",
        }
    else:
        overall = {
            "assessment": "diagnostics_generated_without_warnings",
            "meaning": (
                "Diagnostics were generated without triggering configured heuristic warnings. "
                "This does not prove that integration behavior or final image quality is correct."
            ),
            "next_step": "Review the diagnostic sections and representative PNG evidence.",
        }

    result: dict[str, Any] = {
        "human_readable_report": "interpretation.md",
        "overall": overall,
        "preflight": preflight_result,
    }
    if status == "PASS":
        if "jitter" in selected_diagnostics:
            result["jitter"] = _jitter_interpretation(frame_results, aggregate)
        if "histogram" in selected_diagnostics:
            result["histogram"] = _histogram_interpretation(aggregate)
        if "motion" in selected_diagnostics:
            result["motion"] = _motion_interpretation(frame_results, warnings)
    result["limitations"] = [
        "These diagnostics use captured inputs and public approximations, not exact NSS runtime state.",
        "Scene motion, animation, occlusion, transparency, lighting, and capture noise can affect image metrics.",
        "Heuristic warnings identify evidence to investigate and do not establish a root cause by themselves.",
    ]
    return result


def _format_interpretation_value(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return ", ".join(f"{key}: {_format_interpretation_value(item)}" for key, item in value.items())
    return str(value)


def _render_interpretation_markdown(report: dict[str, Any]) -> str:
    interpretation = report["interpretation"]
    lines = [
        "# NSS Diagnostic Interpretation",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Profile: `{report['profile']}`",
        f"- Sequence: `{report['sequence']}`",
        f"- Scale: `{report['scale']}`",
        f"- Status: **{report['status']}**",
        "",
        "## Overall",
        "",
        f"**Assessment:** `{interpretation['overall']['assessment']}`",
        "",
        interpretation["overall"]["meaning"],
        "",
        f"**Next step:** {interpretation['overall']['next_step']}",
        "",
        "## Preflight",
        "",
        f"**Assessment:** `{interpretation['preflight']['assessment']}`",
        "",
        interpretation["preflight"]["meaning"],
        "",
        (
            f"Errors: {interpretation['preflight']['error_count']}; "
            f"warnings: {interpretation['preflight']['warning_count']}; "
            f"information findings: {interpretation['preflight']['info_count']}."
        ),
    ]
    if interpretation["preflight"]["blocking_errors"]:
        lines.extend(["", "**Blocking errors:**", ""])
        for finding in interpretation["preflight"]["blocking_errors"]:
            path = f" (`{finding['path']}`)" if finding["path"] else ""
            lines.append(f"- `{finding['check_id']}`{path}: {finding['message']}")
    if interpretation["preflight"]["review_warnings"]:
        lines.extend(["", "**Review warnings:**", ""])
        for finding in interpretation["preflight"]["review_warnings"]:
            path = f" (`{finding['path']}`)" if finding["path"] else ""
            lines.append(f"- `{finding['check_id']}`{path}: {finding['message']}")
    for section_name in ("jitter", "histogram", "motion"):
        section = interpretation.get(section_name)
        if section is None:
            continue
        lines.extend(
            [
                "",
                f"## {section_name.title()}",
                "",
                f"**Assessment:** `{section['assessment']}`",
                "",
                section["meaning"],
                "",
                "**Evidence:**",
                "",
            ]
        )
        for name, value in section["evidence"].items():
            label = name.replace("_", " ")
            lines.append(f"- {label}: {_format_interpretation_value(value)}")
        lines.extend(
            [
                "",
                f"**What this does not mean:** {section['does_not_mean']}",
                "",
                "**Inspect next:**",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in section["inspect_next"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in interpretation["limitations"])
    return "\n".join(lines) + "\n"


def _write_report_files(directory: Path, report: dict[str, Any]) -> None:
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (directory / report["interpretation"]["human_readable_report"]).write_text(
        _render_interpretation_markdown(report),
        encoding="utf-8",
    )


def run_diagnostics(
    dataset_path: Path,
    *,
    profile: str,
    sequence: str,
    scale: str,
    frames: str,
    diagnostics: Sequence[str],
    output_dir: Path,
) -> dict[str, Any]:
    dataset_root, output, selected_diagnostics = _validate_request(
        dataset_path,
        output_dir,
        profile,
        sequence,
        scale,
        diagnostics,
    )
    try:
        profile_data = load_profile(profile)
    except ProfileError as error:
        raise DiagnosticError(str(error)) from error
    exposure_override = _NssDataset._read_exposure_override(dataset_root)
    try:
        validation_report = validate_dataset(
            dataset_root,
            profile,
            use_case="debug",
            scale=scale,
            warning_policy="allow",
            max_image_pixels=MAX_IMAGE_PIXELS,
            exposure_override=exposure_override,
        )
    except ValidationError as error:
        raise DiagnosticError(str(error)) from error
    if validation_report.status != "PASS":
        preflight = validation_report.to_dict()
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "status": "FAIL",
            "dataset": str(dataset_root),
            "profile": profile,
            "sequence": sequence,
            "scale": scale,
            "diagnostics": list(selected_diagnostics),
            "exposure": {
                "source": "config.json" if exposure_override is not None else "sequence metadata",
                "override": exposure_override is not None,
                "value": exposure_override,
            },
            "selected_frames": [],
            "summary": {
                "frames": 0,
                "artifacts": 0,
                "warnings": preflight["summary"]["findings"]["warnings"],
                "warning_counts": {
                    "preflight": preflight["summary"]["findings"]["warnings"],
                    "diagnostic": 0,
                    "total": preflight["summary"]["findings"]["warnings"],
                },
            },
            "preflight": {
                "status": preflight["status"],
                "complete_scan": preflight["complete_scan"],
                "summary": preflight["summary"],
                "findings": preflight["findings"],
            },
            "warnings": [],
            "artifacts": [],
            "aggregate": {},
            "frames": [],
        }
        report["interpretation"] = _build_interpretation(
            status=report["status"],
            selected_diagnostics=selected_diagnostics,
            preflight=report["preflight"],
            aggregate=report["aggregate"],
            frame_results=report["frames"],
            warnings=report["warnings"],
        )
        _write_report_directory(output, report)
        return report

    dataset = _NssDataset(
        dataset_root,
        sequence,
        scale,
        profile_data,
        exposure_override,
    )
    selected_frames = _frame_selector(frames, dataset.available_frames)
    warnings: list[dict[str, Any]] = []
    frame_results: list[dict[str, Any]] = []
    artifacts: list[str] = []
    aggregate_histogram = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    aggregate_histogram_samples = 0
    aggregate_dark_pixels = 0
    aggregate_bright_pixels = 0
    aggregate_non_finite_pixels = 0
    aggregate_negative_pixels = 0
    raw_shift_magnitudes: list[float] = []
    residual_shift_magnitudes: list[float] = []
    motion_summaries: list[dict[str, float]] = []

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for frame in selected_frames:
            state = dataset.frame_state(frame)
            result: dict[str, Any] = {
                "frame": frame,
                "exposure": state.exposure,
                "exposure_source": state.exposure_source,
                "jitter_source": state.jitter_source,
                "jitter_pixels": {
                    "x": state.jitter_pixels[0],
                    "y": state.jitter_pixels[1],
                },
            }

            if "jitter" in selected_diagnostics:
                relative_path = Path("jitter") / f"frame_{frame:04d}.png"
                _save_panels(
                    (
                        ("Jittered input", _to_display_rgb(state.processed_color)),
                        ("Supplied jitter removed", _to_display_rgb(state.unjittered_color)),
                    ),
                    staging / relative_path,
                )
                jitter_result: dict[str, Any] = {
                    "artifact": relative_path.as_posix(),
                    "panel_order": ["jittered_input", "supplied_jitter_removed"],
                }
                previous_frame = frame - 1
                if previous_frame in dataset.frames:
                    previous = dataset.frame_state(previous_frame)
                    jitter_result["raw_temporal_mse"] = float(
                        np.mean(np.square(state.processed_color - previous.processed_color))
                    )
                    jitter_result["unjittered_temporal_mse"] = float(
                        np.mean(np.square(state.unjittered_color - previous.unjittered_color))
                    )
                    raw_translation = _estimate_translation(
                        previous.processed_color,
                        state.processed_color,
                    )
                    residual_translation = _estimate_translation(
                        previous.unjittered_color,
                        state.unjittered_color,
                    )
                    jitter_result["estimated_raw_translation_pixels"] = raw_translation
                    jitter_result["estimated_residual_translation_pixels"] = residual_translation
                    if raw_translation is not None:
                        raw_shift_magnitudes.append(raw_translation["magnitude"])
                    if residual_translation is not None:
                        residual_shift_magnitudes.append(residual_translation["magnitude"])
                result["jitter"] = jitter_result
                artifacts.append(relative_path.as_posix())

            if "histogram" in selected_diagnostics:
                histogram, histogram_result = _histogram_metrics(state)
                relative_path = Path("histogram") / f"frame_{frame:04d}.png"
                _save_histogram(state, histogram, staging / relative_path)
                histogram_result["artifact"] = relative_path.as_posix()
                histogram_result["panel_order"] = [
                    "unjittered_tonemapped_input",
                    "luminance_histogram",
                ]
                aggregate_histogram += histogram
                aggregate_histogram_samples += histogram_result["sample_count"]
                aggregate_dark_pixels += histogram_result["dark_pixels"]
                aggregate_bright_pixels += histogram_result["bright_pixels"]
                aggregate_non_finite_pixels += histogram_result["non_finite_pixels"]
                aggregate_negative_pixels += histogram_result["negative_input_pixels"]
                result["histogram"] = histogram_result
                artifacts.append(relative_path.as_posix())

            if "motion" in selected_diagnostics:
                previous_frame = frame - 1
                if previous_frame not in dataset.frames:
                    result["motion"] = {
                        "status": "skipped",
                        "reason": "No immediately preceding frame is present.",
                    }
                else:
                    previous = dataset.frame_state(previous_frame)
                    motion = dataset.motion(frame)
                    if motion.shape[:2] != state.unjittered_color.shape[:2]:
                        raise DiagnosticError(
                            f"Motion and color dimensions differ at frame {frame}: "
                            f"{motion.shape[:2]} versus {state.unjittered_color.shape[:2]}"
                        )
                    height, width = state.unjittered_color.shape[:2]
                    motion_scale_source, motion_vector_scale = dataset.motion_vector_scale(
                        frame,
                        width,
                        height,
                    )
                    scale_x, scale_y = motion_vector_scale
                    polarities = {
                        "documented": (scale_x, scale_y),
                        "flip-x": (-scale_x, scale_y),
                        "flip-y": (scale_x, -scale_y),
                        "flip-xy": (-scale_x, -scale_y),
                    }
                    candidate_data: dict[str, _MotionCandidate] = {}
                    for name, (sign_x, sign_y) in polarities.items():
                        candidate_data[name] = _motion_candidate(
                            state.unjittered_color,
                            previous.unjittered_color,
                            motion,
                            sign_x,
                            sign_y,
                        )
                    comparison_valid = np.logical_and.reduce(
                        [candidate.valid for candidate in candidate_data.values()]
                    )
                    candidate_results = {
                        name: _mean_masked_error(candidate.error, comparison_valid)
                        for name, candidate in candidate_data.items()
                    }
                    documented_candidate = candidate_data["documented"]
                    warped = documented_candidate.warped
                    error = documented_candidate.error
                    geometric_valid = documented_candidate.valid
                    documented_mse = candidate_results["documented"]
                    current_depth = _shift_image(
                        dataset.depth(frame),
                        source_offset_x=-state.jitter_pixels[0],
                        source_offset_y=-state.jitter_pixels[1],
                    )
                    previous_depth = _shift_image(
                        dataset.depth(previous_frame),
                        source_offset_x=-previous.jitter_pixels[0],
                        source_offset_y=-previous.jitter_pixels[1],
                    )
                    if (
                        current_depth.shape[:2] != state.unjittered_color.shape[:2]
                        or previous_depth.shape[:2] != state.unjittered_color.shape[:2]
                    ):
                        raise DiagnosticError(
                            f"Depth and color dimensions differ at frame {frame} or {previous_frame}"
                        )
                    valid, depth_threshold = _depth_consistency_mask(
                        current_depth,
                        previous_depth,
                        motion,
                        motion_vector_scale,
                        geometric_valid,
                    )
                    mse = _mean_masked_error(error, valid)
                    finite_candidates = {
                        name: value for name, value in candidate_results.items() if value is not None
                    }
                    best_polarity = (
                        min(finite_candidates, key=finite_candidates.get)
                        if finite_candidates
                        else None
                    )
                    if (
                        documented_mse is not None
                        and best_polarity is not None
                        and best_polarity != "documented"
                        and finite_candidates[best_polarity]
                        < documented_mse * POLARITY_IMPROVEMENT_RATIO
                    ):
                        warnings.append(
                            {
                                "code": "motion.polarity.suspect",
                                "frame": frame,
                                "message": (
                                    f"{best_polarity} produced substantially lower photometric MSE "
                                    "than the documented dataset convention."
                                ),
                            }
                        )
                    geometric_fraction = float(np.mean(geometric_valid))
                    valid_fraction = float(np.mean(valid))
                    if geometric_fraction > 0.0 and valid_fraction < geometric_fraction * 0.5:
                        warnings.append(
                            {
                                "code": "motion.depth_mask.low_coverage",
                                "frame": frame,
                                "message": (
                                    "The depth-consistency mask retained less than half of the "
                                    "geometrically valid reprojections; inspect the mask before "
                                    "using the masked MSE."
                                ),
                            }
                        )

                    relative_path = Path("motion") / f"frame_{frame:04d}.png"
                    mask_rgb = np.repeat((valid.astype(np.uint8) * 255)[..., None], 3, axis=2)
                    _save_panels(
                        (
                            ("Current, unjittered", _to_display_rgb(state.unjittered_color)),
                            ("Previous warped", _to_display_rgb(warped)),
                            ("Squared-error heatmap", _error_heatmap(error, valid)),
                            ("Valid reprojection mask", mask_rgb),
                        ),
                        staging / relative_path,
                    )
                    result["motion"] = {
                        "status": "generated",
                        "previous_frame": previous_frame,
                        "mse": mse,
                        "unmasked_mse": documented_mse,
                        "p95_squared_error": (
                            float(np.percentile(error[valid], 95.0)) if np.any(valid) else None
                        ),
                        "valid_fraction": valid_fraction,
                        "geometric_valid_fraction": geometric_fraction,
                        "depth_consistency_threshold": depth_threshold,
                        "candidate_mse": candidate_results,
                        "polarity_comparison_valid_fraction": float(np.mean(comparison_valid)),
                        "best_polarity": best_polarity,
                        "motion_vector_scale": {
                            "source": motion_scale_source,
                            "x": scale_x,
                            "y": scale_y,
                        },
                        "artifact": relative_path.as_posix(),
                        "panel_order": [
                            "current_unjittered",
                            "previous_warped",
                            "squared_error_heatmap",
                            "valid_reprojection_mask",
                        ],
                        "mask_scope": (
                            "The mask excludes non-finite and out-of-bounds reprojections, then "
                            "uses a robust frame-local depth-consistency threshold to exclude "
                            "likely visibility changes. It is diagnostic rather than the exact "
                            "runtime disocclusion mask."
                        ),
                    }
                    if mse is not None:
                        motion_summaries.append(
                            {
                                "mse": mse,
                                "p95": float(np.percentile(error[valid], 95.0)),
                                "valid_fraction": valid_fraction,
                            }
                        )
                    artifacts.append(relative_path.as_posix())

            frame_results.append(result)

        aggregate: dict[str, Any] = {}
        if "jitter" in selected_diagnostics:
            aggregate["jitter"] = {
                "compared_frame_pairs": len(residual_shift_magnitudes),
                "mean_raw_translation_magnitude_pixels": (
                    float(np.mean(raw_shift_magnitudes)) if raw_shift_magnitudes else None
                ),
                "mean_residual_translation_magnitude_pixels": (
                    float(np.mean(residual_shift_magnitudes))
                    if residual_shift_magnitudes
                    else None
                ),
                "maximum_residual_translation_magnitude_pixels": (
                    float(np.max(residual_shift_magnitudes))
                    if residual_shift_magnitudes
                    else None
                ),
                "interpretation": (
                    "Phase-correlation translation is diagnostic for mostly static content; "
                    "scene or camera motion can dominate it."
                ),
            }
        if "histogram" in selected_diagnostics:
            relative_path = Path("histogram") / "aggregate.png"
            _write_png(staging / relative_path, _histogram_chart(aggregate_histogram))
            artifacts.append(relative_path.as_posix())
            aggregate["histogram"] = {
                "artifact": relative_path.as_posix(),
                "sample_count": aggregate_histogram_samples,
                "non_finite_pixels": aggregate_non_finite_pixels,
                "negative_input_pixels": aggregate_negative_pixels,
                "dark_pixels": aggregate_dark_pixels,
                "bright_pixels": aggregate_bright_pixels,
                "dark_fraction": (
                    aggregate_dark_pixels / aggregate_histogram_samples
                    if aggregate_histogram_samples
                    else None
                ),
                "bright_fraction": (
                    aggregate_bright_pixels / aggregate_histogram_samples
                    if aggregate_histogram_samples
                    else None
                ),
                "bins": aggregate_histogram.astype(int).tolist(),
            }
        if "motion" in selected_diagnostics:
            aggregate["motion"] = {
                "generated_frames": len(motion_summaries),
                "mean_mse": (
                    float(np.mean([item["mse"] for item in motion_summaries]))
                    if motion_summaries
                    else None
                ),
                "maximum_p95_squared_error": (
                    float(np.max([item["p95"] for item in motion_summaries]))
                    if motion_summaries
                    else None
                ),
                "mean_valid_fraction": (
                    float(np.mean([item["valid_fraction"] for item in motion_summaries]))
                    if motion_summaries
                    else None
                ),
            }

        preflight = validation_report.to_dict()
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "status": "PASS",
            "dataset": str(dataset_root),
            "profile": profile,
            "sequence": sequence,
            "scale": scale,
            "diagnostics": list(selected_diagnostics),
            "exposure": {
                "source": (
                    "config.json"
                    if dataset.exposure_override is not None
                    else "sequence metadata"
                ),
                "override": dataset.exposure_override is not None,
                "value": dataset.exposure_override,
            },
            "selected_frames": selected_frames,
            "summary": {
                "frames": len(selected_frames),
                "artifacts": len(artifacts),
                "warnings": (
                    preflight["summary"]["findings"]["warnings"] + len(warnings)
                ),
                "warning_counts": {
                    "preflight": preflight["summary"]["findings"]["warnings"],
                    "diagnostic": len(warnings),
                    "total": (
                        preflight["summary"]["findings"]["warnings"]
                        + len(warnings)
                    ),
                },
            },
            "preflight": {
                "status": preflight["status"],
                "complete_scan": preflight["complete_scan"],
                "summary": preflight["summary"],
                "findings": preflight["findings"],
            },
            "warnings": warnings,
            "artifacts": artifacts,
            "aggregate": aggregate,
            "frames": frame_results,
        }
        report["interpretation"] = _build_interpretation(
            status=report["status"],
            selected_diagnostics=selected_diagnostics,
            preflight=report["preflight"],
            aggregate=report["aggregate"],
            frame_results=report["frames"],
            warnings=report["warnings"],
        )
        _write_report_files(staging, report)
        staging.rename(output)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
