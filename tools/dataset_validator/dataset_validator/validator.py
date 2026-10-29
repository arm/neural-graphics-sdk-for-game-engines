# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .exr_reader import ExrReadError, read_exr
from .model import Finding, ValidationReport
from .profiles import ProfileError, load_profile


USE_CASES = {"debug", "replay"}
WARNING_POLICIES = {"allow", "fail"}


class ValidationError(ValueError):
    pass


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _metadata_frame_index(frame_metadata: dict[str, Any]) -> int | None:
    has_frame = "Frame" in frame_metadata
    has_frame_number = "FrameNumber" in frame_metadata
    if has_frame and has_frame_number:
        frame = frame_metadata["Frame"]
        frame_number = frame_metadata["FrameNumber"]
        if (
            not _is_non_negative_int(frame)
            or not _is_non_negative_int(frame_number)
            or frame != frame_number
        ):
            return None
        return frame

    value = (
        frame_metadata.get("Frame")
        if has_frame
        else frame_metadata.get("FrameNumber")
    )
    return value if _is_non_negative_int(value) else None


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _finding(
    report: ValidationReport,
    severity: str,
    check_id: str,
    message: str,
    remediation: str,
    *,
    sequence: str | None = None,
    frame: int | None = None,
    path: str | None = None,
    observed: Any = None,
    expected: Any = None,
) -> None:
    report.findings.append(
        Finding(
            severity=severity,
            check_id=check_id,
            message=message,
            remediation=remediation,
            sequence=sequence,
            frame=frame,
            path=path,
            observed=observed,
            expected=expected,
        )
    )


def _validate_camera_planes(
    report: ValidationReport,
    values: dict[str, Any],
    required_fields: list[str],
    metadata_path: Path,
    sequence: str,
    frame_number: int | None,
) -> None:
    relative_metadata_path = _relative_path(Path(report.dataset), metadata_path)
    for field_name in ("CameraNearPlane", "NearPlane", "CameraFarPlane", "FarPlane"):
        if field_name not in required_fields or field_name not in values:
            continue
        value = values[field_name]
        numeric = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
        is_near = "Near" in field_name
        if numeric:
            numeric_value = float(value)
            valid = numeric_value > 0.0 if is_near else numeric_value == -1.0 or numeric_value > 0.0
        else:
            valid = False
        if valid:
            continue
        expected = "finite positive number" if is_near else "positive finite number or -1"
        _finding(
            report,
            "error",
            "metadata.camera_plane.invalid",
            f"Metadata field {field_name!r} is not a valid camera plane.",
            (
                f"Export the positive {field_name} value used for this capture."
                if is_near
                else f"Export a positive finite {field_name}, or -1 for an infinite far plane."
            ),
            sequence=sequence,
            frame=frame_number,
            path=relative_metadata_path,
            observed={field_name: value},
            expected={field_name: expected},
        )


def _metadata_frame_numbers(
    report: ValidationReport,
    metadata: dict[str, Any],
    metadata_path: Path,
    sequence: str,
) -> list[int]:
    frames = metadata.get("Frames")
    relative_metadata_path = _relative_path(Path(report.dataset), metadata_path)
    if not isinstance(frames, list):
        _finding(
            report,
            "error",
            "metadata.frames.missing",
            "Sequence metadata does not contain a Frames array.",
            "Write one metadata object per captured frame in the Frames array.",
            sequence=sequence,
            path=relative_metadata_path,
            observed=type(frames).__name__,
            expected="array",
        )
        return []

    frame_numbers: list[int] = []
    invalid_entries: list[int] = []
    conflicting_entries: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            invalid_entries.append(index)
            continue

        frame_index = _metadata_frame_index(frame)
        if frame_index is not None:
            frame_numbers.append(frame_index)
        elif (
            "Frame" in frame
            and "FrameNumber" in frame
            and _is_non_negative_int(frame["Frame"])
            and _is_non_negative_int(frame["FrameNumber"])
        ):
            conflicting_entries.append(
                {
                    "entry": index,
                    "Frame": frame["Frame"],
                    "FrameNumber": frame["FrameNumber"],
                }
            )
        else:
            invalid_entries.append(index)

    if invalid_entries:
        _finding(
            report,
            "error",
            "metadata.frame_index.invalid",
            "One or more frame metadata entries have no valid integer Frame or FrameNumber index.",
            "Add a non-negative integer Frame or FrameNumber value to every frame metadata object.",
            sequence=sequence,
            path=relative_metadata_path,
            observed=invalid_entries,
            expected="integer Frame or FrameNumber value for every entry",
        )

    if conflicting_entries:
        _finding(
            report,
            "error",
            "metadata.frame_index.conflict",
            "Frame and FrameNumber disagree in one or more metadata entries.",
            "Make both index fields equal, or provide only one of them.",
            sequence=sequence,
            path=relative_metadata_path,
            observed=conflicting_entries,
            expected="matching non-negative integer Frame and FrameNumber values",
        )

    if len(frame_numbers) != len(set(frame_numbers)):
        duplicates = sorted({frame for frame in frame_numbers if frame_numbers.count(frame) > 1})
        _finding(
            report,
            "error",
            "metadata.frame_index.duplicate",
            "Frame metadata contains duplicate frame indices.",
            "Capture or regenerate metadata so each frame index appears exactly once.",
            sequence=sequence,
            path=relative_metadata_path,
            observed=duplicates,
            expected="unique frame indices",
        )

    # Frames array order is not semantic; replay uses each entry's explicit frame index.
    sorted_frames = sorted(set(frame_numbers))
    expected_frames = list(range(len(frames)))
    if sorted_frames != expected_frames:
        missing = sorted(set(expected_frames) - set(sorted_frames))
        out_of_range = sorted(set(sorted_frames) - set(expected_frames))
        _finding(
            report,
            "error",
            "metadata.frame_index.range",
            "Frame metadata indices must cover every replay frame exactly once.",
            "Use each integer Frame or FrameNumber index from zero through the last Frames entry exactly once.",
            sequence=sequence,
            path=relative_metadata_path,
            observed={"frames": sorted_frames, "missing": missing, "out_of_range": out_of_range},
            expected=expected_frames,
        )

    declared_count = metadata.get("Samples", {}).get("Count") if isinstance(metadata.get("Samples"), dict) else None
    if not _is_non_negative_int(declared_count):
        _finding(
            report,
            "error",
            "metadata.samples.count.invalid",
            "Samples.Count is missing or is not an integer.",
            "Set Samples.Count to the number of frame metadata entries.",
            sequence=sequence,
            path=relative_metadata_path,
            observed=declared_count,
            expected=len(frames),
        )
    elif declared_count != len(frames):
        _finding(
            report,
            "error",
            "metadata.samples.count.mismatch",
            "Samples.Count does not match the number of frame metadata entries.",
            "Update Samples.Count or recapture the incomplete metadata sequence.",
            sequence=sequence,
            path=relative_metadata_path,
            observed=declared_count,
            expected=len(frames),
        )

    return sorted_frames


def _validate_required_frame_fields(
    report: ValidationReport,
    metadata: dict[str, Any],
    metadata_path: Path,
    sequence: str,
    required_fields: list[str],
) -> None:
    relative_metadata_path = _relative_path(Path(report.dataset), metadata_path)
    frames = metadata.get("Frames")
    if not isinstance(frames, list):
        return

    for frame_metadata in frames:
        if not isinstance(frame_metadata, dict):
            continue
        frame_number = _metadata_frame_index(frame_metadata)
        for field_name in required_fields:
            if field_name == "Frame" or field_name in frame_metadata:
                continue
            _finding(
                report,
                "error",
                "metadata.frame_field.missing",
                f"Frame metadata field {field_name!r} is missing.",
                f"Export {field_name!r} for every frame required by the selected profile.",
                sequence=sequence,
                frame=frame_number if _is_non_negative_int(frame_number) else None,
                path=relative_metadata_path,
                observed=sorted(frame_metadata),
                expected=field_name,
            )

        for field_name in ("FovX", "FovY"):
            if field_name not in required_fields or field_name not in frame_metadata:
                continue
            value = frame_metadata[field_name]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                _finding(
                    report,
                    "error",
                    "metadata.fov.invalid",
                    f"Frame metadata field {field_name!r} is not a finite positive number.",
                    f"Export the positive {field_name} value used for this frame.",
                    sequence=sequence,
                    frame=frame_number if _is_non_negative_int(frame_number) else None,
                    path=relative_metadata_path,
                    observed={field_name: value},
                    expected={field_name: "finite positive number"},
                )

        if "ViewProjection" in required_fields and "ViewProjection" in frame_metadata:
            matrix = frame_metadata["ViewProjection"]
            valid_matrix = (
                isinstance(matrix, list)
                and len(matrix) == 16
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in matrix
                )
            )
            if not valid_matrix:
                _finding(
                    report,
                    "error",
                    "metadata.view_projection.invalid",
                    "Frame ViewProjection is not a finite 4x4 matrix.",
                    "Export ViewProjection as 16 finite numeric values in the documented order.",
                    sequence=sequence,
                    frame=frame_number if _is_non_negative_int(frame_number) else None,
                    path=relative_metadata_path,
                    observed=matrix,
                    expected="array of 16 finite numbers",
                )

        _validate_camera_planes(
            report,
            frame_metadata,
            required_fields,
            metadata_path,
            sequence,
            frame_number if _is_non_negative_int(frame_number) else None,
        )


def _validate_target_resolution(
    report: ValidationReport,
    metadata: dict[str, Any],
    metadata_path: Path,
    sequence: str,
) -> tuple[int, int] | None:
    target_resolution = metadata.get("TargetResolution")
    if target_resolution is None:
        return None

    width = target_resolution.get("X") if isinstance(target_resolution, dict) else None
    height = target_resolution.get("Y") if isinstance(target_resolution, dict) else None
    valid_width = isinstance(width, int) and not isinstance(width, bool) and width > 0
    valid_height = isinstance(height, int) and not isinstance(height, bool) and height > 0
    if not valid_width or not valid_height:
        _finding(
            report,
            "error",
            "metadata.target_resolution.invalid",
            "TargetResolution must contain positive integer X and Y dimensions.",
            "Set TargetResolution.X and TargetResolution.Y to the exported target dimensions.",
            sequence=sequence,
            path=_relative_path(Path(report.dataset), metadata_path),
            observed=target_resolution,
            expected={"X": "positive integer", "Y": "positive integer"},
        )
        return None
    return width, height


def _resolve_lane_directory(
    dataset_root: Path,
    alternatives: list[str],
    sequence: str,
    scale: str,
) -> Path | None:
    for template in alternatives:
        candidate = dataset_root / template.format(sequence=sequence, scale=scale)
        if candidate.is_dir():
            return candidate
    return None


def _frames_for_cadence(frame_numbers: list[int], cadence: str) -> list[int]:
    if cadence == "all":
        return frame_numbers
    if cadence == "even":
        return [frame for frame in frame_numbers if frame % 2 == 0]
    if cadence == "odd":
        return [frame for frame in frame_numbers if frame % 2 != 0]
    if cadence == "midpoints_between_even":
        even_frames = [frame for frame in frame_numbers if frame % 2 == 0]
        return [
            (left + right) // 2
            for left, right in zip(even_frames, even_frames[1:])
            if right - left == 2
        ]
    raise ValidationError(f"Unsupported frame cadence in profile: {cadence!r}")


def _scale_factor(scale: str) -> float:
    try:
        factor = float(scale[1:].replace("_", "."))
    except ValueError as error:
        raise ValidationError(
            f"Scale must use dataset directory form such as 'x2' or 'x1_7', got {scale!r}"
        ) from error
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValidationError(f"Scale factor must be finite and positive, got {scale!r}")
    return factor


def _frame_metadata_by_index(metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    frames = metadata.get("Frames", [])
    if not isinstance(frames, list):
        return {}
    indexed_frames: dict[int, dict[str, Any]] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_index = _metadata_frame_index(frame)
        if frame_index is not None:
            indexed_frames[frame_index] = frame
    return indexed_frames


def _repeating_phase_length(points: list[tuple[float, float]]) -> int:
    if not points:
        return 0
    rounded_points = [tuple(round(value, 8) for value in point) for point in points]
    for period in range(1, len(rounded_points)):
        if rounded_points[period] != rounded_points[0]:
            continue
        if all(
            point == rounded_points[index % period]
            for index, point in enumerate(rounded_points)
        ):
            return period
    return len(rounded_points)


def _indexed_exr_files(
    report: ValidationReport,
    dataset_root: Path,
    lane_directory: Path,
    sequence: str,
    lane_name: str,
) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for path in sorted(lane_directory.iterdir()):
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            report.complete_scan = False
            _finding(
                report,
                "error",
                "file.path.unreadable",
                "An EXR path cannot be resolved.",
                "Repair or remove the broken file entry.",
                sequence=sequence,
                path=_relative_path(dataset_root, path),
                observed=str(error),
                expected="readable file under the dataset root",
            )
            continue
        if dataset_root not in resolved.parents:
            report.complete_scan = False
            _finding(
                report,
                "error",
                "file.path.outside_dataset",
                "A dataset file resolves outside the selected dataset root.",
                "Replace external links with files stored inside the dataset.",
                sequence=sequence,
                path=_relative_path(dataset_root, path),
                observed=str(resolved),
                expected=str(dataset_root),
            )
            continue
        if path.suffix.lower() != ".exr":
            continue
        if not resolved.is_file():
            _finding(
                report,
                "warning",
                "lane.file.unexpected",
                f"{lane_name} lane contains a non-file EXR entry; its contents were not scanned.",
                "Replace the entry with a readable EXR file or remove it from the frame directory.",
                sequence=sequence,
                path=_relative_path(dataset_root, path),
                observed="directory" if resolved.is_dir() else "non-file entry",
                expected="regular *.exr file",
            )
            continue
        try:
            frame = int(path.stem)
        except ValueError:
            report.complete_scan = False
            _finding(
                report,
                "error",
                "lane.frame_name.invalid",
                "An EXR has no numeric frame index and was not scanned.",
                "If this is a capture frame, restore its correct index; otherwise keep it outside the frame directory.",
                sequence=sequence,
                path=_relative_path(dataset_root, path),
                observed=path.name,
                expected="numeric .exr filename",
            )
            continue
        if frame in indexed:
            _finding(
                report,
                "error",
                "lane.frame_index.duplicate",
                f"{lane_name} lane contains duplicate numeric frame indices.",
                "Keep exactly one EXR file for each numeric frame index.",
                sequence=sequence,
                frame=frame,
                path=_relative_path(dataset_root, path),
                observed=[
                    _relative_path(dataset_root, indexed[frame]),
                    _relative_path(dataset_root, path),
                ],
                expected="one EXR file per frame index",
            )
            continue
        indexed[frame] = path
    return indexed


def _validate_exr_files(
    report: ValidationReport,
    dataset_root: Path,
    sequence: str,
    lane_name: str,
    lane_spec: dict[str, Any],
    indexed: dict[int, Path],
    max_image_pixels: int | None,
) -> dict[str, Any]:
    valid_dimensions: set[tuple[int, int]] = set()
    observed_channels: set[tuple[str, ...]] = set()
    expected_channels = set(lane_spec.get("channels", []))
    channel_aliases = set(lane_spec.get("channel_aliases", []))
    expected_pixel_types = set(lane_spec.get("pixel_types", []))
    value_range = lane_spec.get("value_range")
    channel_extrema: dict[str, dict[str, float]] = {}

    for frame, path in sorted(indexed.items()):
        relative_path = _relative_path(dataset_root, path)
        try:
            info = read_exr(path, max_pixels=max_image_pixels)
        except ExrReadError as error:
            report.complete_scan = False
            _finding(
                report,
                "error",
                "exr.unreadable",
                f"{lane_name} frame is not a readable OpenEXR image.",
                "Replace the damaged file or recapture the affected frame.",
                sequence=sequence,
                frame=frame,
                path=relative_path,
                observed=str(error),
                expected="readable OpenEXR image",
            )
            continue

        valid_dimensions.add((info.width, info.height))
        observed_channels.add(info.channels)
        actual_channels = set(info.channels)
        selected_channels = expected_channels
        channel_match = expected_channels <= actual_channels
        if not channel_match and len(expected_channels) == 1 and channel_aliases & actual_channels:
            selected_channels = {sorted(channel_aliases & actual_channels)[0]}
            channel_match = True
        if not channel_match:
            _finding(
                report,
                "error",
                "exr.channels.mismatch",
                f"{lane_name} frame does not contain the channels required by its profile.",
                "Export the lane using the documented channel layout.",
                sequence=sequence,
                frame=frame,
                path=relative_path,
                observed=sorted(actual_channels),
                expected={
                    "required": sorted(expected_channels),
                    "accepted_single_channel_aliases": sorted(channel_aliases),
                },
            )
            selected_channels = actual_channels

        mismatched_pixel_types = {
            channel: info.pixel_types[channel].rsplit(".", 1)[-1]
            for channel in sorted(selected_channels & actual_channels)
            if (
                expected_pixel_types
                and info.pixel_types[channel].rsplit(".", 1)[-1] not in expected_pixel_types
            )
        }
        if mismatched_pixel_types:
            _finding(
                report,
                "error",
                "exr.pixel_type.mismatch",
                f"{lane_name} frame uses an unsupported EXR pixel type.",
                "Export required channels using a floating-point EXR pixel type.",
                sequence=sequence,
                frame=frame,
                path=relative_path,
                observed=mismatched_pixel_types,
                expected=sorted(expected_pixel_types),
            )

        non_finite = {
            channel: {
                key: info.statistics[channel][key]
                for key in ("nan", "posinf", "neginf")
            }
            for channel in sorted(selected_channels & actual_channels)
            if any(info.statistics[channel][key] for key in ("nan", "posinf", "neginf"))
        }
        if non_finite:
            _finding(
                report,
                "error",
                "exr.non_finite",
                f"{lane_name} frame contains NaN or infinite values.",
                "Fix the capture source or recapture the affected frame without invalid pixels.",
                sequence=sequence,
                frame=frame,
                path=relative_path,
                observed=non_finite,
                expected="all required channel values are finite",
            )

        if value_range:
            minimum, maximum = (float(value) for value in value_range)
            outside = {}
            for channel in sorted(selected_channels & actual_channels):
                statistics = info.statistics[channel]
                observed_min = statistics["min"]
                observed_max = statistics["max"]
                if observed_min is None or observed_max is None:
                    continue
                if observed_min < minimum or observed_max > maximum:
                    outside[channel] = {
                        "min": observed_min,
                        "max": observed_max,
                    }
            if outside:
                _finding(
                    report,
                    "error",
                    "exr.value_range",
                    f"{lane_name} frame contains values outside the profile range.",
                    "Verify the exported value convention and scaling before using the dataset.",
                    sequence=sequence,
                    frame=frame,
                    path=relative_path,
                    observed=outside,
                    expected={"min": minimum, "max": maximum},
                )

        for channel in sorted(selected_channels & actual_channels):
            statistics = info.statistics[channel]
            observed_min = statistics["min"]
            observed_max = statistics["max"]
            if observed_min is None or observed_max is None:
                continue
            extrema = channel_extrema.setdefault(
                channel,
                {"min": observed_min, "max": observed_max},
            )
            extrema["min"] = min(extrema["min"], observed_min)
            extrema["max"] = max(extrema["max"], observed_max)

    if len(valid_dimensions) > 1:
        _finding(
            report,
            "error",
            "lane.dimensions.inconsistent",
            f"{lane_name} frames do not all have the same dimensions.",
            "Recapture or resize the lane so every frame uses one consistent resolution.",
            sequence=sequence,
            path=None,
            observed=[{"width": width, "height": height} for width, height in sorted(valid_dimensions)],
            expected="one width and height for the entire lane",
        )

    if lane_spec.get("content") == "motion" and expected_channels <= channel_extrema.keys():
        constant_values = {
            channel: channel_extrema[channel]["min"]
            for channel in sorted(expected_channels)
            if channel_extrema[channel]["min"] == channel_extrema[channel]["max"]
        }
        if len(constant_values) == len(expected_channels) and any(
            abs(value) > 1e-8 for value in constant_values.values()
        ):
            _finding(
                report,
                "warning",
                "motion.constant_non_zero",
                f"{lane_name} contains one constant non-zero vector across the scanned sequence.",
                "Confirm the capture with a warp diagnostic before applying any motion-vector correction.",
                sequence=sequence,
                observed=constant_values,
                expected="motion varies spatially or temporally, or is identically zero",
            )

    dimensions = next(iter(valid_dimensions)) if len(valid_dimensions) == 1 else None
    return {
        "dimensions": (
            {"width": dimensions[0], "height": dimensions[1]}
            if dimensions is not None
            else None
        ),
        "channel_layouts": [list(channels) for channels in sorted(observed_channels)],
        "value_ranges": channel_extrema,
    }


def _validate_jitter(
    report: ValidationReport,
    metadata: dict[str, Any],
    sequence: str,
    metadata_path: Path,
    frame_numbers: list[int],
    render_dimensions: dict[str, int] | None,
    jitter_spec: dict[str, Any],
) -> dict[str, Any] | None:
    if report.use_case not in jitter_spec.get("required_use_cases", []):
        return None

    relative_metadata_path = _relative_path(Path(report.dataset), metadata_path)
    if render_dimensions is None:
        _finding(
            report,
            "error",
            "jitter.render_dimensions.missing",
            "Jitter cannot be converted to render-pixel units because render dimensions are unavailable.",
            "Restore the required render-resolution image lanes.",
            sequence=sequence,
            path=relative_metadata_path,
            observed=None,
            expected="known render width and height",
        )
        return None

    indices = metadata.get("UpscalingRatiosIndices")
    scale_index_key = f"{report.scale.replace('.', '_')}_index"
    scale_index = indices.get(scale_index_key) if isinstance(indices, dict) else None
    if scale_index is None:
        _finding(
            report,
            "error",
            "jitter.scale_index.missing",
            "The selected scale has no jitter index in UpscalingRatiosIndices.",
            "Add the selected scale mapping or choose a scale present in the metadata.",
            sequence=sequence,
            path=relative_metadata_path,
            observed=indices,
            expected=scale_index_key,
        )
        return None
    if not _is_non_negative_int(scale_index):
        _finding(
            report,
            "error",
            "jitter.scale_index.invalid",
            "The selected scale's jitter index is not a non-negative integer.",
            "Set the scale index to the matching zero-based jitter array position.",
            sequence=sequence,
            path=relative_metadata_path,
            observed=scale_index,
            expected="non-negative integer",
        )
        return None

    metadata_by_frame = _frame_metadata_by_index(metadata)
    selected_frames = _frames_for_cadence(
        frame_numbers,
        jitter_spec.get("cadence", "all"),
    )

    samples: list[tuple[int, float, float]] = []
    jitter_keys = jitter_spec["keys"]
    minimum, maximum = (float(value) for value in jitter_spec["range_pixels"])
    width = render_dimensions["width"]
    height = render_dimensions["height"]

    for frame_number in selected_frames:
        frame_metadata = metadata_by_frame.get(frame_number)
        if frame_metadata is None:
            continue
        jitter_values = None
        jitter_key = None
        present_jitter_values = {}
        for candidate_key in jitter_keys:
            if candidate_key not in frame_metadata:
                continue
            candidate_values = frame_metadata[candidate_key]
            present_jitter_values[candidate_key] = candidate_values
            if isinstance(candidate_values, list) and scale_index < len(candidate_values):
                jitter_key = candidate_key
                jitter_values = candidate_values
                break
        if jitter_values is None:
            if present_jitter_values:
                _finding(
                    report,
                    "error",
                    "jitter.scale_value.missing",
                    "Frame jitter does not contain the selected scale entry.",
                    "Export one jitter entry for every scale index declared by the dataset.",
                    sequence=sequence,
                    frame=frame_number,
                    path=relative_metadata_path,
                    observed=present_jitter_values,
                    expected={"scale_index": scale_index},
                )
            else:
                _finding(
                    report,
                    "error",
                    "jitter.missing",
                    "Frame metadata does not contain a supported jitter value.",
                    "Export the per-ratio jitter used to render this frame.",
                    sequence=sequence,
                    frame=frame_number,
                    path=relative_metadata_path,
                    observed=sorted(frame_metadata),
                    expected=jitter_keys,
                )
            continue

        jitter = jitter_values[scale_index]
        if not isinstance(jitter, dict):
            _finding(
                report,
                "error",
                "jitter.value.invalid",
                "Frame jitter entry is not an object.",
                "Store jitter as an object with numeric X and Y fields.",
                sequence=sequence,
                frame=frame_number,
                path=relative_metadata_path,
                observed=jitter,
                expected={"X": "number", "Y": "number"},
            )
            continue
        jitter_x = jitter.get("X")
        jitter_y = jitter.get("Y")
        if (
            not isinstance(jitter_x, (int, float))
            or isinstance(jitter_x, bool)
            or not isinstance(jitter_y, (int, float))
            or isinstance(jitter_y, bool)
        ):
            _finding(
                report,
                "error",
                "jitter.value.invalid",
                "Frame jitter X or Y value is not numeric.",
                "Store finite numeric X and Y jitter values.",
                sequence=sequence,
                frame=frame_number,
                path=relative_metadata_path,
                observed={"X": jitter_x, "Y": jitter_y},
                expected={"X": "number", "Y": "number"},
            )
            continue

        pixel_x = float(jitter_x) * width
        pixel_y = float(jitter_y) * height
        if not math.isfinite(pixel_x) or not math.isfinite(pixel_y):
            _finding(
                report,
                "error",
                "jitter.non_finite",
                "Frame jitter contains a NaN or infinite value.",
                "Fix the jitter generator or recapture the affected frame.",
                sequence=sequence,
                frame=frame_number,
                path=relative_metadata_path,
                observed={"x": pixel_x, "y": pixel_y},
                expected="finite render-pixel jitter",
            )
            continue
        samples.append((frame_number, pixel_x, pixel_y))
        if pixel_x < minimum or pixel_x > maximum or pixel_y < minimum or pixel_y > maximum:
            _finding(
                report,
                "error",
                "jitter.out_of_range",
                "Frame jitter is outside the profile's render-pixel range.",
                "Verify jitter normalization, scale selection, and sign before recapturing.",
                sequence=sequence,
                frame=frame_number,
                path=relative_metadata_path,
                observed={"x": pixel_x, "y": pixel_y},
                expected={"min": minimum, "max": maximum, "units": "render pixels"},
            )

    expected_sample_count = len(selected_frames)
    if len(samples) != expected_sample_count:
        _finding(
            report,
            "error",
            "jitter.sample_count.mismatch",
            "The number of valid jitter samples does not match the selected metadata timeline.",
            "Provide one valid jitter sample for every frame selected by the profile cadence.",
            sequence=sequence,
            path=relative_metadata_path,
            observed={"valid_jitter_samples": len(samples)},
            expected={"selected_metadata_frames": expected_sample_count},
        )

    declared_frame_count = metadata.get("Samples", {}).get("Count") if isinstance(
        metadata.get("Samples"), dict
    ) else None
    if not samples:
        return {
            "declared_frame_count": declared_frame_count,
            "expected_sample_count": expected_sample_count,
            "frame_count": 0,
            "unique_count": 0,
        }

    all_points = [(x, y) for _, x, y in samples]
    phase_length = _repeating_phase_length(all_points)
    points = np.asarray(all_points[:phase_length], dtype=np.float64)
    rounded_points = [tuple(round(value, 8) for value in point) for point in points]
    unique_count = len(set(rounded_points))
    duplicate_fraction = 1.0 - (unique_count / phase_length)
    mean_x, mean_y = points.mean(axis=0)
    span_x, span_y = points.max(axis=0) - points.min(axis=0)
    quadrants = {
        (1 if x >= 0.0 else -1, 1 if y >= 0.0 else -1)
        for x, y in points
    }
    nearest_spacing = None
    if len(points) > 1:
        nearest_spacing = min(
            float(np.linalg.norm(points[left] - points[right]))
            for left in range(len(points))
            for right in range(left + 1, len(points))
        )
    correlation = None
    if len(points) > 1 and float(np.std(points[:, 0])) > 0.0 and float(np.std(points[:, 1])) > 0.0:
        correlation = float(np.corrcoef(points[:, 0], points[:, 1])[0, 1])

    min_phase_length = int(jitter_spec.get("min_phase_length", 1))
    if phase_length < min_phase_length:
        _finding(
            report,
            "warning",
            "jitter.phase_length",
            "The inferred jitter phase is shorter than the profile minimum.",
            "Use a longer jitter phase with adequate spatial coverage.",
            sequence=sequence,
            path=relative_metadata_path,
            observed={"phase_length": phase_length},
            expected={"minimum_phase_length": min_phase_length},
        )

    max_duplicate_fraction = float(jitter_spec.get("max_duplicate_fraction", 1.0))
    if duplicate_fraction > max_duplicate_fraction:
        _finding(
            report,
            "warning",
            "jitter.duplicates",
            "Jitter sequence contains too many duplicate sample positions.",
            "Use a longer, non-repeating sequence or correct repeated frame association.",
            sequence=sequence,
            path=relative_metadata_path,
            observed={"unique": unique_count, "phase_length": phase_length, "duplicate_fraction": duplicate_fraction},
            expected={"maximum_duplicate_fraction": max_duplicate_fraction},
        )

    max_abs_mean = float(jitter_spec.get("max_abs_mean_pixels", math.inf))
    if abs(mean_x) > max_abs_mean or abs(mean_y) > max_abs_mean:
        _finding(
            report,
            "warning",
            "jitter.centering",
            "Jitter sequence is not sufficiently centered around zero.",
            "Remove systematic jitter bias and regenerate the sequence.",
            sequence=sequence,
            path=relative_metadata_path,
            observed={"mean_x": float(mean_x), "mean_y": float(mean_y)},
            expected={"maximum_absolute_mean": max_abs_mean, "units": "render pixels"},
        )

    min_span = float(jitter_spec.get("min_span_pixels", 0.0))
    if span_x < min_span or span_y < min_span:
        _finding(
            report,
            "warning",
            "jitter.coverage",
            "Jitter sequence does not cover enough of the sample area.",
            "Use a sequence with wider and more even X/Y coverage.",
            sequence=sequence,
            path=relative_metadata_path,
            observed={"span_x": float(span_x), "span_y": float(span_y)},
            expected={"minimum_span": min_span, "units": "render pixels"},
        )

    min_quadrants = int(jitter_spec.get("min_quadrants", 0))
    if phase_length >= min_quadrants and len(quadrants) < min_quadrants:
        _finding(
            report,
            "warning",
            "jitter.quadrant_coverage",
            "Jitter sequence does not cover all required quadrants.",
            "Use an unbiased sequence that samples positive and negative X/Y offsets.",
            sequence=sequence,
            path=relative_metadata_path,
            observed={"quadrants": len(quadrants)},
            expected={"minimum_quadrants": min_quadrants},
        )

    min_spacing = float(jitter_spec.get("min_nearest_spacing_pixels", 0.0))
    if nearest_spacing is not None and nearest_spacing < min_spacing:
        _finding(
            report,
            "warning",
            "jitter.spacing",
            "Jitter samples are closer together than the profile threshold.",
            "Regenerate the sequence with better-distributed sample positions.",
            sequence=sequence,
            path=relative_metadata_path,
            observed={"minimum_spacing": nearest_spacing},
            expected={"minimum_spacing": min_spacing, "units": "render pixels"},
        )

    max_abs_correlation = float(jitter_spec.get("max_abs_correlation", 1.0))
    if correlation is not None and abs(correlation) > max_abs_correlation:
        _finding(
            report,
            "warning",
            "jitter.correlation",
            "Jitter X and Y values are too strongly correlated.",
            "Use a two-dimensional low-discrepancy sequence without diagonal bias.",
            sequence=sequence,
            path=relative_metadata_path,
            observed={"correlation": correlation},
            expected={"maximum_absolute_correlation": max_abs_correlation},
        )

    return {
        "declared_frame_count": declared_frame_count,
        "expected_sample_count": expected_sample_count,
        "frame_count": len(samples),
        "phase_length": phase_length,
        "unique_count": unique_count,
        "range_pixels": {
            "min_x": float(points[:, 0].min()),
            "max_x": float(points[:, 0].max()),
            "min_y": float(points[:, 1].min()),
            "max_y": float(points[:, 1].max()),
        },
        "mean_pixels": {"x": float(mean_x), "y": float(mean_y)},
        "span_pixels": {"x": float(span_x), "y": float(span_y)},
        "quadrants": len(quadrants),
        "minimum_spacing_pixels": nearest_spacing,
        "xy_correlation": correlation,
    }


def _validate_exposure(
    report: ValidationReport,
    metadata: dict[str, Any],
    sequence: str,
    metadata_path: Path,
    frame_numbers: list[int],
    exposure_spec: dict[str, Any],
    exposure_override: float | None,
) -> dict[str, Any] | None:
    if report.use_case not in exposure_spec.get("required_use_cases", []):
        return None

    relative_metadata_path = _relative_path(Path(report.dataset), metadata_path)
    metadata_by_frame = _frame_metadata_by_index(metadata)
    selected_frames = _frames_for_cadence(
        frame_numbers,
        exposure_spec.get("cadence", "all"),
    )

    field_name = exposure_spec.get("field", "Exposure")
    values: list[float] = []
    minimum_exclusive = exposure_spec.get("minimum_exclusive")
    minimum_inclusive = exposure_spec.get("minimum_inclusive")
    warning_maximum = exposure_spec.get("warning_maximum")

    if exposure_override is not None:
        config_path = _relative_path(Path(report.dataset), Path(report.dataset) / "config.json")
        if minimum_exclusive is not None and exposure_override <= float(minimum_exclusive):
            _finding(
                report,
                "error",
                "exposure.override.non_positive",
                "Dataset config Exposure does not satisfy the profile's positive-value requirement.",
                "Set config.json Exposure to the positive value used by the presentation tonemapper.",
                sequence=sequence,
                path=config_path,
                observed=exposure_override,
                expected=f"finite scalar greater than {float(minimum_exclusive)}",
            )
        elif minimum_inclusive is not None and exposure_override < float(minimum_inclusive):
            _finding(
                report,
                "error",
                "exposure.override.out_of_range",
                "Dataset config Exposure is below the profile range.",
                "Verify the config.json Exposure representation.",
                sequence=sequence,
                path=config_path,
                observed=exposure_override,
                expected={"minimum": float(minimum_inclusive)},
            )
        if warning_maximum is not None and exposure_override > float(warning_maximum):
            _finding(
                report,
                "warning",
                "exposure.override.implausible",
                "Dataset config Exposure is above the profile's diagnostic threshold.",
                "Verify that config.json Exposure is stored in linear scalar form rather than log exposure.",
                sequence=sequence,
                path=config_path,
                observed=exposure_override,
                expected={"maximum_recommended": float(warning_maximum)},
            )
        return {
            "frame_count": len(selected_frames),
            "minimum": exposure_override,
            "maximum": exposure_override,
        }

    for frame_number in selected_frames:
        frame_metadata = metadata_by_frame.get(frame_number)
        if frame_metadata is None:
            continue
        if field_name not in frame_metadata:
            _finding(
                report,
                "error",
                "exposure.missing",
                "Frame metadata does not contain the required scalar exposure.",
                "Export the exposure used for tonemapping this frame.",
                sequence=sequence,
                frame=frame_number,
                path=relative_metadata_path,
                observed=None,
                expected=field_name,
            )
            continue
        value = frame_metadata[field_name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            _finding(
                report,
                "error",
                "exposure.invalid",
                "Frame exposure is not a numeric scalar.",
                "Store exposure as one finite numeric value per frame.",
                sequence=sequence,
                frame=frame_number,
                path=relative_metadata_path,
                observed=value,
                expected="numeric scalar",
            )
            continue
        exposure = float(value)
        if not math.isfinite(exposure):
            _finding(
                report,
                "error",
                "exposure.non_finite",
                "Frame exposure is NaN or infinite.",
                "Fix exposure capture and recapture the affected frame.",
                sequence=sequence,
                frame=frame_number,
                path=relative_metadata_path,
                observed=exposure,
                expected="finite scalar",
            )
            continue
        values.append(exposure)

        if minimum_exclusive is not None and exposure <= float(minimum_exclusive):
            threshold = float(minimum_exclusive)
            _finding(
                report,
                "error",
                "exposure.non_positive",
                "Frame exposure does not satisfy the profile's positive-value requirement.",
                "Provide the positive exposure used by the presentation tonemapper.",
                sequence=sequence,
                frame=frame_number,
                path=relative_metadata_path,
                observed=exposure,
                expected=f"finite scalar greater than {threshold}",
            )
        elif minimum_inclusive is not None and exposure < float(minimum_inclusive):
            threshold = float(minimum_inclusive)
            _finding(
                report,
                "error",
                "exposure.out_of_range",
                "Frame exposure is below the profile range.",
                "Verify the exposure representation and recapture the affected frame.",
                sequence=sequence,
                frame=frame_number,
                path=relative_metadata_path,
                observed=exposure,
                expected={"minimum": threshold},
            )

        if warning_maximum is not None and exposure > float(warning_maximum):
            _finding(
                report,
                "warning",
                "exposure.implausible",
                "Frame exposure is above the profile's diagnostic threshold.",
                "Verify that exposure is stored in linear scalar form rather than log exposure.",
                sequence=sequence,
                frame=frame_number,
                path=relative_metadata_path,
                observed=exposure,
                expected={"maximum_recommended": float(warning_maximum)},
            )

    return {
        "frame_count": len(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _validate_sequence(
    report: ValidationReport,
    profile: dict[str, Any],
    metadata_path: Path,
    max_image_pixels: int | None,
    exposure_override: float | None,
) -> None:
    dataset_root = Path(report.dataset)
    sequence = metadata_path.stem
    relative_metadata_path = _relative_path(dataset_root, metadata_path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        report.complete_scan = False
        _finding(
            report,
            "error",
            "metadata.encoding.invalid",
            "Sequence metadata is not valid UTF-8.",
            "Rewrite or regenerate the sequence metadata as UTF-8 JSON.",
            sequence=sequence,
            path=relative_metadata_path,
            observed={"encoding": "utf-8", "byte_offset": error.start},
            expected="valid UTF-8 JSON",
        )
        return
    except OSError as error:
        report.complete_scan = False
        _finding(
            report,
            "error",
            "metadata.unreadable",
            "Sequence metadata cannot be read.",
            "Restore file permissions or replace the damaged metadata file.",
            sequence=sequence,
            path=relative_metadata_path,
            observed=str(error),
            expected="readable UTF-8 JSON",
        )
        return
    except json.JSONDecodeError as error:
        report.complete_scan = False
        _finding(
            report,
            "error",
            "metadata.json.invalid",
            "Sequence metadata is not valid JSON.",
            "Fix the JSON syntax or regenerate the sequence metadata.",
            sequence=sequence,
            path=relative_metadata_path,
            observed=f"line {error.lineno}, column {error.colno}: {error.msg}",
            expected="valid JSON object",
        )
        return

    if not isinstance(metadata, dict):
        report.complete_scan = False
        _finding(
            report,
            "error",
            "metadata.root.invalid",
            "Sequence metadata root is not an object.",
            "Write the sequence metadata as a JSON object.",
            sequence=sequence,
            path=relative_metadata_path,
            observed=type(metadata).__name__,
            expected="object",
        )
        return

    metadata_spec = profile["metadata"]
    required_top_level = list(metadata_spec["required_top_level"])
    required_frame = list(metadata_spec.get("required_frame", []))
    if exposure_override is not None:
        exposure_field = profile.get("exposure", {}).get("field", "Exposure")
        required_frame = [
            field_name for field_name in required_frame if field_name != exposure_field
        ]
    if metadata_spec.get("camera_planes") == "versioned_or_legacy":
        if "Version" in metadata:
            required_frame.extend(("CameraNearPlane", "CameraFarPlane"))
        else:
            required_top_level.extend(("NearPlane", "FarPlane"))

    for key in required_top_level:
        if key not in metadata:
            _finding(
                report,
                "error",
                "metadata.field.missing",
                f"Required top-level metadata field {key!r} is missing.",
                f"Add {key!r} to the sequence metadata using the selected profile contract.",
                sequence=sequence,
                path=relative_metadata_path,
                observed=None,
                expected=key,
            )

    _validate_camera_planes(
        report,
        metadata,
        required_top_level,
        metadata_path,
        sequence,
        None,
    )
    frame_numbers = _metadata_frame_numbers(report, metadata, metadata_path, sequence)
    _validate_required_frame_fields(
        report,
        metadata,
        metadata_path,
        sequence,
        required_frame,
    )
    validated_target_resolution = _validate_target_resolution(
        report,
        metadata,
        metadata_path,
        sequence,
    )
    lane_reports = []
    for lane_name, configured_lane_spec in profile["lanes"].items():
        lane_spec = {
            **profile.get("lane_defaults", {}),
            **configured_lane_spec,
        }
        required = report.use_case in lane_spec.get("required_use_cases", [])
        recommended = report.use_case in lane_spec.get("recommended_use_cases", [])
        lane_directory = _resolve_lane_directory(
            dataset_root,
            lane_spec["paths"],
            sequence,
            report.scale,
        )
        if lane_directory is None:
            if required:
                _finding(
                    report,
                    "error",
                    f"lane.{lane_name}.missing",
                    f"Required {lane_name} lane is missing.",
                    f"Capture or restore the {lane_name} files at one of the profile-supported paths.",
                    sequence=sequence,
                    path=None,
                    observed=None,
                    expected=[
                        path.format(sequence=sequence, scale=report.scale) for path in lane_spec["paths"]
                    ],
                )
            elif recommended:
                _finding(
                    report,
                    "warning",
                    f"lane.{lane_name}.recommended_missing",
                    f"Recommended {lane_name} lane is missing for the selected use case.",
                    f"Capture or generate the {lane_name} files to enable complete diagnostics.",
                    sequence=sequence,
                    path=None,
                    observed=None,
                    expected=[
                        path.format(sequence=sequence, scale=report.scale) for path in lane_spec["paths"]
                    ],
                )
            continue

        indexed = _indexed_exr_files(
            report,
            dataset_root,
            lane_directory,
            sequence,
            lane_name,
        )
        actual_frames = sorted(indexed)
        if required and not actual_frames:
            _finding(
                report,
                "error",
                f"lane.{lane_name}.empty",
                f"Required {lane_name} lane contains no numeric EXR frames.",
                f"Restore the {lane_name} frame files for sequence {sequence}.",
                sequence=sequence,
                path=_relative_path(dataset_root, lane_directory),
                observed=0,
                expected="one or more frames",
            )

        accepted_cadences = lane_spec.get(
            "accepted_cadences",
            [lane_spec.get("cadence", "all")],
        )
        expected_frames_by_cadence = {
            cadence: _frames_for_cadence(frame_numbers, cadence)
            for cadence in accepted_cadences
        }
        matched_cadence = next(
            (
                cadence
                for cadence, expected_frames in expected_frames_by_cadence.items()
                if actual_frames == expected_frames
            ),
            None,
        )
        if actual_frames and frame_numbers and matched_cadence is None:
            closest_expected_frames = min(
                expected_frames_by_cadence.values(),
                key=lambda candidate: len(set(actual_frames) ^ set(candidate)),
            )
            affected_frames = sorted(set(actual_frames) ^ set(closest_expected_frames))
            expected_frames: Any
            if len(expected_frames_by_cadence) == 1:
                expected_frames = next(iter(expected_frames_by_cadence.values()))
            else:
                expected_frames = expected_frames_by_cadence
            _finding(
                report,
                "error",
                f"lane.{lane_name}.frame_alignment",
                f"{lane_name} frames do not match the profile timeline.",
                f"Recapture or restore {lane_name} so its numeric frames follow the selected profile cadence.",
                sequence=sequence,
                frame=affected_frames[0] if affected_frames else None,
                path=_relative_path(dataset_root, lane_directory),
                observed=actual_frames,
                expected=expected_frames,
            )

        report.files_scanned += len(indexed)
        exr_report = _validate_exr_files(
            report,
            dataset_root,
            sequence,
            lane_name,
            lane_spec,
            indexed,
            max_image_pixels,
        )
        lane_reports.append(
            {
                "name": lane_name,
                "path": _relative_path(dataset_root, lane_directory),
                "dimension_group": lane_spec.get("dimension_group"),
                "frame_count": len(indexed),
                "first_frame": actual_frames[0] if actual_frames else None,
                "last_frame": actual_frames[-1] if actual_frames else None,
                "cadence": matched_cadence,
                **exr_report,
            }
        )

    dimension_groups: dict[str, dict[str, dict[str, int]]] = {}
    for lane_report in lane_reports:
        group = lane_report["dimension_group"]
        dimensions = lane_report["dimensions"]
        if group and dimensions is not None:
            dimension_groups.setdefault(group, {})[lane_report["name"]] = dimensions
    for group, lane_dimensions in dimension_groups.items():
        unique_dimensions = {
            (dimensions["width"], dimensions["height"]) for dimensions in lane_dimensions.values()
        }
        if len(unique_dimensions) > 1:
            _finding(
                report,
                "error",
                "dimensions.group_mismatch",
                f"Dataset lanes in the {group} dimension group do not match.",
                f"Export every {group} lane at the same resolution.",
                sequence=sequence,
                observed=lane_dimensions,
                expected=f"all {group} lanes use one width and height",
            )

    render_dimensions_by_lane = dimension_groups.get("render", {})
    unique_render_dimensions = {
        (dimensions["width"], dimensions["height"])
        for dimensions in render_dimensions_by_lane.values()
    }
    target_dimensions_by_lane = dimension_groups.get("target", {})
    unique_target_dimensions = {
        (dimensions["width"], dimensions["height"])
        for dimensions in target_dimensions_by_lane.values()
    }
    if validated_target_resolution is not None:
        target_width, target_height = validated_target_resolution
        expected_target_dimensions = {
            "width": target_width,
            "height": target_height,
        }
        if len(unique_target_dimensions) == 1:
            target_lane_width, target_lane_height = next(iter(unique_target_dimensions))
            observed_target_dimensions = {
                "width": target_lane_width,
                "height": target_lane_height,
            }
            if observed_target_dimensions != expected_target_dimensions:
                _finding(
                    report,
                    "error",
                    "dimensions.target_mismatch",
                    "Target-resolution lane dimensions do not match TargetResolution.",
                    "Export target-resolution images using the dimensions declared by the metadata.",
                    sequence=sequence,
                    observed=observed_target_dimensions,
                    expected=expected_target_dimensions,
                )

        if len(unique_render_dimensions) == 1:
            factor = _scale_factor(report.scale)
            expected_render_dimensions = {
                "width": round(target_width / factor),
                "height": round(target_height / factor),
            }
            render_width, render_height = next(iter(unique_render_dimensions))
            observed_render_dimensions = {
                "width": render_width,
                "height": render_height,
            }
            if observed_render_dimensions != expected_render_dimensions:
                _finding(
                    report,
                    "error",
                    "dimensions.scale_mismatch",
                    "Render dimensions do not match TargetResolution and the selected scale.",
                    "Export the selected scale at its expected render resolution or choose the matching scale.",
                    sequence=sequence,
                    observed=observed_render_dimensions,
                    expected={
                        **expected_render_dimensions,
                        "target": {
                            "width": target_width,
                            "height": target_height,
                        },
                        "scale": report.scale,
                    },
                )

    dimension_lane = profile.get("jitter", {}).get("dimension_lane")
    render_dimensions = next(
        (
            lane_report["dimensions"]
            for lane_report in lane_reports
            if lane_report["name"] == dimension_lane
        ),
        None,
    )
    jitter_report = _validate_jitter(
        report,
        metadata,
        sequence,
        metadata_path,
        frame_numbers,
        render_dimensions,
        profile.get("jitter", {}),
    )
    exposure_report = _validate_exposure(
        report,
        metadata,
        sequence,
        metadata_path,
        frame_numbers,
        profile.get("exposure", {}),
        exposure_override,
    )

    report.sequences.append(
        {
            "sequence": sequence,
            "metadata": relative_metadata_path,
            "frame_count": len(frame_numbers),
            "lanes": lane_reports,
            "jitter": jitter_report,
            "exposure": exposure_report,
        }
    )


def _validate_orphan_sequences(
    report: ValidationReport,
    profile: dict[str, Any],
    metadata_sequences: set[str],
) -> None:
    dataset_root = Path(report.dataset)
    reported: set[tuple[str, str]] = set()
    for lane_name, lane_spec in profile["lanes"].items():
        for template in lane_spec["paths"]:
            pattern = template.format(sequence="*", scale=report.scale)
            for lane_directory in sorted(dataset_root.glob(pattern)):
                if (
                    not lane_directory.is_dir()
                    or lane_directory.name in metadata_sequences
                ):
                    continue
                relative_lane_path = _relative_path(dataset_root, lane_directory)
                finding_key = (lane_name, relative_lane_path)
                if finding_key in reported:
                    continue
                reported.add(finding_key)
                if not lane_directory.name.isdigit():
                    _finding(
                        report,
                        "info",
                        "lane.directory.excluded",
                        f"{lane_name} directory has no numeric sequence identifier; its contents were not scanned.",
                        "Validate experimental results separately using a layout with matching sequence metadata.",
                        path=relative_lane_path,
                        observed=lane_directory.name,
                        expected="numeric sequence directory matching root metadata",
                    )
                    continue
                report.complete_scan = False
                _finding(
                    report,
                    "error",
                    f"lane.{lane_name}.orphan_sequence",
                    f"{lane_name} lane contains a sequence with no matching root metadata.",
                    "Add the matching numeric sequence JSON or remove the orphaned lane directory.",
                    sequence=lane_directory.name,
                    path=relative_lane_path,
                    observed=lane_directory.name,
                    expected="matching <sequence>.json at the dataset root",
                )


def validate_dataset(
    dataset_path: str | Path,
    profile_name: str,
    *,
    use_case: str = "replay",
    scale: str = "x2",
    warning_policy: str = "allow",
    max_image_pixels: int | None = None,
    exposure_override: float | None = None,
) -> ValidationReport:
    if use_case not in USE_CASES:
        raise ValidationError(f"Unknown use case {use_case!r}; choose from {sorted(USE_CASES)}")
    if warning_policy not in WARNING_POLICIES:
        raise ValidationError(
            f"Unknown warning policy {warning_policy!r}; choose from {sorted(WARNING_POLICIES)}"
        )
    if max_image_pixels is not None and max_image_pixels <= 0:
        raise ValidationError("max_image_pixels must be positive when provided")
    if exposure_override is not None:
        if (
            not isinstance(exposure_override, (int, float))
            or isinstance(exposure_override, bool)
            or not math.isfinite(float(exposure_override))
            or float(exposure_override) <= 0.0
        ):
            raise ValidationError("exposure_override must be a finite value greater than zero")
        exposure_override = float(exposure_override)
    if not scale.startswith("x") or len(scale) == 1:
        raise ValidationError(f"Scale must use dataset directory form such as 'x2', got {scale!r}")
    _scale_factor(scale)

    dataset_root = Path(dataset_path).expanduser().resolve()
    if not dataset_root.is_dir():
        raise ValidationError(f"Dataset path is not a directory: {dataset_root}")

    try:
        profile = load_profile(profile_name)
    except ProfileError as error:
        raise ValidationError(str(error)) from error
    if use_case not in profile["supported_use_cases"]:
        raise ValidationError(f"Profile {profile_name!r} does not support use case {use_case!r}")

    report = ValidationReport(
        dataset=str(dataset_root),
        profile=profile_name,
        use_case=use_case,
        scale=scale,
        warning_policy=warning_policy,
    )
    metadata_paths: list[Path] = []
    for metadata_path in sorted(
        path for path in dataset_root.glob("*.json") if path.stem.isdigit()
    ):
        try:
            resolved_metadata_path = metadata_path.resolve(strict=True)
        except OSError as error:
            report.complete_scan = False
            _finding(
                report,
                "error",
                "file.path.unreadable",
                "A sequence metadata path cannot be resolved.",
                "Repair or remove the broken metadata file entry.",
                sequence=metadata_path.stem,
                path=_relative_path(dataset_root, metadata_path),
                observed=str(error),
                expected="readable JSON file under the dataset root",
            )
            continue
        if dataset_root not in resolved_metadata_path.parents:
            report.complete_scan = False
            _finding(
                report,
                "error",
                "file.path.outside_dataset",
                "A sequence metadata file resolves outside the selected dataset root.",
                "Replace external links with metadata stored inside the dataset.",
                sequence=metadata_path.stem,
                path=_relative_path(dataset_root, metadata_path),
                observed=str(resolved_metadata_path),
                expected=str(dataset_root),
            )
            continue
        if resolved_metadata_path.is_file():
            metadata_paths.append(metadata_path)

    _validate_orphan_sequences(
        report,
        profile,
        {path.stem for path in metadata_paths},
    )
    if not metadata_paths:
        report.complete_scan = False
        _finding(
            report,
            "error",
            "dataset.sequence_metadata.missing",
            "No numeric sequence metadata files were found at the dataset root.",
            "Add sequence metadata such as 0000.json at the dataset root.",
            path=".",
            observed=0,
            expected="one or more numeric .json files",
        )
        return report

    for metadata_path in metadata_paths:
        _validate_sequence(
            report,
            profile,
            metadata_path,
            max_image_pixels,
            exposure_override,
        )
    return report
