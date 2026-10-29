# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import OpenEXR


MAX_IMAGE_PIXELS = 64 * 1024 * 1024


class ExrReadError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExrInfo:
    width: int
    height: int
    channels: tuple[str, ...]
    pixel_types: dict[str, str]
    statistics: dict[str, dict[str, Any]]


def _point_coordinates(point: Any) -> tuple[int, int]:
    if hasattr(point, "x") and hasattr(point, "y"):
        return int(point.x), int(point.y)
    values = np.asarray(point, dtype=np.int64).reshape(-1)
    if values.size != 2:
        raise ValueError("OpenEXR data-window point must contain two coordinates")
    return int(values[0]), int(values[1])


def exr_dimensions(image: Any) -> tuple[int, int]:
    try:
        data_window = image.header()["dataWindow"]
        if hasattr(data_window, "min") and hasattr(data_window, "max"):
            minimum = data_window.min
            maximum = data_window.max
        else:
            minimum, maximum = data_window
        min_x, min_y = _point_coordinates(minimum)
        max_x, max_y = _point_coordinates(maximum)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ExrReadError(f"OpenEXR image has an invalid data window: {error}") from error

    width = max_x - min_x + 1
    height = max_y - min_y + 1
    if width <= 0 or height <= 0:
        raise ExrReadError(f"OpenEXR image has invalid dimensions: {width}x{height}")
    return width, height


def read_exr(path: Path, *, max_pixels: int | None = None) -> ExrInfo:
    if max_pixels is not None and max_pixels <= 0:
        raise ValueError("max_pixels must be positive when provided")

    try:
        with OpenEXR.File(str(path), separate_channels=True) as image:
            width, height = exr_dimensions(image)
            if max_pixels is not None and width > max_pixels // height:
                raise ExrReadError(
                    f"OpenEXR dimensions are unsupported for diagnostics: {width}x{height}"
                )

            channels = image.channels()
            if not channels:
                raise ExrReadError("OpenEXR image contains no channels")

            shapes = {name: tuple(np.asarray(channel.pixels).shape) for name, channel in channels.items()}
            invalid_shapes = {name: shape for name, shape in shapes.items() if len(shape) != 2}
            if invalid_shapes:
                raise ExrReadError(f"OpenEXR channels are not two-dimensional: {invalid_shapes}")
            if len(set(shapes.values())) != 1:
                raise ExrReadError(f"OpenEXR channels have inconsistent dimensions: {shapes}")

            actual_height, actual_width = next(iter(shapes.values()))
            if (actual_width, actual_height) != (width, height):
                raise ExrReadError(
                    "OpenEXR channel dimensions do not match the image data window: "
                    f"{actual_width}x{actual_height} versus {width}x{height}"
                )
            statistics: dict[str, dict[str, Any]] = {}
            pixel_types: dict[str, str] = {}
            for name in sorted(channels):
                channel = channels[name]
                pixels = np.asarray(channel.pixels)
                finite = np.isfinite(pixels) if np.issubdtype(pixels.dtype, np.floating) else np.ones(
                    pixels.shape, dtype=bool
                )
                finite_values = pixels[finite]
                statistics[name] = {
                    "nan": int(np.isnan(pixels).sum()) if np.issubdtype(pixels.dtype, np.floating) else 0,
                    "posinf": int(np.isposinf(pixels).sum()) if np.issubdtype(pixels.dtype, np.floating) else 0,
                    "neginf": int(np.isneginf(pixels).sum()) if np.issubdtype(pixels.dtype, np.floating) else 0,
                    "min": float(finite_values.min()) if finite_values.size else None,
                    "max": float(finite_values.max()) if finite_values.size else None,
                }
                pixel_types[name] = str(channel.type())

            return ExrInfo(
                width=int(width),
                height=int(height),
                channels=tuple(sorted(channels)),
                pixel_types=pixel_types,
                statistics=statistics,
            )
    except ExrReadError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise ExrReadError(str(error)) from error
