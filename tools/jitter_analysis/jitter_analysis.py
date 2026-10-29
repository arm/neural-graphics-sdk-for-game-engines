#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

"""Evaluate dataset jitter suitability for TAA-like temporal accumulation."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JITTER_KEYS = ("QuantisedPerRatioJitter", "NormalizedPerRatioJitter")
DEFAULT_DPI = 300
MIN_EXPECTED_UNIQUE_FRACTION = 0.50
MAX_REPEAT_FREQUENCY_CV = 0.50
MAX_TEMPORAL_WINDOW_CENTROID = 0.20
MAX_CRITICAL_TEMPORAL_WINDOW_CENTROID = 0.50
MIN_TEMPORAL_WINDOW_FRAMES = 8
TEMPORAL_WINDOW_FRACTION = 0.25


@dataclass(frozen=True)
class JitterData:
    points: np.ndarray
    frame_ids: list[int]
    ratio: str
    source_size: tuple[int, int] | None
    target_size: tuple[int, int] | None
    source_resolution_from: str
    jitter_key: str
    jitter_sources: list[str]
    fallback_reasons: dict[int, str]
    declared_samples: int | None


@dataclass(frozen=True)
class Analysis:
    verdict: bool
    severity: str
    severity_reasons: list[str]
    score: int
    total_checks: int
    metrics: dict[str, float | int]
    checks: dict[str, bool]
    reasons: list[str]


def _ratio_name(value: str) -> str:
    value = value.strip().lower()
    if value == "auto":
        return value
    if value.startswith("x"):
        return value.replace(".", "_")
    return f"x{value.replace('.', '_')}"


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _select_ratio(metadata: dict[str, Any], requested: str) -> tuple[str, int]:
    ratio_indices = metadata.get("UpscalingRatiosIndices")
    if "UpscalingRatiosIndices" not in metadata:
        if requested not in ("auto", "x2"):
            raise ValueError("The metadata has no UpscalingRatiosIndices mapping")
        return "x2", 0
    if not isinstance(ratio_indices, dict) or not ratio_indices:
        raise ValueError("UpscalingRatiosIndices must be a non-empty object")

    available = {}
    for key, index in ratio_indices.items():
        if not key.endswith("_index"):
            continue
        if not _is_non_negative_int(index):
            raise ValueError(
                f"UpscalingRatiosIndices.{key} must be a non-negative integer"
            )
        available[key[: -len("_index")]] = index
    if not available:
        raise ValueError("UpscalingRatiosIndices has no ratio_index entries")
    ratio = min(available, key=available.get) if requested == "auto" else requested
    if ratio not in available:
        raise ValueError(
            f"Ratio {ratio!r} is unavailable; choose from {', '.join(sorted(available))}"
        )
    return ratio, available[ratio]


def _target_size(metadata: dict[str, Any]) -> tuple[int, int] | None:
    target = metadata.get("TargetResolution")
    if "TargetResolution" not in metadata:
        return None
    if not isinstance(target, dict) or not {"X", "Y"} <= target.keys():
        raise ValueError("TargetResolution must contain X and Y dimensions")
    width, height = target["X"], target["Y"]
    if not all(
        _is_non_negative_int(value) and value > 0 for value in (width, height)
    ):
        raise ValueError("TargetResolution dimensions must be positive integers")
    return width, height


def _ffx_nss_jitter_phase_count(render_size: int, display_size: int) -> int:
    """Match ffxNssGetJitterPhaseCount from the SDK."""
    if render_size <= 0 or display_size <= 0:
        raise ValueError("Render and display dimensions must be positive")
    return int(8.0 * (float(display_size) / float(render_size)) ** 2.0)


def _read_exr_size(path: Path) -> tuple[int, int]:
    try:
        import OpenEXR
    except ImportError as error:
        raise ValueError(
            "OpenEXR is required to read the source color resolution"
        ) from error

    exr = OpenEXR.InputFile(str(path))
    try:
        data_window = exr.header()["dataWindow"]
        return (
            data_window.max.x - data_window.min.x + 1,
            data_window.max.y - data_window.min.y + 1,
        )
    finally:
        if hasattr(exr, "close"):
            exr.close()


def _source_size(
    dataset_json: Path,
    ratio: str,
    sequence: str,
    width_override: int | None,
    height_override: int | None,
) -> tuple[tuple[int, int], str]:
    if (width_override is None) != (height_override is None):
        raise ValueError("--source-width and --source-height must be used together")
    if width_override is not None and height_override is not None:
        if width_override <= 0 or height_override <= 0:
            raise ValueError("Source dimensions must be positive")
        return (width_override, height_override), "command-line override"

    ratio_root = dataset_json.parent / ratio
    searched = []
    for color_name in ("colour", "color"):
        color_root = ratio_root / color_name
        sequence_root = color_root / sequence
        searched.append(sequence_root)
        candidates = (
            sorted(sequence_root.glob("*.exr"))
            if sequence_root.is_dir()
            else sorted(color_root.glob("*.exr"))
            if color_root.is_dir()
            else []
        )
        if candidates:
            source_file = candidates[0]
            return _read_exr_size(source_file), str(source_file)

    searched_text = ", ".join(str(path) for path in searched)
    raise ValueError(
        f"No low-resolution color EXR found for ratio {ratio} and sequence "
        f"{sequence}; searched {searched_text}. Use --source-width and "
        "--source-height only for metadata without input images."
    )


def _frame_id(frame: Any, position: int) -> int:
    if not isinstance(frame, dict):
        raise ValueError(f"Frames[{position}] must be an object")
    indices = [frame[key] for key in ("Frame", "FrameNumber") if key in frame]
    if not indices or not all(_is_non_negative_int(value) for value in indices):
        raise ValueError(
            f"Frames[{position}] needs a non-negative integer Frame or FrameNumber"
        )
    if len(indices) == 2 and indices[0] != indices[1]:
        raise ValueError(f"Frame and FrameNumber disagree: {indices[0]} != {indices[1]}")
    return indices[0]


def _jitter_pair(sample: Any) -> tuple[float, float] | None:
    if not isinstance(sample, dict):
        return None
    components = (sample.get("X"), sample.get("Y"))
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in components
    ):
        return None
    try:
        x, y = (float(value) for value in components)
    except OverflowError:
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _ratio_jitter(
    frame: dict[str, Any], key: str, ratio_index: int
) -> tuple[tuple[float, float] | None, str]:
    samples = frame.get(key)
    if not isinstance(samples, list) or ratio_index >= len(samples):
        return None, f"{key}[{ratio_index}] is missing or unavailable"
    pair = _jitter_pair(samples[ratio_index])
    if pair is None:
        return None, f"{key}[{ratio_index}] must contain finite numeric X and Y"
    return pair, ""


def _load_jitter(
    path: Path,
    requested_ratio: str,
    width_override: int | None,
    height_override: int | None,
) -> JitterData:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("The dataset JSON must contain an object")
    frames = metadata.get("Frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("The dataset JSON has no non-empty Frames array")

    indexed_frames = sorted(
        ((_frame_id(frame, position), frame) for position, frame in enumerate(frames)),
        key=lambda entry: entry[0],
    )
    for (previous, _), (current, _) in zip(indexed_frames, indexed_frames[1:]):
        if current == previous:
            raise ValueError(f"Duplicate frame index {current}")
        if current != previous + 1:
            raise ValueError(
                f"Frame gap between {previous} and {current}; analyze a contiguous interval "
                "so temporal windows do not bridge missing frames"
            )

    ratio, ratio_index = _select_ratio(metadata, requested_ratio)
    target_size = _target_size(metadata)
    per_ratio = any(key in frame for frame in frames for key in JITTER_KEYS)

    source_size = None
    source_resolution_from = "pixel-valued jitter"
    if per_ratio:
        if "UpscalingRatiosIndices" not in metadata:
            raise ValueError("Per-ratio jitter requires UpscalingRatiosIndices")
        source_size, source_resolution_from = _source_size(
            path, ratio, path.stem, width_override, height_override
        )

    points = []
    frame_ids = []
    jitter_sources = []
    fallback_reasons = {}
    for frame_id, frame in indexed_frames:
        if per_ratio:
            jitter_key = JITTER_KEYS[0]
            pair, reason = _ratio_jitter(frame, jitter_key, ratio_index)
            if pair is None:
                jitter_key = JITTER_KEYS[1]
                pair, fallback_error = _ratio_jitter(frame, jitter_key, ratio_index)
                if pair is None:
                    raise ValueError(f"Frame {frame_id}: {reason}; {fallback_error}")
                fallback_reasons[frame_id] = reason
        else:
            jitter_key = next(
                (
                    key for key in ("Jitter", "NonQuantisedJitter")
                    if isinstance(frame.get(key), dict)
                ),
                "Jitter",
            )
            pair = _jitter_pair(frame.get(jitter_key))
            if pair is None:
                raise ValueError(
                    f"Frame {frame_id}: {jitter_key} must contain finite numeric X and Y"
                )

        x, y = pair
        if source_size is not None:
            x *= source_size[0]
            y *= source_size[1]
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(
                f"Frame {frame_id}: jitter conversion produced non-finite pixels"
            )
        points.append((x, y))
        frame_ids.append(frame_id)
        jitter_sources.append(jitter_key)

    samples = metadata.get("Samples", {})
    if not isinstance(samples, dict):
        raise ValueError("Samples must be an object")
    declared_samples = samples.get("Count")
    if "Count" in samples and not (
        _is_non_negative_int(declared_samples) and declared_samples > 0
    ):
        raise ValueError("Samples.Count must be a positive integer")
    return JitterData(
        points=np.asarray(points, dtype=np.float64),
        frame_ids=frame_ids,
        ratio=ratio,
        source_size=source_size,
        target_size=target_size,
        source_resolution_from=source_resolution_from,
        jitter_key=jitter_sources[0] if len(set(jitter_sources)) == 1 else "mixed",
        jitter_sources=jitter_sources,
        fallback_reasons=fallback_reasons,
        declared_samples=declared_samples,
    )


def _unique_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.unique(
        np.round(points, decimals=8), axis=0, return_index=True, return_counts=True
    )


def _sequence_values(
    data: JitterData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, first_indices, counts = _unique_points(data.points)
    order = np.argsort(first_indices)
    first_indices = first_indices[order]
    return (
        data.points[first_indices],
        first_indices,
        counts[order],
        np.asarray(data.frame_ids)[first_indices],
    )


def _approximate_star_discrepancy(points: np.ndarray) -> float:
    unit = np.clip(points + 0.5, 0.0, 1.0)
    grid = np.linspace(0.0, 1.0, 33)
    discrepancy = 0.0
    for x in grid:
        for y in grid:
            observed = np.mean((unit[:, 0] <= x) & (unit[:, 1] <= y))
            discrepancy = max(discrepancy, abs(float(observed) - x * y))
    return discrepancy


def _nearest_neighbor_cv(points: np.ndarray) -> float:
    if len(points) < 3:
        return math.inf
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    mean = float(np.mean(nearest))
    return float(np.std(nearest) / mean) if mean > 0.0 else math.inf


def _classify_severity(
    unique_count: int,
    min_window_quadrants: int,
    max_window_centroid_bias: float,
    score: int,
    total_checks: int,
) -> tuple[str, list[str]]:
    reasons = []
    if unique_count <= 1:
        reasons.append(
            "the sequence contains no meaningful temporal sample variation "
            f"({unique_count} unique offset)"
        )
    if min_window_quadrants <= 1:
        reasons.append(
            "a temporal window covers only "
            f"{min_window_quadrants}/4 quadrants"
        )
    if max_window_centroid_bias >= MAX_CRITICAL_TEMPORAL_WINDOW_CENTROID:
        reasons.append(
            "a temporal window is biased by "
            f"{max_window_centroid_bias:.3f} pixels "
            f"(critical limit {MAX_CRITICAL_TEMPORAL_WINDOW_CENTROID:.2f})"
        )
    severity = (
        "ERROR"
        if reasons
        else "PASS"
        if score == total_checks
        else "WARNING"
    )
    return severity, reasons


def _analyze(data: JitterData) -> Analysis:
    points = data.points
    unique, _, repeat_counts = _unique_points(points)
    finite = bool(np.isfinite(points).all())
    in_bounds = float(np.mean(np.all(np.abs(points) <= 0.500001, axis=1)))
    centroid = np.mean(points, axis=0)
    centroid_distance = float(np.linalg.norm(centroid))
    spans = np.ptp(unique, axis=0)

    non_axis = unique[np.all(np.abs(unique) > 1e-8, axis=1)]
    quadrants = len({(int(x > 0.0), int(y > 0.0)) for x, y in non_axis})

    grid_size = max(2, min(8, math.ceil(math.sqrt(len(unique)))))
    bounded = np.clip(unique + 0.5, 0.0, 1.0 - np.finfo(float).eps)
    cells = np.floor(bounded * grid_size).astype(int)
    occupancy = len({tuple(cell) for cell in cells}) / float(grid_size**2)
    discrepancy = _approximate_star_discrepancy(unique)
    nearest_cv = _nearest_neighbor_cv(unique)
    correlation = (
        abs(float(np.corrcoef(unique[:, 0], unique[:, 1])[0, 1]))
        if len(unique) >= 3
        and np.std(unique[:, 0]) > 0.0
        and np.std(unique[:, 1]) > 0.0
        else 1.0
    )
    if data.source_size is not None and data.target_size is not None:
        expected_unique_width = _ffx_nss_jitter_phase_count(
            data.source_size[0], data.target_size[0]
        )
        expected_unique_height = _ffx_nss_jitter_phase_count(
            data.source_size[1], data.target_size[1]
        )
    else:
        expected_unique_width = data.declared_samples or len(points)
        expected_unique_height = expected_unique_width
    expected_unique = expected_unique_width
    expected_unique_fraction = len(unique) / float(max(1, expected_unique))
    mean_repeat_count = float(np.mean(repeat_counts))
    repeat_frequency_cv = (
        float(np.std(repeat_counts) / mean_repeat_count)
        if mean_repeat_count > 0.0
        else math.inf
    )
    temporal_window_size = max(
        1,
        min(
            len(points),
            max(
                MIN_TEMPORAL_WINDOW_FRAMES,
                math.ceil(expected_unique * TEMPORAL_WINDOW_FRACTION),
            ),
        ),
    )
    temporal_windows = [
        points[start : start + temporal_window_size]
        for start in range(len(points) - temporal_window_size + 1)
    ]
    max_window_centroid_bias = max(
        (
            float(np.linalg.norm(np.mean(window, axis=0)))
            for window in temporal_windows
        ),
        default=math.inf,
    )
    min_window_quadrants = min(
        (
            len(
                {
                    (int(x > 0.0), int(y > 0.0))
                    for x, y in window
                    if abs(x) > 1e-8 and abs(y) > 1e-8
                }
            )
            for window in temporal_windows
        ),
        default=0,
    )

    checks = {
        "finite values": finite,
        "inside one pixel": in_bounds >= 0.95,
        "centered": centroid_distance <= 0.10,
        "wide x/y span": bool(np.all(spans >= 0.70)),
        "all quadrants covered": quadrants == 4,
        "grid coverage": occupancy >= 0.45,
        "low discrepancy": discrepancy <= 0.30,
        "balanced spacing": nearest_cv <= 0.75,
        "low axis correlation": correlation <= 0.75,
        "balanced offset frequency": repeat_frequency_cv
        <= MAX_REPEAT_FREQUENCY_CV,
        "temporal window centering": max_window_centroid_bias
        <= MAX_TEMPORAL_WINDOW_CENTROID,
        "temporal window quadrants": min_window_quadrants == 4,
        "expected unique samples": expected_unique_fraction
        >= MIN_EXPECTED_UNIQUE_FRACTION,
    }
    essential = all(
        checks[name]
        for name in (
            "finite values",
            "inside one pixel",
            "centered",
            "wide x/y span",
            "all quadrants covered",
            "balanced offset frequency",
            "temporal window centering",
            "temporal window quadrants",
            "expected unique samples",
        )
    )
    score = int(sum(checks.values()))
    verdict = bool(essential and score >= 8)

    reasons = []
    if not checks["inside one pixel"]:
        reasons.append(f"{(1.0 - in_bounds):.1%} of samples are outside [-0.5, 0.5]")
    if not checks["centered"]:
        reasons.append(
            f"centroid is biased by {centroid_distance:.3f} pixels "
            f"(x={centroid[0]:+.3f}, y={centroid[1]:+.3f})"
        )
    if not checks["wide x/y span"]:
        reasons.append(f"sample span is only x={spans[0]:.3f}, y={spans[1]:.3f}")
    if not checks["all quadrants covered"]:
        reasons.append(f"only {quadrants}/4 pixel quadrants are covered")
    if not checks["expected unique samples"]:
        reasons.append(
            f"only {len(unique)}/{expected_unique} expected samples are unique "
            f"({expected_unique_fraction:.1%}; need at least "
            f"{MIN_EXPECTED_UNIQUE_FRACTION:.0%})"
        )
    if not checks["balanced offset frequency"]:
        reasons.append(
            f"offset repeat frequencies are unbalanced "
            f"(coefficient of variation={repeat_frequency_cv:.3f})"
        )
    if not checks["temporal window centering"]:
        reasons.append(
            f"a {temporal_window_size}-frame window is biased by "
            f"{max_window_centroid_bias:.3f} pixels "
            f"(limit {MAX_TEMPORAL_WINDOW_CENTROID:.2f})"
        )
    if not checks["temporal window quadrants"]:
        reasons.append(
            f"a {temporal_window_size}-frame window covers only "
            f"{min_window_quadrants}/4 quadrants"
        )
    if not checks["low discrepancy"]:
        reasons.append(f"star discrepancy is high ({discrepancy:.3f})")
    if not checks["low axis correlation"]:
        reasons.append(f"x/y correlation is high ({correlation:.3f})")

    severity, severity_reasons = _classify_severity(
        len(unique),
        min_window_quadrants,
        max_window_centroid_bias,
        score,
        len(checks),
    )

    metrics: dict[str, float | int] = {
        "frames": len(points),
        "unique_points": len(unique),
        "expected_unique": expected_unique,
        "expected_unique_width": expected_unique_width,
        "expected_unique_height": expected_unique_height,
        "expected_unique_fraction": expected_unique_fraction,
        "in_bounds_fraction": in_bounds,
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "span_x": float(spans[0]),
        "span_y": float(spans[1]),
        "quadrants": quadrants,
        "grid_size": grid_size,
        "grid_occupancy": occupancy,
        "star_discrepancy": discrepancy,
        "nearest_neighbor_cv": nearest_cv,
        "axis_correlation": correlation,
        "repeat_frequency_cv": repeat_frequency_cv,
        "temporal_window_size": temporal_window_size,
        "max_window_centroid_bias": max_window_centroid_bias,
        "min_window_quadrants": min_window_quadrants,
    }
    return Analysis(
        verdict,
        severity,
        severity_reasons,
        score,
        len(checks),
        metrics,
        checks,
        reasons,
    )


def _plot(data: JitterData, analysis: Analysis, output: Path, dpi: int) -> None:
    unique, _, counts, frame_ids = _sequence_values(data)

    figure_height = max(9.0, 0.32 * len(unique) + 2.5)
    fig, (ax, values_ax) = plt.subplots(
        1,
        2,
        figsize=(13, figure_height),
        gridspec_kw={"width_ratios": (3.0, 1.7)},
    )
    ax.add_patch(
        plt.Rectangle(
            (-0.5, -0.5), 1.0, 1.0, fill=False, edgecolor="0.35", linewidth=1.8
        )
    )
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.plot(
        0.0, 0.0, "o", markersize=7, markerfacecolor="white", markeredgecolor="black"
    )

    if len(unique) > 1:
        ax.plot(unique[:, 0], unique[:, 1], color="0.78", linewidth=1.0, zorder=1)
    colors = plt.cm.tab20(np.arange(len(unique)) % 20)
    ax.scatter(
        unique[:, 0],
        unique[:, 1],
        s=105,
        c=colors,
        edgecolors="black",
        linewidths=0.9,
        zorder=3,
    )
    for sequence_index, point in enumerate(unique):
        ax.annotate(
            str(sequence_index),
            xy=point,
            xytext=(7, -7),
            textcoords="offset points",
            fontsize=11,
        )

    ax.text(0.0, 0.0, " centre", fontsize=11, va="top")
    quadrant_labels = (
        (-0.25, -0.25, "top-left offset"),
        (0.25, -0.25, "top-right offset"),
        (-0.25, 0.25, "bottom-left offset"),
        (0.25, 0.25, "bottom-right offset"),
    )
    for x, y, label in quadrant_labels:
        ax.text(x, y, label, color="0.65", ha="center", va="center", fontsize=11)

    limit = max(0.58, float(np.max(np.abs(data.points))) + 0.08)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(limit, -limit)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.82", linewidth=0.8)
    ax.set_xlabel("x offset from pixel centre, pixels", fontsize=14)
    ax.set_ylabel("y offset from pixel centre, pixels", fontsize=14)
    status_text = {
        "PASS": "JITTER DISTRIBUTION CHECKS PASSED",
        "WARNING": "WARNING - Jitter sequence has room for improvement",
        "ERROR": "ERROR - Jitter may impair temporal accumulation",
    }[analysis.severity]
    ax.set_title(
        f"Jitter sample points relative to input pixel centre\n"
        f"{status_text}\n"
        f"score {analysis.score}/{analysis.total_checks}\n"
        f"repeat CV={analysis.metrics['repeat_frequency_cv']:.3f}, "
        f"{analysis.metrics['temporal_window_size']}-frame bias="
        f"{analysis.metrics['max_window_centroid_bias']:.3f}px\n"
        f"min window quadrants={analysis.metrics['min_window_quadrants']}/4",
        fontsize=14,
    )

    values_ax.axis("off")
    values_ax.set_title("Jitter offsets in sequence order", fontsize=16, pad=12)
    table_rows = [
        (
            str(index),
            str(int(frame_id)),
            f"{point[0]:+.6f}",
            f"{point[1]:+.6f}",
            str(int(count)),
        )
        for index, (point, frame_id, count) in enumerate(zip(unique, frame_ids, counts))
    ]
    table = values_ax.table(
        cellText=table_rows,
        colLabels=("#", "first frame", "x (px)", "y (px)", "repeat"),
        cellLoc="right",
        colLoc="right",
        loc="upper center",
        colWidths=(0.09, 0.22, 0.25, 0.25, 0.12),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(max(8.0, min(12.0, 280.0 / max(1, len(unique)))))
    table.scale(1.0, 1.35)
    for row, color in enumerate(colors, start=1):
        cell = table[row, 0]
        cell.set_facecolor(color)
        red, green, blue = color[:3]
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        cell.get_text().set_color("black" if luminance > 0.5 else "white")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _print_report(
    input_path: Path, output_path: Path, data: JitterData, analysis: Analysis
) -> None:
    metrics = analysis.metrics
    status_text = {
        "PASS": "PASS - all jitter distribution checks passed",
        "WARNING": "WARNING - Jitter sequence has room for improvement",
        "ERROR": "ERROR - jitter may impair temporal accumulation",
    }[analysis.severity]
    print(status_text)
    print(f"input: {input_path}")
    print(
        f"jitter: {data.jitter_key}, ratio={data.ratio}, "
        f"source_size={data.source_size or 'pixel offsets'}"
    )
    print(f"source resolution from: {data.source_resolution_from}")
    if data.source_size is not None:
        for key in JITTER_KEYS:
            selected_frames = [
                frame for frame, source in zip(data.frame_ids, data.jitter_sources)
                if source == key
            ]
            print(f"{key}: {len(selected_frames)} frames; frame_ids={selected_frames}")
        for frame, reason in data.fallback_reasons.items():
            print(f"  frame {frame}: using {JITTER_KEYS[1]} because {reason}")
    if data.target_size is not None:
        print(f"display_size: {data.target_size}")
    print(
        f"samples: {metrics['frames']} frames, {metrics['unique_points']} unique "
        f"(expected width={metrics['expected_unique_width']}, "
        f"height={metrics['expected_unique_height']}, "
        f"expected coverage={metrics['expected_unique_fraction']:.1%})"
    )
    points, _, counts, frame_ids = _sequence_values(data)
    print("jitter offsets in sequence order (pixels):")
    for index, (point, frame_id, count) in enumerate(zip(points, frame_ids, counts)):
        print(
            f"  [{index:3d}] first_frame={int(frame_id):4d} "
            f"x={point[0]:+.6f} y={point[1]:+.6f} repeat={int(count)}"
        )
    print(
        "distribution: "
        f"centroid=({metrics['centroid_x']:+.4f}, {metrics['centroid_y']:+.4f}), "
        f"span=({metrics['span_x']:.4f}, {metrics['span_y']:.4f}), "
        f"quadrants={metrics['quadrants']}/4, "
        f"in_bounds={metrics['in_bounds_fraction']:.1%}"
    )
    print(
        "quality: "
        f"grid_occupancy={metrics['grid_occupancy']:.1%}, "
        f"star_discrepancy={metrics['star_discrepancy']:.3f}, "
        f"spacing_cv={metrics['nearest_neighbor_cv']:.3f}, "
        f"|xy_correlation|={metrics['axis_correlation']:.3f}, "
        f"repeat_frequency_cv={metrics['repeat_frequency_cv']:.3f}, "
        f"temporal_window={metrics['temporal_window_size']} frames, "
        f"max_window_centroid_bias={metrics['max_window_centroid_bias']:.3f}px, "
        f"min_window_quadrants={metrics['min_window_quadrants']}/4"
    )
    print(f"score: {analysis.score}/{analysis.total_checks}")
    if analysis.severity_reasons:
        print("temporal error reasons:")
        for reason in analysis.severity_reasons:
            print(f"  - {reason}")
    if analysis.reasons:
        print("distribution checks to review:")
        for reason in analysis.reasons:
            print(f"  - {reason}")
    print(f"plot: {output_path}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json_report(
    path: Path,
    input_path: Path,
    output_path: Path,
    data: JitterData,
    analysis: Analysis,
) -> None:
    report = _json_safe(
        {
            "schema_version": 2,
            "input": str(input_path),
            "plot": str(output_path),
            "ratio": data.ratio,
            "source_size": data.source_size,
            "target_size": data.target_size,
            "source_resolution_from": data.source_resolution_from,
            "jitter_key": data.jitter_key,
            "jitter_selection": [
                {
                    "frame": frame,
                    "source": source,
                    "fallback_reason": data.fallback_reasons.get(frame),
                }
                for frame, source in zip(data.frame_ids, data.jitter_sources)
            ],
            "jitter_source_counts": {
                key: data.jitter_sources.count(key)
                for key in dict.fromkeys((*JITTER_KEYS, *data.jitter_sources))
            },
            "declared_samples": data.declared_samples,
            "frame_ids": data.frame_ids,
            "verdict": analysis.verdict,
            "severity": analysis.severity,
            "severity_reasons": analysis.severity_reasons,
            "score": analysis.score,
            "total_checks": analysis.total_checks,
            "metrics": analysis.metrics,
            "checks": analysis.checks,
            "reasons": analysis.reasons,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_json", type=Path, help="Dataset metadata JSON")
    parser.add_argument(
        "--ratio",
        type=_ratio_name,
        default="auto",
        help="Upscale ratio to inspect (default: first ratio in metadata)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output PNG path (default: <json_stem>_jitter_analysis.png)",
    )
    parser.add_argument(
        "--source-width",
        type=int,
        help="Override source width when low-resolution color EXRs are unavailable",
    )
    parser.add_argument(
        "--source-height",
        type=int,
        help="Override source height when low-resolution color EXRs are unavailable",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Output image resolution (default: {DEFAULT_DPI})",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional machine-readable analysis report path",
    )
    args = parser.parse_args()

    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    input_path = args.dataset_json
    output_path = args.output or input_path.with_name(
        f"{input_path.stem}_jitter_analysis.png"
    )
    try:
        data = _load_jitter(
            input_path, args.ratio, args.source_width, args.source_height
        )
        analysis = _analyze(data)
        _plot(data, analysis, output_path, args.dpi)
        _print_report(input_path, output_path, data, analysis)
        if args.json_out is not None:
            _write_json_report(args.json_out, input_path, output_path, data, analysis)
            print(f"json report: {args.json_out}")
    except (OSError, ValueError, KeyError, OverflowError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
