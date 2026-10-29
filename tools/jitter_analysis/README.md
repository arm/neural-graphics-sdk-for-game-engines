<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->
<!-- SPDX-License-Identifier: MIT -->

# NSS Jitter Analysis

This command analyzes the jitter offsets in one NSS sequence and creates the
project's standard two-panel PNG: the unique sample positions and their
sequence-order table. Keep this plot implementation as the source of truth for
the jitter-validation skill so future reports use the same visualization.

## Requirements

Use Python 3.11 or newer with the packages in `requirements.txt`:

```text
python -m pip install -r tools/jitter_analysis/requirements.txt
```

## Usage

Pass the sequence JSON, scale, and output paths explicitly. Keep generated
artifacts outside the dataset:

```text
python tools/jitter_analysis/jitter_analysis.py DATASET_PATH/0000.json \
  --ratio x2 \
  --output OUTPUT_DIRECTORY/jitter_analysis.png \
  --json-out OUTPUT_DIRECTORY/jitter-analysis.json
```

`--ratio auto` selects the ratio with the lowest index in
`UpscalingRatiosIndices`. Indices must be non-negative integers. If the
selected scale has normalized per-ratio jitter, the tool reads the first
low-resolution color EXR header to determine its source dimensions and converts
the offsets to render pixels. When those EXRs are unavailable, provide both
`--source-width` and `--source-height`. The sequence index may use `Frame` or
`FrameNumber`; each frame must have a non-negative integer index, and if both
fields are present they must agree. Frames are sorted by index before temporal
analysis. Duplicate indices and frame gaps are rejected with exit code `2`;
select a contiguous interval rather than treating separated frames as neighbors.

For each frame, the tool prefers a valid selected-scale
`QuantisedPerRatioJitter` entry, then falls back to `NormalizedPerRatioJitter`
when the preferred entry is missing, too short, malformed, or non-finite. It
reports each source's frame count separately and identifies fallback frames and
reasons. If neither entry is usable, analysis fails with exit code `2`. This is
the same selection policy as the dataset validator and offline NSS diagnostics.
For metadata without per-ratio fields, finite pixel-valued `Jitter` or
`NonQuantisedJitter` entries are supported.

The JSON report includes the legacy heuristic `verdict`, independent severity,
score, checks, metrics, reasons, selected ratio/jitter field, and sorted frame
IDs. `severity` is `PASS` only when all distribution checks pass, `WARNING`
when one or more checks need review without a clear severe temporal defect,
and `ERROR` only when there is no meaningful offset variation, a sliding
temporal window covers at most one quadrant, or a window's centroid bias is at
least 0.5 pixel. Ordinary score failures therefore remain warnings; the score
is still available to identify areas for improvement. `severity_reasons`
explains any temporal error condition. `verdict` retains its prior boolean
meaning for the analyzer's TAA-like suitability heuristic; consumers should
use `severity` to distinguish warnings from errors.

`jitter_source_counts` reports counts
for both per-ratio fields; `jitter_selection` records the chosen source and
fallback reason for each frame. `jitter_key` is `mixed` when sources differ
across frames. Scores and plots use only the selected offsets, not the unused
alternative field. A completed analysis returns exit code `0`
regardless of severity; inspect the severity and score. Exit code `2` means the
input could not be analyzed.

The score is a heuristic for TAA-like temporal accumulation suitability, not
an NSS image-quality score. It uses the fixed checks in the tool and should be
interpreted with the dataset validator findings and runtime evidence. A
warning means the sequence has room for improvement; it is not by itself
evidence of a visible quality problem. An error marks a clear temporal sampling
degeneration that may impair temporal accumulation, not a measured IQ result.

## Interpreting jitter coverage

`all quadrants covered` reports whether the unique offsets with non-zero X and
Y components visit all four sign quadrants over the full sequence. The
`temporal window quadrants` check also evaluates every sliding window of
consecutive frames. Its window length is based on the expected unique sample
count, with a minimum of eight frames and a maximum of the available sequence
length. `min_window_quadrants` reports the fewest quadrants visited by any one
window; compare it with `quadrants` for full-sequence coverage. A failed
window check indicates that at least one interval does not cover all four
quadrants, which may signal local sampling bias. It is not proof of runtime
shake or final image-quality failure.

The input frames must form a contiguous sequence; the analyzer rejects gaps so
that a temporal window never bridges missing frames.
