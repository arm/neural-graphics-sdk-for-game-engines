<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->
<!-- SPDX-License-Identifier: MIT -->

# NSS Integration Debug

This tool generates offline diagnostics from a captured NSS dataset without
adding SDK runtime APIs, shaders, resources, or GPU work. It uses the
`nss-replay-v1` dataset contract and runs the existing dataset validator before
creating any artifacts.

## Agent workflow

The [jitter-validation skill](../../.agents/skills/jitter-validation/SKILL.md)
owns sequence-distribution, pixel-domain jitter, and unjittered-color replay
analysis. The [nss-integration-debug
skill](../../.agents/skills/nss-integration-debug/SKILL.md) invokes it and uses
this CLI for exposure/histogram and motion-vector diagnostics only, then
includes the jitter report in its Markdown aggregate. The CLI still supports
`--diagnostic jitter` for the jitter-validation workflow and direct users.
High-level skill reports default to a unique run directory beneath the
repository-root `agents_output/` directory, which is ignored by Git.

## Requirements

- Python 3.11 or newer
- Packages listed in `requirements.txt`

From the SDK root:

```text
python -m pip install -r tools/nss_integration_debug/requirements.txt
```

## Usage

Generate all diagnostics for up to eight representative frames:

```text
python tools/nss_integration_debug/diagnose_nss.py DATASET_PATH \
  --profile nss-replay-v1 \
  --sequence 0000 \
  --scale x2 \
  --frames auto \
  --output-dir OUTPUT_DIRECTORY
```

`OUTPUT_DIRECTORY` must be a new path outside the source dataset. Use
`--frames all`, a single frame, or comma-separated frames and inclusive ranges,
for example `--frames 0,4-8`. A run may select at most 256 frames; split larger
captures into separate bounded runs and output directories. Repeat
`--diagnostic` to select only the required work:

```text
python tools/nss_integration_debug/diagnose_nss.py DATASET_PATH \
  --sequence 0000 \
  --frames 12-16 \
  --diagnostic jitter \
  --diagnostic motion \
  --output-dir OUTPUT_DIRECTORY
```

An optional `config.json` at the dataset root may contain a positive
`Exposure` value. When present, it overrides every frame's `Exposure` metadata,
matching the NSS dataset loader. The report records whether this override was
used and identifies the source for every selected frame.

Frame metadata may use `Frame` or the `FrameNumber` compatibility alias.
The validator preflight requires non-negative integer indices and equal values
when both fields are present. Diagnostics index metadata by these values,
independently of JSON array order.

## Outputs

The command writes:

- `interpretation.md`: the recommended human-readable starting point, with an
  assessment, supporting evidence, interpretation limits, and suggested
  follow-up for each requested diagnostic.
- `report.json`: the complete machine-readable report. Its top-level
  `interpretation` object contains the same assessment in structured form.
- Selected PNG artifacts:

| Directory | Diagnostic |
|---|---|
| `jitter/` | Jittered and supplied-jitter-removed low-resolution color, with temporal MSE values when the previous frame exists. |
| `histogram/` | Per-frame unjittered, exposure-tonemapped input and 64-bin BT.709 luminance histogram, plus `aggregate.png`. |
| `motion/` | Current input, motion-warped previous input, squared-error heatmap, and a geometric/depth-consistency mask. |

Contact-sheet panels appear from left to right in the order listed above; the
same order is recorded explicitly in each frame's JSON result.

The JSON report records the validator preflight, exposure source and values,
selected frames, jitter values, estimated residual translation, per-frame and
aggregate histograms, motion MSE summaries, the effective per-frame
motion-vector scale, common-mask polarity comparison, valid-pixel fraction,
warnings, panel order, artifact paths, and structured interpretation.
`summary.warnings` is the total of preflight and diagnostic warnings;
`summary.warning_counts` keeps those two sources separate.

## How to read the result

Read `interpretation.md` first, then use `report.json` and the referenced PNGs
to verify its evidence.

| Result | What it means | What it does not mean |
|---|---|---|
| Preflight failed | Required dataset inputs are invalid, so visual diagnostics were not generated. | An absent diagnostic image is an NSS quality result. |
| Jitter compensation improves temporal metrics | Applying the supplied offsets made sampled frames more temporally similar, which supports the metadata sign and scale. | Intentional capture jitter is defective, or all residual motion comes from jitter. |
| Jitter compensation consistently worsens metrics | Jitter sign, scale, ratio selection, or frame ordering needs investigation on a mostly static range. | A single dynamic frame proves the metadata is wrong. |
| Histogram is dark- or bright-heavy | The sampled, tonemapped scene content is concentrated in that range. Inspect visible detail and exposure changes. | The input is automatically underexposed or overexposed, or final NSS IQ is proven. |
| Documented motion polarity is supported | It produced the lowest photometric MSE in every compared frame. | Motion vectors or final output are otherwise correct. |
| Alternative polarity is consistently better | The capture/integration motion convention is a strong investigation lead. | The alternative should be applied automatically without checking the documented convention and images. |
| Low valid-mask coverage | The masked MSE represents a smaller part of the frame and needs careful visual review. | The retained or rejected pixels exactly match runtime disocclusion. |
| Exposure source is `config.json` | One positive dataset-level override was applied to every selected frame. | The per-frame exposure metadata was used for those frames. |

## Interpretation limits

- Histogram images use the runtime Karis tonemapping half-precision and
  `[0, 65504]` clamping domain, but do not decode the runtime's quantized model
  tensor. Treat them as input/exposure evidence, not proof that exposure or
  final IQ is correct.
- Jitter compensation samples the raw scene-linear RGB EXR values with
  bilinear offsets in render pixels, then applies the exposure-aware
  half-precision Karis tonemap. The unshifted panel uses the same tonemap
  without the jitter offset; both outputs are normalized to the diagnostic
  display domain `[0, 1]`.
- EXR data windows larger than 64M pixels are rejected from diagnostics before
  channel pixel arrays are materialized.
- Motion reprojection uses each frame's finite `MotionVectorScale` metadata when
  present. When the field is absent, it uses the DatasetReplay fallback of
  negative render width and height. Metadata scales must be representable as
  finite float32 values. Alternative candidates flip signs relative to that
  effective scale and are compared over one common geometric-validity mask;
  they are reported as evidence, not applied automatically.
- Depth input follows the selected profile's required channel and accepted
  single-channel aliases, including `Y` when the NSS profile permits it.
- The motion mask excludes non-finite and out-of-bounds samples, then applies a
  robust frame-local depth-consistency threshold to suppress likely visibility
  changes. It is not an exact reproduction of the runtime disocclusion mask;
  inspect its retained fraction and the unmasked MSE.
- Temporal MSE can be affected by scene motion, animation, lighting, exposure,
  transparency, and capture noise. Inspect the images before changing an
  integration.
- Phase-correlation residual translation is meaningful mainly for static or
  nearly static content.

The source dataset is opened read-only. Exit code `0` means diagnostics were
generated, `1` means dataset validation found errors and only a failure
`report.json` and `interpretation.md` were written, and `2` means invocation,
dependency, input, or output setup failed.
