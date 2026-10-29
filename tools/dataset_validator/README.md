# Dataset Validator

The dataset validator performs deterministic, read-only checks on NSS and
NFRU capture datasets. It validates structure, metadata, frame timelines, EXR
content, jitter, motion vectors, exposure, and output resolution.

## Agent workflow

The normal [dataset-validation skill](../../.agents/skills/dataset-validation/SKILL.md)
runs this command, compares NSS inputs with the
[published NSS Dataset Specification](https://github.com/arm/neural-graphics-model-gym/blob/main/docs/nss/nss_dataset_specification.md),
and writes a Markdown report alongside the JSON findings. Give the agent a
dataset path and the desired profile/use case/scale, or use the command
directly below.
By default, skill reports are stored under the repository-root
`agents_output/` directory, which is ignored by Git.

This is a static check only; it does not run DatasetReplay or make pixel-domain
quality conclusions. See [agent workflows](../../.agents/README.md) for
discovery and publication conventions.

## Requirements

- Python 3.11 or newer
- The packages listed in `requirements.txt`

Install the dependencies from the SDK repository root:

```text
python -m pip install -r tools/dataset_validator/requirements.txt
```

## Usage

Run the command from the SDK repository root:

```text
python tools/dataset_validator/validate_dataset.py DATASET_PATH \
  --profile nss-replay-v1 \
  --use-case replay \
  --scale x2
```

Write the complete machine-readable report with `--json-out`:

```text
python tools/dataset_validator/validate_dataset.py DATASET_PATH \
  --profile nfru-replay-v1 \
  --use-case debug \
  --scale x2 \
  --json-out validation-report.json
```

The report path must be outside the dataset root so validation remains
read-only with respect to the captured data.

Use `--warning-policy fail` when warnings must fail a CI job. The default
policy, `allow`, reports warnings but returns success when there are no errors.

Exit codes:

| Code | Meaning |
|---|---|
| `0` | Validation completed and the selected policy passed. |
| `1` | Validation completed and found errors or policy-blocking warnings. |
| `2` | The command, profile, dependency setup, output path, or invocation is invalid. |

## Profiles

| Profile | Supported use cases | Dataset contract |
|---|---|---|
| `nss-replay-v1` | `debug`, `replay` | Lock-step NSS colour, depth, and motion inputs; ground truth is optional for replay and recommended for debug. |
| `nfru-replay-legacy` | `debug`, `replay` | Legacy `motion_t-2` layout; source lanes may contain the full timeline or only replay-consumed even frames. |
| `nfru-replay-v1` | `debug`, `replay` | Source colour/depth may be full or even-only, `motion_m2` uses even frames, and optional predictions use odd midpoints. |

`debug` requires replay inputs and warns when recommended diagnostic lanes are
missing. NFRU replay uses the `ground_truth` directory as its required source
colour lane; it is not optional in that layout. Profile files are versioned JSON
documents under `profiles`; select one explicitly so validation does not guess a
dataset convention.

## Checks and report

The validator decodes numeric EXR frames in the selected profile lanes and
scale, for sequences with root metadata. It reports:

- missing, empty, non-numeric, duplicate, or misaligned EXR frame files;
- malformed metadata, frame counts, required per-frame values, camera planes,
  target resolution, and render-to-target scale;
- frame indices from non-negative integer `Frame` or `FrameNumber` fields; if
  both are present, they must agree;
- EXR readability, dimensions, channel layouts, floating-point pixel types,
  non-finite values, and configured value ranges;
- jitter presence, scale selection, pixel-space range, centering, coverage,
  valid sample count, duplicates, spacing, and X/Y correlation;
- exposure presence, finite values, and profile-specific bounds;
- constant non-zero motion fields that require a follow-up warp diagnostic.

Non-EXR sidecars, such as preview videos and notes, are outside the frame scan
and do not produce findings. A nonnumeric EXR filename produces a naming error
and sets `complete_scan` to false: it cannot be associated with the captured
timeline. This does not assert that its image content is damaged.

Named experimental directories beneath profile lane roots are reported as
`lane.directory.excluded` information findings, not as numeric orphan
sequences. They are not decoded or included in `files_scanned`. Other scales
and lanes not declared in the chosen profile are outside the scan scope.
`complete_scan` describes the selected scope, not every file under the dataset
root; read it together with findings, including exclusions.

### Hard errors versus diagnostic thresholds

Missing required signals, invalid metadata, frame misalignment, unreadable
images, non-finite values, and violations of the selected representation are
data errors. Jitter centering, coverage, phase length, spacing, duplicates,
correlation, constant non-zero motion, and high exposure are diagnostic
warnings, not universal correctness criteria. For example, exposure above
`1024` is a prompt to review its representation, not a hard rejection.

Repeating jitter timelines are analyzed over their inferred phase so a valid
periodic sequence is not reported as duplicate sampling. Its phase length is
inferred from observed values, not declared by `Samples.Count`, which is the
metadata frame count. Quality thresholds are heuristic and need contextual
interpretation. Only an explicit `--warning-policy fail` makes warnings block
the selected policy; it does not reclassify them as data errors.

The JSON report contains:

```text
schema_version
validator_version
dataset
profile
use_case
scale
warning_policy
complete_scan
status
summary
sequences
findings
```

Each finding includes severity, check identifier, location, observed and
expected values, and remediation. Findings are deterministically ordered.
NaN and infinity observations are emitted as JSON strings so reports remain
strict JSON.

## Python interface

The command and tests use the same public interface:

```python
from dataset_validator import validate_dataset

report = validate_dataset(
    dataset_path,
    "nss-replay-v1",
    use_case="replay",
    scale="x2",
    warning_policy="allow",
)

print(report.status)
print(report.to_json())
```

When importing from an SDK checkout, add `tools/dataset_validator` to
`PYTHONPATH`, or run the command directly as shown above.

## Scope and safety

Validation is local and read-only. A structural PASS does not establish image
quality, motion-vector polarity, or temporal reconstruction quality. For NSS
jitter-sequence and unjittered-color analysis, use the
[jitter-validation skill](../../.agents/skills/jitter-validation/SKILL.md).
For exposure-aware histograms and motion warp/error images, use the
[nss-integration-debug skill](../../.agents/skills/nss-integration-debug/SKILL.md);
it aggregates the jitter-validation report rather than repeating jitter analysis.
