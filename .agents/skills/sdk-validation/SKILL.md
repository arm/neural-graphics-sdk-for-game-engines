---
name: sdk-validation
description: Run focused SDK tests, inspect failures, or check a built bundle. Follow the linked unit-test guide for runner options, build setup, and execution requirements.
---

<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->
<!-- SPDX-License-Identifier: MIT -->

# Focused SDK validation

Use the [unit-test guide](../../../unit_test/README.md), existing test binary,
and public SDK contracts. The guide is authoritative for runner options, test
build requirements, and runtime assets. Use the
[compatibility matrix](../../../docs/compatibility_matrix.md) for the validated
emulation-layer version. Locate the SDK root three directories above this skill;
run commands there without requiring Git metadata. Collect the changed behavior
or symptom, backend/configuration, test selector, device, and report destination.

## Build and discover

Use [sdk-build](../sdk-build/SKILL.md) with
unit tests enabled when the binary is missing or stale. Follow the unit-test
guide for build options, runner assets, and runtime requirements. GPU/ML tests
also need a compatible device or emulation environment.

Discover runner options with the guide or the binary's `--help`, then use the
native test binary from the SDK root. Native invocations preserve the caller's
Vulkan layer configuration.

Choose the smallest relevant selector after discovery. Prefer suite/test names
to numeric IDs, which reflect the current registration order. Examples:

| Behavior | Focus |
|---|---|
| API parameter/provider contracts | `--test nss_api.entry_point_contracts` |
| Logging callback/filtering | Relevant `logging` tests |
| Vulkan resources/backend lifetime | Relevant `backend_vk` tests |
| NSS or NFRU component behavior | Named tests from `nss_api` or `nfru_api` |
| Captured input data | [dataset-validation](../dataset-validation/SKILL.md) |
| Sample startup or asset deployment | [sdk-samples](../sdk-samples/SKILL.md) |

Read the selected test's requirements: even a suite with mostly CPU tests may
contain a GPU-dependent case. Report API-only coverage separately from rendering
coverage, and report an unavailable GPU case as blocked.

## Run and interpret

For ML emulation, use the dedicated helper; it configures the emulation layers
and differs from the native runner. Follow the [User Guide](../../../docs/user_guide.md#ml-emulation-layer-for-vulkan)
and [unit-test guide](../../../unit_test/README.md#running-through-python-helper)
for compatible package setup and helper options.

Preserve an existing managed copy when replacement is not intended. Discover
devices before selection and pass `--select-device` explicitly when it matters.

Keep subprocess isolation enabled by default. Use `--verbose` for an actionable
failure trace. `EXPECTED_FAIL` cases are not executed unless
`--expected_failure` is requested; report them separately, not as passes.

Record the exact command, SDK binary/configuration metadata, device/layers,
exit code, test names, pass/fail/expected-fail counts, and log location. Require
the selected tests to have actually executed; an empty selection, missing
summary, crash, timeout, or unavailable binary is not a successful validation.
For failures, retain the first relevant assertion and subprocess diagnostic.
Escalate to related suites only when the affected behavior justifies it.

For an explicitly requested timing comparison, use the benchmark option
documented by the runner. Fix configuration, device, layers, selector, and
iteration count across comparisons. Treat the output as repeated test duration
and compare only equivalent environments.

## Bundle checks and boundaries

For a supplied SDK bundle, check that its referenced normal skill files,
runtime libraries, selected sample assets, and optional unit-test binary plus
sibling `shader` directory are present. Run only the relevant supported command
from that bundle.

Report package contents, smoke execution, and whitelist approval as separate
checks, and list only the checks actually run. Keep reports outside the workflow
tree.

Rerun the focused selector against the current binary and effective device/layer
configuration.
