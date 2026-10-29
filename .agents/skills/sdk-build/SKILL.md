---
name: sdk-build
description: Build the SDK or requested samples/tests and diagnose build failures. Follow the linked SDK build guide for supported targets, prerequisites, and options.
---

<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->
<!-- SPDX-License-Identifier: MIT -->

# SDK build

Use `build.py` and the existing CMake targets. Locate the SDK root three
directories above this skill and run from there. A source package need not
contain Git metadata. Follow the [SDK User Guide - Build the SDK](../../../docs/user_guide.md#build-the-sdk)
and `python build.py --help` for supported targets, prerequisites, options, and
build commands. Read the [compatibility matrix](../../../docs/compatibility_matrix.md)
for validated dependencies, and the [sample guide](../../../samples/README.md)
when samples are requested.

## Select the build

Record the host OS/architecture, target backend, configuration, SDK-only versus
samples/unit tests, job limit, and any requested CMake options. Default to the
native supported backend, `Release`, and incremental SDK-only compilation for
an otherwise unspecified build. Preserve any explicitly selected target.

Inspect existing dependencies and CMake cache before changing configuration.
Use existing tools first; install only a confirmed missing prerequisite into an
appropriate environment. Preserve submodule changes and use documented
dependency sources. Fetch sample dependencies only when needed and authorized;
follow the sample guide for the supported procedure.

## Configure and build

Run only the selected build command from the SDK User Guide, adapting paths to
the host shell. Preserve intended options and use incremental builds by
default. Run a clean build only after a deliberate rebuild decision,
preserving user changes and unrelated build trees.

## Interpret and hand off

Save the exact command, working directory, log path, exit code, backend, and
configuration. Distinguish configuration failure, compiler/shader failure,
linker failure, and missing runtime dependencies. Report the first actionable
error with its file/target and required prerequisite.

A successful configure is not a completed build. Wait for completion and
confirm the requested artifacts exist: SDK libraries in `bin`, optional
`bin/sdk_unit_test[.exe]`, and sample executable/APK at the paths in the sample
guide. Report warnings and any target not built. Inspect artifact timestamps
and configuration to avoid handing off a stale binary.

For execution use [sdk-samples](../sdk-samples/SKILL.md), or
[sdk-validation](../sdk-validation/SKILL.md) for selected tests. Report build,
runtime, image-quality, and performance results as separate evidence.

Rerun the applicable build command for the current revision and platform.
