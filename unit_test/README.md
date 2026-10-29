<!--
/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */
-->

# Unit Test Guide

This directory contains the CLI unit-test framework and suites for backend and
component validation on Linux and Windows.

## Validate Your Development Environment Before Integration

For SDK customers, the main purpose of these tests is to check the development
environment on Linux or Windows before integrating the SDK into an engine or
application. Running the tests independently of your engine helps identify
toolchain, SDK library loading, Vulkan driver, and ML emulation layer setup
issues before introducing engine-specific integration code.

Supported test platforms are Windows x64, Linux x64, and Linux ARM64. On a PC
without native support for the required Vulkan ML extensions, configure the
ML emulation layers and use the [Python helper](#running-through-python-helper).

Before starting integration:

1. [Build](#build) the tests on your development platform.
2. [List Vulkan devices](#run-binsdk_unit_test) and select the intended device.
3. Run the tests and check the summary for failures and skipped tests. A skipped
   test does not validate the corresponding capability.

A successful run provides a baseline for the tested SDK paths on the selected
development device. It does not replace validation of your engine integration
or testing on the target device.

## Writing Focused Tests

The suites exercise public API and observable component behavior. Keep each
test focused on a distinct contract, boundary, failure mode, or regression
risk:

- Search existing suites before adding coverage.
- Combine equivalent inputs when named rows still make failures clear, and
  report independent failing rows instead of stopping at the first one.
- Avoid private control flow, duplicate failure signals, and incidental output.
- Assert the expected effect: callback tests should verify delivery, and
  image-output tests should inspect SDK output rather than accept valid input
  as a substitute.
- When consolidating cases, preserve coverage of distinct public API entry
  points and documented numerical boundaries.

For example, retain a case proving that the public API rejects an invalid
descriptor with its documented error. Do not add separate tests for many
ordinary values when they execute the same path and make the same assertion,
or a test that only checks how many times a private helper was called.

## Layout

```text
unit_test/
  backend_suites/      # backend and vk_wrapper coverage
  component_suites/    # public-API suites (NSS/NFRU, including NSS entry-point contracts)
  framework/           # test runner, Vulkan helpers, shader assets
    shader/
      triangle.vert.glsl
      triangle.frag.glsl
      triangle.vert.spv
      triangle.frag.spv
```

## Build

Use `UNIT_TEST` to enable or disable the unit-test target. Dataset capture is
compiled by default (`FFX_CAPTURE_ENABLED=ON`). When reusing a build directory
created before capture became the default, use `--clean` or pass
`-DFFX_CAPTURE_ENABLED=ON` explicitly.

```bash
# Windows x64
python build.py -t Debug -b vk_windows_x64 --cmake-args="-DUNIT_TEST=ON -DFFX_CAPTURE_ENABLED=ON"

# Linux x64
python build.py -t Debug -b vk_linux_x64 --cmake-args="-DUNIT_TEST=ON -DFFX_CAPTURE_ENABLED=ON"

# Linux ARM64
python build.py --clean -t Debug -b vk_linux_arm --vulkan-lib=/path/to/vulkan/install --cmake-args="-DUNIT_TEST=ON -DFFX_CAPTURE_ENABLED=ON"

```

Windows shared-library builds compile the private API capture helpers into the
test executable as well as the SDK DLL. This keeps capture tests available
without exporting those internal helpers or requiring a static SDK build.
`FFX_CAPTURE_ENABLED=OFF` omits both the capture suite and its private helpers.

## Run `./bin/sdk_unit_test`

By default, all tests from all suites run.

```bash
# all tests
./bin/sdk_unit_test

# help
./bin/sdk_unit_test -h

# verbose mode (prints framework + subprocess logs)
./bin/sdk_unit_test -v --suite backend_vk --test-range 1:1

# list suites
./bin/sdk_unit_test --list-suites

# list Vulkan devices and select one by index/name
./bin/sdk_unit_test --list-device
./bin/sdk_unit_test --select-device 2
./bin/sdk_unit_test --select-device lavapipe

# list all tests
./bin/sdk_unit_test --list-tests

# list tests in one suite (by id or name)
./bin/sdk_unit_test --list-tests --suite 1
./bin/sdk_unit_test --list-tests --suite backend_vk

# capture suite
./bin/sdk_unit_test --suite capture

# run one suite / range of suites
./bin/sdk_unit_test --suite backend_vk
./bin/sdk_unit_test --suite-range 1:2

# run a test range within selected suite(s)
./bin/sdk_unit_test --suite backend_vk --test-range 1:3

# run a specific test
./bin/sdk_unit_test --test nss_api.query_jitter_without_context

# benchmark mode
./bin/sdk_unit_test --suite backend_vk --benchmark=5

# run expected-failure tests as normal tests
./bin/sdk_unit_test --expected_failure
```

Notes:
- Suite IDs are one-based and stable for the current binary registration order.
- Test IDs are one-based within each suite.
- Expected-failure tests are reported as `EXPECTED_FAIL` and are not executed.
- `--expected_failure` overrides that behavior and executes expected-failure tests normally.
- Capability-dependent tests can use `UNIT_TEST_SKIP(reason)` or `MarkTestSkipped(reason)` when a required device feature is unavailable. The runner reports `SKIP` with the reason and includes a `skipped` summary count; a skipped path is not reported as a pass.
- Subprocess isolation is enabled by default; use `--no-isolation` for in-process execution.
- Failed tests are listed with reasons in the summary; if any fail, `sdk_unit_test` exits non-zero.
- `--test` selects a registered test, not an individual table row. Rerunning a
  consolidated test executes all its named rows.
- Image-output tests require completed dispatches and output readback on the
  selected device; a valid input frame or setup-only smoke run is not a pass.
- Use `--select-device` to force a Vulkan implementation for all test execution subprocesses.
- Default run output is sectioned into:
  - metadata banner (selected Vulkan backend, full parameters, SDK commit),
  - suite headers and per-test separators,
  - summary banner.
- `--verbose` / `-v` enables extra runner logs and forwards subprocess logs.

## Running Through Python Helper

`unit_test/run_test_on_emulation_layer.py` forwards arguments to `sdk_unit_test` and configures the Vulkan ML emulation layers. Follow the [User Guide - ML Emulation Layer for Vulkan](../docs/user_guide.md#ml-emulation-layer-for-vulkan) to download and configure a compatible release before running the helper.

The helper defaults `--vk-emul-layer-path` to the downloaded platform directory. The option can also point directly to its `lib` directory on Linux or `bin` directory on Windows. The helper locates the layer manifests and libraries, configures the Vulkan loader search paths, and enables both emulation layers for the test subprocesses.

The test executable uses the default Vulkan device unless `SDK_UNIT_TEST_VK_DEVICE` is set. The value can be a device index or a case-insensitive device-name substring.

```bash
# Linux: list Vulkan devices
python3 unit_test/run_test_on_emulation_layer.py --list-device

# Linux: run a filtered subset
python3 unit_test/run_test_on_emulation_layer.py --vk-emul-layer-path /path/to/VulkanML --suite backend_vk --test-range 1:1

# Linux: select a device by name
SDK_UNIT_TEST_VK_DEVICE=lavapipe python3 unit_test/run_test_on_emulation_layer.py

# Windows PowerShell: list Vulkan devices
py -3 unit_test\run_test_on_emulation_layer.py --vk-emul-layer-path C:\path\to\VulkanML --list-device

# Windows PowerShell: select an NVIDIA device
$env:SDK_UNIT_TEST_VK_DEVICE = "NVIDIA"
py -3 unit_test\run_test_on_emulation_layer.py
```

## Test Shader Assets

The triangle-rendering tests use precompiled SPIR-V assets:

- Source assets: `unit_test/framework/shader/triangle.vert.spv` and `triangle.frag.spv`
- Build staging: CMake copies both files to `shader/` next to `sdk_unit_test`
- Runtime lookup: the test executable reads `<executable-directory>/shader/*.spv`

This keeps the test bundle relocatable: deployment must include `sdk_unit_test` and its sibling `shader/` directory. The GLSL files are source artifacts only; they are not compiled automatically during the SDK build.

When changing a triangle shader:
1. Edit its GLSL source file.
2. Regenerate the matching SPIR-V asset:

```bash
glslangValidator -V unit_test/framework/shader/triangle.vert.glsl -o unit_test/framework/shader/triangle.vert.spv
glslangValidator -V unit_test/framework/shader/triangle.frag.glsl -o unit_test/framework/shader/triangle.frag.spv
```

3. Commit the GLSL and matching `.spv` file together.
