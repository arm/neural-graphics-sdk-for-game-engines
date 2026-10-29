<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: MIT
-->

# Bin2Exr

`Bin2Exr` is a public SDK tool that converts raw dataset-capture payloads to
EXR files.

## Build dependency

The SDK does not distribute `nlohmann/json.hpp`. A release-package build with
`BUILD_TOOLS=ON` must provide a directory containing that header:

```text
-DFFX_CAPTURE_JSON_INCLUDE_DIR=<directory containing json.hpp>
```

When the source checkout contains `samples/third_party/tinygltf/json.hpp`,
CMake uses it as a convenience default. The release package does not include
that samples dependency, so callers must provide the cache variable above.
Configuration fails immediately when the selected directory does not contain
`json.hpp`.

## Usage

```text
Bin2Exr <capture-directory> [--depth-is-d24]
```

The tool reads the capture metadata at `<capture-directory>/0000.json` and
converts its raw payloads in place. Use `--depth-is-d24` only for datasets
whose depth stream uses 24-bit depth packed into a 32-bit value.
