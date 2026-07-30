# Neural Graphics SDK for Game Engines

The Neural Graphics Software Development Kit (SDK) is Arm's unified graphics SDK for multiple rendering use cases across diverse game engines and platforms. Derived from [AMD FidelityFX SDK 1.1.3](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK), it provides a modular, engine-agnostic framework that makes high-quality Neural Super Sampling (NSS) and Neural Frame Rate Upscaling (NFRU) straightforward to integrate. For complete build, integration, API, and sample instructions, see the [User Guide](docs/user_guide.md).

---

## Features

| Component | Description | Platforms |
|-----------|-------------|-----------|
| **NSS** — Neural Super Sampling | Temporal upscaling from render resolution to display resolution using a neural network. Supports Quality / Balanced / Performance presets and flexible upscale ratios. | Windows 11 x64, Linux x64, Android AArch64 |
| **NFRU** — Neural Frame Rate Upscaling | Frame interpolation that generates one extra frame between every two rendered frames, increasing perceived frame rate with predictable latency. | Windows 11 x64, Linux x64, Android AArch64 |

---

## Requirements

**Supported platforms:** Windows 11 x64, Linux x64, Android AArch64.

| Tool category | Notes |
|---------------|-------|
| Git | Required to clone the repository and initialize its submodules |
| CMake | Build system (see [user guide — setup](docs/user_guide.md#set-up-your-environment) for exact version range) |
| Python 3 | Used by `build.py` |
| Vulkan SDK | Vulkan-only backend; see [user guide — setup](docs/user_guide.md#set-up-your-environment) for recommended version |
| Android NDK | Required for Android AArch64 target; see [user guide — platform requirements](docs/user_guide.md#platform-support) for details |

### Vulkan Emulation Layer

The SDK targets the Vulkan backend exclusively. For devices without native Vulkan ML extension support (`VK_ARM_tensors`, `VK_ARM_data_graph`), use the [Arm Vulkan ML Emulation Layer](https://github.com/arm/ai-ml-emulation-layer-for-vulkan/releases).

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/arm/neural-graphics-sdk-for-game-engines.git
cd neural-graphics-sdk-for-game-engines
git submodule update --init --recursive
```

### 2. Build

```bash
# Show all available options and targets
python build.py -h

# Example: Release build for Linux x64
python build.py -t Release -b vk_linux_x64
```

Additional build targets (Windows x64, Android AArch64) and Docker-based builds are described in the [user guide](docs/user_guide.md#build-the-sdk).

### 3. Integrate

Add the SDK to your project as a CMake subproject (`add_subdirectory`) or link the prebuilt library from the `bin/` folder. Then create and dispatch an NSS or NFRU context using the SDK API. See the [user guide](docs/user_guide.md#integration-guidelines) for the complete integration walkthrough.

### 4. Run a Sample

```bash
# Build with samples included
python build.py -t Release -b vk_linux_x64 --build-samples
```

For NSS and NFRU sample configuration, dataset setup, and run instructions, see the [user guide — NSS sample](docs/user_guide.md#neural-super-sampling-sample) and [user guide — NFRU sample](docs/user_guide.md#neural-frame-rate-upscaling-sample) sections.

---

## Known Issues

- In very rare cases, certain upscale ratios can trigger float32 precision issues in the dynamic offset LUT generation path in NSS, which may manifest as visible black line artifacts in the upscaled output. As a workaround, users are advised to adjust the upscale ratio to avoid affected configurations until a permanent fix is available.

---

## Documentation

- [User Guide](docs/user_guide.md) — build, integration, API reference, and samples
- [Release Notes](RELEASE-NOTES.md)

---

## License

The Arm Neural Graphics SDK software in this repository is licensed under the [MIT License](LICENSES/MIT.txt)

The [Arm Neural Graphics SDK Developer Guide](docs/user_guide.md) is not licensed under the MIT License.

The [Arm Neural Graphics SDK Developer Guide](docs/user_guide.md) is licensed separately under the Creative Commons Attribution-NoDerivatives 4.0 International License (CC BY-ND 4.0): https://creativecommons.org/licenses/by-nd/4.0/  – see [CC-BY-4.0](LICENSES/CC-BY-4.0.txt)

Copyright © 2025–2026 Arm Limited.

Except for the rights expressly granted under that license, Arm reserves all rights in the Developer Guide.

No patent or trademark rights are granted by that license.

The MIT License applying to the Arm Neural Graphics SDK software does not apply to the Developer Guide.

---

## Trademarks and Copyrights

AMD is a trademark of Advanced Micro Devices, Inc.

AMD FidelityFX™ is a trademark of Advanced Micro Devices, Inc.

Arm® is a registered trademark of Arm Limited (or its subsidiaries) in the US and/or elsewhere.

Vulkan is a registered trademark and the Vulkan SC logo is a trademark of the Khronos Group Inc.

Visual Studio, Windows are registered trademarks or trademarks of Microsoft Corporation in the US and other jurisdictions.
