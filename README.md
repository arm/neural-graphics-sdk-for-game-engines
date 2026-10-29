# Neural Graphics SDK for Game Engines

The Arm Neural Graphics SDK is a unified neural-rendering software development kit for mobile GPU game engines.

With the Arm Neural Graphics SDK, you can combine neural network inference with Vulkan ML extensions to deliver high image quality and high frame rates. The SDK is derived from [AMD FidelityFX SDK 1.1.3](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK) and can be integrated through a Vulkan-only, engine-agnostic API.

## Table of contents

1. [Component overview](#component-overview)
2. [Documentation](#documentation)
3. [Neural Graphics Development Kit](#neural-graphics-development-kit)
4. [Limitations](#limitations)
5. [License](#license)
6. [Trademarks and copyrights](#trademarks-and-copyrights)

---

## Component overview

Currently, the SDK includes the following components:

| Component | Description | Platforms | Detailed documentation |
|-----------|-------------|-----------|------------------------|
| **NSS** - Neural Super Sampling | Temporal upscaling from render resolution to display resolution using a neural network. | Windows x64, Linux x64/AArch64, Android AArch64 | [NSS guide](docs/NSS/NSS.md) |
| **NFRU** - Neural Frame Rate Upscaling | Frame interpolation that generates one extra frame between rendered frames. | Windows x64, Linux x64/AArch64, Android AArch64 | [NFRU guide](docs/NFRU/NFRU.md) |

> **API layer:** The SDK is derived from AMD FidelityFX SDK 1.1.3 and adds Arm-specific NSS and NFRU descriptors. This is not a drop-in implementation of every upstream FidelityFX effect or backend. Components share one Vulkan backend and can be integrated independently.

---

## Documentation

The following documentation is available:

- [User Guide](docs/user_guide.md) - build and integration instructions, project layout, shared API, resource states, synchronization, lifecycle, diagnostics, and memory queries.
- [Migration Note](docs/migration_note.md) - check this note when upgrading the SDK for source, ABI, and behavior changes.
- [Compatibility matrix](docs/compatibility_matrix.md) - supported public dependency and tooling versions.
- [NSS technical documentation](docs/NSS/NSS.md) - NSS context flags, resource contracts, jitter, motion vectors, dispatch, and debug views.
- [NFRU technical documentation](docs/NFRU/NFRU.md) - NFRU context flags, prepare/dispatch flow, resource coordinate systems, and presentation pacing.
- [Dataset capture guide](docs/Capture/Capture.md) - capture runtime NSS and NFRU data for offline replay and validation.
- [NSS and NFRU integration troubleshooting](docs/integration_troubleshooting.md) - symptom-based input validation, debug-view interpretation, and visual examples.
- [Sample documentation](samples/README.md) - sample dependency setup, build, run, and configuration instructions.
- [Unit test guide](unit_test/README.md) - build and run unit tests for backend + component validation on linux/windows.
- [Agent workflows](.agents/README.md) **(Experimental)** - SDK build, sample execution, focused tests, and dataset validation.

---

## Neural Graphics Development Kit

The Neural Graphics SDK is part of the broader Neural Graphics Development Kit. The stack below links to each product or its documentation.

| Neural Graphics Development Kit |
|:------------------------------:|
| [Game Engine Plugins](https://github.com/arm/neural-graphics-for-unreal) |
| **[Neural Graphics SDK for Game Engines](https://github.com/arm/neural-graphics-sdk-for-game-engines) — (this repository)** |
| [Vulkan Emulation Layer on PC](https://github.com/arm/ai-ml-emulation-layer-for-vulkan) |
| Neural Graphics Models — [NSS](https://huggingface.co/Arm/neural-super-sampling) / [NFRU](https://huggingface.co/Arm/neural-frame-rate-upscaling) on Hugging Face |
| [Neural Graphics Model Gym](https://github.com/arm/neural-graphics-model-gym) |
| [Dataset capture](https://github.com/arm/neural-graphics-data-capture-for-unreal) / [Debug and Optimization tools](https://developer.arm.com/tools-and-software/arm-performance-studio) |
| [Documentation and Resources](#additional-documentation-and-resources) |


---

### Additional Documentation and Resources

- [Arm Neural technology](https://www.arm.com/technologies/neural-technology) - overview of Arm Neural technology.
- [Arm Neural technology developer resources](https://developer.arm.com/mobile-graphics-and-gaming/neural-technology) - Neural technology information and resources on the Arm Developer website.

---

## Limitations

The SDK currently has the following limitations:

- **Vulkan only.** DirectX and Metal backends are not provided.
- **Quantized data graphs only.** Both NSS and NFRU require quantized models.
- **Application-owned synchronization.** The SDK does not synchronize work outside its recorded commands.

---

## License

The Arm Neural Graphics SDK software in this repository is licensed under the [MIT License](LICENSES/MIT.txt).

The [Arm Neural Graphics SDK Developer Guide](docs/user_guide.md) is not licensed under the MIT License.

The [Arm Neural Graphics SDK Developer Guide](docs/user_guide.md) is licensed separately under the Creative Commons Attribution-NoDerivatives 4.0 International License (CC BY-ND 4.0): https://creativecommons.org/licenses/by-nd/4.0/. You can refer to [CC-BY-ND 4.0](LICENSES/CC-BY-ND-4.0.txt).

The VGF model files under `sdk/src/backends/vk/data_graphs/` are licensed under the
[Arm AI Model Community License v1.0 (PRE-1154)](https://huggingface.co/Arm/neural-frame-rate-upscaling/blob/main/Arm_AI_Model_Community_License_v1_0_PRE-1154.pdf).

Copyright © 2025-2026 Arm Limited.

Except for the rights expressly granted under that license, Arm reserves all rights in the Developer Guide.

No patent or trademark rights are granted by that license.

The MIT License applying to the Arm Neural Graphics SDK software does not apply to the Developer Guide.

---

## Trademarks and copyrights

AMD is a trademark of Advanced Micro Devices, Inc.

AMD FidelityFX™ is a trademark of Advanced Micro Devices, Inc.

Arm® is a registered trademark of Arm Limited (or its subsidiaries) in the US and/or elsewhere.

Vulkan is a registered trademark and the Vulkan SC logo is a trademark of the Khronos Group Inc.

Visual Studio and Windows are registered trademarks or trademarks of Microsoft Corporation in the US and/or elsewhere.
