# Public v1.1.0 Release Notes
Arm® Neural Graphics SDK for Game Engines public v1.1.0 introduces Neural Frame Rate Upscaling (NFRU), expands platform support, and delivers significant improvements in image quality, performance, stability, and developer tooling.

## New Features
- Added support for Neural Frame Rate Upscaling (NFRU), a neural-assisted frame interpolation technology that increases perceived frame rate by generating intermediate frames between rendered frames. The system combines engine‑provided motion vectors, optical‑flow‑derived correspondence fields, and a neural network model to produce temporally consistent and visually smooth results.
- Added Android™ (AArch64) support for NSS and NFRU on supported Arm GPU platforms.
- Added a Vulkan layer to simplify SDK integration and validation.
- Added NFRU sample code and a dataset replayer to accelerate development and testing workflows.
- Restructured NSS sample code and reduced the library size.
- Added fragment-pass support for both NSS and NFRU.

## Improvements
- Upgraded the Neural Super Sampling (NSS) algorithm to improve image quality, reduce Neural Engine bandwidth consumption, and provide quality/balanced/performance presets for different performance and quality targets.
- Added new debugging and visualization capabilities for NSS and NFRU.
- Added additional API configuration controls for NFRU.
- Optimized memory bandwidth usage and overall runtime performance across the SDK.
- Improved SDK stability and fixed multiple reliability issues.

# Public v1.0.1 Release Notes

- Added sample code for Neural Super Sampling (NSS), the following platform combinations are supported:
  - Windows® 11 - x86-64.
  - Android™ - AArch64. (Validated on Android™ 16)
- Added Android™ AArch64 support for Neural Super Sampling (NSS) in the SDK.
- Added the padding feature support for Neural Super Sampling (NSS) in the SDK.
- Bug fixes for Neural Super Sampling(NSS) in the SDK.

# Public v1.0.0 Release Notes

Arm® Neural Graphics SDK for Game Engines public v1.0.0 marks the first public release of Arm’s unified graphics software development kit. It is designed to support multiple rendering use cases across diverse engines and platforms, with a modular and engine-agnostic architecture. This release focuses on Neural Super Sampling (NSS) and lays the foundation for future technologies.

## Key Features

### Neural Super Sampling(NSS)

Neural Super Sampling (NSS) is a temporal frame upsampling technique that improves visual fidelity by reconstructing high-resolution frames from lower-resolution inputs. It uses neural networks to assist with the upscaling process and is specifically optimized for mobile applications to achieve high image quality, computational efficiency, and minimal bandwidth usage.

### SDK Architecture

API Layer: Compatible with FidelityFX API 1.1.3.

Component Layer: Modular implementation of NSS, will include more modules.

Backend Layer: Vulkan-based execution.

### NSS Models

The models can be downloaded from [Hugging face](https://huggingface.co/Arm/neural-super-sampling). Please note that SDK only supports quantized models.

### Supported Platforms
The following platform combinations are supported:

Linux - x86-64
Windows® 11 - x86-64

## Limitation

This is a Vulkan focused delivery so the sdk only supports the Vulkan backend.

To run the SDK on devices that do not support ML extensions for Vulkan, the ML Emulation Layer for Vulkan needs to be enabled. Details can be found in [README.md](/README.md#Vulkan-emulation-layer). Please note that performance based on the Vulkan emulation layer does not represent the performance on actual devices.

# Documentation and resources

SDK readme can be found in [README.md](/README.md).

RenderDoc which supports ML extensions for Vulkan can be downloaded from [Arm developer](https://developer.arm.com/Tools%20and%20Software/RenderDoc%20for%20Arm%20GPUs).

Vulkan Emulation layer libs will be under the release package. Source code can be found in [github](https://github.com/arm/ai-ml-emulation-layer-for-vulkan).
