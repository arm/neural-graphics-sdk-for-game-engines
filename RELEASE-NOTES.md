## Public v1.0.0 Release Notes

Arm® Neural Graphics SDK for Game Engines public v1.0.0 marks the first private release of Arm’s unified graphics software development kit. It is designed to support multiple rendering use cases across diverse engines and platforms, with a modular and engine-agnostic architecture. This release focuses on Neural Super Sampling (NSS) and lays the foundation for future technologies.

## Key Features

### Neural Super Sampling(NSS)
Neural Super Sampling (NSS) is a temporal frame upsampling technique that improves visual fidelity by reconstructing high-resolution frames from lower-resolution inputs. It uses neural networks to assist with the upscaling process and is specifically optimized for mobile applications to achieve high image quality, computational efficiency, and minimal bandwidth usage.

### SDK Architecture

API Layer: Compatible with FidelityFX API 1.1.3.

Component Layer: Modular implementation of NSS, will include more modules.

Backend Layer: Vulkan-based execution.

### NSS Models

The models can be downloaded from [Hugging face](https://huggingface.co/Arm/neural-super-sampling). Please note that SDK only supports quantized models.

### Platform Support

Windows 11 x64.

Linux x64.

## Limitation

This is a Vulkan focused delivery so the sdk only supports the Vulkan backend.

To run the SDK on devices that do not support ML extensions for Vulkan, the ML Emulation Layer for Vulkan needs to be enabled. Details can be found in [README.md](/README.md#Vulkan-emulation-layer). Please note that performance based on the Vulkan eimulation layer does not represent the performance on actual devices.

## Documentation and resources

SDK readme can be found in [README.md](/README.md).

NSS readme can be found in [NSS.md](/docs/NSS/NSS.md).

RenderDoc which support ML extensions for Vulkan can be downloaded from [Arm developer](https://developer.arm.com/Tools%20and%20Software/RenderDoc%20for%20Arm%20GPUs). 

Vulkan Emulation layer libs will be under the release package. Source code can be found in [github](https://github.com/arm/ai-ml-emulation-layer-for-vulkan).