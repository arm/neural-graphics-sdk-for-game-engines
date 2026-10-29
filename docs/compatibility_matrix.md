# Compatibility Matrix


## Usecase model
| Usecase | Version | Source |
|-----------|-------------------|--------|
| NSS | 1.0.1 | https://huggingface.co/Arm/neural-super-sampling |
| NFRU | 2.0.0  | https://huggingface.co/Arm/neural-frame-rate-upscaling |

## Build Tool Versions

The following versions and recommendations cover build dependencies shared by
the SDK and samples. Sample-specific requirements are documented in the
[sample prerequisites](../samples/README.md#prerequisites).

| Tool | Version | Scope |
|-----------|-------------------|--------|
| CMake | 3.23 through 3.31. | SDK and sample configuration and builds. |
| Ninja | 1.11.1 or later recommended. | Linux and Android builds. |
| Vulkan SDK | 1.4.350.0 or later recommended. | Vulkan headers and shader build tools. |
| Android NDK | 28.2.13676358. `build.py` uses Android API 33 by default. | Required for native Android AArch64 builds. |
| Python | 3.x. | SDK build script and sample tooling. |

Other versions may also work with the SDK, but have not been validated as part of this release.

## Git Submodule Dependencies

See [`.gitmodules`](../.gitmodules) for the SDK's submodule paths, repository
URLs, and tracking branches. The SDK repository pins each submodule to a
specific commit through a Git gitlink. The pinned commits for this SDK revision are:

| Submodule | Path | Commit |
|-----------|------|--------|
| SPIRV-Cross | `sdk/tools/ffx_shader_compiler/libs/SPIRV-Cross` | `072444287f4e139c178d6d8fe32e04a0d2c34e8b` |
| SPIRV-Headers | `sdk/libs/SPIRV-Headers` | `9268f3057354a2cb65991ba5f38b16d81e803692` |
| SPIRV-Tools | `sdk/libs/SPIRV-Tools` | `336270aaf277f616361e690e2c1e9eef15fc4417` |

## External tools
The following versions have been validated for this release and are recommended for use with the SDK.

| Tools | Version | Source |
|-----------|-------------------|--------|
| Arm ML Emulation Layer for Vulkan | `v0.11.0` | [GitHub release](https://github.com/arm/ai-ml-emulation-layer-for-vulkan/releases/tag/release/v0.11.0) |
| RenderDoc for Arm GPUs | `2026.5` | Included in [Arm Performance Studio](https://developer.arm.com/tools-and-software/arm-performance-studio#Downloads) |

Other versions may also work with the SDK, but have not been validated as part of this release.

## Vulkan Runtime Requirements

This section consolidates the Vulkan requirements for integrating NSS and NFRU.
These are runtime and device-creation requirements, not Vulkan SDK tool-version
recommendations or a claim that every device and shader configuration has been
validated. Use the [User Guide](user_guide.md#integrate-the-sdk) for integration
steps and the [NSS](NSS/NSS.md) and [NFRU](NFRU/NFRU.md) guides for path selection
and resource contracts.

Extension support, extension enablement, feature support, and feature enablement
are separate checks. Enumerate extensions and query features before creating
`VkDevice`, then explicitly enable the requirements for the selected paths.
Physical-device capability queries cannot establish which optional features the
application enabled on an existing device. Do not assume that enabling an
extension also enables its feature bits.

### API Version and Extensions

The ARM tensor and data-graph paths require Vulkan 1.3 or later. Request at least
Vulkan 1.3 in `VkApplicationInfo::apiVersion` and select a physical device that
supports that version. The samples currently request Vulkan 1.4; this is not the
minimum version requirement of the ARM extensions.

The following device-extension requirements apply to both native drivers and
extensions exposed by the ML emulation layers. An extension promoted to core
does not need its extension name enabled when the applicable core version is
used, but any required feature bit still needs to be enabled.

| Requirement | NSS | NFRU | Condition or Core Alternative | NG Layer Responsibility |
|-------------|:---:|:----:|-------------------------------|-------------------------|
| `VK_ARM_tensors` | Required | Required | Tensor resources used by neural processing. | Conditionally adds the extension and tensor feature structure. |
| `VK_ARM_data_graph` | Required | Required | Neural data-graph pipelines; also requires the dependencies below. | Conditionally adds the extension and graph feature structure. |
| `VK_ARM_data_graph_optical_flow` | - | Required | NFRU optical-flow processing; depends on `VK_ARM_data_graph`. | Adds the extension and feature structure when downstream support is available and the structure is absent. |
| `VK_KHR_deferred_host_operations` | Required | Required | Dependency of `VK_ARM_data_graph`. | Conditionally adds the extension name. |
| `VK_KHR_extended_flags` or `VK_KHR_maintenance5` | Required | Required | Enable one dependency path; maintenance5 functionality is core in Vulkan 1.4. | Conditionally adds one advertised extension, preferring maintenance5; does not independently recognize its Vulkan 1.4 core alternative. |
| `VK_KHR_synchronization2` functionality | Required | Required | Core in Vulkan 1.3; enable `synchronization2`, not necessarily the extension name. | Conditionally enables the feature; does not add the extension name. |
| `VK_KHR_dynamic_rendering` functionality | Dependency alternative | Dependency alternative | Maintenance5 depends on Vulkan 1.3 or Vulkan 1.1 plus this extension. The ARM paths already require Vulkan 1.3, so no additional dynamic-rendering extension name is required. | Does not add the extension name or enable `dynamicRendering`. |

The extension dependencies are defined by the Khronos specifications for
[tensors](https://docs.vulkan.org/refpages/latest/refpages/source/VK_ARM_tensors.html),
[data graphs](https://docs.vulkan.org/refpages/latest/refpages/source/VK_ARM_data_graph.html),
[optical flow](https://docs.vulkan.org/refpages/latest/refpages/source/VK_ARM_data_graph_optical_flow.html),
and [maintenance5](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_maintenance5.html).
Dynamic rendering is not a separate NSS/NFRU shader-path requirement merely
because the samples list its extension.

### Device Features

Enable the applicable feature bits through `VkDeviceCreateInfo::pNext` or the
core feature structure. The integer and buffer-storage requirements below are
also present in generated shader SPIR-V, not just in sample feature requests.
Conditional requirements depend on the shader permutation selected by the SDK;
they are not requirements for every permutation.

| Feature | Applies To | Enablement Condition | NG Layer Responsibility |
|---------|------------|----------------------|-------------------------|
| `tensors` | NSS and NFRU | Required to create and use ARM tensor resources. | Conditionally enables it in a newly inserted tensor structure. |
| `dataGraph` | NSS and NFRU | Required for ARM data-graph execution. | Conditionally enables it in a newly inserted graph structure. |
| `dataGraphOpticalFlow` | NFRU | Required for the optical-flow data graph. | Conditionally enables it in a newly inserted optical-flow structure. |
| `shaderTensorAccess` | Native tensor shader paths | Required when selected shaders access tensors directly. A stage without tensor access may use the SDK's buffer-alias path instead, which has separate storage requirements. | Requires downstream support for this bit before ARM injection and enables it in a newly inserted tensor structure. |
| `synchronization2` | NSS and NFRU | Required by the backend's tensor/data-graph barriers and `vkCmdPipelineBarrier2`. Use either `VkPhysicalDeviceVulkan13Features` or `VkPhysicalDeviceSynchronization2Features`, not both. | Conditionally inserts a structure or sets an existing carrier's bit to `VK_TRUE`. |
| `shaderInt8` | NSS and NFRU | Required by quantized shader processing (`Int8` capability). Available in Vulkan 1.2 core features. | Application must enable. |
| `shaderInt16` | Selected NSS and NFRU shaders | Required whenever the selected shader declares `Int16`; includes NSS postprocess variants and some NFRU FP16 variants. This is a core `VkPhysicalDeviceFeatures` bit, separate from FP16 and 16-bit storage. | Application must enable. |
| `storageBuffer8BitAccess` | Tensor buffer-alias paths | Required for selected SSBO shaders declaring `StorageBuffer8BitAccess`. Available in Vulkan 1.2 core features. | Application must enable. |
| `shaderFloat16`, `storageBuffer16BitAccess`, `uniformAndStorageBuffer16BitAccess` | FP16 paths | Enable all three when physical-device capabilities cause the SDK to select FP16. Shader float16 is core in Vulkan 1.2; 16-bit storage is core in Vulkan 1.1. | Application must enable. |
| `fragmentStoresAndAtomics` | NFRU fragment warp and preprocess | Required for the default paths. If unsupported, select both `FFX_API_FG_CONTEXT_FLAG_WARP_COMPUTE` and `FFX_API_FG_CONTEXT_FLAG_PRE_PROCESS_COMPUTE`. | Application must enable. |
| `shaderStorageImageExtendedFormats` | Selected compute paths | Required by shaders declaring `StorageImageExtendedFormats`, including NSS compute preprocess/postprocess and NFRU compute warp variants. The actual image format must also support storage-image usage. NSS compute postprocess/debug requires R11G11B10 optimal-tiling storage-image support. | Application must enable; format support must be checked separately. |
| `timelineSemaphore` | Frame-interpolation swapchain integration | Required when using the SDK's timeline-semaphore-based swapchain path; not a blanket requirement for manual NSS/NFRU dispatch. Available in Vulkan 1.2 core features. | On Vulkan 1.3+ devices that pass the ARM tensor/data-graph support gate, the NG layer adds the advertised `VK_KHR_timeline_semaphore` extension when missing, and injects `timelineSemaphore = VK_TRUE` when supported and no timeline or Vulkan 1.2 feature carrier was supplied. If the application supplies a carrier, it must enable the bit itself; the layer preserves its value. The layer does not provide unsupported capability. |

NSS can select FP32 using `FFX_API_NSS_CONTEXT_FLAG_DISABLE_16BIT`. This flag
does not guarantee that the selected shaders are free of integer `Int16`
operations or 8-bit storage requirements. NFRU does not expose a context flag
to force FP32 when the backend reports FP16 support.

For native tensor access, also check `shaderTensorSupportedStages` for the
selected compute or fragment stages. If a required integer or storage feature
is unsupported, enabling a related extension is not a substitute. Select only
a shader path that avoids that capability, where available; otherwise the
device/configuration cannot run that path. The requirements here do not replace
the component guides' image-format, optical-flow property, queue, and resource
compatibility checks.

### NG Layer Coverage and Limits

See the [NG layer coverage and limits](../layer/README.md#ng-layer-coverage-and-limits)
for automatic enablement conditions, application responsibilities, and layer order.

### Current Backend Constraints

The shared Vulkan backend currently treats surface and swapchain entry points
as mandatory during initial function-table loading. This also affects manual
and offscreen NSS/NFRU use, even when no swapchain is created:

| Requirement | Scope | Reason | NG Layer Responsibility |
|-------------|-------|--------|-------------------------|
| `VK_KHR_surface` | Instance extension | The backend requires `vkGetPhysicalDeviceSurfaceSupportKHR`. | Application must enable. |
| `VK_KHR_swapchain` | Device extension | The backend requires swapchain creation, acquisition, destruction, and presentation entry points. | Application must enable. |

These are current backend initialization constraints, not intrinsic
requirements of neural super sampling or frame interpolation. Missing mandatory
entry points cause Vulkan wrapper initialization to fail. Platform-specific
surface extensions are additionally required when the application actually
creates a platform surface; they depend on the application's window system.
Merely loading the core timeline-semaphore entry points does not require
enabling `timelineSemaphore` unless the selected integration uses that feature.

The Vulkan function table is process-global. `InitVulkanWrapper` loads it only
until the first successful initialization; it does not lock initialization,
key the table by device, or reset it when contexts are destroyed. Serialize
first backend creation and retain the originating device and loader module
while the table is in use. Independent multi-device operation and
device/loader recreation in the same process are not guaranteed by this
implementation, even though contexts otherwise have separate effect state.
