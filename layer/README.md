# NG Vulkan Layer

The root SDK CMake build includes the `VK_LAYER_ARM_NG` Vulkan layer and stages
its shared library and platform manifest into `bin/`. The layer target requires
CMake 3.21 or later and a C++17 compiler. It uses the SDK's bundled Vulkan
headers, including the ARM tensor and data-graph declarations; a standalone
build can override the header location with `NGLAYER_VULKAN_INCLUDE_DIR`.

Platform-specific library names are used so Linux and Android binaries can be
packaged together without renaming a built shared library:

| Platform | Library | Manifest |
| --- | --- | --- |
| Linux | `libVkLayer_arm_NG_linux.so` | `VkLayer_arm_NG_linux.json` |
| Android | `libVkLayer_arm_NG_android.so` | `VkLayer_arm_NG_android.json` |
| Windows (MSVC) | `VkLayer_arm_NG_windows.dll` | `VkLayer_arm_NG_windows.json` |

## NG Layer Coverage and Limits

`VK_LAYER_ARM_NG` is an enablement helper, not an ML implementation. It forwards
capability queries to the downstream driver/layers and does not provide missing
tensor, graph, or optical-flow functionality. On devices without native support,
use the compatible ML emulation-layer version listed in the
[Compatibility Matrix](../docs/compatibility_matrix.md#external-tools) and configure
the order as `VK_LAYER_ARM_NG`, `VK_LAYER_ML_Graph_Emulation`, then
`VK_LAYER_ML_Tensor_Emulation`. See the
[ML Emulation Layer setup](../docs/user_guide.md#ml-emulation-layer-for-vulkan).

The current layer's automatic enablement has these limits:

- ARM injection requires a physical-device API version of at least 1.3, support
	for `tensors` and `shaderTensorAccess`, and support for `dataGraph`. Otherwise
	device creation is passed through without ARM injection. The gate checks the
	device version, not the application's requested instance API version.
- ARM feature structures are considered supplied when their `sType` is present.
	Existing ARM feature bits set to `VK_FALSE` are not changed to `VK_TRUE`, and
	the corresponding extension name is not independently added for an existing
	structure. Applications supplying ARM structures must enable their feature
	bits and extension names themselves.
- Synchronization2 and data-graph dependency completion occurs only when the
	layer inserts at least one ARM feature structure. If all applicable ARM
	structures are already present, it does not fill missing dependencies or
	enable synchronization2. Missing advertised dependency extensions are logged,
	not implemented by the layer.
- On supported ARM devices that advertise the timeline feature and
	`VK_KHR_timeline_semaphore`, the layer adds the missing extension name and
	feature structure independently of ARM feature injection. Existing timeline
	and Vulkan 1.2 feature structures and their values are preserved. Missing
	timeline support is skipped without introducing new device-creation errors.
	The layer uses the KHR path without tracking the application's API version.
- Shader integer, buffer-storage, FP16, fragment atomic, storage-image, and WSI
	requirements remain the application's responsibility.

For predictable integration, explicitly enable the complete selected-path
[Vulkan runtime requirements](../docs/compatibility_matrix.md#vulkan-runtime-requirements)
even when using NG layer. Do not rely on partial automatic injection to repair
an incomplete application feature chain.
