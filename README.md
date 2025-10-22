# Welcome to the Neural Graphics SDK for Game Engines

Neural Graphics SDK for Game Engines is Arm’s unified graphics software development kit designed to support multiple rendering use cases across diverse engines and platforms. It is derived from AMD's FFX SDK 1.1.3(https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK). It consolidates technologies like NSS (Neural Super Sampling), ASR(Accuracy Super Resolution), and NFRU (Neural Frame Rate Upsampling) into a modular, engine-agnostic framework.

Currently, Neural Graphics SDK for Game Engines supports [NSS](/docs/NSS/NSS.md).

![invert](/docs/media/overview.svg "A diagram showing the sdk structure.")

'*' means not supported yet.

*ASR: Not supported yet. But it will be a fallback for NSS, for some devices cannot support ML extensions for Vulkan.

api_layer: we are compliant with AMD FidelityFX API 1.1.3.

components: Currently we support [NSS](/docs/NSS/NSS.md). All components are implemented independently in this layer.

backend: All components share one backend. It will access deeper layers via Vulkan APIs.

[Vulkan Emulation Layer](#vulkan-emulation-layer): Emulation layers for supporting the ML extensions and Vulkan headers before the real device is ready.(https://github.com/arm/ai-ml-emulation-layer-for-vulkan/releases)

## System requirement

Cmake: minimun version 3.21, maxmum version 3.31

Vulkan SDK: recommend version 1.4.321.0. You can download from Lunarg(https://www.lunarg.com/vulkan-sdk/).

## Vulkan Emulation Layer

After the SDK has been integrated, you are ready to run Neural Graphics technologies, like [NSS](/docs/NSS/NSS.md) in your project. But before real devices are ready for ML extensions for Vulkan, you need to enable Vulkan emulation layers.

You can use "Vulkan configurator" in your vulkan sdk:

![invert](/docs/media/Enable_vulkan_emulation_layer.png "A screenshot to enable vulkan emulation layer.")

Or add Vulkan emulation layer libs' path to your system environment. For example:
Windows:

```bash
set "VK_ADD_LAYER_PATH=path\to\VulkanML"
set "VK_INSTANCE_LAYERS=VK_LAYER_ML_Graph_Emulation;VK_LAYER_ML_Tensor_Emulation"
```

Linux:

```bash
export VK_ADD_LAYER_PATH="path/to/VulkanMLLib"
export LD_LIBRARY_PATH="path/to/VulkanMLLib"
export VK_INSTANCE_LAYERS="VK_LAYER_ML_Graph_Emulation:VK_LAYER_ML_Tensor_Emulation"
```

Whatever method you use, make sure VK_LAYER_ML_Graph_Emulation is enabled before VK_LAYER_ML_Tensor_Emulation.

## Build SDK

Before building the SDK, you need to make sure the repository is fully cloned, including all submodules and Git LFS files:

1. **Clone submodules**: The SDK uses third-party libraries tracked as git submodules. After cloning the main repository, initialize and update the submodules:
    ```bash
    git submodule sync --recursive
    git submodule update --init --recursive
    ```

2. **Pull LFS files**: Some dependent files are stored using Git LFS. Pull these files with:
    ```bash
    git lfs install
    git lfs pull
    ```

There are build.bat and build.sh for windows and linux to build the SDK. You can build the sdk by calling these scripts file.

If everything is ok, libs for SDK will be generated into folder "./bin".

## Enable extensions

When create vulkan physical device, must enable necessary extensions: VulkanExtensions::kVK_ARM_tensors and VulkanExtensions::kVK_ARM_data_graph. Sample code:

```cpp
VkDeviceCreateInfo ci = {};
ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
... //Other create info settings
 
VkPhysicalDeviceTensorFeaturesARM tensorFeature = {};
{
    tensorFeature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM;
 
    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &tensorFeature;
    vkGetPhysicalDeviceFeatures2(m_physicalDevice, &features2);
 
    tensorFeature.pNext = const_cast<void*>(ci.pNext);
    ci.pNext = &tensorFeature;
}
 
VkPhysicalDeviceDataGraphFeaturesARM dataGraphFeature = {};
{
    dataGraphFeature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM;
 
    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &dataGraphFeature;
    vkGetPhysicalDeviceFeatures2(m_physicalDevice, &features2);
 
    dataGraphFeature.pNext = const_cast<void*>(ci.pNext);
    ci.pNext = &tensorFeature; 
}
 
vkCreateDevice(m_physicalDevice, &ci, nullptr, &m_vkdevice)
```

## Limitations

Only supports vulkan backend.

## References

For more information, refer to the [Developer guide](https://developer.arm.com/documentation/111167/latest).

## Trademarks and Copyrights

AMD is a trademark of Advanced Micro Devices, Inc.

AMD FidelityFX™ is a trademark of Advanced Micro Devices, Inc.

Arm® is a registered trademark of Arm Limited (or its subsidiaries) in the US and/or elsewhere.

Vulkan is a registered trademark and the Vulkan SC logo is a trademark of the Khronos Group Inc.

Visual Studio, Windows are registered trademarks or trademarks of Microsoft Corporation in the US and other jurisdictions.