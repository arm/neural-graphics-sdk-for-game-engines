/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "vk_test_utils.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

namespace
{
    std::filesystem::path gShaderDirectory = "shader";

    void SetError(std::string* errorMessage, const std::string& message)
    {
        if (errorMessage)
        {
            *errorMessage = message;
        }
    }

    std::filesystem::path GetShaderDir()
    {
        return gShaderDirectory;
    }

    bool LoadSpirvFile(const char* fileName, std::vector<uint32_t>& outWords, std::string* errorMessage)
    {
        const std::filesystem::path shaderPath = GetShaderDir() / fileName;
        std::ifstream               file(shaderPath, std::ios::binary | std::ios::ate);
        if (!file.is_open())
        {
            SetError(errorMessage, "Failed to open SPIR-V shader: " + shaderPath.string());
            return false;
        }

        const std::streamsize fileSize = file.tellg();
        if (fileSize <= 0 || (fileSize % static_cast<std::streamsize>(sizeof(uint32_t))) != 0)
        {
            SetError(errorMessage, "Invalid SPIR-V shader size: " + shaderPath.string());
            return false;
        }

        file.seekg(0, std::ios::beg);
        std::vector<char> bytes(static_cast<size_t>(fileSize));
        if (!file.read(bytes.data(), fileSize))
        {
            SetError(errorMessage, "Failed to read SPIR-V shader: " + shaderPath.string());
            return false;
        }

        outWords.resize(static_cast<size_t>(fileSize) / sizeof(uint32_t));
        std::memcpy(outWords.data(), bytes.data(), static_cast<size_t>(fileSize));
        return true;
    }

    bool LoadTriangleShaders(std::vector<uint32_t>& vertexSpv, std::vector<uint32_t>& fragmentSpv, std::string* errorMessage)
    {
        if (!LoadSpirvFile("triangle.vert.spv", vertexSpv, errorMessage))
        {
            return false;
        }
        if (!LoadSpirvFile("triangle.frag.spv", fragmentSpv, errorMessage))
        {
            return false;
        }
        return true;
    }

    constexpr const char* kSelectedVulkanDeviceEnv = "SDK_UNIT_TEST_VK_DEVICE";

    std::string TrimCopy(const std::string& value)
    {
        size_t begin = 0;
        while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0)
        {
            ++begin;
        }

        size_t end = value.size();
        while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0)
        {
            --end;
        }

        return value.substr(begin, end - begin);
    }

    std::string ToLowerCopy(const std::string& value)
    {
        std::string lowered = value;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](const unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        return lowered;
    }

    bool TryParseDeviceIndex(const std::string& selector, uint32_t& outIndex)
    {
        if (selector.empty())
        {
            return false;
        }

        for (const char ch : selector)
        {
            if (std::isdigit(static_cast<unsigned char>(ch)) == 0)
            {
                return false;
            }
        }

        errno                      = 0;
        char*               endPtr = nullptr;
        const unsigned long parsed = std::strtoul(selector.c_str(), &endPtr, 10);
        if (errno != 0 || endPtr == nullptr || *endPtr != '\0' || parsed > std::numeric_limits<uint32_t>::max())
        {
            return false;
        }

        outIndex = static_cast<uint32_t>(parsed);
        return true;
    }

    bool HasExtension(const std::vector<VkExtensionProperties>& availableExtensions, const char* extensionName)
    {
        return std::any_of(availableExtensions.begin(), availableExtensions.end(), [extensionName](const VkExtensionProperties& extension) {
            return std::strcmp(extension.extensionName, extensionName) == 0;
        });
    }

    void AddExtensionIfAvailable(const std::vector<VkExtensionProperties>& availableExtensions,
                                 const char*                               extensionName,
                                 std::vector<const char*>&                 enabledExtensions)
    {
        if (!HasExtension(availableExtensions, extensionName))
        {
            return;
        }

        const bool alreadyEnabled = std::any_of(enabledExtensions.begin(), enabledExtensions.end(), [extensionName](const char* enabledName) {
            return std::strcmp(enabledName, extensionName) == 0;
        });
        if (!alreadyEnabled)
        {
            enabledExtensions.push_back(extensionName);
        }
    }

    bool CreateUnitTestInstance(VkInstance& instance, std::string* errorMessage)
    {
        instance = VK_NULL_HANDLE;

        uint32_t extensionCount = 0;
        if (vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkEnumerateInstanceExtensionProperties failed");
            return false;
        }

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        if (extensionCount > 0 && vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data()) != VK_SUCCESS)
        {
            SetError(errorMessage, "failed to enumerate Vulkan instance extensions");
            return false;
        }

        std::vector<const char*> enabledInstanceExtensions;
        for (const VkExtensionProperties& extension : availableExtensions)
        {
            if (std::strcmp(extension.extensionName, VK_KHR_SURFACE_EXTENSION_NAME) == 0)
            {
                enabledInstanceExtensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
                break;
            }
        }

        VkApplicationInfo appInfo{};
        appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName   = "ngsdk_unit_test";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName        = "ngsdk_unit_test";
        appInfo.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion         = VK_API_VERSION_1_3;

        VkInstanceCreateInfo instanceCreateInfo{};
        instanceCreateInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceCreateInfo.pApplicationInfo        = &appInfo;
        instanceCreateInfo.enabledExtensionCount   = static_cast<uint32_t>(enabledInstanceExtensions.size());
        instanceCreateInfo.ppEnabledExtensionNames = enabledInstanceExtensions.empty() ? nullptr : enabledInstanceExtensions.data();

        if (vkCreateInstance(&instanceCreateInfo, nullptr, &instance) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkCreateInstance failed");
            return false;
        }

        return true;
    }

    bool EnumeratePhysicalDevices(VkInstance instance, std::vector<VkPhysicalDevice>& physicalDevices, std::string* errorMessage)
    {
        uint32_t physicalDeviceCount = 0;
        if (vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr) != VK_SUCCESS || physicalDeviceCount == 0)
        {
            SetError(errorMessage, "no Vulkan physical device found");
            return false;
        }

        physicalDevices.resize(physicalDeviceCount);
        if (vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data()) != VK_SUCCESS)
        {
            SetError(errorMessage, "failed to enumerate Vulkan physical devices");
            physicalDevices.clear();
            return false;
        }

        return true;
    }

    std::vector<unit_test::vk::VulkanPhysicalDeviceInfo> BuildDeviceInfos(const std::vector<VkPhysicalDevice>& physicalDevices)
    {
        std::vector<unit_test::vk::VulkanPhysicalDeviceInfo> infos;
        infos.reserve(physicalDevices.size());

        for (uint32_t i = 0; i < static_cast<uint32_t>(physicalDevices.size()); ++i)
        {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(physicalDevices[i], &properties);

            unit_test::vk::VulkanPhysicalDeviceInfo info;
            info.index    = i;
            info.vendorId = properties.vendorID;
            info.name     = properties.deviceName;
            infos.push_back(std::move(info));
        }

        return infos;
    }

    std::string FormatDeviceList(const std::vector<unit_test::vk::VulkanPhysicalDeviceInfo>& infos)
    {
        std::ostringstream stream;
        for (const unit_test::vk::VulkanPhysicalDeviceInfo& info : infos)
        {
            stream << "[" << info.index << "] " << info.name << '\n';
        }
        return stream.str();
    }

    bool SelectPhysicalDevice(const std::vector<VkPhysicalDevice>& physicalDevices,
                              const std::string&                   requestedSelector,
                              VkPhysicalDevice&                    outDevice,
                              std::string*                         errorMessage)
    {
        const std::vector<unit_test::vk::VulkanPhysicalDeviceInfo> infos         = BuildDeviceInfos(physicalDevices);
        uint32_t                                                   selectedIndex = 0;
        if (!unit_test::vk::SelectVulkanDeviceIndex(infos, requestedSelector, selectedIndex, errorMessage))
        {
            return false;
        }

        outDevice = physicalDevices[selectedIndex];
        return true;
    }

    bool IsDepthFormat(const VkFormat format)
    {
        switch (format)
        {
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_D16_UNORM_S8_UINT:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return true;
        default:
            return false;
        }
    }

    VkImageAspectFlags GetImageAspectMask(const VkFormat format)
    {
        if (IsDepthFormat(format))
        {
            return VK_IMAGE_ASPECT_DEPTH_BIT;
        }
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }

    bool FindMemoryType(const VkPhysicalDeviceMemoryProperties& memoryProperties,
                        const uint32_t                          typeFilter,
                        const VkMemoryPropertyFlags             requiredProperties,
                        uint32_t&                               outMemoryType)
    {
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
        {
            if ((typeFilter & (1u << i)) == 0)
            {
                continue;
            }
            if ((memoryProperties.memoryTypes[i].propertyFlags & requiredProperties) != requiredProperties)
            {
                continue;
            }
            outMemoryType = i;
            return true;
        }
        return false;
    }

    void ResetContext(unit_test::vk::VulkanContext& context)
    {
        context.instance            = VK_NULL_HANDLE;
        context.physicalDevice      = VK_NULL_HANDLE;
        context.device              = VK_NULL_HANDLE;
        context.queue               = VK_NULL_HANDLE;
        context.queueFamilyIndex    = 0;
        context.commandPool         = VK_NULL_HANDLE;
        context.memoryProperties    = {};
        context.getInstanceProcAddr = nullptr;
        context.getDeviceProcAddr   = nullptr;
    }

    bool TransitionImage(VkCommandBuffer commandBuffer, VkImage image, VkImageAspectFlags aspectMask, VkImageLayout oldLayout, VkImageLayout newLayout)
    {
        if (oldLayout == newLayout)
        {
            return true;
        }

        VkImageMemoryBarrier barrier{};
        barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout                       = oldLayout;
        barrier.newLayout                       = newLayout;
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.image                           = image;
        barrier.subresourceRange.aspectMask     = aspectMask;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;
        barrier.srcAccessMask                   = (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) ? 0 : (VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        barrier.dstAccessMask                   = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        return true;
    }

    template <typename RecorderFn>
    bool RunOneTimeCommands(unit_test::vk::VulkanContext& context, RecorderFn&& recorder, std::string* errorMessage)
    {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        if (!unit_test::vk::BeginCommandBuffer(context, commandBuffer, errorMessage))
        {
            return false;
        }

        recorder(commandBuffer);

        if (!unit_test::vk::SubmitCommandBuffer(context, commandBuffer, errorMessage))
        {
            return false;
        }
        return true;
    }

}  // namespace

namespace unit_test::vk
{
    void SetShaderDirectory(const std::string& shaderDirectory)
    {
        gShaderDirectory = shaderDirectory;
    }

    std::string ResolveVulkanDeviceSelector(const std::string& commandLineSelector, const char* environmentSelector)
    {
        const std::string trimmedCommandLineSelector = TrimCopy(commandLineSelector);
        if (!trimmedCommandLineSelector.empty())
        {
            return trimmedCommandLineSelector;
        }
        return environmentSelector ? TrimCopy(environmentSelector) : std::string{};
    }

    bool SelectVulkanDeviceIndex(const std::vector<VulkanPhysicalDeviceInfo>& devices,
                                 const std::string&                           requestedSelector,
                                 uint32_t&                                    selectedIndex,
                                 std::string*                                 errorMessage)
    {
        if (devices.empty())
        {
            SetError(errorMessage, "no Vulkan physical device found");
            return false;
        }

        const std::string selector = TrimCopy(requestedSelector);
        if (selector.empty())
        {
            selectedIndex = 0;
            return true;
        }

        uint32_t parsedIndex = 0;
        if (TryParseDeviceIndex(selector, parsedIndex))
        {
            if (parsedIndex >= devices.size())
            {
                SetError(errorMessage, "Requested Vulkan device index '" + selector + "' was not found. Available devices:\n" + FormatDeviceList(devices));
                return false;
            }
            selectedIndex = parsedIndex;
            return true;
        }

        const std::string selectorLower  = ToLowerCopy(selector);
        const bool        lavapipeAlias  = selectorLower == "lavapipe";
        const bool        nvidiaAlias    = selectorLower == "nvidia";
        const bool        intelAlias     = selectorLower == "intel";
        const uint32_t    nvidiaVendorId = static_cast<uint32_t>(FfxGpuVendorId::Nvidia);
        const uint32_t    intelVendorId  = static_cast<uint32_t>(FfxGpuVendorId::Intel);
        for (uint32_t index = 0; index < static_cast<uint32_t>(devices.size()); ++index)
        {
            const VulkanPhysicalDeviceInfo& device          = devices[index];
            const std::string               deviceNameLower = ToLowerCopy(device.name);
            if (deviceNameLower.find(selectorLower) != std::string::npos || (lavapipeAlias && deviceNameLower.find("llvmpipe") != std::string::npos) ||
                (nvidiaAlias && device.vendorId == nvidiaVendorId) || (intelAlias && device.vendorId == intelVendorId))
            {
                selectedIndex = index;
                return true;
            }
        }

        SetError(errorMessage, "Requested Vulkan device '" + selector + "' was not found. Available devices:\n" + FormatDeviceList(devices));
        return false;
    }

    bool EnumerateVulkanDevices(std::vector<VulkanPhysicalDeviceInfo>& devices, std::string* errorMessage)
    {
        devices.clear();

        VkInstance instance = VK_NULL_HANDLE;
        if (!CreateUnitTestInstance(instance, errorMessage))
        {
            return false;
        }

        std::vector<VkPhysicalDevice> physicalDevices;
        if (!EnumeratePhysicalDevices(instance, physicalDevices, errorMessage))
        {
            vkDestroyInstance(instance, nullptr);
            return false;
        }

        devices = BuildDeviceInfos(physicalDevices);
        vkDestroyInstance(instance, nullptr);
        return true;
    }

    bool InitializeVulkanContext(VulkanContext& context, std::string* errorMessage)
    {
        ResetContext(context);

        context.getInstanceProcAddr = vkGetInstanceProcAddr;
        context.getDeviceProcAddr   = vkGetDeviceProcAddr;

        if (!CreateUnitTestInstance(context.instance, errorMessage))
        {
            return false;
        }

        std::vector<VkPhysicalDevice> physicalDevices;
        if (!EnumeratePhysicalDevices(context.instance, physicalDevices, errorMessage))
        {
            DestroyVulkanContext(context);
            return false;
        }

        const std::string requestedDeviceSelector = ResolveVulkanDeviceSelector({}, std::getenv(kSelectedVulkanDeviceEnv));

        if (!SelectPhysicalDevice(physicalDevices, requestedDeviceSelector, context.physicalDevice, errorMessage))
        {
            DestroyVulkanContext(context);
            return false;
        }

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(context.physicalDevice, &queueFamilyCount, nullptr);
        if (queueFamilyCount == 0)
        {
            SetError(errorMessage, "no queue family available");
            DestroyVulkanContext(context);
            return false;
        }

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(context.physicalDevice, &queueFamilyCount, queueFamilies.data());

        bool foundQueue = false;
        for (uint32_t i = 0; i < queueFamilyCount; ++i)
        {
            const bool hasGraphics = (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
            const bool hasCompute  = (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
            if (hasGraphics && hasCompute)
            {
                context.queueFamilyIndex = i;
                foundQueue               = true;
                break;
            }
        }
        if (!foundQueue)
        {
            for (uint32_t i = 0; i < queueFamilyCount; ++i)
            {
                if ((queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0)
                {
                    context.queueFamilyIndex = i;
                    foundQueue               = true;
                    break;
                }
            }
        }
        if (!foundQueue)
        {
            for (uint32_t i = 0; i < queueFamilyCount; ++i)
            {
                if ((queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0)
                {
                    context.queueFamilyIndex = i;
                    foundQueue               = true;
                    break;
                }
            }
        }
        if (!foundQueue)
        {
            SetError(errorMessage, "unable to find graphics/compute queue");
            DestroyVulkanContext(context);
            return false;
        }

        const float             queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = context.queueFamilyIndex;
        queueCreateInfo.queueCount       = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo deviceCreateInfo{};
        deviceCreateInfo.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos    = &queueCreateInfo;

        uint32_t deviceExtensionCount = 0;
        if (vkEnumerateDeviceExtensionProperties(context.physicalDevice, nullptr, &deviceExtensionCount, nullptr) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkEnumerateDeviceExtensionProperties failed");
            DestroyVulkanContext(context);
            return false;
        }

        std::vector<VkExtensionProperties> availableDeviceExtensions(deviceExtensionCount);
        if (deviceExtensionCount > 0 &&
            vkEnumerateDeviceExtensionProperties(context.physicalDevice, nullptr, &deviceExtensionCount, availableDeviceExtensions.data()) != VK_SUCCESS)
        {
            SetError(errorMessage, "failed to enumerate Vulkan device extensions");
            DestroyVulkanContext(context);
            return false;
        }

        std::vector<const char*> enabledDeviceExtensions;
        enabledDeviceExtensions.reserve(8);
        AddExtensionIfAvailable(availableDeviceExtensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME, enabledDeviceExtensions);
        AddExtensionIfAvailable(availableDeviceExtensions, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME, enabledDeviceExtensions);
        AddExtensionIfAvailable(availableDeviceExtensions, VK_KHR_MAINTENANCE_5_EXTENSION_NAME, enabledDeviceExtensions);
        AddExtensionIfAvailable(availableDeviceExtensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, enabledDeviceExtensions);
        AddExtensionIfAvailable(availableDeviceExtensions, VK_ARM_TENSORS_EXTENSION_NAME, enabledDeviceExtensions);
        AddExtensionIfAvailable(availableDeviceExtensions, VK_ARM_DATA_GRAPH_EXTENSION_NAME, enabledDeviceExtensions);
        AddExtensionIfAvailable(availableDeviceExtensions, VK_ARM_DATA_GRAPH_OPTICAL_FLOW_EXTENSION_NAME, enabledDeviceExtensions);

        const bool tensorExtensionEnabled    = std::any_of(enabledDeviceExtensions.begin(), enabledDeviceExtensions.end(), [](const char* extensionName) {
            return std::strcmp(extensionName, VK_ARM_TENSORS_EXTENSION_NAME) == 0;
        });
        const bool dataGraphExtensionEnabled = std::any_of(enabledDeviceExtensions.begin(), enabledDeviceExtensions.end(), [](const char* extensionName) {
            return std::strcmp(extensionName, VK_ARM_DATA_GRAPH_EXTENSION_NAME) == 0;
        });
        const bool dataGraphOpticalFlowExtensionEnabled =
            std::any_of(enabledDeviceExtensions.begin(), enabledDeviceExtensions.end(), [](const char* extensionName) {
                return std::strcmp(extensionName, VK_ARM_DATA_GRAPH_OPTICAL_FLOW_EXTENSION_NAME) == 0;
            });

        VkPhysicalDeviceFeatures2 supportedFeatures{};
        supportedFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

        VkPhysicalDeviceVulkan11Features supportedVulkan11Features{};
        supportedVulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;

        VkPhysicalDeviceVulkan12Features supportedVulkan12Features{};
        supportedVulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

        VkPhysicalDeviceVulkan13Features supportedVulkan13Features{};
        supportedVulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

        VkPhysicalDeviceTensorFeaturesARM supportedTensorFeatures{};
        supportedTensorFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM;

        VkPhysicalDeviceDataGraphFeaturesARM supportedDataGraphFeatures{};
        supportedDataGraphFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM;

        VkPhysicalDeviceDataGraphOpticalFlowFeaturesARM supportedDataGraphOpticalFlowFeatures{};
        supportedDataGraphOpticalFlowFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_OPTICAL_FLOW_FEATURES_ARM;

        void** supportedFeatureChainTail = reinterpret_cast<void**>(&supportedFeatures.pNext);
        auto   appendSupportedFeature    = [&supportedFeatureChainTail](auto* featureStruct) {
            *supportedFeatureChainTail = featureStruct;
            supportedFeatureChainTail  = &featureStruct->pNext;
        };
        appendSupportedFeature(&supportedVulkan11Features);
        appendSupportedFeature(&supportedVulkan12Features);
        appendSupportedFeature(&supportedVulkan13Features);
        if (tensorExtensionEnabled)
        {
            appendSupportedFeature(&supportedTensorFeatures);
        }
        if (dataGraphExtensionEnabled)
        {
            appendSupportedFeature(&supportedDataGraphFeatures);
        }
        if (dataGraphOpticalFlowExtensionEnabled)
        {
            appendSupportedFeature(&supportedDataGraphOpticalFlowFeatures);
        }
        *supportedFeatureChainTail = nullptr;

        vkGetPhysicalDeviceFeatures2(context.physicalDevice, &supportedFeatures);

        VkPhysicalDeviceFeatures2 enabledFeatures{};
        enabledFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

        VkPhysicalDeviceVulkan11Features enabledVulkan11Features{};
        enabledVulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;

        VkPhysicalDeviceVulkan12Features enabledVulkan12Features{};
        enabledVulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

        VkPhysicalDeviceVulkan13Features enabledVulkan13Features{};
        enabledVulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

        VkPhysicalDeviceTensorFeaturesARM enabledTensorFeatures{};
        enabledTensorFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM;

        VkPhysicalDeviceDataGraphFeaturesARM enabledDataGraphFeatures{};
        enabledDataGraphFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM;

        VkPhysicalDeviceDataGraphOpticalFlowFeaturesARM enabledDataGraphOpticalFlowFeatures{};
        enabledDataGraphOpticalFlowFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_OPTICAL_FLOW_FEATURES_ARM;

        void** enabledFeatureChainTail = reinterpret_cast<void**>(&enabledFeatures.pNext);
        auto   appendEnabledFeature    = [&enabledFeatureChainTail](auto* featureStruct) {
            *enabledFeatureChainTail = featureStruct;
            enabledFeatureChainTail  = &featureStruct->pNext;
        };
        appendEnabledFeature(&enabledVulkan11Features);
        appendEnabledFeature(&enabledVulkan12Features);
        appendEnabledFeature(&enabledVulkan13Features);
        if (tensorExtensionEnabled)
        {
            appendEnabledFeature(&enabledTensorFeatures);
        }
        if (dataGraphExtensionEnabled)
        {
            appendEnabledFeature(&enabledDataGraphFeatures);
        }
        if (dataGraphOpticalFlowExtensionEnabled)
        {
            appendEnabledFeature(&enabledDataGraphOpticalFlowFeatures);
        }
        *enabledFeatureChainTail = nullptr;

        enabledVulkan11Features.storageBuffer16BitAccess           = supportedVulkan11Features.storageBuffer16BitAccess;
        enabledVulkan11Features.uniformAndStorageBuffer16BitAccess = supportedVulkan11Features.uniformAndStorageBuffer16BitAccess;
        enabledVulkan12Features.shaderInt8                         = supportedVulkan12Features.shaderInt8;
        enabledVulkan12Features.shaderFloat16                      = supportedVulkan12Features.shaderFloat16;
        enabledFeatures.features.fragmentStoresAndAtomics          = supportedFeatures.features.fragmentStoresAndAtomics;
        enabledFeatures.features.shaderStorageImageExtendedFormats = supportedFeatures.features.shaderStorageImageExtendedFormats;
        enabledVulkan13Features.synchronization2                   = supportedVulkan13Features.synchronization2;
        enabledVulkan13Features.dynamicRendering                   = supportedVulkan13Features.dynamicRendering;

        if (tensorExtensionEnabled)
        {
            enabledTensorFeatures.shaderTensorAccess = supportedTensorFeatures.shaderTensorAccess;
            enabledTensorFeatures.tensors            = supportedTensorFeatures.tensors;
        }
        if (dataGraphExtensionEnabled)
        {
            enabledDataGraphFeatures.dataGraph = supportedDataGraphFeatures.dataGraph;
        }
        if (dataGraphOpticalFlowExtensionEnabled)
        {
            enabledDataGraphOpticalFlowFeatures.dataGraphOpticalFlow = supportedDataGraphOpticalFlowFeatures.dataGraphOpticalFlow;
        }

        deviceCreateInfo.enabledExtensionCount   = static_cast<uint32_t>(enabledDeviceExtensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = enabledDeviceExtensions.empty() ? nullptr : enabledDeviceExtensions.data();
        deviceCreateInfo.pNext                   = &enabledFeatures;

        if (vkCreateDevice(context.physicalDevice, &deviceCreateInfo, nullptr, &context.device) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkCreateDevice failed");
            DestroyVulkanContext(context);
            return false;
        }

        vkGetDeviceQueue(context.device, context.queueFamilyIndex, 0, &context.queue);
        if (context.queue == VK_NULL_HANDLE)
        {
            SetError(errorMessage, "vkGetDeviceQueue returned null queue");
            DestroyVulkanContext(context);
            return false;
        }

        VkCommandPoolCreateInfo commandPoolCreateInfo{};
        commandPoolCreateInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandPoolCreateInfo.queueFamilyIndex = context.queueFamilyIndex;
        if (vkCreateCommandPool(context.device, &commandPoolCreateInfo, nullptr, &context.commandPool) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkCreateCommandPool failed");
            DestroyVulkanContext(context);
            return false;
        }

        vkGetPhysicalDeviceMemoryProperties(context.physicalDevice, &context.memoryProperties);
        return true;
    }

    void DestroyVulkanContext(VulkanContext& context)
    {
        if (context.device != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(context.device);
        }

        if (context.commandPool != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(context.device, context.commandPool, nullptr);
            context.commandPool = VK_NULL_HANDLE;
        }
        if (context.device != VK_NULL_HANDLE)
        {
            vkDestroyDevice(context.device, nullptr);
            context.device = VK_NULL_HANDLE;
        }
        if (context.instance != VK_NULL_HANDLE)
        {
            vkDestroyInstance(context.instance, nullptr);
            context.instance = VK_NULL_HANDLE;
        }
        context.physicalDevice = VK_NULL_HANDLE;
        context.queue          = VK_NULL_HANDLE;
    }

    bool BeginCommandBuffer(VulkanContext& context, VkCommandBuffer& commandBuffer, std::string* errorMessage)
    {
        commandBuffer = VK_NULL_HANDLE;

        VkCommandBufferAllocateInfo allocateInfo{};
        allocateInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocateInfo.commandPool        = context.commandPool;
        allocateInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(context.device, &allocateInfo, &commandBuffer) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkAllocateCommandBuffers failed");
            return false;
        }

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkBeginCommandBuffer failed");
            vkFreeCommandBuffers(context.device, context.commandPool, 1, &commandBuffer);
            commandBuffer = VK_NULL_HANDLE;
            return false;
        }

        return true;
    }

    bool SubmitCommandBuffer(VulkanContext& context, VkCommandBuffer commandBuffer, std::string* errorMessage)
    {
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkEndCommandBuffer failed");
            vkFreeCommandBuffers(context.device, context.commandPool, 1, &commandBuffer);
            return false;
        }

        VkFenceCreateInfo fenceCreateInfo{};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence         = VK_NULL_HANDLE;
        if (vkCreateFence(context.device, &fenceCreateInfo, nullptr, &fence) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkCreateFence failed");
            vkFreeCommandBuffers(context.device, context.commandPool, 1, &commandBuffer);
            return false;
        }

        VkSubmitInfo submitInfo{};
        submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &commandBuffer;
        if (vkQueueSubmit(context.queue, 1, &submitInfo, fence) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkQueueSubmit failed");
            vkDestroyFence(context.device, fence, nullptr);
            vkFreeCommandBuffers(context.device, context.commandPool, 1, &commandBuffer);
            return false;
        }

        if (vkWaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkWaitForFences failed");
            vkDestroyFence(context.device, fence, nullptr);
            vkFreeCommandBuffers(context.device, context.commandPool, 1, &commandBuffer);
            return false;
        }

        vkDestroyFence(context.device, fence, nullptr);
        vkFreeCommandBuffers(context.device, context.commandPool, 1, &commandBuffer);
        return true;
    }

    bool CreateBuffer(VulkanContext&        context,
                      VkDeviceSize          size,
                      VkBufferUsageFlags    usageFlags,
                      VkMemoryPropertyFlags memoryProperties,
                      Buffer&               outBuffer,
                      std::string*          errorMessage)
    {
        outBuffer      = {};
        outBuffer.size = size;

        VkBufferCreateInfo bufferCreateInfo{};
        bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size        = size;
        bufferCreateInfo.usage       = usageFlags;
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(context.device, &bufferCreateInfo, nullptr, &outBuffer.buffer) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkCreateBuffer failed");
            return false;
        }

        VkMemoryRequirements memoryRequirements{};
        vkGetBufferMemoryRequirements(context.device, outBuffer.buffer, &memoryRequirements);

        uint32_t memoryTypeIndex = 0;
        if (!FindMemoryType(context.memoryProperties, memoryRequirements.memoryTypeBits, memoryProperties, memoryTypeIndex))
        {
            SetError(errorMessage, "failed to find suitable memory type for buffer");
            DestroyBuffer(context, outBuffer);
            return false;
        }

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize  = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;
        if (vkAllocateMemory(context.device, &allocateInfo, nullptr, &outBuffer.memory) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkAllocateMemory for buffer failed");
            DestroyBuffer(context, outBuffer);
            return false;
        }

        if (vkBindBufferMemory(context.device, outBuffer.buffer, outBuffer.memory, 0) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkBindBufferMemory failed");
            DestroyBuffer(context, outBuffer);
            return false;
        }
        return true;
    }

    void DestroyBuffer(VulkanContext& context, Buffer& buffer)
    {
        if (buffer.buffer != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(context.device, buffer.buffer, nullptr);
            buffer.buffer = VK_NULL_HANDLE;
        }
        if (buffer.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(context.device, buffer.memory, nullptr);
            buffer.memory = VK_NULL_HANDLE;
        }
        buffer.size = 0;
    }

    bool CreateImage2D(
        VulkanContext& context, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usageFlags, Image& outImage, std::string* errorMessage)
    {
        outImage = {};

        outImage.createInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        outImage.createInfo.imageType     = VK_IMAGE_TYPE_2D;
        outImage.createInfo.format        = format;
        outImage.createInfo.extent.width  = width;
        outImage.createInfo.extent.height = height;
        outImage.createInfo.extent.depth  = 1;
        outImage.createInfo.mipLevels     = 1;
        outImage.createInfo.arrayLayers   = 1;
        outImage.createInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
        outImage.createInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
        outImage.createInfo.usage         = usageFlags;
        outImage.createInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        outImage.createInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (vkCreateImage(context.device, &outImage.createInfo, nullptr, &outImage.image) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkCreateImage failed");
            return false;
        }

        VkMemoryRequirements memoryRequirements{};
        vkGetImageMemoryRequirements(context.device, outImage.image, &memoryRequirements);

        uint32_t memoryTypeIndex = 0;
        if (!FindMemoryType(context.memoryProperties, memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memoryTypeIndex))
        {
            if (!FindMemoryType(context.memoryProperties, memoryRequirements.memoryTypeBits, 0, memoryTypeIndex))
            {
                SetError(errorMessage, "failed to find suitable memory type for image");
                DestroyImage(context, outImage);
                return false;
            }
        }

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize  = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;
        if (vkAllocateMemory(context.device, &allocateInfo, nullptr, &outImage.memory) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkAllocateMemory for image failed");
            DestroyImage(context, outImage);
            return false;
        }

        if (vkBindImageMemory(context.device, outImage.image, outImage.memory, 0) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkBindImageMemory failed");
            DestroyImage(context, outImage);
            return false;
        }

        VkImageViewCreateInfo imageViewCreateInfo{};
        imageViewCreateInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCreateInfo.image                           = outImage.image;
        imageViewCreateInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCreateInfo.format                          = format;
        imageViewCreateInfo.subresourceRange.aspectMask     = GetImageAspectMask(format);
        imageViewCreateInfo.subresourceRange.baseMipLevel   = 0;
        imageViewCreateInfo.subresourceRange.levelCount     = 1;
        imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        imageViewCreateInfo.subresourceRange.layerCount     = 1;
        if (vkCreateImageView(context.device, &imageViewCreateInfo, nullptr, &outImage.view) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkCreateImageView failed");
            DestroyImage(context, outImage);
            return false;
        }

        outImage.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        return true;
    }

    void DestroyImage(VulkanContext& context, Image& image)
    {
        if (image.view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(context.device, image.view, nullptr);
            image.view = VK_NULL_HANDLE;
        }
        if (image.image != VK_NULL_HANDLE)
        {
            vkDestroyImage(context.device, image.image, nullptr);
            image.image = VK_NULL_HANDLE;
        }
        if (image.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(context.device, image.memory, nullptr);
            image.memory = VK_NULL_HANDLE;
        }
        image.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    bool ClearImage(VulkanContext& context, Image& image, const std::array<float, 4>& clearColor, std::string* errorMessage)
    {
        const VkImageAspectFlags aspectMask = GetImageAspectMask(image.createInfo.format);
        if (!RunOneTimeCommands(
                context,
                [&](VkCommandBuffer commandBuffer) {
                    TransitionImage(commandBuffer, image.image, aspectMask, image.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

                    VkImageSubresourceRange range{};
                    range.aspectMask     = aspectMask;
                    range.baseMipLevel   = 0;
                    range.levelCount     = 1;
                    range.baseArrayLayer = 0;
                    range.layerCount     = 1;

                    if (aspectMask == VK_IMAGE_ASPECT_COLOR_BIT)
                    {
                        VkClearColorValue value{};
                        value.float32[0] = clearColor[0];
                        value.float32[1] = clearColor[1];
                        value.float32[2] = clearColor[2];
                        value.float32[3] = clearColor[3];
                        vkCmdClearColorImage(commandBuffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &value, 1, &range);
                    }
                    else
                    {
                        VkClearDepthStencilValue value{};
                        value.depth = clearColor[0];
                        vkCmdClearDepthStencilImage(commandBuffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &value, 1, &range);
                    }

                    TransitionImage(commandBuffer, image.image, aspectMask, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
                },
                errorMessage))
        {
            return false;
        }

        image.layout = VK_IMAGE_LAYOUT_GENERAL;
        return true;
    }

    bool RenderTriangle(VulkanContext& context, Image& image, std::string* errorMessage)
    {
        if ((image.createInfo.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0)
        {
            SetError(errorMessage, "RenderTriangle requires VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT");
            return false;
        }

        VkShaderModule        vertShaderModule = VK_NULL_HANDLE;
        VkShaderModule        fragShaderModule = VK_NULL_HANDLE;
        VkRenderPass          renderPass       = VK_NULL_HANDLE;
        VkFramebuffer         framebuffer      = VK_NULL_HANDLE;
        VkPipelineLayout      pipelineLayout   = VK_NULL_HANDLE;
        VkPipeline            pipeline         = VK_NULL_HANDLE;
        std::vector<uint32_t> vertexSpv;
        std::vector<uint32_t> fragmentSpv;

        if (!LoadTriangleShaders(vertexSpv, fragmentSpv, errorMessage))
        {
            return false;
        }

        VkShaderModuleCreateInfo shaderModuleCreateInfo{};
        shaderModuleCreateInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderModuleCreateInfo.codeSize = vertexSpv.size() * sizeof(uint32_t);
        shaderModuleCreateInfo.pCode    = vertexSpv.data();
        if (vkCreateShaderModule(context.device, &shaderModuleCreateInfo, nullptr, &vertShaderModule) != VK_SUCCESS)
        {
            SetError(errorMessage, "failed to create vertex shader module");
            return false;
        }

        shaderModuleCreateInfo.codeSize = fragmentSpv.size() * sizeof(uint32_t);
        shaderModuleCreateInfo.pCode    = fragmentSpv.data();
        if (vkCreateShaderModule(context.device, &shaderModuleCreateInfo, nullptr, &fragShaderModule) != VK_SUCCESS)
        {
            vkDestroyShaderModule(context.device, vertShaderModule, nullptr);
            SetError(errorMessage, "failed to create fragment shader module");
            return false;
        }

        VkAttachmentDescription colorAttachment{};
        colorAttachment.format         = image.createInfo.format;
        colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentReference{};
        colorAttachmentReference.attachment = 0;
        colorAttachmentReference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments    = &colorAttachmentReference;

        VkRenderPassCreateInfo renderPassCreateInfo{};
        renderPassCreateInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassCreateInfo.attachmentCount = 1;
        renderPassCreateInfo.pAttachments    = &colorAttachment;
        renderPassCreateInfo.subpassCount    = 1;
        renderPassCreateInfo.pSubpasses      = &subpass;
        if (vkCreateRenderPass(context.device, &renderPassCreateInfo, nullptr, &renderPass) != VK_SUCCESS)
        {
            vkDestroyShaderModule(context.device, fragShaderModule, nullptr);
            vkDestroyShaderModule(context.device, vertShaderModule, nullptr);
            SetError(errorMessage, "failed to create render pass");
            return false;
        }

        VkFramebufferCreateInfo framebufferCreateInfo{};
        framebufferCreateInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferCreateInfo.renderPass      = renderPass;
        framebufferCreateInfo.attachmentCount = 1;
        framebufferCreateInfo.pAttachments    = &image.view;
        framebufferCreateInfo.width           = image.createInfo.extent.width;
        framebufferCreateInfo.height          = image.createInfo.extent.height;
        framebufferCreateInfo.layers          = 1;
        if (vkCreateFramebuffer(context.device, &framebufferCreateInfo, nullptr, &framebuffer) != VK_SUCCESS)
        {
            vkDestroyRenderPass(context.device, renderPass, nullptr);
            vkDestroyShaderModule(context.device, fragShaderModule, nullptr);
            vkDestroyShaderModule(context.device, vertShaderModule, nullptr);
            SetError(errorMessage, "failed to create framebuffer");
            return false;
        }

        VkPipelineShaderStageCreateInfo shaderStages[2]{};
        shaderStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages[0].module = vertShaderModule;
        shaderStages[0].pName  = "main";
        shaderStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages[1].module = fragShaderModule;
        shaderStages[1].pName  = "main";

        VkPipelineVertexInputStateCreateInfo vertexInputState{};
        vertexInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState{};
        inputAssemblyState.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport viewport{};
        viewport.x        = 0.0f;
        viewport.y        = 0.0f;
        viewport.width    = static_cast<float>(image.createInfo.extent.width);
        viewport.height   = static_cast<float>(image.createInfo.extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = {image.createInfo.extent.width, image.createInfo.extent.height};

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports    = &viewport;
        viewportState.scissorCount  = 1;
        viewportState.pScissors     = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizationState{};
        rasterizationState.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizationState.polygonMode             = VK_POLYGON_MODE_FILL;
        rasterizationState.lineWidth               = 1.0f;
        rasterizationState.cullMode                = VK_CULL_MODE_NONE;
        rasterizationState.frontFace               = VK_FRONT_FACE_CLOCKWISE;
        rasterizationState.depthClampEnable        = VK_FALSE;
        rasterizationState.rasterizerDiscardEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampleState{};
        multisampleState.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable    = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlendState{};
        colorBlendState.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendState.attachmentCount = 1;
        colorBlendState.pAttachments    = &colorBlendAttachment;

        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
        pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        if (vkCreatePipelineLayout(context.device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
        {
            vkDestroyFramebuffer(context.device, framebuffer, nullptr);
            vkDestroyRenderPass(context.device, renderPass, nullptr);
            vkDestroyShaderModule(context.device, fragShaderModule, nullptr);
            vkDestroyShaderModule(context.device, vertShaderModule, nullptr);
            SetError(errorMessage, "failed to create pipeline layout");
            return false;
        }

        VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
        pipelineCreateInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.stageCount          = 2;
        pipelineCreateInfo.pStages             = shaderStages;
        pipelineCreateInfo.pVertexInputState   = &vertexInputState;
        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pViewportState      = &viewportState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pMultisampleState   = &multisampleState;
        pipelineCreateInfo.pColorBlendState    = &colorBlendState;
        pipelineCreateInfo.layout              = pipelineLayout;
        pipelineCreateInfo.renderPass          = renderPass;
        pipelineCreateInfo.subpass             = 0;

        if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline) != VK_SUCCESS)
        {
            vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
            vkDestroyFramebuffer(context.device, framebuffer, nullptr);
            vkDestroyRenderPass(context.device, renderPass, nullptr);
            vkDestroyShaderModule(context.device, fragShaderModule, nullptr);
            vkDestroyShaderModule(context.device, vertShaderModule, nullptr);
            SetError(errorMessage, "failed to create graphics pipeline");
            return false;
        }

        const VkImageAspectFlags aspectMask  = GetImageAspectMask(image.createInfo.format);
        const bool               drawSuccess = RunOneTimeCommands(
            context,
            [&](VkCommandBuffer commandBuffer) {
                TransitionImage(commandBuffer, image.image, aspectMask, image.layout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

                VkClearValue clearValue{};
                clearValue.color.float32[0] = 0.0f;
                clearValue.color.float32[1] = 0.0f;
                clearValue.color.float32[2] = 0.0f;
                clearValue.color.float32[3] = 1.0f;

                VkRenderPassBeginInfo renderPassBeginInfo{};
                renderPassBeginInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                renderPassBeginInfo.renderPass        = renderPass;
                renderPassBeginInfo.framebuffer       = framebuffer;
                renderPassBeginInfo.renderArea.offset = {0, 0};
                renderPassBeginInfo.renderArea.extent = {image.createInfo.extent.width, image.createInfo.extent.height};
                renderPassBeginInfo.clearValueCount   = 1;
                renderPassBeginInfo.pClearValues      = &clearValue;

                vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                vkCmdDraw(commandBuffer, 3, 1, 0, 0);
                vkCmdEndRenderPass(commandBuffer);

                TransitionImage(commandBuffer, image.image, aspectMask, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
            },
            errorMessage);

        vkDestroyPipeline(context.device, pipeline, nullptr);
        vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
        vkDestroyFramebuffer(context.device, framebuffer, nullptr);
        vkDestroyRenderPass(context.device, renderPass, nullptr);
        vkDestroyShaderModule(context.device, fragShaderModule, nullptr);
        vkDestroyShaderModule(context.device, vertShaderModule, nullptr);

        if (!drawSuccess)
        {
            return false;
        }

        image.layout = VK_IMAGE_LAYOUT_GENERAL;
        return true;
    }

    bool ReadbackRgba8Image(VulkanContext& context, Image& image, std::vector<uint8_t>& rgbaData, std::string* errorMessage)
    {
        const uint32_t width  = image.createInfo.extent.width;
        const uint32_t height = image.createInfo.extent.height;
        if (image.createInfo.format != VK_FORMAT_R8G8B8A8_UNORM && image.createInfo.format != VK_FORMAT_B10G11R11_UFLOAT_PACK32)
        {
            SetError(errorMessage, "ReadbackRgba8Image supports VK_FORMAT_R8G8B8A8_UNORM and VK_FORMAT_B10G11R11_UFLOAT_PACK32");
            return false;
        }

        Buffer             stagingBuffer{};
        const VkDeviceSize byteSize = static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * 4u;
        if (!CreateBuffer(context,
                          byteSize,
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          stagingBuffer,
                          errorMessage))
        {
            return false;
        }

        const VkImageLayout      originalLayout = image.layout;
        const VkImageAspectFlags aspectMask     = GetImageAspectMask(image.createInfo.format);
        const bool               copySuccess    = RunOneTimeCommands(
            context,
            [&](VkCommandBuffer commandBuffer) {
                TransitionImage(commandBuffer, image.image, aspectMask, originalLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

                VkBufferImageCopy copyRegion{};
                copyRegion.bufferOffset                    = 0;
                copyRegion.bufferRowLength                 = 0;
                copyRegion.bufferImageHeight               = 0;
                copyRegion.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
                copyRegion.imageSubresource.mipLevel       = 0;
                copyRegion.imageSubresource.baseArrayLayer = 0;
                copyRegion.imageSubresource.layerCount     = 1;
                copyRegion.imageOffset                     = {0, 0, 0};
                copyRegion.imageExtent                     = {width, height, 1};

                vkCmdCopyImageToBuffer(commandBuffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuffer.buffer, 1, &copyRegion);
                TransitionImage(commandBuffer, image.image, aspectMask, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, originalLayout);
            },
            errorMessage);

        if (!copySuccess)
        {
            DestroyBuffer(context, stagingBuffer);
            return false;
        }

        void* mappedData = nullptr;
        if (vkMapMemory(context.device, stagingBuffer.memory, 0, byteSize, 0, &mappedData) != VK_SUCCESS)
        {
            SetError(errorMessage, "vkMapMemory failed for readback buffer");
            DestroyBuffer(context, stagingBuffer);
            return false;
        }

        rgbaData.resize(static_cast<size_t>(byteSize));
        std::memcpy(rgbaData.data(), mappedData, static_cast<size_t>(byteSize));
        vkUnmapMemory(context.device, stagingBuffer.memory);

        DestroyBuffer(context, stagingBuffer);
        image.layout = originalLayout;
        return true;
    }

    ffxCreateBackendVKDesc BuildBackendDesc(const VulkanContext& context)
    {
        ffxCreateBackendVKDesc backendDesc{};
        backendDesc.header.type           = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
        backendDesc.header.pNext          = nullptr;
        backendDesc.vkDevice              = context.device;
        backendDesc.vkPhysicalDevice      = context.physicalDevice;
        backendDesc.vkInstance            = context.instance;
        backendDesc.vkDeviceProcAddr      = context.getDeviceProcAddr;
        backendDesc.vkGetInstanceProcAddr = context.getInstanceProcAddr;
        return backendDesc;
    }

    FfxApiResource MakeApiImageResource(const Image& image, const uint32_t additionalUsage, const uint32_t state)
    {
        FfxApiResource resource{};
        resource.resource    = reinterpret_cast<void*>(image.image);
        resource.description = ffxApiGetImageResourceDescriptionVK(image.image, image.createInfo, additionalUsage);
        resource.state       = state;
        return resource;
    }

    FfxResource MakeFfxImageResource(const Image& image, const FfxResourceUsage additionalUsage, const FfxResourceStates state, const char* name)
    {
        const FfxResourceDescription description = ffxGetImageResourceDescriptionVK(image.image, image.createInfo, additionalUsage);
        return ffxGetResourceVK(reinterpret_cast<void*>(image.image), description, name, state);
    }

    bool PixelHasColor(
        const std::vector<uint8_t>& rgbaData, const uint32_t width, const uint32_t x, const uint32_t y, const uint8_t minRed, const uint8_t maxBlue)
    {
        if (width == 0)
        {
            return false;
        }
        const size_t index = (static_cast<size_t>(y) * width + x) * 4;
        if (index + 3 >= rgbaData.size())
        {
            return false;
        }

        const uint8_t red   = rgbaData[index + 0];
        const uint8_t blue  = rgbaData[index + 2];
        const uint8_t alpha = rgbaData[index + 3];
        return red >= minRed && blue <= maxBlue && alpha > 0;
    }

    void SetIdentityMatrix(FfxApiFloat32x4x4 matrix)
    {
        for (int i = 0; i < 16; ++i)
        {
            matrix[i] = 0.0f;
        }
        matrix[0]  = 1.0f;
        matrix[5]  = 1.0f;
        matrix[10] = 1.0f;
        matrix[15] = 1.0f;
    }

}  // namespace unit_test::vk
