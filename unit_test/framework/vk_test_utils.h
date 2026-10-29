/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <ffx_api/vk/ffx_api_vk.h>

#include <array>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace unit_test::vk
{
    void SetShaderDirectory(const std::string& shaderDirectory);

    struct VulkanContext
    {
        VkInstance                       instance         = VK_NULL_HANDLE;
        VkPhysicalDevice                 physicalDevice   = VK_NULL_HANDLE;
        VkDevice                         device           = VK_NULL_HANDLE;
        VkQueue                          queue            = VK_NULL_HANDLE;
        uint32_t                         queueFamilyIndex = 0;
        VkCommandPool                    commandPool      = VK_NULL_HANDLE;
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        PFN_vkGetInstanceProcAddr        getInstanceProcAddr = nullptr;
        PFN_vkGetDeviceProcAddr          getDeviceProcAddr   = nullptr;
    };

    struct Buffer
    {
        VkBuffer       buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize   size   = 0;
    };

    struct Image
    {
        VkImage           image  = VK_NULL_HANDLE;
        VkImageView       view   = VK_NULL_HANDLE;
        VkDeviceMemory    memory = VK_NULL_HANDLE;
        VkImageLayout     layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageCreateInfo createInfo{};
    };

    struct VulkanPhysicalDeviceInfo
    {
        uint32_t    index    = 0;
        uint32_t    vendorId = 0;
        std::string name;
    };

    std::string ResolveVulkanDeviceSelector(const std::string& commandLineSelector, const char* environmentSelector);
    bool        SelectVulkanDeviceIndex(const std::vector<VulkanPhysicalDeviceInfo>& devices,
                                        const std::string&                           requestedSelector,
                                        uint32_t&                                    selectedIndex,
                                        std::string*                                 errorMessage);
    bool        EnumerateVulkanDevices(std::vector<VulkanPhysicalDeviceInfo>& devices, std::string* errorMessage);
    bool        InitializeVulkanContext(VulkanContext& context, std::string* errorMessage);
    void        DestroyVulkanContext(VulkanContext& context);

    bool BeginCommandBuffer(VulkanContext& context, VkCommandBuffer& commandBuffer, std::string* errorMessage);
    bool SubmitCommandBuffer(VulkanContext& context, VkCommandBuffer commandBuffer, std::string* errorMessage);

    bool CreateBuffer(VulkanContext&        context,
                      VkDeviceSize          size,
                      VkBufferUsageFlags    usageFlags,
                      VkMemoryPropertyFlags memoryProperties,
                      Buffer&               outBuffer,
                      std::string*          errorMessage);
    void DestroyBuffer(VulkanContext& context, Buffer& buffer);

    bool CreateImage2D(
        VulkanContext& context, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usageFlags, Image& outImage, std::string* errorMessage);
    void DestroyImage(VulkanContext& context, Image& image);

    bool ClearImage(VulkanContext& context, Image& image, const std::array<float, 4>& clearColor, std::string* errorMessage);
    bool RenderTriangle(VulkanContext& context, Image& image, std::string* errorMessage);
    bool ReadbackRgba8Image(VulkanContext& context, Image& image, std::vector<uint8_t>& rgbaData, std::string* errorMessage);

    ffxCreateBackendVKDesc BuildBackendDesc(const VulkanContext& context);
    FfxApiResource         MakeApiImageResource(const Image& image, uint32_t additionalUsage, uint32_t state);
    FfxResource            MakeFfxImageResource(const Image& image, FfxResourceUsage additionalUsage, FfxResourceStates state, const char* name);

    bool PixelHasColor(const std::vector<uint8_t>& rgbaData, uint32_t width, uint32_t x, uint32_t y, uint8_t minRed, uint8_t maxBlue);
    void SetIdentityMatrix(FfxApiFloat32x4x4 matrix);

}  // namespace unit_test::vk
