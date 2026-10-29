/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <FidelityFX/host/ffx_interface.h>
#include <FidelityFX/host/ffx_types.h>

#ifdef FFX_BACKEND_VK
#include <vulkan/vulkan.h>

// Defined in sdk/src/backends/vk/ffx_vk.cpp. This is the backend's authoritative
// mapping from FfxResourceStates to Vulkan synchronization state and image layout.
VkAccessFlags2        getVKAccessFlagsFromResourceState(FfxResourceStates state);
VkPipelineStageFlags2 getVKPipelineStageFlagsFromResourceState(FfxResourceStates state);
VkImageLayout         getVKImageLayoutFromResourceState(FfxResourceStates state);
#endif

namespace arm::capture::vk
{
#ifdef FFX_BACKEND_VK
    struct DeviceInfo
    {
        VkDevice         device         = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    };

    struct ImageReadbackDesc
    {
        VkImage            image         = VK_NULL_HANDLE;
        VkImageLayout      currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageAspectFlags aspect        = VK_IMAGE_ASPECT_COLOR_BIT;
        VkExtent2D         extent{};
        VkDeviceSize       byteSize = 0;
        uint32_t           format   = 0;
    };

    struct BufferReadbackDesc
    {
        VkBuffer          buffer       = VK_NULL_HANDLE;
        VkDeviceSize      byteSize     = 0;
        FfxResourceStates currentState = FFX_RESOURCE_STATE_COMMON;
        uint32_t          format       = 0;
    };

    bool               GetDeviceInfo(void* backendDevice, DeviceInfo& info);
    PFN_vkVoidFunction GetDeviceProc(VkDevice device, const char* name);
    bool               BuildImageReadbackDesc(const FfxResource& source, ImageReadbackDesc& desc);
    void RecordImageReadback(VkCommandBuffer commandBuffer, const ImageReadbackDesc& source, VkBuffer destination, VkDeviceSize destinationOffset);
    bool BuildBufferReadbackDesc(const FfxResource& source, BufferReadbackDesc& desc);
    VkBufferMemoryBarrier2 MakeBufferReadbackRestoreBarrier(VkBuffer source);
    void RecordBufferReadback(VkCommandBuffer commandBuffer, const BufferReadbackDesc& source, VkBuffer destination, VkDeviceSize destinationOffset);
#endif

}  // namespace arm::capture::vk
