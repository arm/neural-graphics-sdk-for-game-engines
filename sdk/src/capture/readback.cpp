/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "readback.h"

#ifdef FFX_BACKEND_VK
#include "format.h"

#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <FidelityFX/host/backends/vk/vk_wrapper.h>
#include <limits>
#endif

namespace arm::capture::vk
{
#ifdef FFX_BACKEND_VK
    namespace
    {
        bool IsSupportedImageState(FfxResourceStates state)
        {
            switch (state)
            {
            case FFX_RESOURCE_STATE_COMMON:
            case FFX_RESOURCE_STATE_GENERIC_READ:
            case FFX_RESOURCE_STATE_COMPUTE_UAV:
            case FFX_RESOURCE_STATE_PIXEL_UAV:
            case FFX_RESOURCE_STATE_GENERIC_UAV:
            case FFX_RESOURCE_STATE_COMPUTE_READ:
            case FFX_RESOURCE_STATE_PIXEL_READ:
            case FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ:
            case FFX_RESOURCE_STATE_COPY_SRC:
            case FFX_RESOURCE_STATE_PRESENT:
            case FFX_RESOURCE_STATE_RENDER_TARGET:
            case FFX_RESOURCE_STATE_DATA_GRAPH_READ:
            case FFX_RESOURCE_STATE_DATA_GRAPH_WRITE:
            case FFX_RESOURCE_STATE_DATA_GRAPH_TEXTURE_READ:
            case FFX_RESOURCE_STATE_DATA_GRAPH_TEXTURE_WRITE:
            case FFX_RESOURCE_STATE_DEPTH_ATTACHEMENT:
                return true;
            default:
                return false;
            }
        }

        bool IsSupportedBufferState(FfxResourceStates state)
        {
            switch (state)
            {
            case FFX_RESOURCE_STATE_COMMON:
            case FFX_RESOURCE_STATE_GENERIC_READ:
            case FFX_RESOURCE_STATE_COMPUTE_UAV:
            case FFX_RESOURCE_STATE_PIXEL_UAV:
            case FFX_RESOURCE_STATE_GENERIC_UAV:
            case FFX_RESOURCE_STATE_COMPUTE_READ:
            case FFX_RESOURCE_STATE_PIXEL_READ:
            case FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ:
            case FFX_RESOURCE_STATE_COPY_SRC:
            case FFX_RESOURCE_STATE_COPY_DEST:
            case FFX_RESOURCE_STATE_INDIRECT_ARGUMENT:
            case FFX_RESOURCE_STATE_DATA_GRAPH_READ:
            case FFX_RESOURCE_STATE_DATA_GRAPH_WRITE:
                return true;
            default:
                return false;
            }
        }
    }  // namespace

    bool GetDeviceInfo(void* backendDevice, DeviceInfo& info)
    {
        const VkDeviceContext* context = reinterpret_cast<const VkDeviceContext*>(backendDevice);
        if (context == nullptr || context->vkDevice == VK_NULL_HANDLE || context->vkPhysicalDevice == VK_NULL_HANDLE)
            return false;

        info.device         = context->vkDevice;
        info.physicalDevice = context->vkPhysicalDevice;
        return true;
    }

    PFN_vkVoidFunction GetDeviceProc(VkDevice device, const char* name)
    {
        if (device == VK_NULL_HANDLE || name == nullptr || VulkanWrapper().vkGetDeviceProcAddr == nullptr)
            return nullptr;
        return VulkanWrapper().vkGetDeviceProcAddr(device, name);
    }

    bool BuildImageReadbackDesc(const FfxResource& source, ImageReadbackDesc& desc)
    {
        if (!IsSupportedImageState(source.state))
            return false;
        if (source.description.type != FFX_RESOURCE_TYPE_TEXTURE2D || source.description.width == 0 || source.description.height == 0 ||
            source.description.depth > 1 || source.description.mipCount > 1)
            return false;

        uint64_t byteSize = 0;
        if (source.description.type == FFX_RESOURCE_TYPE_BUFFER || !AddResourceBytes(source, byteSize) || byteSize > std::numeric_limits<VkDeviceSize>::max())
        {
            return false;
        }

        // The resource type and single-2D-subresource constraints above are
        // the Vulkan image-handle tag for this backend conversion.
        desc.image         = reinterpret_cast<VkImage>(source.resource);
        desc.currentLayout = getVKImageLayoutFromResourceState(source.state);
        desc.aspect        = (source.description.usage & FFX_RESOURCE_USAGE_DEPTHTARGET) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        desc.extent        = {source.description.width, source.description.height};
        // A tensor's format describes one element; channel is a separate tensor
        // dimension and must be included in the readback byte count.
        desc.byteSize = static_cast<VkDeviceSize>(byteSize);
        desc.format   = static_cast<uint32_t>(source.description.format);
        return true;
    }

    void RecordImageReadback(VkCommandBuffer commandBuffer, const ImageReadbackDesc& source, VkBuffer destination, VkDeviceSize destinationOffset)
    {
        VkImageMemoryBarrier2 before{};
        before.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        before.srcStageMask        = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        before.srcAccessMask       = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
        before.dstStageMask        = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        before.dstAccessMask       = VK_ACCESS_2_TRANSFER_READ_BIT;
        before.oldLayout           = source.currentLayout;
        before.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before.image               = source.image;
        before.subresourceRange    = {source.aspect, 0, 1, 0, 1};

        VkDependencyInfo dependency{};
        dependency.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers    = &before;
        VulkanWrapper().vkCmdPipelineBarrier2(commandBuffer, &dependency);

        VkBufferImageCopy region{};
        region.bufferOffset     = destinationOffset;
        region.imageSubresource = {source.aspect, 0, 0, 1};
        region.imageExtent      = {source.extent.width, source.extent.height, 1};
        VulkanWrapper().vkCmdCopyImageToBuffer(commandBuffer, source.image, VK_IMAGE_LAYOUT_GENERAL, destination, 1, &region);

        VkBufferMemoryBarrier2 hostRead{};
        hostRead.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        hostRead.srcStageMask        = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        hostRead.srcAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        hostRead.dstStageMask        = VK_PIPELINE_STAGE_2_HOST_BIT;
        hostRead.dstAccessMask       = VK_ACCESS_2_HOST_READ_BIT;
        hostRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostRead.buffer              = destination;
        hostRead.size                = VK_WHOLE_SIZE;

        VkImageMemoryBarrier2 after = before;
        after.srcStageMask          = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        after.srcAccessMask         = VK_ACCESS_2_TRANSFER_READ_BIT;
        after.dstStageMask          = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        after.dstAccessMask         = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        after.oldLayout             = VK_IMAGE_LAYOUT_GENERAL;
        after.newLayout             = source.currentLayout;

        dependency.bufferMemoryBarrierCount = 1;
        dependency.pBufferMemoryBarriers    = &hostRead;
        dependency.pImageMemoryBarriers     = &after;
        VulkanWrapper().vkCmdPipelineBarrier2(commandBuffer, &dependency);
    }

    bool BuildBufferReadbackDesc(const FfxResource& source, BufferReadbackDesc& desc)
    {
        if (!IsSupportedBufferState(source.state) || source.resource == nullptr || source.description.type != FFX_RESOURCE_TYPE_BUFFER ||
            source.description.size == 0 || (source.description.size % 4) != 0)
        {
            return false;
        }

        desc.buffer       = reinterpret_cast<VkBuffer>(source.resource);
        desc.byteSize     = static_cast<VkDeviceSize>(source.description.size);
        desc.currentState = source.state;
        desc.format       = static_cast<uint32_t>(source.description.format);
        return true;
    }

    VkBufferMemoryBarrier2 MakeBufferReadbackRestoreBarrier(VkBuffer source)
    {
        VkBufferMemoryBarrier2 restore{};
        restore.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        restore.srcStageMask        = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        restore.srcAccessMask       = VK_ACCESS_2_TRANSFER_READ_BIT;
        restore.dstStageMask        = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        restore.dstAccessMask       = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        restore.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore.buffer              = source;
        restore.size                = VK_WHOLE_SIZE;
        return restore;
    }

    void RecordBufferReadback(VkCommandBuffer commandBuffer, const BufferReadbackDesc& source, VkBuffer destination, VkDeviceSize destinationOffset)
    {
        VkBufferMemoryBarrier2 before{};
        before.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        before.srcStageMask        = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        before.srcAccessMask       = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
        before.dstStageMask        = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        before.dstAccessMask       = VK_ACCESS_2_TRANSFER_READ_BIT;
        before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before.buffer              = source.buffer;
        before.size                = VK_WHOLE_SIZE;

        VkDependencyInfo dependency{};
        dependency.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.bufferMemoryBarrierCount = 1;
        dependency.pBufferMemoryBarriers    = &before;
        VulkanWrapper().vkCmdPipelineBarrier2(commandBuffer, &dependency);

        VkBufferCopy region{};
        region.srcOffset = 0;
        region.dstOffset = destinationOffset;
        region.size      = source.byteSize;
        VulkanWrapper().vkCmdCopyBuffer(commandBuffer, source.buffer, destination, 1, &region);

        VkBufferMemoryBarrier2 hostRead{};
        hostRead.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        hostRead.srcStageMask        = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        hostRead.srcAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        hostRead.dstStageMask        = VK_PIPELINE_STAGE_2_HOST_BIT;
        hostRead.dstAccessMask       = VK_ACCESS_2_HOST_READ_BIT;
        hostRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostRead.buffer              = destination;
        hostRead.size                = VK_WHOLE_SIZE;

        const VkBufferMemoryBarrier2 restore = MakeBufferReadbackRestoreBarrier(source.buffer);

        const VkBufferMemoryBarrier2 barriers[] = {hostRead, restore};
        dependency.bufferMemoryBarrierCount     = 2;
        dependency.pBufferMemoryBarriers        = barriers;
        VulkanWrapper().vkCmdPipelineBarrier2(commandBuffer, &dependency);
    }
#endif

}  // namespace arm::capture::vk
