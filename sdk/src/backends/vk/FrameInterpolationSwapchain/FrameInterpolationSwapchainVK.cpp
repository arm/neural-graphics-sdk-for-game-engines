// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "FrameInterpolationSwapchainVK.h"
#include "FrameInterpolationSwapchainVK_DebugPacing.h"
#include "FrameInterpolationSwapchainVK_UiComposition.h"

#include <FidelityFX/host/ffx_assert.h>
#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <FidelityFX/host/backends/vk/vk_wrapper.h>
#define ENABLE_EXR_EXPORT
#ifdef ENABLE_EXR_EXPORT
#ifndef DISABLE_SDK_TINYEXR_IMPLEMENTATION
#define TINYEXR_IMPLEMENTATION
#endif
#include <tinyexr/tinyexr.h>
#endif
#include <vector>
#include <mutex>

struct AsyncDumpOperation
{
    VkBuffer               stagingBuffer;
    VkDeviceMemory         stagingBufferMemory;
    VkDeviceSize           imageSize;
    std::string            filename;
    FfxResourceDescription resourceDesc;
    uint32_t               width;
    uint32_t               height;
    uint64_t               framesSentForPresentation;
    VkFence                completionFence;
};

std::vector<AsyncDumpOperation> g_pendingDumpOps;
std::mutex                      g_dumpOpsMutex;

static void dumpResourceAsyncNoCmdBuffer(
    VkDevice device, VkPhysicalDevice physicalDevice, VkQueue queue, const FfxResource& resource, const char* baseFilename, uint64_t frameID);
static void processAsyncDumpOperations(VkDevice device, uint64_t currentFrameCount);
///////////////////////////////////////////////////////////////////
//                    MODES EXPLAINED
///////////////////////////////////////////////////////////////////
//
// 1. FGSwapchainCompositionMode::eComposeOnPresentQueue
//    - closest mode to DX12
//    - the present queue needs to have graphics and compute capabilities
//    - in FrameInterpolationSwapChainVK::queuePresent:
//      - game queue signals the game semaphore
//      - dispatches the interpolation on the interpolation queue (can be the game queue). This waits for the game semaphore. Signals the interpolation semaphore
//    - interpolation thread:
//      - computes the execution time of the interpolation by waiting on the interpolation semaphore
//      - computes when the second present should be called
//    - present thread (composeAndPresent_presenterThread):
//      - acquires a new backbuffer image.
//      - Present queue waits for the interpolation and image available semaphores then executes the UI composition of the interpolated image. Signals the composition and frame rendered semaphores.
//      - presents the interpolated image (waiting on the frame rendered semaphore).
//      - acquires a new backbuffer image.
//      - Present queue waits for the image available semaphore then executes the UI composition of the real image. Signals the composition, frame rendered and the present semaphores.
//      - thread waits for a given time
//      - presents the real image (waiting on the frame rendered semaphore).
//
//
// 2. FGSwapchainCompositionMode::eComposeOnGameQueue
//    - legacy vulkan mode
//    - the present queue needs to have transfer capability
//    - in FrameInterpolationSwapChainVK::queuePresent:
//      - game queue signals the game semaphore
//      - dispatches the interpolation on the interpolation queue (can be the game queue). This waits for the game semaphore. Signals the interpolation semaphore.
//      - game queue waits for the interpolation semaphore then executes the composition of the interpolated image into an intermediate texture. Copies this texture back into the interpolation buffer. Signals the composition semaphore.
//      - game queue executes the composition of the real image on the game queue into an intermediate texture. Copies this texture back into the replacement buffer. Signals the composition semaphore.
//    - interpolation thread:
//      - computes the execution time of the interpolation by waiting on the interpolation semaphore
//      - computes when the second present should be called
//    - present thread (copyAndPresent_presenterThread):
//      - acquires a new backbuffer image.
//      - present queue waits for composition and image available semaphores then copies the interpolated image (composed with UI) into the backbuffer image. Signals frame rendered and present semaphores.
//      - presents the interpolated image (waiting on the frame rendered semaphore).
//      - acquires a new backbuffer image.
//      - present queue waits for composition and image available semaphores then copies the replacement image (composed with UI) into the backbuffer image. Signals frame rendered and present semaphores.
//      - thread waits for a given time
//      - presents the real image (waiting on the frame rendered semaphore).
//

//////////////////////////////////////////////
/// Helper functions and classes
//////////////////////////////////////////////

#define EXIT_ON_VKRESULT_NOT_SUCCESS(res) \
    if (res != VK_SUCCESS)                \
        return res;

#define FFX_ASSERT_MESSAGE_FORMAT(test, ...)            \
    {                                                   \
        char msg[128];                                  \
        snprintf(msg, 128 * sizeof(char), __VA_ARGS__); \
        FFX_ASSERT_MESSAGE(test, msg);                  \
    }

#define GET_DEVICE_PROC_ADDR(name) name##Proc = (PFN_##name)VulkanWrapper().vkGetDeviceProcAddr(device, #name);

#define HAS_FLAG(options, flags) (((options) & (flags)) == (flags))

// Some known states during queue family ownership transfer
struct ImageState
{
    VkAccessFlags accessMask;
    VkImageLayout layout;
};
constexpr ImageState ReplacementBufferTransferState   = {VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
constexpr ImageState InterpolationBufferTransferState = {VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL};

VkImageMemoryBarrier getImageMemoryBarrier(VkImage       image,
                                           VkAccessFlags srcAccessMask,
                                           VkAccessFlags dstAccessMask,
                                           VkImageLayout oldLayout,
                                           VkImageLayout newLayout,
                                           uint32_t      srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                           uint32_t      dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                           uint32_t      levelCount          = VK_REMAINING_MIP_LEVELS,
                                           uint32_t      layerCount          = VK_REMAINING_ARRAY_LAYERS)
{
    VkImageMemoryBarrier barrier            = {};
    barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.pNext                           = nullptr;
    barrier.srcAccessMask                   = srcAccessMask;
    barrier.dstAccessMask                   = dstAccessMask;
    barrier.oldLayout                       = oldLayout;
    barrier.newLayout                       = newLayout;
    barrier.srcQueueFamilyIndex             = srcQueueFamilyIndex;
    barrier.dstQueueFamilyIndex             = dstQueueFamilyIndex;
    barrier.image                           = image;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = levelCount;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = layerCount;

    return barrier;
}

struct ImageBarrierHelper
{
    static const uint32_t Capacity           = 3;
    VkImageMemoryBarrier  barriers[Capacity] = {};
    uint32_t              count              = 0;

    void add(VkImageMemoryBarrier barrier)
    {
        FFX_ASSERT_MESSAGE(count < Capacity, "ImageBarrierHelper capacity exceeded. Please increase it.");
        barriers[count] = barrier;
        ++count;
    }

    template <typename... T>
    void add(T... args)
    {
        add(getImageMemoryBarrier(args...));
    }

    void record(VkCommandBuffer      commandBuffer,
                VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
    {
        if (count > 0)
        {
            VulkanWrapper().vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, 0, 0, nullptr, 0, nullptr, count, barriers);
        }
    }
};

inline void flipBarrier(VkImageMemoryBarrier& barrier)
{
    VkAccessFlags dstAccessMask = barrier.dstAccessMask;
    barrier.dstAccessMask       = barrier.srcAccessMask;
    barrier.srcAccessMask       = dstAccessMask;

    VkImageLayout newLayout = barrier.newLayout;
    barrier.newLayout       = barrier.oldLayout;
    barrier.oldLayout       = newLayout;
}

void recordCopy(VkCommandBuffer commandBuffer, VkImage srcImage, VkImage dstImage, uint32_t width, uint32_t height, uint32_t depth = 1)
{
    VkImageCopy imageCopy                   = {};
    imageCopy.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    imageCopy.srcSubresource.mipLevel       = 0;
    imageCopy.srcSubresource.baseArrayLayer = 0;
    imageCopy.srcSubresource.layerCount     = 1;
    imageCopy.srcOffset.x                   = 0;
    imageCopy.srcOffset.y                   = 0;
    imageCopy.srcOffset.z                   = 0;
    imageCopy.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    imageCopy.dstSubresource.mipLevel       = 0;
    imageCopy.dstSubresource.baseArrayLayer = 0;
    imageCopy.dstSubresource.layerCount     = 1;
    imageCopy.dstOffset.x                   = 0;
    imageCopy.dstOffset.y                   = 0;
    imageCopy.dstOffset.z                   = 0;
    imageCopy.extent.width                  = width;
    imageCopy.extent.height                 = height;
    imageCopy.extent.depth                  = depth;
    VulkanWrapper().vkCmdCopyImage(
        commandBuffer, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopy);
}

/// Helper class to set the debug name
struct DebugNameSetter
{
    VkDevice                         device;
    PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXTProc;

    DebugNameSetter(VkDevice dev)
    {
        device = dev;
        GET_DEVICE_PROC_ADDR(vkSetDebugUtilsObjectNameEXT);
    }

    VkResult setDebugName(void* pObject, VkObjectType type, const char* name)
    {
        if (pObject != nullptr && vkSetDebugUtilsObjectNameEXTProc != nullptr && name != nullptr)
        {
            VkDebugUtilsObjectNameInfoEXT nameInfo = {};
            nameInfo.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
            nameInfo.pNext                         = nullptr;
            nameInfo.objectType                    = type;
            nameInfo.objectHandle                  = (uint64_t)pObject;
            nameInfo.pObjectName                   = name;
            return vkSetDebugUtilsObjectNameEXTProc(device, &nameInfo);
        }
        return VK_SUCCESS;
    }

    VkResult setDebugName(void* pObject, VkObjectType type, const char* name, uint32_t i)
    {
        constexpr size_t cBufferSize = 64;
        char             finalName[cBufferSize];
        snprintf(finalName, cBufferSize, name, i);
        return setDebugName(pObject, type, finalName);
    }
};

VkAccessFlags getVKAccessFlagsFromResourceState2(FfxResourceStates state)
{
    switch (state)
    {
    case FFX_RESOURCE_STATE_COMMON:
        return VK_ACCESS_NONE;
    case FFX_RESOURCE_STATE_GENERIC_READ:
        return VK_ACCESS_SHADER_READ_BIT;
    case FFX_RESOURCE_STATE_GENERIC_UAV:
        return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    case FFX_RESOURCE_STATE_COMPUTE_READ:
    case FFX_RESOURCE_STATE_PIXEL_READ:
    case FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ:
        return VK_ACCESS_SHADER_READ_BIT;
    case FFX_RESOURCE_STATE_COPY_SRC:
        return VK_ACCESS_TRANSFER_READ_BIT;
    case FFX_RESOURCE_STATE_COPY_DEST:
        return VK_ACCESS_TRANSFER_WRITE_BIT;
    case FFX_RESOURCE_STATE_INDIRECT_ARGUMENT:
        return VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    case FFX_RESOURCE_STATE_PRESENT:
        return VK_ACCESS_NONE;
    case FFX_RESOURCE_STATE_RENDER_TARGET:
        return VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    case FFX_RESOURCE_STATE_DEPTH_ATTACHEMENT:
        return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    default:
        FFX_ASSERT_MESSAGE(false, "State flag not yet supported");
        return VK_ACCESS_SHADER_READ_BIT;
    }
}

VkImageLayout getVKImageLayoutFromResourceState2(FfxResourceStates state)
{
    switch (state)
    {
    case FFX_RESOURCE_STATE_COMMON:
        return VK_IMAGE_LAYOUT_GENERAL;
    case FFX_RESOURCE_STATE_GENERIC_READ:
        return VK_IMAGE_LAYOUT_GENERAL;
    case FFX_RESOURCE_STATE_GENERIC_UAV:
        return VK_IMAGE_LAYOUT_GENERAL;
    case FFX_RESOURCE_STATE_COMPUTE_READ:
    case FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ:
    case FFX_RESOURCE_STATE_PIXEL_READ:
        return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    case FFX_RESOURCE_STATE_COPY_SRC:
        return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    case FFX_RESOURCE_STATE_COPY_DEST:
        return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    case FFX_RESOURCE_STATE_PRESENT:
        return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    case FFX_RESOURCE_STATE_RENDER_TARGET:
        return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    case FFX_RESOURCE_STATE_DEPTH_ATTACHEMENT:
        return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    case FFX_RESOURCE_STATE_INDIRECT_ARGUMENT:
        // this case is for buffers
    default:
        FFX_ASSERT_MESSAGE(false, "Image layout flag not yet supported");
        return VK_IMAGE_LAYOUT_GENERAL;
    }
}

VkPipelineStageFlags getVKPipelineStageFlagsFromResourceState2(FfxResourceStates state)
{
    switch (state)
    {
    case (FFX_RESOURCE_STATE_COMMON):
    case (FFX_RESOURCE_STATE_GENERIC_READ):
    case (FFX_RESOURCE_STATE_GENERIC_UAV):
    case (FFX_RESOURCE_STATE_COMPUTE_READ):
        return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    case (FFX_RESOURCE_STATE_PIXEL_READ):
        return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    case (FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ):
        return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    case (FFX_RESOURCE_STATE_INDIRECT_ARGUMENT):
        return VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
    case FFX_RESOURCE_STATE_COPY_SRC:
    case FFX_RESOURCE_STATE_COPY_DEST:
        return VK_PIPELINE_STAGE_TRANSFER_BIT;
    case FFX_RESOURCE_STATE_PRESENT:
        return VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    case FFX_RESOURCE_STATE_RENDER_TARGET:
        return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    default:
        FFX_ASSERT_MESSAGE(false, "Pipeline stage flag not yet supported");
        return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
}

// Put the wait semaphores from the VkPresentInfo into the SubmissionSemaphores
void addPresentInfoSemaphores(const VkPresentInfoKHR* pPresentInfo, SubmissionSemaphores& toWait)
{
    for (uint32_t i = 0; i < pPresentInfo->waitSemaphoreCount; ++i)
    {
        toWait.add(pPresentInfo->pWaitSemaphores[i]);  // those aren't timeline semaphores
    }
}

uint32_t findMemoryType(const VkPhysicalDeviceMemoryProperties& memProperties, uint32_t memoryTypeBits, VkMemoryPropertyFlags properties)
{
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((memoryTypeBits & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }
    return 0u;
};

bool waitForSemaphoreValue(
    VkDevice device, VkSemaphore semaphore, uint64_t value, uint64_t nanoseconds = UINT64_MAX, FfxWaitCallbackFunc waitCallback = nullptr)
{
    if (semaphore != VK_NULL_HANDLE)
    {
        VkSemaphoreWaitInfo waitInfo = {};
        waitInfo.sType               = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        waitInfo.pNext               = nullptr;
        waitInfo.flags               = 0;
        waitInfo.semaphoreCount      = 1;
        waitInfo.pSemaphores         = &semaphore;
        waitInfo.pValues             = &value;
        VkResult res                 = VK_TIMEOUT;
        if (nanoseconds == UINT64_MAX)
        {
            if (waitCallback)
            {
                uint64_t waitIntervalInNanoSeconds = 1000000;  //1ms
                res                                = VulkanWrapper().vkWaitSemaphores(device, &waitInfo, waitIntervalInNanoSeconds);
                while (res == VK_TIMEOUT)
                {
                    res = VulkanWrapper().vkWaitSemaphores(device, &waitInfo, waitIntervalInNanoSeconds);
                    waitCallback("FenceName", value);
                }
            }
            else
            {
                res = VulkanWrapper().vkWaitSemaphores(device, &waitInfo, nanoseconds);
            }
        }
        else
        {
            res = VulkanWrapper().vkWaitSemaphores(device, &waitInfo, nanoseconds);
        }

        return (res == VK_SUCCESS);
    }

    return false;
}

#ifdef FFX_PLATFORM_WINDOWS
inline void SafeCloseHandle(HANDLE& handle)
{
    if (handle)
    {
        CloseHandle(handle);
        handle = NULL;
    }
}
#endif

FrameInterpolationSwapChainVK* createFrameInterpolationSwapChain(const VkAllocationCallbacks* pAllocator)
{
    FrameInterpolationSwapChainVK* pSwapChainVK = nullptr;
    if (pAllocator != nullptr && pAllocator->pfnAllocation != nullptr)
    {
        void* pData  = pAllocator->pfnAllocation(pAllocator->pUserData,
                                                sizeof(FrameInterpolationSwapChainVK),
                                                std::alignment_of<FrameInterpolationSwapChainVK>::value,
                                                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
        pSwapChainVK = new (pData) FrameInterpolationSwapChainVK();
    }
    else
    {
        pSwapChainVK = new FrameInterpolationSwapChainVK();
        if (pAllocator != nullptr && pAllocator->pfnInternalAllocation != nullptr)
        {
            pAllocator->pfnInternalAllocation(
                pAllocator->pUserData, sizeof(FrameInterpolationSwapChainVK), VK_INTERNAL_ALLOCATION_TYPE_EXECUTABLE, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
        }
    }
    return pSwapChainVK;
}

void deleteFrameInterpolationSwapChain(FrameInterpolationSwapChainVK* pSwapChainVK, const VkAllocationCallbacks* pAllocator)
{
    if (pSwapChainVK != nullptr)
    {
        if (pAllocator != nullptr && pAllocator->pfnFree != nullptr)
        {
            pSwapChainVK->~FrameInterpolationSwapChainVK();
            pAllocator->pfnFree(pAllocator->pUserData, pSwapChainVK);
        }
        else
        {
            delete pSwapChainVK;
            if (pAllocator != nullptr && pAllocator->pfnInternalFree != nullptr)
            {
                pAllocator->pfnInternalFree(
                    pAllocator->pUserData, sizeof(FrameInterpolationSwapChainVK), VK_INTERNAL_ALLOCATION_TYPE_EXECUTABLE, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
            }
        }
    }
}

VkResult FrameInterpolationSwapChainVK::createImage(ReplacementResource&                    resource,
                                                    VkImageCreateInfo&                      info,
                                                    FfxSurfaceFormat                        format,
                                                    const char*                             name,
                                                    const VkPhysicalDeviceMemoryProperties& memProperties,
                                                    const VkAllocationCallbacks*            pAllocator)
{
    VkResult res = VulkanWrapper().vkCreateImage(presentInfo.device, &info, pAllocator, &resource.image);
    EXIT_ON_VKRESULT_NOT_SUCCESS(res);

    if (res == VK_SUCCESS)
    {
        DebugNameSetter debugNameSetter(presentInfo.device);
        debugNameSetter.setDebugName(resource.image, VK_OBJECT_TYPE_IMAGE, name);  // it's fine if this fails

        resource.description.type     = FFX_RESOURCE_TYPE_TEXTURE2D;
        resource.description.format   = format;
        resource.description.width    = info.extent.width;
        resource.description.height   = info.extent.height;
        resource.description.depth    = info.extent.depth;
        resource.description.mipCount = info.mipLevels;
        resource.description.flags    = FFX_RESOURCE_FLAGS_NONE;
        resource.description.usage    = static_cast<FfxResourceUsage>(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV);

        VkMemoryRequirements memRequirements;
        VulkanWrapper().vkGetImageMemoryRequirements(presentInfo.device, resource.image, &memRequirements);

        VkMemoryAllocateInfo allocateInfo = {};
        allocateInfo.sType                = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.pNext                = nullptr;
        allocateInfo.allocationSize       = memRequirements.size;
        allocateInfo.memoryTypeIndex      = findMemoryType(memProperties, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        res                               = VulkanWrapper().vkAllocateMemory(presentInfo.device, &allocateInfo, pAllocator, &resource.memory);

        if (res == VK_SUCCESS)
        {
            resource.allocationSize = allocateInfo.allocationSize;
            totalUsageInBytes += resource.allocationSize;
        }
    }

    if (res == VK_SUCCESS)
        res = VulkanWrapper().vkBindImageMemory(presentInfo.device, resource.image, resource.memory, 0);

    if (res != VK_SUCCESS)
        destroyImage(resource, pAllocator);

    return res;
}

VkResult FrameInterpolationSwapChainVK::createImage(ReplacementResource&                    resource,
                                                    VkImageCreateInfo&                      info,
                                                    FfxSurfaceFormat                        format,
                                                    const char*                             name,
                                                    uint32_t                                index,
                                                    const VkPhysicalDeviceMemoryProperties& memProperties,
                                                    const VkAllocationCallbacks*            pAllocator)
{
    constexpr size_t cBufferSize = 64;
    char             finalName[cBufferSize];
    snprintf(finalName, cBufferSize, name, index);
    return createImage(resource, info, format, finalName, memProperties, pAllocator);
}

void FrameInterpolationSwapChainVK::destroyImage(ReplacementResource& resource, const VkAllocationCallbacks* pAllocator)
{
    if (resource.image != VK_NULL_HANDLE)
    {
        VulkanWrapper().vkDestroyImage(presentInfo.device, resource.image, pAllocator);
        resource.image = VK_NULL_HANDLE;
    }
    if (resource.memory != VK_NULL_HANDLE)
    {
        VulkanWrapper().vkFreeMemory(presentInfo.device, resource.memory, pAllocator);
        resource.memory = VK_NULL_HANDLE;
        totalUsageInBytes -= resource.allocationSize;
        resource.allocationSize = 0;
    }
}

//////////////////////////////////////////////
/// Vulkan API overridden functions
//////////////////////////////////////////////

VkResult vkAcquireNextImageFFX(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore, VkFence fence, uint32_t* pImageIndex)
{
    if (swapchain != VK_NULL_HANDLE)
    {
        FrameInterpolationSwapChainVK* pSwapChainVK = reinterpret_cast<FrameInterpolationSwapChainVK*>(swapchain);
        return pSwapChainVK->acquireNextImage(device, swapchain, timeout, semaphore, fence, pImageIndex);
    }
    else
    {
        // vkAcquireNextImageKHR would crash if swapchain is null.
        return VK_ERROR_SURFACE_LOST_KHR;
    }
}

VkResult vkCreateSwapchainFFX(VkDevice                           device,
                              const VkSwapchainCreateInfoKHR*    pCreateInfo,
                              const VkAllocationCallbacks*       pAllocator,
                              VkSwapchainKHR*                    pSwapchain,
                              const VkFrameInterpolationInfoFFX* pFrameInterpolationInfo)
{
    if (pCreateInfo == nullptr || pFrameInterpolationInfo == nullptr)
        return VK_ERROR_INITIALIZATION_FAILED;

    FrameInterpolationSwapChainVK* pSwapChainVK = createFrameInterpolationSwapChain(pAllocator);
    VkResult                       result       = pSwapChainVK->init(pCreateInfo, pFrameInterpolationInfo);
    if (result == VK_SUCCESS)
    {
        *pSwapchain = reinterpret_cast<VkSwapchainKHR>(pSwapChainVK);
    }
    else
    {
        pSwapChainVK->destroySwapchain(device, pAllocator);
        deleteFrameInterpolationSwapChain(pSwapChainVK, pAllocator);
        pSwapChainVK = nullptr;
    }
    return result;
}

void vkDestroySwapchainFFX(VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks* pAllocator)
{
    if (swapchain != VK_NULL_HANDLE)
    {
        FrameInterpolationSwapChainVK* pSwapChainVK = reinterpret_cast<FrameInterpolationSwapChainVK*>(swapchain);
        pSwapChainVK->destroySwapchain(device, pAllocator);
        delete pSwapChainVK;
    }
}

VkResult vkGetSwapchainImagesFFX(VkDevice device, VkSwapchainKHR swapchain, uint32_t* pSwapchainImageCount, VkImage* pSwapchainImages)
{
    if (swapchain != VK_NULL_HANDLE)
    {
        FrameInterpolationSwapChainVK* pSwapChainVK = reinterpret_cast<FrameInterpolationSwapChainVK*>(swapchain);
        return pSwapChainVK->getSwapchainImages(device, pSwapchainImageCount, pSwapchainImages);
    }
    else
    {
        // vkGetSwapchainImagesKHR would crash if swapchain is null.
        // no need to handle the case where pSwapchainImageCount is null as Vulkan itself doesn't handle it
        pSwapchainImageCount = 0;
        return VK_INCOMPLETE;
    }
}

VkResult vkQueuePresentFFX(VkQueue queue, const VkPresentInfoKHR* pPresentInfo)
{
    if (pPresentInfo->swapchainCount == 0)
    {
        return VK_SUCCESS;
    }
    // We DO NOT support multiple swapchains for now as there is no way to know which swapchain is a frame interpolation one.
    // We need to assume that the only one that is passed is indeed a frame interpolation one.
    FFX_ASSERT_MESSAGE(pPresentInfo->swapchainCount == 1, "vkQueuePresentFFX doesn't support multiple swapchains");
    FrameInterpolationSwapChainVK* pSwapChainVK = reinterpret_cast<FrameInterpolationSwapChainVK*>(pPresentInfo->pSwapchains[0]);
    if (pSwapChainVK != nullptr)
    {
        VkResult res = pSwapChainVK->queuePresent(queue, pPresentInfo);
        if (pPresentInfo->pResults != nullptr)
        {
            pPresentInfo->pResults[0] = res;
        }
        return res;
    }

    return VK_ERROR_SURFACE_LOST_KHR;
}

// Provided by VK_EXT_hdr_metadata
void vkSetHdrMetadataFFX(VkDevice device, uint32_t swapchainCount, const VkSwapchainKHR* pSwapchains, const VkHdrMetadataEXT* pMetadata)
{
    for (uint32_t i = 0; i < swapchainCount; ++i)
    {
        FrameInterpolationSwapChainVK* pSwapChainVK = reinterpret_cast<FrameInterpolationSwapChainVK*>(pSwapchains[i]);
        pSwapChainVK->setHdrMetadata(device, &pMetadata[i]);
    }
}

//////////////////////////////////////////////
/// FFX additional functions
//////////////////////////////////////////////

uint64_t getLastPresentCountFFX(VkSwapchainKHR swapchain)
{
    FrameInterpolationSwapChainVK* pSwapChainVK = reinterpret_cast<FrameInterpolationSwapChainVK*>(swapchain);
    return pSwapChainVK->getLastPresentCount();
}

//////////////////////////////////////////////
/// FFX API overridden functions
//////////////////////////////////////////////

FFX_API FfxErrorCode ffxGetSwapchainReplacementFunctionsVK(FfxDevice ffxDevice, FfxSwapchainReplacementFunctions* functions)
{
    functions->createSwapchainFFX    = vkCreateSwapchainFFX;
    functions->destroySwapchainKHR   = vkDestroySwapchainFFX;
    functions->getSwapchainImagesKHR = vkGetSwapchainImagesFFX;
    functions->acquireNextImageKHR   = vkAcquireNextImageFFX;
    functions->queuePresentKHR       = vkQueuePresentFFX;

    // for extensions, make sure the base functions exist
    VkDevice device = static_cast<VkDevice>(ffxDevice);
    FFX_ASSERT(device != VK_NULL_HANDLE);

    // VK_EXT_hdr_metadata
    if (VulkanWrapper().vkGetDeviceProcAddr(device, "vkSetHdrMetadataEXT") != nullptr)
        functions->setHdrMetadataEXT = vkSetHdrMetadataFFX;
    else
        functions->setHdrMetadataEXT = nullptr;

    // additional functions only available for frame interpolation swapchain
    functions->getLastPresentCountFFX = getLastPresentCountFFX;

    return FFX_OK;
}

FfxErrorCode ffxRegisterFrameinterpolationUiResourceVK(FfxSwapchain gameSwapChain, const FfxResource& uiResource, uint32_t flags)
{
    FrameInterpolationSwapChainVK* pSwapChainVK = reinterpret_cast<FrameInterpolationSwapChainVK*>(gameSwapChain);
    pSwapChainVK->registerUiResource(uiResource, flags);
    return FFX_OK;
}

FFX_API FfxErrorCode ffxSetFrameGenerationConfigToSwapchainVK(FfxFrameGenerationConfig const* config)
{
    FfxErrorCode result = FFX_ERROR_INVALID_ARGUMENT;

    if (config->swapChain)
    {
        FrameInterpolationSwapChainVK* frameinterpolationSwapchain = reinterpret_cast<FrameInterpolationSwapChainVK*>(config->swapChain);
        if (frameinterpolationSwapchain != VK_NULL_HANDLE)
        {
            frameinterpolationSwapchain->setFrameGenerationConfig(config);
            result = FFX_OK;
        }
    }

    return result;
}

FfxErrorCode ffxConfigureFrameInterpolationSwapchainVK(FfxSwapchain gameSwapChain, FfxFrameInterpolationSwapchainConfigureKey key, void* valuePtr)
{
    if (gameSwapChain)
    {
        FrameInterpolationSwapChainVK* pSwapChainVK = reinterpret_cast<FrameInterpolationSwapChainVK*>(gameSwapChain);
        switch (key)
        {
        case FFX_FI_SWAPCHAIN_CONFIGURE_KEY_WAITCALLBACK:
            pSwapChainVK->setWaitCallback(reinterpret_cast<FfxWaitCallbackFunc>(valuePtr));
            break;
        case FFX_FI_SWAPCHAIN_CONFIGURE_KEY_FRAMEPACINGTUNING:
            if (valuePtr != nullptr)
            {
                pSwapChainVK->setFramePacingTuning(static_cast<FfxSwapchainFramePacingTuning*>(valuePtr));
            }
            break;
            return FFX_OK;
        }
    }

    return FFX_ERROR_INVALID_ARGUMENT;
}

FfxResource ffxGetFrameinterpolationTextureVK(FfxSwapchain gameSwapChain)
{
    FrameInterpolationSwapChainVK* pSwapChainVK = reinterpret_cast<FrameInterpolationSwapChainVK*>(gameSwapChain);
    FfxResource                    res          = pSwapChainVK->interpolationOutput(0);
    return res;
}

FfxErrorCode ffxGetFrameinterpolationCommandlistVK(FfxSwapchain gameSwapChain, FfxCommandList& gameCommandlist)
{
    FrameInterpolationSwapChainVK* frameinterpolationSwapchain = reinterpret_cast<FrameInterpolationSwapChainVK*>(gameSwapChain);

    gameCommandlist = reinterpret_cast<FfxCommandList>(frameinterpolationSwapchain->getInterpolationCommandList());

    return FFX_OK;
}

FfxErrorCode ffxReplaceSwapchainForFrameinterpolationVK(FfxCommandQueue                    gameQueue,
                                                        FfxSwapchain&                      gameSwapChain,
                                                        const VkSwapchainCreateInfoKHR*    swapchainCreateInfo,
                                                        const VkFrameInterpolationInfoFFX* frameInterpolationInfo)
{
    FfxErrorCode status = FFX_OK;

    VkSwapchainKHR gameSwapchain = reinterpret_cast<VkSwapchainKHR>(gameSwapChain);
    FFX_ASSERT(swapchainCreateInfo != VK_NULL_HANDLE);

    if (frameInterpolationInfo == nullptr)
        return FFX_ERROR_INVALID_ARGUMENT;

    FFX_ASSERT(frameInterpolationInfo->device != VK_NULL_HANDLE);
    FFX_ASSERT(frameInterpolationInfo->physicalDevice != VK_NULL_HANDLE);
    FFX_ASSERT(gameQueue != VK_NULL_HANDLE);
    FFX_ASSERT(gameQueue == frameInterpolationInfo->gameQueue.queue);

    const VkAllocationCallbacks* pAllocator = frameInterpolationInfo->pAllocator;
    VkSwapchainCreateInfoKHR     createInfo = *swapchainCreateInfo;  // copy

    // createInfo.oldSwapchain should be the same as gameSwapchain if not VK_NULL_HANDLE
    if (createInfo.oldSwapchain != VK_NULL_HANDLE && createInfo.oldSwapchain != gameSwapchain)
        return FFX_ERROR_INVALID_ARGUMENT;

    // use the old swapchain to help with resource reuse
    createInfo.oldSwapchain = gameSwapchain;

    FrameInterpolationSwapChainVK* pSwapChainVK = createFrameInterpolationSwapChain(pAllocator);
    VkResult                       result       = pSwapChainVK->init(&createInfo, frameInterpolationInfo);
    if (result != VK_SUCCESS)
    {
        pSwapChainVK->destroySwapchain(frameInterpolationInfo->device, pAllocator);
        deleteFrameInterpolationSwapChain(pSwapChainVK, pAllocator);
        pSwapChainVK = nullptr;
        status       = FFX_ERROR_BACKEND_API_ERROR;
    }

    // as per Vulkan documentation, oldSwapchain is retired - even if creation of the new swapchain fails.
    VulkanWrapper().vkDestroySwapchainKHR(frameInterpolationInfo->device, gameSwapchain, pAllocator);

    gameSwapChain = reinterpret_cast<VkSwapchainKHR>(pSwapChainVK);

    return status;
}

FfxErrorCode ffxWaitForPresents(FfxSwapchain gameSwapChain)
{
    FrameInterpolationSwapChainVK* frameinterpolationSwapchain = reinterpret_cast<FrameInterpolationSwapChainVK*>(gameSwapChain);

    frameinterpolationSwapchain->waitForPresents();

    return FFX_OK;
}

FfxErrorCode ffxFrameInterpolationSwapchainGetGpuMemoryUsageVK(FfxSwapchain gameSwapChain, FfxEffectMemoryUsage* vramUsage)
{
    FFX_RETURN_ON_ERROR(vramUsage, FFX_ERROR_INVALID_POINTER);
    FrameInterpolationSwapChainVK* pSwapChainVK = reinterpret_cast<FrameInterpolationSwapChainVK*>(gameSwapChain);

    pSwapChainVK->getGpuMemoryUsage(vramUsage);

    return FFX_OK;
}

//////////////////////////////////////////////
/// Present
//////////////////////////////////////////////
VkResult presentToSwapChain(FrameinterpolationPresentInfo* pPresenter, uint32_t imageIndex, uint32_t semaphoreIndex = 0)
{
    VkPresentInfoKHR presentInfoKHR   = {};
    presentInfoKHR.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfoKHR.pNext              = nullptr;
    presentInfoKHR.waitSemaphoreCount = 1;
    presentInfoKHR.pWaitSemaphores    = &pPresenter->frameRenderedSemaphores[semaphoreIndex];
    presentInfoKHR.swapchainCount     = 1;
    presentInfoKHR.pSwapchains        = &pPresenter->realSwapchain;
    presentInfoKHR.pImageIndices      = &imageIndex;
    presentInfoKHR.pResults           = nullptr;  // Optional
    VkResult res                      = VK_SUCCESS;

#ifdef FFX_PLATFORM_WINDOWS
    EnterCriticalSection(&pPresenter->swapchainCriticalSection);
    res = VulkanWrapper().vkQueuePresentKHR(pPresenter->presentQueue.queue, &presentInfoKHR);
    LeaveCriticalSection(&pPresenter->swapchainCriticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_lock(&pPresenter->swapchainMutex);
    res = VulkanWrapper().vkQueuePresentKHR(pPresenter->presentQueue.queue, &presentInfoKHR);
    pthread_mutex_unlock(&pPresenter->swapchainMutex);
#endif
    ++(pPresenter->realPresentCount);
    return res;
}

VkResult compositeSwapChainFrame(FrameinterpolationPresentInfo* pPresenter,
                                 const PacingData*              pPacingEntry,
                                 const PacingData::FrameType    frameType,
                                 const uint32_t                 realSwapchainImageIndex,
                                 const VulkanQueue              compositionQueue,
                                 SubmissionSemaphores&          semaphoresToWait,
                                 SubmissionSemaphores&          semaphoresToSignal,
                                 bool&                          uiSurfaceTransfered)
{
    const PacingData::FrameInfo& frameInfo = pPacingEntry->frames[frameType];

    semaphoresToWait.add(pPresenter->interpolationSemaphore, frameInfo.interpolationCompletedSemaphoreValue);
    semaphoresToSignal.add(pPresenter->compositionSemaphore, frameInfo.presentIndex);

    if (pPacingEntry->drawDebugPacingLines)
    {
        auto            compositeCommandList   = pPresenter->commandPool.get(pPresenter->device, compositionQueue, "compositeCommandList");
        VkCommandBuffer compositeCommandBuffer = compositeCommandList->reset();

        FfxPresentCallbackDescription desc{};
        desc.commandList         = ffxGetCommandListVK(compositeCommandBuffer);
        desc.device              = pPresenter->device;
        desc.isInterpolatedFrame = frameType != PacingData::FrameType::Real;
        desc.outputSwapChainBuffer =
            ffxGetResourceVK(pPresenter->compositionOutput.image, pPresenter->compositionOutput.description, nullptr, FFX_RESOURCE_STATE_COPY_SRC);
        desc.currentBackBuffer = frameInfo.resource;
        desc.currentUI         = pPacingEntry->uiSurface;
        desc.usePremulAlpha    = pPacingEntry->usePremulAlphaComposite;
        desc.frameID           = pPacingEntry->currentFrameID;

        FfxDebugPacingContext debugPacingContext;
        debugPacingContext.physicalDevice = pPresenter->physicalDevice;

        ffxFrameInterpolationDebugPacing(&desc, &debugPacingContext);

        SubmissionSemaphores toSignalDummy;
        compositeCommandList->execute(semaphoresToWait, toSignalDummy);
    }

    if (pPacingEntry->presentCallback)
    {
        auto            compositeCommandList   = pPresenter->commandPool.get(pPresenter->device, compositionQueue, "compositeCommandList");
        VkCommandBuffer compositeCommandBuffer = compositeCommandList->reset();

        FfxPresentCallbackDescription desc{};
        desc.commandList         = ffxGetCommandListVK(compositeCommandBuffer);
        desc.device              = pPresenter->device;
        desc.isInterpolatedFrame = frameType != PacingData::FrameType::Real;
        if (pPresenter->compositionMode == FGSwapchainCompositionMode::eComposeOnPresentQueue)
        {
            desc.outputSwapChainBuffer = ffxGetResourceVK(
                pPresenter->realSwapchainImages[realSwapchainImageIndex], pPresenter->realSwapchainImageDescription, nullptr, FFX_RESOURCE_STATE_PRESENT);
        }
        if (pPresenter->compositionMode == FGSwapchainCompositionMode::eComposeOnGameQueue)
        {
            desc.outputSwapChainBuffer =
                ffxGetResourceVK(pPresenter->compositionOutput.image, pPresenter->compositionOutput.description, nullptr, FFX_RESOURCE_STATE_COPY_SRC);
        }
        desc.currentBackBuffer = frameInfo.resource;
        desc.currentUI         = pPacingEntry->uiSurface;
        desc.usePremulAlpha    = pPacingEntry->usePremulAlphaComposite;
        desc.frameID           = pPacingEntry->currentFrameID;

        // queue family ownership transfer for interpolation output & UI surface
        ImageBarrierHelper preCallbackBarriers;
        if (pPresenter->interpolationQueue.familyIndex != compositionQueue.familyIndex)
        {
            if (frameType == PacingData::FrameType::Interpolated_1)
            {
                // this is the interpolation buffer
                preCallbackBarriers.add(static_cast<VkImage>(frameInfo.resource.resource),
                                        InterpolationBufferTransferState.accessMask,
                                        InterpolationBufferTransferState.accessMask,
                                        InterpolationBufferTransferState.layout,
                                        InterpolationBufferTransferState.layout,
                                        pPresenter->interpolationQueue.familyIndex,
                                        compositionQueue.familyIndex);
            }
            else if (frameType == PacingData::FrameType::Real)
            {
                // this is the replacement buffer
                preCallbackBarriers.add(static_cast<VkImage>(frameInfo.resource.resource),
                                        ReplacementBufferTransferState.accessMask,
                                        ReplacementBufferTransferState.accessMask,
                                        ReplacementBufferTransferState.layout,
                                        ReplacementBufferTransferState.layout,
                                        pPresenter->interpolationQueue.familyIndex,
                                        compositionQueue.familyIndex);
            }
        }
        if (!uiSurfaceTransfered)
        {
            preCallbackBarriers.add(pPresenter->queueFamilyOwnershipTransferGameToPresent(pPacingEntry->uiSurface));
            uiSurfaceTransfered = true;
        }

        if (pPresenter->compositionMode == FGSwapchainCompositionMode::eComposeOnPresentQueue)
        {
            // change real image to present layout
            preCallbackBarriers.add(pPresenter->realSwapchainImages[realSwapchainImageIndex], 0, 0, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        }
        else if (pPresenter->compositionMode == FGSwapchainCompositionMode::eComposeOnGameQueue)
        {
            // prepare for copy
            preCallbackBarriers.add(
                pPresenter->compositionOutput.image, 0, VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        }

        preCallbackBarriers.record(compositeCommandBuffer);

        pPacingEntry->presentCallback(&desc, pPacingEntry->presentCallbackContext);

        if (pPresenter->compositionMode == FGSwapchainCompositionMode::eComposeOnGameQueue)
        {
            // copy back the content of the composition in the replacement or the interpolation output buffer
            ImageBarrierHelper preCopyBarriers;

            // composition output buffer is already in a VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL layout on the compose queue
            // just set resource to copy dest layout
            preCopyBarriers.add(static_cast<VkImage>(frameInfo.resource.resource),
                                0,
                                VK_ACCESS_TRANSFER_WRITE_BIT,
                                VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            preCopyBarriers.record(compositeCommandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

            recordCopy(compositeCommandBuffer,
                       pPresenter->compositionOutput.image,
                       static_cast<VkImage>(frameInfo.resource.resource),
                       pPresenter->realSwapchainImageDescription.width,
                       pPresenter->realSwapchainImageDescription.height);

            ImageBarrierHelper postCopyBarriers;

            postCopyBarriers.add(static_cast<VkImage>(frameInfo.resource.resource),
                                 VK_ACCESS_TRANSFER_WRITE_BIT,
                                 VK_ACCESS_TRANSFER_READ_BIT,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 // this will handle the queue family ownership transfer if any
                                 compositionQueue.familyIndex,
                                 pPresenter->presentQueue.familyIndex);

            postCopyBarriers.record(compositeCommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        }

        return compositeCommandList->execute(semaphoresToWait, semaphoresToSignal);
    }
    else
    {
        return pPresenter->presentQueue.submit(VK_NULL_HANDLE, semaphoresToWait, semaphoresToSignal);
    }
}

#ifdef FFX_PLATFORM_WINDOWS
DWORD WINAPI copyAndPresent_presenterThread(LPVOID pParam)
{
    FrameinterpolationPresentInfo* presenter = static_cast<FrameinterpolationPresentInfo*>(pParam);

    if (presenter)
    {
        uint64_t numFramesSentForPresentation = 0;
        int64_t  previousPresentQpc           = 0;

        while (!presenter->shutdown)
        {
            WaitForSingleObject(presenter->pacerEvent, INFINITE);

            if (!presenter->shutdown)
            {
                EnterCriticalSection(&presenter->scheduledFrameCriticalSection);

                PacingData entry = presenter->scheduledPresents;
                presenter->scheduledPresents.invalidate();

                LeaveCriticalSection(&presenter->scheduledFrameCriticalSection);

                if (entry.numFramesToPresent > 0)
                {
                    // we might have dropped entries so have to update here, otherwise we might deadlock
                    // we need to track the latest signaled value to avoid validation warnings
                    if (presenter->lastPresentSemaphoreValue != entry.numFramesSentForPresentationBase)
                    {
                        presenter->presentQueue.submit(VK_NULL_HANDLE, presenter->presentSemaphore, entry.numFramesSentForPresentationBase);
                        presenter->lastPresentSemaphoreValue = entry.numFramesSentForPresentationBase;
                    }

                    for (uint32_t frameType = 0; frameType < PacingData::FrameType::Count; frameType++)
                    {
                        const PacingData::FrameInfo& frameInfo = entry.frames[frameType];
                        if (frameInfo.doPresent)
                        {
                            uint32_t    imageIndex              = 0;
                            VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
                            VkResult    res                     = presenter->acquireNextRealImage(imageIndex, imageAvailableSemaphore);
                            FFX_ASSERT_MESSAGE_FORMAT(
                                res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_NOT_READY || res == VK_TIMEOUT,
                                "[copyAndPresent_presenterThread] failed to acquire swapchain image");

                            SubmissionSemaphores toSignal;
                            SubmissionSemaphores toWait;
                            toWait.add(presenter->compositionSemaphore, frameInfo.presentIndex);  // composition to finish
                            toSignal.add(presenter->frameRenderedSemaphores[imageIndex]);         // frame ready for present. Not a timeline semaphore

                            // no image was acquired, just skip everything and signal the appropriate semaphores

                            // signal replacement buffer availability
                            // this is the last present of this entry
                            if (frameInfo.presentIndex == entry.replacementBufferSemaphoreSignal)
                            {
                                toSignal.add(presenter->replacementBufferSemaphore, entry.replacementBufferSemaphoreSignal);
                            }

                            // There is no way to signal a semaphore after Present, so signal it before it.
                            if (frameInfo.presentIndex != entry.numFramesSentForPresentationBase)
                            {
                                // no need to signal twice
                                toSignal.add(presenter->presentSemaphore, frameInfo.presentIndex);
                                presenter->lastPresentSemaphoreValue = frameInfo.presentIndex;
                            }

                            if (res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR)
                            {
                                toWait.add(imageAvailableSemaphore);  // swapchain image to be available

                                auto presentCommandList = presenter->commandPool.get(presenter->device, presenter->presentQueue, "presentCommandList");
                                VkCommandBuffer presentCommandBuffer = presentCommandList->reset();

                                ImageBarrierHelper preCopyBarriers;

                                // newly acquired image transition
                                preCopyBarriers.add(presenter->realSwapchainImages[imageIndex],
                                                    0,
                                                    VK_ACCESS_TRANSFER_WRITE_BIT,
                                                    VK_IMAGE_LAYOUT_UNDEFINED,
                                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

                                // queue family ownership transfer for the texture containing the final image
                                if (presenter->gameQueue.familyIndex != presenter->presentQueue.familyIndex)
                                {
                                    preCopyBarriers.add(static_cast<VkImage>(frameInfo.resource.resource),
                                                        VK_ACCESS_TRANSFER_WRITE_BIT,
                                                        VK_ACCESS_TRANSFER_READ_BIT,
                                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                        presenter->gameQueue.familyIndex,
                                                        presenter->presentQueue.familyIndex);
                                }

                                preCopyBarriers.record(presentCommandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

                                // copy from resource to the swapchain image
                                recordCopy(presentCommandBuffer,
                                           static_cast<VkImage>(frameInfo.resource.resource),
                                           presenter->realSwapchainImages[imageIndex],
                                           presenter->realSwapchainImageDescription.width,
                                           presenter->realSwapchainImageDescription.height);

                                ImageBarrierHelper postCopyBarriers;
                                // transition to present
                                postCopyBarriers.add(presenter->realSwapchainImages[imageIndex],
                                                     VK_ACCESS_TRANSFER_WRITE_BIT,
                                                     0,
                                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

                                postCopyBarriers.record(presentCommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

                                res = presentCommandList->execute(toWait, toSignal);

                                waitForPerformanceCount(previousPresentQpc + frameInfo.presentQpcDelta);
                                QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&previousPresentQpc));

                                res = presentToSwapChain(presenter, imageIndex, imageIndex);
                                // VK_SUBOPTIMAL_KHR & VK_ERROR_OUT_OF_DATE_KHR: the swapchain has been recreated
                                FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR,
                                                          "presentToSwapChain failed with error %d",
                                                          res);
                            }
                            else
                            {
                                // no image has been acquired, just wait on and signal the semaphores
                                presenter->presentQueue.submit(VK_NULL_HANDLE, toWait, toSignal);
                            }

                            // store result to send it back to next queuePresent call
                            presenter->lastPresentResult.store(res);
                        }
                    }

                    numFramesSentForPresentation = entry.numFramesSentForPresentationBase + entry.numFramesToPresent;
                }
            }
        }

        waitForSemaphoreValue(presenter->device, presenter->presentSemaphore, numFramesSentForPresentation);
    }

    return 0;
}
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
void* copyAndPresent_presenterThread(void* arg)
{
    FrameinterpolationPresentInfo* presenter = static_cast<FrameinterpolationPresentInfo*>(arg);

    if (presenter)
    {
        uint64_t numFramesSentForPresentation = 0;
        int64_t previousPresentQpc = 0;

        while (!presenter->shutdown)
        {
            pthread_mutex_lock(&presenter->pacerMutex);
            while (!presenter->pacerEventSignaled)
                pthread_cond_wait(&presenter->pacerCond, &presenter->pacerMutex);
            presenter->pacerEventSignaled = false;
            pthread_mutex_unlock(&presenter->pacerMutex);

            if (!presenter->shutdown)
            {
                pthread_mutex_lock(&presenter->scheduledFrameMutex);

                PacingData entry = presenter->scheduledPresents;
                presenter->scheduledPresents.invalidate();

                pthread_mutex_unlock(&presenter->scheduledFrameMutex);

                if (entry.numFramesToPresent > 0)
                {
                    // we might have dropped entries so have to update here, otherwise we might deadlock
                    // we need to track the latest signaled value to avoid validation warnings
                    if (presenter->lastPresentSemaphoreValue != entry.numFramesSentForPresentationBase)
                    {
                        presenter->presentQueue.submit(VK_NULL_HANDLE, presenter->presentSemaphore, entry.numFramesSentForPresentationBase);
                        presenter->lastPresentSemaphoreValue = entry.numFramesSentForPresentationBase;
                    }

                    for (uint32_t frameType = 0; frameType < PacingData::FrameType::Count; frameType++)
                    {
                        const PacingData::FrameInfo& frameInfo = entry.frames[frameType];
                        if (frameInfo.doPresent)
                        {
                            uint32_t imageIndex = 0;
                            VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
                            VkResult res = presenter->acquireNextRealImage(imageIndex, imageAvailableSemaphore);
                            FFX_ASSERT_MESSAGE_FORMAT(
                                res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_NOT_READY || res == VK_TIMEOUT,
                                "[copyAndPresent_presenterThread] failed to acquire swapchain image");

                            SubmissionSemaphores toSignal;
                            SubmissionSemaphores toWait;
                            toWait.add(presenter->compositionSemaphore, frameInfo.presentIndex);  // composition to finish
                            toSignal.add(presenter->frameRenderedSemaphores[imageIndex]);         // frame ready for present. Not a timeline semaphore

                            // no image was acquired, just skip everything and signal the appropriate semaphores

                            // signal replacement buffer availability
                            // this is the last present of this entry
                            if (frameInfo.presentIndex == entry.replacementBufferSemaphoreSignal)
                            {
                                toSignal.add(presenter->replacementBufferSemaphore, entry.replacementBufferSemaphoreSignal);
                            }

                            // There is no way to signal a semaphore after Present, so signal it before it.
                            if (frameInfo.presentIndex != entry.numFramesSentForPresentationBase)
                            {
                                // no need to signal twice
                                toSignal.add(presenter->presentSemaphore, frameInfo.presentIndex);
                                presenter->lastPresentSemaphoreValue = frameInfo.presentIndex;
                            }

                            if (res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR)
                            {
                                toWait.add(imageAvailableSemaphore);  // swapchain image to be available

                                auto presentCommandList = presenter->commandPool.get(presenter->device, presenter->presentQueue, "presentCommandList");
                                VkCommandBuffer presentCommandBuffer = presentCommandList->reset();

                                ImageBarrierHelper preCopyBarriers;

                                // newly acquired image transition
                                preCopyBarriers.add(presenter->realSwapchainImages[imageIndex],
                                                    0,
                                                    VK_ACCESS_TRANSFER_WRITE_BIT,
                                                    VK_IMAGE_LAYOUT_UNDEFINED,
                                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

                                // queue family ownership transfer for the texture containing the final image
                                if (presenter->gameQueue.familyIndex != presenter->presentQueue.familyIndex)
                                {
                                    preCopyBarriers.add(static_cast<VkImage>(frameInfo.resource.resource),
                                                        VK_ACCESS_TRANSFER_WRITE_BIT,
                                                        VK_ACCESS_TRANSFER_READ_BIT,
                                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                        presenter->gameQueue.familyIndex,
                                                        presenter->presentQueue.familyIndex);
                                }

                                preCopyBarriers.record(presentCommandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

                                // copy from resource to the swapchain image
                                recordCopy(presentCommandBuffer,
                                           static_cast<VkImage>(frameInfo.resource.resource),
                                           presenter->realSwapchainImages[imageIndex],
                                           presenter->realSwapchainImageDescription.width,
                                           presenter->realSwapchainImageDescription.height);

                                ImageBarrierHelper postCopyBarriers;
                                // transition to present
                                postCopyBarriers.add(presenter->realSwapchainImages[imageIndex],
                                                     VK_ACCESS_TRANSFER_WRITE_BIT,
                                                     0,
                                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

                                postCopyBarriers.record(presentCommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

                                res = presentCommandList->execute(toWait, toSignal);

                                waitForPerformanceCount(previousPresentQpc + frameInfo.presentQpcDelta);
                                struct timespec ts;
                                clock_gettime(CLOCK_MONOTONIC, &ts);
                                previousPresentQpc = ts.tv_sec * 1000000000LL + ts.tv_nsec;

                                res = presentToSwapChain(presenter, imageIndex, imageIndex);
                                // VK_SUBOPTIMAL_KHR & VK_ERROR_OUT_OF_DATE_KHR: the swapchain has been recreated
                                FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR,
                                                          "presentToSwapChain failed with error %d",
                                                          res);
                            }
                            else
                            {
                                // no image has been acquired, just wait on and signal the semaphores
                                presenter->presentQueue.submit(VK_NULL_HANDLE, toWait, toSignal);
                            }

                            // store result to send it back to next queuePresent call
                            presenter->lastPresentResult.store(res);
                        }
                    }

                    numFramesSentForPresentation = entry.numFramesSentForPresentationBase + entry.numFramesToPresent;
                }
            }
        }

        waitForSemaphoreValue(presenter->device, presenter->presentSemaphore, numFramesSentForPresentation);
    }

    return 0;
}
#endif

#ifdef FFX_PLATFORM_WINDOWS
DWORD WINAPI composeAndPresent_presenterThread(LPVOID pParam)
{
    FrameinterpolationPresentInfo* presenter = static_cast<FrameinterpolationPresentInfo*>(pParam);

    if (presenter)
    {
        uint64_t numFramesSentForPresentation = 0;
        int64_t  previousPresentQpc           = 0;

        while (!presenter->shutdown)
        {
            WaitForSingleObject(presenter->pacerEvent, INFINITE);

            if (!presenter->shutdown)
            {
                EnterCriticalSection(&presenter->scheduledFrameCriticalSection);

                PacingData entry = presenter->scheduledPresents;
                presenter->scheduledPresents.invalidate();

                LeaveCriticalSection(&presenter->scheduledFrameCriticalSection);

                if (entry.numFramesToPresent > 0)
                {
                    // we might have dropped entries so have to update here, otherwise we might deadlock
                    // we need to track the latest signaled value to avoid validation warnings
                    if (presenter->lastPresentSemaphoreValue != entry.numFramesSentForPresentationBase)
                    {
                        presenter->presentQueue.submit(VK_NULL_HANDLE, presenter->presentSemaphore, entry.numFramesSentForPresentationBase);
                        presenter->lastPresentSemaphoreValue = entry.numFramesSentForPresentationBase;
                    }

                    SubmissionSemaphores toWait;
                    toWait.add(presenter->gameSemaphore, entry.gameSemaphoreValue);

                    bool uiSurfaceTransfered =
                        (entry.uiSurface.resource == nullptr) || (presenter->gameQueue.familyIndex == presenter->presentQueue.familyIndex);

                    for (uint32_t frameType = 0; frameType < PacingData::FrameType::Count; frameType++)
                    {
                        const PacingData::FrameInfo& frameInfo = entry.frames[frameType];
                        if (frameInfo.doPresent)
                        {
                            SubmissionSemaphores toSignal;
                            toSignal.add(presenter->frameRenderedSemaphores[0]);  // not a timeline semaphore

                            // signal replacement buffer availability
                            // this is the last present of this entry
                            if (frameInfo.presentIndex == entry.replacementBufferSemaphoreSignal)
                            {
                                toSignal.add(presenter->replacementBufferSemaphore, entry.replacementBufferSemaphoreSignal);
                            }

                            // this should be signaled after present, but this cannot be done in Vulkan. Signaling here
                            if (frameInfo.presentIndex != entry.numFramesSentForPresentationBase)
                            {
                                // no need to signal twice
                                toSignal.add(presenter->presentSemaphore, frameInfo.presentIndex);
                                presenter->lastPresentSemaphoreValue = frameInfo.presentIndex;
                            }

                            // get the swapchain image
                            uint32_t    realSwapchainImageIndex = 0;
                            VkSemaphore acquireSemaphore        = VK_NULL_HANDLE;
                            VkResult    res                     = presenter->acquireNextRealImage(realSwapchainImageIndex, acquireSemaphore);
                            FFX_ASSERT_MESSAGE_FORMAT(
                                res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_NOT_READY || res == VK_TIMEOUT,
                                "[composeAndPresent_presenterThread] failed to acquire swapchain image");

                            if (res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR)
                            {
                                toWait.add(acquireSemaphore);

                                res = compositeSwapChainFrame(presenter,
                                                              &entry,
                                                              (PacingData::FrameType)frameType,
                                                              realSwapchainImageIndex,
                                                              presenter->presentQueue,
                                                              toWait,
                                                              toSignal,
                                                              uiSurfaceTransfered);
                                FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS, "compositeSwapChainFrame failed with error %d", res);

                                waitForPerformanceCount(previousPresentQpc + frameInfo.presentQpcDelta);
                                QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&previousPresentQpc));

                                res = presentToSwapChain(presenter, realSwapchainImageIndex);
                                // VK_SUBOPTIMAL_KHR & VK_ERROR_OUT_OF_DATE_KHR: the swapchain has been recreated
                                FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR,
                                                          "presentToSwapChain failed with error %d",
                                                          res);
                            }
                            else
                            {
                                // no image has been acquired, just wait on and signal the semaphores
                                // just signal the composition semaphore as if composition has been executed
                                toSignal.add(presenter->compositionSemaphore, frameInfo.presentIndex);
                                presenter->presentQueue.submit(VK_NULL_HANDLE, toWait, toSignal);
                            }
                            // store result to send it back to next queuePresent call
                            presenter->lastPresentResult.store(res);
                        }
                    }

                    // if no frame was presented, we still need to update the semaphore
                    if (toWait.count > 0)
                    {
                        presenter->presentQueue.submit(VK_NULL_HANDLE, toWait, SubmissionSemaphores());
                    }

                    numFramesSentForPresentation = entry.numFramesSentForPresentationBase + entry.numFramesToPresent;
                    processAsyncDumpOperations(presenter->device, entry.currentFrameID);
                }
            }
        }

        waitForSemaphoreValue(presenter->device, presenter->presentSemaphore, numFramesSentForPresentation);
    }

    return 0;
}
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
void* composeAndPresent_presenterThread(void* pParam)
{
    FrameinterpolationPresentInfo* presenter = static_cast<FrameinterpolationPresentInfo*>(pParam);

    if (presenter)
    {
        uint64_t numFramesSentForPresentation = 0;
        int64_t previousPresentQpc = 0;

        while (!presenter->shutdown)
        {
            pthread_mutex_lock(&presenter->pacerMutex);
            while (!presenter->pacerEventSignaled)
                pthread_cond_wait(&presenter->pacerCond, &presenter->pacerMutex);
            presenter->pacerEventSignaled = false;
            pthread_mutex_unlock(&presenter->pacerMutex);

            if (!presenter->shutdown)
            {
                pthread_mutex_lock(&presenter->scheduledFrameMutex);
                PacingData entry = presenter->scheduledPresents;
                presenter->scheduledPresents.invalidate();
                pthread_mutex_unlock(&presenter->scheduledFrameMutex);

                if (entry.numFramesToPresent > 0)
                {
                    // we might have dropped entries so have to update here, otherwise we might deadlock
                    // we need to track the latest signaled value to avoid validation warnings
                    if (presenter->lastPresentSemaphoreValue != entry.numFramesSentForPresentationBase)
                    {
                        presenter->presentQueue.submit(VK_NULL_HANDLE, presenter->presentSemaphore, entry.numFramesSentForPresentationBase);
                        presenter->lastPresentSemaphoreValue = entry.numFramesSentForPresentationBase;
                    }

                    SubmissionSemaphores toWait;
                    toWait.add(presenter->gameSemaphore, entry.gameSemaphoreValue);

                    bool uiSurfaceTransfered =
                        (entry.uiSurface.resource == nullptr) || (presenter->gameQueue.familyIndex == presenter->presentQueue.familyIndex);

                    for (uint32_t frameType = 0; frameType < PacingData::FrameType::Count; frameType++)
                    {
                        const PacingData::FrameInfo& frameInfo = entry.frames[frameType];
                        if (frameInfo.doPresent)
                        {
                            SubmissionSemaphores toSignal;
                            toSignal.add(presenter->frameRenderedSemaphores[0]);  // not a timeline semaphore

                            // signal replacement buffer availability
                            // this is the last present of this entry
                            if (frameInfo.presentIndex == entry.replacementBufferSemaphoreSignal)
                            {
                                toSignal.add(presenter->replacementBufferSemaphore, entry.replacementBufferSemaphoreSignal);
                            }

                            // this should be signaled after present, but this cannot be done in Vulkan. Signaling here
                            if (frameInfo.presentIndex != entry.numFramesSentForPresentationBase)
                            {
                                // no need to signal twice
                                toSignal.add(presenter->presentSemaphore, frameInfo.presentIndex);
                                presenter->lastPresentSemaphoreValue = frameInfo.presentIndex;
                            }

                            // get the swapchain image
                            uint32_t realSwapchainImageIndex = 0;
                            VkSemaphore acquireSemaphore = VK_NULL_HANDLE;
                            VkResult res = presenter->acquireNextRealImage(realSwapchainImageIndex, acquireSemaphore);
                            FFX_ASSERT_MESSAGE_FORMAT(
                                res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_NOT_READY || res == VK_TIMEOUT,
                                "[composeAndPresent_presenterThread] failed to acquire swapchain image");

                            if (res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR)
                            {
                                toWait.add(acquireSemaphore);

                                res = compositeSwapChainFrame(presenter,
                                                              &entry,
                                                              (PacingData::FrameType)frameType,
                                                              realSwapchainImageIndex,
                                                              presenter->presentQueue,
                                                              toWait,
                                                              toSignal,
                                                              uiSurfaceTransfered);
                                FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS, "compositeSwapChainFrame failed with error %d", res);

                                waitForPerformanceCount(previousPresentQpc + frameInfo.presentQpcDelta);
                                struct timespec ts;
                                clock_gettime(CLOCK_MONOTONIC, &ts);
                                previousPresentQpc = ts.tv_sec * 1000000000LL + ts.tv_nsec;

                                res = presentToSwapChain(presenter, realSwapchainImageIndex);
                                // VK_SUBOPTIMAL_KHR & VK_ERROR_OUT_OF_DATE_KHR: the swapchain has been recreated
                                FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR,
                                                          "presentToSwapChain failed with error %d",
                                                          res);
                            }
                            else
                            {
                                // no image has been acquired, just wait on and signal the semaphores
                                // just signal the composition semaphore as if composition has been executed
                                toSignal.add(presenter->compositionSemaphore, frameInfo.presentIndex);
                                presenter->presentQueue.submit(VK_NULL_HANDLE, toWait, toSignal);
                            }
                            // store result to send it back to next queuePresent call
                            presenter->lastPresentResult.store(res);
                        }
                    }

                    // if no frame was presented, we still need to update the semaphore
                    if (toWait.count > 0)
                    {
                        SubmissionSemaphores semaphores;
                        presenter->presentQueue.submit(VK_NULL_HANDLE, toWait, semaphores);
                    }

                    numFramesSentForPresentation = entry.numFramesSentForPresentationBase + entry.numFramesToPresent;
                    processAsyncDumpOperations(presenter->device, entry.currentFrameID);
                }
            }
        }

        waitForSemaphoreValue(presenter->device, presenter->presentSemaphore, numFramesSentForPresentation);
    }

    return 0;
}
#endif

#ifdef FFX_PLATFORM_WINDOWS
DWORD WINAPI interpolationThread(LPVOID param)
{
    FrameinterpolationPresentInfo* presenter = static_cast<FrameinterpolationPresentInfo*>(param);

    if (presenter)
    {
        HANDLE presenterThreadHandle = NULL;
        if (presenter->compositionMode == FGSwapchainCompositionMode::eComposeOnPresentQueue)
        {
            presenterThreadHandle = CreateThread(nullptr, 0, composeAndPresent_presenterThread, param, 0, nullptr);
        }
        else if (presenter->compositionMode == FGSwapchainCompositionMode::eComposeOnGameQueue)
        {
            presenterThreadHandle = CreateThread(nullptr, 0, copyAndPresent_presenterThread, param, 0, nullptr);
        }

        FFX_ASSERT(presenterThreadHandle != NULL);

        if (presenterThreadHandle != 0)
        {
            SetThreadPriority(presenterThreadHandle, THREAD_PRIORITY_HIGHEST);
            SetThreadDescription(presenterThreadHandle, L"AMD FSR Presenter Thread");

            SimpleMovingAverage<10, double> frameTime{};
            int64_t                         previousQpc = 0;

            while (!presenter->shutdown)
            {
                WaitForSingleObject(presenter->presentEvent, INFINITE);

                if (!presenter->shutdown)
                {
                    EnterCriticalSection(&presenter->scheduledFrameCriticalSection);

                    PacingData entry = presenter->scheduledInterpolations;
                    presenter->scheduledInterpolations.invalidate();

                    LeaveCriticalSection(&presenter->scheduledFrameCriticalSection);

                    waitForSemaphoreValue(presenter->device,
                                          presenter->interpolationSemaphore,
                                          entry.frames[PacingData::FrameType::Interpolated_1].interpolationCompletedSemaphoreValue);
                    SetEvent(presenter->interpolationEvent);  // unlocks the queuePresent method

                    int64_t currentQpc = 0;
                    QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&currentQpc));

                    const double deltaQpc = double(currentQpc - previousQpc) * (previousQpc > 0);
                    previousQpc           = currentQpc;

                    // reset pacing averaging if delta > 10 fps,
                    int64_t qpcFrequency;
                    QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&qpcFrequency));
                    const float fTimeoutInSeconds       = 0.1f;
                    double      deltaQpcResetThreashold = double(qpcFrequency * fTimeoutInSeconds);
                    if ((deltaQpc > deltaQpcResetThreashold) || presenter->resetTimer)
                    {
                        frameTime.reset();
                    }
                    else
                    {
                        frameTime.update(deltaQpc);
                    }

                    // set presentation time: reduce based on variance and subract safety margin so we don't lock on a framerate lower than necessary
                    int64_t       qpcSafetyMargin = int64_t(qpcFrequency * presenter->safetyMarginInSec);
                    const int64_t conservativeAvg = int64_t(frameTime.getAverage() * 0.5 - frameTime.getVariance() * presenter->varianceFactor);
                    const int64_t deltaToUse      = conservativeAvg > qpcSafetyMargin ? (conservativeAvg - qpcSafetyMargin) : 0;
                    entry.frames[PacingData::FrameType::Interpolated_1].presentQpcDelta = deltaToUse;
                    entry.frames[PacingData::FrameType::Real].presentQpcDelta           = deltaToUse;

                    // schedule presents
                    EnterCriticalSection(&presenter->scheduledFrameCriticalSection);
                    presenter->scheduledPresents = entry;
                    LeaveCriticalSection(&presenter->scheduledFrameCriticalSection);
                    SetEvent(presenter->pacerEvent);
                }
            }

            // signal event to allow thread to finish
            SetEvent(presenter->pacerEvent);
            WaitForSingleObject(presenterThreadHandle, INFINITE);
            SafeCloseHandle(presenterThreadHandle);
        }
    }

    return 0;
}
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
void* interpolationThread(void* param)
{
    FrameinterpolationPresentInfo* presenter = static_cast<FrameinterpolationPresentInfo*>(param);

    if (presenter)
    {
        pthread_t presenterThreadHandle = 0;

        if (presenter->compositionMode == FGSwapchainCompositionMode::eComposeOnPresentQueue)
        {
            pthread_create(&presenterThreadHandle, nullptr, composeAndPresent_presenterThread, param);
        }
        else if (presenter->compositionMode == FGSwapchainCompositionMode::eComposeOnGameQueue)
        {
            pthread_create(&presenterThreadHandle, nullptr, copyAndPresent_presenterThread, param);
        }

        FFX_ASSERT(presenterThreadHandle != 0);

        if (presenterThreadHandle != 0)
        {
            struct sched_param sch_params;
            sch_params.sched_priority = sched_get_priority_max(SCHED_FIFO);
            pthread_setschedparam(presenterThreadHandle, SCHED_FIFO, &sch_params);
            pthread_setname_np(presenterThreadHandle, "Presenter");

            SimpleMovingAverage<10, double> frameTime{};
            int64_t previousQpc = 0;

            while (!presenter->shutdown)
            {
                pthread_mutex_lock(&presenter->presentMutex);
                while (!presenter->presentEventSignaled)
                    pthread_cond_wait(&presenter->presentCond, &presenter->presentMutex);
                presenter->presentEventSignaled = false;
                pthread_mutex_unlock(&presenter->presentMutex);

                if (!presenter->shutdown)
                {
                    pthread_mutex_lock(&presenter->scheduledFrameMutex);

                    PacingData entry = presenter->scheduledInterpolations;
                    presenter->scheduledInterpolations.invalidate();

                    pthread_mutex_unlock(&presenter->scheduledFrameMutex);

                    waitForSemaphoreValue(presenter->device,
                                          presenter->interpolationSemaphore,
                                          entry.frames[PacingData::FrameType::Interpolated_1].interpolationCompletedSemaphoreValue);

                    pthread_mutex_lock(&presenter->interpolationMutex);
                    presenter->interpolationEventSignaled = true;
                    pthread_cond_signal(&presenter->interpolationCond);
                    pthread_mutex_unlock(&presenter->interpolationMutex);

                    struct timespec ts;
                    clock_gettime(CLOCK_MONOTONIC, &ts);
                    int64_t currentQpc = ts.tv_sec * 1000000000LL + ts.tv_nsec;

                    const double deltaQpc = double(currentQpc - previousQpc) * (previousQpc > 0);
                    previousQpc = currentQpc;

                    // reset pacing averaging if delta > 10 fps,
                    int64_t qpcFrequency = 1000000000LL;  // 1 second in nanoseconds

                    const float fTimeoutInSeconds = 0.1f;
                    double deltaQpcResetThreashold = double(qpcFrequency * fTimeoutInSeconds);
                    if ((deltaQpc > deltaQpcResetThreashold) || presenter->resetTimer)
                    {
                        frameTime.reset();
                    }
                    else
                    {
                        frameTime.update(deltaQpc);
                    }

                    // set presentation time: reduce based on variance and subract safety margin so we don't lock on a framerate lower than necessary
                    int64_t qpcSafetyMargin = int64_t(qpcFrequency * presenter->safetyMarginInSec);
                    const int64_t conservativeAvg = int64_t(frameTime.getAverage() * 0.5 - frameTime.getVariance() * presenter->varianceFactor);
                    const int64_t deltaToUse = conservativeAvg > qpcSafetyMargin ? (conservativeAvg - qpcSafetyMargin) : 0;
                    entry.frames[PacingData::FrameType::Interpolated_1].presentQpcDelta = deltaToUse;
                    entry.frames[PacingData::FrameType::Real].presentQpcDelta = deltaToUse;
                    // schedule presents
                    pthread_mutex_lock(&presenter->scheduledFrameMutex);
                    presenter->scheduledPresents = entry;
                    pthread_mutex_unlock(&presenter->scheduledFrameMutex);

                    pthread_mutex_lock(&presenter->pacerMutex);
                    presenter->pacerEventSignaled = true;
                    pthread_cond_signal(&presenter->pacerCond);
                    pthread_mutex_unlock(&presenter->pacerMutex);
                }
            }

            // signal event to allow thread to finish
            pthread_mutex_lock(&presenter->pacerMutex);
            presenter->pacerEventSignaled = true;
            pthread_cond_signal(&presenter->pacerCond);
            pthread_mutex_unlock(&presenter->pacerMutex);

            pthread_join(presenterThreadHandle, nullptr);
        }
    }

    return 0;
}
#endif
//////////////////////////////////////////////
/// FrameInterpolationSwapChainVK
//////////////////////////////////////////////

FrameInterpolationSwapChainVK::FrameInterpolationSwapChainVK()
{
    for (uint32_t i = 0; i < FFX_FRAME_INTERPOLATION_SWAP_CHAIN_MAX_BUFFER_COUNT; ++i)
    {
        presentInfo.realSwapchainImages[i]     = VK_NULL_HANDLE;
        presentInfo.frameRenderedSemaphores[i] = VK_NULL_HANDLE;
    }
    for (uint32_t i = 0; i < FFX_FRAME_INTERPOLATION_SWAP_CHAIN_MAX_ACQUIRE_SEMAPHORE_COUNT; ++i)
    {
        presentInfo.acquireSemaphores[i] = VK_NULL_HANDLE;
    }
#if defined(FFX_PLATFORM_ANDROID) || defined(FFX_PLATFORM_LINUX)
    pthread_mutex_init(&presentInfo.presentMutex, nullptr);
    pthread_cond_init(&presentInfo.presentCond, nullptr);
    pthread_mutex_init(&presentInfo.interpolationMutex, nullptr);
    pthread_cond_init(&presentInfo.interpolationCond, nullptr);
    pthread_mutex_init(&presentInfo.scheduledFrameMutex, nullptr);
    pthread_mutex_init(&presentInfo.swapchainMutex, nullptr);
    pthread_mutex_init(&presentInfo.pacerMutex, nullptr);
    pthread_cond_init(&presentInfo.pacerCond, nullptr);
    presentInfo.presentEventSignaled       = false;
    presentInfo.interpolationEventSignaled = false;
    presentInfo.pacerEventSignaled         = false;
#endif
}

FrameInterpolationSwapChainVK::~FrameInterpolationSwapChainVK()
{
}

VkResult FrameInterpolationSwapChainVK::acquireNextImage(
    VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore, VkFence fence, uint32_t* pImageIndex)
{
    // this should always be the case
    if (semaphore == VK_NULL_HANDLE && fence == VK_NULL_HANDLE)
        return VK_INCOMPLETE;
    if (pImageIndex == nullptr)
        return VK_INCOMPLETE;

#ifdef FFX_PLATFORM_WINDOWS
    EnterCriticalSection(&criticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_lock(&criticalSection);
#endif

    *pImageIndex = (uint32_t)(acquiredCount % gameBufferCount);

    if (replacementSwapBuffers[*pImageIndex].image == VK_NULL_HANDLE)
    {
#ifdef FFX_PLATFORM_WINDOWS
        LeaveCriticalSection(&criticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
        pthread_mutex_unlock(&criticalSection);
#endif
        return VK_ERROR_SURFACE_LOST_KHR;
    }

    // limit the acquired count
    if (acquiredCount > presentCount && (acquiredCount - presentCount) >= gameBufferCount)
    {
#ifdef FFX_PLATFORM_WINDOWS
        LeaveCriticalSection(&criticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
        pthread_mutex_unlock(&criticalSection);
#endif
        return VK_NOT_READY;
    }

    SubmissionSemaphores toWait;
    SubmissionSemaphores toSignal;

    toWait.add(presentInfo.replacementBufferSemaphore, replacementSwapBuffers[*pImageIndex].availabilitySemaphoreValue);

    if (semaphore != VK_NULL_HANDLE)
        toSignal.add(semaphore);  // not a timeline semaphore

    VkResult res = imageAcquireQueue.submit(VK_NULL_HANDLE, toWait, toSignal, fence);

    ++acquiredCount;
#ifdef FFX_PLATFORM_WINDOWS
    LeaveCriticalSection(&criticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_unlock(&criticalSection);
#endif
    return res;
}

// structure holding all the structures that can extend VkSwapchainCreateInfoKHR
struct SwapchainCreationInfo
{
    VkSwapchainCreateInfoKHR swapchain;

    VkImageCompressionControlEXT imageCompressionControl;
    VkImageFormatListCreateInfo  imageFormatList;
#ifdef FFX_PLATFORM_WINDOWS
    VkSurfaceFullScreenExclusiveInfoEXT      surfaceFullScreenExclusive;
    VkSurfaceFullScreenExclusiveWin32InfoEXT surfaceFullScreenExclusiveWin32;
#endif
    VkSwapchainCounterCreateInfoEXT          swapchainCounter;
    VkSwapchainDisplayNativeHdrCreateInfoAMD swapchainDisplayNativeHdr;
    VkSwapchainPresentModesCreateInfoEXT     swapchainPresentModes;
};

#define FFX_USE_PNEXT_AS_IS(name, type)                                                 \
    realSwapchainCreateInfo.name            = *reinterpret_cast<const type*>(pCurrent); \
    realSwapchainCreateInfo.name.pNext      = realSwapchainCreateInfo.swapchain.pNext;  \
    realSwapchainCreateInfo.swapchain.pNext = &realSwapchainCreateInfo.name;

VkResult getRealSwapchainCreateInfo(const VkSwapchainCreateInfoKHR* pCreateInfo, SwapchainCreationInfo& realSwapchainCreateInfo)
{
    // fix VkSwapchainCreateInfoKHR
    realSwapchainCreateInfo.swapchain       = *pCreateInfo;
    realSwapchainCreateInfo.swapchain.pNext = nullptr;

    realSwapchainCreateInfo.swapchain.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    realSwapchainCreateInfo.swapchain.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
    realSwapchainCreateInfo.swapchain.queueFamilyIndexCount = 0;
    realSwapchainCreateInfo.swapchain.pQueueFamilyIndices   = nullptr;

    const VkBaseInStructure* pCurrent = reinterpret_cast<const VkBaseInStructure*>(pCreateInfo);
    while (pCurrent->pNext != nullptr)
    {
        pCurrent = pCurrent->pNext;
        switch (pCurrent->sType)
        {
        case VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT:
            FFX_USE_PNEXT_AS_IS(imageCompressionControl, VkImageCompressionControlEXT);
            break;
        case VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO:
            FFX_USE_PNEXT_AS_IS(imageFormatList, VkImageFormatListCreateInfo);
            break;
        case VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT:
#ifdef FFX_PLATFORM_WINDOWS
            realSwapchainCreateInfo.surfaceFullScreenExclusive = *reinterpret_cast<const VkSurfaceFullScreenExclusiveInfoEXT*>(pCurrent);
            realSwapchainCreateInfo.surfaceFullScreenExclusive.pNext =
                const_cast<void*>(realSwapchainCreateInfo.swapchain.pNext);  // because pNext is void* instead of const void* in vulkan header
            realSwapchainCreateInfo.swapchain.pNext = &realSwapchainCreateInfo.surfaceFullScreenExclusive;
#endif
            break;
        case VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_WIN32_INFO_EXT:
#ifdef FFX_PLATFORM_WINDOWS
            FFX_USE_PNEXT_AS_IS(surfaceFullScreenExclusiveWin32, VkSurfaceFullScreenExclusiveWin32InfoEXT);
#endif
            break;
        case VK_STRUCTURE_TYPE_SWAPCHAIN_COUNTER_CREATE_INFO_EXT:
            FFX_USE_PNEXT_AS_IS(swapchainCounter, VkSwapchainCounterCreateInfoEXT);
            break;
        case VK_STRUCTURE_TYPE_SWAPCHAIN_DISPLAY_NATIVE_HDR_CREATE_INFO_AMD:
            FFX_USE_PNEXT_AS_IS(swapchainDisplayNativeHdr, VkSwapchainDisplayNativeHdrCreateInfoAMD);
            break;
        case VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT:
            FFX_USE_PNEXT_AS_IS(swapchainPresentModes, VkSwapchainPresentModesCreateInfoEXT);
            break;
        default:
            FFX_ASSERT_MESSAGE_FORMAT(false, "Swapchain creation structure extension %d not supported", pCurrent->sType);
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }

    return VK_SUCCESS;
}

VkResult FrameInterpolationSwapChainVK::init(const VkSwapchainCreateInfoKHR* pCreateInfo, const VkFrameInterpolationInfoFFX* pFrameInterpolationInfo)
{
    if (pFrameInterpolationInfo == nullptr || pCreateInfo == nullptr)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (pFrameInterpolationInfo->device == VK_NULL_HANDLE || pFrameInterpolationInfo->physicalDevice == VK_NULL_HANDLE)
        return VK_ERROR_INITIALIZATION_FAILED;

    VkDevice device                         = pFrameInterpolationInfo->device;
    presentInfo.device                      = device;
    presentInfo.physicalDevice              = pFrameInterpolationInfo->physicalDevice;
    const VkAllocationCallbacks* pAllocator = pFrameInterpolationInfo->pAllocator;

    DebugNameSetter debugNameSetter(device);

    // create semaphores
    VkSemaphoreTypeCreateInfo semaphoreTypeCreateInfo = {};
    semaphoreTypeCreateInfo.sType                     = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    semaphoreTypeCreateInfo.pNext                     = nullptr;
    semaphoreTypeCreateInfo.semaphoreType             = VK_SEMAPHORE_TYPE_TIMELINE;
    semaphoreTypeCreateInfo.initialValue              = 0;

    VkSemaphoreCreateInfo semaphoreCreateInfo = {};
    semaphoreCreateInfo.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreCreateInfo.pNext                 = &semaphoreTypeCreateInfo;
    semaphoreCreateInfo.flags                 = 0;

    auto createSemaphore = [device, &semaphoreCreateInfo, &debugNameSetter](
                               VkSemaphore& semaphore, const char* name, const VkAllocationCallbacks* pAllocator = nullptr) {
        VkResult res = VulkanWrapper().vkCreateSemaphore(device, &semaphoreCreateInfo, pAllocator, &semaphore);

        if (res == VK_SUCCESS)
            debugNameSetter.setDebugName(semaphore, VK_OBJECT_TYPE_SEMAPHORE, name);  // it's fine if this fails

        return res;
    };

    auto createSemaphoreFormat = [device, &semaphoreCreateInfo, &debugNameSetter](
                                     VkSemaphore& semaphore, const char* name, uint32_t i, const VkAllocationCallbacks* pAllocator = nullptr) {
        VkResult res = VulkanWrapper().vkCreateSemaphore(device, &semaphoreCreateInfo, pAllocator, &semaphore);

        if (res == VK_SUCCESS)
            debugNameSetter.setDebugName(semaphore, VK_OBJECT_TYPE_SEMAPHORE, name, i);  // it's fine if this fails

        return res;
    };

    VkResult res = VK_SUCCESS;
    res          = createSemaphore(presentInfo.gameSemaphore, "AMD FSR GameSemaphore");
    EXIT_ON_VKRESULT_NOT_SUCCESS(res);
    res = createSemaphore(presentInfo.interpolationSemaphore, "AMD FSR InterpolationSemaphore");
    EXIT_ON_VKRESULT_NOT_SUCCESS(res);
    res = createSemaphore(presentInfo.presentSemaphore, "AMD FSR PresentSemaphore");
    EXIT_ON_VKRESULT_NOT_SUCCESS(res);
    res = createSemaphore(presentInfo.replacementBufferSemaphore, "AMD FSR ReplacementBufferSemaphore");
    EXIT_ON_VKRESULT_NOT_SUCCESS(res);
    res = createSemaphore(presentInfo.compositionSemaphore, "AMD FSR CompositionSemaphore");
    EXIT_ON_VKRESULT_NOT_SUCCESS(res);

    // create non-timeline semaphores
    semaphoreCreateInfo.pNext = nullptr;

    // semaphores used for present to wait until rendering is over
    for (uint32_t i = 0; i < FFX_FRAME_INTERPOLATION_SWAP_CHAIN_MAX_BUFFER_COUNT; ++i)
    {
        res = createSemaphoreFormat(presentInfo.frameRenderedSemaphores[i], "AMD FSR FrameRenderedSemaphore %d", i);
        EXIT_ON_VKRESULT_NOT_SUCCESS(res);
    }
    for (uint32_t i = 0; i < FFX_FRAME_INTERPOLATION_SWAP_CHAIN_MAX_ACQUIRE_SEMAPHORE_COUNT; ++i)
    {
        res = createSemaphoreFormat(presentInfo.acquireSemaphores[i], "AMD FSR AcquireSemaphore %d", i);
        EXIT_ON_VKRESULT_NOT_SUCCESS(res);
    }

    // queues
    presentInfo.gameQueue          = pFrameInterpolationInfo->gameQueue;
    presentInfo.interpolationQueue = presentInfo.gameQueue;
    presentInfo.asyncComputeQueue  = pFrameInterpolationInfo->asyncComputeQueue;
    presentInfo.presentQueue       = pFrameInterpolationInfo->presentQueue;
    imageAcquireQueue              = pFrameInterpolationInfo->imageAcquireQueue;

    // TODO: for now we don't support shared queues
    // check that the queues are separate
    // if (presentInfo.gameQueue.queue == presentInfo.asyncComputeQueue.queue || presentInfo.gameQueue.queue == presentInfo.presentQueue.queue ||
    //     presentInfo.gameQueue.queue == imageAcquireQueue.queue || presentInfo.asyncComputeQueue.queue == presentInfo.presentQueue.queue ||
    //     presentInfo.asyncComputeQueue.queue == imageAcquireQueue.queue || presentInfo.presentQueue.queue == imageAcquireQueue.queue)
    // {
    //     return VK_ERROR_INITIALIZATION_FAILED;
    // }

    if (presentInfo.presentQueue.queue != presentInfo.gameQueue.queue)
        debugNameSetter.setDebugName(presentInfo.presentQueue.queue, VK_OBJECT_TYPE_QUEUE, "AMD FSR PresentQueue");
    debugNameSetter.setDebugName(presentInfo.asyncComputeQueue.queue, VK_OBJECT_TYPE_QUEUE, "AMD FSR AsyncComputeQueue");
    debugNameSetter.setDebugName(imageAcquireQueue.queue, VK_OBJECT_TYPE_QUEUE, "AMD FSR ImageAcquireQueue");

    // check the mode
    const uint32_t cMaxQueueFamilyCount = 16;
    uint32_t       queueFamilyCount     = 0;
    VulkanWrapper().vkGetPhysicalDeviceQueueFamilyProperties(presentInfo.physicalDevice, &queueFamilyCount, nullptr);

    uint32_t maxFamilyIndex = presentInfo.presentQueue.familyIndex;
    maxFamilyIndex          = maxFamilyIndex > presentInfo.gameQueue.familyIndex ? maxFamilyIndex : presentInfo.gameQueue.familyIndex;
    maxFamilyIndex          = maxFamilyIndex > presentInfo.asyncComputeQueue.familyIndex ? maxFamilyIndex : presentInfo.asyncComputeQueue.familyIndex;
    maxFamilyIndex          = maxFamilyIndex > imageAcquireQueue.familyIndex ? maxFamilyIndex : imageAcquireQueue.familyIndex;

    if (maxFamilyIndex >= queueFamilyCount || queueFamilyCount > cMaxQueueFamilyCount)
        return VK_ERROR_INITIALIZATION_FAILED;

    VkQueueFamilyProperties properties[cMaxQueueFamilyCount];
    queueFamilyCount = maxFamilyIndex + 1;
    VulkanWrapper().vkGetPhysicalDeviceQueueFamilyProperties(presentInfo.physicalDevice, &queueFamilyCount, properties);

    if (presentInfo.presentQueue.familyIndex >= queueFamilyCount)
        return VK_ERROR_INITIALIZATION_FAILED;

    const VkQueueFamilyProperties presentQueueProperties = properties[presentInfo.presentQueue.familyIndex];
    VkBool32                      supportsPresent        = VK_FALSE;
    res                                                  = VulkanWrapper().vkGetPhysicalDeviceSurfaceSupportKHR(
        presentInfo.physicalDevice, presentInfo.presentQueue.familyIndex, pCreateInfo->surface, &supportsPresent);
    EXIT_ON_VKRESULT_NOT_SUCCESS(res);

    if (supportsPresent == VK_FALSE)
        return VK_ERROR_INITIALIZATION_FAILED;

    if (HAS_FLAG(presentQueueProperties.queueFlags, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
    {
        presentInfo.compositionMode = FGSwapchainCompositionMode::eComposeOnPresentQueue;
    }
    else if (HAS_FLAG(presentQueueProperties.queueFlags, VK_QUEUE_GRAPHICS_BIT) || HAS_FLAG(presentQueueProperties.queueFlags, VK_QUEUE_COMPUTE_BIT) ||
             HAS_FLAG(presentQueueProperties.queueFlags, VK_QUEUE_TRANSFER_BIT))
    {
        presentInfo.compositionMode = FGSwapchainCompositionMode::eComposeOnGameQueue;
    }
    else
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // force if there is an override
    switch (pFrameInterpolationInfo->compositionMode)
    {
    case VK_COMPOSITION_MODE_GAME_QUEUE_FFX:
        presentInfo.compositionMode = FGSwapchainCompositionMode::eComposeOnGameQueue;
        break;
    case VK_COMPOSITION_MODE_PRESENT_QUEUE_FFX:
        presentInfo.compositionMode = FGSwapchainCompositionMode::eComposeOnPresentQueue;
        break;
    default:
        break;
    }

    // check other queues
    const VkQueueFamilyProperties gameQueueProperties = properties[presentInfo.gameQueue.familyIndex];
    if (!HAS_FLAG(gameQueueProperties.queueFlags, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
        return VK_ERROR_INITIALIZATION_FAILED;
    if (presentInfo.asyncComputeQueue.queue != VK_NULL_HANDLE)
    {
        const VkQueueFamilyProperties asyncComputeQueueProperties = properties[presentInfo.asyncComputeQueue.familyIndex];
        if (!HAS_FLAG(asyncComputeQueueProperties.queueFlags, VK_QUEUE_COMPUTE_BIT))
            return VK_ERROR_INITIALIZATION_FAILED;
    }

#ifdef FFX_PLATFORM_WINDOWS
    InitializeCriticalSection(&criticalSection);
    InitializeCriticalSection(&criticalSectionUpdateConfig);
    InitializeCriticalSection(&presentInfo.scheduledFrameCriticalSection);
    InitializeCriticalSection(&presentInfo.swapchainCriticalSection);

    presentInfo.presentEvent       = CreateEvent(NULL, FALSE, FALSE, TEXT("PresentEvent"));
    presentInfo.interpolationEvent = CreateEvent(NULL, FALSE, TRUE, TEXT("InterpolationEvent"));
    presentInfo.pacerEvent         = CreateEvent(NULL, FALSE, FALSE, TEXT("PacerEvent"));
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_init(&criticalSection, nullptr);
    pthread_mutex_init(&criticalSectionUpdateConfig, nullptr);
    pthread_mutex_init(&presentInfo.scheduledFrameMutex, nullptr);
    pthread_mutex_init(&presentInfo.swapchainMutex, nullptr);
    pthread_mutex_init(&presentInfo.presentMutex, NULL);
    pthread_mutex_init(&presentInfo.interpolationMutex, NULL);
    pthread_mutex_init(&presentInfo.pacerMutex, NULL);
#endif

    // create the real swapchain
    SwapchainCreationInfo realSwapchainCreateInfo;
    res = getRealSwapchainCreateInfo(pCreateInfo, realSwapchainCreateInfo);
    EXIT_ON_VKRESULT_NOT_SUCCESS(res);

    res = VulkanWrapper().vkCreateSwapchainKHR(device, &realSwapchainCreateInfo.swapchain, pAllocator, &presentInfo.realSwapchain);
    EXIT_ON_VKRESULT_NOT_SUCCESS(res);

    presentMode      = realSwapchainCreateInfo.swapchain.presentMode;
    tearingSupported = (presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR || presentMode == VK_PRESENT_MODE_FIFO_RELAXED_KHR);

    // get all the images of the swapchain
    presentInfo.realSwapchainImageCount = 0;
    res = VulkanWrapper().vkGetSwapchainImagesKHR(device, presentInfo.realSwapchain, &presentInfo.realSwapchainImageCount, nullptr);
    if (res != VK_SUCCESS || presentInfo.realSwapchainImageCount > FFX_FRAME_INTERPOLATION_SWAP_CHAIN_MAX_BUFFER_COUNT)
        return VK_ERROR_INITIALIZATION_FAILED;
    res = VulkanWrapper().vkGetSwapchainImagesKHR(device, presentInfo.realSwapchain, &presentInfo.realSwapchainImageCount, presentInfo.realSwapchainImages);
    EXIT_ON_VKRESULT_NOT_SUCCESS(res);

    for (uint32_t i = 0; i < presentInfo.realSwapchainImageCount; ++i)
    {
        debugNameSetter.setDebugName(presentInfo.realSwapchainImages[i],
                                     VK_OBJECT_TYPE_IMAGE,
                                     "AMD FSR Real swapchain image %d",
                                     i);  // it's fine if this fails
    }

    presentInfo.realSwapchainImageDescription.type     = FFX_RESOURCE_TYPE_TEXTURE2D;
    presentInfo.realSwapchainImageDescription.format   = ffxGetSurfaceFormatVK(pCreateInfo->imageFormat);
    presentInfo.realSwapchainImageDescription.width    = pCreateInfo->imageExtent.width;
    presentInfo.realSwapchainImageDescription.height   = pCreateInfo->imageExtent.height;
    presentInfo.realSwapchainImageDescription.depth    = 1;
    presentInfo.realSwapchainImageDescription.mipCount = 1;
    presentInfo.realSwapchainImageDescription.flags    = FFX_RESOURCE_FLAGS_NONE;
    presentInfo.realSwapchainImageDescription.usage    = static_cast<FfxResourceUsage>(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV);

    // create the semaphore for each image to wait on
    for (uint32_t i = 0; i < presentInfo.realSwapchainImageCount; ++i)
    {
        res = createSemaphoreFormat(presentInfo.frameRenderedSemaphores[i], "AMD swapchain image semaphore %d", i);
        EXIT_ON_VKRESULT_NOT_SUCCESS(res);
    }

    // create the fake backbuffer images
    FFX_ASSERT_MESSAGE(pCreateInfo->minImageCount <= FFX_FRAME_INTERPOLATION_SWAP_CHAIN_MAX_BUFFER_COUNT, "Too many backbuffer requested");
    gameBufferCount = pCreateInfo->minImageCount;

    VkImageCreateInfo info = {};
    info.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.pNext             = nullptr;
    info.flags             = 0;
    info.imageType         = VK_IMAGE_TYPE_2D;
    info.format            = pCreateInfo->imageFormat;
    info.extent.width      = pCreateInfo->imageExtent.width;
    info.extent.height     = pCreateInfo->imageExtent.height;
    info.extent.depth      = 1;
    info.mipLevels         = 1;
    info.arrayLayers       = 1;
    info.samples           = VK_SAMPLE_COUNT_1_BIT;
    info.tiling            = VK_IMAGE_TILING_OPTIMAL;
    info.usage             = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;  // queue family ownership transfer will be handled manually

    VkPhysicalDeviceMemoryProperties memProperties;
    VulkanWrapper().vkGetPhysicalDeviceMemoryProperties(presentInfo.physicalDevice, &memProperties);

    for (uint32_t i = 0; i < gameBufferCount; ++i)
    {
        res = createImage(replacementSwapBuffers[i],
                          info,
                          presentInfo.realSwapchainImageDescription.format,
                          "AMD FSR Replacement BackBuffer %d",
                          i,
                          memProperties,
                          pAllocator);
        EXIT_ON_VKRESULT_NOT_SUCCESS(res);
    }
    // No need to set the layout of the images as vulkan API states that a presentable image starts out at an undefined layout.

    for (uint32_t i = 0; i < FFX_COUNTOF(interpolationOutputs); ++i)
    {
        res = createImage(
            interpolationOutputs[i], info, presentInfo.realSwapchainImageDescription.format, "AMD FSR Interpolation Output %d", i, memProperties, pAllocator);
        EXIT_ON_VKRESULT_NOT_SUCCESS(res);
    }

    if (presentInfo.compositionMode == FGSwapchainCompositionMode::eComposeOnGameQueue)
    {
        res = createImage(
            presentInfo.compositionOutput, info, presentInfo.realSwapchainImageDescription.format, "AMD FSR Composition Output", memProperties, pAllocator);
        EXIT_ON_VKRESULT_NOT_SUCCESS(res);
    }

    GET_DEVICE_PROC_ADDR(vkSetHdrMetadataEXT);

    return res;
}

void FrameInterpolationSwapChainVK::destroySwapchain(VkDevice device, const VkAllocationCallbacks* pAllocator)
{
    // Wait for all operations to be finished
    waitForPresents();
    killPresenterThread();

    FFX_ASSERT_MESSAGE(device == presentInfo.device, "Attempting to destroy Frame interpolation swapchain with another device");

    if (device == VK_NULL_HANDLE)
        return;

    auto destroySemaphore = [device, pAllocator](VkSemaphore& semaphore) {
        VulkanWrapper().vkDestroySemaphore(device, semaphore, pAllocator);
        semaphore = VK_NULL_HANDLE;
    };

    // destroy semaphores
    destroySemaphore(presentInfo.gameSemaphore);
    destroySemaphore(presentInfo.interpolationSemaphore);
    destroySemaphore(presentInfo.presentSemaphore);
    destroySemaphore(presentInfo.replacementBufferSemaphore);
    destroySemaphore(presentInfo.compositionSemaphore);

    for (uint32_t i = 0; i < FFX_FRAME_INTERPOLATION_SWAP_CHAIN_MAX_ACQUIRE_SEMAPHORE_COUNT; ++i)
    {
        destroySemaphore(presentInfo.acquireSemaphores[i]);
    }

    // reset queue information
    presentInfo.gameQueue.reset();
    presentInfo.interpolationQueue.reset();
    presentInfo.asyncComputeQueue.reset();
    presentInfo.presentQueue.reset();

#ifdef FFX_PLATFORM_WINDOWS
    // delete win32 objects
    DeleteCriticalSection(&criticalSection);
    DeleteCriticalSection(&criticalSectionUpdateConfig);
    DeleteCriticalSection(&presentInfo.scheduledFrameCriticalSection);
    DeleteCriticalSection(&presentInfo.swapchainCriticalSection);

    SafeCloseHandle(presentInfo.presentEvent);
    SafeCloseHandle(presentInfo.interpolationEvent);
    SafeCloseHandle(presentInfo.pacerEvent);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    // delete android/linux objects
    pthread_mutex_destroy(&criticalSection);
    pthread_mutex_destroy(&criticalSectionUpdateConfig);
    pthread_mutex_destroy(&presentInfo.scheduledFrameMutex);
    pthread_mutex_destroy(&presentInfo.swapchainMutex);
    pthread_mutex_destroy(&presentInfo.presentMutex);
    pthread_mutex_destroy(&presentInfo.interpolationMutex);
    pthread_mutex_destroy(&presentInfo.pacerMutex);
#endif

    // delete real swapchain
    VulkanWrapper().vkDestroySwapchainKHR(device, presentInfo.realSwapchain, pAllocator);
    presentInfo.realSwapchain = VK_NULL_HANDLE;

    // destroy semaphores of each image
    for (uint32_t i = 0; i < FFX_FRAME_INTERPOLATION_SWAP_CHAIN_MAX_BUFFER_COUNT; ++i)
    {
        presentInfo.realSwapchainImages[i] = VK_NULL_HANDLE;
        destroySemaphore(presentInfo.frameRenderedSemaphores[i]);
    }

    // destroy replacement buffers & interpolation outputs
    for (uint32_t i = 0; i < FFX_FRAME_INTERPOLATION_SWAP_CHAIN_MAX_BUFFER_COUNT; ++i)
    {
        destroyImage(replacementSwapBuffers[i], pAllocator);
    }

    for (uint32_t i = 0; i < FFX_COUNTOF(interpolationOutputs); ++i)
    {
        destroyImage(interpolationOutputs[i], pAllocator);
    }

    destroyImage(presentInfo.compositionOutput, pAllocator);

    destroyImage(uiReplacementBuffer, pAllocator);
}

VkResult FrameInterpolationSwapChainVK::getSwapchainImages(VkDevice device, uint32_t* pSwapchainImageCount, VkImage* pSwapchainImages)
{
    FFX_ASSERT_MESSAGE(device == presentInfo.device, "Attempting to get frame interpolation swapchain images with another device");

    if (pSwapchainImages == nullptr)
    {
        *pSwapchainImageCount = gameBufferCount;
    }
    else
    {
        for (uint32_t i = 0; i < gameBufferCount && i < *pSwapchainImageCount; ++i)
        {
            pSwapchainImages[i] = replacementSwapBuffers[i].image;
        }
    }
    return (gameBufferCount <= *pSwapchainImageCount) ? VK_SUCCESS : VK_INCOMPLETE;
}

void FrameInterpolationSwapChainVK::discardOutstandingInterpolationCommandLists()
{
    // drop any outstanding interpolaton command lists
    for (int i = 0; i < FFX_COUNTOF(registeredInterpolationCommandLists); i++)
    {
        if (registeredInterpolationCommandLists[i] != nullptr)
        {
            registeredInterpolationCommandLists[i]->drop();
            registeredInterpolationCommandLists[i] = nullptr;
        }
    }
}

void FrameInterpolationSwapChainVK::setFrameGenerationConfig(FfxFrameGenerationConfig const* config)
{
    FFX_ASSERT(config);
#ifdef FFX_PLATFORM_WINDOWS
    EnterCriticalSection(&criticalSectionUpdateConfig);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_lock(&criticalSectionUpdateConfig);
#endif

    // if config is a pointer to the internal config ::present called this function to apply the changes
    bool applyChangesNow = (config == &nextFrameGenerationConfig);

    FfxPresentCallbackFunc inputPresentCallback    = (nullptr != config->presentCallback) ? config->presentCallback : ffxFrameInterpolationUiComposition;
    void*                  inputPresentCallbackCtx = (nullptr != config->presentCallback) ? config->presentCallbackContext : nullptr;
    VulkanQueue            inputInterpolationQueue = config->allowAsyncWorkloads ? presentInfo.asyncComputeQueue : presentInfo.gameQueue;

    // if this is called externally just copy the new config to the internal copy to avoid potentially stalling on criticalSection
    if (!applyChangesNow)
    {
        nextFrameGenerationConfig = *config;

        if (presentInfo.compositionMode == FGSwapchainCompositionMode::eComposeOnGameQueue)
        {
            // since we compose on gameQueue anyway we should run interpolation on it too, to avoid semaphores
            nextFrameGenerationConfig.allowAsyncWorkloads = false;
            inputInterpolationQueue                       = presentInfo.gameQueue;
        }

        // in case of actual reconfiguration: apply the changes immediately
        if (presentInfo.interpolationQueue.queue != inputInterpolationQueue.queue || interpolationEnabled != config->frameGenerationEnabled ||
            presentCallback != inputPresentCallback || presentCallbackContext != inputPresentCallbackCtx ||
            frameGenerationCallback != config->frameGenerationCallback || frameGenerationCallbackContext != config->frameGenerationCallbackContext ||
            drawDebugPacingLines != config->drawDebugPacingLines)
        {
            applyChangesNow = true;
        }
    }

    if (applyChangesNow)
    {
#ifdef FFX_PLATFORM_WINDOWS
        EnterCriticalSection(&criticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
        pthread_mutex_lock(&criticalSection);
#endif

        currentFrameID          = config->frameID;
        presentInterpolatedOnly = config->onlyPresentInterpolated;
        drawDebugPacingLines    = config->drawDebugPacingLines;
        dumpGeneratedFrame      = config->dumpGeneratedFrame;
        dumpGeneratedFramePath  = config->dumpGeneratedFramePath;

        if (presentInfo.interpolationQueue.queue != inputInterpolationQueue.queue)
        {
            waitForPresents();
            discardOutstandingInterpolationCommandLists();

            // change interpolation queue
            presentInfo.interpolationQueue = inputInterpolationQueue;
        }

        if (interpolationEnabled != config->frameGenerationEnabled || presentCallback != inputPresentCallback ||
            frameGenerationCallback != config->frameGenerationCallback)
        {
            waitForPresents();
            presentCallback                = inputPresentCallback;
            presentCallbackContext         = inputPresentCallbackCtx;
            frameGenerationCallback        = config->frameGenerationCallback;
            frameGenerationCallbackContext = config->frameGenerationCallbackContext;

            // handle interpolation mode change
            if (interpolationEnabled != config->frameGenerationEnabled)
            {
                interpolationEnabled = config->frameGenerationEnabled;
                if (interpolationEnabled)
                {
                    frameInterpolationResetCondition = true;

                    spawnPresenterThread();
                }
                else
                {
                    killPresenterThread();
                }
            }
        }
#ifdef FFX_PLATFORM_WINDOWS
        LeaveCriticalSection(&criticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
        pthread_mutex_unlock(&criticalSection);
#endif
    }
#ifdef FFX_PLATFORM_WINDOWS
    LeaveCriticalSection(&criticalSectionUpdateConfig);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_unlock(&criticalSectionUpdateConfig);
#endif
}

bool FrameInterpolationSwapChainVK::waitForPresents()
{
    // wait for interpolation to finish
    bool waitRes = waitForSemaphoreValue(presentInfo.device, presentInfo.gameSemaphore, gameSemaphoreValue, UINT64_MAX, presentInfo.waitCallback);
    waitRes &= waitForSemaphoreValue(presentInfo.device, presentInfo.interpolationSemaphore, interpolationSemaphoreValue, UINT64_MAX, presentInfo.waitCallback);

    if (framesSentForPresentation > 0)
    {
        waitRes &= waitForSemaphoreValue(presentInfo.device, presentInfo.presentSemaphore, framesSentForPresentation, UINT64_MAX, presentInfo.waitCallback);
    }

    FFX_ASSERT(waitRes);

    // wait for queue idle too, to make sure all the semaphores are signaled
    VkResult res = VK_SUCCESS;
    if (res == VK_SUCCESS && presentInfo.asyncComputeQueue.queue != VK_NULL_HANDLE)
        res = VulkanWrapper().vkQueueWaitIdle(presentInfo.asyncComputeQueue.queue);
    if (res == VK_SUCCESS && presentInfo.presentQueue.queue != VK_NULL_HANDLE)
        res = VulkanWrapper().vkQueueWaitIdle(presentInfo.presentQueue.queue);
    if (res == VK_SUCCESS && presentInfo.gameQueue.queue != VK_NULL_HANDLE)
        res = VulkanWrapper().vkQueueWaitIdle(presentInfo.gameQueue.queue);
    if (res == VK_SUCCESS && imageAcquireQueue.queue != VK_NULL_HANDLE)
        res = VulkanWrapper().vkQueueWaitIdle(imageAcquireQueue.queue);

    if (framesSentForPresentation > 0)
    {
        if (res == VK_SUCCESS)
            res = VulkanWrapper().vkDeviceWaitIdle(presentInfo.device);
    }

    return res == VK_SUCCESS;
}

FfxResource FrameInterpolationSwapChainVK::interpolationOutput(int index)
{
    index = interpolationBufferIndex;

    FfxResourceDescription interpolateDesc = interpolationOutputs[index].description;
    return ffxGetResourceVK(interpolationOutputs[index].image, interpolateDesc, nullptr, FFX_RESOURCE_STATE_GENERIC_UAV);
}

void FrameInterpolationSwapChainVK::dispatchInterpolationCommands(uint32_t              currentBackBufferIndex,
                                                                  FfxResource*          pInterpolatedFrame,
                                                                  FfxResource*          pRealFrame,
                                                                  SubmissionSemaphores& semaphoresToWait)
{
    FFX_ASSERT(pInterpolatedFrame);
    FFX_ASSERT(pRealFrame);

    VkImage                currentBackBuffer = replacementSwapBuffers[currentBackBufferIndex].image;
    FfxResourceDescription gameFrameDesc     = replacementSwapBuffers[currentBackBufferIndex].description;
    FfxResource            backbuffer        = ffxGetResourceVK(currentBackBuffer, gameFrameDesc, nullptr, FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ);

    *pRealFrame = backbuffer;

    // interpolation queue must wait for output resource to become available
    if (presentInfo.compositionMode == FGSwapchainCompositionMode::eComposeOnPresentQueue)
    {
        // in those cases, interpolation output is available after composition
        semaphoresToWait.add(presentInfo.compositionSemaphore, interpolationOutputs[interpolationBufferIndex].availabilitySemaphoreValue);
    }
    else if (presentInfo.compositionMode == FGSwapchainCompositionMode::eComposeOnGameQueue)
    {
        // when there is a copy on the present queue from the present thread, the interpolation output will be available when the present semaphore is signaled
        semaphoresToWait.add(presentInfo.presentSemaphore, interpolationOutputs[interpolationBufferIndex].availabilitySemaphoreValue);
    }

    // check necessary transitions
    ImageBarrierHelper preInterpolationBarriers;
    if (presentInfo.gameQueue.familyIndex != presentInfo.interpolationQueue.familyIndex)
    {
        preInterpolationBarriers.add(currentBackBuffer,
                                     ReplacementBufferTransferState.accessMask,
                                     ReplacementBufferTransferState.accessMask,
                                     ReplacementBufferTransferState.layout,
                                     ReplacementBufferTransferState.layout,
                                     presentInfo.gameQueue.familyIndex,
                                     presentInfo.interpolationQueue.familyIndex);
    }

    // interpolation texture previous content doesn't matter here so we can use an undefined source layout
    preInterpolationBarriers.add(
        interpolationOutputs[interpolationBufferIndex].image, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    auto pRegisteredCommandList = registeredInterpolationCommandLists[currentBackBufferIndex];
    if (pRegisteredCommandList != nullptr)
    {
        SubmissionSemaphores semaphoresToSignal;

        if (preInterpolationBarriers.count > 0)
        {
            // we have some necessary transitions, but we need a new command list

            VkCommands* interpolationCommandList = presentInfo.commandPool.get(presentInfo.device, presentInfo.interpolationQueue, "interpolationCommandList");
            VkCommandBuffer interpolationCommandBuffer = interpolationCommandList->reset();

            preInterpolationBarriers.record(interpolationCommandBuffer);
            // there is nothing to signal
            VkResult res = interpolationCommandList->execute(semaphoresToWait, semaphoresToSignal);
            FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS, "barriers prior to dispatchInterpolationCommands failed with error %d", res);
        }

        semaphoresToSignal.add(presentInfo.interpolationSemaphore, ++interpolationSemaphoreValue);

        VkResult res = pRegisteredCommandList->execute(semaphoresToWait, semaphoresToSignal);
        FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS, "dispatchInterpolationCommands failed with error %d", res);

        frameInterpolationResetCondition = false;
        *pInterpolatedFrame              = interpolationOutput();
    }
    else
    {
        VkCommands*     interpolationCommandList = presentInfo.commandPool.get(presentInfo.device, presentInfo.interpolationQueue, "interpolationCommandList");
        VkCommandBuffer interpolationCommandBuffer = interpolationCommandList->reset();

        FfxResource interpolatedResource = interpolationOutput();
        VkImage     interpolatedImage    = static_cast<VkImage>(interpolatedResource.resource);

        FFX_ASSERT_MESSAGE(interpolationOutputs[interpolationBufferIndex].image == interpolatedImage, "Error when querying interpolation output");

        FfxFrameGenerationDispatchDescription desc{};
        desc.commandList           = ffxGetCommandListVK(interpolationCommandBuffer);
        desc.outputs[0]            = interpolatedResource;
        desc.presentColor          = backbuffer;
        desc.reset                 = frameInterpolationResetCondition;
        desc.numInterpolatedFrames = 1;
        desc.frameID               = currentFrameID;

        // execute the necessary transitions
        preInterpolationBarriers.record(interpolationCommandBuffer);

        if (frameGenerationCallback(&desc, frameGenerationCallbackContext) == FFX_OK)
        {
            SubmissionSemaphores semaphoresToSignal;

            ImageBarrierHelper postInterpolationBarriers;

            uint32_t compositionQueueFamily = 0xffffffff;
            if (presentInfo.compositionMode == FGSwapchainCompositionMode::eComposeOnPresentQueue)
            {
                compositionQueueFamily = presentInfo.presentQueue.familyIndex;
            }
            else if (presentInfo.compositionMode == FGSwapchainCompositionMode::eComposeOnPresentQueue)
            {
                compositionQueueFamily = presentInfo.gameQueue.familyIndex;
            }

            if (presentInfo.interpolationQueue.familyIndex != compositionQueueFamily)
            {
                // interpolation output queue family ownership transfer
                postInterpolationBarriers.add(interpolatedImage,
                                              InterpolationBufferTransferState.accessMask,
                                              InterpolationBufferTransferState.accessMask,
                                              InterpolationBufferTransferState.layout,
                                              InterpolationBufferTransferState.layout,
                                              presentInfo.interpolationQueue.familyIndex,
                                              compositionQueueFamily);

                // backbuffer queue family ownership transfer
                if (!presentInterpolatedOnly)
                {
                    postInterpolationBarriers.add(currentBackBuffer,
                                                  ReplacementBufferTransferState.accessMask,
                                                  ReplacementBufferTransferState.accessMask,
                                                  ReplacementBufferTransferState.layout,
                                                  ReplacementBufferTransferState.layout,
                                                  presentInfo.interpolationQueue.familyIndex,
                                                  compositionQueueFamily);
                }
                postInterpolationBarriers.record(interpolationCommandBuffer);
            }

            semaphoresToSignal.add(presentInfo.interpolationSemaphore, ++interpolationSemaphoreValue);

            VkResult res = interpolationCommandList->execute(semaphoresToWait, semaphoresToSignal);
            FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS, "dispatchInterpolationCommands failed with error %d", res);
        }
        else
        {
            interpolationCommandList->drop();
        }

        // reset condition if at least one frame was interpolated
        *pInterpolatedFrame = interpolationOutput();
        if (frameInterpolationResetCondition)
        {
            pInterpolatedFrame->resource = nullptr;
        }
        frameInterpolationResetCondition = false;
    }
}

VkImageMemoryBarrier FrameinterpolationPresentInfo::queueFamilyOwnershipTransferGameToPresent(const FfxResource& resource) const
{
    VkAccessFlags accessMask  = getVKAccessFlagsFromResourceState2(resource.state);
    VkImageLayout imageLayout = getVKImageLayoutFromResourceState2(resource.state);
    return getImageMemoryBarrier(
        static_cast<VkImage>(resource.resource), accessMask, accessMask, imageLayout, imageLayout, gameQueue.familyIndex, presentQueue.familyIndex);
}

VkResult FrameInterpolationSwapChainVK::presentInterpolated(const VkPresentInfoKHR* pPresentInfo, uint32_t currentBackBufferIndex, bool needUICopy)
{
    SubmissionSemaphores ToWaitGameQueue;

    SubmissionSemaphores toSignalGameQueue;
    toSignalGameQueue.add(presentInfo.gameSemaphore, ++gameSemaphoreValue);

    // wait on the provided semaphores on the game queue
    addPresentInfoSemaphores(pPresentInfo, ToWaitGameQueue);  // we need to wait on the semaphores the user provided

    // transfer the replacement buffer from game queue to interpolation queue
    const bool replacementBufferNeedsOwnershipTransfer = (presentInfo.gameQueue.familyIndex != presentInfo.interpolationQueue.familyIndex);
    bool       uiSurfaceNeedsOwnershipTransfer         = false;
    if (presentInfo.compositionMode == FGSwapchainCompositionMode::eComposeOnPresentQueue)
    {
        uiSurfaceNeedsOwnershipTransfer =
            (presentInfo.gameQueue.familyIndex != presentInfo.presentQueue.familyIndex) && (presentInfo.currentUiSurface.resource != VK_NULL_HANDLE);
    }

    if (replacementBufferNeedsOwnershipTransfer || uiSurfaceNeedsOwnershipTransfer || needUICopy)
    {
        VkCommands*     ownershipTransferCommandList = presentInfo.commandPool.get(presentInfo.device, presentInfo.gameQueue, "ownershipTranferCommandList");
        VkCommandBuffer ownershipCommandBuffer       = ownershipTransferCommandList->reset();

        ImageBarrierHelper imageBarriers;
        if (replacementBufferNeedsOwnershipTransfer)
        {
            // transfer the fake backbuffer from game queue to interpolation queue
            imageBarriers.add(replacementSwapBuffers[currentBackBufferIndex].image,
                              ReplacementBufferTransferState.accessMask,
                              ReplacementBufferTransferState.accessMask,
                              ReplacementBufferTransferState.layout,
                              ReplacementBufferTransferState.layout,
                              presentInfo.gameQueue.familyIndex,
                              presentInfo.interpolationQueue.familyIndex);
        }
        if (needUICopy)
        {
            // copy the UI texture and transfer the internal resource to the present queue
            copyUiResource(ownershipCommandBuffer, ToWaitGameQueue, uiSurfaceNeedsOwnershipTransfer);
        }
        else if (uiSurfaceNeedsOwnershipTransfer)
        {
            // transfer the UI texture from game queue to present queue
            imageBarriers.add(presentInfo.queueFamilyOwnershipTransferGameToPresent(presentInfo.currentUiSurface));
        }

        imageBarriers.record(ownershipCommandBuffer);

        VkResult res = ownershipTransferCommandList->execute(ToWaitGameQueue, toSignalGameQueue);
        FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS, "[presentInterpolated] queue family ownership transfer failed with error %d", res);
    }
    else
    {
        // add a signal on the game queue for the interpolation queue to wait on
        VkResult res = presentInfo.gameQueue.submit(VK_NULL_HANDLE, ToWaitGameQueue, toSignalGameQueue);
        FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS, "[presentInterpolated] game queue signal failed with error %d", res);
    }

    // interpolation should wait for the game semaphore
    SubmissionSemaphores ToWaitInterpolationQueue;
    ToWaitInterpolationQueue.add(presentInfo.gameSemaphore, gameSemaphoreValue);

    FfxResource interpolatedFrame{}, realFrame{};
    dispatchInterpolationCommands(currentBackBufferIndex, &interpolatedFrame, &realFrame, ToWaitInterpolationQueue);
#ifdef FFX_PLATFORM_WINDOWS
    EnterCriticalSection(&presentInfo.scheduledFrameCriticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_lock(&presentInfo.scheduledFrameMutex);
#endif

    PacingData entry{};
    entry.presentCallback        = presentCallback;
    entry.presentCallbackContext = presentCallbackContext;
    entry.drawDebugPacingLines   = drawDebugPacingLines;
    if (presentInfo.uiCompositionFlags & FFX_UI_COMPOSITION_FLAG_ENABLE_INTERNAL_UI_DOUBLE_BUFFERING)
    {
        entry.uiSurface = ffxGetResourceVK(uiReplacementBuffer.image, uiReplacementBuffer.description, nullptr, FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ);
    }
    else
    {
        entry.uiSurface = presentInfo.currentUiSurface;
    }
    entry.presentMode                      = presentMode;
    entry.numFramesSentForPresentationBase = framesSentForPresentation;
    entry.gameSemaphoreValue               = gameSemaphoreValue;
    entry.usePremulAlphaComposite          = (presentInfo.uiCompositionFlags & FFX_UI_COMPOSITION_FLAG_USE_PREMUL_ALPHA) != 0;
    entry.currentFrameID                   = currentFrameID;

    // interpolated
    PacingData::FrameInfo& fiInterpolated = entry.frames[PacingData::FrameType::Interpolated_1];
    if (interpolatedFrame.resource != nullptr)
    {
        fiInterpolated.doPresent                            = true;
        fiInterpolated.resource                             = interpolatedFrame;
        fiInterpolated.interpolationCompletedSemaphoreValue = interpolationSemaphoreValue;
        fiInterpolated.presentIndex                         = ++framesSentForPresentation;
    }

    if (dumpGeneratedFrame && dumpGeneratedFramePath != nullptr && fiInterpolated.doPresent)
    {
        std::string basePath(dumpGeneratedFramePath);
        uint64_t    frameID = 2 * currentFrameID - 1;

        dumpResourceAsyncNoCmdBuffer(
            presentInfo.device, presentInfo.physicalDevice, presentInfo.interpolationQueue.queue, interpolatedFrame, basePath.c_str(), frameID);
    }

    // real
    if (!presentInterpolatedOnly)
    {
        PacingData::FrameInfo& fiReal = entry.frames[PacingData::FrameType::Real];
        if (realFrame.resource != nullptr)
        {
            fiReal.doPresent                            = true;
            fiReal.resource                             = realFrame;
            fiReal.interpolationCompletedSemaphoreValue = interpolationSemaphoreValue;
            fiReal.presentIndex                         = ++framesSentForPresentation;
        }
    }

    entry.replacementBufferSemaphoreSignal = framesSentForPresentation;
    entry.numFramesToPresent               = static_cast<uint32_t>(framesSentForPresentation - entry.numFramesSentForPresentationBase);

    interpolationOutputs[interpolationBufferIndex].availabilitySemaphoreValue = entry.numFramesSentForPresentationBase + fiInterpolated.doPresent;

    if (presentInfo.compositionMode == FGSwapchainCompositionMode::eComposeOnGameQueue)
    {
        // launch composition now that PacingData has been set
        submitCompositionOnGameQueue(entry);
    }

    presentInfo.resetTimer              = frameInterpolationResetCondition;
    presentInfo.scheduledInterpolations = entry;
#ifdef FFX_PLATFORM_WINDOWS
    LeaveCriticalSection(&presentInfo.scheduledFrameCriticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_unlock(&presentInfo.scheduledFrameMutex);
#endif

    // Set event to kick off async CPU present thread
#ifdef FFX_PLATFORM_WINDOWS
    SetEvent(presentInfo.presentEvent);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_lock(&presentInfo.presentMutex);
    presentInfo.presentEventSignaled = true;
    pthread_cond_signal(&presentInfo.presentCond);
    pthread_mutex_unlock(&presentInfo.presentMutex);
#endif
    return presentInfo.lastPresentResult.load();
}

void FrameInterpolationSwapChainVK::registerUiResource(const FfxResource& uiResource, uint32_t flags)
{
#ifdef FFX_PLATFORM_WINDOWS
    EnterCriticalSection(&criticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_lock(&criticalSection);
#endif

    presentInfo.currentUiSurface   = uiResource;
    presentInfo.uiCompositionFlags = flags;
    if (nullptr == uiResource.resource)
        presentInfo.uiCompositionFlags &= ~FFX_UI_COMPOSITION_FLAG_ENABLE_INTERNAL_UI_DOUBLE_BUFFERING;
#ifdef FFX_PLATFORM_WINDOWS
    LeaveCriticalSection(&criticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_unlock(&criticalSection);
#endif
}

void FrameInterpolationSwapChainVK::setWaitCallback(FfxWaitCallbackFunc waitCallbackFunc)
{
    presentInfo.waitCallback = waitCallbackFunc;
}

void FrameInterpolationSwapChainVK::getGpuMemoryUsage(FfxEffectMemoryUsage* vramUsage)
{
    vramUsage->totalUsageInBytes     = totalUsageInBytes;
    vramUsage->aliasableUsageInBytes = aliasableUsageInBytes;
}

void FrameInterpolationSwapChainVK::setFramePacingTuning(const FfxSwapchainFramePacingTuning* framePacingTuning)
{
    presentInfo.safetyMarginInSec = static_cast<double>(framePacingTuning->safetyMarginInMs) / 1000.0;
    presentInfo.varianceFactor    = static_cast<double>(framePacingTuning->varianceFactor);
}

VkResult FrameInterpolationSwapChainVK::queuePresentNonInterpolated(VkCommands* pCommands, uint32_t imageIndex, SubmissionSemaphores& semaphoresToWait)
{
    SubmissionSemaphores semaphoresToSignal;
    semaphoresToSignal.add(presentInfo.replacementBufferSemaphore, ++framesSentForPresentation);
    semaphoresToSignal.add(presentInfo.frameRenderedSemaphores[imageIndex]);  // not a timeline semaphore

    semaphoresToSignal.add(presentInfo.presentSemaphore, framesSentForPresentation);
    presentInfo.lastPresentSemaphoreValue = framesSentForPresentation;

    pCommands->execute(semaphoresToWait, semaphoresToSignal);

    VkResult presentRes = presentToSwapChain(&presentInfo, imageIndex, imageIndex);
    FFX_ASSERT_MESSAGE_FORMAT(presentRes == VK_SUCCESS || presentRes == VK_SUBOPTIMAL_KHR || presentRes == VK_ERROR_OUT_OF_DATE_KHR,
                              "[queuePresentNonInterpolated] present failed with error %d",
                              presentRes);

    semaphoresToWait.add(presentInfo.presentSemaphore, framesSentForPresentation);
    VkResult res = presentInfo.gameQueue.submit(VK_NULL_HANDLE, semaphoresToWait, semaphoresToSignal);
    FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS, "[queuePresentNonInterpolated] game queue submit failed with error %d", res);

    return presentRes;
}

VkResult FrameInterpolationSwapChainVK::presentPassthrough(uint32_t              imageIndex,
                                                           SubmissionSemaphores& gameQueueWait,
                                                           SubmissionSemaphores& gameQueueSignal,
                                                           SubmissionSemaphores& presentQueueWait)
{
    VkImage srcImage = replacementSwapBuffers[replacementSwapBufferIndex].image;
    VkImage dstImage = presentInfo.realSwapchainImages[imageIndex];

    ImageBarrierHelper imageBarriers;

    // replacement buffer barrier
    VkImageMemoryBarrier srcBarrier = getImageMemoryBarrier(
        srcImage,
        ReplacementBufferTransferState.accessMask,
        VK_ACCESS_TRANSFER_READ_BIT,
        ReplacementBufferTransferState.layout,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        presentInfo.gameQueue
            .familyIndex,  // if game queue and present queue families are different, this will automatically be a queue family ownership transfer barrier)
        presentInfo.presentQueue.familyIndex);

    if (presentInfo.presentQueue.familyIndex != presentInfo.gameQueue.familyIndex)
    {
        // queue family ownership transfer for replacement buffer
        // srcBarrier needs to be executed on the game queue
        auto            ownershipList          = presentInfo.commandPool.get(presentInfo.device, presentInfo.gameQueue, "ownershipList");
        VkCommandBuffer ownershipCommandBuffer = ownershipList->reset();

        ImageBarrierHelper ownershipBarriers;
        ownershipBarriers.add(srcBarrier);
        ownershipBarriers.record(ownershipCommandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkResult res = ownershipList->execute(gameQueueWait, gameQueueSignal);
        FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS, "[presentPassthrough] queue submit failed with error %d", res);
    }
    else
    {
        // just wait/signal the semaphores
        VkResult res = presentInfo.gameQueue.submit(VK_NULL_HANDLE, gameQueueWait, gameQueueSignal);
        FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS, "[presentPassthrough] queue submit failed with error %d", res);
    }

    auto            passthroughList          = presentInfo.commandPool.get(presentInfo.device, presentInfo.presentQueue, "passthroughList");
    VkCommandBuffer passthroughCommandBuffer = passthroughList->reset();

    // real swapchain buffer (that was just acquried) barrier
    VkImageMemoryBarrier dstBarrier =
        getImageMemoryBarrier(dstImage, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // change layout for copy & queue family ownership transfer (if necessary)
    ImageBarrierHelper preCopyBarriers;
    if (presentInfo.presentQueue.familyIndex != presentInfo.gameQueue.familyIndex || srcBarrier.oldLayout != srcBarrier.newLayout)
        preCopyBarriers.add(srcBarrier);
    preCopyBarriers.add(dstBarrier);
    preCopyBarriers.record(passthroughCommandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    // copy
    recordCopy(passthroughCommandBuffer, srcImage, dstImage, presentInfo.realSwapchainImageDescription.width, presentInfo.realSwapchainImageDescription.height);

    // revert replacement buffer to shader read layout
    flipBarrier(srcBarrier);

    // set real swapchain buffer to present layout
    dstBarrier = getImageMemoryBarrier(dstImage, VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    ImageBarrierHelper postCopyBarriers;
    postCopyBarriers.add(srcBarrier);
    postCopyBarriers.add(dstBarrier);
    postCopyBarriers.record(passthroughCommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    return queuePresentNonInterpolated(passthroughList, imageIndex, presentQueueWait);
}

VkResult FrameInterpolationSwapChainVK::presentNonInterpolatedWithUiCompositionOnPresentQueue(
    uint32_t imageIndex, SubmissionSemaphores& gameQueueWait, SubmissionSemaphores& gameQueueSignal, SubmissionSemaphores& presentQueueWait, bool needUICopy)
{
    bool needQueueOwnershipTransfer = presentInfo.presentQueue.familyIndex != presentInfo.gameQueue.familyIndex;

    VkImage srcImage = replacementSwapBuffers[replacementSwapBufferIndex].image;
    VkImage dstImage = presentInfo.realSwapchainImages[imageIndex];

    ImageBarrierHelper presentQueueBarriers;

    if (needQueueOwnershipTransfer || needUICopy)
    {
        auto            ownershipList          = presentInfo.commandPool.get(presentInfo.device, presentInfo.gameQueue, "ownershipList");
        VkCommandBuffer ownershipCommandBuffer = ownershipList->reset();

        ImageBarrierHelper gameQueueBarriers;

        if (needUICopy)
        {
            // copy on game queue and transfer to present queue if necessary
            VkImageMemoryBarrier uiReplacementBufferBarrier = copyUiResource(ownershipCommandBuffer, gameQueueWait, needQueueOwnershipTransfer);

            // transition of the UI replacement resource
            if (needQueueOwnershipTransfer)
            {
                presentQueueBarriers.add(uiReplacementBufferBarrier);
            }
        }

        if (needQueueOwnershipTransfer)
        {
            // queue family ownership transfer for replacement buffer
            VkImageMemoryBarrier ownershipBarrier = getImageMemoryBarrier(srcImage,
                                                                          ReplacementBufferTransferState.accessMask,
                                                                          ReplacementBufferTransferState.accessMask,
                                                                          ReplacementBufferTransferState.layout,
                                                                          ReplacementBufferTransferState.layout,
                                                                          presentInfo.gameQueue.familyIndex,
                                                                          presentInfo.presentQueue.familyIndex);

            // record the queue family ownership transfer on both queues
            presentQueueBarriers.add(ownershipBarrier);
            gameQueueBarriers.add(ownershipBarrier);

            // queue family ownership transfer for UI surface if not copied
            if (!needUICopy && presentInfo.currentUiSurface.resource != VK_NULL_HANDLE)
            {
                ownershipBarrier = presentInfo.queueFamilyOwnershipTransferGameToPresent(presentInfo.currentUiSurface);
                // record the queue family ownership transfer on both queues
                presentQueueBarriers.add(ownershipBarrier);
                gameQueueBarriers.add(ownershipBarrier);
            }
        }

        gameQueueBarriers.record(ownershipCommandBuffer);

        VkResult res = ownershipList->execute(gameQueueWait, gameQueueSignal);
        FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS, "[presentWithUiComposition] queue submit failed with error %d", res);
    }
    else
    {
        VkResult res = presentInfo.gameQueue.submit(VK_NULL_HANDLE, gameQueueWait, gameQueueSignal);
        FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS, "[presentWithUiComposition] queue submit failed with error %d", res);
    }

    // FFX doesn't have a undefined state. Transition to Present here. It will come back as Present after the callback
    presentQueueBarriers.add(dstImage, 0, 0, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    auto            uiCompositionList          = presentInfo.commandPool.get(presentInfo.device, presentInfo.presentQueue, "uiCompositionList");
    VkCommandBuffer uiCompositionCommandBuffer = uiCompositionList->reset();

    // transition layout & queue family ownership transfer (if necessary)
    presentQueueBarriers.record(uiCompositionCommandBuffer);

    FfxResourceDescription srcResourceDesc = replacementSwapBuffers[replacementSwapBufferIndex].description;
    FfxResourceDescription dstResourceDesc = presentInfo.realSwapchainImageDescription;

    FfxPresentCallbackDescription desc{};
    desc.commandList           = ffxGetCommandListVK(uiCompositionCommandBuffer);
    desc.device                = presentInfo.device;
    desc.isInterpolatedFrame   = false;
    desc.outputSwapChainBuffer = ffxGetResourceVK(dstImage, dstResourceDesc, nullptr, FFX_RESOURCE_STATE_PRESENT);
    desc.currentBackBuffer     = ffxGetResourceVK(srcImage, srcResourceDesc, nullptr, FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ);
    if (presentInfo.uiCompositionFlags & FFX_UI_COMPOSITION_FLAG_ENABLE_INTERNAL_UI_DOUBLE_BUFFERING)
    {
        desc.currentUI = ffxGetResourceVK(uiReplacementBuffer.image, uiReplacementBuffer.description, nullptr, FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ);
    }
    else
    {
        desc.currentUI = presentInfo.currentUiSurface;
    }
    desc.frameID = currentFrameID;

    presentCallback(&desc, presentCallbackContext);

    return queuePresentNonInterpolated(uiCompositionList, imageIndex, presentQueueWait);
}

VkResult FrameInterpolationSwapChainVK::presentNonInterpolatedWithUiCompositionOnGameQueue(
    uint32_t imageIndex, SubmissionSemaphores& gameQueueWait, SubmissionSemaphores& gameQueueSignal, SubmissionSemaphores& presentQueueWait, bool needUICopy)
{
    auto            uiCompositionList          = presentInfo.commandPool.get(presentInfo.device, presentInfo.gameQueue, "uiCompositionList");
    VkCommandBuffer uiCompositionCommandBuffer = uiCompositionList->reset();

    if (needUICopy)
        copyUiResource(uiCompositionCommandBuffer, gameQueueWait, false);

    VkImage srcImage = replacementSwapBuffers[replacementSwapBufferIndex].image;
    VkImage dstImage = presentInfo.realSwapchainImages[imageIndex];

    ImageBarrierHelper gameQueueBarriers;

    // FFX doesn't have a undefined state. Transition to Present here. It will come back as Present after the callback
    gameQueueBarriers.add(dstImage, 0, 0, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    // transition layout & queue family ownership transfer (if necessary)
    gameQueueBarriers.record(uiCompositionCommandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

    FfxResourceDescription srcResourceDesc = replacementSwapBuffers[replacementSwapBufferIndex].description;
    FfxResourceDescription dstResourceDesc = presentInfo.realSwapchainImageDescription;

    FfxPresentCallbackDescription desc{};
    desc.commandList           = ffxGetCommandListVK(uiCompositionCommandBuffer);
    desc.device                = presentInfo.device;
    desc.isInterpolatedFrame   = false;
    desc.outputSwapChainBuffer = ffxGetResourceVK(dstImage, dstResourceDesc, nullptr, FFX_RESOURCE_STATE_PRESENT);
    desc.currentBackBuffer     = ffxGetResourceVK(srcImage, srcResourceDesc, nullptr, FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ);
    if (presentInfo.uiCompositionFlags & FFX_UI_COMPOSITION_FLAG_ENABLE_INTERNAL_UI_DOUBLE_BUFFERING)
    {
        desc.currentUI = ffxGetResourceVK(uiReplacementBuffer.image, uiReplacementBuffer.description, nullptr, FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ);
    }
    else
    {
        desc.currentUI = presentInfo.currentUiSurface;
    }
    desc.usePremulAlpha = ((presentInfo.uiCompositionFlags & FFX_UI_COMPOSITION_FLAG_USE_PREMUL_ALPHA) != 0);
    desc.frameID        = currentFrameID;
    presentCallback(&desc, presentCallbackContext);

    gameQueueSignal.add(presentInfo.replacementBufferSemaphore, ++framesSentForPresentation);

    ImageBarrierHelper presentQueueBarriers;
    if (presentInfo.presentQueue.familyIndex != presentInfo.gameQueue.familyIndex)
    {
        // queue family ownership transfer for swapchain buffer
        presentQueueBarriers.add(dstImage,
                                 0,
                                 0,
                                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                 presentInfo.gameQueue.familyIndex,
                                 presentInfo.presentQueue.familyIndex);
        presentQueueBarriers.record(uiCompositionCommandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    }
    else
    {
        gameQueueSignal.add(presentInfo.frameRenderedSemaphores[imageIndex]);  // not a timeline semaphore
    }

    // cannot signal after present on the present queue, so signal here
    gameQueueSignal.add(presentInfo.presentSemaphore, framesSentForPresentation);
    presentInfo.lastPresentSemaphoreValue = framesSentForPresentation;

    VkResult res = uiCompositionList->execute(gameQueueWait, gameQueueSignal);
    FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS, "[presentWithUiComposition2] queue submit failed with error %d", res);

    if (presentInfo.presentQueue.familyIndex != presentInfo.gameQueue.familyIndex)
    {
        auto            ownershipList          = presentInfo.commandPool.get(presentInfo.device, presentInfo.presentQueue, "ownershipList");
        VkCommandBuffer ownershipCommandBuffer = ownershipList->reset();

        presentQueueBarriers.record(ownershipCommandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

        SubmissionSemaphores presentQueueSignal;
        presentQueueSignal.add(presentInfo.frameRenderedSemaphores[imageIndex]);  // not a timeline semaphore
        res = ownershipList->execute(presentQueueWait, presentQueueSignal);
        FFX_ASSERT_MESSAGE_FORMAT(
            res == VK_SUCCESS, "[queuePresentNonInterpolated] queue family ownership transfer to present queue failed with error %d", res);
    }

    VkResult presentRes = presentToSwapChain(&presentInfo, imageIndex, imageIndex);
    FFX_ASSERT_MESSAGE_FORMAT(presentRes == VK_SUCCESS || presentRes == VK_SUBOPTIMAL_KHR || presentRes == VK_ERROR_OUT_OF_DATE_KHR,
                              "[queuePresentNonInterpolated] present failed with error %d",
                              presentRes);

    SubmissionSemaphores semaphoresToWait;
    SubmissionSemaphores semaphoresToSignal;
    semaphoresToWait.add(presentInfo.presentSemaphore, framesSentForPresentation);
    res = presentInfo.gameQueue.submit(VK_NULL_HANDLE, semaphoresToWait, semaphoresToSignal);
    FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS, "[queuePresentNonInterpolated] game queue submit failed with error %d", res);

    return presentRes;
}

VkResult FrameinterpolationPresentInfo::acquireNextRealImage(uint32_t& imageIndex, VkSemaphore& acquireSemaphore)
{
    acquireSemaphore = acquireSemaphores[nextAcquireSemaphoreIndex];
#ifdef FFX_PLATFORM_WINDOWS
    EnterCriticalSection(&swapchainCriticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_lock(&swapchainMutex);
#endif
    VkResult res = VulkanWrapper().vkAcquireNextImageKHR(device, realSwapchain, UINT64_MAX, acquireSemaphore, VK_NULL_HANDLE, &imageIndex);
#ifdef FFX_PLATFORM_WINDOWS
    LeaveCriticalSection(&swapchainCriticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_unlock(&swapchainMutex);
#endif

    // only increment on success
    // VK_NOT_READY shouldn't be returned according to the Vulkan spec, as timeout isn't 0
    // https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#vkAcquireNextImageKHR
    // but in practice, it happens...
    if (res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR || res == VK_NOT_READY || res == VK_TIMEOUT)
    {
        nextAcquireSemaphoreIndex = (nextAcquireSemaphoreIndex + 1) % FFX_FRAME_INTERPOLATION_SWAP_CHAIN_MAX_ACQUIRE_SEMAPHORE_COUNT;
    }

    return res;
}

VkResult FrameInterpolationSwapChainVK::queuePresent(VkQueue queue, const VkPresentInfoKHR* pPresentInfo)
{
    if (queue == VK_NULL_HANDLE || pPresentInfo == nullptr)
    {
        return VK_INCOMPLETE;
    }

    VkResult res = VK_SUCCESS;

    setFrameGenerationConfig(&nextFrameGenerationConfig);
#ifdef FFX_PLATFORM_WINDOWS
    EnterCriticalSection(&criticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_lock(&criticalSection);
#endif

    uint32_t currentBackBufferIndex = replacementSwapBufferIndex;

    // ensure that we aren't running too ahead of the
    FFX_ASSERT_MESSAGE(pPresentInfo->pImageIndices[0] == replacementSwapBufferIndex,
                       "Presented image and internal replacement swap buffer index aren't in sync.");

    // first determine which codepath to run
    bool bRunInterpolation = true;

    // determine what present path to execute
    const bool fgCallbackConfigured    = frameGenerationCallback != nullptr;
    const bool fgCommandListConfigured = registeredInterpolationCommandLists[currentBackBufferIndex] != nullptr;
    const bool runInterpolation        = interpolationEnabled && (fgCallbackConfigured || fgCommandListConfigured);

    // Verify integrity of internal UI resource
    bool needUICopy = false;
    if (verifyUiDuplicateResource())
    {
        // check here if the UI resource will have to be copied to the internal resource
        needUICopy = ((presentInfo.uiCompositionFlags & FFX_UI_COMPOSITION_FLAG_ENABLE_INTERNAL_UI_DOUBLE_BUFFERING) != 0) &&
                     (presentInfo.currentUiSurface.resource != VK_NULL_HANDLE);
    }

    if (runInterpolation)
    {
#ifdef FFX_PLATFORM_WINDOWS
        WaitForSingleObject(presentInfo.interpolationEvent, INFINITE);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
        pthread_mutex_lock(&presentInfo.interpolationMutex);
        while (!presentInfo.interpolationEventSignaled)
            pthread_cond_wait(&presentInfo.interpolationCond, &presentInfo.interpolationMutex);
        presentInfo.interpolationEventSignaled = false;
        pthread_mutex_unlock(&presentInfo.interpolationMutex);
#endif
        res = presentInterpolated(pPresentInfo, currentBackBufferIndex, needUICopy);
    }
    else
    {
        // if no interpolation, then we copy directly to the swapchain. Render UI, present and be done
        // it should also wait for the semaphore acquiring the image.

        // if there are some semaphores the present was supposed to wait on, let the game queue wait on them
        SubmissionSemaphores gameQueueWait;
        addPresentInfoSemaphores(pPresentInfo, gameQueueWait);

        // game queue will increment the gameSemaphore
        // we don't signal here because some queue family ownership transfer might be needed
        SubmissionSemaphores gameQueueSignal;
        gameQueueSignal.add(presentInfo.gameSemaphore, ++gameSemaphoreValue);

        // present queue should wait for the game queue to be finished
        SubmissionSemaphores presentQueueWait;  // wait on the present queue
        presentQueueWait.add(presentInfo.gameSemaphore, gameSemaphoreValue);

        // fails
        FFX_ASSERT_MESSAGE(pPresentInfo->pNext == nullptr, "VkPresentInfoKHR::pNext not yet supported.");

        uint32_t    imageIndex       = 0;
        VkSemaphore acquireSemaphore = VK_NULL_HANDLE;

        VkResult res = presentInfo.acquireNextRealImage(imageIndex, acquireSemaphore);
        FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_NOT_READY || res == VK_TIMEOUT,
                                  "[queuePresent] acquiring next image failed with error %d",
                                  res);
        if (res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR)
        {
            // composition queue should wait until the image is available to render into it
            if (presentInfo.compositionMode == FGSwapchainCompositionMode::eComposeOnPresentQueue)
            {
                presentQueueWait.add(acquireSemaphore);
            }
            else if (presentInfo.compositionMode == FGSwapchainCompositionMode::eComposeOnGameQueue)
            {
                gameQueueWait.add(acquireSemaphore);
            }

            if (presentCallback != nullptr)
            {
                if (presentInfo.compositionMode == FGSwapchainCompositionMode::eComposeOnPresentQueue)
                {
                    res = presentNonInterpolatedWithUiCompositionOnPresentQueue(imageIndex, gameQueueWait, gameQueueSignal, presentQueueWait, needUICopy);
                }
                else if (presentInfo.compositionMode == FGSwapchainCompositionMode::eComposeOnGameQueue)
                {
                    res = presentNonInterpolatedWithUiCompositionOnGameQueue(imageIndex, gameQueueWait, gameQueueSignal, presentQueueWait, needUICopy);
                }
            }
            else
            {
                res = presentPassthrough(imageIndex, gameQueueWait, gameQueueSignal, presentQueueWait);
            }
        }
        else
        {
            // no image was acquired. Skip present. Just signal the replacement buffer semaphore for it to be used at a later point
            gameQueueSignal.add(presentInfo.replacementBufferSemaphore, ++framesSentForPresentation);
            presentInfo.gameQueue.submit(VK_NULL_HANDLE, gameQueueWait, gameQueueSignal);
        }
    }

    previousFrameWasInterpolated = runInterpolation;

    replacementSwapBuffers[currentBackBufferIndex].availabilitySemaphoreValue = framesSentForPresentation;

    // Unregister any potential command list
    registeredInterpolationCommandLists[currentBackBufferIndex] = nullptr;
    presentCount++;
    interpolationBufferIndex = presentCount % FFX_COUNTOF(interpolationOutputs);

    // update active backbuffer and block when no buffer is available
    replacementSwapBufferIndex = presentCount % gameBufferCount;

#ifdef FFX_PLATFORM_WINDOWS
    LeaveCriticalSection(&criticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_unlock(&criticalSection);
#endif

    waitForSemaphoreValue(presentInfo.device,
                          presentInfo.replacementBufferSemaphore,
                          replacementSwapBuffers[replacementSwapBufferIndex].availabilitySemaphoreValue,
                          UINT64_MAX,
                          presentInfo.waitCallback);

    return res;
}

bool FrameInterpolationSwapChainVK::spawnPresenterThread()
{
#ifdef FFX_PLATFORM_WINDOWS
    if (interpolationThreadHandle == NULL)
    {
        presentInfo.shutdown      = false;
        interpolationThreadHandle = CreateThread(nullptr, 0, interpolationThread, reinterpret_cast<void*>(&presentInfo), 0, nullptr);

        FFX_ASSERT(interpolationThreadHandle != NULL);

        if (interpolationThreadHandle != 0)
        {
            SetThreadPriority(interpolationThreadHandle, THREAD_PRIORITY_HIGHEST);
            SetThreadDescription(interpolationThreadHandle, L"AMD FSR Interpolation Thread");
        }
        SetEvent(presentInfo.interpolationEvent);  // unlocks the queuePresent method
    }

    return interpolationThreadHandle != NULL;
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    if (interpolationThreadHandle == 0)
    {
        presentInfo.shutdown = false;
        pthread_create(&interpolationThreadHandle, nullptr, interpolationThread, reinterpret_cast<void*>(&presentInfo));
        FFX_ASSERT(interpolationThreadHandle != 0);

        if (interpolationThreadHandle != 0)
        {
            struct sched_param sch_params;
            sch_params.sched_priority = sched_get_priority_max(SCHED_FIFO);
            pthread_setschedparam(interpolationThreadHandle, SCHED_FIFO, &sch_params);
            pthread_setname_np(interpolationThreadHandle, "Interpolation");
        }

        pthread_mutex_lock(&presentInfo.interpolationMutex);
        presentInfo.interpolationEventSignaled = true;
        pthread_cond_signal(&presentInfo.interpolationCond);
        pthread_mutex_unlock(&presentInfo.interpolationMutex);
    }

    return interpolationThreadHandle != 0;
#endif
}

bool FrameInterpolationSwapChainVK::killPresenterThread()
{
#ifdef FFX_PLATFORM_WINDOWS
    if (interpolationThreadHandle != NULL)
    {
        // prepare present CPU thread for shutdown
        presentInfo.shutdown = true;

        // signal event to allow thread to finish
        SetEvent(presentInfo.presentEvent);
        WaitForSingleObject(interpolationThreadHandle, INFINITE);
        SafeCloseHandle(interpolationThreadHandle);
    }

    return interpolationThreadHandle == NULL;
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    if (interpolationThreadHandle != 0)
    {
        // prepare present CPU thread for shutdown
        presentInfo.shutdown = true;

        // signal event to allow thread to finish
        pthread_mutex_lock(&presentInfo.presentMutex);
        presentInfo.presentEventSignaled = true;
        pthread_cond_signal(&presentInfo.presentCond);
        pthread_mutex_unlock(&presentInfo.presentMutex);

        // Wait for thread to actually finish (equivalent to WaitForSingleObject)
        pthread_join(interpolationThreadHandle, nullptr);
        interpolationThreadHandle = 0;
    }

    return interpolationThreadHandle == 0;
#endif
}

void FrameInterpolationSwapChainVK::setHdrMetadata(VkDevice device, const VkHdrMetadataEXT* pMetadata)
{
    if (pMetadata != nullptr)
    {
        if (vkSetHdrMetadataEXTProc != nullptr)
            vkSetHdrMetadataEXTProc(device, 1, &presentInfo.realSwapchain, pMetadata);
    }
}

uint64_t FrameInterpolationSwapChainVK::getLastPresentCount()
{
    return presentInfo.realPresentCount;
}

VkCommandBuffer FrameInterpolationSwapChainVK::getInterpolationCommandList()
{
#ifdef FFX_PLATFORM_WINDOWS
    EnterCriticalSection(&criticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_lock(&criticalSection);
#endif

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    // store active backbuffer index to the command list, used to verify list usage later
    if (interpolationEnabled)
    {
        uint32_t currentBackBufferIndex = replacementSwapBufferIndex;

        VkCommands* registeredCommands = registeredInterpolationCommandLists[currentBackBufferIndex];

        // drop if already existing
        if (registeredCommands != nullptr)
        {
            registeredCommands->drop();
        }

        registeredCommands = presentInfo.commandPool.get(presentInfo.device, presentInfo.interpolationQueue, "getInterpolationCommandList");
        FFX_ASSERT(registeredCommands);

        commandBuffer = registeredCommands->reset();

        registeredInterpolationCommandLists[currentBackBufferIndex] = registeredCommands;
    }

#ifdef FFX_PLATFORM_WINDOWS
    LeaveCriticalSection(&criticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_unlock(&criticalSection);
#endif

    return commandBuffer;
}

VkResult FrameInterpolationSwapChainVK::submitCompositionOnGameQueue(const PacingData& entry)
{
    VkResult res = VK_SUCCESS;

    FFX_ASSERT_MESSAGE(entry.numFramesToPresent > 0, "[submitCompositionOnGameQueue] need at least one frame to present");
    if (entry.numFramesToPresent > 0)
    {
        SubmissionSemaphores toWait;
        toWait.add(presentInfo.gameSemaphore, entry.gameSemaphoreValue);

        bool uiSurfaceTransfered = true;  // the UI surface is already on the game queue. so there is no need to transition it

        for (uint32_t frameType = 0; frameType < PacingData::FrameType::Count; frameType++)
        {
            const PacingData::FrameInfo& frameInfo = entry.frames[frameType];
            if (frameInfo.doPresent)
            {
                SubmissionSemaphores toSignal;

                // unused value
                uint32_t realSwapchainImageIndex = 0;

                res = compositeSwapChainFrame(&presentInfo,
                                              &entry,
                                              (PacingData::FrameType)frameType,
                                              realSwapchainImageIndex,
                                              presentInfo.gameQueue,
                                              toWait,
                                              toSignal,
                                              uiSurfaceTransfered);
                FFX_ASSERT_MESSAGE_FORMAT(res == VK_SUCCESS, "compositeSwapChainFrame failed with error %d", res);
            }
        }

        // if no frame was presented, we still need to update the semaphore
        if (toWait.count > 0)
        {
            SubmissionSemaphores sema;
            res = presentInfo.gameQueue.submit(VK_NULL_HANDLE, toWait, sema);
        }
    }

    return res;
}

bool FrameInterpolationSwapChainVK::verifyUiDuplicateResource()
{
    VkResult res = VK_SUCCESS;

    const VkAllocationCallbacks* pAllocator = nullptr;  // we don't have it at this point

    VkImage                uiResource     = static_cast<VkImage>(presentInfo.currentUiSurface.resource);
    FfxResourceDescription uiResourceDesc = presentInfo.currentUiSurface.description;

    if ((0 == (presentInfo.uiCompositionFlags & FFX_UI_COMPOSITION_FLAG_ENABLE_INTERNAL_UI_DOUBLE_BUFFERING)) || (VK_NULL_HANDLE == uiResource))
    {
        if (uiReplacementBuffer.image != VK_NULL_HANDLE)
        {
            waitForSemaphoreValue(presentInfo.device, presentInfo.compositionSemaphore, framesSentForPresentation, UINT64_MAX, presentInfo.waitCallback);
            destroyImage(uiReplacementBuffer, pAllocator);
            uiReplacementBuffer = {};
        }
    }
    else
    {
        if (uiReplacementBuffer.image != VK_NULL_HANDLE)
        {
            auto internalDesc = uiReplacementBuffer.description;

            if (uiResourceDesc.format != internalDesc.format || uiResourceDesc.width != internalDesc.width || uiResourceDesc.height != internalDesc.height)
            {
                waitForSemaphoreValue(presentInfo.device, presentInfo.compositionSemaphore, framesSentForPresentation, UINT64_MAX, presentInfo.waitCallback);
                destroyImage(uiReplacementBuffer, pAllocator);
                uiReplacementBuffer = {};
            }
        }

        if (uiReplacementBuffer.image == VK_NULL_HANDLE)
        {
            VkImageCreateInfo info = {};
            info.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            info.pNext             = nullptr;
            info.flags             = 0;
            info.imageType         = VK_IMAGE_TYPE_2D;
            info.format            = ffxGetVkFormatFromSurfaceFormat(uiResourceDesc.format);
            info.extent.width      = uiResourceDesc.width;
            info.extent.height     = uiResourceDesc.height;
            info.extent.depth      = 1;
            info.mipLevels         = 1;
            info.arrayLayers       = 1;
            info.samples           = VK_SAMPLE_COUNT_1_BIT;
            info.tiling            = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;  // queue family ownership transfer will be handled manually

            VkPhysicalDeviceMemoryProperties memProperties;
            VulkanWrapper().vkGetPhysicalDeviceMemoryProperties(presentInfo.physicalDevice, &memProperties);

            res = createImage(uiReplacementBuffer, info, uiResourceDesc.format, "AMD FSR Ui Doublebuffer Resource", memProperties, pAllocator);
        }
    }

    return VK_SUCCESS == res;
}

VkImageMemoryBarrier FrameInterpolationSwapChainVK::copyUiResource(VkCommandBuffer       commandBuffer,
                                                                   SubmissionSemaphores& gameQueueWait,
                                                                   bool                  transferToPresentQueue)
{
    if ((previousFrameWasInterpolated) && (uiReplacementBuffer.image != VK_NULL_HANDLE))
    {
        // if doublebuffering is enabled and uiReplacementbuffer has been used last frame
        // game queue should wait for any outstanding composition workloads to finish
        gameQueueWait.add(presentInfo.compositionSemaphore, framesSentForPresentation);
    }

    VkImage imageSrc = reinterpret_cast<VkImage>(presentInfo.currentUiSurface.resource);
    VkImage imageDst = uiReplacementBuffer.image;

    // transitions for copy
    VkImageMemoryBarrier srcBarrier = getImageMemoryBarrier(imageSrc,
                                                            getVKAccessFlagsFromResourceState2(presentInfo.currentUiSurface.state),
                                                            VK_ACCESS_TRANSFER_READ_BIT,
                                                            getVKImageLayoutFromResourceState2(presentInfo.currentUiSurface.state),
                                                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                            VK_QUEUE_FAMILY_IGNORED,
                                                            VK_QUEUE_FAMILY_IGNORED,
                                                            1,   // we only consider the first mip
                                                            1);  // we don't support array

    VkImageMemoryBarrier dstBarrier =
        getImageMemoryBarrier(imageDst, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    ImageBarrierHelper preCopyBarriers;
    preCopyBarriers.add(srcBarrier);
    preCopyBarriers.add(dstBarrier);
    preCopyBarriers.record(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    // copy
    recordCopy(commandBuffer,
               imageSrc,
               imageDst,
               uiReplacementBuffer.description.width,
               uiReplacementBuffer.description.height,
               uiReplacementBuffer.description.depth);

    // transition back to previous state
    flipBarrier(srcBarrier);

    // transition the ui replacement texture to shader read state
    dstBarrier = getImageMemoryBarrier(
        imageDst, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    ImageBarrierHelper postCopyBarriers;
    postCopyBarriers.add(srcBarrier);
    postCopyBarriers.add(dstBarrier);

    if (transferToPresentQueue)
    {
        // NOTE: it might be possible to merge this barrier with dstBarrier above but it needs to be tracked
        VkImageMemoryBarrier owwnershipBarrier = getImageMemoryBarrier(
            imageDst, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        owwnershipBarrier.srcQueueFamilyIndex = presentInfo.gameQueue.familyIndex;
        owwnershipBarrier.dstQueueFamilyIndex = presentInfo.presentQueue.familyIndex;
        postCopyBarriers.add(owwnershipBarrier);
    }

    postCopyBarriers.record(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    presentInfo.currentUiSurface.resource = nullptr;
    return dstBarrier;
}

static void convertAndExportRGBImageToExr(const char* filename, const uint32_t width, const uint32_t height, const float* rgbData)
{
#ifdef ENABLE_EXR_EXPORT
    EXRHeader header;
    InitEXRHeader(&header);
    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;

    std::vector<float> images[3];
    images[0].resize(width * height);
    images[1].resize(width * height);
    images[2].resize(width * height);

    // Split RGBRGBRGB... into R, G and B layer
    for (uint32_t i = 0; i < width * height; i++)
    {
        images[0][i] = rgbData[4 * i + 0];
        images[1][i] = rgbData[4 * i + 1];
        images[2][i] = rgbData[4 * i + 2];
    }

    float* image_ptr[3];
    image_ptr[0] = &(images[2].at(0));  // B
    image_ptr[1] = &(images[1].at(0));  // G
    image_ptr[2] = &(images[0].at(0));  // R

    image.images = (unsigned char**)image_ptr;
    image.width  = width;
    image.height = height;

    header.num_channels = 3;
    header.channels     = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
    // Must be (A)BGR order, since most of EXR viewers expect this channel order.
    strncpy(header.channels[0].name, "B", 255);
    header.channels[0].name[strlen("B")] = '\0';
    strncpy(header.channels[1].name, "G", 255);
    header.channels[1].name[strlen("G")] = '\0';
    strncpy(header.channels[2].name, "R", 255);
    header.channels[2].name[strlen("R")] = '\0';

    header.pixel_types           = (int*)malloc(sizeof(int) * header.num_channels);
    header.requested_pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++)
    {
        header.pixel_types[i]           = TINYEXR_PIXELTYPE_FLOAT;
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF;
    }

    const char* err = nullptr;
    int         ret = SaveEXRImageToFile(&image, &header, filename, &err);
    if (ret != TINYEXR_SUCCESS)
    {
        fprintf(stderr, "Save EXR err: %s\n", err);
        FreeEXRErrorMessage(err);  // free's buffer for an error message
    }

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
#else
    // Fallback: save as raw binary
    FILE* file = fopen(filename, "wb");
    if (file)
    {
        fwrite(rgbData, sizeof(float), width * height * 4, file);
        fclose(file);
    }
#endif
}

static float halfToFloat(uint16_t halfValue)
{
    union
    {
        uint32_t i;
        float    f;
    } result;

    uint32_t sign     = (halfValue & 0x8000) << 16;
    uint32_t exponent = (halfValue & 0x7C00) >> 10;
    uint32_t mantissa = halfValue & 0x03FF;

    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            // Zero
            result.i = sign;
        }
        else
        {
            // Denormalized number
            exponent = 127 - 14;
            while ((mantissa & 0x400) == 0)
            {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            result.i = sign | (exponent << 23) | (mantissa << 13);
        }
    }
    else if (exponent == 31)
    {
        // Infinity or NaN
        result.i = sign | 0x7F800000 | (mantissa << 13);
    }
    else
    {
        // Normalized number
        result.i = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    return result.f;
}

static void dumpResourceAsyncNoCmdBuffer(
    VkDevice device, VkPhysicalDevice physicalDevice, VkQueue queue, const FfxResource& resource, const char* baseFilename, uint64_t frameID)
{
    if (resource.resource == nullptr)
        return;

    VkImage sourceImage = static_cast<VkImage>(resource.resource);

    // Get image properties
    uint32_t width  = resource.description.width;
    uint32_t height = resource.description.height;
    VkFormat format = ffxGetVkFormatFromSurfaceFormat(resource.description.format);

    // Get queue family index for command pool creation
    uint32_t queueFamilyIndex = 0;
    // vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);  // This assumes queue is from family 0
    // Better approach: get the actual queue family index if available

    // Create our own command pool
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags                   = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex        = queueFamilyIndex;  // You'll need to determine this

    VkCommandPool commandPool;
    VkResult      result = VulkanWrapper().vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
    if (result != VK_SUCCESS)
        return;

    // Create staging buffer for readback
    VkDeviceSize imageSize = (VkDeviceSize)width * (VkDeviceSize)height * 4;  // Assuming RGBA format

    VkBuffer       stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size               = imageSize;
    bufferInfo.usage              = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;

    result = VulkanWrapper().vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer);
    if (result != VK_SUCCESS)
    {
        VulkanWrapper().vkDestroyCommandPool(device, commandPool, nullptr);
        return;
    }

    VkMemoryRequirements memRequirements;
    VulkanWrapper().vkGetBufferMemoryRequirements(device, stagingBuffer, &memRequirements);

    VkPhysicalDeviceMemoryProperties memProperties;
    VulkanWrapper().vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType                = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize       = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memProperties, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    result = VulkanWrapper().vkAllocateMemory(device, &allocInfo, nullptr, &stagingBufferMemory);
    if (result != VK_SUCCESS)
    {
        VulkanWrapper().vkDestroyBuffer(device, stagingBuffer, nullptr);
        VulkanWrapper().vkDestroyCommandPool(device, commandPool, nullptr);
        return;
    }

    VulkanWrapper().vkBindBufferMemory(device, stagingBuffer, stagingBufferMemory, 0);

    // Create command buffer for the copy operation
    VkCommandBufferAllocateInfo allocInfo2 = {};
    allocInfo2.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo2.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo2.commandPool                 = commandPool;
    allocInfo2.commandBufferCount          = 1;

    VkCommandBuffer commandBuffer;
    VulkanWrapper().vkAllocateCommandBuffers(device, &allocInfo2, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VulkanWrapper().vkBeginCommandBuffer(commandBuffer, &beginInfo);

    // Transition image layout for transfer
    VkImageMemoryBarrier barrier            = {};
    barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout                       = VK_IMAGE_LAYOUT_GENERAL;  // Assuming current layout
    barrier.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.image                           = sourceImage;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = 1;
    barrier.srcAccessMask                   = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask                   = VK_ACCESS_TRANSFER_READ_BIT;

    VulkanWrapper().vkCmdPipelineBarrier(
        commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Copy image to buffer
    VkBufferImageCopy region               = {};
    region.bufferOffset                    = 0;
    region.bufferRowLength                 = 0;
    region.bufferImageHeight               = 0;
    region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel       = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount     = 1;
    region.imageOffset                     = {0, 0, 0};
    region.imageExtent                     = {width, height, 1};

    VulkanWrapper().vkCmdCopyImageToBuffer(commandBuffer, sourceImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuffer, 1, &region);

    // Transition back to original layout
    barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    VulkanWrapper().vkCmdPipelineBarrier(
        commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    VulkanWrapper().vkEndCommandBuffer(commandBuffer);

    // Submit command buffer
    VkSubmitInfo submitInfo       = {};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &commandBuffer;

    VkFence           completionFence;
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VulkanWrapper().vkCreateFence(device, &fenceInfo, nullptr, &completionFence);

    VulkanWrapper().vkQueueSubmit(queue, 1, &submitInfo, completionFence);

    // Generate filename
    char frameStr[16];
    snprintf(frameStr, sizeof(frameStr), "%04ld", frameID);
    std::string filename = std::string(baseFilename) + frameStr + ".exr";

    // Store async operation for later completion
    AsyncDumpOperation asyncOp        = {};
    asyncOp.stagingBuffer             = stagingBuffer;
    asyncOp.stagingBufferMemory       = stagingBufferMemory;
    asyncOp.imageSize                 = imageSize;
    asyncOp.filename                  = std::move(filename);
    asyncOp.resourceDesc              = resource.description;
    asyncOp.width                     = width;
    asyncOp.height                    = height;
    asyncOp.framesSentForPresentation = frameID;
    asyncOp.completionFence           = completionFence;

    {
        std::lock_guard<std::mutex> lock(g_dumpOpsMutex);
        g_pendingDumpOps.push_back(std::move(asyncOp));
    }
}

static void processAsyncDumpOperations(VkDevice device, uint64_t currentFrameCount)
{
    std::lock_guard<std::mutex> lock(g_dumpOpsMutex);

    auto it = g_pendingDumpOps.begin();
    while (it != g_pendingDumpOps.end())
    {
        VkResult result = VulkanWrapper().vkWaitForFences(device, 1, &it->completionFence, VK_TRUE, UINT64_MAX);
        if (result == VK_SUCCESS)
        {
            // Operation completed, process the image data
            void* data;
            VulkanWrapper().vkMapMemory(device, it->stagingBufferMemory, 0, it->imageSize, 0, &data);

            // Process image data based on format and save to file
            if (it->resourceDesc.format == FFX_SURFACE_FORMAT_B8G8R8A8_UNORM || it->resourceDesc.format == FFX_SURFACE_FORMAT_R8G8B8A8_UNORM)
            {
                // Convert uint8 data to float
                uint8_t*           uint8Data = static_cast<uint8_t*>(data);
                std::vector<float> floatData(it->width * it->height * 4);

                for (uint32_t i = 0; i < it->width * it->height; ++i)
                {
                    if (it->resourceDesc.format == FFX_SURFACE_FORMAT_B8G8R8A8_UNORM)
                    {
                        floatData[i * 4 + 0] = uint8Data[i * 4 + 2] / 255.0f;  // R (from B)
                        floatData[i * 4 + 1] = uint8Data[i * 4 + 1] / 255.0f;  // G
                        floatData[i * 4 + 2] = uint8Data[i * 4 + 0] / 255.0f;  // B (from R)
                        floatData[i * 4 + 3] = uint8Data[i * 4 + 3] / 255.0f;  // A
                    }
                    else  // RGBA
                    {
                        floatData[i * 4 + 0] = uint8Data[i * 4 + 0] / 255.0f;  // R
                        floatData[i * 4 + 1] = uint8Data[i * 4 + 1] / 255.0f;  // G
                        floatData[i * 4 + 2] = uint8Data[i * 4 + 2] / 255.0f;  // B
                        floatData[i * 4 + 3] = uint8Data[i * 4 + 3] / 255.0f;  // A
                    }
                }

                convertAndExportRGBImageToExr(it->filename.c_str(), it->width, it->height, floatData.data());
            }
            else if (it->resourceDesc.format == FFX_SURFACE_FORMAT_R16G16_FLOAT)
            {
                // Convert R16G16 float data to RGBA float for EXR export
                uint16_t*          uint16Data = static_cast<uint16_t*>(data);
                std::vector<float> floatData(it->width * it->height * 4);

                for (uint32_t i = 0; i < it->width * it->height; ++i)
                {
                    floatData[i * 4 + 0] = halfToFloat(uint16Data[i * 2 + 0]);  // R
                    floatData[i * 4 + 1] = halfToFloat(uint16Data[i * 2 + 1]);  // G
                    floatData[i * 4 + 2] = 0.0f;                                // B (default to 0)
                    floatData[i * 4 + 3] = 1.0f;                                // A (default to 1)
                }

                convertAndExportRGBImageToExr(it->filename.c_str(), it->width, it->height, floatData.data());
            }
            else
            {
                convertAndExportRGBImageToExr(it->filename.c_str(), it->width, it->height, static_cast<float*>(data));
            }

            VulkanWrapper().vkUnmapMemory(device, it->stagingBufferMemory);

            // Cleanup resources
            VulkanWrapper().vkDestroyFence(device, it->completionFence, nullptr);
            VulkanWrapper().vkFreeMemory(device, it->stagingBufferMemory, nullptr);
            VulkanWrapper().vkDestroyBuffer(device, it->stagingBuffer, nullptr);

            // Remove from pending operations
            it = g_pendingDumpOps.erase(it);
        }
        else
        {
            // Operation still in progress, check next one
            ++it;
        }
    }
}
