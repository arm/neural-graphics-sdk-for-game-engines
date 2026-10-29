/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "../framework/test_framework.h"
#include "../framework/vk_test_utils.h"

#include <FidelityFX/host/backends/vk/vk_wrapper.h>

#include <array>
#include <cstring>
#include <vector>

namespace
{
    constexpr uint32_t     barrierTestImageWidth    = 64;
    constexpr uint32_t     barrierTestImageHeight   = 64;
    constexpr uint32_t     barrierTestBytesPerPixel = 4;
    constexpr VkDeviceSize barrierTestImageByteSize = static_cast<VkDeviceSize>(barrierTestImageWidth) * barrierTestImageHeight * barrierTestBytesPerPixel;
    constexpr VkDeviceSize barrierTestBufferSize    = 64;

    VkDeviceContext MakeDeviceContext(const unit_test::vk::VulkanContext& vkContext)
    {
        VkDeviceContext deviceContext{};
        deviceContext.vkDevice              = vkContext.device;
        deviceContext.vkPhysicalDevice      = vkContext.physicalDevice;
        deviceContext.vkInstance            = vkContext.instance;
        deviceContext.vkDeviceProcAddr      = vkContext.getDeviceProcAddr;
        deviceContext.vkGetInstanceProcAddr = vkContext.getInstanceProcAddr;
        return deviceContext;
    }

    bool CreateBackendInterface(unit_test::vk::VulkanContext& vkContext, FfxInterface& outInterface, std::vector<uint8_t>& scratch)
    {
        VkDeviceContext deviceContext = MakeDeviceContext(vkContext);

        const size_t scratchSize = ffxGetScratchMemorySizeVK(deviceContext, 8);
        UNIT_TEST_CHECK(scratchSize > 0);
        scratch.resize(scratchSize);

        const FfxDevice device = ffxGetDeviceVK(&deviceContext);
        UNIT_TEST_CHECK(device != nullptr);
        UNIT_TEST_CHECK(ffxGetInterfaceVK(&outInterface, device, scratch.data(), scratch.size(), 8) == FFX_OK);
        return true;
    }

}  // namespace

UNIT_TEST_CASE(backend_vk, vk_wrapper_initialization)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!unit_test::vk::InitializeVulkanContext(vkContext, &errorMessage))
    {
        UNIT_TEST_FAIL(std::string("Vulkan unavailable: ") + errorMessage);
    }

    const VkDeviceContext deviceContext = MakeDeviceContext(vkContext);
    if (!InitVulkanWrapper(deviceContext))
    {
        unit_test::vk::DestroyVulkanContext(vkContext);
        UNIT_TEST_FAIL("Vulkan wrapper initialization unsupported on this Vulkan runtime.");
    }

    const VkFunctionTable& table = VulkanWrapper();
    UNIT_TEST_CHECK(table.initialized);
    UNIT_TEST_CHECK(table.vkCreateImage != nullptr);
    UNIT_TEST_CHECK(table.vkCmdDispatch != nullptr);
    UNIT_TEST_CHECK(table.vkQueueSubmit != nullptr);

    unit_test::vk::DestroyVulkanContext(vkContext);
    return true;
}

UNIT_TEST_CASE(backend_vk, create_and_destroy_backend_context)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!unit_test::vk::InitializeVulkanContext(vkContext, &errorMessage))
    {
        UNIT_TEST_FAIL(std::string("Vulkan unavailable: ") + errorMessage);
    }

    if (!InitVulkanWrapper(MakeDeviceContext(vkContext)))
    {
        unit_test::vk::DestroyVulkanContext(vkContext);
        UNIT_TEST_FAIL("Vulkan wrapper initialization unsupported on this Vulkan runtime.");
    }

    FfxInterface         backendInterface{};
    std::vector<uint8_t> scratch;
    UNIT_TEST_CHECK(CreateBackendInterface(vkContext, backendInterface, scratch));
    UNIT_TEST_CHECK(backendInterface.fpCreateBackendContext != nullptr);
    UNIT_TEST_CHECK(backendInterface.fpDestroyBackendContext != nullptr);

    uint32_t effectContextId = 0;
    UNIT_TEST_CHECK(backendInterface.fpCreateBackendContext(&backendInterface, FFX_EFFECT_SHAREDAPIBACKEND, nullptr, &effectContextId) == FFX_OK);
    UNIT_TEST_CHECK(backendInterface.fpDestroyBackendContext(&backendInterface, effectContextId) == FFX_OK);

    unit_test::vk::DestroyVulkanContext(vkContext);
    return true;
}

UNIT_TEST_CASE(backend_vk, create_register_and_destroy_resources)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!unit_test::vk::InitializeVulkanContext(vkContext, &errorMessage))
    {
        UNIT_TEST_FAIL(std::string("Vulkan unavailable: ") + errorMessage);
    }

    if (!InitVulkanWrapper(MakeDeviceContext(vkContext)))
    {
        unit_test::vk::DestroyVulkanContext(vkContext);
        UNIT_TEST_FAIL("Vulkan wrapper initialization unsupported on this Vulkan runtime.");
    }

    FfxInterface         backendInterface{};
    std::vector<uint8_t> scratch;
    UNIT_TEST_CHECK(CreateBackendInterface(vkContext, backendInterface, scratch));

    uint32_t effectContextId = 0;
    UNIT_TEST_CHECK(backendInterface.fpCreateBackendContext(&backendInterface, FFX_EFFECT_SHAREDAPIBACKEND, nullptr, &effectContextId) == FFX_OK);

    FfxCreateResourceDescription createDescription{};
    createDescription.heapType                     = FFX_HEAP_TYPE_DEFAULT;
    createDescription.resourceDescription.type     = FFX_RESOURCE_TYPE_TEXTURE2D;
    createDescription.resourceDescription.format   = FFX_SURFACE_FORMAT_R8G8B8A8_UNORM;
    createDescription.resourceDescription.width    = 64;
    createDescription.resourceDescription.height   = 64;
    createDescription.resourceDescription.depth    = 1;
    createDescription.resourceDescription.mipCount = 1;
    createDescription.resourceDescription.flags    = FFX_RESOURCE_FLAGS_NONE;
    createDescription.resourceDescription.usage    = FFX_RESOURCE_USAGE_UAV;
    createDescription.initialState                 = FFX_RESOURCE_STATE_GENERIC_UAV;
    createDescription.name                         = "unit_test_internal_resource";
    createDescription.id                           = 1;
    createDescription.initData.type                = FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED;

    FfxResourceInternal internalResource{};
    UNIT_TEST_CHECK(backendInterface.fpCreateResource(&backendInterface, &createDescription, effectContextId, &internalResource) == FFX_OK);

    FfxGpuJobDescription barrierJob{};
    barrierJob.jobType                    = FFX_GPU_JOB_BARRIER;
    barrierJob.barrierDescriptor.resource = internalResource;
    barrierJob.barrierDescriptor.newState = FFX_RESOURCE_STATE_COPY_DEST;
    UNIT_TEST_CHECK(backendInterface.fpScheduleGpuJob(&backendInterface, &barrierJob) == FFX_OK);

    VkCommandBuffer abortCommandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(vkContext, abortCommandBuffer, &errorMessage));
    UNIT_TEST_CHECK(backendInterface.fpUnregisterResources(&backendInterface, ffxGetCommandListVK(abortCommandBuffer), effectContextId) == FFX_OK);
    UNIT_TEST_CHECK(backendInterface.fpExecuteGpuJobs(&backendInterface, ffxGetCommandListVK(abortCommandBuffer), effectContextId) == FFX_OK);
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(vkContext, abortCommandBuffer, &errorMessage));
    UNIT_TEST_CHECK(backendInterface.fpGetResource(&backendInterface, internalResource).state == FFX_RESOURCE_STATE_GENERIC_UAV);

    UNIT_TEST_CHECK(backendInterface.fpDestroyResource(&backendInterface, internalResource, effectContextId) == FFX_OK);

    unit_test::vk::Image externalImage{};
    UNIT_TEST_CHECK(unit_test::vk::CreateImage2D(vkContext,
                                                 64,
                                                 64,
                                                 VK_FORMAT_R8G8B8A8_UNORM,
                                                 VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                                 externalImage,
                                                 &errorMessage));
    UNIT_TEST_CHECK(unit_test::vk::ClearImage(vkContext, externalImage, {0.0f, 0.0f, 0.0f, 1.0f}, &errorMessage));

    const FfxResource externalResource =
        unit_test::vk::MakeFfxImageResource(externalImage, FFX_RESOURCE_USAGE_UAV, FFX_RESOURCE_STATE_GENERIC_UAV, "unit_test_external_resource");

    FfxResourceInternal registeredResource{};
    UNIT_TEST_CHECK(backendInterface.fpRegisterResource(&backendInterface, &externalResource, effectContextId, &registeredResource) == FFX_OK);

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(vkContext, commandBuffer, &errorMessage));
    UNIT_TEST_CHECK(backendInterface.fpUnregisterResources(&backendInterface, ffxGetCommandListVK(commandBuffer), effectContextId) == FFX_OK);
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(vkContext, commandBuffer, &errorMessage));

    unit_test::vk::DestroyImage(vkContext, externalImage);
    UNIT_TEST_CHECK(backendInterface.fpDestroyBackendContext(&backendInterface, effectContextId) == FFX_OK);

    unit_test::vk::DestroyVulkanContext(vkContext);
    return true;
}

UNIT_TEST_CASE(backend_vk, unregister_resources_flush_barriers_at_capacity)
{
    constexpr uint32_t resourceCount = FFX_MAX_BARRIERS + 2;
    constexpr uint32_t bufferFill    = 0x12345678u;

    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!unit_test::vk::InitializeVulkanContext(vkContext, &errorMessage))
    {
        UNIT_TEST_FAIL(std::string("Vulkan unavailable: ") + errorMessage);
    }

    if (!InitVulkanWrapper(MakeDeviceContext(vkContext)))
    {
        unit_test::vk::DestroyVulkanContext(vkContext);
        UNIT_TEST_FAIL("Vulkan wrapper initialization unsupported on this Vulkan runtime.");
    }

    FfxInterface         backendInterface{};
    std::vector<uint8_t> scratch;
    UNIT_TEST_CHECK(CreateBackendInterface(vkContext, backendInterface, scratch));

    uint32_t effectContextId = 0;
    UNIT_TEST_CHECK(backendInterface.fpCreateBackendContext(&backendInterface, FFX_EFFECT_SHAREDAPIBACKEND, nullptr, &effectContextId) == FFX_OK);

    std::array<unit_test::vk::Image, resourceCount>  images{};
    std::array<unit_test::vk::Buffer, resourceCount> buffers{};
    std::array<FfxResourceInternal, resourceCount>   imageResources{};
    std::array<FfxResourceInternal, resourceCount>   bufferResources{};
    const VkImageUsageFlags  imageUsage  = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    const VkBufferUsageFlags bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size        = barrierTestBufferSize;
    bufferCreateInfo.usage       = bufferUsage;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    for (uint32_t index = 0; index < resourceCount; ++index)
    {
        UNIT_TEST_CHECK(unit_test::vk::CreateImage2D(
            vkContext, barrierTestImageWidth, barrierTestImageHeight, VK_FORMAT_R8G8B8A8_UNORM, imageUsage, images[index], &errorMessage));
        UNIT_TEST_CHECK(
            unit_test::vk::CreateBuffer(vkContext, barrierTestBufferSize, bufferUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffers[index], &errorMessage));
    }

    std::array<VkImageMemoryBarrier2, resourceCount> initialImageBarriers{};
    for (uint32_t index = 0; index < resourceCount; ++index)
    {
        VkImageMemoryBarrier2& barrier          = initialImageBarriers[index];
        barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask                    = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.dstStageMask                    = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.dstAccessMask                   = VK_ACCESS_2_TRANSFER_READ_BIT;
        barrier.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.image                           = images[index].image;
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;
    }

    VkCommandBuffer initializationCommandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(vkContext, initializationCommandBuffer, &errorMessage));
    VkDependencyInfo initialImageDependencies{};
    initialImageDependencies.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    initialImageDependencies.imageMemoryBarrierCount = resourceCount;
    initialImageDependencies.pImageMemoryBarriers    = initialImageBarriers.data();
    VulkanWrapper().vkCmdPipelineBarrier2(initializationCommandBuffer, &initialImageDependencies);
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(vkContext, initializationCommandBuffer, &errorMessage));

    const VkDeviceSize    imageReadbackSize    = static_cast<VkDeviceSize>(resourceCount) * barrierTestImageByteSize;
    const VkDeviceSize    bufferReadbackOffset = imageReadbackSize;
    const VkDeviceSize    readbackSize         = imageReadbackSize + static_cast<VkDeviceSize>(resourceCount) * barrierTestBufferSize;
    unit_test::vk::Buffer readbackBuffer{};
    UNIT_TEST_CHECK(unit_test::vk::CreateBuffer(vkContext,
                                                readbackSize,
                                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                readbackBuffer,
                                                &errorMessage));

    for (uint32_t index = 0; index < resourceCount; ++index)
    {
        const FfxResource externalImage =
            unit_test::vk::MakeFfxImageResource(images[index], FFX_RESOURCE_USAGE_UAV, FFX_RESOURCE_STATE_COPY_SRC, "unit_test_barrier_image");
        UNIT_TEST_CHECK(backendInterface.fpRegisterResource(&backendInterface, &externalImage, effectContextId, &imageResources[index]) == FFX_OK);
    }

    for (uint32_t index = 0; index < resourceCount; ++index)
    {
        const FfxResourceDescription bufferDescription = ffxGetBufferResourceDescriptionVK(buffers[index].buffer, bufferCreateInfo, FFX_RESOURCE_USAGE_UAV);
        const FfxResource            externalBuffer =
            ffxGetResourceVK(reinterpret_cast<void*>(buffers[index].buffer), bufferDescription, "unit_test_barrier_buffer", FFX_RESOURCE_STATE_COPY_SRC);
        UNIT_TEST_CHECK(backendInterface.fpRegisterResource(&backendInterface, &externalBuffer, effectContextId, &bufferResources[index]) == FFX_OK);
    }

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(vkContext, commandBuffer, &errorMessage));

    constexpr std::array<float, 4> clearColor{0.25f, 0.5f, 0.75f, 1.0f};
    for (uint32_t index = 0; index < resourceCount; ++index)
    {
        FfxGpuJobDescription clearImageJob{};
        clearImageJob.jobType                   = FFX_GPU_JOB_CLEAR_FLOAT;
        clearImageJob.clearJobDescriptor.target = imageResources[index];
        for (uint32_t channel = 0; channel < clearColor.size(); ++channel)
            clearImageJob.clearJobDescriptor.color[channel] = clearColor[channel];
        UNIT_TEST_CHECK(backendInterface.fpScheduleGpuJob(&backendInterface, &clearImageJob) == FFX_OK);

        FfxGpuJobDescription clearBufferJob{};
        clearBufferJob.jobType                         = FFX_GPU_JOB_CLEAR_UINT;
        clearBufferJob.clearUintJobDescriptor.target   = bufferResources[index];
        clearBufferJob.clearUintJobDescriptor.color[0] = bufferFill;
        UNIT_TEST_CHECK(backendInterface.fpScheduleGpuJob(&backendInterface, &clearBufferJob) == FFX_OK);
    }

    UNIT_TEST_CHECK(backendInterface.fpExecuteGpuJobs(&backendInterface, ffxGetCommandListVK(commandBuffer), effectContextId) == FFX_OK);
    UNIT_TEST_CHECK(backendInterface.fpUnregisterResources(&backendInterface, ffxGetCommandListVK(commandBuffer), effectContextId) == FFX_OK);

    for (uint32_t index = 0; index < resourceCount; ++index)
    {
        VkBufferImageCopy imageCopy{};
        imageCopy.bufferOffset                = static_cast<VkDeviceSize>(index) * barrierTestImageByteSize;
        imageCopy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageCopy.imageSubresource.layerCount = 1;
        imageCopy.imageExtent                 = {barrierTestImageWidth, barrierTestImageHeight, 1};
        VulkanWrapper().vkCmdCopyImageToBuffer(commandBuffer, images[index].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readbackBuffer.buffer, 1, &imageCopy);

        VkBufferCopy bufferCopy{};
        bufferCopy.dstOffset = bufferReadbackOffset + static_cast<VkDeviceSize>(index) * barrierTestBufferSize;
        bufferCopy.size      = barrierTestBufferSize;
        VulkanWrapper().vkCmdCopyBuffer(commandBuffer, buffers[index].buffer, readbackBuffer.buffer, 1, &bufferCopy);
    }

    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(vkContext, commandBuffer, &errorMessage));

    void* mappedReadback = nullptr;
    UNIT_TEST_CHECK(VulkanWrapper().vkMapMemory(vkContext.device, readbackBuffer.memory, 0, readbackSize, 0, &mappedReadback) == VK_SUCCESS);
    const auto*                      readbackBytes = static_cast<const uint8_t*>(mappedReadback);
    constexpr std::array<uint8_t, 4> expectedColor{64, 128, 191, 255};
    for (uint32_t index = 0; index < resourceCount; ++index)
    {
        const VkDeviceSize imageOffset = static_cast<VkDeviceSize>(index) * barrierTestImageByteSize;
        for (uint32_t channel = 0; channel < expectedColor.size(); ++channel)
        {
            const int difference = static_cast<int>(readbackBytes[imageOffset + channel]) - expectedColor[channel];
            UNIT_TEST_CHECK(difference >= -1 && difference <= 1);
        }

        uint32_t           bufferValue  = 0;
        const VkDeviceSize bufferOffset = bufferReadbackOffset + static_cast<VkDeviceSize>(index) * barrierTestBufferSize;
        std::memcpy(&bufferValue, readbackBytes + bufferOffset, sizeof(bufferValue));
        UNIT_TEST_CHECK(bufferValue == bufferFill);
    }
    VulkanWrapper().vkUnmapMemory(vkContext.device, readbackBuffer.memory);

    UNIT_TEST_CHECK(backendInterface.fpDestroyBackendContext(&backendInterface, effectContextId) == FFX_OK);

    for (uint32_t index = 0; index < resourceCount; ++index)
    {
        unit_test::vk::DestroyImage(vkContext, images[index]);
        unit_test::vk::DestroyBuffer(vkContext, buffers[index]);
    }

    unit_test::vk::DestroyBuffer(vkContext, readbackBuffer);
    unit_test::vk::DestroyVulkanContext(vkContext);
    return true;
}

// Unregistration visits dynamically registered resources; backend-created aliases use the GPU-job API.
UNIT_TEST_CASE(backend_vk, api_gpu_job_clears_buffer_aliased_image)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!unit_test::vk::InitializeVulkanContext(vkContext, &errorMessage))
    {
        UNIT_TEST_FAIL(std::string("Vulkan unavailable: ") + errorMessage);
    }

    if (!InitVulkanWrapper(MakeDeviceContext(vkContext)))
    {
        unit_test::vk::DestroyVulkanContext(vkContext);
        UNIT_TEST_FAIL("Vulkan wrapper initialization unsupported on this Vulkan runtime.");
    }

    FfxInterface         backendInterface{};
    std::vector<uint8_t> scratch;
    UNIT_TEST_CHECK(CreateBackendInterface(vkContext, backendInterface, scratch));

    uint32_t effectContextId = 0;
    UNIT_TEST_CHECK(backendInterface.fpCreateBackendContext(&backendInterface, FFX_EFFECT_SHAREDAPIBACKEND, nullptr, &effectContextId) == FFX_OK);

    FfxCreateResourceDescription createDescription{};
    createDescription.heapType                     = FFX_HEAP_TYPE_DEFAULT;
    createDescription.resourceDescription.type     = FFX_RESOURCE_TYPE_TEXTURE2D;
    createDescription.resourceDescription.format   = FFX_SURFACE_FORMAT_R8G8B8A8_UNORM;
    createDescription.resourceDescription.width    = barrierTestImageWidth;
    createDescription.resourceDescription.height   = barrierTestImageHeight;
    createDescription.resourceDescription.depth    = 1;
    createDescription.resourceDescription.mipCount = 1;
    createDescription.resourceDescription.flags    = FFX_RESOURCE_FLAGS_BUFFER_ALIASED;
    createDescription.resourceDescription.usage    = FFX_RESOURCE_USAGE_UAV;
    createDescription.initialState                 = FFX_RESOURCE_STATE_GENERIC_UAV;
    createDescription.name                         = "unit_test_buffer_aliased_image";
    createDescription.initData.type                = FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED;

    FfxResourceInternal resource{};
    UNIT_TEST_CHECK(backendInterface.fpCreateResource(&backendInterface, &createDescription, effectContextId, &resource) == FFX_OK);

    FfxGpuJobDescription clearJob{};
    clearJob.jobType                     = FFX_GPU_JOB_CLEAR_FLOAT;
    clearJob.clearJobDescriptor.target   = resource;
    clearJob.clearJobDescriptor.color[0] = 0.25f;
    clearJob.clearJobDescriptor.color[1] = 0.5f;
    clearJob.clearJobDescriptor.color[2] = 0.75f;
    clearJob.clearJobDescriptor.color[3] = 1.0f;
    UNIT_TEST_CHECK(backendInterface.fpScheduleGpuJob(&backendInterface, &clearJob) == FFX_OK);

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(vkContext, commandBuffer, &errorMessage));
    UNIT_TEST_CHECK(backendInterface.fpExecuteGpuJobs(&backendInterface, ffxGetCommandListVK(commandBuffer), effectContextId) == FFX_OK);
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(vkContext, commandBuffer, &errorMessage));

    const FfxResource clearedResource = backendInterface.fpGetResource(&backendInterface, resource);
    UNIT_TEST_CHECK((clearedResource.description.flags & FFX_RESOURCE_FLAGS_BUFFER_ALIASED) != 0);
    UNIT_TEST_CHECK(clearedResource.state == FFX_RESOURCE_STATE_COPY_DEST);

    // FfxResource exposes the image handle; the backend-owned alias buffer stays private.
    unit_test::vk::Image readbackImage{};
    readbackImage.image             = reinterpret_cast<VkImage>(clearedResource.resource);
    readbackImage.createInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    readbackImage.createInfo.extent = {barrierTestImageWidth, barrierTestImageHeight, 1};
    readbackImage.layout            = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    std::vector<uint8_t> readbackPixels;
    UNIT_TEST_CHECK(unit_test::vk::ReadbackRgba8Image(vkContext, readbackImage, readbackPixels, &errorMessage));
    UNIT_TEST_CHECK(readbackPixels.size() == barrierTestImageByteSize);
    constexpr std::array<uint8_t, 4> expectedColor{64, 128, 191, 255};
    for (uint32_t channel = 0; channel < expectedColor.size(); ++channel)
    {
        const int difference = static_cast<int>(readbackPixels[channel]) - expectedColor[channel];
        UNIT_TEST_CHECK(difference >= -1 && difference <= 1);
    }

    UNIT_TEST_CHECK(backendInterface.fpDestroyResource(&backendInterface, resource, effectContextId) == FFX_OK);
    UNIT_TEST_CHECK(backendInterface.fpDestroyBackendContext(&backendInterface, effectContextId) == FFX_OK);
    unit_test::vk::DestroyVulkanContext(vkContext);
    return true;
}

UNIT_TEST_CASE(backend_vk, sdk_resource_undefined_state_is_shared_across_effect_contexts)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!unit_test::vk::InitializeVulkanContext(vkContext, &errorMessage))
    {
        UNIT_TEST_FAIL(std::string("Vulkan unavailable: ") + errorMessage);
    }

    if (!InitVulkanWrapper(MakeDeviceContext(vkContext)))
    {
        unit_test::vk::DestroyVulkanContext(vkContext);
        UNIT_TEST_FAIL("Vulkan wrapper initialization unsupported on this Vulkan runtime.");
    }

    FfxInterface         backendInterface{};
    std::vector<uint8_t> scratch;
    UNIT_TEST_CHECK(CreateBackendInterface(vkContext, backendInterface, scratch));

    uint32_t ownerEffectContextId  = 0;
    uint32_t firstEffectContextId  = 0;
    uint32_t secondEffectContextId = 0;
    UNIT_TEST_CHECK(backendInterface.fpCreateBackendContext(&backendInterface, FFX_EFFECT_SHAREDAPIBACKEND, nullptr, &ownerEffectContextId) == FFX_OK);
    UNIT_TEST_CHECK(backendInterface.fpCreateBackendContext(&backendInterface, ARM_EFFECT_OPTICALFLOW, nullptr, &firstEffectContextId) == FFX_OK);
    UNIT_TEST_CHECK(backendInterface.fpCreateBackendContext(&backendInterface, ARM_EFFECT_FRAMEINTERPOLATION, nullptr, &secondEffectContextId) == FFX_OK);

    FfxCreateResourceDescription createDescription{};
    createDescription.heapType                     = FFX_HEAP_TYPE_DEFAULT;
    createDescription.resourceDescription.type     = FFX_RESOURCE_TYPE_TEXTURE2D;
    createDescription.resourceDescription.format   = FFX_SURFACE_FORMAT_R8G8B8A8_UNORM;
    createDescription.resourceDescription.width    = 64;
    createDescription.resourceDescription.height   = 64;
    createDescription.resourceDescription.depth    = 1;
    createDescription.resourceDescription.mipCount = 1;
    createDescription.resourceDescription.flags    = FFX_RESOURCE_FLAGS_NONE;
    createDescription.resourceDescription.usage    = static_cast<FfxResourceUsage>(FFX_RESOURCE_USAGE_READ_ONLY | FFX_RESOURCE_USAGE_UAV);
    createDescription.initialState                 = FFX_RESOURCE_STATE_GENERIC_UAV;
    createDescription.name                         = "unit_test_shared_undefined_resource";
    createDescription.id                           = 1;
    createDescription.initData.type                = FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED;

    FfxResourceInternal ownerResource{};
    UNIT_TEST_CHECK(backendInterface.fpCreateResource(&backendInterface, &createDescription, ownerEffectContextId, &ownerResource) == FFX_OK);

    const FfxResource exportedResource = backendInterface.fpGetResource(&backendInterface, ownerResource);
    UNIT_TEST_CHECK((exportedResource.description.flags & FFX_RESOURCE_FLAGS_UNDEFINED) != 0);

    FfxResourceInternal firstRegisteredResource{};
    FfxResourceInternal secondRegisteredResource{};
    UNIT_TEST_CHECK(backendInterface.fpRegisterResource(&backendInterface, &exportedResource, firstEffectContextId, &firstRegisteredResource) == FFX_OK);
    UNIT_TEST_CHECK(backendInterface.fpRegisterResource(&backendInterface, &exportedResource, secondEffectContextId, &secondRegisteredResource) == FFX_OK);
    UNIT_TEST_CHECK(firstRegisteredResource.internalIndex != ownerResource.internalIndex);
    UNIT_TEST_CHECK(secondRegisteredResource.internalIndex != ownerResource.internalIndex);
    UNIT_TEST_CHECK(firstRegisteredResource.internalIndex != secondRegisteredResource.internalIndex);

    FfxGpuJobDescription barrierJob{};
    barrierJob.jobType                    = FFX_GPU_JOB_BARRIER;
    barrierJob.barrierDescriptor.resource = firstRegisteredResource;
    barrierJob.barrierDescriptor.newState = FFX_RESOURCE_STATE_COPY_DEST;
    UNIT_TEST_CHECK(backendInterface.fpScheduleGpuJob(&backendInterface, &barrierJob) == FFX_OK);

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(vkContext, commandBuffer, &errorMessage));
    UNIT_TEST_CHECK(backendInterface.fpExecuteGpuJobs(&backendInterface, ffxGetCommandListVK(commandBuffer), firstEffectContextId) == FFX_OK);
    UNIT_TEST_CHECK(backendInterface.fpUnregisterResources(&backendInterface, ffxGetCommandListVK(commandBuffer), firstEffectContextId) == FFX_OK);

    const FfxResource ownerAfterFirstUse = backendInterface.fpGetResource(&backendInterface, ownerResource);
    UNIT_TEST_CHECK((ownerAfterFirstUse.description.flags & FFX_RESOURCE_FLAGS_UNDEFINED) == 0);
    UNIT_TEST_CHECK(ownerAfterFirstUse.state == FFX_RESOURCE_STATE_GENERIC_UAV);

    const FfxResource secondRegistrationAfterFirstUse = backendInterface.fpGetResource(&backendInterface, secondRegisteredResource);
    UNIT_TEST_CHECK((secondRegistrationAfterFirstUse.description.flags & FFX_RESOURCE_FLAGS_UNDEFINED) == 0);

    barrierJob.barrierDescriptor.resource = secondRegisteredResource;
    barrierJob.barrierDescriptor.newState = FFX_RESOURCE_STATE_COMPUTE_READ;
    UNIT_TEST_CHECK(backendInterface.fpScheduleGpuJob(&backendInterface, &barrierJob) == FFX_OK);
    UNIT_TEST_CHECK(backendInterface.fpExecuteGpuJobs(&backendInterface, ffxGetCommandListVK(commandBuffer), secondEffectContextId) == FFX_OK);
    UNIT_TEST_CHECK(backendInterface.fpUnregisterResources(&backendInterface, ffxGetCommandListVK(commandBuffer), secondEffectContextId) == FFX_OK);
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(vkContext, commandBuffer, &errorMessage));

    UNIT_TEST_CHECK(backendInterface.fpDestroyBackendContext(&backendInterface, secondEffectContextId) == FFX_OK);
    UNIT_TEST_CHECK(backendInterface.fpDestroyBackendContext(&backendInterface, firstEffectContextId) == FFX_OK);
    UNIT_TEST_CHECK(backendInterface.fpDestroyResource(&backendInterface, ownerResource, ownerEffectContextId) == FFX_OK);
    UNIT_TEST_CHECK(backendInterface.fpDestroyBackendContext(&backendInterface, ownerEffectContextId) == FFX_OK);

    unit_test::vk::DestroyVulkanContext(vkContext);
    return true;
}
