// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#pragma once

#include <FidelityFX/host/backends/vk/vk_wrapper.h>

/// @brief Helper class for loading Vulkan function pointers
class VkFuncLoader
{
public:
    explicit VkFuncLoader(const VkDeviceContext& vkDeviceContext)
        : getInstanceProcAddrPtr(vkDeviceContext.vkGetInstanceProcAddr)
        , getDeviceProcAddrPtr(vkDeviceContext.vkDeviceProcAddr)
        , device(vkDeviceContext.vkDevice)
        , instance(vkDeviceContext.vkInstance)
    {
        FFX_ASSERT(NULL != getInstanceProcAddrPtr);
        FFX_ASSERT(NULL != getDeviceProcAddrPtr);
        FFX_ASSERT(VK_NULL_HANDLE != device);
        FFX_ASSERT(VK_NULL_HANDLE != instance);
    }

    template <typename T>
    bool getDeviceProc(T& funcPtr, const char* name)
    {
        return getProcInternal(funcPtr, name, getDeviceProcAddrPtr, device);
    }

    template <typename T>
    bool getInstanceProc(T& funcPtr, const char* name)
    {
        return getProcInternal(funcPtr, name, getInstanceProcAddrPtr, instance);
    }

private:
    template <typename T, typename FuncType, typename HandleType>
    bool getProcInternal(T& funcPtr, const char* name, const FuncType& getProcAddrPtr, HandleType handle)
    {
        FFX_ASSERT(NULL != getProcAddrPtr);

        funcPtr = reinterpret_cast<T>(getProcAddrPtr(handle, name));

        if (!funcPtr)
        {
            // Try to load the khr version function if the function is not available
            char aliasNameKHR[128];
            snprintf(aliasNameKHR, sizeof(aliasNameKHR), "%sKHR", name);
            funcPtr = reinterpret_cast<T>(getProcAddrPtr(handle, aliasNameKHR));
        }

        if (!funcPtr)
        {
            // Try to load the ext version function if the function is not available
            char aliasNameEXT[128];
            snprintf(aliasNameEXT, sizeof(aliasNameEXT), "%sEXT", name);
            funcPtr = reinterpret_cast<T>(getProcAddrPtr(handle, aliasNameEXT));
        }

        return funcPtr != nullptr;
    }

    PFN_vkGetInstanceProcAddr getInstanceProcAddrPtr;
    PFN_vkGetDeviceProcAddr   getDeviceProcAddrPtr;
    VkInstance                instance = VK_NULL_HANDLE;
    VkDevice                  device   = VK_NULL_HANDLE;
};

static bool LoadVulkanFunctions(const VkDeviceContext& vkDeviceContext, VkFunctionTable& tb)
{
    // Load the required Vulkan functions
    VkFuncLoader loader(vkDeviceContext);
    bool         success = true;

    // vulkan instance level functions
    {
        success &= loader.getInstanceProc(tb.vkEnumerateDeviceExtensionProperties, "vkEnumerateDeviceExtensionProperties");
        success &= loader.getInstanceProc(tb.vkGetPhysicalDeviceProperties, "vkGetPhysicalDeviceProperties");
        success &= loader.getInstanceProc(tb.vkGetPhysicalDeviceMemoryProperties, "vkGetPhysicalDeviceMemoryProperties");
        success &= loader.getInstanceProc(tb.vkGetPhysicalDeviceFeatures2, "vkGetPhysicalDeviceFeatures2");
        success &= loader.getInstanceProc(tb.vkGetPhysicalDeviceProperties2, "vkGetPhysicalDeviceProperties2");
        success &= loader.getInstanceProc(tb.vkGetPhysicalDeviceQueueFamilyProperties, "vkGetPhysicalDeviceQueueFamilyProperties");
        success &= loader.getInstanceProc(tb.vkGetPhysicalDeviceSurfaceSupportKHR, "vkGetPhysicalDeviceSurfaceSupportKHR");

        // Optional ML extensions for Vulkan support
        loader.getInstanceProc(tb.vkGetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM,
                               "vkGetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM");
    }

    // vulkan device level functions
    {
        success &= loader.getDeviceProc(tb.vkGetDeviceProcAddr, "vkGetDeviceProcAddr");
        success &= loader.getDeviceProc(tb.vkFlushMappedMemoryRanges, "vkFlushMappedMemoryRanges");
        success &= loader.getDeviceProc(tb.vkCreateDescriptorPool, "vkCreateDescriptorPool");
        success &= loader.getDeviceProc(tb.vkCreateSampler, "vkCreateSampler");

        success &= loader.getDeviceProc(tb.vkCreateDescriptorSetLayout, "vkCreateDescriptorSetLayout");
        success &= loader.getDeviceProc(tb.vkCreateBuffer, "vkCreateBuffer");
        success &= loader.getDeviceProc(tb.vkCreateBufferView, "vkCreateBufferView");
        success &= loader.getDeviceProc(tb.vkCreateImage, "vkCreateImage");
        success &= loader.getDeviceProc(tb.vkCreateImageView, "vkCreateImageView");
        success &= loader.getDeviceProc(tb.vkGetImageSubresourceLayout, "vkGetImageSubresourceLayout");

        success &= loader.getDeviceProc(tb.vkCreateShaderModule, "vkCreateShaderModule");
        success &= loader.getDeviceProc(tb.vkCreatePipelineLayout, "vkCreatePipelineLayout");
        success &= loader.getDeviceProc(tb.vkCreateComputePipelines, "vkCreateComputePipelines");
        success &= loader.getDeviceProc(tb.vkCmdPushConstants, "vkCmdPushConstants");

        success &= loader.getDeviceProc(tb.vkCreateGraphicsPipelines, "vkCreateGraphicsPipelines");
        success &= loader.getDeviceProc(tb.vkCreateRenderPass, "vkCreateRenderPass");
        success &= loader.getDeviceProc(tb.vkCreateFramebuffer, "vkCreateFramebuffer");

        success &= loader.getDeviceProc(tb.vkDestroyPipelineLayout, "vkDestroyPipelineLayout");
        success &= loader.getDeviceProc(tb.vkDestroyPipeline, "vkDestroyPipeline");

        success &= loader.getDeviceProc(tb.vkDestroyRenderPass, "vkDestroyRenderPass");
        success &= loader.getDeviceProc(tb.vkDestroyFramebuffer, "vkDestroyFramebuffer");

        success &= loader.getDeviceProc(tb.vkDestroyImage, "vkDestroyImage");
        success &= loader.getDeviceProc(tb.vkDestroyImageView, "vkDestroyImageView");
        success &= loader.getDeviceProc(tb.vkDestroyBuffer, "vkDestroyBuffer");
        success &= loader.getDeviceProc(tb.vkDestroyBufferView, "vkDestroyBufferView");
        success &= loader.getDeviceProc(tb.vkDestroyDescriptorSetLayout, "vkDestroyDescriptorSetLayout");
        success &= loader.getDeviceProc(tb.vkDestroyDescriptorPool, "vkDestroyDescriptorPool");
        success &= loader.getDeviceProc(tb.vkDestroySampler, "vkDestroySampler");
        success &= loader.getDeviceProc(tb.vkDestroyShaderModule, "vkDestroyShaderModule");
        success &= loader.getDeviceProc(tb.vkGetBufferMemoryRequirements, "vkGetBufferMemoryRequirements");
        success &= loader.getDeviceProc(tb.vkGetBufferMemoryRequirements2, "vkGetBufferMemoryRequirements2");
        success &= loader.getDeviceProc(tb.vkGetImageMemoryRequirements, "vkGetImageMemoryRequirements");
        success &= loader.getDeviceProc(tb.vkGetImageMemoryRequirements2, "vkGetImageMemoryRequirements2");
        success &= loader.getDeviceProc(tb.vkAllocateDescriptorSets, "vkAllocateDescriptorSets");
        success &= loader.getDeviceProc(tb.vkFreeDescriptorSets, "vkFreeDescriptorSets");
        success &= loader.getDeviceProc(tb.vkAllocateMemory, "vkAllocateMemory");
        success &= loader.getDeviceProc(tb.vkFreeMemory, "vkFreeMemory");
        success &= loader.getDeviceProc(tb.vkMapMemory, "vkMapMemory");
        success &= loader.getDeviceProc(tb.vkUnmapMemory, "vkUnmapMemory");
        success &= loader.getDeviceProc(tb.vkBindBufferMemory, "vkBindBufferMemory");
        success &= loader.getDeviceProc(tb.vkBindImageMemory, "vkBindImageMemory");
        success &= loader.getDeviceProc(tb.vkBindImageMemory2, "vkBindImageMemory2");
        success &= loader.getDeviceProc(tb.vkUpdateDescriptorSets, "vkUpdateDescriptorSets");
        success &= loader.getDeviceProc(tb.vkCmdPipelineBarrier, "vkCmdPipelineBarrier");
        success &= loader.getDeviceProc(tb.vkCmdPipelineBarrier2, "vkCmdPipelineBarrier2");
        success &= loader.getDeviceProc(tb.vkCmdBindPipeline, "vkCmdBindPipeline");
        success &= loader.getDeviceProc(tb.vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets");
        success &= loader.getDeviceProc(tb.vkCmdDispatch, "vkCmdDispatch");
        success &= loader.getDeviceProc(tb.vkCmdDispatchIndirect, "vkCmdDispatchIndirect");
        success &= loader.getDeviceProc(tb.vkCmdCopyBuffer, "vkCmdCopyBuffer");
        success &= loader.getDeviceProc(tb.vkCmdCopyImage, "vkCmdCopyImage");
        success &= loader.getDeviceProc(tb.vkCmdCopyBufferToImage, "vkCmdCopyBufferToImage");
        success &= loader.getDeviceProc(tb.vkCmdClearColorImage, "vkCmdClearColorImage");
        success &= loader.getDeviceProc(tb.vkCmdFillBuffer, "vkCmdFillBuffer");

        success &= loader.getDeviceProc(tb.vkCmdBeginRenderPass, "vkCmdBeginRenderPass");
        success &= loader.getDeviceProc(tb.vkCmdEndRenderPass, "vkCmdEndRenderPass");
        success &= loader.getDeviceProc(tb.vkCmdDraw, "vkCmdDraw");
        success &= loader.getDeviceProc(tb.vkCmdCopyImageToBuffer, "vkCmdCopyImageToBuffer");

        // Semaphore and synchronization functions
        success &= loader.getDeviceProc(tb.vkWaitSemaphores, "vkWaitSemaphores");
        success &= loader.getDeviceProc(tb.vkCreateSemaphore, "vkCreateSemaphore");
        success &= loader.getDeviceProc(tb.vkDestroySemaphore, "vkDestroySemaphore");
        success &= loader.getDeviceProc(tb.vkGetSemaphoreCounterValue, "vkGetSemaphoreCounterValue");

        // Swapchain functions
        success &= loader.getDeviceProc(tb.vkCreateSwapchainKHR, "vkCreateSwapchainKHR");
        success &= loader.getDeviceProc(tb.vkDestroySwapchainKHR, "vkDestroySwapchainKHR");
        success &= loader.getDeviceProc(tb.vkGetSwapchainImagesKHR, "vkGetSwapchainImagesKHR");
        success &= loader.getDeviceProc(tb.vkAcquireNextImageKHR, "vkAcquireNextImageKHR");

        // Queue functions
        success &= loader.getDeviceProc(tb.vkQueuePresentKHR, "vkQueuePresentKHR");
        success &= loader.getDeviceProc(tb.vkQueueWaitIdle, "vkQueueWaitIdle");
        success &= loader.getDeviceProc(tb.vkQueueSubmit, "vkQueueSubmit");
        success &= loader.getDeviceProc(tb.vkDeviceWaitIdle, "vkDeviceWaitIdle");

        // Command buffer and pool functions
        success &= loader.getDeviceProc(tb.vkCreateCommandPool, "vkCreateCommandPool");
        success &= loader.getDeviceProc(tb.vkDestroyCommandPool, "vkDestroyCommandPool");
        success &= loader.getDeviceProc(tb.vkAllocateCommandBuffers, "vkAllocateCommandBuffers");
        success &= loader.getDeviceProc(tb.vkFreeCommandBuffers, "vkFreeCommandBuffers");
        success &= loader.getDeviceProc(tb.vkResetCommandPool, "vkResetCommandPool");
        success &= loader.getDeviceProc(tb.vkResetCommandBuffer, "vkResetCommandBuffer");
        success &= loader.getDeviceProc(tb.vkBeginCommandBuffer, "vkBeginCommandBuffer");
        success &= loader.getDeviceProc(tb.vkEndCommandBuffer, "vkEndCommandBuffer");

        // Command buffer state functions
        success &= loader.getDeviceProc(tb.vkCmdSetViewport, "vkCmdSetViewport");
        success &= loader.getDeviceProc(tb.vkCmdSetScissor, "vkCmdSetScissor");

        // Fence functions
        success &= loader.getDeviceProc(tb.vkCreateFence, "vkCreateFence");
        success &= loader.getDeviceProc(tb.vkDestroyFence, "vkDestroyFence");
        success &= loader.getDeviceProc(tb.vkGetFenceStatus, "vkGetFenceStatus");
        success &= loader.getDeviceProc(tb.vkWaitForFences, "vkWaitForFences");

        // Optional debug markers
        loader.getDeviceProc(tb.vkSetDebugUtilsObjectNameEXT, "vkSetDebugUtilsObjectNameEXT");
        loader.getDeviceProc(tb.vkCmdWriteBufferMarkerAMD, "vkCmdWriteBufferMarkerAMD");
        loader.getDeviceProc(tb.vkCmdWriteBufferMarker2AMD, "vkCmdWriteBufferMarker2AMD");
        loader.getDeviceProc(tb.vkCmdBeginDebugUtilsLabelEXT, "vkCmdBeginDebugUtilsLabelEXT");
        loader.getDeviceProc(tb.vkCmdEndDebugUtilsLabelEXT, "vkCmdEndDebugUtilsLabelEXT");

        // Optional ML extensions for Vulkan support
        loader.getDeviceProc(tb.vkCreateTensorARM, "vkCreateTensorARM");
        loader.getDeviceProc(tb.vkCreateTensorViewARM, "vkCreateTensorViewARM");
        loader.getDeviceProc(tb.vkGetTensorMemoryRequirementsARM, "vkGetTensorMemoryRequirementsARM");
        loader.getDeviceProc(tb.vkBindTensorMemoryARM, "vkBindTensorMemoryARM");
        loader.getDeviceProc(tb.vkCreateDataGraphPipelinesARM, "vkCreateDataGraphPipelinesARM");
        loader.getDeviceProc(tb.vkCreateDataGraphPipelineSessionARM, "vkCreateDataGraphPipelineSessionARM");
        loader.getDeviceProc(tb.vkCmdDispatchDataGraphARM, "vkCmdDispatchDataGraphARM");
        loader.getDeviceProc(tb.vkGetDataGraphPipelineSessionBindPointRequirementsARM, "vkGetDataGraphPipelineSessionBindPointRequirementsARM");
        loader.getDeviceProc(tb.vkGetDataGraphPipelineSessionMemoryRequirementsARM, "vkGetDataGraphPipelineSessionMemoryRequirementsARM");
        loader.getDeviceProc(tb.vkBindDataGraphPipelineSessionMemoryARM, "vkBindDataGraphPipelineSessionMemoryARM");
        loader.getDeviceProc(tb.vkDestroyDataGraphPipelineSessionARM, "vkDestroyDataGraphPipelineSessionARM");
        loader.getDeviceProc(tb.vkDestroyTensorARM, "vkDestroyTensorARM");
        loader.getDeviceProc(tb.vkDestroyTensorViewARM, "vkDestroyTensorViewARM");
    }

    tb.initialized = success;

    return success;
}

/// @brief Returns a reference to the singleton Vulkan function table
/// @return Reference to VKFunctionTable containing all Vulkan function pointers
///
/// @endcode
static VkFunctionTable& GetVulkanFunctionTable()
{
    static VkFunctionTable instance;
    return instance;
}

bool InitVulkanWrapper(const VkDeviceContext& deviceContext)
{
    auto& vktb = GetVulkanFunctionTable();
    if (!vktb.initialized)
    {
        return LoadVulkanFunctions(deviceContext, vktb);
    }

    return true;
}

const VkFunctionTable& VulkanWrapper()
{
    const auto& vktb = GetVulkanFunctionTable();
    FFX_ASSERT(vktb.initialized && "Vulkan function table not initialized!");
    return vktb;
}
