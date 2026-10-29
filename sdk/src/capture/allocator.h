/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "config.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#ifdef FFX_BACKEND_VK
#include <vulkan/vulkan.h>
#endif

namespace arm::capture
{
#ifdef FFX_BACKEND_VK
    class StagingPagePlatform
    {
    public:
        virtual ~StagingPagePlatform() = default;

        virtual void     GetPhysicalDeviceMemoryProperties(VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties* properties) const = 0;
        virtual void     GetPhysicalDeviceProperties2(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2* properties) const           = 0;
        virtual VkResult CreateBuffer(VkDevice device, const VkBufferCreateInfo* createInfo, VkBuffer* buffer) const                            = 0;
        virtual void     GetBufferMemoryRequirements(VkDevice device, VkBuffer buffer, VkMemoryRequirements* requirements) const                = 0;
        virtual VkResult AllocateMemory(VkDevice device, const VkMemoryAllocateInfo* allocateInfo, VkDeviceMemory* memory) const                = 0;
        virtual VkResult BindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset) const                   = 0;
        virtual VkResult MapMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, void** data) const           = 0;
        virtual void     UnmapMemory(VkDevice device, VkDeviceMemory memory) const                                                              = 0;
        virtual void     DestroyBuffer(VkDevice device, VkBuffer buffer) const                                                                  = 0;
        virtual void     FreeMemory(VkDevice device, VkDeviceMemory memory) const                                                               = 0;

        VkDeviceSize GetHostVisibleHeapBytes(VkPhysicalDevice physicalDevice) const;
        VkDeviceSize GetMaxMemoryAllocationSize(VkPhysicalDevice physicalDevice) const;
        VkDeviceSize GetStagingPoolOffsetAlignment(VkPhysicalDevice physicalDevice) const;
        uint32_t     FindHostMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter) const;
    };

    std::shared_ptr<StagingPagePlatform> CreateDefaultStagingPagePlatform();

    VkDeviceSize GetStagingPoolOffsetAlignment(VkPhysicalDevice physicalDevice);

    class StagingPageSink
    {
    public:
        virtual ~StagingPageSink() = default;

        virtual void PublishPage(VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize bytes) = 0;
        virtual void MarkReleaseUnsafe()                                                     = 0;
        virtual bool IsReleaseSafe() const                                                   = 0;
    };

    struct StagingPageAllocatorConfig
    {
        VkDeviceSize unitSize               = 0;
        float        hostVisibleMemoryRatio = 0.125f;
    };

    class StagingPageAllocator
    {
    public:
        struct Request
        {
            // The request can shrink to the remaining budget, but it must still
            // fit one useful block.
            VkDeviceSize desiredBytes = 0;
            VkDeviceSize minimumBytes = 0;
        };

        StagingPageAllocator(VkDevice                             device,
                             VkPhysicalDevice                     physicalDevice,
                             const StagingPageAllocatorConfig&    config,
                             StagingPageSink*                     pageSink,
                             std::shared_ptr<StagingPagePlatform> platform = nullptr);
        ~StagingPageAllocator();

        bool CanAllocateBlock(VkDeviceSize blockBytes) const;
        // Must allocate a page or fail.
        bool EnsurePageForBlock(VkDeviceSize blockBytes);

    private:
        struct PageCreateResult
        {
            VkBuffer       buffer          = VK_NULL_HANDLE;
            VkDeviceMemory memory          = VK_NULL_HANDLE;
            VkDeviceSize   bytes           = 0;
            VkDeviceSize   allocationBytes = 0;
            bool           bound           = false;
            VkResult       result          = VK_SUCCESS;
        };

        void             UpdateWithRetained(const Request& request, VkDeviceSize retainedBytes);
        bool             ReserveLocked(Request& request);
        bool             ReserveActualAllocationLocked(Request& request, VkDeviceSize allocationBytes);
        VkDeviceSize     AvailableBudgetLocked() const;
        bool             ExecuteOrEnqueue(Request request);
        bool             Process(Request request);
        PageCreateResult AllocatePage(Request& request);
        bool             ReleasePage(PageCreateResult& page, bool markReleaseUnsafe);

        VkDevice                             device_             = VK_NULL_HANDLE;
        VkPhysicalDevice                     physicalDevice_     = VK_NULL_HANDLE;
        VkDeviceSize                         unitSize_           = 0;
        VkDeviceSize                         budgetBytes_        = 0;
        VkDeviceSize                         maxAllocationBytes_ = 0;
        StagingPageSink*                     pageSink_           = nullptr;
        std::shared_ptr<StagingPagePlatform> platform_;
        mutable std::mutex                   mutex_;
        // Includes completed allocations and reservations for requests currently
        // being allocated.
        VkDeviceSize usedBudgetBytes_ = 0;
    };
#endif

}  // namespace arm::capture
