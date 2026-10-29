/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "allocator.h"
#include "config.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <functional>
#include <vector>

#ifdef FFX_BACKEND_VK
#include <vulkan/vulkan.h>
#endif

namespace arm::capture
{
#ifdef FFX_BACKEND_VK
    namespace detail
    {
        class Worker;
    }

    class StagingPool;
    class StagingPagePlatform;
    struct StagingPage;

    struct StagingBlock
    {
        VkBuffer       buffer    = VK_NULL_HANDLE;
        VkDeviceMemory memory    = VK_NULL_HANDLE;
        VkDeviceSize   offset    = 0;
        VkDeviceSize   size      = 0;
        uint32_t       pageIndex = UINT32_MAX;
        uint32_t       unitCount = 0;

        explicit operator bool() const
        {
            return buffer != VK_NULL_HANDLE;
        }
    };

    // The allocator is shared per VkDevice and owns global byte-budget accounting.
    // Acquisition creates a page synchronously when no published page is available.
    class StagingPool : public std::enable_shared_from_this<StagingPool>, public StagingPageSink
    {
    public:
        ~StagingPool();

        static std::shared_ptr<StagingPool> Create(VkDevice                             device,
                                                   VkPhysicalDevice                     physicalDevice,
                                                   const StagingPoolConfig&             config,
                                                   std::shared_ptr<StagingPagePlatform> platform = nullptr);

        bool         Matches(VkDevice device, VkPhysicalDevice physicalDevice) const;
        bool         UsesDevice(VkDevice device) const;
        VkDeviceSize OffsetAlignment() const;
        bool         Supports(VkDeviceSize blockBytes) const;
        StagingBlock Acquire(VkDeviceSize blockBytes);
        void         Return(StagingBlock& block);
        // Permanently suppress Vulkan buffer/memory destruction after device loss.
        void Abandon();

    private:
        friend class detail::Worker;

        VkResult     Readback(const StagingBlock& block, VkDeviceSize offset, VkDeviceSize size, const std::function<void(void*)>& consume) const;
        StagingBlock AcquireRangeLocked(VkDeviceSize bytes);
        bool         HasAtLeastBlocksLocked(VkDeviceSize bytes, uint32_t minimumBlocks) const;
        void         PublishPage(VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize bytes) override;
        void         MarkReleaseUnsafe() override;
        bool         IsReleaseSafe() const override;
        void         ReleasePages();

        VkDevice                              device_          = VK_NULL_HANDLE;
        VkPhysicalDevice                      physicalDevice_  = VK_NULL_HANDLE;
        VkDeviceSize                          unitSize_        = 0;
        VkDeviceSize                          offsetAlignment_ = 1;
        mutable std::mutex                    mutex_;
        std::atomic_bool                      releaseSafe_{true};
        std::unique_ptr<StagingPageAllocator> allocator_;
        std::shared_ptr<StagingPagePlatform>  platform_;
        std::vector<StagingPage>              pages_;
    };

#endif

}  // namespace arm::capture
