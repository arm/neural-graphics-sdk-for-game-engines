/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "staging_pool.h"

#include "allocator.h"
#include "log.h"
#include "utils.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#ifdef FFX_BACKEND_VK
#include <FidelityFX/host/backends/vk/vk_wrapper.h>
#endif

namespace arm::capture
{
#ifdef FFX_BACKEND_VK
    namespace
    {
        class ScopedMemoryMap final
        {
        public:
            ScopedMemoryMap(const StagingPagePlatform& platform, VkDevice device, VkDeviceMemory memory)
                : platform_(platform)
                , device_(device)
                , memory_(memory)
            {
            }

            ~ScopedMemoryMap()
            {
                platform_.UnmapMemory(device_, memory_);
            }

            ScopedMemoryMap(const ScopedMemoryMap&) = delete;
            ScopedMemoryMap& operator=(const ScopedMemoryMap&) = delete;

        private:
            const StagingPagePlatform& platform_;
            VkDevice                   device_;
            VkDeviceMemory             memory_;
        };

        class UnitRanges
        {
        public:
            UnitRanges() = default;
            explicit UnitRanges(size_t unitCount)
                : unitCount_(unitCount)
            {
                if (unitCount != 0)
                    availableRanges_.push_back({0, static_cast<uint32_t>(unitCount)});
            }

            size_t Capacity() const
            {
                return unitCount_;
            }

            uint32_t Capacity(uint32_t requiredUnits, uint32_t limit) const
            {
                if (requiredUnits == 0)
                    return 0;

                uint32_t blockCount = 0;
                for (const AvailableRange& range : availableRanges_)
                {
                    blockCount += range.count / requiredUnits;
                    if (blockCount >= limit)
                        return limit;
                }
                return blockCount;
            }

            std::optional<uint32_t> Claim(uint32_t requiredUnits)
            {
                for (auto it = availableRanges_.begin(); it != availableRanges_.end(); ++it)
                {
                    if (it->count < requiredUnits)
                        continue;

                    const uint32_t first = it->first;
                    it->first += requiredUnits;
                    it->count -= requiredUnits;
                    if (it->count == 0)
                        availableRanges_.erase(it);
                    return first;
                }
                return std::nullopt;
            }

            void ReleaseRange(uint32_t firstUnit, uint32_t unitCount)
            {
                if (unitCount == 0)
                    return;

                auto next = std::lower_bound(availableRanges_.begin(), availableRanges_.end(), firstUnit, [](const AvailableRange& range, uint32_t first) {
                    return range.first < first;
                });
                const uint64_t end = static_cast<uint64_t>(firstUnit) + unitCount;
                if (next != availableRanges_.begin())
                {
                    auto previous = std::prev(next);
                    if (static_cast<uint64_t>(previous->first) + previous->count == firstUnit)
                    {
                        previous->count += unitCount;
                        if (next != availableRanges_.end() && end == next->first)
                        {
                            previous->count += next->count;
                            availableRanges_.erase(next);
                        }
                        return;
                    }
                }
                if (next != availableRanges_.end() && end == next->first)
                {
                    next->first = firstUnit;
                    next->count += unitCount;
                    return;
                }
                availableRanges_.insert(next, {firstUnit, unitCount});
            }

        private:
            struct AvailableRange
            {
                uint32_t first;
                uint32_t count;
            };

            std::vector<AvailableRange> availableRanges_;
            size_t                      unitCount_ = 0;
        };
    }  // namespace

    struct StagingPage
    {
        VkBuffer       buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize   bytes  = 0;
        UnitRanges     unitRanges;
    };

    StagingPool::~StagingPool()
    {
        ReleasePages();
    }

    void StagingPool::MarkReleaseUnsafe()
    {
        releaseSafe_.store(false, std::memory_order_release);
    }

    void StagingPool::Abandon()
    {
        MarkReleaseUnsafe();
    }

    bool StagingPool::IsReleaseSafe() const
    {
        return releaseSafe_.load(std::memory_order_acquire);
    }

    void StagingPool::ReleasePages()
    {
        std::unique_ptr<StagingPageAllocator> allocator;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            allocator = std::move(allocator_);
        }
        allocator.reset();

        std::lock_guard<std::mutex> lock(mutex_);
        if (IsReleaseSafe())
        {
            for (const StagingPage& page : pages_)
            {
                if (page.buffer != VK_NULL_HANDLE)
                    platform_->DestroyBuffer(device_, page.buffer);
                if (page.memory != VK_NULL_HANDLE)
                    platform_->FreeMemory(device_, page.memory);
            }
        }
        pages_.clear();
    }

    std::shared_ptr<StagingPool> StagingPool::Create(VkDevice                             device,
                                                     VkPhysicalDevice                     physicalDevice,
                                                     const StagingPoolConfig&             config,
                                                     std::shared_ptr<StagingPagePlatform> platform)
    {
        if (device == VK_NULL_HANDLE || physicalDevice == VK_NULL_HANDLE)
            return nullptr;

        auto               pagePlatform    = platform ? std::move(platform) : CreateDefaultStagingPagePlatform();
        const VkDeviceSize offsetAlignment = pagePlatform->GetStagingPoolOffsetAlignment(physicalDevice);
        const VkDeviceSize unitSize        = detail::StagingAllocationUnitSize(offsetAlignment);
        if (unitSize == 0)
            return nullptr;

        auto pool              = std::shared_ptr<StagingPool>(new StagingPool());
        pool->device_          = device;
        pool->physicalDevice_  = physicalDevice;
        pool->unitSize_        = unitSize;
        pool->offsetAlignment_ = offsetAlignment;
        pool->platform_        = std::move(pagePlatform);
        const StagingPageAllocatorConfig allocatorConfig{unitSize, config.hostVisibleMemoryRatio};
        pool->allocator_ = std::make_unique<StagingPageAllocator>(device, physicalDevice, allocatorConfig, pool.get(), pool->platform_);
        return pool;
    }

    bool StagingPool::Matches(VkDevice device, VkPhysicalDevice physicalDevice) const
    {
        return device_ == device && physicalDevice_ == physicalDevice;
    }

    bool StagingPool::UsesDevice(VkDevice device) const
    {
        return device_ == device;
    }

    VkDeviceSize StagingPool::OffsetAlignment() const
    {
        return offsetAlignment_;
    }

    bool StagingPool::Supports(VkDeviceSize blockBytes) const
    {
        return allocator_ && blockBytes != 0 && blockBytes % unitSize_ == 0 && allocator_->CanAllocateBlock(blockBytes);
    }

    StagingBlock StagingPool::Acquire(VkDeviceSize blockBytes)
    {
        if (!Supports(blockBytes))
            return {};

        auto tryAcquire = [&]() {
            std::lock_guard<std::mutex> lock(mutex_);
            return AcquireRangeLocked(blockBytes);
        };

        StagingBlock block = tryAcquire();
        if (!block)
        {
            if (!allocator_->EnsurePageForBlock(blockBytes))
                return {};
            block = tryAcquire();
            if (!block)
                return {};
        }

        return block;
    }

    void StagingPool::Return(StagingBlock& block)
    {
        if (!block)
            return;
        std::lock_guard<std::mutex> lock(mutex_);
        if (block.pageIndex < pages_.size())
        {
            StagingPage&   page  = pages_[block.pageIndex];
            const uint32_t first = static_cast<uint32_t>(block.offset / unitSize_);
            if (block.buffer == page.buffer && block.memory == page.memory && first <= page.unitRanges.Capacity() &&
                block.unitCount <= page.unitRanges.Capacity() - first)
            {
                page.unitRanges.ReleaseRange(first, block.unitCount);
            }
        }
        block = {};
    }

    VkResult StagingPool::Readback(const StagingBlock& block, VkDeviceSize offset, VkDeviceSize size, const std::function<void(void*)>& consume) const
    {
        void*          mapped = nullptr;
        const VkResult result = platform_->MapMemory(device_, block.memory, block.offset + offset, size, &mapped);
        if (result != VK_SUCCESS)
            return result;

        const ScopedMemoryMap mapping(*platform_, device_, block.memory);
        consume(mapped);
        return VK_SUCCESS;
    }

    StagingBlock StagingPool::AcquireRangeLocked(VkDeviceSize bytes)
    {
        const uint32_t needed = static_cast<uint32_t>(bytes / unitSize_);
        for (uint32_t pageIndex = 0; pageIndex < pages_.size(); ++pageIndex)
        {
            StagingPage& page = pages_[pageIndex];
            if (const std::optional<uint32_t> first = page.unitRanges.Claim(needed))
                return {page.buffer, page.memory, static_cast<VkDeviceSize>(*first) * unitSize_, bytes, pageIndex, needed};
        }
        return {};
    }

    bool StagingPool::HasAtLeastBlocksLocked(VkDeviceSize bytes, uint32_t minimumBlocks) const
    {
        const uint32_t needed          = static_cast<uint32_t>(bytes / unitSize_);
        uint32_t       availableBlocks = 0;
        for (const StagingPage& page : pages_)
        {
            availableBlocks += page.unitRanges.Capacity(needed, minimumBlocks - availableBlocks);
            if (availableBlocks >= minimumBlocks)
                return true;
        }
        return false;
    }

    void StagingPool::PublishPage(VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize bytes)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        StagingPage                 page{};
        page.buffer     = buffer;
        page.memory     = memory;
        page.bytes      = bytes;
        page.unitRanges = UnitRanges(static_cast<size_t>(bytes / unitSize_));
        pages_.push_back(std::move(page));
        ARM_CAPTURE_LOG_INFO("POOL", "allocated shared page=%llu MiB", static_cast<unsigned long long>(pages_.back().bytes / (1024 * 1024)));
    }

#endif

}  // namespace arm::capture
