/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "allocator.h"

#include "log.h"
#include "utils.h"

#include <algorithm>
#include <limits>
#include <new>
#include <system_error>
#include <utility>

#ifdef FFX_BACKEND_VK
#include <FidelityFX/host/backends/vk/vk_wrapper.h>
#endif

namespace arm::capture
{
#ifdef FFX_BACKEND_VK
    namespace
    {
        class DefaultStagingPagePlatform final : public StagingPagePlatform
        {
        public:
            void GetPhysicalDeviceMemoryProperties(VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties* properties) const override
            {
                VulkanWrapper().vkGetPhysicalDeviceMemoryProperties(physicalDevice, properties);
            }

            void GetPhysicalDeviceProperties2(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2* properties) const override
            {
                if (VulkanWrapper().vkGetPhysicalDeviceProperties2 != nullptr)
                    VulkanWrapper().vkGetPhysicalDeviceProperties2(physicalDevice, properties);
            }

            VkResult CreateBuffer(VkDevice device, const VkBufferCreateInfo* createInfo, VkBuffer* buffer) const override
            {
                return VulkanWrapper().vkCreateBuffer(device, createInfo, nullptr, buffer);
            }

            void GetBufferMemoryRequirements(VkDevice device, VkBuffer buffer, VkMemoryRequirements* requirements) const override
            {
                VulkanWrapper().vkGetBufferMemoryRequirements(device, buffer, requirements);
            }

            VkResult AllocateMemory(VkDevice device, const VkMemoryAllocateInfo* allocateInfo, VkDeviceMemory* memory) const override
            {
                return VulkanWrapper().vkAllocateMemory(device, allocateInfo, nullptr, memory);
            }

            VkResult BindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset) const override
            {
                return VulkanWrapper().vkBindBufferMemory(device, buffer, memory, offset);
            }

            VkResult MapMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, void** data) const override
            {
                return VulkanWrapper().vkMapMemory(device, memory, offset, size, 0, data);
            }

            void UnmapMemory(VkDevice device, VkDeviceMemory memory) const override
            {
                VulkanWrapper().vkUnmapMemory(device, memory);
            }

            void DestroyBuffer(VkDevice device, VkBuffer buffer) const override
            {
                VulkanWrapper().vkDestroyBuffer(device, buffer, nullptr);
            }

            void FreeMemory(VkDevice device, VkDeviceMemory memory) const override
            {
                VulkanWrapper().vkFreeMemory(device, memory, nullptr);
            }
        };

        constexpr VkDeviceSize kNormalPageSlotBudgetBytes = 256ull * 1024 * 1024;

        uint32_t NormalPageSlots(VkDeviceSize blockBytes)
        {
            const VkDeviceSize slots = blockBytes == 0 ? 1 : kNormalPageSlotBudgetBytes / blockBytes;
            return static_cast<uint32_t>(std::max<VkDeviceSize>(1, std::min<VkDeviceSize>(slots, std::numeric_limits<uint32_t>::max())));
        }

        VkDeviceSize PageBytes(VkDeviceSize blockBytes, uint32_t slots, VkDeviceSize maxAllocationBytes, VkDeviceSize unitSize)
        {
            const VkDeviceSize bytes =
                blockBytes <= std::numeric_limits<VkDeviceSize>::max() / slots ? blockBytes * slots : std::numeric_limits<VkDeviceSize>::max();
            return detail::AlignUp(std::min(bytes, maxAllocationBytes == 0 ? bytes : maxAllocationBytes), unitSize);
        }

        VkDeviceSize FitPageBytes(VkDeviceSize desiredBytes, VkDeviceSize availableBytes, VkDeviceSize maxAllocationBytes, VkDeviceSize blockBytes)
        {
            VkDeviceSize pageBytes = std::min(desiredBytes, availableBytes);
            if (maxAllocationBytes != 0)
                pageBytes = std::min(pageBytes, maxAllocationBytes);
            return pageBytes - pageBytes % blockBytes;
        }

    }  // namespace

    VkDeviceSize StagingPagePlatform::GetHostVisibleHeapBytes(VkPhysicalDevice physicalDevice) const
    {
        VkPhysicalDeviceMemoryProperties properties{};
        GetPhysicalDeviceMemoryProperties(physicalDevice, &properties);
        bool hostVisibleHeap[VK_MAX_MEMORY_HEAPS] = {};
        for (uint32_t index = 0; index < properties.memoryTypeCount; ++index)
        {
            if ((properties.memoryTypes[index].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
                hostVisibleHeap[properties.memoryTypes[index].heapIndex] = true;
        }

        VkDeviceSize bytes = 0;
        for (uint32_t index = 0; index < properties.memoryHeapCount; ++index)
        {
            if (hostVisibleHeap[index] && bytes <= std::numeric_limits<VkDeviceSize>::max() - properties.memoryHeaps[index].size)
                bytes += properties.memoryHeaps[index].size;
        }
        return bytes;
    }

    VkDeviceSize StagingPagePlatform::GetMaxMemoryAllocationSize(VkPhysicalDevice physicalDevice) const
    {
        VkPhysicalDeviceMaintenance3Properties maintenance3{};
        maintenance3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
        VkPhysicalDeviceProperties2 properties{};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties.pNext = &maintenance3;
        GetPhysicalDeviceProperties2(physicalDevice, &properties);
        return maintenance3.maxMemoryAllocationSize;
    }

    VkDeviceSize StagingPagePlatform::GetStagingPoolOffsetAlignment(VkPhysicalDevice physicalDevice) const
    {
        VkPhysicalDeviceProperties2 properties{};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        GetPhysicalDeviceProperties2(physicalDevice, &properties);
        return std::max(properties.properties.limits.optimalBufferCopyOffsetAlignment, properties.properties.limits.minMemoryMapAlignment);
    }

    uint32_t StagingPagePlatform::FindHostMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter) const
    {
        VkPhysicalDeviceMemoryProperties properties{};
        GetPhysicalDeviceMemoryProperties(physicalDevice, &properties);
        const VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t index = 0; index < properties.memoryTypeCount; ++index)
        {
            const VkMemoryPropertyFlags flags = properties.memoryTypes[index].propertyFlags;
            if ((typeFilter & (1u << index)) != 0 && (flags & required) == required && (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0)
                return index;
        }
        for (uint32_t index = 0; index < properties.memoryTypeCount; ++index)
        {
            if ((typeFilter & (1u << index)) != 0 && (properties.memoryTypes[index].propertyFlags & required) == required)
                return index;
        }
        return UINT32_MAX;
    }

    std::shared_ptr<StagingPagePlatform> CreateDefaultStagingPagePlatform()
    {
        static std::shared_ptr<StagingPagePlatform> platform = std::make_shared<DefaultStagingPagePlatform>();
        return platform;
    }

    VkDeviceSize GetStagingPoolOffsetAlignment(VkPhysicalDevice physicalDevice)
    {
        return CreateDefaultStagingPagePlatform()->GetStagingPoolOffsetAlignment(physicalDevice);
    }

    StagingPageAllocator::StagingPageAllocator(VkDevice                             device,
                                               VkPhysicalDevice                     physicalDevice,
                                               const StagingPageAllocatorConfig&    config,
                                               StagingPageSink*                     pageSink,
                                               std::shared_ptr<StagingPagePlatform> platform)
        : device_(device)
        , physicalDevice_(physicalDevice)
        , unitSize_(config.unitSize)
        , pageSink_(pageSink)
        , platform_(platform ? std::move(platform) : CreateDefaultStagingPagePlatform())
    {
        const VkDeviceSize hostVisible = platform_->GetHostVisibleHeapBytes(physicalDevice_);
        budgetBytes_                   = static_cast<VkDeviceSize>(static_cast<double>(hostVisible) * config.hostVisibleMemoryRatio);
        maxAllocationBytes_            = platform_->GetMaxMemoryAllocationSize(physicalDevice_);
        if (budgetBytes_ < unitSize_)
            return;

        ARM_CAPTURE_LOG_INFO("POOL",
                             "mode=Serial hostVisible=%llu MiB ratio=%.3f budget=%llu MiB normalPageFormula=floor(256MiB/slot)",
                             static_cast<unsigned long long>(hostVisible / (1024 * 1024)),
                             static_cast<double>(config.hostVisibleMemoryRatio),
                             static_cast<unsigned long long>(budgetBytes_ / (1024 * 1024)));
    }

    StagingPageAllocator::PageCreateResult StagingPageAllocator::AllocatePage(Request& request)
    {
        PageCreateResult   result{};
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size        = request.desiredBytes;
        bufferInfo.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        result.result          = platform_->CreateBuffer(device_, &bufferInfo, &result.buffer);
        if (result.result != VK_SUCCESS)
            return result;

        VkMemoryRequirements requirements{};
        platform_->GetBufferMemoryRequirements(device_, result.buffer, &requirements);
        result.bytes           = request.desiredBytes;
        result.allocationBytes = requirements.size;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!ReserveActualAllocationLocked(request, requirements.size))
            {
                result.result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
                return result;
            }
        }
        const uint32_t memoryType = platform_->FindHostMemoryType(physicalDevice_, requirements.memoryTypeBits);
        if (memoryType == UINT32_MAX)
        {
            result.result = VK_ERROR_FEATURE_NOT_PRESENT;
            return result;
        }

        VkMemoryAllocateInfo allocation{};
        allocation.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize  = requirements.size;
        allocation.memoryTypeIndex = memoryType;
        result.result              = platform_->AllocateMemory(device_, &allocation, &result.memory);
        if (result.result != VK_SUCCESS)
            return result;

        result.result = platform_->BindBufferMemory(device_, result.buffer, result.memory, 0);
        result.bound  = result.result == VK_SUCCESS;
        return result;
    }

    StagingPageAllocator::~StagingPageAllocator() = default;

    bool StagingPageAllocator::CanAllocateBlock(VkDeviceSize blockBytes) const
    {
        return unitSize_ != 0 && blockBytes != 0 && blockBytes % unitSize_ == 0 && blockBytes / unitSize_ <= std::numeric_limits<uint32_t>::max() &&
               blockBytes <= budgetBytes_ && (maxAllocationBytes_ == 0 || blockBytes <= maxAllocationBytes_);
    }

    bool StagingPageAllocator::EnsurePageForBlock(VkDeviceSize blockBytes)
    {
        if (!CanAllocateBlock(blockBytes))
            return false;
        const uint32_t normalSlots = NormalPageSlots(blockBytes);
        return ExecuteOrEnqueue({PageBytes(blockBytes, normalSlots, maxAllocationBytes_, unitSize_), blockBytes});
    }

    bool StagingPageAllocator::ExecuteOrEnqueue(Request request)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!ReserveLocked(request))
                return false;
        }
        return Process(std::move(request));
    }

    // Update the reserved bytes before with bytes actually used.
    void StagingPageAllocator::UpdateWithRetained(const Request& request, VkDeviceSize retainedBytes)
    {
        // Replace the in-flight reservation with the memory that must remain
        // accounted for after this request completes.
        usedBudgetBytes_ = usedBudgetBytes_ - request.desiredBytes + retainedBytes;
    }

    bool StagingPageAllocator::ReserveLocked(Request& request)
    {
        const VkDeviceSize pageBytes = FitPageBytes(request.desiredBytes, AvailableBudgetLocked(), maxAllocationBytes_, request.minimumBytes);
        if (pageBytes < request.minimumBytes)
            return false;
        request.desiredBytes = pageBytes;
        usedBudgetBytes_ += pageBytes;
        return true;
    }

    bool StagingPageAllocator::ReserveActualAllocationLocked(Request& request, VkDeviceSize allocationBytes)
    {
        if ((maxAllocationBytes_ != 0 && allocationBytes > maxAllocationBytes_) || usedBudgetBytes_ < request.desiredBytes)
            return false;

        const VkDeviceSize completedBytes = usedBudgetBytes_ - request.desiredBytes;
        if (completedBytes > budgetBytes_ || allocationBytes > budgetBytes_ - completedBytes)
            return false;

        usedBudgetBytes_     = completedBytes + allocationBytes;
        request.desiredBytes = allocationBytes;
        return true;
    }

    VkDeviceSize StagingPageAllocator::AvailableBudgetLocked() const
    {
        return budgetBytes_ > usedBudgetBytes_ ? budgetBytes_ - usedBudgetBytes_ : 0;
    }

    bool StagingPageAllocator::ReleasePage(PageCreateResult& page, bool markReleaseUnsafe)
    {
        if (markReleaseUnsafe)
            pageSink_->MarkReleaseUnsafe();
        const bool releaseSafe = pageSink_->IsReleaseSafe();
        if (releaseSafe)
        {
            if (page.buffer != VK_NULL_HANDLE)
                platform_->DestroyBuffer(device_, page.buffer);
            if (page.memory != VK_NULL_HANDLE)
                platform_->FreeMemory(device_, page.memory);
        }
        return releaseSafe;
    }

    bool StagingPageAllocator::Process(Request request)
    {
        PageCreateResult page    = AllocatePage(request);
        const bool       success = page.bound && page.buffer != VK_NULL_HANDLE && page.memory != VK_NULL_HANDLE && page.bytes != 0;
        if (success)
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                UpdateWithRetained(request, page.allocationBytes);
            }
            pageSink_->PublishPage(page.buffer, page.memory, page.bytes);
        }
        else
        {
            const bool                  releaseSafe = ReleasePage(page, page.result == VK_ERROR_DEVICE_LOST);
            std::lock_guard<std::mutex> lock(mutex_);
            UpdateWithRetained(request, !releaseSafe && page.memory != VK_NULL_HANDLE ? page.allocationBytes : 0);
            ARM_CAPTURE_LOG_WARNING(
                "POOL", "page allocation failed VkResult=%d; partial resources %s", static_cast<int>(page.result), releaseSafe ? "rolled back" : "abandoned");
        }
        return success;
    }

#endif

}  // namespace arm::capture
