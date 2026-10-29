/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "worker.h"

#include "log.h"
#include "utils.h"
#include "readback.h"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <limits>
#include <memory>

#ifdef FFX_BACKEND_VK
#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <FidelityFX/host/backends/vk/vk_wrapper.h>
#endif

namespace arm::capture
{
    namespace detail
    {
#ifdef FFX_BACKEND_VK
        static std::shared_ptr<WorkerPlatform> CreateDefaultWorkerPlatform();

        Worker::Worker()
            : Worker(CreateDefaultWorkerPlatform())
        {
        }

        Worker::Worker(std::shared_ptr<WorkerPlatform> platform, uint32_t queryPoolChunkSize)
            : stagingPoolConfig_(LoadDefaultStagingPoolConfig())
            , queryPoolChunkSize_(queryPoolChunkSize)
            , platform_(std::move(platform))
        {
            worker_ = std::thread([this]() { Run(); });
        }

        Worker::~Worker()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                aborting_ = true;
            }
            cv_.notify_all();
            if (worker_.joinable())
                worker_.join();

            std::vector<std::shared_ptr<CaptureSession>> sessionsToDestroy;
            std::vector<std::shared_ptr<StagingPool>>    poolsToRelease;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                for (auto& entry : sessions_)
                    sessionsToDestroy.push_back(std::move(entry.second));
                sessions_.clear();
                poolsToRelease = std::move(stagingPools_);
                stagingPools_.clear();
            }
            for (const std::shared_ptr<StagingPool>& pool : poolsToRelease)
                pool->MarkReleaseUnsafe();
            for (const std::shared_ptr<CaptureSession>& session : sessionsToDestroy)
                session->Abandon();
            sessionsToDestroy.clear();
            for (const std::shared_ptr<StagingPool>& pool : poolsToRelease)
                pool->ReleasePages();
        }

        Worker& GetWorker()
        {
            static Worker worker;
            return worker;
        }

        namespace
        {
            VkQueryPool CreateVulkanTimestampQueryPool(VkDevice device, uint32_t queryCount)
            {
                VkQueryPoolCreateInfo createInfo{};
                createInfo.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                createInfo.queryType  = VK_QUERY_TYPE_TIMESTAMP;
                createInfo.queryCount = queryCount;

                VkQueryPool queryPool = VK_NULL_HANDLE;
                auto        create    = reinterpret_cast<PFN_vkCreateQueryPool>(vk::GetDeviceProc(device, "vkCreateQueryPool"));
                if (create == nullptr)
                {
                    ARM_CAPTURE_LOG_ERROR("CAPTURE", "failed to create timestamp query pool: vkCreateQueryPool is unavailable");
                    return VK_NULL_HANDLE;
                }
                const VkResult result = create(device, &createInfo, nullptr, &queryPool);
                if (result != VK_SUCCESS)
                {
                    ARM_CAPTURE_LOG_ERROR(
                        "CAPTURE", "failed to create timestamp query pool with %u queries; VkResult=%d", queryCount, static_cast<int>(result));
                    return VK_NULL_HANDLE;
                }
                return queryPool;
            }

            void DestroyVulkanQueryPool(VkDevice device, VkQueryPool queryPool)
            {
                auto destroy = reinterpret_cast<PFN_vkDestroyQueryPool>(vk::GetDeviceProc(device, "vkDestroyQueryPool"));
                if (destroy != nullptr && queryPool != VK_NULL_HANDLE)
                    destroy(device, queryPool, nullptr);
            }

            class VulkanWorkerPlatform final : public WorkerPlatform
            {
            public:
                bool SupportsCapture(VkDevice device, VkPhysicalDevice) const override
                {
                    // The SDK exposes only the device and an opaque command buffer, not its
                    // submission queue family index. Queue-family timestamp capability is
                    // therefore a caller-side contract; this check only validates the device
                    // entry points needed by the completion-query implementation.
                    return vk::GetDeviceProc(device, "vkCmdResetQueryPool") != nullptr && vk::GetDeviceProc(device, "vkCmdWriteTimestamp") != nullptr &&
                           vk::GetDeviceProc(device, "vkGetQueryPoolResults") != nullptr;
                }

                VkQueryPool CreateTimestampQueryPool(VkDevice device, uint32_t queryCount) override
                {
                    return CreateVulkanTimestampQueryPool(device, queryCount);
                }

                void DestroyQueryPool(VkDevice device, VkQueryPool queryPool) override
                {
                    DestroyVulkanQueryPool(device, queryPool);
                }

                VkResult GetQueryPoolResults(VkDevice           device,
                                             VkQueryPool        queryPool,
                                             uint32_t           firstQuery,
                                             uint32_t           queryCount,
                                             size_t             dataSize,
                                             void*              data,
                                             VkDeviceSize       stride,
                                             VkQueryResultFlags flags) const override
                {
                    auto getResults = reinterpret_cast<PFN_vkGetQueryPoolResults>(vk::GetDeviceProc(device, "vkGetQueryPoolResults"));
                    return getResults == nullptr ? VK_ERROR_INITIALIZATION_FAILED
                                                 : getResults(device, queryPool, firstQuery, queryCount, dataSize, data, stride, flags);
                }

                std::shared_ptr<StagingPool> CreateStagingPool(VkDevice device, VkPhysicalDevice physicalDevice, const StagingPoolConfig& config) override
                {
                    return StagingPool::Create(device, physicalDevice, config);
                }
            };
        }  // namespace

        static std::shared_ptr<WorkerPlatform> CreateDefaultWorkerPlatform()
        {
            return std::make_shared<VulkanWorkerPlatform>();
        }

        bool QuerySet::Initialize(std::shared_ptr<WorkerPlatform> platform, VkDevice device, uint32_t queryCount)
        {
            platform_   = std::move(platform);
            device_     = device;
            queryCount_ = queryCount;
            if (!platform_ || device_ == VK_NULL_HANDLE || queryCount_ == 0)
                return false;

            const VkQueryPool initialQueryPool = platform_->CreateTimestampQueryPool(device_, queryCount_);
            if (initialQueryPool == VK_NULL_HANDLE)
            {
                Abandon();
                return false;
            }
            queryPools_.push_back(QueryPool{initialQueryPool, 0, 0});
            return true;
        }

        QuerySet::~QuerySet()
        {
            ReleaseAll();
        }

        std::optional<CompletionQuery> QuerySet::Acquire()
        {
            if (!platform_ || queryCount_ == 0)
                return std::nullopt;
            if (queryPools_.empty() || queryPools_.back().nextIndex >= queryCount_)
            {
                const VkQueryPool nextQueryPool = platform_->CreateTimestampQueryPool(device_, queryCount_);
                if (nextQueryPool == VK_NULL_HANDLE)
                    return std::nullopt;
                queryPools_.push_back(QueryPool{nextQueryPool, 0, 0});
            }
            QueryPool& pool = queryPools_.back();
            ++pool.inUseCount;
            return CompletionQuery{pool.pool, pool.nextIndex++};
        }

        void QuerySet::ReleaseQuery(const CompletionQuery& query)
        {
            if (!query.IsValid())
                return;
            for (auto pool = queryPools_.begin(); pool != queryPools_.end(); ++pool)
            {
                if (pool->pool != query.pool)
                    continue;
                if (pool->inUseCount != 0)
                    --pool->inUseCount;
                const bool isCurrentPool = std::next(pool) == queryPools_.end();
                // Retire an old pool as soon as all of its queries are released.
                // Keep a partially used current pool so its remaining query indices can be reused;
                // once the current pool is exhausted, it can also be destroyed when no query is in use.
                if (pool->inUseCount == 0 && (!isCurrentPool || pool->nextIndex >= queryCount_))
                {
                    platform_->DestroyQueryPool(device_, pool->pool);
                    queryPools_.erase(pool);
                }
                return;
            }
        }

        void QuerySet::ReleaseAll(const std::atomic_bool* shutdownDetached)
        {
            if (platform_ && device_ != VK_NULL_HANDLE)
                for (const QueryPool& queryPool : queryPools_)
                {
                    if (shutdownDetached && shutdownDetached->load(std::memory_order_acquire))
                        break;
                    if (queryPool.pool != VK_NULL_HANDLE)
                        platform_->DestroyQueryPool(device_, queryPool.pool);
                }
            Abandon();
        }

        void QuerySet::Abandon()
        {
            platform_.reset();
            device_     = VK_NULL_HANDLE;
            queryCount_ = 0;
            queryPools_.clear();
        }

        CaptureSession::~CaptureSession()
        {
            if (stagingPool)
            {
                for (CaptureSlot& slot : slots)
                    if (slot.state != SlotState::Quarantined)
                        stagingPool->Return(slot.stagingBlock);
            }
            slots.clear();
            stagingPool.reset();
        }

        void CaptureSession::Abandon()
        {
            if (stagingPool)
                stagingPool->Abandon();
            querySet.Abandon();
        }

        bool CaptureSession::IsShutdownDetached() const
        {
            return shutdownDetached && shutdownDetached->load(std::memory_order_acquire);
        }

        SessionHandle Worker::Create(const SessionDescriptor& desc)
        {
            if (!worker_.joinable() || desc.estimatedSlotRequirements.attachmentBytes.empty() || !desc.metadataSink || !platform_)
                return InvalidSession;
            vk::DeviceInfo deviceInfo{};
            if (desc.device != VK_NULL_HANDLE && desc.physicalDevice != VK_NULL_HANDLE)
            {
                deviceInfo.device         = desc.device;
                deviceInfo.physicalDevice = desc.physicalDevice;
            }
            else if (!desc.backendDevice || !vk::GetDeviceInfo(desc.backendDevice, deviceInfo))
                return InvalidSession;
            if (!platform_->SupportsCapture(deviceInfo.device, deviceInfo.physicalDevice))
                return InvalidSession;

            auto session              = std::make_shared<CaptureSession>();
            session->device           = deviceInfo.device;
            session->physicalDevice   = deviceInfo.physicalDevice;
            session->metadataSink     = desc.metadataSink;
            session->shutdownDetached = desc.shutdownDetached ? desc.shutdownDetached : std::make_shared<std::atomic_bool>(false);
            if (desc.onFailure)
            {
                session->onFailure = [shutdownDetached = session->shutdownDetached, onFailure = desc.onFailure]() {
                    if (!shutdownDetached->load(std::memory_order_acquire))
                        onFailure();
                };
            }
            if (!session->querySet.Initialize(platform_, deviceInfo.device, queryPoolChunkSize_))
                return InvalidSession;
            session->slots.resize(kMaxSessionSlots);

            bool createdStagingPool = false;
            {
                std::lock_guard<std::mutex>  lock(mutex_);
                const bool                   poolExisted = FindStagingPoolLocked(deviceInfo.device, deviceInfo.physicalDevice) != nullptr;
                std::shared_ptr<StagingPool> stagingPool = FindOrCreateStagingPoolLocked(deviceInfo.device, deviceInfo.physicalDevice);
                if (!stagingPool)
                {
                    session->querySet.ReleaseAll();
                    return InvalidSession;
                }
                createdStagingPool            = !poolExisted;
                const VkDeviceSize unitSize   = detail::StagingAllocationUnitSize(stagingPool->OffsetAlignment());
                const VkDeviceSize blockBytes = detail::SlotBytesForAttachments(desc.estimatedSlotRequirements.attachmentBytes,
                                                                                desc.estimatedSlotRequirements.attachmentAlignments,
                                                                                stagingPool->OffsetAlignment(),
                                                                                unitSize);
                if (blockBytes == 0 || !stagingPool->Supports(blockBytes))
                {
                    if (createdStagingPool)
                    {
                        stagingPools_.erase(std::remove(stagingPools_.begin(), stagingPools_.end(), stagingPool), stagingPools_.end());
                    }
                    session->querySet.ReleaseAll();
                    return InvalidSession;
                }
                session->stagingPool = std::move(stagingPool);
                session->handle      = nextSessionHandle_++;
                sessions_.emplace(session->handle, session);
            }
            return session->handle;
        }

        bool Worker::AcquireSlot(SessionHandle handle, const SlotRequirements& requirements, uint32_t& slotIndex)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            auto                         session = Find(handle);
            if (!session || session->IsShutdownDetached() || session->state != SessionState::Accepting || requirements.attachmentBytes.empty())
                return false;
            const VkDeviceSize alignment = session->stagingPool->OffsetAlignment();
            const VkDeviceSize unitSize  = detail::StagingAllocationUnitSize(alignment);
            const VkDeviceSize blockBytes =
                detail::SlotBytesForAttachments(requirements.attachmentBytes, requirements.attachmentAlignments, alignment, unitSize);
            if (blockBytes == 0 || !session->stagingPool->Supports(blockBytes))
                return false;
            for (;;)
            {
                const uint32_t slotCount = static_cast<uint32_t>(session->slots.size());
                for (uint32_t offset = 0; offset < slotCount; ++offset)
                {
                    const uint32_t index = (session->nextAcquireSlot + offset) % slotCount;
                    if (session->slots[index].state != SlotState::Free)
                        continue;

                    std::vector<VkDeviceSize> attachmentAlignments = requirements.attachmentAlignments;
                    const auto                query                = session->querySet.Acquire();
                    if (!query)
                    {
                        ARM_CAPTURE_LOG_WARNING("CAPTURE",
                                                "failed to reserve completion query for session=%llu before recording slot=%u",
                                                static_cast<unsigned long long>(handle),
                                                index);
                        return false;
                    }
                    if (session->IsShutdownDetached())
                    {
                        session->slots[index].query = *query;
                        session->slots[index].state = SlotState::Quarantined;
                        return false;
                    }

                    const auto   acquireBegin = std::chrono::steady_clock::now();
                    StagingBlock stagingBlock = session->stagingPool->Acquire(blockBytes);
                    const auto   acquireMicroseconds =
                        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - acquireBegin).count();
                    if (acquireMicroseconds >= 1000)
                    {
                        ARM_CAPTURE_LOG_WARNING(
                            "CAPTURE", "synchronous render-thread slot acquire took %llu us", static_cast<unsigned long long>(acquireMicroseconds));
                    }
                    if (!stagingBlock)
                    {
                        if (session->IsShutdownDetached())
                        {
                            session->slots[index].query = *query;
                            session->slots[index].state = SlotState::Quarantined;
                            return false;
                        }
                        session->querySet.ReleaseQuery(*query);
                        ARM_CAPTURE_LOG_WARNING(
                            "CAPTURE", "session=%llu exhausted its staging-memory budget; rendering will not block", static_cast<unsigned long long>(handle));
                        return false;
                    }
                    if (session->IsShutdownDetached())
                    {
                        session->slots[index].attachmentAlignments = std::move(attachmentAlignments);
                        session->slots[index].stagingBlock         = std::move(stagingBlock);
                        session->slots[index].query                = *query;
                        session->slots[index].state                = SlotState::Quarantined;
                        return false;
                    }

                    session->slots[index].attachmentAlignments = std::move(attachmentAlignments);
                    session->slots[index].stagingBlock         = std::move(stagingBlock);
                    session->slots[index].state                = SlotState::Recording;
                    session->slots[index].query                = *query;
                    session->nextAcquireSlot                   = (index + 1) % slotCount;
                    slotIndex                                  = index;
                    return true;
                }
                ARM_CAPTURE_LOG_WARNING(
                    "CAPTURE", "session=%llu reached its outstanding-slot cap; stopping capture without waiting", static_cast<unsigned long long>(handle));
                return false;
            }
        }

        bool Worker::AcquireSlot(SessionHandle handle, uint64_t requiredSlotBytes, uint32_t& slotIndex)
        {
            return AcquireSlot(handle, SlotRequirements{{requiredSlotBytes}, {4}}, slotIndex);
        }

        bool Worker::Submit(SessionHandle handle, uint32_t slotIndex, VkCommandBuffer commandBuffer, std::unique_ptr<FrameMeta>* frameMeta)
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto                        session = Find(handle);
                if (!session || session->IsShutdownDetached() || commandBuffer == VK_NULL_HANDLE || slotIndex >= session->slots.size() ||
                    session->slots[slotIndex].state != SlotState::Recording || session->slots[slotIndex].gpuReadbacks.empty())
                {
                    ARM_CAPTURE_LOG_ERROR("CAPTURE",
                                          "failed to submit transaction: session=%s commandBuffer=%p slot=%u",
                                          session ? "valid" : "invalid",
                                          commandBuffer,
                                          slotIndex);
                    return false;
                }
                CaptureSlot& slot = session->slots[slotIndex];
                if (!slot.query.IsValid())
                {
                    ARM_CAPTURE_LOG_ERROR("CAPTURE", "missing reserved completion query for slot %u", slotIndex);
                    return false;
                }
                if (!RecordCompletionQuery(*session, commandBuffer, slot.query))
                {
                    ARM_CAPTURE_LOG_ERROR("CAPTURE", "failed to record completion query for slot %u", slotIndex);
                    return false;
                }
                if (session->IsShutdownDetached())
                {
                    slot.state = SlotState::Quarantined;
                    return false;
                }
                if (frameMeta != nullptr)
                    slot.frameMeta = std::move(*frameMeta);
                slot.state = SlotState::GpuPending;
            }
            cv_.notify_one();
            return true;
        }

        bool Worker::Abort(SessionHandle handle, uint32_t slotIndex, VkCommandBuffer commandBuffer)
        {
            std::shared_ptr<CaptureSession> session;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                session = Find(handle);
                if (!session || session->IsShutdownDetached() || slotIndex >= session->slots.size())
                    return false;
                if (session->slots[slotIndex].state == SlotState::Quarantined)
                {
                    // Device-loss quarantine already discarded CPU-side
                    // payloads and forbids Vulkan cleanup. Clear the owner
                    // transaction without touching the quarantined handles.
                    return true;
                }
                if (session->slots[slotIndex].state != SlotState::Recording)
                    return false;
                if (session->slots[slotIndex].gpuReadbacks.empty())
                {
                    // No GPU copies were recorded, so no completion query is needed.
                    session->slots[slotIndex].frameMeta.reset();
                    session->querySet.ReleaseQuery(session->slots[slotIndex].query);
                    session->slots[slotIndex].query = {};
                    session->stagingPool->Return(session->slots[slotIndex].stagingBlock);
                    session->slots[slotIndex].state = SlotState::Free;
                    return true;
                }
                if (commandBuffer == VK_NULL_HANDLE)
                {
                    // Abort is only used for a command buffer that the caller
                    // will discard; no recorded copy may execute after reuse.
                    session->slots[slotIndex].gpuReadbacks.clear();
                    session->slots[slotIndex].frameMeta.reset();
                    session->querySet.ReleaseQuery(session->slots[slotIndex].query);
                    session->slots[slotIndex].query = {};
                    session->stagingPool->Return(session->slots[slotIndex].stagingBlock);
                    session->slots[slotIndex].state = SlotState::Free;
                    return true;
                }
                if (session->slots[slotIndex].query.IsValid() && RecordCompletionQuery(*session, commandBuffer, session->slots[slotIndex].query))
                {
                    if (session->IsShutdownDetached())
                    {
                        session->slots[slotIndex].state = SlotState::Quarantined;
                        return false;
                    }
                    session->slots[slotIndex].state = SlotState::GpuPendingDiscard;
                    cv_.notify_one();
                    return true;
                }
                ARM_CAPTURE_LOG_ERROR("CAPTURE", "failed to record reserved completion query for cancelled slot %u", slotIndex);
                return false;
            }
        }

        void Worker::Close(SessionHandle handle)
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto                        session = Find(handle);
                if (!session || session->IsShutdownDetached() || session->state != SessionState::Accepting)
                    return;
                uint32_t activeSlotCount = 0;
                for (const CaptureSlot& slot : session->slots)
                {
                    if (IsActiveSlot(slot.state))
                        ++activeSlotCount;
                }
                ARM_CAPTURE_LOG_INFO("CAPTURE", "close session=%llu activeSlots=%u", static_cast<unsigned long long>(handle), activeSlotCount);
                session->state = SessionState::Draining;
                for (CaptureSlot& slot : session->slots)
                {
                    if (slot.state == SlotState::Recording)
                    {
                        // A slot with GPU readbacks has already emitted copy commands.
                        // Its producer must Submit() or Abort() it before closing.
                        if (slot.gpuReadbacks.empty())
                        {
                            slot.frameMeta.reset();
                            session->stagingPool->Return(slot.stagingBlock);
                            slot.state = SlotState::Free;
                        }
                    }
                }
            }
            cv_.notify_one();
        }

        bool Worker::Wait(SessionHandle handle, uint32_t timeoutMs)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            auto                         predicate = [&]() {
                auto session = Find(handle);
                return !session || session->state == SessionState::Finalized;
            };
            if (timeoutMs == UINT32_MAX)
            {
                cv_.wait(lock, predicate);
                return true;
            }
            return cv_.wait_for(lock, std::chrono::milliseconds(timeoutMs), predicate);
        }

        void Worker::Destroy(SessionHandle handle)
        {
            std::shared_ptr<StagingPool> poolToRelease;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto                        it = sessions_.find(handle);
                if (it != sessions_.end() && it->second->state == SessionState::Finalized)
                {
                    const VkDevice device = it->second->device;
                    sessions_.erase(it);
                    const bool deviceStillInUse =
                        std::any_of(sessions_.begin(), sessions_.end(), [device](const auto& entry) { return entry.second->device == device; });
                    if (!deviceStillInUse)
                    {
                        const auto poolIt =
                            std::find_if(stagingPools_.begin(), stagingPools_.end(), [device](const auto& pool) { return pool->UsesDevice(device); });
                        if (poolIt != stagingPools_.end())
                        {
                            poolToRelease = *poolIt;
                            stagingPools_.erase(poolIt);
                        }
                    }
                }
            }
            if (poolToRelease)
                poolToRelease->ReleasePages();
        }

        std::shared_ptr<StagingPool> Worker::FindStagingPoolLocked(VkDevice device, VkPhysicalDevice physicalDevice) const
        {
            for (const std::shared_ptr<StagingPool>& pool : stagingPools_)
            {
                if (pool->Matches(device, physicalDevice))
                    return pool;
            }
            return nullptr;
        }

        std::shared_ptr<StagingPool> Worker::FindOrCreateStagingPoolLocked(VkDevice device, VkPhysicalDevice physicalDevice)
        {
            const std::shared_ptr<StagingPool> pool = FindStagingPoolLocked(device, physicalDevice);
            if (pool)
                return pool;

            std::shared_ptr<StagingPool> newPool = platform_->CreateStagingPool(device, physicalDevice, stagingPoolConfig_);
            if (newPool)
                stagingPools_.push_back(newPool);
            return newPool;
        }

        std::shared_ptr<CaptureSession> Worker::Find(SessionHandle handle)
        {
            auto it = sessions_.find(handle);
            return it == sessions_.end() ? nullptr : it->second;
        }

        bool Worker::Record(SessionHandle handle, uint32_t slotIndex, FfxCommandList commandList, const FfxResource& source, std::unique_ptr<Payload> payload)
        {
            std::lock_guard<std::mutex>           lock(mutex_);
            const std::shared_ptr<CaptureSession> session = Find(handle);
            if (!session || session->IsShutdownDetached() || slotIndex >= session->slots.size() || session->slots[slotIndex].state != SlotState::Recording)
                return false;
            vk::ImageReadbackDesc image{};
            if (!payload || commandList == nullptr || !vk::BuildImageReadbackDesc(source, image))
            {
                ARM_CAPTURE_LOG_ERROR("CAPTURE",
                                      "failed to prepare image attachment (%s): payload=%s commandList=%p resource=%p state=%u type=%u format=%u "
                                      "mips=%u usage=0x%x",
                                      payload ? payload->GetName() : "<none>",
                                      payload ? "valid" : "null",
                                      commandList,
                                      source.resource,
                                      static_cast<unsigned>(source.state),
                                      static_cast<unsigned>(source.description.type),
                                      static_cast<unsigned>(source.description.format),
                                      source.description.mipCount,
                                      static_cast<unsigned>(source.description.usage));
                return false;
            }

            CaptureSlot& slot = session->slots[slotIndex];
            if (slot.gpuReadbacks.size() >= slot.attachmentAlignments.size())
                return false;
            const VkDeviceSize alignment =
                detail::LeastCommonMultiple(session->stagingPool->OffsetAlignment(), slot.attachmentAlignments[slot.gpuReadbacks.size()]);
            VkDeviceSize payloadOffset = 0;
            if (!TryGetNextReadbackOffset(slot, alignment, payloadOffset))
            {
                ARM_CAPTURE_LOG_ERROR("CAPTURE", "failed to calculate payload offset for %s", payload->GetName());
                return false;
            }
            if (!slot.stagingBlock || payloadOffset > slot.stagingBlock.size || image.byteSize > slot.stagingBlock.size - payloadOffset)
            {
                ARM_CAPTURE_LOG_ERROR("CAPTURE",
                                      "frame block exhausted for %s: required=%llu offset=%llu block=%llu attachments=%zu",
                                      payload->GetName(),
                                      static_cast<unsigned long long>(image.byteSize),
                                      static_cast<unsigned long long>(payloadOffset),
                                      static_cast<unsigned long long>(slot.stagingBlock.size),
                                      slot.gpuReadbacks.size());
                return false;
            }

            if (slot.gpuReadbacks.size() == slot.gpuReadbacks.max_size())
                return false;
            slot.gpuReadbacks.reserve(slot.gpuReadbacks.size() + 1);

            payload->size_   = static_cast<uint64_t>(image.byteSize);
            payload->format_ = image.format;
            payload->extent_ = {image.extent.width, image.extent.height};
            vk::RecordImageReadback(reinterpret_cast<VkCommandBuffer>(commandList), image, slot.stagingBlock.buffer, slot.stagingBlock.offset + payloadOffset);
            if (session->IsShutdownDetached())
                return false;
            slot.gpuReadbacks.push_back({std::move(payload), payloadOffset});
            return true;
        }

        bool Worker::RecordBuffer(
            SessionHandle handle, uint32_t slotIndex, FfxCommandList commandList, const FfxResource& source, std::unique_ptr<Payload> payload)
        {
            std::lock_guard<std::mutex>           lock(mutex_);
            const std::shared_ptr<CaptureSession> session = Find(handle);
            if (!session || session->IsShutdownDetached() || slotIndex >= session->slots.size() || session->slots[slotIndex].state != SlotState::Recording)
                return false;
            vk::BufferReadbackDesc buffer{};
            if (!payload || commandList == nullptr || !vk::BuildBufferReadbackDesc(source, buffer))
            {
                ARM_CAPTURE_LOG_ERROR("CAPTURE",
                                      "failed to prepare buffer attachment (%s): payload=%s commandList=%p",
                                      payload ? payload->GetName() : "<none>",
                                      payload ? "valid" : "null",
                                      commandList);
                return false;
            }

            CaptureSlot& slot = session->slots[slotIndex];
            if (slot.gpuReadbacks.size() >= slot.attachmentAlignments.size())
                return false;
            const VkDeviceSize alignment =
                detail::LeastCommonMultiple(session->stagingPool->OffsetAlignment(), slot.attachmentAlignments[slot.gpuReadbacks.size()]);
            VkDeviceSize payloadOffset = 0;
            if (!TryGetNextReadbackOffset(slot, alignment, payloadOffset))
            {
                ARM_CAPTURE_LOG_ERROR("CAPTURE", "failed to calculate payload offset for %s", payload->GetName());
                return false;
            }
            if (!slot.stagingBlock || payloadOffset > slot.stagingBlock.size || buffer.byteSize > slot.stagingBlock.size - payloadOffset)
            {
                ARM_CAPTURE_LOG_ERROR("CAPTURE",
                                      "frame block exhausted for %s: required=%llu offset=%llu block=%llu attachments=%zu",
                                      payload->GetName(),
                                      static_cast<unsigned long long>(buffer.byteSize),
                                      static_cast<unsigned long long>(payloadOffset),
                                      static_cast<unsigned long long>(slot.stagingBlock.size),
                                      slot.gpuReadbacks.size());
                return false;
            }

            if (slot.gpuReadbacks.size() == slot.gpuReadbacks.max_size())
                return false;
            slot.gpuReadbacks.reserve(slot.gpuReadbacks.size() + 1);

            payload->size_   = static_cast<uint64_t>(buffer.byteSize);
            payload->format_ = buffer.format;
            payload->extent_ = {0, 0};
            vk::RecordBufferReadback(
                reinterpret_cast<VkCommandBuffer>(commandList), buffer, slot.stagingBlock.buffer, slot.stagingBlock.offset + payloadOffset);
            if (session->IsShutdownDetached())
                return false;
            slot.gpuReadbacks.push_back({std::move(payload), payloadOffset});
            return true;
        }

        bool Worker::RecordBackendReadback(SessionHandle, uint32_t, FfxInterface&, uint32_t, FfxResourceInternal, FfxCommandList, std::unique_ptr<Payload>)
        {
            ARM_CAPTURE_LOG_WARNING("CAPTURE", "backend resource readback is not supported");
            return false;
        }

        bool Worker::TryGetNextReadbackOffset(const CaptureSlot& slot, VkDeviceSize alignment, VkDeviceSize& offset)
        {
            if (alignment == 0)
                return false;

            offset = 0;
            if (!slot.gpuReadbacks.empty())
            {
                const GpuReadback& previous = slot.gpuReadbacks.back();
                if (previous.stagingOffset > std::numeric_limits<VkDeviceSize>::max() - previous.payload->size_)
                    return false;
                offset = previous.stagingOffset + previous.payload->size_;
            }

            const VkDeviceSize aligned = detail::AlignUp(offset, alignment);
            if (aligned == 0 && offset != 0)
                return false;
            offset = aligned;
            return true;
        }

        bool Worker::RecordCompletionQuery(const CaptureSession& session, VkCommandBuffer commandBuffer, const CompletionQuery& query)
        {
            auto reset = reinterpret_cast<PFN_vkCmdResetQueryPool>(vk::GetDeviceProc(session.device, "vkCmdResetQueryPool"));
            auto write = reinterpret_cast<PFN_vkCmdWriteTimestamp>(vk::GetDeviceProc(session.device, "vkCmdWriteTimestamp"));
            if (reset == nullptr || write == nullptr || !query.IsValid())
                return false;
            reset(commandBuffer, query.pool, query.index, 1);
            write(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, query.pool, query.index);
            return true;
        }

        Worker::QueryState Worker::QuerySlot(const CaptureSession& session, const CompletionQuery& query, VkResult& result) const
        {
            uint64_t resultValues[2] = {};
            result                   = platform_->GetQueryPoolResults(session.device,
                                                    query.pool,
                                                    query.index,
                                                    1,
                                                    sizeof(resultValues),
                                                    resultValues,
                                                    sizeof(resultValues),
                                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
            if (result == VK_SUCCESS)
                return resultValues[1] != 0 ? QueryState::Ready : QueryState::NotReady;
            if (result == VK_NOT_READY)
                return QueryState::NotReady;
            return QueryState::Failed;
        }

        bool Worker::IsActiveSlot(SlotState state)
        {
            return state == SlotState::Recording || state == SlotState::GpuPending || state == SlotState::GpuPendingDiscard || state == SlotState::Consuming ||
                   state == SlotState::ConsumingDiscard;
        }

        void Worker::QuarantineSlot(CaptureSlot& slot)
        {
            slot.gpuReadbacks.clear();
            slot.frameMeta.reset();
            slot.state = SlotState::Quarantined;
        }

        void Worker::RecycleOrQuarantineSlotLocked(CaptureSession& session, CaptureSlot& slot)
        {
            slot.gpuReadbacks.clear();
            slot.frameMeta.reset();
            if (session.IsShutdownDetached())
            {
                slot.state = SlotState::Quarantined;
                return;
            }

            session.stagingPool->Return(slot.stagingBlock);
            if (session.IsShutdownDetached())
            {
                slot.state = SlotState::Quarantined;
                return;
            }

            session.querySet.ReleaseQuery(slot.query);
            slot.query = {};
            slot.state = SlotState::Free;
        }

        bool Worker::HasActiveSlots(const CaptureSession& session)
        {
            for (const CaptureSlot& slot : session.slots)
            {
                if (IsActiveSlot(slot.state))
                    return true;
            }
            return false;
        }

        std::optional<SlotRef> Worker::FindPendingSlotLocked()
        {
            if (sessions_.empty())
                return std::nullopt;

            auto start = pollingCursor_ == InvalidSession ? sessions_.begin() : sessions_.find(pollingCursor_);
            if (start == sessions_.end())
                start = sessions_.begin();

            auto entry = start;
            do
            {
                const auto& session = entry->second;
                if (session->state != SessionState::Finalized && !session->IsShutdownDetached())
                {
                    const uint32_t slotCount = static_cast<uint32_t>(session->slots.size());
                    for (uint32_t offset = 0; offset < slotCount; ++offset)
                    {
                        const uint32_t  index     = (session->nextPollSlot + offset) % slotCount;
                        const SlotState slotState = session->slots[index].state;
                        if (slotState != SlotState::GpuPending && slotState != SlotState::GpuPendingDiscard)
                            continue;

                        session->nextPollSlot = (index + 1) % slotCount;
                        auto next             = std::next(entry);
                        if (next == sessions_.end())
                            next = sessions_.begin();
                        pollingCursor_ = next->first;
                        return SlotRef{session, index};
                    }
                }

                ++entry;
                if (entry == sessions_.end())
                    entry = sessions_.begin();
            } while (entry != start);

            return std::nullopt;
        }

        bool Worker::TryClaimReadySlotLocked(const SlotRef& candidate)
        {
            const auto sessionIt = sessions_.find(candidate.session->handle);
            if (sessionIt == sessions_.end() || sessionIt->second != candidate.session || candidate.session->state == SessionState::Finalized)
                return false;
            if (candidate.slotIndex >= candidate.session->slots.size())
                return false;

            CaptureSlot& slot = candidate.session->slots[candidate.slotIndex];
            if (slot.state != SlotState::GpuPending && slot.state != SlotState::GpuPendingDiscard)
                return false;
            if (candidate.session->IsShutdownDetached())
            {
                slot.state = SlotState::Quarantined;
                return false;
            }

            slot.state = slot.state == SlotState::GpuPendingDiscard ? SlotState::ConsumingDiscard : SlotState::Consuming;
            return true;
        }

        void Worker::ProcessFailedSlots(
            std::unique_lock<std::mutex>& lock, VkDevice device, SessionHandle failedSession, bool deviceLost, const std::optional<SlotRef>& claimedSlot)
        {
            if (deviceLost)
            {
                // A device-loss result invalidates every in-flight Vulkan operation
                // on this device, including Recording slots that have not reached a
                // timestamp query yet. Quarantine all active slots and notify each
                // affected session once; no Vulkan cleanup is safe on this path.
                for (;;)
                {
                    std::function<void()> failureCallback;
                    bool                  processed = false;
                    for (const auto& entry : sessions_)
                    {
                        const std::shared_ptr<CaptureSession>& session = entry.second;
                        if (session->device != device || session->state == SessionState::Finalized)
                            continue;

                        bool hasActiveSlot = false;
                        for (CaptureSlot& slot : session->slots)
                        {
                            if (IsActiveSlot(slot.state))
                            {
                                QuarantineSlot(slot);
                                hasActiveSlot = true;
                            }
                        }
                        if (!hasActiveSlot && !session->onFailure)
                            continue;

                        if (deviceLost)
                            if (deviceLost)
                                session->stagingPool->MarkReleaseUnsafe();
                        failureCallback = std::move(session->onFailure);
                        processed       = true;
                        lock.unlock();
                        if (failureCallback)
                            failureCallback();
                        lock.lock();
                        break;
                    }
                    if (!processed)
                        break;
                }
                return;
            }

            std::function<void()> claimedFailureCallback;
            if (claimedSlot)
            {
                claimedSlot->session->slots[claimedSlot->slotIndex].state = SlotState::Quarantined;
                claimedFailureCallback                                    = std::move(claimedSlot->session->onFailure);
            }

            if (claimedFailureCallback)
            {
                lock.unlock();
                claimedFailureCallback();
                lock.lock();
            }

            std::function<void()> failureCallback;
            const auto            sessionIt = sessions_.find(failedSession);
            if (sessionIt != sessions_.end())
            {
                const std::shared_ptr<CaptureSession>& session = sessionIt->second;
                if (session->device == device && session->state != SessionState::Finalized)
                {
                    session->stagingPool->MarkReleaseUnsafe();
                    for (CaptureSlot& slot : session->slots)
                    {
                        if (slot.state == SlotState::GpuPending || slot.state == SlotState::GpuPendingDiscard)
                            QuarantineSlot(slot);
                    }
                    failureCallback = std::move(session->onFailure);
                }
            }
            if (failureCallback)
            {
                lock.unlock();
                failureCallback();
                lock.lock();
            }
        }

        std::shared_ptr<CaptureSession> Worker::TryTakeDrainingSession()
        {
            for (const auto& entry : sessions_)
            {
                const auto& session = entry.second;
                if (session->IsShutdownDetached())
                {
                    for (CaptureSlot& slot : session->slots)
                    {
                        if (slot.state == SlotState::GpuPending || slot.state == SlotState::GpuPendingDiscard)
                            slot.state = SlotState::Quarantined;
                    }
                }
                if (session->state == SessionState::Draining && !HasActiveSlots(*session))
                    return session;
            }
            return nullptr;
        }

        bool Worker::HasGpuPendingSlots() const
        {
            for (const auto& entry : sessions_)
            {
                for (const CaptureSlot& slot : entry.second->slots)
                {
                    if (slot.state == SlotState::GpuPending || slot.state == SlotState::GpuPendingDiscard)
                        return true;
                }
            }
            return false;
        }

        Worker::ConsumeResult Worker::Consume(CaptureSession& session, CaptureSlot& slot)
        {
            ARM_CAPTURE_LOG_INFO("CAPTURE",
                                 "consume begin session=%llu gpuReadbacks=%zu discard=%s",
                                 static_cast<unsigned long long>(session.handle),
                                 slot.gpuReadbacks.size(),
                                 slot.state == SlotState::ConsumingDiscard ? "true" : "false");
            VkResult result = VK_SUCCESS;
            if (slot.state != SlotState::ConsumingDiscard && !session.IsShutdownDetached())
            {
                for (GpuReadback& readback : slot.gpuReadbacks)
                {
                    if (session.IsShutdownDetached())
                        break;
                    bool payloadWritten = false;
                    result              = session.stagingPool->Readback(slot.stagingBlock, readback.stagingOffset, readback.payload->size_, [&](void* mapped) {
                        if (session.IsShutdownDetached())
                            return;
                        PayloadView view{mapped, readback.payload->size_, readback.payload->format_, readback.payload->extent_};
                        payloadWritten = readback.payload->Write(view);
                    });
                    if (session.IsShutdownDetached())
                        break;
                    if (result != VK_SUCCESS)
                        break;
                    if (!payloadWritten)
                    {
                        slot.gpuReadbacks.clear();
                        if (slot.frameMeta)
                            slot.frameMeta.reset();
                        ARM_CAPTURE_LOG_ERROR("CAPTURE", "payload write failed for session=%llu", static_cast<unsigned long long>(session.handle));
                        return {ConsumeStatus::IoFailure, VK_SUCCESS};
                    }
                }
            }
            if (result != VK_SUCCESS)
            {
                slot.gpuReadbacks.clear();
                slot.frameMeta.reset();
                return {ConsumeStatus::VulkanFailure, result};
            }
            slot.gpuReadbacks.clear();
            if (result == VK_SUCCESS && !session.IsShutdownDetached() && slot.state != SlotState::ConsumingDiscard && slot.frameMeta && session.metadataSink)
            {
                if (!session.metadataSink->CompleteFrame(std::move(slot.frameMeta)))
                {
                    ARM_CAPTURE_LOG_ERROR("CAPTURE", "metadata write failed for session=%llu", static_cast<unsigned long long>(session.handle));
                    return {ConsumeStatus::IoFailure, VK_SUCCESS};
                }
            }
            else
                slot.frameMeta.reset();
            ARM_CAPTURE_LOG_INFO("CAPTURE", "consume end session=%llu", static_cast<unsigned long long>(session.handle));
            return {ConsumeStatus::Success, VK_SUCCESS};
        }

        void Worker::Run()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!aborting_)
            {
                SlotRef workSlot;

                // 1. Snapshot a pending slot under the mutex.
                const std::optional<SlotRef> candidate = FindPendingSlotLocked();
                if (candidate)
                {
                    // 2. Query the external Vulkan driver without the worker mutex.
                    lock.unlock();
                    VkResult           queryResult = VK_SUCCESS;
                    const CaptureSlot& pendingSlot = candidate->session->slots[candidate->slotIndex];
                    const QueryState   queryState  = QuerySlot(*candidate->session, pendingSlot.query, queryResult);
                    lock.lock();

                    // 3. Revalidate and claim the slot under the mutex.
                    if (aborting_)
                        break;
                    if (queryState == QueryState::Failed)
                    {
                        ARM_CAPTURE_LOG_ERROR("CAPTURE",
                                              "query failed for session=%llu slot=%u VkResult=%d; discarding pending capture slots",
                                              static_cast<unsigned long long>(candidate->session->handle),
                                              candidate->slotIndex,
                                              static_cast<int>(queryResult));
                        const bool deviceLost = queryResult == VK_ERROR_DEVICE_LOST;
                        ProcessFailedSlots(lock, candidate->session->device, candidate->session->handle, deviceLost);
                        cv_.notify_all();
                        continue;
                    }
                    if (queryState == QueryState::Ready && TryClaimReadySlotLocked(*candidate))
                    {
                        workSlot = *candidate;
                        lock.unlock();
                        ConsumeResult consumeResult{};
                        consumeResult = Consume(*workSlot.session, workSlot.session->slots[workSlot.slotIndex]);
                        lock.lock();
                        if (workSlot.session->IsShutdownDetached())
                        {
                            QuarantineSlot(workSlot.session->slots[workSlot.slotIndex]);
                            cv_.notify_all();
                            continue;
                        }
                        const auto releaseFailedSlot = [&]() {
                            const std::function<void()> failureCallback = workSlot.session->onFailure;
                            RecycleOrQuarantineSlotLocked(*workSlot.session, workSlot.session->slots[workSlot.slotIndex]);
                            lock.unlock();
                            if (failureCallback)
                                failureCallback();
                            lock.lock();
                        };
                        if (consumeResult.status == ConsumeStatus::Success)
                        {
                            RecycleOrQuarantineSlotLocked(*workSlot.session, workSlot.session->slots[workSlot.slotIndex]);
                        }
                        else if (consumeResult.status == ConsumeStatus::VulkanFailure && consumeResult.vkResult == VK_ERROR_DEVICE_LOST)
                        {
                            ARM_CAPTURE_LOG_ERROR("CAPTURE",
                                                  "map failed for session=%llu slot=%u VkResult=%d; discarding pending capture slots",
                                                  static_cast<unsigned long long>(workSlot.session->handle),
                                                  workSlot.slotIndex,
                                                  static_cast<int>(consumeResult.vkResult));
                            ProcessFailedSlots(lock, workSlot.session->device, workSlot.session->handle, true, std::optional<SlotRef>(workSlot));
                        }
                        else if (consumeResult.status == ConsumeStatus::VulkanFailure)
                        {
                            ARM_CAPTURE_LOG_ERROR("CAPTURE",
                                                  "readback consumption failed for session=%llu slot=%u VkResult=%d",
                                                  static_cast<unsigned long long>(workSlot.session->handle),
                                                  workSlot.slotIndex,
                                                  static_cast<int>(consumeResult.vkResult));
                            releaseFailedSlot();
                        }
                        else
                        {
                            ARM_CAPTURE_LOG_ERROR("CAPTURE",
                                                  "capture output I/O failed for session=%llu slot=%u",
                                                  static_cast<unsigned long long>(workSlot.session->handle),
                                                  workSlot.slotIndex);
                            releaseFailedSlot();
                        }
                        cv_.notify_all();
                        continue;
                    }
                }

                // 4. A closed session has drained every slot, so finalize its metadata.
                const std::shared_ptr<CaptureSession> drainingSession = TryTakeDrainingSession();
                if (drainingSession)
                {
                    lock.unlock();
                    if (drainingSession->IsShutdownDetached())
                        drainingSession->Abandon();
                    else
                    {
                        ARM_CAPTURE_LOG_INFO("CAPTURE", "metadata finalize begin session=%llu", static_cast<unsigned long long>(drainingSession->handle));
                        const bool metadataFinalized = drainingSession->metadataSink->FinalizeMetadata();
                        ARM_CAPTURE_LOG_INFO("CAPTURE", "metadata finalize end session=%llu", static_cast<unsigned long long>(drainingSession->handle));
                        if (!metadataFinalized && drainingSession->onFailure)
                            drainingSession->onFailure();
                        if (drainingSession->IsShutdownDetached())
                            drainingSession->Abandon();
                        else if (drainingSession->stagingPool->IsReleaseSafe())
                            drainingSession->querySet.ReleaseAll(drainingSession->shutdownDetached.get());
                        else
                            // Do not destroy query pools after an unsafe staging/device
                            // failure; the device may already be lost.
                            drainingSession->Abandon();
                    }
                    lock.lock();
                    drainingSession->state = SessionState::Finalized;
                    cv_.notify_all();
                    continue;
                }

                // 5. Idle: wait for submission/close, or poll pending GPU work.
                if (HasGpuPendingSlots())
                    cv_.wait_for(lock, std::chrono::milliseconds(2));
                else
                    cv_.wait(lock);
            }
        }
#endif

    }  // namespace detail

}  // namespace arm::capture
