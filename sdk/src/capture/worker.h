/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "config.h"
#include "payload.h"
#include "session.h"
#include "staging_pool.h"

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#include <FidelityFX/host/ffx_interface.h>
#include <FidelityFX/host/ffx_types.h>

namespace arm::capture::detail
{
#ifdef FFX_BACKEND_VK
    constexpr uint32_t kMaxSessionSlots = 500;

    enum class SlotState : uint8_t
    {
        Free,
        Recording,
        GpuPending,
        GpuPendingDiscard,
        Consuming,
        ConsumingDiscard,
        Quarantined,
    };

    enum class SessionState : uint8_t
    {
        Accepting,
        Draining,
        Finalized,
    };

    struct GpuReadback
    {
        std::unique_ptr<Payload> payload;
        VkDeviceSize             stagingOffset = 0;
    };

    // Keeps Worker lifecycle tests independent of a Vulkan driver while the
    // production implementation continues to use the Vulkan dispatch wrapper.
    class WorkerPlatform
    {
    public:
        virtual ~WorkerPlatform() = default;

        virtual bool                         SupportsCapture(VkDevice device, VkPhysicalDevice physicalDevice) const                              = 0;
        virtual VkQueryPool                  CreateTimestampQueryPool(VkDevice device, uint32_t queryCount)                                       = 0;
        virtual void                         DestroyQueryPool(VkDevice device, VkQueryPool queryPool)                                             = 0;
        virtual VkResult                     GetQueryPoolResults(VkDevice           device,
                                                                 VkQueryPool        queryPool,
                                                                 uint32_t           firstQuery,
                                                                 uint32_t           queryCount,
                                                                 size_t             dataSize,
                                                                 void*              data,
                                                                 VkDeviceSize       stride,
                                                                 VkQueryResultFlags flags) const                                                  = 0;
        virtual std::shared_ptr<StagingPool> CreateStagingPool(VkDevice device, VkPhysicalDevice physicalDevice, const StagingPoolConfig& config) = 0;
    };

    struct CompletionQuery
    {
        VkQueryPool pool  = VK_NULL_HANDLE;
        uint32_t    index = UINT32_MAX;

        bool IsValid() const
        {
            return pool != VK_NULL_HANDLE && index != UINT32_MAX;
        }
    };

    class QuerySet
    {
    public:
        ~QuerySet();

        bool Initialize(std::shared_ptr<WorkerPlatform> platform, VkDevice device, uint32_t queryCount);

        // Requires Worker::mutex_.
        std::optional<CompletionQuery> Acquire();
        // Releases one reserved query. Requires Worker::mutex_.
        void ReleaseQuery(const CompletionQuery& query);
        // Releases all query pools. The device must still be valid. If shutdown
        // detaches while a destroy call is blocked, no later pool is destroyed.
        void ReleaseAll(const std::atomic_bool* shutdownDetached = nullptr);
        // Detaches all Vulkan state without calling into Vulkan. Use when the
        // device or its dispatch lifetime is no longer safe.
        void Abandon();

    private:
        std::shared_ptr<WorkerPlatform> platform_;
        VkDevice                        device_     = VK_NULL_HANDLE;
        uint32_t                        queryCount_ = 0;
        struct QueryPool
        {
            VkQueryPool pool       = VK_NULL_HANDLE;
            uint32_t    nextIndex  = 0;
            uint32_t    inUseCount = 0;
        };
        std::vector<QueryPool> queryPools_;
    };

    struct CaptureSlot
    {
        SlotState state = SlotState::Free;
        // Reserved when the slot is acquired; recorded after capture copies on submit or abort.
        CompletionQuery            query;
        std::vector<GpuReadback>   gpuReadbacks;
        std::vector<uint64_t>      attachmentAlignments;
        StagingBlock               stagingBlock;
        std::unique_ptr<FrameMeta> frameMeta;
    };

    struct CaptureSession
    {
        ~CaptureSession();
        // Detaches Vulkan-owned resources so destruction only releases C++
        // state after device loss or backend teardown.
        void Abandon();
        bool IsShutdownDetached() const;

        SessionHandle                      handle         = InvalidSession;
        VkDevice                           device         = VK_NULL_HANDLE;
        VkPhysicalDevice                   physicalDevice = VK_NULL_HANDLE;
        QuerySet                           querySet;
        std::vector<CaptureSlot>           slots;
        std::shared_ptr<StagingPool>       stagingPool;
        uint32_t                           nextAcquireSlot = 0;
        uint32_t                           nextPollSlot    = 0;
        std::shared_ptr<FrameMetadataSink> metadataSink;
        std::function<void()>              onFailure;
        std::shared_ptr<std::atomic_bool>  shutdownDetached;
        SessionState                       state = SessionState::Accepting;
    };

    struct SlotRef
    {
        std::shared_ptr<CaptureSession> session;
        uint32_t                        slotIndex = UINT32_MAX;
    };

    // Capture-runtime interface shared by the session, transaction, and
    // attachment-context implementation files.
    class Worker
    {
    public:
        Worker();
        explicit Worker(std::shared_ptr<WorkerPlatform> platform, uint32_t queryPoolChunkSize = kMaxSessionSlots);
        ~Worker();

        SessionHandle Create(const SessionDescriptor& desc);
        bool          AcquireSlot(SessionHandle handle, const SlotRequirements& requirements, uint32_t& slotIndex);
        bool          AcquireSlot(SessionHandle handle, uint64_t requiredSlotBytes, uint32_t& slotIndex);
        bool          Record(SessionHandle handle, uint32_t slotIndex, FfxCommandList commandList, const FfxResource& source, std::unique_ptr<Payload> payload);
        bool RecordBuffer(SessionHandle handle, uint32_t slotIndex, FfxCommandList commandList, const FfxResource& source, std::unique_ptr<Payload> payload);
#ifdef FFX_BACKEND_VK
        bool RecordBackendReadback(SessionHandle            handle,
                                   uint32_t                 slotIndex,
                                   FfxInterface&            backend,
                                   uint32_t                 effectContextId,
                                   FfxResourceInternal      source,
                                   FfxCommandList           commandList,
                                   std::unique_ptr<Payload> payload);
        bool Submit(SessionHandle handle, uint32_t slotIndex, VkCommandBuffer commandBuffer, std::unique_ptr<FrameMeta>* frameMeta);
        bool Abort(SessionHandle handle, uint32_t slotIndex, VkCommandBuffer commandBuffer);
#endif
        void Close(SessionHandle handle);
        bool Wait(SessionHandle handle, uint32_t timeoutMs);
        void Destroy(SessionHandle handle);

    private:
        enum class QueryState : uint8_t
        {
            Ready,
            NotReady,
            Failed,
        };

        enum class ConsumeStatus : uint8_t
        {
            Success,
            VulkanFailure,
            IoFailure,
        };

        struct ConsumeResult
        {
            ConsumeStatus status   = ConsumeStatus::Success;
            VkResult      vkResult = VK_SUCCESS;
        };

        static bool TryGetNextReadbackOffset(const CaptureSlot& slot, VkDeviceSize alignment, VkDeviceSize& offset);
        static bool RecordCompletionQuery(const CaptureSession& session, VkCommandBuffer commandBuffer, const CompletionQuery& query);
        static bool IsActiveSlot(SlotState state);
        static void QuarantineSlot(CaptureSlot& slot);
        // Requires Worker::mutex_. Retains the slot if shutdown detached while
        // recycling; otherwise returns its staging block and completion query.
        static void                     RecycleOrQuarantineSlotLocked(CaptureSession& session, CaptureSlot& slot);
        QueryState                      QuerySlot(const CaptureSession& session, const CompletionQuery& query, VkResult& result) const;
        std::shared_ptr<StagingPool>    FindStagingPoolLocked(VkDevice device, VkPhysicalDevice physicalDevice) const;
        std::shared_ptr<StagingPool>    FindOrCreateStagingPoolLocked(VkDevice device, VkPhysicalDevice physicalDevice);
        std::shared_ptr<CaptureSession> Find(SessionHandle handle);
        static bool                     HasActiveSlots(const CaptureSession& session);
        std::optional<SlotRef>          FindPendingSlotLocked();
        bool                            TryClaimReadySlotLocked(const SlotRef& candidate);
        void                            ProcessFailedSlots(std::unique_lock<std::mutex>& lock,
                                                           VkDevice                      device,
                                                           SessionHandle                 failedSession,
                                                           bool                          deviceLost,
                                                           const std::optional<SlotRef>& claimedSlot = std::nullopt);
        std::shared_ptr<CaptureSession> TryTakeDrainingSession();
        bool                            HasGpuPendingSlots() const;
        ConsumeResult                   Consume(CaptureSession& session, CaptureSlot& slot);
        void                            Run();

        std::mutex                                                         mutex_;
        std::condition_variable                                            cv_;
        StagingPoolConfig                                                  stagingPoolConfig_;
        const uint32_t                                                     queryPoolChunkSize_ = kMaxSessionSlots;
        std::shared_ptr<WorkerPlatform>                                    platform_;
        std::vector<std::shared_ptr<StagingPool>>                          stagingPools_;
        std::unordered_map<SessionHandle, std::shared_ptr<CaptureSession>> sessions_;
        // Monotonic session identity allocator. Handles are not reused after
        // a session is destroyed, preventing stale asynchronous requests from
        // targeting a later session.
        // *Don't* replace it with the size of sessions_ as handles must remain
        // unique even if sessions are removed.
        SessionHandle nextSessionHandle_ = 1;
        // Round-robin polling cursor. This is independent from handle
        // allocation and ensures one not-ready session cannot starve others.
        SessionHandle pollingCursor_ = InvalidSession;
        bool          aborting_      = false;
        std::thread   worker_;
    };

    Worker& GetWorker();
#endif

}  // namespace arm::capture::detail
