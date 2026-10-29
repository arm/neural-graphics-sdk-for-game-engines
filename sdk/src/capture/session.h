/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "meta.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#ifdef FFX_BACKEND_VK
#include <vulkan/vulkan.h>
#endif

namespace arm::capture
{
    using SessionHandle                    = uint64_t;
    constexpr SessionHandle InvalidSession = 0;

    struct SlotRequirements
    {
        std::vector<uint64_t> attachmentBytes;
        std::vector<uint64_t> attachmentAlignments;
    };

    // Provider-owned frame persistence boundary. The generic runtime transfers
    // typed ownership here but does not inspect provider metadata or JSON schema.
    class FrameMetadataSink
    {
    public:
        virtual ~FrameMetadataSink() = default;
        // Completion callbacks may arrive out of dataset-index order. Each
        // FrameMeta remains paired with its slot's payloads; consumers must use
        // datasetIndex rather than callback or JSON-array position.
        virtual bool CompleteFrame(std::unique_ptr<FrameMeta> frameMeta) = 0;
        virtual bool FinalizeMetadata()                                  = 0;
    };

    struct SessionDescriptor
    {
        SessionDescriptor(void* device, uint64_t slotBytes, const std::shared_ptr<FrameMetadataSink>& sink)
            : backendDevice(device)
            , estimatedSlotRequirements{{slotBytes}, {4}}
            , metadataSink(sink)
        {
        }

        SessionDescriptor(void* device, SlotRequirements requirements, const std::shared_ptr<FrameMetadataSink>& sink)
            : backendDevice(device)
            , estimatedSlotRequirements(std::move(requirements))
            , metadataSink(sink)
        {
        }

        // Borrowed backend device context. The active capture backend resolves
        // this opaque pointer; providers never need to know native device types.
        void* backendDevice;
#ifdef FFX_BACKEND_VK
        // Legacy provider path. Capture owns the conversion to its backend state.
        VkDevice         device         = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
#endif
        // Vulkan completion uses timestamp queries recorded into the caller's command buffer.
        // The opaque FfxCommandList does not expose its executing queue family, so the caller
        // must submit it on a queue family with timestamp support; the capture runtime can only
        // validate the device-side timestamp entry points.
        SlotRequirements                   estimatedSlotRequirements;
        std::shared_ptr<FrameMetadataSink> metadataSink;
        // Invoked as soon as the worker detects a terminal query failure.
        // It must only signal the owner thread and must not touch provider state.
        std::function<void()> onFailure;
        // Shutdown-only logical detach intent shared with the capture owner.
        // Publishing it must not wait for a worker mutex or an external call.
        std::shared_ptr<std::atomic_bool> shutdownDetached;
    };

    // All sessions share one process-wide worker. A transaction is accepted only
    // when its complete staging slot can be acquired without waiting.
    SessionHandle CreateSession(const SessionDescriptor& desc);

    // Close rejects new slots, abandons Recording slots with no recorded payloads,
    // drains submitted or aborted work, invokes finalize once, and then transitions
    // to Finalized. A Recording slot with payloads must be submitted or aborted
    // before closing so staging resources remain valid for its recorded GPU copies.
    void CloseSession(SessionHandle session);
    bool WaitForSession(SessionHandle session, uint32_t timeoutMs);
    void DestroySession(SessionHandle session);
}  // namespace arm::capture
