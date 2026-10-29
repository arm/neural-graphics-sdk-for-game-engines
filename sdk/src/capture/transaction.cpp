/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "transaction.h"

#include "worker.h"

namespace arm::capture
{
#ifdef FFX_BACKEND_VK
    bool BeginTransaction(
        SessionHandle session, const SlotRequirements& requirements, FfxCommandList commandList, std::unique_ptr<FrameMeta> frameMeta, Transaction& transaction)
    {
        if (!frameMeta || transaction.active || commandList == nullptr)
            return false;

        uint32_t slotIndex = UINT32_MAX;
        if (!detail::GetWorker().AcquireSlot(session, requirements, slotIndex))
            return false;

        transaction.session     = session;
        transaction.slotIndex   = slotIndex;
        transaction.commandList = commandList;
        transaction.frameMeta   = std::move(frameMeta);
        transaction.active      = true;
        return true;
    }

    bool BeginTransaction(
        SessionHandle session, uint64_t requiredSlotBytes, FfxCommandList commandList, std::unique_ptr<FrameMeta> frameMeta, Transaction& transaction)
    {
        return BeginTransaction(session, SlotRequirements{{requiredSlotBytes}, {4}}, commandList, std::move(frameMeta), transaction);
    }

    bool AddAttachment(Transaction& transaction, const FfxResource& source, std::unique_ptr<Payload> payload)
    {
        if (!transaction.active || !payload || source.resource == nullptr)
            return false;
        if (source.description.type == FFX_RESOURCE_TYPE_BUFFER)
        {
            return detail::GetWorker().RecordBuffer(transaction.session, transaction.slotIndex, transaction.commandList, source, std::move(payload));
        }
        return detail::GetWorker().Record(transaction.session, transaction.slotIndex, transaction.commandList, source, std::move(payload));
    }

    bool CommitTransaction(Transaction& transaction)
    {
        if (!transaction.active)
            return false;

        std::unique_ptr<FrameMeta>* const frameMeta = transaction.frameMeta ? &transaction.frameMeta : nullptr;
        const bool                        committed =
            detail::GetWorker().Submit(transaction.session, transaction.slotIndex, reinterpret_cast<VkCommandBuffer>(transaction.commandList), frameMeta);
        if (committed)
            transaction = {};
        return committed;
    }

    bool CancelTransaction(Transaction& transaction)
    {
        if (!transaction.active)
            return true;

        if (!detail::GetWorker().Abort(transaction.session, transaction.slotIndex, reinterpret_cast<VkCommandBuffer>(transaction.commandList)))
            return false;
        transaction = {};
        return true;
    }

    bool DiscardTransaction(Transaction& transaction)
    {
        if (!transaction.active)
            return true;

        if (!detail::GetWorker().Abort(transaction.session, transaction.slotIndex, VK_NULL_HANDLE))
            return false;
        transaction = {};
        return true;
    }
#endif

}  // namespace arm::capture
