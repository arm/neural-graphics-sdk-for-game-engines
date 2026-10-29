/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "session.h"
#include "payload.h"

#include <memory>
#include <stdint.h>

#include <FidelityFX/host/ffx_interface.h>
#include <FidelityFX/host/ffx_types.h>

namespace arm::capture
{
    // One transaction owns one logical capture frame. Attachments may be recorded
    // before and after an algorithm dispatch, but are submitted atomically through
    // one completion token on the same command list.
    struct Transaction
    {
        SessionHandle              session     = InvalidSession;
        uint32_t                   slotIndex   = UINT32_MAX;
        FfxCommandList             commandList = nullptr;
        std::unique_ptr<FrameMeta> frameMeta;
        bool                       active = false;
    };

    bool BeginTransaction(SessionHandle              session,
                          const SlotRequirements&    requirements,
                          FfxCommandList             commandList,
                          std::unique_ptr<FrameMeta> frameMeta,
                          Transaction&               transaction);
    bool BeginTransaction(
        SessionHandle session, uint64_t requiredSlotBytes, FfxCommandList commandList, std::unique_ptr<FrameMeta> frameMeta, Transaction& transaction);
    bool AddAttachment(Transaction& transaction, const FfxResource& source, std::unique_ptr<Payload> payload);
    bool CommitTransaction(Transaction& transaction);
    bool CancelTransaction(Transaction& transaction);
    bool DiscardTransaction(Transaction& transaction);

}  // namespace arm::capture
