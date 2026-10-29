/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "session.h"

#include "worker.h"

namespace arm::capture
{
#ifdef FFX_BACKEND_VK
    SessionHandle CreateSession(const SessionDescriptor& desc)
    {
        return detail::GetWorker().Create(desc);
    }

    void CloseSession(SessionHandle session)
    {
        detail::GetWorker().Close(session);
    }

    bool WaitForSession(SessionHandle session, uint32_t timeoutMs)
    {
        return detail::GetWorker().Wait(session, timeoutMs);
    }

    void DestroySession(SessionHandle session)
    {
        detail::GetWorker().Destroy(session);
    }

#endif

}  // namespace arm::capture
