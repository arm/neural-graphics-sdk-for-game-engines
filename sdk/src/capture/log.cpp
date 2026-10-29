/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "log.h"

#include "ffx_logging.h"

#include <cstdarg>
#include <cstdio>

namespace arm::capture
{
    namespace
    {
        FfxMsgType ToMessageType(LogLevel level)
        {
            switch (level)
            {
            case LogLevel::Warning:
                return FFX_MESSAGE_TYPE_WARNING;
            case LogLevel::Error:
                return FFX_MESSAGE_TYPE_ERROR;
            default:
                return FFX_MESSAGE_TYPE_INFO;
            }
        }
    }  // namespace

    void Log::Write(LogLevel level, const char* component, const char* format, ...)
    {
        char    message[2048];
        va_list args;
        va_start(args, format);
        const int messageLength = std::vsnprintf(message, sizeof(message), format, args);
        va_end(args);
        if (messageLength < 0)
            return;

        ffxLogMessage(component && component[0] ? component : "CAPTURE", ToMessageType(level), message);
    }

}  // namespace arm::capture
