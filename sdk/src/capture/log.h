/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

namespace arm::capture
{
    enum class LogLevel
    {
        Info,
        Warning,
        Error,
    };

    // Compatibility facade for capture call sites using the SDK logger.
    class Log
    {
    public:
        static void Write(LogLevel level, const char* component, const char* format, ...);
    };

}  // namespace arm::capture

#define ARM_CAPTURE_LOG_INFO(component, format, ...)    ::arm::capture::Log::Write(::arm::capture::LogLevel::Info, component, format, ##__VA_ARGS__)
#define ARM_CAPTURE_LOG_WARNING(component, format, ...) ::arm::capture::Log::Write(::arm::capture::LogLevel::Warning, component, format, ##__VA_ARGS__)
#define ARM_CAPTURE_LOG_ERROR(component, format, ...)   ::arm::capture::Log::Write(::arm::capture::LogLevel::Error, component, format, ##__VA_ARGS__)
