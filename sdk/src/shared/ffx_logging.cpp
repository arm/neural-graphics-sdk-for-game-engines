/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "ffx_logging.h"

#include <atomic>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <string>
#include <utility>

#ifdef __ANDROID__
#include <sys/system_properties.h>
#include <spdlog/sinks/android_sink.h>
#else
#include <spdlog/sinks/stdout_color_sinks.h>
#endif
#include <spdlog/spdlog.h>

static constexpr const char* ffxLogComponentName()
{
    return "SDK";
}

static constexpr const char* DEFAULT_NULL_MSG_WARNING   = "[SDK Warning]: the log message is nullptr!";
static constexpr uint32_t    MAX_ERROR_LOG_SIZE         = 1024;
static constexpr size_t      STACK_LOG_MESSAGE_CAPACITY = 240;

namespace
{
    std::once_flag                  g_loggerInitialization;
    std::shared_ptr<spdlog::logger> g_logger;
    std::atomic<bool>               g_sdkVersionLogged{false};
    thread_local uint32_t           g_logSuppressionDepth = 0;

    spdlog::level::level_enum getLogLevel()
    {
        const char* configuredLevel = std::getenv(SDK_LOG_LEVEL_ENV_VARIABLE);
#ifdef __ANDROID__
        char androidPropertyLevel[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.arm.logging.level", androidPropertyLevel) > 0)
        {
            configuredLevel = androidPropertyLevel;
        }
#endif
        if (configuredLevel == nullptr)
        {
            return spdlog::level::warn;
        }

        const auto level = spdlog::level::from_str(configuredLevel);
        return level == spdlog::level::off && std::strcmp(configuredLevel, "off") != 0 ? spdlog::level::warn : level;
    }

    void initializeLogger()
    {
        std::shared_ptr<spdlog::logger> logger = spdlog::get("ffx_sdk");
        if (!logger)
        {
#ifdef __ANDROID__
            logger = spdlog::android_logger_mt("ffx_sdk", "FFX");
#else
            logger                       = spdlog::stderr_color_mt("ffx_sdk");
#endif
        }
        logger->set_level(getLogLevel());
        logger->set_pattern("%v");
        g_logger = std::move(logger);
    }

#if !defined(__ANDROID__)
    const char* getLevelName(FfxMsgType type)
    {
        switch (type)
        {
        case FFX_MESSAGE_TYPE_ERROR:
            return "error";
        case FFX_MESSAGE_TYPE_INFO:
            return "info";
        case FFX_MESSAGE_TYPE_WARNING:
        default:
            return "warning";
        }
    }
#endif

    spdlog::level::level_enum getMessageLogLevel(FfxMsgType type)
    {
        switch (type)
        {
        case FFX_MESSAGE_TYPE_ERROR:
            return spdlog::level::err;
        case FFX_MESSAGE_TYPE_INFO:
            return spdlog::level::info;
        case FFX_MESSAGE_TYPE_WARNING:
        default:
            return spdlog::level::warn;
        }
    }

}  // namespace

namespace
{
    bool ensureLogger() noexcept
    {
        std::call_once(g_loggerInitialization, initializeLogger);
        return g_logger != nullptr;
    }

    void invokeCallback(FfxLogMessageCallback callback, FfxMsgType type, const char* message) noexcept
    {
        if (callback == nullptr)
        {
            return;
        }

        callback(type, message);
    }

    void logMessage(const char* component, FfxMsgType type, const char* message) noexcept
    {
        if (message == nullptr)
        {
            return;
        }

        if (!ensureLogger())
        {
            return;
        }

        const auto level = getMessageLogLevel(type);
        if (g_logger->should_log(level))
        {
            const char* const componentName = component == nullptr ? ffxLogComponentName() : component;
            char              stackMessage[STACK_LOG_MESSAGE_CAPACITY];
#if !defined(__ANDROID__)
            const int stackMessageLength = std::snprintf(stackMessage, sizeof(stackMessage), "[FFX][%s] %s: %s", componentName, getLevelName(type), message);
#else
            const int stackMessageLength = std::snprintf(stackMessage, sizeof(stackMessage), "[FFX][%s] %s", componentName, message);
#endif
            if (stackMessageLength >= 0 && static_cast<size_t>(stackMessageLength) < sizeof(stackMessage))
            {
                g_logger->log(level, spdlog::string_view_t(stackMessage, static_cast<size_t>(stackMessageLength)));
                return;
            }

            std::string formattedMessage = std::string("[FFX][") + componentName + "] ";
#if !defined(__ANDROID__)
            formattedMessage += getLevelName(type);
            formattedMessage += ": ";
#endif
            formattedMessage += message;
            g_logger->log(level, spdlog::string_view_t(formattedMessage.data(), formattedMessage.size()));
        }
    }
}  // namespace

FfxScopedLogSuppression::FfxScopedLogSuppression(bool enabled) noexcept
    : enabled_(enabled)
{
    if (enabled_)
    {
        ++g_logSuppressionDepth;
    }
}

FfxScopedLogSuppression::~FfxScopedLogSuppression() noexcept
{
    if (enabled_)
    {
        --g_logSuppressionDepth;
    }
}

bool ffxLogIsSuppressed() noexcept
{
    return g_logSuppressionDepth != 0;
}

void ffxLogMessage(const char* component, FfxMsgType type, const char* message, FfxLogMessageCallback callback) noexcept
{
    if (ffxLogIsSuppressed())
    {
        return;
    }

    const char* normalizedMessage = message == nullptr ? DEFAULT_NULL_MSG_WARNING : message;
    while (*normalizedMessage == '\r' || *normalizedMessage == '\n')
    {
        ++normalizedMessage;
    }

    const size_t messageLength = std::strlen(normalizedMessage);
    size_t       trimmedLength = messageLength;
    while (trimmedLength > 0 && (normalizedMessage[trimmedLength - 1] == '\r' || normalizedMessage[trimmedLength - 1] == '\n'))
    {
        --trimmedLength;
    }

    if (trimmedLength == messageLength)
    {
        logMessage(component, type, normalizedMessage);
        invokeCallback(callback, type, normalizedMessage);
        return;
    }

    const std::string trimmedMessage(normalizedMessage, trimmedLength);
    logMessage(component, type, trimmedMessage.c_str());
    invokeCallback(callback, type, trimmedMessage.c_str());
}

void ffxLogMessageOnceToSink(
    const char* component, FfxMsgType type, const char* message, std::atomic<bool>& sinkLogged, FfxLogMessageCallback callback) noexcept
{
    if (ffxLogIsSuppressed())
    {
        return;
    }

    if (!sinkLogged.exchange(true, std::memory_order_relaxed))
    {
        ffxLogMessage(component, type, message, callback);
        return;
    }
    invokeCallback(callback, type, message);
}

void ffxLogSdkVersion(FfxLogMessageCallback callback) noexcept
{
    constexpr size_t messageCapacity = 128;
    char             message[messageCapacity];
    std::snprintf(message, sizeof(message), "event=versions sdk_version=%d.%d.%d", FFX_SDK_VERSION_MAJOR, FFX_SDK_VERSION_MINOR, FFX_SDK_VERSION_PATCH);
    ffxLogMessageOnceToSink(ffxLogComponentName(), FFX_MESSAGE_TYPE_INFO, message, g_sdkVersionLogged, callback);
}

const char* ffxGetSurfaceFormatName(FfxSurfaceFormat format) noexcept
{
    switch (format)
    {
#define FFX_FORMAT_NAME(name)       \
    case FFX_SURFACE_FORMAT_##name: \
        return #name
        FFX_FORMAT_NAME(UNKNOWN);
        FFX_FORMAT_NAME(R32G32B32A32_TYPELESS);
        FFX_FORMAT_NAME(R32G32B32A32_UINT);
        FFX_FORMAT_NAME(R32G32B32A32_FLOAT);
        FFX_FORMAT_NAME(R16G16B16A16_FLOAT);
        FFX_FORMAT_NAME(R32G32B32_FLOAT);
        FFX_FORMAT_NAME(R32G32_FLOAT);
        FFX_FORMAT_NAME(R8_UINT);
        FFX_FORMAT_NAME(R8_SINT);
        FFX_FORMAT_NAME(R32_UINT);
        FFX_FORMAT_NAME(R8G8B8A8_TYPELESS);
        FFX_FORMAT_NAME(R8G8B8A8_UNORM);
        FFX_FORMAT_NAME(R8G8B8A8_SNORM);
        FFX_FORMAT_NAME(R8G8B8A8_SRGB);
        FFX_FORMAT_NAME(B8G8R8A8_TYPELESS);
        FFX_FORMAT_NAME(B8G8R8A8_UNORM);
        FFX_FORMAT_NAME(B8G8R8A8_SRGB);
        FFX_FORMAT_NAME(R11G11B10_FLOAT);
        FFX_FORMAT_NAME(R10G10B10A2_UNORM);
        FFX_FORMAT_NAME(R16G16_FLOAT);
        FFX_FORMAT_NAME(R16G16_UINT);
        FFX_FORMAT_NAME(R16G16_SINT);
        FFX_FORMAT_NAME(R16_FLOAT);
        FFX_FORMAT_NAME(R16_UINT);
        FFX_FORMAT_NAME(R16_UNORM);
        FFX_FORMAT_NAME(R16_SNORM);
        FFX_FORMAT_NAME(R8_UNORM);
        FFX_FORMAT_NAME(R8_SNORM);
        FFX_FORMAT_NAME(R8G8_UNORM);
        FFX_FORMAT_NAME(R8G8_UINT);
        FFX_FORMAT_NAME(R8G8_SINT);
        FFX_FORMAT_NAME(R32_FLOAT);
        FFX_FORMAT_NAME(R9G9B9E5_SHAREDEXP);
        FFX_FORMAT_NAME(R16G16B16A16_TYPELESS);
        FFX_FORMAT_NAME(R32G32_TYPELESS);
        FFX_FORMAT_NAME(R10G10B10A2_TYPELESS);
        FFX_FORMAT_NAME(R16G16_TYPELESS);
        FFX_FORMAT_NAME(R16_TYPELESS);
        FFX_FORMAT_NAME(R8_TYPELESS);
        FFX_FORMAT_NAME(R8G8_TYPELESS);
        FFX_FORMAT_NAME(R32_TYPELESS);
#undef FFX_FORMAT_NAME
    default:
        return "UNKNOWN";
    }
}

void ffxLogError(const char* component, FfxErrorCode errorCode, const char* errorMessage, const char* funcName) noexcept
{
    char message[MAX_ERROR_LOG_SIZE] = {};

    const char* finalFuncName     = funcName ? funcName : "UnknownFunc";
    const char* errorName         = ffxGetErrorCodeName(errorCode);
    const char* finalErrorMessage = (errorMessage && errorMessage[0] != '\0') ? errorMessage : "";
    std::snprintf(message, sizeof(message), "error code=%d(%s) %s in %s", errorCode, errorName, finalErrorMessage, finalFuncName);

    ffxLogMessage(component, FFX_MESSAGE_TYPE_ERROR, message);
}
