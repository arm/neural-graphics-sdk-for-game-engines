/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <type_traits>

#include "FidelityFX/host/ffx_interface.h"
#include "FidelityFX/host/ffx_error.h"
#include "FidelityFX/host/ffx_types.h"

static constexpr const char* SDK_LOG_LEVEL_ENV_VARIABLE = "FFX_SDK_LOG_LEVEL";

// Each translation unit must define the translation-unit-local constexpr accessor
// ffxLogComponentName() after including this header.
static constexpr const char* ffxLogComponentName();

void ffxLogMessage(const char* component, FfxMsgType type, const char* message, FfxLogMessageCallback callback = nullptr) noexcept;

// Sink once; callback every call.
void ffxLogMessageOnceToSink(
    const char* component, FfxMsgType type, const char* message, std::atomic<bool>& sinkLogged, FfxLogMessageCallback callback = nullptr) noexcept;

void ffxLogSdkVersion(FfxLogMessageCallback callback = nullptr) noexcept;

const char* ffxGetSurfaceFormatName(FfxSurfaceFormat format) noexcept;

bool ffxLogIsSuppressed() noexcept;

class FfxScopedLogSuppression
{
public:
    explicit FfxScopedLogSuppression(bool enabled = true) noexcept;
    ~FfxScopedLogSuppression() noexcept;

    FfxScopedLogSuppression(const FfxScopedLogSuppression&) = delete;
    FfxScopedLogSuppression& operator=(const FfxScopedLogSuppression&) = delete;

private:
    bool enabled_;
};

#if defined(__GNUC__) || defined(__clang__) || (defined(_MSC_VER) && _MSC_VER >= 1927)
#define FFX_BUILTIN_FUNCTION __builtin_FUNCTION()
#else
#define FFX_BUILTIN_FUNCTION nullptr
#endif

void ffxLogError(const char* component, FfxErrorCode errorCode, const char* errorMessage = "", const char* funcName = FFX_BUILTIN_FUNCTION) noexcept;

// Unified compile-time component and operation preserving logging return helpers.
template <typename ErrorCodeType>
static inline ErrorCodeType ffxLogError(const char*   component,
                                        ErrorCodeType errorCode,
                                        const char*   errorMessage = "",
                                        const char*   funcName     = FFX_BUILTIN_FUNCTION) noexcept
{
    static_assert(std::is_same_v<std::decay_t<ErrorCodeType>, FfxErrorCode> || std::is_same_v<std::decay_t<ErrorCodeType>, FfxErrorCodes>,
                  "errorCode must be of type FfxErrorCode or FfxErrorCodes");
    ffxLogError(component, static_cast<FfxErrorCode>(errorCode), errorMessage, funcName);
    return errorCode;
}

template <typename ErrorCodeType>
static inline ErrorCodeType ffxLogError(ErrorCodeType errorCode, const char* errorMessage = "", const char* funcName = FFX_BUILTIN_FUNCTION) noexcept
{
    ffxLogError(ffxLogComponentName(), static_cast<FfxErrorCode>(errorCode), errorMessage, funcName);
    return errorCode;
}

#undef FFX_ERROR_LOG
#define FFX_ERROR_LOG(condition, errorCode) ffxLogError((errorCode), "validation (" condition ")", __func__)
