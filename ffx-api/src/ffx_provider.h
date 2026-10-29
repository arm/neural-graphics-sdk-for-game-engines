// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once
#include <FidelityFX/host/ffx_types.h>
#include "ffx_logging.h"
#include <ffx_api/ffx_api.hpp>
#include <ffx_api/ffx_api_types.h>
#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
#include <ffx_heap_tracker.hpp>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static constexpr uint32_t MSG_BUFFER_SIZE = 256;

/// Validates a boolean condition and returns an FfxApiReturnCodes code if the condition is not met.
///
/// Use this macro to validate parameters and invariants in ffx-api provider entry points.
///
/// @param [in] _cond    The condition expression to evaluate (e.g., ptr != nullptr).
/// @param [in] _retcode The API return code to return on failure (e.g., FFX_API_RETURN_ERROR_PARAMETER).
///                      Must be of type FfxApiReturnCodes or ffxReturnCode_t, not an internal FfxErrorCode.
#define VERIFY(_cond, _retcode)                                                                                                                       \
    do                                                                                                                                                \
    {                                                                                                                                                 \
        static_assert(                                                                                                                                \
            std::is_same_v<std::decay_t<decltype(_retcode)>, FfxApiReturnCodes> || std::is_same_v<std::decay_t<decltype(_retcode)>, ffxReturnCode_t>, \
            "VERIFY requires an FfxApiReturnCodes / ffxReturnCode_t value");                                                                          \
        if (!(_cond))                                                                                                                                 \
        {                                                                                                                                             \
            char ffxMsg[MSG_BUFFER_SIZE];                                                                                                             \
            snprintf(ffxMsg, sizeof(ffxMsg), "validation failed: %s (return code %u)", #_cond, static_cast<uint32_t>(_retcode));                      \
            ffxLogMessage(ffxLogComponentName(), FFX_MESSAGE_TYPE_ERROR, ffxMsg);                                                                     \
            return _retcode;                                                                                                                          \
        }                                                                                                                                             \
    } while (0)

/// Executes an expression returning an API return code (FfxApiReturnCodes / ffxReturnCode_t).
///
/// If the expression does not return FFX_API_RETURN_OK, logs the failure and propagates
/// the returned FfxApiReturnCodes error code to the caller.
///
/// @note Use TRY when invoking functions that return FfxApiReturnCodes / ffxReturnCode_t
///       (e.g., CreateBackend, MustCreateBackend, or other provider APIs).
///       Do NOT use TRY for internal SDK APIs returning FfxErrorCode; use TRY2 instead.
///
/// @param [in] _expr An expression returning FfxApiReturnCodes or ffxReturnCode_t.
#define TRY(_expr)                                                                                                                                        \
    do                                                                                                                                                    \
    {                                                                                                                                                     \
        static_assert(std::is_same_v<std::decay_t<decltype(_expr)>, FfxApiReturnCodes> || std::is_same_v<std::decay_t<decltype(_expr)>, ffxReturnCode_t>, \
                      "TRY requires an expression returning FfxApiReturnCodes / ffxReturnCode_t (use TRY2 for FfxErrorCode)");                            \
        const auto ffxReturnCode = (_expr);                                                                                                               \
        if (ffxReturnCode != FFX_API_RETURN_OK)                                                                                                           \
        {                                                                                                                                                 \
            char ffxMsg[MSG_BUFFER_SIZE];                                                                                                                 \
            snprintf(ffxMsg, sizeof(ffxMsg), "call failed: %s (FfxApiReturnCodes=%u)", #_expr, static_cast<uint32_t>(ffxReturnCode));                     \
            ffxLogMessage(ffxLogComponentName(), FFX_MESSAGE_TYPE_ERROR, ffxMsg);                                                                         \
            return ffxReturnCode;                                                                                                                         \
        }                                                                                                                                                 \
    } while (0)

/// Executes an expression returning an internal SDK error code (FfxErrorCode / FfxErrorCodes).
///
/// If the expression does not return FFX_OK, logs the internal SDK error code and reason,
/// and translates the failure into FFX_API_RETURN_ERROR_RUNTIME_ERROR for the ffx-api caller.
///
/// @note Use TRY2 when invoking internal FidelityFX SDK host or backend APIs
///       (e.g., ffxCreateContext, fpCreateResource, ffxGetInterfaceVK).
///       Do NOT use TRY2 for expressions returning FfxApiReturnCodes; use TRY instead.
///
/// @param [in] _expr An expression returning FfxErrorCode or FfxErrorCodes.
#define TRY2(_expr)                                                                                                                                \
    do                                                                                                                                             \
    {                                                                                                                                              \
        static_assert(std::is_same_v<std::decay_t<decltype(_expr)>, FfxErrorCode> || std::is_same_v<std::decay_t<decltype(_expr)>, FfxErrorCodes>, \
                      "TRY2 requires an expression returning FfxErrorCode or FfxErrorCodes");                                                      \
        const FfxErrorCode ffxErrorCode = (_expr);                                                                                                 \
        if (ffxErrorCode != FFX_OK)                                                                                                                \
        {                                                                                                                                          \
            ffxLogError(ffxErrorCode, "call failed: " #_expr);                                                                                     \
            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;                                                                                             \
        }                                                                                                                                          \
    } while (0)

struct Allocator
{
    const ffxAllocationCallbacks* cb;
#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
    ffx::EffectHeapTracker* heapTracker = nullptr;
#endif

    void* alloc(size_t sz)
    {
        void* allocation = nullptr;
        if (cb)
            allocation = cb->alloc(cb->pUserData, static_cast<uint64_t>(sz));
        else
            allocation = malloc(sz);
#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
        if (allocation && heapTracker)
            heapTracker->Add(sz);
#endif
        return allocation;
    }

    void dealloc(void* ptr, size_t sz)
    {
#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
        if (ptr && heapTracker)
            heapTracker->Remove(sz);
#endif
        if (cb)
            cb->dealloc(cb->pUserData, ptr);
        else
            free(ptr);
    }

#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
    Allocator withTracker(ffx::EffectHeapTracker& tracker) const
    {
        return {cb, &tracker};
    }
#endif

    template <typename T>
    T* construct()
    {
        void* addr = alloc(sizeof(T));
        return reinterpret_cast<T*>(addr);
    }

    template <typename T, typename... Args>
    T* construct(Args&&... args)
    {
        void* addr = alloc(sizeof(T));
        return ::new (addr) T(std::forward<Args>(args)...);
    }
};

class ffxProvider
{
public:
    virtual ~ffxProvider() = default;

    virtual bool CanProvide(uint64_t) const
    {
        return false;
    }

    virtual uint64_t GetId() const = 0;

    virtual const char* GetVersionName() const = 0;

    virtual ffxReturnCode_t CreateContext(ffxContext* context, ffxCreateContextDescHeader* desc, Allocator& alloc) const = 0;

    virtual ffxReturnCode_t DestroyContext(ffxContext* context, Allocator& alloc) const = 0;

    virtual ffxReturnCode_t Configure(ffxContext* context, const ffxConfigureDescHeader* desc) const = 0;

    virtual ffxReturnCode_t Query(ffxContext* context, ffxQueryDescHeader* desc) const = 0;

    virtual ffxReturnCode_t Dispatch(ffxContext* context, const ffxDispatchDescHeader* desc) const = 0;
};

const ffxProvider* GetffxProvider(ffxStructType_t descType, uint64_t overrideId, void* device);

const ffxProvider* GetAssociatedProvider(ffxContext* context);

uint64_t GetProviderCount(ffxStructType_t descType, void* device);

uint64_t GetProviderVersions(ffxStructType_t descType, void* device, uint64_t capacity, uint64_t* versionIds, const char** versionNames);

struct InternalContextHeader
{
    const ffxProvider* provider;
};

static inline void ffxProviderLogMessage(ffxApiMessage callback, uint32_t type, const char* message)
{
    ffxLogMessage(ffxLogComponentName(), static_cast<FfxMsgType>(type), message);
    if (callback && message)
    {
        callback(type, message);
    }
}

template <typename T>
static inline T ConvertEnum(uint32_t enumVal)
{
    return static_cast<T>(enumVal);
}

template <typename T>
static inline uint32_t ReverseConvertEnum(T enumVal)
{
    return static_cast<T>(enumVal);
}

static inline FfxResource Convert(const FfxApiResource& inRes)
{
    FfxResource outRes{};
    memset(outRes.name, 0, sizeof(outRes.name));
    outRes.resource             = inRes.resource;
    outRes.state                = ConvertEnum<FfxResourceStates>(inRes.state);
    outRes.description.type     = ConvertEnum<FfxResourceType>(inRes.description.type);
    outRes.description.format   = ConvertEnum<FfxSurfaceFormat>(inRes.description.format);
    outRes.description.width    = inRes.description.width;
    outRes.description.height   = inRes.description.height;
    outRes.description.depth    = inRes.description.depth;
    outRes.description.mipCount = inRes.description.mipCount;
    outRes.description.flags    = ConvertEnum<FfxResourceFlags>(inRes.description.flags);
    outRes.description.usage    = ConvertEnum<FfxResourceUsage>(inRes.description.usage);
    return outRes;
}

static inline FfxApiResource Convert(const FfxResource& inRes)
{
    FfxApiResource outRes{};
    outRes.resource             = inRes.resource;
    outRes.state                = ReverseConvertEnum<FfxResourceStates>(inRes.state);
    outRes.description.type     = ReverseConvertEnum<FfxResourceType>(inRes.description.type);
    outRes.description.format   = ReverseConvertEnum<FfxSurfaceFormat>(inRes.description.format);
    outRes.description.width    = inRes.description.width;
    outRes.description.height   = inRes.description.height;
    outRes.description.depth    = inRes.description.depth;
    outRes.description.mipCount = inRes.description.mipCount;
    outRes.description.flags    = ReverseConvertEnum<FfxResourceFlags>(inRes.description.flags);
    outRes.description.usage    = ReverseConvertEnum<FfxResourceUsage>(inRes.description.usage);
    return outRes;
}
