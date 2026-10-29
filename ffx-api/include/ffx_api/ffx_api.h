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

#if defined(__cplusplus)
extern "C" {
#endif  // #if defined(__cplusplus)

#if defined(_MSC_VER)
#define FFX_API_ENTRY __declspec(dllexport)  // Microsoft
#elif defined(__GNUC__)
#define FFX_API_ENTRY __attribute__((visibility("default")))  // GCC
#else
#define FFX_API_ENTRY  // Most compilers export all the symbols by default. We hope for the best here.
#pragma warning Unknown dynamic link import / export semantics.
#endif

#include <stdint.h>

enum FfxApiReturnCodes
{
    FFX_API_RETURN_OK    = 0,  ///< The oparation was successful.
    FFX_API_RETURN_ERROR = 1,  ///< An error occurred that is not further specified.
    FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE =
        2,  ///< The structure type given was not recognized for the function or context with which it was used. This is likely a programming error.
    FFX_API_RETURN_ERROR_RUNTIME_ERROR = 3,  ///< The underlying runtime (e.g. D3D12, Vulkan) or effect returned an error code.
    FFX_API_RETURN_NO_PROVIDER         = 4,  ///< No provider was found for the given structure type. This is likely a programming error.
    FFX_API_RETURN_ERROR_MEMORY        = 5,  ///< A memory allocation failed.
    FFX_API_RETURN_ERROR_PARAMETER     = 6,  ///< A parameter was invalid, e.g. a null pointer, empty resource or out-of-bounds enum value.
};

typedef void*    ffxContext;
typedef uint32_t ffxReturnCode_t;

#define FFX_API_EFFECT_MASK       0xffff0000u
#define FFX_API_EFFECT_ID_GENERAL 0x00000000u

// Base Descriptor types
typedef uint64_t ffxStructType_t;
typedef struct ffxApiHeader
{
    ffxStructType_t      type;   ///< The structure type. Must always be set to the corresponding value for any structure (found nearby with a similar name).
    struct ffxApiHeader* pNext;  ///< Pointer to next structure, used for optional parameters and extensions. Can be null.
} ffxApiHeader;

typedef ffxApiHeader ffxCreateContextDescHeader;
typedef ffxApiHeader ffxConfigureDescHeader;
typedef ffxApiHeader ffxQueryDescHeader;
typedef ffxApiHeader ffxDispatchDescHeader;

// Extensions for global debug
#define FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_SILENCE  0x0000000u
#define FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_ERRORS   0x0000001u
#define FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_WARNINGS 0x0000002u
#define FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE  0xfffffffu

enum FfxApiMsgType
{
    FFX_API_MESSAGE_TYPE_ERROR   = 0,
    FFX_API_MESSAGE_TYPE_WARNING = 1,
    FFX_API_MESSAGE_TYPE_INFO    = 2,
    FFX_API_MESSAGE_TYPE_COUNT
};

typedef void (*ffxApiMessage)(uint32_t type, const char* message);

#define FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1 0x0000001u
struct ffxConfigureDescGlobalDebug1
{
    ffxConfigureDescHeader header;
    ffxApiMessage          fpMessage;
    uint32_t               debugLevel;
};

typedef enum FfxApiCaptureOperation
{
    FFX_API_CAPTURE_BEGIN  = 0,  ///< Queue a capture using the descriptor settings.
    FFX_API_CAPTURE_CANCEL = 2,  ///< Stop an active capture early and cancel its incomplete transaction.
} FfxApiCaptureOperation;

typedef enum FfxApiCapturePhase
{
    FFX_API_CAPTURE_PHASE_IDLE      = 0,
    FFX_API_CAPTURE_PHASE_CAPTURING = 1,
    FFX_API_CAPTURE_PHASE_DRAINING  = 2,
} FfxApiCapturePhase;

typedef enum FfxApiCaptureResult
{
    FFX_API_CAPTURE_RESULT_NONE      = 0,
    FFX_API_CAPTURE_RESULT_SUCCESS   = 1,
    FFX_API_CAPTURE_RESULT_FAILED    = 2,
    FFX_API_CAPTURE_RESULT_CANCELLED = 3,
} FfxApiCaptureResult;

#define FFX_API_CONFIGURE_DESC_TYPE_CAPTURE 6u
typedef struct ffxApiConfigureDescCapture
{
    ffxConfigureDescHeader header;
    uint32_t               operation;  ///< One of FfxApiCaptureOperation.
    const char*
             outputDirectory;  ///< BEGIN only: non-null, NUL-terminated output root of at most 512 bytes; copied before ffxConfigure returns. Longer values return FFX_API_RETURN_ERROR_PARAMETER. For a valid descriptor, BEGIN is accepted for an idle context even without environment/property configuration; state-based rejection applies when an API BEGIN is already pending or the context is capturing or draining. Invalid descriptor values return FFX_API_RETURN_ERROR_PARAMETER.
    uint32_t startDispatch;    ///< First component dispatch eligible for capture.
    uint32_t frameCount;       ///< Maximum number of frames to request; capture may stop earlier on budget or runtime failure.
    float    motionVectorUnormBias;   ///< Finite motion-vector conversion bias; use 0.5 for R8G8_UNORM inputs.
    float    motionVectorUnormScale;  ///< Finite non-zero scale whose bias-adjusted R8G8_UNORM result remains finite; use 2.0 for R8G8_UNORM inputs.
} ffxApiConfigureDescCapture;

typedef struct FfxApiCaptureStatus
{
    // The lifecycle phase and the most recent capture result are independent:
    // a draining capture can already have a known result.
    uint32_t phase;            ///< One of FfxApiCapturePhase: IDLE, CAPTURING, or DRAINING.
    uint32_t result;           ///< One of FfxApiCaptureResult; FAILED is retained after asynchronous failure.
    uint32_t capturedFrames;   ///< Frames submitted for asynchronous readback; later payload/I/O failure is reported in result.
    uint32_t requestedFrames;  ///< Requested frame count for the active or most recent capture.
} FfxApiCaptureStatus;

#define FFX_API_QUERY_DESC_TYPE_CAPTURESTATUS 7u
typedef struct ffxApiQueryDescCaptureStatus
{
    ffxQueryDescHeader   header;
    FfxApiCaptureStatus* pOutStatus;  ///< Non-null output storage written synchronously by ffxQuery.
} ffxApiQueryDescCaptureStatus;

#define FFX_API_QUERY_DESC_TYPE_GET_VERSIONS 4u
struct ffxQueryDescGetVersions
{
    ffxQueryDescHeader header;
    uint64_t           createDescType;  ///< Create description for the effect whose versions should be enumerated.
    void*              device;          ///< For DX12: pointer to ID3D12Device.
    uint64_t*
              outputCount;  ///< Input capacity of id and name arrays. Output number of returned versions. If initially zero, output is number of available versions.
    uint64_t* versionIds;  ///< Output array of version ids to be used as version overrides. If null, only names and count are returned.
    const char**
        versionNames;  ///< Output array of version names for display. If null, only ids and count are returned. If both this and versionIds are null, only count is returned.
};

#define FFX_API_DESC_TYPE_OVERRIDE_VERSION 5u
struct ffxOverrideVersion
{
    ffxApiHeader header;
    uint64_t     versionId;  ///< Id of version to use. Must be a value returned from a query in ffxQueryDescGetVersions.versionIds array.
};

// Memory allocation function. Must return a valid pointer to at least size bytes of memory aligned to hold any type.
// May return null to indicate failure. Standard library malloc fulfills this requirement.
typedef void* (*ffxAlloc)(void* pUserData, uint64_t size);

// Memory deallocation function. May be called with null pointer as second argument.
typedef void (*ffxDealloc)(void* pUserData, void* pMem);

typedef struct ffxAllocationCallbacks
{
    void*      pUserData;
    ffxAlloc   alloc;
    ffxDealloc dealloc;
} ffxAllocationCallbacks;

// Context objects are thread-affine. Calls operating on the same context must
// be externally serialized by its owning thread. Different contexts may be
// used concurrently from different threads.

// Creates a FFX object context.
// Depending on the desc structures provided to this function, the context will be created with the desired version and attributes.
// Non-zero return indicates error code.
// Pointers passed in desc must remain live until ffxDestroyContext is called on the context.
// MemCb may be null; the system allocator (malloc/free) will be used in this case.
FFX_API_ENTRY ffxReturnCode_t ffxCreateContext(ffxContext* context, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* memCb);
typedef ffxReturnCode_t (*PfnFfxCreateContext)(ffxContext* context, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* memCb);

// Destroys an FFX object context.
// Non-zero return indicates error code.
// MemCb must be compatible with the callbacks passed into ffxCreateContext.
FFX_API_ENTRY ffxReturnCode_t ffxDestroyContext(ffxContext* context, const ffxAllocationCallbacks* memCb);
typedef ffxReturnCode_t (*PfnFfxDestroyContext)(ffxContext* context, const ffxAllocationCallbacks* memCb);

// Configures the provided FFX object context.
// Capture configuration is per context; a null context is invalid for capture descriptors.
// For capture BEGIN, outputDirectory is copied before this function returns.
// Capture readback and file output are asynchronous; poll capture status with ffxQuery until
// the phase is IDLE after issuing CANCEL and asynchronous drain completion.
// Capture BEGIN is accepted while the context is IDLE and no API BEGIN is pending,
// regardless of whether environment/property configuration exists. An accepted API
// BEGIN supersedes pending environment/property requests.
// CANCEL is accepted only while a capture is CAPTURING.
// CANCEL preserves completed output but discards the incomplete transaction.
// Non-zero return indicates error code.
FFX_API_ENTRY ffxReturnCode_t ffxConfigure(ffxContext* context, const ffxConfigureDescHeader* desc);
typedef ffxReturnCode_t (*PfnFfxConfigure)(ffxContext* context, const ffxConfigureDescHeader* desc);

// Queries the provided FFX object context.
// Capture status is written to the descriptor's pOutStatus; the pointer must be valid for the call.
// Non-zero return indicates error code.
FFX_API_ENTRY ffxReturnCode_t ffxQuery(ffxContext* context, ffxQueryDescHeader* desc);
typedef ffxReturnCode_t (*PfnFfxQuery)(ffxContext* context, ffxQueryDescHeader* desc);

// Dispatches work on the given FFX object context defined by the dispatch descriptor.
// Non-zero return indicates error code.
FFX_API_ENTRY ffxReturnCode_t ffxDispatch(ffxContext* context, const ffxDispatchDescHeader* desc);
typedef ffxReturnCode_t (*PfnFfxDispatch)(ffxContext* context, const ffxDispatchDescHeader* desc);

#if defined(__cplusplus)
}
#endif  // #if defined(__cplusplus)
