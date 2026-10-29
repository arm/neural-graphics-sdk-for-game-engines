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
//
/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "ffx_provider_nss.h"
#if FFX_CAPTURE_ENABLED
#include "capture/neuralssuperresolution.h"
#endif
#include "backends.h"
#include "validation.h"
#include <ffx_api/ffx_nss.hpp>
#ifdef FFX_BACKEND_VK
#include <ffx_api/vk/ffx_api_vk.h>
#endif  // #ifdef FFX_BACKEND_VK
#include <ffx_heap_tracker.hpp>
#include <FidelityFX/host/ffx_nss.h>

#include <new>
#include <stdlib.h>
#include <cstdint>
#include <cstring>

static constexpr const char* ffxLogComponentName()
{
    return "NSSProvider";
}

static_assert(static_cast<uint32_t>(FfxNssShaderQualityMode::FFX_NSS_SHADER_QUALITY_MODE_QUALITY) ==
              static_cast<uint32_t>(FfxApiNssShaderQualityMode::FFX_API_NSS_SHADER_QUALITY_MODE_QUALITY));
static_assert(static_cast<uint32_t>(FfxNssShaderQualityMode::FFX_NSS_SHADER_QUALITY_MODE_BALANCED) ==
              static_cast<uint32_t>(FfxApiNssShaderQualityMode::FFX_API_NSS_SHADER_QUALITY_MODE_BALANCED));
static_assert(static_cast<uint32_t>(FfxNssShaderQualityMode::FFX_NSS_SHADER_QUALITY_MODE_PERFORMANCE) ==
              static_cast<uint32_t>(FfxApiNssShaderQualityMode::FFX_API_NSS_SHADER_QUALITY_MODE_PERFORMANCE));

static uint32_t ConvertContextFlagsNss(uint32_t apiFlags)
{
    uint32_t outFlags = 0;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_QUANTIZED)
        outFlags |= FFX_NSS_CONTEXT_FLAG_QUANTIZED;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_LOW_DYNAMIC_RANGE)
        outFlags |= FFX_NSS_CONTEXT_FLAG_LOW_DYNAMIC_RANGE;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_DEPTH_INVERTED)
        outFlags |= FFX_NSS_CONTEXT_FLAG_DEPTH_INVERTED;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_DEPTH_INFINITE)
        outFlags |= FFX_NSS_CONTEXT_FLAG_DEPTH_INFINITE;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_RESAMPLE_BICUBIC)
        outFlags |= FFX_NSS_CONTEXT_FLAG_RESAMPLE_BICUBIC;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_DISABLE_16BIT)
        outFlags |= FFX_NSS_CONTEXT_FLAG_DISABLE_16BIT;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_MANAGE_HISTORY)
        outFlags |= FFX_NSS_CONTEXT_FLAG_MANAGE_HISTORY;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_PRE_PROCESS_COMPUTE)
        outFlags |= FFX_NSS_CONTEXT_FLAG_PRE_PROCESS_COMPUTE;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_POST_PROCESS_COMPUTE)
        outFlags |= FFX_NSS_CONTEXT_FLAG_POST_PROCESS_COMPUTE;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_EXPOSURE_TEXTURE)
        outFlags |= FFX_NSS_CONTEXT_FLAG_EXPOSURE_TEXTURE;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION)
        outFlags |= FFX_NSS_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION;
    return outFlags;
}

static uint32_t ConvertDispatchFlagsNss(uint32_t apiFlags)
{
    uint32_t outFlags = 0;
    if (apiFlags & FFX_API_NSS_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING)
        outFlags |= FFX_NSS_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING;
    if (apiFlags & FFX_API_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW)
        outFlags |= FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW;
    return outFlags;
}

bool ffxProvider_Nss::CanProvide(uint64_t type) const
{
    return (type & FFX_API_EFFECT_MASK) == FFX_API_EFFECT_ID_NSS;
}

#define STRINGIFY_(X)                            #X
#define STRINGIFY(X)                             STRINGIFY_(X)
#define MAKE_VERSION_STRING(major, minor, patch) STRINGIFY major "." STRINGIFY minor "." STRINGIFY patch

uint64_t ffxProvider_Nss::GetId() const
{
    // NSS Scale, version from header
    return 0x2555'CA1Eull << 32 | (FFX_SDK_MAKE_VERSION(FFX_NSS_VERSION_MAJOR, FFX_NSS_VERSION_MINOR, FFX_NSS_VERSION_PATCH) & 0xFFFF'FFFF);
}

const char* ffxProvider_Nss::GetVersionName() const
{
    return MAKE_VERSION_STRING(FFX_NSS_VERSION_MAJOR, FFX_NSS_VERSION_MINOR, FFX_NSS_VERSION_PATCH);
}

#if FFX_CAPTURE_ENABLED
using CaptureController = arm::capture::neuralssuperresolution::Controller;
#endif

struct InternalNssContext
{
    InternalContextHeader header;
#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
    ffx::EffectHeapTracker heapTracker;
#endif
    FfxInterface  backendInterface;
    FfxNssContext context;
    ffxApiMessage fpMessage;
#if FFX_CAPTURE_ENABLED
    CaptureController* controller;
#endif
};

static void FreeInternalNssContext(InternalNssContext* context, Allocator& alloc)
{
#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
    Allocator trackedAlloc = alloc.withTracker(context->heapTracker);
#else
    Allocator& trackedAlloc = alloc;
#endif
    if (context->backendInterface.scratchBuffer)
    {
        trackedAlloc.dealloc(context->backendInterface.scratchBuffer, context->backendInterface.scratchBufferSize);
    }
#if FFX_CAPTURE_ENABLED
    if (context->controller)
    {
        arm::capture::DestroyCaptureController(trackedAlloc, context->controller);
    }
#endif
#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
    context->heapTracker.Remove(sizeof(InternalNssContext));
    context->heapTracker.~EffectHeapTracker();
#endif
    alloc.dealloc(context, sizeof(InternalNssContext));
}

ffxReturnCode_t ffxProvider_Nss::CreateContext(ffxContext* context, ffxCreateContextDescHeader* header, Allocator& alloc) const
{
    VERIFY(context, FFX_API_RETURN_ERROR_PARAMETER);
    VERIFY(header, FFX_API_RETURN_ERROR_PARAMETER);

    if (auto desc = ffx::DynamicCast<ffxApiCreateContextDescNss>(header))
    {
        if (desc->fpMessage)
        {
#ifdef FFX_BACKEND_VK
            Validator{desc->fpMessage, header}.AcceptExtensions({FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK, FFX_API_DESC_TYPE_OVERRIDE_VERSION});
#endif  // FFX_BACKEND_VK
        }
        InternalNssContext* internal_context = alloc.construct<InternalNssContext>();
        VERIFY(internal_context, FFX_API_RETURN_ERROR_MEMORY);
        memset(internal_context, 0, sizeof(*internal_context));
#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
        ::new (&internal_context->heapTracker) ffx::EffectHeapTracker(sizeof(InternalNssContext));
        Allocator trackedAlloc = alloc.withTracker(internal_context->heapTracker);
#else
        Allocator& trackedAlloc = alloc;
#endif
        internal_context->header.provider = this;

        const ffxReturnCode_t backendResult = MustCreateBackend(header, &internal_context->backendInterface, 1, trackedAlloc);
        if (backendResult != FFX_API_RETURN_OK)
        {
            FreeInternalNssContext(internal_context, alloc);
            return backendResult;
        }

        FfxNssContextDescription initializationParameters = {};
        initializationParameters.backendInterface         = internal_context->backendInterface;
        initializationParameters.renderSize.width         = desc->maxRenderSize.width;
        initializationParameters.renderSize.height        = desc->maxRenderSize.height;
        initializationParameters.upscaleSize.width        = desc->maxUpscaleSize.width;
        initializationParameters.upscaleSize.height       = desc->maxUpscaleSize.height;

        initializationParameters.displaySize.width  = desc->maxUpscaleSize.width;
        initializationParameters.displaySize.height = desc->maxUpscaleSize.height;

        const FfxApiNssShaderQualityMode qualityMode =
            (desc->qualityMode >= FFX_API_NSS_SHADER_QUALITY_MODE_QUALITY && desc->qualityMode <= FFX_API_NSS_SHADER_QUALITY_MODE_PERFORMANCE)
                ? desc->qualityMode
                : FFX_API_NSS_SHADER_QUALITY_MODE_BALANCED;
        initializationParameters.qualityMode = ConvertEnum<FfxNssShaderQualityMode>(qualityMode);
        initializationParameters.flags       = ConvertContextFlagsNss(desc->flags);
        // Calling this casted function is undefined behaviour, but it's probably safe.
        initializationParameters.fpMessage = reinterpret_cast<FfxNssMessage>(desc->fpMessage);

        // Grab this fp for use in extensions later
        internal_context->fpMessage = desc->fpMessage;

        // Create the NSS context
        const FfxErrorCode createResult = ffxNssContextCreate(&internal_context->context, &initializationParameters);
        if (createResult != FFX_OK)
        {
            FreeInternalNssContext(internal_context, alloc);
            return createResult == FFX_ERROR_INVALID_ARGUMENT ? FFX_API_RETURN_ERROR_PARAMETER : FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        }

#if FFX_CAPTURE_ENABLED
        // A successfully created provider context owns a capture controller when
        // runtime capture integration is enabled for this build.
        internal_context->controller =
            arm::capture::MakeCaptureController<CaptureController>(trackedAlloc, internal_context->backendInterface.device, desc->flags);
        if (!internal_context->controller)
        {
            ffxNssContextDestroy(&internal_context->context);
            FreeInternalNssContext(internal_context, alloc);
            return FFX_API_RETURN_ERROR_MEMORY;
        }
#endif

        *context = internal_context;
        return FFX_API_RETURN_OK;
    }
    else
    {
        ffxLogMessage(ffxLogComponentName(), FFX_MESSAGE_TYPE_ERROR, "unsupported NSS create descriptor");
        return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }
}

ffxReturnCode_t ffxProvider_Nss::DestroyContext(ffxContext* context, Allocator& alloc) const
{
    VERIFY(context, FFX_API_RETURN_ERROR_PARAMETER);
    VERIFY(*context, FFX_API_RETURN_ERROR_PARAMETER);

    InternalNssContext* internal_context = reinterpret_cast<InternalNssContext*>(*context);
#if FFX_CAPTURE_ENABLED
    if (internal_context->controller)
    {
        if (!internal_context->controller->Shutdown())
            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
#endif

    if (ffxNssContextDestroy(&internal_context->context) != FFX_OK)
    {
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
    FreeInternalNssContext(internal_context, alloc);
    *context = nullptr;

    return FFX_API_RETURN_OK;
}

ffxReturnCode_t ffxProvider_Nss::Configure(ffxContext* context, const ffxConfigureDescHeader* header) const
{
    VERIFY(context, FFX_API_RETURN_ERROR_PARAMETER);
    VERIFY(*context, FFX_API_RETURN_ERROR_PARAMETER);
    VERIFY(header, FFX_API_RETURN_ERROR_PARAMETER);

    InternalNssContext* internal_context = reinterpret_cast<InternalNssContext*>(*context);
    if (auto desc = ffx::DynamicCast<ffxApiConfigureDescCapture>(header))
    {
#if !FFX_CAPTURE_ENABLED
        (void)desc;
        return FFX_API_RETURN_ERROR_PARAMETER;
#else
        return internal_context->controller->Configure(*desc);
#endif
    }
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t ffxProvider_Nss::Query(ffxContext* context, ffxQueryDescHeader* header) const
{
    VERIFY(header, FFX_API_RETURN_ERROR_PARAMETER);

    InternalNssContext* internal_context = context && *context ? reinterpret_cast<InternalNssContext*>(*context) : nullptr;
    if (internal_context)
    {
        if (internal_context->fpMessage)
        {
            Validator{internal_context->fpMessage, header}.NoExtensions();
        }
    }

    switch (header->type)
    {
    case FFX_API_QUERY_DESC_TYPE_CAPTURESTATUS:
    {
        VERIFY(context, FFX_API_RETURN_ERROR_PARAMETER);
        VERIFY(*context, FFX_API_RETURN_ERROR_PARAMETER);
        auto desc = reinterpret_cast<ffxApiQueryDescCaptureStatus*>(header);
        VERIFY(desc->pOutStatus, FFX_API_RETURN_ERROR_PARAMETER);
#if !FFX_CAPTURE_ENABLED
        return FFX_API_RETURN_ERROR_PARAMETER;
#else
        internal_context->controller->GetStatus(*desc->pOutStatus);
        return FFX_API_RETURN_OK;
#endif
    }
    case FFX_API_QUERY_DESC_TYPE_NSS_GETJITTEROFFSET:
    {
        auto  desc = reinterpret_cast<ffxApiQueryDescNssGetJitterOffset*>(header);
        float jitterX, jitterY;
        TRY2(ffxNssGetJitterOffset(&jitterX, &jitterY, desc->index, desc->phaseCount));
        if (desc->pOutX != nullptr)
        {
            *desc->pOutX = jitterX;
        }
        if (desc->pOutY != nullptr)
        {
            *desc->pOutY = jitterY;
        }
        break;
    }
    case FFX_API_QUERY_DESC_TYPE_NSS_GETJITTERPHASECOUNT:
    {
        auto          desc             = reinterpret_cast<ffxApiQueryDescNssGetJitterPhaseCount*>(header);
        const int32_t jitterPhaseCount = ffxNssGetJitterPhaseCount(desc->renderWidth, desc->displayWidth);

        if (desc->pOutPhaseCount != nullptr)
        {
            *desc->pOutPhaseCount = jitterPhaseCount;
        }
        break;
    }
    case FFX_API_QUERY_DESC_TYPE_NSS_GPU_MEMORY_USAGE:
    {
        VERIFY(context, FFX_API_RETURN_ERROR_PARAMETER);
        VERIFY(*context, FFX_API_RETURN_ERROR_PARAMETER);

        auto desc = reinterpret_cast<ffxApiQueryDescNssGetGPUMemoryUsage*>(header);
        VERIFY(desc->gpuMemoryUsageNss, FFX_API_RETURN_ERROR_PARAMETER);

        FfxEffectMemoryUsage usage{};
        FfxEffectMemoryUsage backendUsage{};
        TRY2(ffxNssContextGetGpuMemoryUsage(&internal_context->context, &usage));
        TRY2(internal_context->backendInterface.fpGetBackendGpuMemoryUsage(&internal_context->backendInterface, &backendUsage));

        desc->gpuMemoryUsageNss->totalUsageInBytes     = usage.totalUsageInBytes + backendUsage.totalUsageInBytes;
        desc->gpuMemoryUsageNss->textureUsageInBytes   = usage.textureUsageInBytes + backendUsage.textureUsageInBytes;
        desc->gpuMemoryUsageNss->bufferUsageInBytes    = usage.bufferUsageInBytes + backendUsage.bufferUsageInBytes;
        desc->gpuMemoryUsageNss->tensorUsageInBytes    = usage.tensorUsageInBytes + backendUsage.tensorUsageInBytes;
        desc->gpuMemoryUsageNss->dataGraphUsageInBytes = usage.dataGraphUsageInBytes + backendUsage.dataGraphUsageInBytes;
        desc->gpuMemoryUsageNss->aliasableUsageInBytes = usage.aliasableUsageInBytes + backendUsage.aliasableUsageInBytes;
        break;
    }
#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
    case FFX_API_QUERY_DESC_TYPE_NSS_CPU_MEMORY_USAGE:
    {
        VERIFY(context, FFX_API_RETURN_ERROR_PARAMETER);
        VERIFY(*context, FFX_API_RETURN_ERROR_PARAMETER);

        auto desc = reinterpret_cast<ffxApiQueryDescNssGetCPUMemoryUsage*>(header);
        VERIFY(desc->cpuMemoryUsageNss, FFX_API_RETURN_ERROR_PARAMETER);

        FfxApiEffectCpuMemoryUsage& usage = *desc->cpuMemoryUsageNss;
        usage.currentUsageInBytes         = internal_context->heapTracker.GetCurrentUsageInBytes();
        usage.peakUsageInBytes            = internal_context->heapTracker.GetPeakUsageInBytes();
        break;
    }
#endif
    default:
        ffxLogMessage(ffxLogComponentName(), FFX_MESSAGE_TYPE_ERROR, "unsupported NSS query descriptor");
        return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }

    return FFX_API_RETURN_OK;
}

ffxReturnCode_t ffxProvider_Nss::Dispatch(ffxContext* context, const ffxDispatchDescHeader* header) const
{
    VERIFY(context, FFX_API_RETURN_ERROR_PARAMETER);
    VERIFY(*context, FFX_API_RETURN_ERROR_PARAMETER);
    VERIFY(header, FFX_API_RETURN_ERROR_PARAMETER);

    InternalNssContext* internal_context = reinterpret_cast<InternalNssContext*>(*context);
    if (internal_context->fpMessage)
    {
        Validator{internal_context->fpMessage, header}.NoExtensions();
    }

    switch (header->type)
    {
    case FFX_API_DISPATCH_DESC_TYPE_NSS:
    {
        auto desc = reinterpret_cast<const ffxApiDispatchDescNss*>(header);

#if FFX_CAPTURE_ENABLED
        const bool captureActive = internal_context->controller->BeginDispatchCapture(*desc, internal_context->context);
#else
        constexpr bool captureActive = false;
#endif

        FfxNssDispatchDescription dispatchParameters = {};
        dispatchParameters.commandList               = desc->commandList;
        dispatchParameters.color                     = Convert(desc->color);
        dispatchParameters.depth                     = Convert(desc->depth);
        dispatchParameters.outputTm1                 = Convert(desc->outputTm1);
        dispatchParameters.motionVectors             = Convert(desc->motionVectors);
        dispatchParameters.output                    = Convert(desc->output);
        dispatchParameters.jitterOffset.x            = desc->jitterOffset.x;
        dispatchParameters.jitterOffset.y            = desc->jitterOffset.y;
        dispatchParameters.cameraFar                 = desc->cameraFar;
        dispatchParameters.cameraNear                = desc->cameraNear;
        dispatchParameters.cameraFovAngleVertical    = desc->cameraFovAngleVertical;
        dispatchParameters.exposure                  = desc->exposure;
        dispatchParameters.exposureTexture           = Convert(desc->exposureTexture);
        dispatchParameters.motionVectorScale.x       = desc->motionVectorScale.x;
        dispatchParameters.motionVectorScale.y       = desc->motionVectorScale.y;
#if FFX_CAPTURE_ENABLED
        dispatchParameters.reset = desc->reset || (captureActive && internal_context->controller->ShouldResetHistoryForDispatch());
#else
        dispatchParameters.reset     = desc->reset;
#endif
        dispatchParameters.frameTimeDelta     = desc->frameTimeDelta;
        dispatchParameters.upscaleSize.width  = desc->upscaleSize.width;
        dispatchParameters.upscaleSize.height = desc->upscaleSize.height;
        dispatchParameters.renderSize.width   = desc->renderSize.width;
        dispatchParameters.renderSize.height  = desc->renderSize.height;
        dispatchParameters.flags              = ConvertDispatchFlagsNss(desc->flags);
        if (desc->flags & FFX_API_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW)
        {
            dispatchParameters.debugViews    = Convert(desc->debugViews);
            dispatchParameters.debugViewMode = desc->debugViewMode;
        }

#if FFX_CAPTURE_ENABLED
        ARM_LOG_DISPATCH_TIMESTAMP_IF_ENABLED("NSS-PERF", captureActive);
#endif

#if FFX_CAPTURE_ENABLED
        TRY2_CAPTURE(ffxNssContextDispatch(&internal_context->context, &dispatchParameters));
#else
        TRY2(ffxNssContextDispatch(&internal_context->context, &dispatchParameters));
#endif

#if FFX_CAPTURE_ENABLED
        if (captureActive)
            static_cast<void>(internal_context->controller->CompleteDispatchCapture(*desc, internal_context->context));
#endif

        break;
    }
    default:
        ffxLogMessage(ffxLogComponentName(), FFX_MESSAGE_TYPE_ERROR, "unsupported NSS dispatch descriptor");
        return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }

    return FFX_API_RETURN_OK;
}

ffxProvider_Nss ffxProvider_Nss::Instance;
