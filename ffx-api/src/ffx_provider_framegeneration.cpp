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

#include "ffx_provider_framegeneration.h"
#include "ffx_provider_framegeneration_diagnostics.h"
#include "backends.h"
#include <ffx_api/ffx_framegeneration.hpp>
#include <FidelityFX/host/ffx_frameinterpolation.h>
#include <FidelityFX/host/ffx_frameinterpolation_resolution.h>
#include "ffx_frameinterpolation_trace.h"
#include <ffx_heap_tracker.hpp>
#include <FidelityFX/host/ffx_opticalflow.h>
#include "ffx_opticalflow_trace.h"
#include <FidelityFX/gpu/frameinterpolation/ffx_frameinterpolation_resources.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cassert>
#include <memory>
#if FFX_CAPTURE_ENABLED
#include "capture/framegenerations.h"
#endif

static constexpr const char* ffxLogComponentName()
{
    return "NFRU";
}

using NfruCpuClock = std::chrono::steady_clock;

struct NfruRuntimeState
{
    bool     enabled;
    bool     requiresRecreation;
    bool     hasDispatched;
    uint64_t previousFrameId;
};

struct PreparedFrame
{
    ffxApiDispatchDescFrameGenerationPrepare description;
    double                                   cpuPrepareUs;
    bool                                     valid;
};

static uint32_t ConvertContextFlagsFI(const uint32_t apiFlags)
{
    uint32_t outFlags = 0;
    if (apiFlags & FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED)
    {
        outFlags |= FFX_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED;
    }
    if (apiFlags & FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INFINITE)
    {
        outFlags |= FFX_FG_CONTEXT_FLAG_ENABLE_DEPTH_INFINITE;
    }
    if (apiFlags & FFX_API_FG_CONTEXT_FLAG_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS)
    {
        outFlags |= FFX_FG_CONTEXT_FLAG_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS;
    }
    if (apiFlags & FFX_API_FG_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION)
    {
        outFlags |= FFX_FG_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION;
    }
    if (apiFlags & FFX_API_FG_CONTEXT_FLAG_ENABLE_HIGH_DYNAMIC_RANGE)
    {
        outFlags |= FFX_FG_CONTEXT_FLAG_ENABLE_HIGH_DYNAMIC_RANGE;
    }
    if (apiFlags & FFX_API_FG_CONTEXT_FLAG_WARP_COMPUTE)
    {
        outFlags |= FFX_FG_CONTEXT_FLAG_WARP_COMPUTE;
    }
    if (apiFlags & FFX_API_FG_CONTEXT_FLAG_FILL_HOLES_COMPUTE)
    {
        outFlags |= FFX_FG_CONTEXT_FLAG_FILL_HOLES_COMPUTE;
    }
    if (apiFlags & FFX_API_FG_CONTEXT_FLAG_PRE_PROCESS_COMPUTE)
    {
        outFlags |= FFX_FG_CONTEXT_FLAG_PRE_PROCESS_COMPUTE;
    }
    if (apiFlags & FFX_API_FG_CONTEXT_FLAG_POSTPROCESS_COMPUTE)
    {
        outFlags |= FFX_FG_CONTEXT_FLAG_POSTPROCESS_COMPUTE;
    }
    if (apiFlags & FFX_API_FG_CONTEXT_FLAG_DEBUG_VIEW_COMPUTE)
    {
        outFlags |= FFX_FG_CONTEXT_FLAG_DEBUG_VIEW_COMPUTE;
    }
    if (apiFlags & FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH)
    {
        outFlags |= FFX_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH;
    }
    if (apiFlags & FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR)
    {
        outFlags |= FFX_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR;
    }
    return outFlags;
}

// Formats accepted by the optical flow lane (those that map to a supported VkFormat).
static bool IsBackBufferFormatSupportedByOpticalFlow(uint32_t apiFormat)
{
    switch (apiFormat)
    {
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM:
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_TYPELESS:
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM:
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_TYPELESS:
    case FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT:
    case FFX_API_SURFACE_FORMAT_R8_UNORM:
        return true;
    default:
        return false;
    }
}

static FfxOpticalFlowGridSize SelectDefaultOpticalFlowGrid(uint32_t supportedGridSizes)
{
    if (supportedGridSizes & FFX_OPTICAL_FLOW_GRID_SIZE_4X4)
    {
        return FFX_OPTICAL_FLOW_GRID_SIZE_4X4;
    }
    if (supportedGridSizes & FFX_OPTICAL_FLOW_GRID_SIZE_8X8)
    {
        return FFX_OPTICAL_FLOW_GRID_SIZE_8X8;
    }
    if (supportedGridSizes & FFX_OPTICAL_FLOW_GRID_SIZE_2X2)
    {
        return FFX_OPTICAL_FLOW_GRID_SIZE_2X2;
    }
    if (supportedGridSizes & FFX_OPTICAL_FLOW_GRID_SIZE_1X1)
    {
        return FFX_OPTICAL_FLOW_GRID_SIZE_1X1;
    }
    return FFX_OPTICAL_FLOW_GRID_SIZE_UNKNOWN;
}

bool ffxProvider_FrameGeneration::CanProvide(uint64_t type) const
{
    return (type & FFX_API_EFFECT_MASK) == FFX_API_EFFECT_ID_FRAMEGENERATION;
}

const uint32_t MAX_QUEUED_FRAMES = 2;

#if FFX_CAPTURE_ENABLED
using CaptureController = arm::capture::framegenerations::Controller;
#endif

struct InternalFgContext
{
    InternalContextHeader header;
#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
    ffx::EffectHeapTracker heapTracker;
#endif

    FfxInterface                 backendInterfaceFi;
    FfxInterface                 backendInterfaceShared;
    FfxOpticalFlowContext        armOFContext;
    FfxFrameInterpolationContext armFiContext;
    FfxResourceInternal          sharedResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNT];
    uint32_t                     effectContextIdShared, effectContextIdFi;
    uint32_t                     flags;
    float                        deltaTime;

    bool             frameGenEnabled;
    uint32_t         frameGenFlags;
    PreparedFrame    preparedFrames[MAX_QUEUED_FRAMES];
    NfruRuntimeState runtimeState;
    NfruDiagnostics  diagnostics;

    struct Callbacks
    {
        FfxApiCallbackFrameGenerationPresentFunc presentCallback;
        void*                                    presentCallbackUserContext;
        FfxApiDispatchFrameGenerationFunc        frameGenerationCallback;
        void*                                    frameGenerationCallbackUserContext;
    } callbacks[MAX_QUEUED_FRAMES];

    uint64_t lastConfigureFrameID;

    bool        dumpGeneratedFrame;
    const char* dumpGeneratedFramePath;

    FfxNfruResolutionPlan resolutionPlan;
    bool                  usesInternalDepthHistory;
    bool                  fp16Supported;
    FfxDimensions2D       opticalFlowSize;
#if FFX_CAPTURE_ENABLED
    CaptureController* controller;
#endif
};

static NfruLogContextInfo GetLogContextInfo(const InternalFgContext* context)
{
    return {
        &context->resolutionPlan, context->opticalFlowSize, context->flags, context->frameGenFlags, context->usesInternalDepthHistory, context->fp16Supported};
}

static double elapsedMicroseconds(NfruCpuClock::time_point begin, NfruCpuClock::time_point end)
{
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

template <typename Callback>
static auto callWithoutNestedLogging(Callback callback) -> decltype(callback())
{
    FfxScopedLogSuppression suppression;
    return callback();
}

#define STRINGIFY_(X)                            #X
#define STRINGIFY(X)                             STRINGIFY_(X)
#define MAKE_VERSION_STRING(major, minor, patch) STRINGIFY major "." STRINGIFY minor "." STRINGIFY patch

uint64_t ffxProvider_FrameGeneration::GetId() const
{
    // FG, version from header
    return 0xF600'0000ull << 32u |
           (FFX_SDK_MAKE_VERSION(FFX_FRAMEINTERPOLATION_VERSION_MAJOR, FFX_FRAMEINTERPOLATION_VERSION_MINOR, FFX_FRAMEINTERPOLATION_VERSION_PATCH) &
            0xFFFF'FFFF);
}

const char* ffxProvider_FrameGeneration::GetVersionName() const
{
    return MAKE_VERSION_STRING(FFX_FRAMEINTERPOLATION_VERSION_MAJOR, FFX_FRAMEINTERPOLATION_VERSION_MINOR, FFX_FRAMEINTERPOLATION_VERSION_PATCH);
}

/// If passed in optical flow is valid, then registers it. Otherwise, creates one.
static FfxErrorCode CreateOpticalFlow(InternalFgContext* internal_context, const FfxCreateResourceDescription* createResourceDesc)
{
    return internal_context->backendInterfaceShared.fpCreateResource(
        &internal_context->backendInterfaceShared,
        createResourceDesc,
        internal_context->effectContextIdShared,
        &internal_context->sharedResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_VECTOR]);
};

/// Create both depth ping-pong buffers (tm1 and tm1_next) as shared resources.
static FfxErrorCode CreateDepthTm1(InternalFgContext*                  internal_context,
                                   const FfxCreateResourceDescription* depthTm1Desc,
                                   const FfxCreateResourceDescription* depthTm1NextDesc)
{
    FfxErrorCode errorCode =
        internal_context->backendInterfaceShared.fpCreateResource(&internal_context->backendInterfaceShared,
                                                                  depthTm1Desc,
                                                                  internal_context->effectContextIdShared,
                                                                  &internal_context->sharedResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1]);
    if (errorCode != FFX_OK)
    {
        return errorCode;
    }
    return internal_context->backendInterfaceShared.fpCreateResource(
        &internal_context->backendInterfaceShared,
        depthTm1NextDesc,
        internal_context->effectContextIdShared,
        &internal_context->sharedResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1_NEXT]);
};

/// If passed in optical flow is valid, then registers it. Otherwise, creates one.
static FfxErrorCode CreateColorTm1(InternalFgContext* internal_context, const FfxCreateResourceDescription* createResourceDesc)
{
    FfxErrorCode errorCode = internal_context->backendInterfaceShared.fpCreateResource(
        &internal_context->backendInterfaceShared,
        createResourceDesc,
        internal_context->effectContextIdShared,
        &internal_context->sharedResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE]);
    if (errorCode != FFX_OK)
    {
        return errorCode;
    }
    return internal_context->backendInterfaceShared.fpCreateResource(
        &internal_context->backendInterfaceShared,
        createResourceDesc,
        internal_context->effectContextIdShared,
        &internal_context->sharedResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE_BACKUP]);
};

/// Create the full-resolution colour history used by FI postprocess. When the
/// topology selects OF-colour downsampling, the FI context owns the separate
/// OF-resolution P1/M1 ping-pong resources. Otherwise this history resource is
/// also supplied directly to optical flow.
/// @param bindAsRenderTarget When true, the colour TM1 resources are additionally created with
///                           FFX_RESOURCE_USAGE_RENDERTARGET so the fragment-shader variant of
///                           the postprocess pass can write to them directly.
static FfxErrorCode CreateColorTm1Resources(InternalFgContext* internal_context, const FfxCreateResourceDescription* ofColorTm1Desc, bool bindAsRenderTarget)
{
    if (internal_context->resolutionPlan.needsOfColourDownsample)
    {
        // Full-res colour history for FI postprocess
        FfxCreateResourceDescription fullResDesc = *ofColorTm1Desc;
        fullResDesc.resourceDescription.width    = internal_context->resolutionPlan.fullResColour.width;
        fullResDesc.resourceDescription.height   = internal_context->resolutionPlan.fullResColour.height;
        fullResDesc.name                         = "ARM_ColorTm1_FullRes";
        if (bindAsRenderTarget)
        {
            fullResDesc.resourceDescription.usage = FfxResourceUsage(FFX_RESOURCE_USAGE_RENDERTARGET);
        }
        // No separate OF-only colour TM1 needed: the FI context's internal M1
        // ping-pong buffer (populated by the downsample + swap) is used directly
        // via ffxFrameInterpolationGetDownsampledOfColourM1().
        return CreateColorTm1(internal_context, &fullResDesc);
    }
    else
    {
        FfxCreateResourceDescription desc = *ofColorTm1Desc;
        if (bindAsRenderTarget)
        {
            desc.resourceDescription.usage = FfxResourceUsage(FFX_RESOURCE_USAGE_RENDERTARGET);
        }
        return CreateColorTm1(internal_context, &desc);
    }
}

struct FgOwnership
{
    bool backendInterface   = false;
    bool backendContext     = false;
    bool opticalFlow        = false;
    bool frameInterpolation = false;
};

static void RecordTeardownFailure(const FfxErrorCode errorCode, const char* source, FfxErrorCode& firstError, const char*& firstErrorSource)
{
    if (errorCode != FFX_OK)
    {
        if (firstError == FFX_OK)
        {
            firstError       = errorCode;
            firstErrorSource = source;
        }
    }
}

static FfxErrorCode DestroyPartialFgContext(InternalFgContext* internal_context, Allocator& alloc, const FgOwnership owned, bool logLifecycle = false)
{
    if (!internal_context)
    {
        return FFX_OK;
    }

    FfxErrorCode firstError       = FFX_OK;
    const char*  firstErrorSource = "provider";

#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
    Allocator trackedAlloc = alloc.withTracker(internal_context->heapTracker);
#else
    Allocator& trackedAlloc = alloc;
#endif

    if (owned.frameInterpolation)
    {
        RecordTeardownFailure(
            callWithoutNestedLogging([&]() { return ffxFrameInterpolationContextDestroyForFrameGeneration(&internal_context->armFiContext); }),
            "frame_interpolation",
            firstError,
            firstErrorSource);
    }

    if (owned.backendContext && internal_context->backendInterfaceShared.fpDestroyResource)
    {
        for (FfxUInt32 i = 0; i < FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNT; ++i)
        {
            RecordTeardownFailure(internal_context->backendInterfaceShared.fpDestroyResource(
                                      &internal_context->backendInterfaceShared, internal_context->sharedResources[i], internal_context->effectContextIdShared),
                                  "backend",
                                  firstError,
                                  firstErrorSource);
        }
    }

    if (owned.opticalFlow)
    {
        RecordTeardownFailure(callWithoutNestedLogging([&]() { return ffxOpticalFlowContextDestroyForFrameGeneration(&internal_context->armOFContext); }),
                              "optical_flow",
                              firstError,
                              firstErrorSource);
    }

    if (owned.backendContext && internal_context->backendInterfaceShared.fpDestroyBackendContext)
    {
        RecordTeardownFailure(internal_context->backendInterfaceShared.fpDestroyBackendContext(&internal_context->backendInterfaceShared,
                                                                                               internal_context->effectContextIdShared),
                              "backend",
                              firstError,
                              firstErrorSource);
    }

    if (owned.backendInterface)
    {
        trackedAlloc.dealloc(internal_context->backendInterfaceShared.scratchBuffer, internal_context->backendInterfaceShared.scratchBufferSize);
        internal_context->backendInterfaceShared.scratchBuffer = nullptr;
        internal_context->backendInterfaceFi.scratchBuffer     = nullptr;
    }

#if FFX_CAPTURE_ENABLED
    if (internal_context->controller)
    {
        arm::capture::DestroyCaptureController(trackedAlloc, internal_context->controller);
    }
#endif
#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
    internal_context->heapTracker.Remove(sizeof(InternalFgContext));
    internal_context->heapTracker.~EffectHeapTracker();
#endif
    if (logLifecycle || firstError != FFX_OK)
    {
        internal_context->diagnostics.LogContextDestroyed(firstError, firstErrorSource);
    }
    alloc.dealloc(internal_context, sizeof(InternalFgContext));
    return firstError;
}

static ffxReturnCode_t CreateFgContextResources(InternalFgContext*                            internal_context,
                                                const ffxApiCreateContextDescFrameGeneration* desc,
                                                const FfxDimensions2D                         mvDepthSize,
                                                ffxCreateContextDescHeader*                   header,
                                                Allocator&                                    trackedAlloc,
                                                FgOwnership&                                  owned)
{
    // Keep shared resources, optical flow, and frame interpolation in one backend so
    // first-use ownership can be preserved across their effect contexts.
    TRY(MustCreateBackend(header, &internal_context->backendInterfaceShared, 3, trackedAlloc));
    owned.backendInterface               = true;
    internal_context->backendInterfaceFi = internal_context->backendInterfaceShared;
    internal_context->flags              = desc->flags;
    internal_context->resolutionPlan =
        ffxNfruDeriveResolutionPlan({desc->displaySize.width, desc->displaySize.height}, mvDepthSize, FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT);
    const bool managesPreviousDepth            = (desc->flags & FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH) != 0;
    internal_context->usesInternalDepthHistory = managesPreviousDepth || internal_context->resolutionPlan.needsAllInputDownsample;

    const FfxErrorCode backendContextResult = internal_context->backendInterfaceShared.fpCreateBackendContext(
        &internal_context->backendInterfaceShared, FFX_EFFECT_SHAREDAPIBACKEND, nullptr, &internal_context->effectContextIdShared);
    if (backendContextResult != FFX_OK)
    {
        return internal_context->diagnostics.LogContextCreationError("backend",
                                                                     "backend_context",
                                                                     backendContextResult,
                                                                     "backend_context_create_failed",
                                                                     "failed to create the shared backend context",
                                                                     FFX_API_RETURN_ERROR_RUNTIME_ERROR);
    }
    owned.backendContext = true;

    FfxDeviceCapabilities capabilities{};
    const FfxErrorCode    capabilitiesResult =
        internal_context->backendInterfaceShared.fpGetDeviceCapabilities(&internal_context->backendInterfaceShared, &capabilities);
    if (capabilitiesResult != FFX_OK)
    {
        return internal_context->diagnostics.LogContextCreationError("backend",
                                                                     "device_capabilities",
                                                                     capabilitiesResult,
                                                                     "device_capabilities_query_failed",
                                                                     "failed to query device capabilities",
                                                                     FFX_API_RETURN_ERROR_RUNTIME_ERROR);
    }
    internal_context->fp16Supported = capabilities.fp16Supported;
    if (!capabilities.dataGraphOFSupported)
    {
        return internal_context->diagnostics.LogContextCreationError("device_capabilities",
                                                                     "optical_flow",
                                                                     FFX_ERROR_NULL_DEVICE,
                                                                     "missing_data_graph_optical_flow",
                                                                     "the device does not support data graph optical flow",
                                                                     FFX_API_RETURN_ERROR_RUNTIME_ERROR,
                                                                     NfruCapabilityFailure::MissingDataGraphOpticalFlow);
    }
    if (!capabilities.dataGraphSupported || !capabilities.tensorSupported)
    {
        return internal_context->diagnostics.LogContextCreationError("device_capabilities",
                                                                     "frame_interpolation",
                                                                     FFX_ERROR_NULL_DEVICE,
                                                                     "frame_interpolation_pipeline_unsupported",
                                                                     "the device does not support the NFRU frame interpolation data graph",
                                                                     FFX_API_RETURN_ERROR_RUNTIME_ERROR,
                                                                     NfruCapabilityFailure::FrameInterpolationPipelineUnsupported);
    }

    ffxProviderLogMessage(desc->fpMessage, FFX_API_MESSAGE_TYPE_INFO, "Request to use data graph optical flow.");

    const FfxOpticalFlowGridSize ofGridSize = SelectDefaultOpticalFlowGrid(capabilities.supportedOutputGridSizes);
    if (!ffxNfruSupportsOpticalFlowGrid(ofGridSize))
    {
        return internal_context->diagnostics.LogContextCreationError("device_capabilities",
                                                                     "optical_flow",
                                                                     FFX_ERROR_NULL_DEVICE,
                                                                     "unsupported_optical_flow_grid",
                                                                     "the device does not expose an NFRU-compatible optical-flow grid",
                                                                     FFX_API_RETURN_ERROR_PARAMETER,
                                                                     NfruCapabilityFailure::UnsupportedOpticalFlowGrid);
    }
    const FfxDimensions2D ofInputColour = internal_context->resolutionPlan.ofInputColour;
    if (ofInputColour.width < capabilities.dataGraphOFMinWidth || ofInputColour.height < capabilities.dataGraphOFMinHeight ||
        ofInputColour.width > capabilities.dataGraphOFMaxWidth || ofInputColour.height > capabilities.dataGraphOFMaxHeight)
    {
        return internal_context->diagnostics.LogContextCreationError("device_capabilities",
                                                                     "optical_flow",
                                                                     FFX_ERROR_NULL_DEVICE,
                                                                     "optical_flow_resolution_out_of_range",
                                                                     "the NFRU optical-flow input resolution is outside the device limits",
                                                                     FFX_API_RETURN_ERROR_RUNTIME_ERROR,
                                                                     NfruCapabilityFailure::OpticalFlowResolutionOutOfRange);
    }

    FfxOpticalFlowContextDescription ofDescription = {};
    ofDescription.backendInterface                 = internal_context->backendInterfaceFi;
    ofDescription.resolution                       = internal_context->resolutionPlan.ofInputColour;
    ofDescription.maxRenderSize.width              = desc->renderSize.width;
    ofDescription.maxRenderSize.height             = desc->renderSize.height;
    ofDescription.mvDepthSize                      = internal_context->resolutionPlan.mvDepthLane;
    memcpy(ofDescription.initialViewProjection, desc->initialViewProjection, sizeof(ofDescription.initialViewProjection));
    ofDescription.gridSize         = ofGridSize;
    ofDescription.performanceLevel = FFX_OPTICAL_FLOW_PERFORMANCE_LEVEL_MEDIUM;
    ofDescription.backBufferFormat = ConvertEnum<FfxSurfaceFormat>(desc->backBufferFormat);

    if (desc->flags & FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED)
    {
        ofDescription.flags |= FFX_OPTICALFLOW_ENABLE_DEPTH_INVERTED;
    }

    if (desc->flags & FFX_API_FG_CONTEXT_FLAG_ENABLE_MV_HINTS)
    {
        ofDescription.flags |= FFX_OPTICALFLOW_ENABLE_MV_HINTS;
        if (desc->flags & FFX_API_FG_CONTEXT_FLAG_MV_HINTS_COMPUTE)
        {
            ofDescription.flags |= FFX_OPTICALFLOW_ENABLE_MV_HINTS_COMPUTE;
        }
    }

    const FfxErrorCode ofCreateResult = callWithoutNestedLogging([&]() {
        return ffxOpticalFlowContextCreateForFrameGenerationWithDepthInput(
            &internal_context->armOFContext, &ofDescription, internal_context->resolutionPlan.mvDepthInput);
    });
    if (ofCreateResult != FFX_OK)
    {
        return internal_context->diagnostics.LogContextCreationError("optical_flow",
                                                                     "context_create",
                                                                     ofCreateResult,
                                                                     "optical_flow_context_create_failed",
                                                                     "failed to create the data graph optical flow context",
                                                                     FFX_API_RETURN_ERROR_RUNTIME_ERROR);
    }
    owned.opticalFlow = true;

    const FfxDimensions2D opticalFlowSize         = ffxOpticalFlowGetSize(&internal_context->armOFContext);
    internal_context->opticalFlowSize             = opticalFlowSize;
    const FfxDimensions2D expectedOpticalFlowSize = ffxNfruGetOpticalFlowOutputSize(internal_context->resolutionPlan.ofInputColour, ofGridSize);
    if (opticalFlowSize.width != expectedOpticalFlowSize.width || opticalFlowSize.height != expectedOpticalFlowSize.height)
    {
        return internal_context->diagnostics.LogContextCreationError("optical_flow",
                                                                     "resolution",
                                                                     FFX_ERROR_INVALID_ARGUMENT,
                                                                     "optical_flow_size_mismatch",
                                                                     "optical flow dimensions do not match the NFRU resolution plan",
                                                                     FFX_API_RETURN_ERROR_PARAMETER);
    }

    FfxOpticalFlowSharedResourceDescriptions ofResourceDescs          = {};
    const FfxErrorCode                       sharedDescriptionsResult = callWithoutNestedLogging(
        [&]() { return ffxOpticalFlowGetSharedResourceDescriptionsForFrameGeneration(&internal_context->armOFContext, &ofResourceDescs); });
    if (sharedDescriptionsResult != FFX_OK)
    {
        return internal_context->diagnostics.LogContextCreationError("optical_flow",
                                                                     "shared_resources",
                                                                     sharedDescriptionsResult,
                                                                     "optical_flow_shared_resources_failed",
                                                                     "failed to query optical flow shared resource descriptions",
                                                                     FFX_API_RETURN_ERROR_RUNTIME_ERROR);
    }
    if (ofResourceDescs.opticalFlowVector.resourceDescription.width != expectedOpticalFlowSize.width ||
        ofResourceDescs.opticalFlowVector.resourceDescription.height != expectedOpticalFlowSize.height)
    {
        return internal_context->diagnostics.LogContextCreationError("optical_flow",
                                                                     "shared_resources",
                                                                     FFX_ERROR_INVALID_ARGUMENT,
                                                                     "optical_flow_resource_size_mismatch",
                                                                     "optical flow resource dimensions do not match the NFRU resolution plan",
                                                                     FFX_API_RETURN_ERROR_PARAMETER);
    }

    const FfxErrorCode opticalFlowResourceResult = CreateOpticalFlow(internal_context, &ofResourceDescs.opticalFlowVector);
    if (opticalFlowResourceResult != FFX_OK)
    {
        return internal_context->diagnostics.LogContextCreationError("backend",
                                                                     "shared_resources",
                                                                     opticalFlowResourceResult,
                                                                     "optical_flow_resource_create_failed",
                                                                     "failed to create the shared optical flow resource",
                                                                     FFX_API_RETURN_ERROR_RUNTIME_ERROR);
    }
    const bool sdkManagesPrevColor = (desc->flags & FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR) != 0;
    {
        const FfxDimensions2D  depthHistory                     = internal_context->resolutionPlan.mvDepthLane;
        const bool             warpUsesFragment                 = (desc->flags & FFX_API_FG_CONTEXT_FLAG_WARP_COMPUTE) == 0;
        const FfxResourceUsage depthUsage                       = ffxNfruGetDepthHistoryUsage(internal_context->resolutionPlan, warpUsesFragment);
        ofResourceDescs.depthTm1.resourceDescription.width      = depthHistory.width;
        ofResourceDescs.depthTm1.resourceDescription.height     = depthHistory.height;
        ofResourceDescs.depthTm1.resourceDescription.usage      = depthUsage;
        ofResourceDescs.depthTm1Next.resourceDescription.width  = depthHistory.width;
        ofResourceDescs.depthTm1Next.resourceDescription.height = depthHistory.height;
        ofResourceDescs.depthTm1Next.resourceDescription.usage  = depthUsage;
        assert(ofResourceDescs.depthTm1.resourceDescription.width == internal_context->resolutionPlan.mvDepthLane.width);
        assert(ofResourceDescs.depthTm1.resourceDescription.height == internal_context->resolutionPlan.mvDepthLane.height);
    }

    if (internal_context->usesInternalDepthHistory)
    {
        const FfxErrorCode depthHistoryResult = CreateDepthTm1(internal_context, &ofResourceDescs.depthTm1, &ofResourceDescs.depthTm1Next);
        if (depthHistoryResult != FFX_OK)
        {
            return internal_context->diagnostics.LogContextCreationError("backend",
                                                                         "shared_resources",
                                                                         depthHistoryResult,
                                                                         "depth_history_create_failed",
                                                                         "failed to create managed depth history resources",
                                                                         FFX_API_RETURN_ERROR_RUNTIME_ERROR);
        }
    }

    if (sdkManagesPrevColor)
    {
        const bool         bindColorTm1AsRenderTarget = (desc->flags & FFX_API_FG_CONTEXT_FLAG_POSTPROCESS_COMPUTE) == 0;
        const FfxErrorCode colorHistoryResult         = CreateColorTm1Resources(internal_context, &ofResourceDescs.colorTm1, bindColorTm1AsRenderTarget);
        if (colorHistoryResult != FFX_OK)
        {
            return internal_context->diagnostics.LogContextCreationError("backend",
                                                                         "shared_resources",
                                                                         colorHistoryResult,
                                                                         "color_history_create_failed",
                                                                         "failed to create managed color history resources",
                                                                         FFX_API_RETURN_ERROR_RUNTIME_ERROR);
        }
    }

    FfxFrameInterpolationContextDescription fiDescription = {};
    fiDescription.backendInterface                        = internal_context->backendInterfaceFi;
    fiDescription.flags                                   = ConvertContextFlagsFI(desc->flags);
    fiDescription.maxRenderSize.width                     = desc->renderSize.width;
    fiDescription.maxRenderSize.height                    = desc->renderSize.height;
    fiDescription.mvDepthSize                             = mvDepthSize;
    fiDescription.displaySize.width                       = desc->displaySize.width;
    fiDescription.displaySize.height                      = desc->displaySize.height;
    fiDescription.backBufferFormat                        = ConvertEnum<FfxSurfaceFormat>(desc->backBufferFormat);
    fiDescription.opticalFlowSize                         = opticalFlowSize;
    fiDescription.opticalFlowGridSize                     = ofGridSize;
    fiDescription.fpMessage                               = reinterpret_cast<FfxFiMessage>(desc->fpMessage);
    memcpy(fiDescription.initialViewProjection, desc->initialViewProjection, sizeof(fiDescription.initialViewProjection));

    const FfxErrorCode frameInterpolationResult =
        callWithoutNestedLogging([&]() { return ffxFrameInterpolationContextCreateForFrameGeneration(&internal_context->armFiContext, &fiDescription); });
    if (frameInterpolationResult != FFX_OK)
    {
        return internal_context->diagnostics.LogContextCreationError("frame_interpolation",
                                                                     "context_create",
                                                                     frameInterpolationResult,
                                                                     "frame_interpolation_context_create_failed",
                                                                     "failed to create the NFRU frame interpolation context",
                                                                     FFX_API_RETURN_ERROR_RUNTIME_ERROR);
    }
    owned.frameInterpolation = true;
    internal_context->diagnostics.LogContextCreated(*desc, GetLogContextInfo(internal_context), ofGridSize);
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t ffxProvider_FrameGeneration::CreateContext(ffxContext* context, ffxCreateContextDescHeader* header, Allocator& alloc) const
{
    if (auto desc = ffx::DynamicCast<ffxApiCreateContextDescFrameGeneration>(header))
    {
        const FfxNfruResolutionValidationError resolutionError = ffxNfruValidateResolutionInputs({desc->displaySize.width, desc->displaySize.height},
                                                                                                 {desc->renderSize.width, desc->renderSize.height},
                                                                                                 {desc->mvDepthSize.width, desc->mvDepthSize.height});
        if (resolutionError != FfxNfruResolutionValidationError::None)
        {
            NfruDiagnostics::LogResolutionValidationError(desc->fpMessage, resolutionError);
            return FFX_API_RETURN_ERROR_PARAMETER;
        }

        const FfxDimensions2D mvDepthSize =
            ffxNfruResolveMvDepthSize({desc->renderSize.width, desc->renderSize.height}, {desc->mvDepthSize.width, desc->mvDepthSize.height});

        // Reject unsupported backbuffer formats before allocating anything, so the early return
        // doesn't leak the context/backends created below.
        if (!IsBackBufferFormatSupportedByOpticalFlow(desc->backBufferFormat))
        {
            NfruDiagnostics::LogUnsupportedBackBufferFormat(desc->fpMessage);
            return FFX_API_RETURN_ERROR_PARAMETER;
        }

        NfruDiagnostics::LogVersions(desc->fpMessage);

        InternalFgContext* internal_context = alloc.construct<InternalFgContext>();
        VERIFY(internal_context, FFX_API_RETURN_ERROR_MEMORY);
        // Allocator::construct<T>() allocates raw storage, so initialize the storage and start
        memset(internal_context, 0, sizeof(*internal_context));
#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
        ::new (&internal_context->heapTracker) ffx::EffectHeapTracker(sizeof(InternalFgContext));
        Allocator trackedAlloc = alloc.withTracker(internal_context->heapTracker);
#else
        Allocator& trackedAlloc = alloc;
#endif
        internal_context->header.provider = this;
        internal_context->diagnostics     = NfruDiagnostics::Create(desc->fpMessage);

        FgOwnership           owned{};
        const ffxReturnCode_t resourceResult = CreateFgContextResources(internal_context, desc, mvDepthSize, header, trackedAlloc, owned);
        if (resourceResult != FFX_API_RETURN_OK)
        {
            if (!internal_context->diagnostics.HasError())
            {
                (void)internal_context->diagnostics.LogContextCreationError("provider",
                                                                            "context_resources",
                                                                            FFX_ERROR_BACKEND_API_ERROR,
                                                                            "context_resource_creation_failed",
                                                                            "failed to create NFRU context resources",
                                                                            resourceResult);
            }
            (void)DestroyPartialFgContext(internal_context, alloc, owned);
            return resourceResult;
        }

#if FFX_CAPTURE_ENABLED
        internal_context->controller = arm::capture::MakeCaptureController<CaptureController>(
            trackedAlloc,
            arm::capture::framegenerations::CreateDesc{internal_context->backendInterfaceShared.device, desc->flags},
            arm::capture::framegenerations::Context{
                &internal_context->backendInterfaceShared, internal_context->sharedResources, &internal_context->armFiContext});
        if (!internal_context->controller)
        {
            DestroyPartialFgContext(internal_context, alloc, owned);
            return FFX_API_RETURN_ERROR_MEMORY;
        }
#endif

        *context = internal_context;
        return FFX_API_RETURN_OK;
    }
    else
    {
        ffxLogMessage(ffxLogComponentName(), FFX_MESSAGE_TYPE_ERROR, "unsupported frame-generation create descriptor");
        return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }
}

ffxReturnCode_t ffxProvider_FrameGeneration::DestroyContext(ffxContext* context, Allocator& alloc) const
{
    VERIFY(context, FFX_API_RETURN_ERROR_PARAMETER);
    VERIFY(*context, FFX_API_RETURN_ERROR_PARAMETER);

    InternalFgContext* internal_context = reinterpret_cast<InternalFgContext*>(*context);
#if FFX_CAPTURE_ENABLED
    if (internal_context->controller)
    {
        if (!internal_context->controller->Shutdown())
            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
#endif

    FgOwnership owned{};
    owned.backendInterface   = true;
    owned.backendContext     = true;
    owned.opticalFlow        = true;
    owned.frameInterpolation = true;
    if (internal_context->runtimeState.enabled)
    {
        internal_context->diagnostics.LogFrameGenerationDisabledOnDestroy();
    }
    const FfxErrorCode destroyResult = DestroyPartialFgContext(internal_context, alloc, owned, true);
    *context                         = nullptr;
    return destroyResult == FFX_OK ? FFX_API_RETURN_OK : FFX_API_RETURN_ERROR_RUNTIME_ERROR;
}

ffxReturnCode_t ffxProvider_FrameGeneration::Configure(ffxContext* context, const ffxConfigureDescHeader* header) const
{
    VERIFY(context, FFX_API_RETURN_ERROR_PARAMETER);
    VERIFY(*context, FFX_API_RETURN_ERROR_PARAMETER);
    VERIFY(header, FFX_API_RETURN_ERROR_PARAMETER);

    InternalFgContext* internal_context = reinterpret_cast<InternalFgContext*>(*context);
    if (auto captureDesc = ffx::DynamicCast<ffxApiConfigureDescCapture>(header))
    {
#if !FFX_CAPTURE_ENABLED
        (void)captureDesc;
        return FFX_API_RETURN_ERROR_PARAMETER;
#else
        if (!internal_context->controller)
            return FFX_API_RETURN_ERROR_PARAMETER;
        return internal_context->controller->Configure(*captureDesc);
#endif
    }
    else if (auto desc = ffx::DynamicCast<ffxApiConfigureDescFrameGeneration>(header))
    {
        FfxFrameGenerationConfig config{};
        config.allowAsyncWorkloads = desc->allowAsyncWorkloads;
        config.flags               = desc->flags & ~FFX_API_FG_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING;

        size_t callbacksIndex = desc->frameID % MAX_QUEUED_FRAMES;

        const InternalFgContext::Callbacks requestedCallbacks{
            desc->presentCallback, desc->presentCallbackUserContext, desc->frameGenerationCallback, desc->frameGenerationCallbackUserContext};
        const InternalFgContext::Callbacks& currentCallbacks = internal_context->callbacks[callbacksIndex];
        const bool                          bPresentCallbackChanged =
            currentCallbacks.presentCallback != requestedCallbacks.presentCallback ||
            (requestedCallbacks.presentCallback && currentCallbacks.presentCallbackUserContext != requestedCallbacks.presentCallbackUserContext);
        const bool bFrameGenerationCallbackChanged = currentCallbacks.frameGenerationCallback != requestedCallbacks.frameGenerationCallback ||
                                                     (requestedCallbacks.frameGenerationCallback && currentCallbacks.frameGenerationCallbackUserContext !=
                                                                                                        requestedCallbacks.frameGenerationCallbackUserContext);

        config.frameGenerationCallback        = nullptr;
        config.frameGenerationCallbackContext = nullptr;
        if (desc->frameGenerationCallback != nullptr)
        {
            config.frameGenerationCallback = [](const FfxFrameGenerationDispatchDescription* desc, void* ctx) -> FfxErrorCode {
                size_t             callbacksIndex   = desc->frameID % MAX_QUEUED_FRAMES;
                InternalFgContext* internal_context = reinterpret_cast<InternalFgContext*>(ctx);
                auto               callbacks        = &internal_context->callbacks[callbacksIndex];
                if (!callbacks->frameGenerationCallback)
                    return ffxLogError(FFX_ERROR_BACKEND_API_ERROR, "no frame-generation callback is registered for the requested frame");

                ffx::DispatchDescFrameGeneration dispatchDesc{};

                dispatchDesc.commandList        = desc->commandList;
                dispatchDesc.numGeneratedFrames = desc->numInterpolatedFrames;
                dispatchDesc.outputs[0]         = Convert(desc->outputs[0]);
                dispatchDesc.outputs[1]         = Convert(desc->outputs[1]);
                dispatchDesc.outputs[2]         = Convert(desc->outputs[2]);
                dispatchDesc.outputs[3]         = Convert(desc->outputs[3]);
                dispatchDesc.presentColor       = Convert(desc->presentColor);
                dispatchDesc.reset              = desc->reset;
                dispatchDesc.frameID            = desc->frameID;

                if (FFX_API_RETURN_OK != callbacks->frameGenerationCallback(&dispatchDesc, callbacks->frameGenerationCallbackUserContext))
                    return ffxLogError(FFX_ERROR_BACKEND_API_ERROR, "the registered frame-generation callback returned an error");
                return FFX_OK;
            };
            config.frameGenerationCallbackContext = internal_context;
        }

        config.presentCallback        = nullptr;
        config.presentCallbackContext = nullptr;
        if (desc->presentCallback != nullptr)
        {
            config.presentCallback = [](const FfxPresentCallbackDescription* params, void* ctx) -> FfxErrorCode {
                size_t             callbacksIndex   = params->frameID % MAX_QUEUED_FRAMES;
                InternalFgContext* internal_context = reinterpret_cast<InternalFgContext*>(ctx);
                auto               callbacks        = &internal_context->callbacks[callbacksIndex];
                if (!callbacks->presentCallback)
                    return ffxLogError(FFX_ERROR_BACKEND_API_ERROR, "no present callback is registered for the requested frame");

                ffxApiCallbackDescFrameGenerationPresent cbDesc{};
                cbDesc.header.pNext = nullptr;
                cbDesc.header.type  = FFX_API_CALLBACK_DESC_TYPE_FRAMEGENERATION_PRESENT;

                cbDesc.commandList           = params->commandList;
                cbDesc.currentBackBuffer     = Convert(params->currentBackBuffer);
                cbDesc.currentUI             = Convert(params->currentUI);
                cbDesc.device                = params->device;
                cbDesc.isGeneratedFrame      = params->isInterpolatedFrame;
                cbDesc.outputSwapChainBuffer = Convert(params->outputSwapChainBuffer);
                cbDesc.frameID               = params->frameID;

                if (FFX_API_RETURN_OK != callbacks->presentCallback(&cbDesc, callbacks->presentCallbackUserContext))
                    return ffxLogError(FFX_ERROR_BACKEND_API_ERROR, "the registered present callback returned an error");
                return FFX_OK;
            };
            config.presentCallbackContext = internal_context;
        }

        config.frameGenerationEnabled  = desc->frameGenerationEnabled;
        config.onlyPresentInterpolated = desc->onlyPresentGenerated;
        config.swapChain               = desc->swapChain;

        config.frameID = desc->frameID;

        if (desc->flags & FFX_API_FG_DISPATCH_FLAG_DRAW_DEBUG_VIEW)
        {
            config.onlyPresentInterpolated = true;
        }

        if (desc->flags & FFX_API_FG_DISPATCH_FLAG_DUMP_GENERATED_FRAME)
        {
            config.dumpGeneratedFrame     = true;
            config.dumpGeneratedFramePath = desc->dumpGeneratedFramePath;
        }

        const bool verbose = (desc->flags & FFX_API_FG_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING) != 0;
        if (!(desc->flags & FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY))
        {
            // When the frame ID is not incrementing by 1 we could end up overwriting a pointer that is in-use, so reset the swap-chain state
            if (((internal_context->lastConfigureFrameID + 1) != desc->frameID) && (bPresentCallbackChanged || bFrameGenerationCallbackChanged))
            {
                FfxFrameGenerationConfig resetConfig       = config;
                resetConfig.frameGenerationCallback        = nullptr;
                resetConfig.frameGenerationCallbackContext = nullptr;
                resetConfig.presentCallback                = nullptr;
                resetConfig.presentCallbackContext         = nullptr;

                const FfxErrorCode resetResult = internal_context->backendInterfaceShared.fpSwapChainConfigureFrameGeneration(&resetConfig);
                if (resetResult != FFX_OK)
                {
                    return internal_context->diagnostics.LogRuntimeError(desc->frameID,
                                                                         "configure",
                                                                         "backend",
                                                                         "swapchain_configure",
                                                                         resetResult,
                                                                         "swapchain_reset_failed",
                                                                         "failed to reset frame-generation callbacks",
                                                                         verbose);
                }
            }

            const FfxErrorCode configureResult = internal_context->backendInterfaceShared.fpSwapChainConfigureFrameGeneration(&config);
            if (configureResult != FFX_OK)
            {
                return internal_context->diagnostics.LogRuntimeError(desc->frameID,
                                                                     "configure",
                                                                     "backend",
                                                                     "swapchain_configure",
                                                                     configureResult,
                                                                     "swapchain_configure_failed",
                                                                     "failed to configure frame generation",
                                                                     verbose);
            }

            internal_context->lastConfigureFrameID = desc->frameID;
        }

        internal_context->callbacks[callbacksIndex] = requestedCallbacks;
        const bool stateChanged                     = internal_context->runtimeState.enabled != desc->frameGenerationEnabled;
        internal_context->frameGenFlags             = desc->flags;
        internal_context->frameGenEnabled           = desc->frameGenerationEnabled;
        internal_context->dumpGeneratedFrame        = config.dumpGeneratedFrame;
        internal_context->dumpGeneratedFramePath    = config.dumpGeneratedFramePath;
        internal_context->runtimeState.enabled      = desc->frameGenerationEnabled;
        internal_context->diagnostics.ClearError();
        if (stateChanged)
        {
            internal_context->diagnostics.LogFrameGenerationState(*desc, desc->frameGenerationEnabled);
        }

        return FFX_API_RETURN_OK;
    }
    else
    {
        ffxLogMessage(ffxLogComponentName(), FFX_MESSAGE_TYPE_ERROR, "unsupported frame-generation configure descriptor");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
}

static void AddEffectMemoryUsage(FfxApiEffectMemoryUsage& destination, const FfxEffectMemoryUsage& source)
{
    destination.totalUsageInBytes += source.totalUsageInBytes;
    destination.textureUsageInBytes += source.textureUsageInBytes;
    destination.bufferUsageInBytes += source.bufferUsageInBytes;
    destination.tensorUsageInBytes += source.tensorUsageInBytes;
    destination.dataGraphUsageInBytes += source.dataGraphUsageInBytes;
    destination.aliasableUsageInBytes += source.aliasableUsageInBytes;
}

ffxReturnCode_t ffxProvider_FrameGeneration::Query(ffxContext* context, ffxQueryDescHeader* header) const
{
    VERIFY(header, FFX_API_RETURN_ERROR_PARAMETER);

    InternalFgContext* internal_context = context && *context ? reinterpret_cast<InternalFgContext*>(*context) : nullptr;
    switch (header->type)
    {
    case FFX_API_QUERY_DESC_TYPE_CAPTURESTATUS:
    {
        VERIFY(context, FFX_API_RETURN_ERROR_PARAMETER);
        VERIFY(*context, FFX_API_RETURN_ERROR_PARAMETER);
        auto* desc = reinterpret_cast<ffxApiQueryDescCaptureStatus*>(header);
        VERIFY(desc->pOutStatus, FFX_API_RETURN_ERROR_PARAMETER);
#if !FFX_CAPTURE_ENABLED
        return FFX_API_RETURN_ERROR_PARAMETER;
#else
        internal_context->controller->GetStatus(*desc->pOutStatus);
        break;
#endif
    }
    case FFX_API_QUERY_DESC_TYPE_FG_GPU_MEMORY_USAGE:
    {
        VERIFY(context, FFX_API_RETURN_ERROR_PARAMETER);
        VERIFY(*context, FFX_API_RETURN_ERROR_PARAMETER);

        auto desc = reinterpret_cast<ffxApiQueryDescFrameGenerationGetGPUMemoryUsage*>(header);
        VERIFY(desc->gpuMemoryUsageFrameGeneration, FFX_API_RETURN_ERROR_PARAMETER);

        FfxEffectMemoryUsage opticalFlowUsage{};
        FfxEffectMemoryUsage frameInterpolationUsage{};
        FfxEffectMemoryUsage sharedUsage{};
        FfxEffectMemoryUsage backendUsage{};

        TRY2(ffxOpticalFlowContextGetGpuMemoryUsage(&internal_context->armOFContext, &opticalFlowUsage));
        TRY2(ffxFrameInterpolationContextGetGpuMemoryUsage(&internal_context->armFiContext, &frameInterpolationUsage));
        TRY2(internal_context->backendInterfaceShared.fpGetEffectGpuMemoryUsage(
            &internal_context->backendInterfaceShared, internal_context->effectContextIdShared, &sharedUsage));
        TRY2(internal_context->backendInterfaceShared.fpGetBackendGpuMemoryUsage(&internal_context->backendInterfaceShared, &backendUsage));

        *desc->gpuMemoryUsageFrameGeneration = {};
        AddEffectMemoryUsage(*desc->gpuMemoryUsageFrameGeneration, opticalFlowUsage);
        AddEffectMemoryUsage(*desc->gpuMemoryUsageFrameGeneration, frameInterpolationUsage);
        AddEffectMemoryUsage(*desc->gpuMemoryUsageFrameGeneration, sharedUsage);
        AddEffectMemoryUsage(*desc->gpuMemoryUsageFrameGeneration, backendUsage);
        break;
    }
#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
    case FFX_API_QUERY_DESC_TYPE_FG_CPU_MEMORY_USAGE:
    {
        VERIFY(context, FFX_API_RETURN_ERROR_PARAMETER);
        VERIFY(*context, FFX_API_RETURN_ERROR_PARAMETER);

        auto desc = reinterpret_cast<ffxApiQueryDescFrameGenerationGetCPUMemoryUsage*>(header);
        VERIFY(desc->cpuMemoryUsageFrameGeneration, FFX_API_RETURN_ERROR_PARAMETER);

        FfxApiEffectCpuMemoryUsage& usage = *desc->cpuMemoryUsageFrameGeneration;
        usage.currentUsageInBytes         = internal_context->heapTracker.GetCurrentUsageInBytes();
        usage.peakUsageInBytes            = internal_context->heapTracker.GetPeakUsageInBytes();
        break;
    }
#endif
    default:
        ffxLogMessage(ffxLogComponentName(), FFX_MESSAGE_TYPE_ERROR, "unsupported frame-generation query descriptor");
        return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }
    return FFX_API_RETURN_OK;
}

static uint64_t getTM1Index(const uint64_t framIndex)
{
    return 2 * framIndex - 2;
}

ffxReturnCode_t ffxProvider_FrameGeneration::Dispatch(ffxContext* context, const ffxDispatchDescHeader* header) const
{
    VERIFY(context, FFX_API_RETURN_ERROR_PARAMETER);
    VERIFY(*context, FFX_API_RETURN_ERROR_PARAMETER);

    InternalFgContext* internal_context    = reinterpret_cast<InternalFgContext*>(*context);
    const bool         sdkManagesPrevColor = (internal_context->flags & FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR) != 0;
    if (auto desc = ffx::DynamicCast<ffxApiDispatchDescFrameGeneration>(header))
    {
        const bool verbose = (internal_context->frameGenFlags & FFX_API_FG_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING) != 0;
        if (internal_context->runtimeState.requiresRecreation)
        {
            return internal_context->diagnostics.LogRuntimeError(desc->frameID,
                                                                 "dispatch",
                                                                 "provider",
                                                                 "context_state",
                                                                 FFX_ERROR_BACKEND_API_ERROR,
                                                                 "context_requires_recreation",
                                                                 "context requires recreation after a backend dispatch failure",
                                                                 verbose);
        }

        PreparedFrame& preparedFrame = internal_context->preparedFrames[desc->frameID % MAX_QUEUED_FRAMES];
        if (!preparedFrame.valid || preparedFrame.description.frameID != desc->frameID)
        {
            return internal_context->diagnostics.LogRuntimeError(desc->frameID,
                                                                 "dispatch",
                                                                 "provider",
                                                                 "prepare_validation",
                                                                 FFX_ERROR_INVALID_ARGUMENT,
                                                                 "prepare_dispatch_frame_mismatch",
                                                                 "dispatch requires a successful prepare with the same frame ID",
                                                                 verbose,
                                                                 FFX_API_RETURN_ERROR_PARAMETER);
        }
        const ffxApiDispatchDescFrameGenerationPrepare* prepDesc = &preparedFrame.description;

        if (!desc->commandList)
        {
            return internal_context->diagnostics.LogRuntimeError(desc->frameID,
                                                                 "dispatch",
                                                                 "provider",
                                                                 "dispatch_validation",
                                                                 FFX_ERROR_INVALID_POINTER,
                                                                 "missing_command_list",
                                                                 "dispatch requires a command list",
                                                                 verbose,
                                                                 FFX_API_RETURN_ERROR_PARAMETER);
        }
        if (!desc->presentColor.resource || desc->presentColor.description.type != FFX_API_RESOURCE_TYPE_TEXTURE2D)
        {
            return internal_context->diagnostics.LogRuntimeError(desc->frameID,
                                                                 "dispatch",
                                                                 "provider",
                                                                 "dispatch_validation",
                                                                 FFX_ERROR_INVALID_ARGUMENT,
                                                                 "invalid_present_color",
                                                                 "dispatch requires a valid 2D present-color resource",
                                                                 verbose,
                                                                 FFX_API_RETURN_ERROR_PARAMETER);
        }
        if (desc->presentColor.description.width != internal_context->resolutionPlan.fullResColour.width ||
            desc->presentColor.description.height != internal_context->resolutionPlan.fullResColour.height)
        {
            return internal_context->diagnostics.LogRuntimeError(desc->frameID,
                                                                 "dispatch",
                                                                 "provider",
                                                                 "dispatch_validation",
                                                                 FFX_ERROR_INVALID_ARGUMENT,
                                                                 "present_color_dimensions_mismatch",
                                                                 "present-color dimensions must match the configured display size",
                                                                 verbose,
                                                                 FFX_API_RETURN_ERROR_PARAMETER);
        }

        const auto& presentColorDescription = desc->presentColor.description;
        const auto& outputDescription       = desc->outputs[0].description;
        if (desc->numGeneratedFrames > 0 &&
            (presentColorDescription.width != outputDescription.width || presentColorDescription.height != outputDescription.height))
        {
            internal_context->diagnostics.LogFrameSizeMismatch(
                desc->frameID, presentColorDescription.width, presentColorDescription.height, outputDescription.width, outputDescription.height);
        }

        const bool  firstDispatch        = !internal_context->runtimeState.hasDispatched;
        const bool  frameIdDiscontinuity = !firstDispatch && (desc->frameID < internal_context->runtimeState.previousFrameId ||
                                                             desc->frameID - internal_context->runtimeState.previousFrameId > 1);
        const bool  effectiveReset       = firstDispatch || desc->reset || frameIdDiscontinuity;
        const char* resetReason = firstDispatch ? "first_dispatch" : (desc->reset ? "explicit" : (frameIdDiscontinuity ? "frame_id_discontinuity" : "none"));
        if (verbose)
        {
            internal_context->diagnostics.LogDispatchBegin(*desc, GetLogContextInfo(internal_context), effectiveReset, resetReason);
        }

        NfruCpuClock::time_point dispatchStart{};
        if (verbose)
        {
            dispatchStart = NfruCpuClock::now();
        }
        double cpuOfColourUs    = 0.0;
        double cpuOpticalFlowUs = 0.0;
        double cpuFiDispatchUs  = 0.0;

#if FFX_CAPTURE_ENABLED
        const bool captureActive     = internal_context->controller->BeginDispatchCapture(*desc, *prepDesc, effectiveReset);
        const bool forceHistoryReset = captureActive && internal_context->controller->ShouldResetHistoryForDispatch();
        ARM_LOG_DISPATCH_TIMESTAMP_IF_ENABLED("NFRU-PERF", captureActive);
#else
        constexpr bool captureActive     = false;
        constexpr bool forceHistoryReset = false;
#endif

        // Optical flow
        {
            FfxResource opticalFlowColour = Convert(desc->presentColor);
            if (internal_context->resolutionPlan.needsOfColourDownsample)
            {
                NfruCpuClock::time_point ofColourStart{};
                if (verbose)
                {
                    ofColourStart = NfruCpuClock::now();
                }
                const FfxErrorCode downsampleResult = callWithoutNestedLogging([&]() {
                    return ffxFrameInterpolationDispatchDownsampleOfColourForFrameGeneration(
                        &internal_context->armFiContext, desc->commandList, opticalFlowColour);
                });
                if (verbose)
                {
                    cpuOfColourUs = elapsedMicroseconds(ofColourStart, NfruCpuClock::now());
                }
                if (downsampleResult != FFX_OK)
                {
#if FFX_CAPTURE_ENABLED
                    if (captureActive)
                        internal_context->controller->AbortDispatchCapture("Optical-flow colour downsample failed.");
#endif

                    internal_context->runtimeState.requiresRecreation = true;
                    return internal_context->diagnostics.LogRuntimeError(desc->frameID,
                                                                         "dispatch",
                                                                         "frame_interpolation",
                                                                         "of_colour",
                                                                         downsampleResult,
                                                                         "of_colour_downsample_failed",
                                                                         "failed to record the optical-flow colour downsample",
                                                                         verbose);
                }
                const FfxErrorCode queryResult = ffxFrameInterpolationGetDownsampledOfColourP1(&internal_context->armFiContext, &opticalFlowColour);
                if (queryResult != FFX_OK)
                {
#if FFX_CAPTURE_ENABLED
                    if (captureActive)
                        internal_context->controller->AbortDispatchCapture("Optical-flow colour downsample query failed.");
#endif

                    internal_context->runtimeState.requiresRecreation = true;
                    return internal_context->diagnostics.LogRuntimeError(desc->frameID,
                                                                         "dispatch",
                                                                         "frame_interpolation",
                                                                         "of_colour",
                                                                         queryResult,
                                                                         "of_colour_query_failed",
                                                                         "failed to query the downsampled optical-flow colour",
                                                                         verbose);
                }
            }

            {
                {
                    FfxOpticalFlowDispatchDescription ofDispatchDesc{};
                    ofDispatchDesc.commandList = desc->commandList;

                    ofDispatchDesc.color = opticalFlowColour;
                    if (internal_context->resolutionPlan.needsOfColourDownsample)
                    {
                        const FfxErrorCode queryResult =
                            ffxFrameInterpolationGetDownsampledOfColourM1(&internal_context->armFiContext, &ofDispatchDesc.colorTm1);
                        if (queryResult != FFX_OK)
                        {
#if FFX_CAPTURE_ENABLED
                            if (captureActive)
                                internal_context->controller->AbortDispatchCapture("Optical-flow colour downsample query failed.");
#endif
                            internal_context->runtimeState.requiresRecreation = true;
                            return internal_context->diagnostics.LogRuntimeError(desc->frameID,
                                                                                 "dispatch",
                                                                                 "frame_interpolation",
                                                                                 "of_colour",
                                                                                 queryResult,
                                                                                 "of_colour_history_query_failed",
                                                                                 "failed to query the previous downsampled optical-flow colour",
                                                                                 verbose);
                        }
                    }
                    else if (!sdkManagesPrevColor)
                    {
                        ofDispatchDesc.color.description.usage = FfxResourceUsage(ofDispatchDesc.color.description.usage);
                        ofDispatchDesc.colorTm1                = Convert(prepDesc->colorTm1);
                    }
                    else
                    {
                        ofDispatchDesc.color.description.usage = FfxResourceUsage(ofDispatchDesc.color.description.usage);
                        // Ping-pong the previous-source buffers each frame so an async interpolation
                        // pass can read this frame's source while the next frame writes the backup.
                        const uint32_t colorTm1Identifier = (desc->frameID % 2 == 0)
                                                                ? FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE
                                                                : FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE_BACKUP;
                        ofDispatchDesc.colorTm1 = internal_context->backendInterfaceShared.fpGetResource(&internal_context->backendInterfaceShared,
                                                                                                         internal_context->sharedResources[colorTm1Identifier]);
                    }

                    ofDispatchDesc.depth = Convert(prepDesc->depth);
                    memcpy(ofDispatchDesc.viewProjection, prepDesc->viewProjection, sizeof(ofDispatchDesc.viewProjection));
                    ofDispatchDesc.reset = effectiveReset || forceHistoryReset;
                    // OF texture should already have data graph usage from creation
                    ofDispatchDesc.opticalFlowVector = internal_context->backendInterfaceShared.fpGetResource(
                        &internal_context->backendInterfaceShared,
                        internal_context->sharedResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_VECTOR]);

                    // All-input downsampling requires reduced internal history; otherwise preserve the public external-history mode.
                    if (!internal_context->usesInternalDepthHistory)
                    {
                        ofDispatchDesc.depthTm1 = Convert(prepDesc->depthTm1);
                    }
                    else
                    {
                        ofDispatchDesc.depthTm1 = internal_context->backendInterfaceShared.fpGetResource(
                            &internal_context->backendInterfaceShared, internal_context->sharedResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1]);
                    }

                    NfruCpuClock::time_point opticalFlowStart{};
                    if (verbose)
                    {
                        opticalFlowStart = NfruCpuClock::now();
                    }
                    FfxNfruOpticalFlowTrace ofTrace{};
                    const FfxErrorCode      opticalFlowResult = callWithoutNestedLogging(
                        [&]() { return ffxOpticalFlowContextDispatchWithTrace(&internal_context->armOFContext, &ofDispatchDesc, &ofTrace); });
                    if (verbose)
                    {
                        cpuOpticalFlowUs = elapsedMicroseconds(opticalFlowStart, NfruCpuClock::now());
                    }
                    if (opticalFlowResult != FFX_OK)
                    {
#if FFX_CAPTURE_ENABLED
                        if (captureActive)
                            internal_context->controller->AbortDispatchCapture("Optical-flow dispatch failed.");
#endif
                        internal_context->runtimeState.requiresRecreation = ofTrace.failedStage != FfxNfruOfStage::Validation;
                        return internal_context->diagnostics.LogOpticalFlowError(desc->frameID, ofTrace.failedStage, opticalFlowResult, verbose);
                    }
                }
            }
        }

        // Frame interpolation
        {
            FfxFrameInterpolationDispatchDescription fiDispatchDesc{};

            // don't dispatch interpolation async for now: use the same commandlist for copy and interpolate
            fiDispatchDesc.commandList       = desc->commandList;
            fiDispatchDesc.currentBackBuffer = Convert(desc->presentColor);
            fiDispatchDesc.reset             = effectiveReset || forceHistoryReset;

            fiDispatchDesc.output            = Convert(desc->outputs[0]);
            fiDispatchDesc.opticalFlowVector = internal_context->backendInterfaceShared.fpGetResource(
                &internal_context->backendInterfaceShared, internal_context->sharedResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_VECTOR]);

            fiDispatchDesc.opticalFlowScale = {1.f / fiDispatchDesc.opticalFlowVector.description.width,
                                               1.f / fiDispatchDesc.opticalFlowVector.description.height};

            fiDispatchDesc.frameTimeDelta          = prepDesc->frameTimeDelta;
            fiDispatchDesc.cameraNear              = prepDesc->cameraNear;
            fiDispatchDesc.cameraFar               = prepDesc->cameraFar;
            fiDispatchDesc.viewSpaceToMetersFactor = prepDesc->viewSpaceToMetersFactor;
            fiDispatchDesc.cameraFovAngleVertical  = prepDesc->cameraFovAngleVertical;

            if (internal_context->frameGenFlags & FFX_API_FG_DISPATCH_FLAG_DRAW_DEBUG_VIEW)
            {
                fiDispatchDesc.flags |= FFX_FRAMEINTERPOLATION_DISPATCH_DRAW_DEBUG_VIEW;
            }

            fiDispatchDesc.frameID = desc->frameID;

            FfxNfruFrameInterpolationTrace fiTrace{};
            fiTrace.collectTimings = verbose;
            NfruCpuClock::time_point fiDispatchStart{};
            if (verbose)
            {
                fiDispatchStart = NfruCpuClock::now();
            }
            const FfxErrorCode fiResult =
                callWithoutNestedLogging([&]() { return ffxFrameInterpolationDispatchWithTrace(&internal_context->armFiContext, &fiDispatchDesc, &fiTrace); });
            if (verbose)
            {
                cpuFiDispatchUs = elapsedMicroseconds(fiDispatchStart, NfruCpuClock::now());
            }
            if (fiResult != FFX_OK)
            {
#if FFX_CAPTURE_ENABLED
                if (captureActive)
                    internal_context->controller->AbortDispatchCapture("Frame interpolation dispatch failed.");
#endif
                internal_context->runtimeState.requiresRecreation = fiTrace.failedStage != FfxNfruFiStage::Validation;
                return internal_context->diagnostics.LogFrameInterpolationError(desc->frameID, fiTrace.failedStage, fiResult, verbose);
            }

            // Independently advance the depth history after either all-input downsample or warp produced P1 feedback.
            if (internal_context->usesInternalDepthHistory)
            {
                std::swap(internal_context->sharedResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1],
                          internal_context->sharedResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1_NEXT]);
            }

            if (verbose)
            {
                const double cpuDispatchUs = elapsedMicroseconds(dispatchStart, NfruCpuClock::now());
                internal_context->diagnostics.LogDispatchCompleted(desc->frameID,
                                                                   {preparedFrame.cpuPrepareUs,
                                                                    cpuOfColourUs,
                                                                    cpuOpticalFlowUs,
                                                                    fiTrace.preScheduleUs,
                                                                    fiTrace.graphScheduleUs,
                                                                    fiTrace.postScheduleUs,
                                                                    cpuFiDispatchUs,
                                                                    cpuDispatchUs});
            }
        }
        preparedFrame.valid                            = false;
        internal_context->runtimeState.hasDispatched   = true;
        internal_context->runtimeState.previousFrameId = desc->frameID;
        internal_context->diagnostics.ClearError();
#if FFX_CAPTURE_ENABLED
        if (captureActive || internal_context->controller->IsPreparePending())
        {
            static_cast<void>(internal_context->controller->CompleteDispatchCapture(Convert(desc->outputs[0])));
        }
#endif
        return FFX_API_RETURN_OK;
    }
    else if (auto desc = ffx::DynamicCast<ffxApiDispatchDescFrameGenerationPrepare>(header))
    {
        const bool verbose = (internal_context->frameGenFlags & FFX_API_FG_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING) != 0;
        if (internal_context->runtimeState.requiresRecreation)
        {
            return internal_context->diagnostics.LogRuntimeError(desc->frameID,
                                                                 "prepare",
                                                                 "provider",
                                                                 "context_state",
                                                                 FFX_ERROR_BACKEND_API_ERROR,
                                                                 "context_requires_recreation",
                                                                 "context requires recreation after a backend dispatch failure",
                                                                 verbose);
        }
        for (const PreparedFrame& pendingFrame : internal_context->preparedFrames)
        {
            if (pendingFrame.valid)
            {
                return internal_context->diagnostics.LogRuntimeError(desc->frameID,
                                                                     "prepare",
                                                                     "provider",
                                                                     "prepare_validation",
                                                                     FFX_ERROR_INVALID_ARGUMENT,
                                                                     "prepare_already_pending",
                                                                     "dispatch the pending prepared frame before preparing another frame",
                                                                     verbose,
                                                                     FFX_API_RETURN_ERROR_PARAMETER);
            }
        }
        PreparedFrame& preparedFrame = internal_context->preparedFrames[desc->frameID % MAX_QUEUED_FRAMES];
        preparedFrame.valid          = false;
        if (verbose)
        {
            FfxSurfaceFormat colorHistoryFormat = ConvertEnum<FfxSurfaceFormat>(desc->colorTm1.description.format);
            if (sdkManagesPrevColor)
            {
                const FfxResourceDescription colorHistory = internal_context->backendInterfaceShared.fpGetResourceDescription(
                    &internal_context->backendInterfaceShared,
                    internal_context->sharedResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE]);
                colorHistoryFormat = colorHistory.format;
            }
            internal_context->diagnostics.LogPrepareBegin(*desc, GetLogContextInfo(internal_context), colorHistoryFormat);
        }
        NfruCpuClock::time_point prepareStart{};
        if (verbose)
        {
            prepareStart = NfruCpuClock::now();
        }

        FfxFrameInterpolationPrepareDescription dispatchDesc{};
        dispatchDesc.commandList                   = desc->commandList;
        dispatchDesc.jitterOffset.x                = desc->jitterOffset.x;
        dispatchDesc.jitterOffset.y                = desc->jitterOffset.y;
        dispatchDesc.motionVectorScale.x           = desc->motionVectorScale.x;
        dispatchDesc.motionVectorScale.y           = desc->motionVectorScale.y;
        dispatchDesc.mvSimilarityThreshold         = desc->mvSimilarityThreshold;
        dispatchDesc.mvSimilarityNoiseThreshold    = desc->mvSimilarityNoiseThreshold;
        dispatchDesc.mvSimilarityRelativeThreshold = desc->mvSimilarityRelativeThreshold;
        dispatchDesc.frameTimeDelta                = desc->frameTimeDelta;
        dispatchDesc.cameraNear                    = desc->cameraNear;
        dispatchDesc.cameraFar                     = desc->cameraFar;
        dispatchDesc.viewSpaceToMetersFactor       = desc->viewSpaceToMetersFactor;
        dispatchDesc.cameraFovAngleVertical        = desc->cameraFovAngleVertical;
        dispatchDesc.depth                         = Convert(desc->depth);
        dispatchDesc.motionVectors                 = Convert(desc->motionVectors);
        dispatchDesc.frameID                       = desc->frameID;

        if (!internal_context->usesInternalDepthHistory && desc->depthTm1.resource == nullptr)
        {
            return internal_context->diagnostics.LogRuntimeError(
                desc->frameID,
                "prepare",
                "provider",
                "prepare_validation",
                FFX_ERROR_INVALID_POINTER,
                "missing_depth_history",
                "no valid depth history was provided while FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH is disabled",
                verbose,
                FFX_API_RETURN_ERROR_PARAMETER);
        }

        if (!sdkManagesPrevColor && desc->colorTm1.resource == nullptr)
        {
            return internal_context->diagnostics.LogRuntimeError(
                desc->frameID,
                "prepare",
                "provider",
                "prepare_validation",
                FFX_ERROR_INVALID_POINTER,
                "missing_color_history",
                "no valid color history was provided while FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR is disabled",
                verbose,
                FFX_API_RETURN_ERROR_PARAMETER);
        }
        if (!internal_context->usesInternalDepthHistory)
        {
            // External-history mode: the application supplies M1 and current P1 is read directly by preprocess.
            dispatchDesc.depthTm1     = Convert(desc->depthTm1);
            dispatchDesc.depthTm1Next = {};  // null: warp pass skips depth write via shader permutation
        }
        else
        {
            // SDK manages: all depth resources internal
            dispatchDesc.depthTm1 = internal_context->backendInterfaceShared.fpGetResource(
                &internal_context->backendInterfaceShared, internal_context->sharedResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1]);
            dispatchDesc.depthTm1Next = internal_context->backendInterfaceShared.fpGetResource(
                &internal_context->backendInterfaceShared, internal_context->sharedResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1_NEXT]);
        }
        if (!sdkManagesPrevColor)
        {
            dispatchDesc.colorTm1 = Convert(desc->colorTm1);
        }
        else
        {
            // Ping-pong the previous-source buffers each frame: the SRV reads this frame's
            // "current" (even/odd) slot, the UAV/RT writes the alternate ("backup") slot.
            const bool     useBackupAsSource     = (desc->frameID % 2 != 0);
            const uint32_t colorTm1SrvIdentifier = useBackupAsSource ? FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE_BACKUP
                                                                     : FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE;
            const uint32_t colorTm1UavIdentifier = useBackupAsSource ? FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE
                                                                     : FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE_BACKUP;
            dispatchDesc.colorTm1       = internal_context->backendInterfaceShared.fpGetResource(&internal_context->backendInterfaceShared,
                                                                                           internal_context->sharedResources[colorTm1SrvIdentifier]);
            dispatchDesc.colorTm1Backup = internal_context->backendInterfaceShared.fpGetResource(&internal_context->backendInterfaceShared,
                                                                                                 internal_context->sharedResources[colorTm1UavIdentifier]);
        }
        // Copy View Projection
        memcpy(dispatchDesc.viewProjection, desc->viewProjection, sizeof(desc->viewProjection));
        const FfxErrorCode prepareResult =
            callWithoutNestedLogging([&]() { return ffxFrameInterpolationPrepareForFrameGeneration(&internal_context->armFiContext, &dispatchDesc); });
        if (prepareResult != FFX_OK)
        {
            const ffxReturnCode_t returnCode = (prepareResult == FFX_ERROR_INVALID_ARGUMENT || prepareResult == FFX_ERROR_INVALID_POINTER)
                                                   ? FFX_API_RETURN_ERROR_PARAMETER
                                                   : FFX_API_RETURN_ERROR_RUNTIME_ERROR;
            return internal_context->diagnostics.LogRuntimeError(desc->frameID,
                                                                 "prepare",
                                                                 "frame_interpolation",
                                                                 "prepare",
                                                                 prepareResult,
                                                                 "frame_interpolation_prepare_failed",
                                                                 "frame interpolation prepare failed",
                                                                 verbose,
                                                                 returnCode);
        }

        preparedFrame.description  = *desc;
        preparedFrame.cpuPrepareUs = verbose ? elapsedMicroseconds(prepareStart, NfruCpuClock::now()) : 0.0;
        preparedFrame.valid        = true;
        internal_context->diagnostics.ClearError();
        if (verbose)
        {
            internal_context->diagnostics.LogPrepareCompleted(desc->frameID, preparedFrame.cpuPrepareUs);
        }
#if FFX_CAPTURE_ENABLED
        internal_context->controller->OnPrepare();
#endif
        return FFX_API_RETURN_OK;
    }
    else
    {
        ffxLogMessage(ffxLogComponentName(), FFX_MESSAGE_TYPE_ERROR, "unsupported frame-generation dispatch descriptor");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
}

ffxProvider_FrameGeneration ffxProvider_FrameGeneration::Instance;
