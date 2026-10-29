// Copyright  © 2023 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string.h>
#include <tuple>

#include "FidelityFX/host/ffx_nss.h"
#define FFX_CPU

#include "FidelityFX/gpu/ffx_core.h"
#include "ffx_object_management.h"

#include "FidelityFX/host/ffx_util.h"

#include "ffx_logging.h"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#include "ffx_nss_private.h"
#include "ffx_nss_jitter_analysis.h"

static constexpr const char* ffxLogComponentName()
{
    return "NSS";
}

static std::atomic<uint64_t> g_nextLogContextId{1};
static std::atomic<bool>     g_nssVersionLogged{false};
static std::atomic<bool>     g_unsupportedDeviceLogged{false};
static constexpr size_t      NSS_LOG_MESSAGE_CAPACITY = 1024;

using NssCpuClock = std::chrono::steady_clock;

/// Dispatch CPU timings.
struct NssCpuTimings
{
    double preScheduleUs;    ///< Pre-processing.
    double graphScheduleUs;  ///< Data graph.
    double postScheduleUs;   ///< Post-processing.
};

/// Per-dispatch diagnostics.
struct NssDispatchTrace
{
    const char*   stage;    ///< Current stage.
    NssCpuTimings timings;  ///< CPU timings.
    bool          verbose;  ///< Verbose logging enabled.
};

static double elapsedMicroseconds(NssCpuClock::time_point begin, NssCpuClock::time_point end)
{
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

static void logNssVersion(FfxLogMessageCallback callback) noexcept
{
    constexpr size_t messageCapacity = 96;
    char             message[messageCapacity];
    std::snprintf(message, sizeof(message), "event=versions nss_version=%d.%d.%d", FFX_NSS_VERSION_MAJOR, FFX_NSS_VERSION_MINOR, FFX_NSS_VERSION_PATCH);
    ffxLogMessageOnceToSink(ffxLogComponentName(), FFX_MESSAGE_TYPE_INFO, message, g_nssVersionLogged, callback);
}

// max queued frames for descriptor management
static const uint32_t NSS_MAX_QUEUED_FRAMES = 16;

constexpr uint32_t FFX_NSS_MAX_JITTER_ANALYSIS_SAMPLES         = 256;
constexpr uint32_t FFX_NSS_JITTER_ANALYSIS_MESSAGE_BUFFER_SIZE = 512;

struct NssJitterAnalyzerState
{
    bool            active;
    uint32_t        expectedUniqueCount;
    uint32_t        sampleCount;
    FfxDimensions2D renderSize;
    FfxDimensions2D upscaleSize;
    NssJitterSample samples[FFX_NSS_MAX_JITTER_ANALYSIS_SAMPLES];
};

// lists to map shader resource bindpoint name to resource identifier
typedef struct ResourceBinding
{
    uint32_t index;
    char     name[64];
} ResourceBinding;

constexpr uint32_t externalCommonResources[] = {
    FFX_NSS_RESOURCE_IDENTIFIER_INPUT_COLOR,
    FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH,
    FFX_NSS_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS,
    FFX_NSS_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT,
};

// External resources that only needed when manage_history flag is set.
constexpr uint32_t externalCondtinalResources[] = {
    FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_UPSCALED_COLOR,
};

static const ResourceBinding srvTextureBindingTable[] = {
    // externally registered srv resources
    {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_COLOR, "r_input_color_jittered"},
    {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH, "r_input_depth"},
    {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS, "r_input_motion_vectors"},
    {FFX_NSS_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT, "r_upscaled_output"},

    // internally created srv resources
    {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH_TM1, "r_prev_depth"},
    {FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_TM1, "r_prev_luma_deriv"},
    {FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV, "r_luma_deriv"},
    {FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_UPSCALED_COLOR, "r_prev_upscaled_color"},
    {FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD, "r_input_nearest_depth_coord"},
    {FFX_NSS_RESOURCE_IDENTIFIER_DISOCCLUSION_MASK_LQ, "r_disocclusion_mask_lq"},
    {FFX_NSS_RESOURCE_IDENTIFIER_OFFSET_LUT, "r_offset_lut"},
    {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_EXPOSURE, "r_input_exposure"},

    // Aliased tensors
    {FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR, "r_temporal_feedback_tensor"},
};

static const ResourceBinding uavTextureBindingTable[] = {
    {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH_TM1, "rw_reconstructed_prev_depth"},
    {FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV, "rw_luma_deriv"},
    {FFX_NSS_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT, "rw_upscaled_output"},
    {FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_UPSCALED_COLOR, "rw_prev_upscaled_color"},
    {FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD, "rw_nearest_depth_coord_out"},
    {FFX_NSS_RESOURCE_IDENTIFIER_OFFSET_LUT, "rw_offset_lut"},
    {FFX_NSS_RESOURCE_IDENTIFIER_DEBUG_VIEWS, "rw_debug_views"},
    {FFX_NSS_RESOURCE_IDENTIFIER_DISOCCLUSION_MASK_LQ, "rw_disocclusion_mask_lq"},
};

static const ResourceBinding srvBufferBindingTable[] = {
    // Aliased tensors
    {FFX_NSS_RESOURCE_IDENTIFIER_KPN_TENSOR, "KpnCoeffBuffer"},
};

static const ResourceBinding uavBufferBindingTable[] = {
    // Aliased tensors
    {FFX_NSS_RESOURCE_IDENTIFIER_PREPROCESS_INPUT_TENSOR, "PreprocessInputTensorBuffer"},
};

static const ResourceBinding rtTextureBindingTable[] = {
    {FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV, "rw_luma_deriv"},
    {FFX_NSS_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT, "rw_upscaled_output"},
    {FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_UPSCALED_COLOR, "rw_prev_upscaled_color"},
    {FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD, "rw_nearest_depth_coord_out"},
    {FFX_NSS_RESOURCE_IDENTIFIER_DEBUG_VIEWS, "rw_debug_views"},
    {FFX_NSS_RESOURCE_IDENTIFIER_DISOCCLUSION_MASK_LQ, "rw_disocclusion_mask_lq"},
};

static const ResourceBinding srvTensorBindingTable[] = {
    {FFX_NSS_RESOURCE_IDENTIFIER_KPN_TENSOR, "r_coefficients_kpn_tensor"},
    {FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR, "r_temporal_feedback_tensor"},
    // Data graph resources
    {FFX_NSS_RESOURCE_IDENTIFIER_PREPROCESS_INPUT_TENSOR, "Resource_0_input"},
};

static const ResourceBinding uavTensorBindingTable[] = {
    // Shader resources - taken from shader reflection information
    {FFX_NSS_RESOURCE_IDENTIFIER_PREPROCESS_INPUT_TENSOR, "rw_preprocessed_tensor"},
    // Data graph resources - taken from data graph reflection information.
    // nss VGF: Resource_1_output = KPN coefficients (36ch, 1/4 res), Resource_2_output = temporal feedback (4ch)
    {FFX_NSS_RESOURCE_IDENTIFIER_KPN_TENSOR, "Resource_1_output"},
    {FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR, "Resource_2_output"},
};

#define FFX_LENGTH(x, y) (sqrt((x) * (x) + (y) * (y)))

static void printMessage(const FfxNssContext_Private* context, const FfxMsgType type, const char* message)
{
    char        contextualMessage[NSS_LOG_MESSAGE_CAPACITY];
    const char* logMessage = message;
    if (context && message)
    {
        std::snprintf(contextualMessage,
                      sizeof(contextualMessage),
                      "ctx_id=%" PRIu64 " dispatch_id=%" PRIu64 " %s",
                      context->diagnostics.contextId,
                      context->diagnostics.dispatchId,
                      message);
        logMessage = contextualMessage;
    }
    ffxLogMessage(ffxLogComponentName(), type, logMessage, context ? context->contextDescription.fpMessage : nullptr);
}

static FfxErrorCode logErrorWithCallback(FfxErrorCode          errorCode,
                                         const char*           errorMessage,
                                         FfxLogMessageCallback callback = nullptr,
                                         const char*           funcName = FFX_BUILTIN_FUNCTION)
{
    if (callback)
    {
        callback(FFX_MESSAGE_TYPE_ERROR, errorMessage);
    }
    return ffxLogError(errorCode, errorMessage, funcName);
}

static FfxErrorCode logDispatchError(FfxNssContext_Private* context, FfxErrorCode errorCode, const char* errorMessage, bool verbose, const char* stage)
{
    NssDispatchErrorState& lastError = context->diagnostics.lastError;
    const bool             repeated  = lastError.code == errorCode && lastError.reason && lastError.stage && std::strcmp(lastError.reason, errorMessage) == 0 &&
                          std::strcmp(lastError.stage, stage) == 0;
    lastError = {errorCode, errorMessage, stage};
    if (verbose || !repeated)
    {
        char message[NSS_LOG_MESSAGE_CAPACITY];
        std::snprintf(message,
                      sizeof(message),
                      "event=dispatch_error stage=%s error_code=%d error_name=%s reason=\"%s\" fallback=application_required",
                      stage,
                      errorCode,
                      ffxGetErrorCodeName(errorCode),
                      errorMessage);
        printMessage(context, FFX_MESSAGE_TYPE_ERROR, message);
    }
    return errorCode;
}

static void logDispatchResource(
    const FfxNssContext_Private* context, const char* role, const char* source, const FfxResourceDescription& description, bool present)
{
    char message[NSS_LOG_MESSAGE_CAPACITY];
    std::snprintf(message,
                  sizeof(message),
                  "event=dispatch_resource role=%s source=%s present=%u resolution=%ux%u format=%s format_id=%u",
                  role,
                  source,
                  static_cast<unsigned>(present),
                  description.width,
                  description.height,
                  ffxGetSurfaceFormatName(description.format),
                  static_cast<unsigned>(description.format));
    printMessage(context, FFX_MESSAGE_TYPE_INFO, message);
}

static void logDispatchBegin(const FfxNssContext_Private* context, const FfxNssDispatchDescription* params)
{
    char message[NSS_LOG_MESSAGE_CAPACITY];
    std::snprintf(message,
                  sizeof(message),
                  "event=dispatch_begin render=%ux%u upscale=%ux%u reset=%u flags=0x%08x debug_view=%u",
                  params->renderSize.width,
                  params->renderSize.height,
                  params->upscaleSize.width,
                  params->upscaleSize.height,
                  static_cast<unsigned>(params->reset),
                  params->flags,
                  static_cast<unsigned>((params->flags & FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW) != 0));
    printMessage(context, FFX_MESSAGE_TYPE_INFO, message);
    std::snprintf(message,
                  sizeof(message),
                  "event=dispatch_camera jitter=%.6gx%.6g fov_y=%.6g near=%.6g far=%.6g exposure=%.6g",
                  params->jitterOffset.x,
                  params->jitterOffset.y,
                  params->cameraFovAngleVertical,
                  params->cameraNear,
                  params->cameraFar,
                  params->exposure);
    printMessage(context, FFX_MESSAGE_TYPE_INFO, message);
    std::snprintf(message,
                  sizeof(message),
                  "event=dispatch_motion mv_scale=%.6gx%.6g frame_time_delta_ms=%.6g",
                  params->motionVectorScale.x,
                  params->motionVectorScale.y,
                  params->frameTimeDelta);
    printMessage(context, FFX_MESSAGE_TYPE_INFO, message);
    std::snprintf(message,
                  sizeof(message),
                  "event=dispatch_pipeline preprocess=%s postprocess=%s disocclusion_mask=%u dynamic_lut=%u",
                  (context->pipelineFlags & NSS_SHADER_PERMUTATION_PRE_PROCESS_COMPUTE) != 0 ? "compute" : "fragment",
                  (context->pipelineFlags & NSS_SHADER_PERMUTATION_POST_PROCESS_COMPUTE) != 0 ? "compute" : "fragment",
                  static_cast<unsigned>(context->useDisocclusionMaskPass),
                  static_cast<unsigned>(context->useDynamicOffsetLut));
    printMessage(context, FFX_MESSAGE_TYPE_INFO, message);
    logDispatchResource(context, "color", "application", params->color.description, params->color.resource != nullptr);
    logDispatchResource(context, "depth", "application", params->depth.description, params->depth.resource != nullptr);
    logDispatchResource(context, "motion_vectors", "application", params->motionVectors.description, params->motionVectors.resource != nullptr);
    if ((context->contextDescription.flags & FFX_NSS_CONTEXT_FLAG_MANAGE_HISTORY) != 0)
    {
        logDispatchResource(
            context, "history", "managed", context->managedHistoryDescription, context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_1].internalIndex != 0);
    }
    else
    {
        logDispatchResource(context, "history", "application", params->outputTm1.description, params->outputTm1.resource != nullptr);
    }
    logDispatchResource(context, "output", "application", params->output.description, params->output.resource != nullptr);
    if ((params->flags & FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW) != 0)
    {
        logDispatchResource(context, "debug_views", "application", params->debugViews.description, params->debugViews.resource != nullptr);
    }
}

static void resetJitterAnalyzer(FfxNssContext_Private* context)
{
    NssJitterAnalyzerState* state = context->jitterAnalyzer;
    if (!state)
        return;

    state->active              = false;
    state->expectedUniqueCount = 0;
    state->sampleCount         = 0;
}

static void releaseJitterAnalyzer(FfxNssContext_Private* context)
{
    delete context->jitterAnalyzer;
    context->jitterAnalyzer = nullptr;
}

static NssJitterAnalyzerState* getJitterAnalyzer(FfxNssContext_Private* context)
{
    NssJitterAnalyzerState* state = context->jitterAnalyzer;
    if (!state)
    {
        state = new (std::nothrow) NssJitterAnalyzerState{};
        if (!state)
        {
            printMessage(context, FFX_MESSAGE_TYPE_WARNING, "Failed to allocate NSS jitter analyzer state; jitter analysis skipped.");
            return nullptr;
        }

        context->jitterAnalyzer = state;
    }

    return state;
}

static void reportJitterAnalysis(FfxNssContext_Private* context, const NssJitterAnalysis& analysis)
{
    char summary[FFX_NSS_JITTER_ANALYSIS_MESSAGE_BUFFER_SIZE];
    snprintf(summary,
             sizeof(summary),
             "NSS jitter analysis: samples=%u unique=%u/%u score=%u/10 centroid=(%.4f, %.4f) span=(%.4f, %.4f) quadrants=%u/4 "
             "in_bounds=%.1f%% grid=%.1f%% discrepancy=%.4f spacing_cv=%.4f xy_correlation=%.4f",
             analysis.sampleCount,
             analysis.uniqueCount,
             analysis.expectedUniqueCount,
             analysis.score,
             analysis.centroidX,
             analysis.centroidY,
             analysis.spanX,
             analysis.spanY,
             analysis.quadrantCount,
             analysis.inBoundsFraction * 100.0f,
             analysis.gridOccupancy * 100.0f,
             analysis.starDiscrepancy,
             analysis.spacingCv,
             analysis.xyCorrelation);
    printMessage(context, FFX_MESSAGE_TYPE_WARNING, summary);

    if (analysis.failedChecks != 0)
    {
        char issues[FFX_NSS_JITTER_ANALYSIS_MESSAGE_BUFFER_SIZE];
        snprintf(issues, sizeof(issues), "NSS jitter analysis failed checks (%u):", analysis.failedChecks);
        const char* names[]  = {"non-finite",
                               "out-of-range",
                               "not-centered",
                               "insufficient-span",
                               "missing-quadrants",
                               "grid-coverage",
                               "star-discrepancy",
                               "nearest-neighbor-spacing",
                               "xy-correlation",
                               "insufficient-unique"};
        const bool  passed[] = {analysis.finite,
                               analysis.inBounds,
                               analysis.centered,
                               analysis.wideSpan,
                               analysis.quadrants,
                               analysis.grid,
                               analysis.discrepancy,
                               analysis.spacing,
                               analysis.correlation,
                               analysis.enoughUnique};

        for (uint32_t index = 0; index < FFX_COUNTOF(names); ++index)
        {
            if (!passed[index])
            {
                const size_t used = strlen(issues);
                snprintf(issues + used, sizeof(issues) - used, " %s", names[index]);
            }
        }
        printMessage(context, FFX_MESSAGE_TYPE_WARNING, issues);
    }
    else
    {
        printMessage(context, FFX_MESSAGE_TYPE_INFO, "NSS jitter analysis passed all checks");
    }
}

static void analyzeJitterDispatch(FfxNssContext_Private* context, const FfxNssDispatchDescription* params)
{
    const uint32_t expectedUniqueCount = std::max(1u, static_cast<uint32_t>(ffxNssGetJitterPhaseCount(params->renderSize.width, params->upscaleSize.width)));
    NssJitterAnalyzerState* state      = getJitterAnalyzer(context);
    if (!state)
        return;

    if (!state->active || state->renderSize.width != params->renderSize.width || state->renderSize.height != params->renderSize.height ||
        state->upscaleSize.width != params->upscaleSize.width || state->upscaleSize.height != params->upscaleSize.height ||
        state->expectedUniqueCount != expectedUniqueCount || params->reset)
    {
        resetJitterAnalyzer(context);
        state->active              = true;
        state->expectedUniqueCount = expectedUniqueCount;
        state->renderSize          = params->renderSize;
        state->upscaleSize         = params->upscaleSize;
    }

    if (state->sampleCount < FFX_NSS_MAX_JITTER_ANALYSIS_SAMPLES)
        state->samples[state->sampleCount++] = {params->jitterOffset.x, params->jitterOffset.y};

    if (state->sampleCount == expectedUniqueCount || state->sampleCount == FFX_NSS_MAX_JITTER_ANALYSIS_SAMPLES)
    {
        const NssJitterAnalysis analysis = AnalyzeNssJitter(state->samples, state->sampleCount, state->expectedUniqueCount);
        reportJitterAnalysis(context, analysis);
        resetJitterAnalyzer(context);
    }
}

static void nssDebugCheckDispatch(FfxNssContext_Private* context, const FfxNssDispatchDescription* params)
{
    if (params->commandList == nullptr)
    {
        printMessage(context, FFX_MESSAGE_TYPE_ERROR, "commandList is null");
    }

    if (params->color.resource == nullptr)
    {
        printMessage(context, FFX_MESSAGE_TYPE_ERROR, "color resource is null");
    }

    if (params->depth.resource == nullptr)
    {
        printMessage(context, FFX_MESSAGE_TYPE_ERROR, "depth resource is null");
    }

    if (params->motionVectors.resource == nullptr)
    {
        printMessage(context, FFX_MESSAGE_TYPE_ERROR, "motionVectors resource is null");
    }

    if (params->output.resource == nullptr)
    {
        printMessage(context, FFX_MESSAGE_TYPE_ERROR, "output resource is null");
    }

    const bool manageHistory = (context->contextDescription.flags & FFX_NSS_CONTEXT_FLAG_MANAGE_HISTORY) != 0;
    if (!manageHistory)
    {
        if (params->outputTm1.resource == nullptr)
        {
            printMessage(context, FFX_MESSAGE_TYPE_ERROR, "No valid history provided when FFX_NSS_CONTEXT_FLAG_MANAGE_HISTORY is not set.");
        }

        const bool hasDebugView = (params->flags & FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW) == FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW;
        if (hasDebugView)
        {
            if (params->debugViews.resource == nullptr)
            {
                printMessage(context,
                             FFX_MESSAGE_TYPE_ERROR,
                             "No valid debug view provided when FFX_NSS_CONTEXT_FLAG_MANAGE_HISTORY is not set while "
                             "FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW is set.");
            }
        }
    }

    const bool useExposureTexture = (context->contextDescription.flags & FFX_NSS_CONTEXT_FLAG_EXPOSURE_TEXTURE) != 0;
    if (useExposureTexture)
    {
        if (params->exposureTexture.resource == nullptr)
        {
            printMessage(context, FFX_MESSAGE_TYPE_ERROR, "No valid exposure texture provided when FFX_NSS_CONTEXT_FLAG_EXPOSURE_TEXTURE is set.");
        }
    }

    if (fabs(params->jitterOffset.x) > 0.500001f || fabs(params->jitterOffset.y) > 0.500001f)
    {
        printMessage(context, FFX_MESSAGE_TYPE_WARNING, "jitterOffset contains value outside of expected range [-0.5, 0.5]");
    }

    if (fabs(params->motionVectorScale.x) > (float)context->contextDescription.renderSize.width ||
        fabs(params->motionVectorScale.y) > (float)context->contextDescription.renderSize.height)
    {
        printMessage(context, FFX_MESSAGE_TYPE_WARNING, "motionVectorScale is greater than context renderSize");
    }
}

static FfxErrorCode patchResourceBindings(FfxPipelineState* inoutPipeline)
{
    FFX_ASSERT(inoutPipeline != nullptr);

    auto patchResourceIdentifier = [&](FfxResourceBinding*    pipelineResourceBindingTable,
                                       const ResourceBinding* resourceIdNameTable,
                                       uint32_t               resourceIdNameTableCount,
                                       uint32_t               resourceCount) -> FfxErrorCode {
        FFX_ASSERT(pipelineResourceBindingTable != nullptr);
        FFX_ASSERT(resourceIdNameTable != nullptr);

        for (uint32_t resourceIndex = 0; resourceIndex < resourceCount; ++resourceIndex)
        {
            int32_t mapIndex = 0;
            for (mapIndex = 0; mapIndex < (int32_t)resourceIdNameTableCount; ++mapIndex)
            {
                if (0 == strcmp(resourceIdNameTable[mapIndex].name, pipelineResourceBindingTable[resourceIndex].name))
                    break;
            }
            if (mapIndex == (int32_t)resourceIdNameTableCount)
                return ffxLogError(FFX_ERROR_INVALID_ARGUMENT, "the shader resource binding name is not recognized");

            pipelineResourceBindingTable[resourceIndex].resourceIdentifier = resourceIdNameTable[mapIndex].index;
        }
        return FFX_OK;
    };

    // SRV textures
    FfxErrorCode errorCode =
        patchResourceIdentifier(inoutPipeline->srvTextureBindings, srvTextureBindingTable, FFX_COUNTOF(srvTextureBindingTable), inoutPipeline->srvTextureCount);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    // UAV textures
    errorCode =
        patchResourceIdentifier(inoutPipeline->uavTextureBindings, uavTextureBindingTable, FFX_COUNTOF(uavTextureBindingTable), inoutPipeline->uavTextureCount);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    // SRV buffers
    errorCode =
        patchResourceIdentifier(inoutPipeline->srvBufferBindings, srvBufferBindingTable, FFX_COUNTOF(srvBufferBindingTable), inoutPipeline->srvBufferCount);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    // UAV buffers
    errorCode =
        patchResourceIdentifier(inoutPipeline->uavBufferBindings, uavBufferBindingTable, FFX_COUNTOF(uavBufferBindingTable), inoutPipeline->uavBufferCount);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    // Render target textures
    errorCode = patchResourceIdentifier(inoutPipeline->rtBindings, rtTextureBindingTable, FFX_COUNTOF(rtTextureBindingTable), inoutPipeline->rtCount);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    // SRV tensors
    errorCode =
        patchResourceIdentifier(inoutPipeline->srvTensorBindings, srvTensorBindingTable, FFX_COUNTOF(srvTensorBindingTable), inoutPipeline->srvTensorCount);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    // UAV tensors
    errorCode =
        patchResourceIdentifier(inoutPipeline->uavTensorBindings, uavTensorBindingTable, FFX_COUNTOF(uavTensorBindingTable), inoutPipeline->uavTensorCount);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    return FFX_OK;
}

static uint32_t getPipelinePermutationFlags(FfxNssContext_Private* context)
{
    FFX_ASSERT(context);
    FFX_ASSERT(context->contextDescription.backendInterface.devCapInitialized);

    FfxDeviceCapabilities capabilities;
    FFX_VALIDATE(context->contextDescription.backendInterface.fpGetDeviceCapabilities(&context->contextDescription.backendInterface, &capabilities));
    const uint32_t contextFlags          = context->contextDescription.flags;
    const bool     fp16Supported         = capabilities.fp16Supported;
    const bool     computeSupportTensor  = capabilities.computeSupportTensor;
    const bool     fragmentSupportTensor = capabilities.fragmentSupportTensor;

    uint32_t flags = 0;
    flags |= (contextFlags & FFX_NSS_CONTEXT_FLAG_QUANTIZED) ? NSS_SHADER_PERMUTATION_QUANTIZED : 0;
    flags |= (contextFlags & FFX_NSS_CONTEXT_FLAG_DEPTH_INVERTED) ? NSS_SHADER_PERMUTATION_REVERSE_Z : 0;
    flags |= (contextFlags & FFX_NSS_CONTEXT_FLAG_RESAMPLE_BICUBIC) ? NSS_SHADER_PERMUTATION_RESAMPLE_BICUBIC : 0;
    flags |= (contextFlags & FFX_NSS_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) ? NSS_SHADER_PERMUTATION_JITTER_MOTION_VECTORS : 0;
    flags |= (contextFlags & FFX_NSS_CONTEXT_FLAG_MANAGE_HISTORY) ? NSS_SHADER_PERMUTATION_MANAGE_HISTORY : 0;
    flags |= (contextFlags & FFX_NSS_CONTEXT_FLAG_PRE_PROCESS_COMPUTE) ? NSS_SHADER_PERMUTATION_PRE_PROCESS_COMPUTE : 0;
    flags |= (contextFlags & FFX_NSS_CONTEXT_FLAG_POST_PROCESS_COMPUTE) ? NSS_SHADER_PERMUTATION_POST_PROCESS_COMPUTE : 0;
    flags |= (contextFlags & FFX_NSS_CONTEXT_FLAG_EXPOSURE_TEXTURE) ? NSS_SHADER_PERMUTATION_EXPOSURE_TEXTURE : 0;
    flags |= (contextFlags & FFX_NSS_CONTEXT_FLAG_LOW_DYNAMIC_RANGE) ? NSS_SHADER_PERMUTATION_LDR_MODE : 0;
    flags |= (computeSupportTensor) ? NSS_SHADER_PERMUTATION_COMPUTE_SUPPORT_TENSOR : 0;
    flags |= (fragmentSupportTensor) ? NSS_SHADER_PERMUTATION_FRAGMENT_SUPPORT_TENSOR : 0;

    const bool requestDisableFp16 = (contextFlags & FFX_NSS_CONTEXT_FLAG_DISABLE_16BIT) != 0;
    const bool disable16bit       = !fp16Supported || requestDisableFp16;
    flags |= disable16bit ? NSS_SHADER_PERMUTATION_DISABLE_16BIT : 0;

    if (!fp16Supported && !requestDisableFp16)
    {
        printMessage(context, FFX_MESSAGE_TYPE_WARNING, "FP16 is not enabled by the device; NSS fallback to 32-bit shader permutations.");
    }

    // If both axes match exact x2, use the static LUT path for better performance.
    if (context->contextDescription.upscaleSize.width == 2 * context->contextDescription.renderSize.width &&
        context->contextDescription.upscaleSize.height == 2 * context->contextDescription.renderSize.height)
    {
        flags |= NSS_SHADER_PERMUTATION_SCALE_PRESET_X2;
    }

    const auto qualityMode = context->contextDescription.qualityMode;
    flags |= (static_cast<uint32_t>(qualityMode) << NSS_SHADER_PERMUTATION_QUALITY_MODE_SHIFT);

    return flags;
}

static uint32_t computeGreatestCommonDivisor(uint32_t a, uint32_t b)
{
    while (b != 0)
    {
        const uint32_t t = b;
        b                = a % b;
        a                = t;
    }
    return a == 0 ? 1u : a;
}

static uint32_t offsetLutGroupsPerTile(uint32_t pipelineFlags)
{
    const auto qualityModeValue =
        static_cast<FfxNssShaderQualityMode>((pipelineFlags >> NSS_SHADER_PERMUTATION_QUALITY_MODE_SHIFT) & NSS_SHADER_QUALITY_MODE_MASK);
    return (qualityModeValue == FFX_NSS_SHADER_QUALITY_MODE_QUALITY) ? 3u : 1u;
}

static FfxErrorCode buildDataGraphTensorInfo(const FfxInternalResourceDescription* internalSurfaceDesc,
                                             uint32_t                              internalSurfaceCount,
                                             FfxDataGraphTensorInfo*               outTensorInfo,
                                             uint32_t                              maxTensorInfoCount,
                                             uint32_t*                             outTensorInfoCount)
{
    FFX_ASSERT(internalSurfaceDesc != nullptr);
    FFX_ASSERT(outTensorInfo != nullptr);
    FFX_ASSERT(outTensorInfoCount != nullptr);

    uint32_t tensorInfoCount = 0;

    const auto appendTensorInfo = [&](const ResourceBinding& binding, bool bufferAliased) -> FfxErrorCode {
        for (uint32_t infoIndex = 0; infoIndex < tensorInfoCount; ++infoIndex)
        {
            if (0 == strcmp(outTensorInfo[infoIndex].resourceName, binding.name))
            {
                outTensorInfo[infoIndex].bufferAliased = bufferAliased;
                return FFX_OK;
            }
        }

        FFX_RETURN_ON_ERROR(tensorInfoCount < maxTensorInfoCount, FFX_ERROR_OUT_OF_RANGE);
        outTensorInfo[tensorInfoCount++] = {binding.name, bufferAliased};
        return FFX_OK;
    };

    for (uint32_t surfaceIndex = 0; surfaceIndex < internalSurfaceCount; ++surfaceIndex)
    {
        const FfxInternalResourceDescription& surfaceDesc = internalSurfaceDesc[surfaceIndex];
        if (surfaceDesc.type != FFX_RESOURCE_TYPE_TENSOR)
        {
            continue;
        }

        const bool bufferAliased = (surfaceDesc.flags & FFX_RESOURCE_FLAGS_BUFFER_ALIASED) == FFX_RESOURCE_FLAGS_BUFFER_ALIASED;
        for (uint32_t bindingIndex = 0; bindingIndex < FFX_COUNTOF(srvTensorBindingTable); ++bindingIndex)
        {
            if (srvTensorBindingTable[bindingIndex].index == surfaceDesc.id)
            {
                FFX_VALIDATE(appendTensorInfo(srvTensorBindingTable[bindingIndex], bufferAliased));
            }
        }

        for (uint32_t bindingIndex = 0; bindingIndex < FFX_COUNTOF(uavTensorBindingTable); ++bindingIndex)
        {
            if (uavTensorBindingTable[bindingIndex].index == surfaceDesc.id)
            {
                FFX_VALIDATE(appendTensorInfo(uavTensorBindingTable[bindingIndex], bufferAliased));
            }
        }
    }

    *outTensorInfoCount = tensorInfoCount;
    return FFX_OK;
}

static FfxErrorCode createPipelineStates(FfxNssContext_Private* context, const FfxDataGraphTensorInfo* dataGraphTensorInfo, uint32_t dataGraphTensorInfoCount)
{
    FFX_ASSERT(context);

    FfxPipelineDescription pipelineDescription = {};
    pipelineDescription.contextFlags           = context->contextDescription.flags;

    // Set up pipeline descriptor (basically RootSignature and binding)

    // Root constants
    pipelineDescription.rootConstantBufferCount = 1;
    const auto CreatePipeline                   = [&](FfxPass pass, const char* name, FfxPipelineState* pipeline, bool useFragment) -> FfxErrorCode {
        ffxSafeReleasePipeline(&context->contextDescription.backendInterface, pipeline, context->effectContextId);

        strncpy(pipelineDescription.name, name, FFX_RESOURCE_NAME_SIZE - 1);
        pipelineDescription.name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';

        FfxRootConstantDescription rootConstantDescs[1] = {
            {sizeof(NssConstants) / sizeof(uint32_t), useFragment ? FFX_BIND_PIXEL_SHADER_STAGE : FFX_BIND_COMPUTE_SHADER_STAGE}};
        pipelineDescription.rootConstants = rootConstantDescs;

        FfxSamplerDescription samplerDescs[2] = {{FFX_FILTER_TYPE_MINMAGMIP_POINT,
                                                  FFX_ADDRESS_MODE_CLAMP,
                                                  FFX_ADDRESS_MODE_CLAMP,
                                                  FFX_ADDRESS_MODE_CLAMP,
                                                  useFragment ? FFX_BIND_PIXEL_SHADER_STAGE : FFX_BIND_COMPUTE_SHADER_STAGE,
                                                  FFX_FALSE},
                                                 {FFX_FILTER_TYPE_MINMAGMIP_LINEAR,
                                                  FFX_ADDRESS_MODE_CLAMP,
                                                  FFX_ADDRESS_MODE_CLAMP,
                                                  FFX_ADDRESS_MODE_CLAMP,
                                                  useFragment ? FFX_BIND_PIXEL_SHADER_STAGE : FFX_BIND_COMPUTE_SHADER_STAGE,
                                                  FFX_FALSE}};

        pipelineDescription.samplerCount = 2;
        pipelineDescription.samplers     = samplerDescs;
        pipelineDescription.stage        = useFragment ? FFX_BIND_PIXEL_SHADER_STAGE : FFX_BIND_COMPUTE_SHADER_STAGE;

        if (useFragment)
        {
            FFX_VALIDATE(context->contextDescription.backendInterface.fpCreateGraphicsPipeline(&context->contextDescription.backendInterface,
                                                                                               FFX_EFFECT_NSS,
                                                                                               pass,
                                                                                               context->pipelineFlags,
                                                                                               &pipelineDescription,
                                                                                               context->effectContextId,
                                                                                               pipeline));
        }
        else
        {
            FFX_VALIDATE(context->contextDescription.backendInterface.fpCreateComputePipeline(&context->contextDescription.backendInterface,
                                                                                              FFX_EFFECT_NSS,
                                                                                              pass,
                                                                                              context->pipelineFlags,
                                                                                              &pipelineDescription,
                                                                                              context->effectContextId,
                                                                                              pipeline));
        }
        FFX_VALIDATE(patchResourceBindings(pipeline));

        return FFX_OK;
    };

    const bool preProcessUseFragment  = (context->pipelineFlags & NSS_SHADER_PERMUTATION_PRE_PROCESS_COMPUTE) == 0;
    const bool postProcessUseFragment = (context->pipelineFlags & NSS_SHADER_PERMUTATION_POST_PROCESS_COMPUTE) == 0;

    FFX_VALIDATE(CreatePipeline(FFX_NSS_PASS_DEPTH_SCATTER, "NSS-DepthScatter", &context->pipelineNssDepthScatter, false));
    if (context->useDisocclusionMaskPass)
    {
        FFX_VALIDATE(CreatePipeline(FFX_NSS_PASS_DISOCCLUSION_MASK, "NSS-DisocclusionMask", &context->pipelineNssDisocclusionMask, preProcessUseFragment));
    }
    FFX_VALIDATE(CreatePipeline(FFX_NSS_PASS_PREPROCESS, "NSS-Preprocess", &context->pipelineNssPreprocess, preProcessUseFragment));
    if (context->useDynamicOffsetLut)
    {
        FFX_VALIDATE(CreatePipeline(FFX_NSS_PASS_GENERATE_OFFSET_LUT, "NSS-GenerateOffsetLut", &context->pipelineNssGenerateOffsetLut, false));
    }
    FFX_VALIDATE(CreatePipeline(FFX_NSS_PASS_POSTPROCESS, "NSS-Postprocess", &context->pipelineNssPostprocess, postProcessUseFragment));
    FFX_VALIDATE(CreatePipeline(FFX_NSS_PASS_DEBUG_VIEW, "NSS-DebugView", &context->pipelineNssDebugView, postProcessUseFragment));

    // DATA GRAPH
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineNssDataGraph, context->effectContextId);
    strncpy(pipelineDescription.name, "NSS-Graph", FFX_RESOURCE_NAME_SIZE - 1);
    pipelineDescription.name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
    pipelineDescription.dataGraphTensorInfo              = dataGraphTensorInfo;
    pipelineDescription.dataGraphTensorInfoCount         = dataGraphTensorInfoCount;
    FFX_VALIDATE(context->contextDescription.backendInterface.fpCreateDataGraphPipeline(&context->contextDescription.backendInterface,
                                                                                        FFX_EFFECT_NSS,
                                                                                        FFX_NSS_PASS_DATA_GRAPH,
                                                                                        context->pipelineFlags,
                                                                                        &pipelineDescription,
                                                                                        context->effectContextId,
                                                                                        context->dataGraphSize.width,
                                                                                        context->dataGraphSize.height,
                                                                                        &context->pipelineNssDataGraph));
    FFX_VALIDATE(patchResourceBindings(&context->pipelineNssDataGraph));
    // END DATA GRAPH

    return FFX_OK;
}

static FfxErrorCode createResourceFromDescription(FfxNssContext_Private* context, const FfxInternalResourceDescription* resDesc)
{
    const FfxResourceType        resourceType        = resDesc->type;
    const FfxResourceDescription resourceDescription = {resourceType,
                                                        resDesc->format,
                                                        resDesc->width,
                                                        resDesc->height,
                                                        (resourceType == FFX_RESOURCE_TYPE_TENSOR) ? resDesc->channel : 1,
                                                        resDesc->mipCount,
                                                        resDesc->flags,
                                                        resDesc->usage,
                                                        resDesc->batchSize,
                                                        resDesc->shapeSize};
    const FfxResourceStates initialState = (resDesc->usage == FFX_RESOURCE_USAGE_READ_ONLY) ? FFX_RESOURCE_STATE_COMPUTE_READ : FFX_RESOURCE_STATE_GENERIC_UAV;
    const FfxCreateResourceDescription createResourceDescription = {
        FFX_HEAP_TYPE_DEFAULT, resourceDescription, initialState, resDesc->name, resDesc->id, resDesc->initData};
    const FfxErrorCode errorCode = context->contextDescription.backendInterface.fpCreateResource(
        &context->contextDescription.backendInterface, &createResourceDescription, context->effectContextId, &context->srvResources[resDesc->id]);
    if (errorCode == FFX_OK && resDesc->id == FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_1)
    {
        context->managedHistoryDescription = resourceDescription;
    }
    return errorCode;
}

static void nssLogContextCreate(const FfxNssContext_Private* context)
{
    FFX_ASSERT(context);
    FFX_ASSERT_MESSAGE(context->dataGraphSize.width % FFX_NSS_RESOURCE_ALIGNMENT == 0, "DataGraph width is not aligned!");
    FFX_ASSERT_MESSAGE(context->dataGraphSize.height % FFX_NSS_RESOURCE_ALIGNMENT == 0, "DataGraph height is not aligned!");

    const auto getQualityModeString = [](const uint32_t mode) -> const char* {
        switch (static_cast<FfxNssShaderQualityMode>(mode))
        {
        case FFX_NSS_SHADER_QUALITY_MODE_QUALITY:
            return "Quality";
        case FFX_NSS_SHADER_QUALITY_MODE_BALANCED:
            return "Balanced";
        case FFX_NSS_SHADER_QUALITY_MODE_PERFORMANCE:
            return "Performance";
        default:
            break;
        }
        FFX_ASSERT_MESSAGE(0, "Unknown shader quality mode!");
        return "Unknown";
    };

    const FfxNssContextDescription& desc          = context->contextDescription;
    const uint32_t                  pipelineFlags = context->pipelineFlags;
    const uint32_t                  qualityMode   = (pipelineFlags >> NSS_SHADER_PERMUTATION_QUALITY_MODE_SHIFT) & NSS_SHADER_QUALITY_MODE_MASK;
    FFX_ASSERT(qualityMode == static_cast<uint32_t>(context->contextDescription.qualityMode));
    const char* preprocessStage  = ((pipelineFlags & NSS_SHADER_PERMUTATION_PRE_PROCESS_COMPUTE) != 0) ? "compute" : "fragment";
    const char* postprocessStage = ((pipelineFlags & NSS_SHADER_PERMUTATION_POST_PROCESS_COMPUTE) != 0) ? "compute" : "fragment";
    const bool  manageHistory    = (pipelineFlags & NSS_SHADER_PERMUTATION_MANAGE_HISTORY) != 0;
    const float scaleX           = static_cast<float>(desc.upscaleSize.width) / desc.renderSize.width;
    const float scaleY           = static_cast<float>(desc.upscaleSize.height) / desc.renderSize.height;
    char        message[NSS_LOG_MESSAGE_CAPACITY];
    std::snprintf(message,
                  sizeof(message),
                  "event=context_create render=%ux%u upscale=%ux%u display=%ux%u scale=%.4fx%.4f quality=%s "
                  "context_flags=0x%08x pipeline_flags=0x%08x preprocess=%s postprocess=%s history=%s "
                  "depth_scatter=%ux%u data_graph=%ux%u dynamic_lut=%u lut=%ux%u disocclusion_mask=%u "
                  "compute_tensor=%u fragment_tensor=%u",
                  desc.renderSize.width,
                  desc.renderSize.height,
                  desc.upscaleSize.width,
                  desc.upscaleSize.height,
                  desc.displaySize.width,
                  desc.displaySize.height,
                  scaleX,
                  scaleY,
                  getQualityModeString(qualityMode),
                  desc.flags,
                  pipelineFlags,
                  preprocessStage,
                  postprocessStage,
                  manageHistory ? "managed" : "external",
                  context->depthScatterSize.width,
                  context->depthScatterSize.height,
                  context->dataGraphSize.width,
                  context->dataGraphSize.height,
                  static_cast<unsigned>(context->useDynamicOffsetLut),
                  context->offsetLutSize.width,
                  context->offsetLutSize.height,
                  static_cast<unsigned>(context->useDisocclusionMaskPass),
                  static_cast<unsigned>((pipelineFlags & NSS_SHADER_PERMUTATION_COMPUTE_SUPPORT_TENSOR) != 0),
                  static_cast<unsigned>((pipelineFlags & NSS_SHADER_PERMUTATION_FRAGMENT_SUPPORT_TENSOR) != 0));
    printMessage(context, FFX_MESSAGE_TYPE_INFO, message);

    if (std::fabs(scaleX - scaleY) > FLT_EPSILON)
    {
        printMessage(context, FFX_MESSAGE_TYPE_WARNING, "width/height has different upscale ratio");
    }

    if (scaleX - 2.0f > FLT_EPSILON)
    {
        printMessage(context, FFX_MESSAGE_TYPE_WARNING, "It's recommanded to use upscale ratio less than x2.");
    }

    if (!manageHistory)
    {
        printMessage(
            context, FFX_MESSAGE_TYPE_WARNING, "FFX_NSS_CONTEXT_FLAG_MANAGE_HISTORY is not set, the user must provide valid history texture each dispatch.");
    }
}

static FfxErrorCode nssCreate(FfxNssContext_Private* context, const FfxNssContextDescription* contextDescription)
{
    FFX_ASSERT(context);
    FFX_ASSERT(contextDescription);

    // Setup the data for implementation.
    memset(context, 0, sizeof(FfxNssContext_Private));
    context->device                = contextDescription->backendInterface.device;
    context->diagnostics.contextId = g_nextLogContextId.fetch_add(1, std::memory_order_relaxed);

    memcpy(&context->contextDescription, contextDescription, sizeof(FfxNssContextDescription));

    const FfxDimensions2D renderSize  = contextDescription->renderSize;
    const FfxDimensions2D displaySize = contextDescription->upscaleSize;

    // Create the context.
    FfxErrorCode errorCode = context->contextDescription.backendInterface.fpCreateBackendContext(
        &context->contextDescription.backendInterface, FFX_EFFECT_NSS, nullptr, &context->effectContextId);
    if (FFX_OK != errorCode)
    {
        printMessage(context, FFX_MESSAGE_TYPE_ERROR, "Failed to create backend interface for NSS context.");
        return errorCode;
    }
    context->backendContextCreated = true;

    // Register the message callback for backend
    errorCode = context->contextDescription.backendInterface.fpSetMessageCallback(&context->contextDescription.backendInterface, contextDescription->fpMessage);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    // call out for device caps.
    FfxDeviceCapabilities capabilities{};
    FFX_VALIDATE(context->contextDescription.backendInterface.fpGetDeviceCapabilities(&context->contextDescription.backendInterface, &capabilities));

    const bool neuralGraphicsSupported = capabilities.tensorSupported && capabilities.dataGraphSupported;
    if (!neuralGraphicsSupported)
    {
        context->unsupportedDevice = true;
        if (!g_unsupportedDeviceLogged.exchange(true, std::memory_order_relaxed))
        {
            char message[NSS_LOG_MESSAGE_CAPACITY];
            std::snprintf(message,
                          sizeof(message),
                          "event=device_unsupported stage=context_create error_code=%d error_name=%s "
                          "tensor_supported=%u data_graph_supported=%u reason_code=missing_neural_capability "
                          "reason=required_neural_features_unavailable fallback=application_required",
                          static_cast<FfxErrorCode>(FFX_ERROR_NULL_DEVICE),
                          ffxGetErrorCodeName(FFX_ERROR_NULL_DEVICE),
                          static_cast<unsigned>(capabilities.tensorSupported),
                          static_cast<unsigned>(capabilities.dataGraphSupported));
            printMessage(context, FFX_MESSAGE_TYPE_ERROR, message);
        }
        return FFX_ERROR_NULL_DEVICE;
    }

    if ((contextDescription->flags & FFX_NSS_CONTEXT_FLAG_POST_PROCESS_COMPUTE) && !capabilities.r11g11b10StorageImageSupported)
    {
        context->unsupportedDevice = true;
        printMessage(context,
                     FFX_MESSAGE_TYPE_ERROR,
                     "NSS compute postprocess requires optimal-tiling R11G11B10 storage-image support and shaderStorageImageExtendedFormats; "
                     "select the fragment postprocess or use a compatible Vulkan device.");
        return FFX_ERROR_INVALID_ARGUMENT;
    }

    // set defaults
    context->firstExecution     = true;
    context->resourceFrameIndex = 0;

    // Need to initialize the pipelineFlags early so we can check if fragment job will be enabled
    // for preprocess and postprocess and then setup the correct resource usage flag.
    context->pipelineFlags = getPipelinePermutationFlags(context);
    // We only need dynamic LUT for non-x2 case, x2 case will use static LUT.
    context->useDynamicOffsetLut = (context->pipelineFlags & NSS_SHADER_PERMUTATION_SCALE_PRESET_X2) == 0;
    const auto qualityModeValue =
        static_cast<FfxNssShaderQualityMode>((context->pipelineFlags >> NSS_SHADER_PERMUTATION_QUALITY_MODE_SHIFT) & NSS_SHADER_QUALITY_MODE_MASK);
    const bool isQualityMode = (qualityModeValue == FFX_NSS_SHADER_QUALITY_MODE_QUALITY);
    // We only need the disocclusion pass for mid/low quality
    context->useDisocclusionMaskPass = !isQualityMode;

    FfxSurfaceFormat tensorFormatSingleChannel = ((contextDescription->flags & FFX_NSS_CONTEXT_FLAG_QUANTIZED) == FFX_NSS_CONTEXT_FLAG_QUANTIZED)
                                                     ? FFX_SURFACE_FORMAT_R8_SINT
                                                     : FFX_SURFACE_FORMAT_R32_FLOAT;

    const bool                 preProcessUseFragment            = (context->pipelineFlags & NSS_SHADER_PERMUTATION_PRE_PROCESS_COMPUTE) == 0;
    const bool                 postProcessUseFragment           = (context->pipelineFlags & NSS_SHADER_PERMUTATION_POST_PROCESS_COMPUTE) == 0;
    const FfxResourceUsage     preProcessOutputImageUsage       = preProcessUseFragment ? FFX_RESOURCE_USAGE_RENDERTARGET : FFX_RESOURCE_USAGE_UAV;
    const FfxResourceUsage     disocclusionMaskOutputImageUsage = preProcessUseFragment ? FFX_RESOURCE_USAGE_RENDERTARGET : FFX_RESOURCE_USAGE_UAV;
    const FfxResourceUsage     postProcessOutputImageUsage      = postProcessUseFragment ? FFX_RESOURCE_USAGE_RENDERTARGET : FFX_RESOURCE_USAGE_UAV;
    const bool                 preprocessSupportTensor  = preProcessUseFragment ? capabilities.fragmentSupportTensor : capabilities.computeSupportTensor;
    const bool                 postprocessSupportTensor = postProcessUseFragment ? capabilities.fragmentSupportTensor : capabilities.computeSupportTensor;
    constexpr FfxResourceUsage tensorUsage              = FFX_RESOURCE_USAGE_UAV;

    // When preprocess doesn't support native tensor ops, the shader access the tensor via a buffer alias (std430 SSBO) or image alias.
    const FfxResourceFlags preprocessTensorFlags = preprocessSupportTensor ? FFX_RESOURCE_FLAGS_NONE : FFX_RESOURCE_FLAGS_BUFFER_ALIASED;
    const FfxResourceFlags kpnTensorFlags        = postprocessSupportTensor ? FFX_RESOURCE_FLAGS_NONE : FFX_RESOURCE_FLAGS_BUFFER_ALIASED;
    const FfxResourceFlags tensorResourceFlag    = FFX_RESOURCE_FLAGS_IMAGE_ALIASED;

    // Note the dimensions for each shader qualit mode should match the macros in ffx_nss_common_glsl.h
    // KPN coefficients tensor channel count depends on quality mode
    // TODO: we'd better query this from shape inference result.
    uint32_t         KpnTensorChannel  = 0;
    FfxSurfaceFormat depthOffsetFormat = FFX_SURFACE_FORMAT_UNKNOWN;
    if (isQualityMode)
    {
        context->depthScatterSize.width  = renderSize.width / 2;
        context->depthScatterSize.height = renderSize.height / 2;
        context->dataGraphSize.width     = FFX_ALIGN_UP(renderSize.width, FFX_NSS_RESOURCE_ALIGNMENT);
        context->dataGraphSize.height    = FFX_ALIGN_UP(renderSize.height, FFX_NSS_RESOURCE_ALIGNMENT);
        // QUALITY (NSS_USE_SPARSE_2X2_FILTER=0) → 36 channels (full 6×6 kernel)
        KpnTensorChannel = 36u;
        // QUALITY (NSS_PACKED_NEAREST_OFFSET_QUAD=0): single r8 unorm offset code.
        depthOffsetFormat = FFX_SURFACE_FORMAT_R8_UNORM;
    }
    else
    {
        context->depthScatterSize.width  = renderSize.width / 4;
        context->depthScatterSize.height = renderSize.height / 4;
        context->dataGraphSize.width     = FFX_ALIGN_UP(renderSize.width / 2, FFX_NSS_RESOURCE_ALIGNMENT);
        context->dataGraphSize.height    = FFX_ALIGN_UP(renderSize.height / 2, FFX_NSS_RESOURCE_ALIGNMENT);
        // BALANCE/PERFORMANCE (NSS_USE_SPARSE_2X2_FILTER=1) → 16 channels (sparse 2×2 kernel)
        KpnTensorChannel = 16u;
        // BALANCE/PERFORMANCE (NSS_PACKED_NEAREST_OFFSET_QUAD=1): four nibble offsets packed into rg8.
        depthOffsetFormat = FFX_SURFACE_FORMAT_R8G8_UNORM;
    }
    // KPN tensor is at 1/4 of the padded input (dataGraph) resolution.
    // TODO: we'd better query this from shape inference result.
    context->kpnDimension.width  = context->dataGraphSize.width / 4;
    context->kpnDimension.height = context->dataGraphSize.height / 4;

    if (context->useDynamicOffsetLut)
    {
        // The dynamic LUT size is T_H * (T_W * lutGroupsPerTile), where T_H/T_W come from
        // the reduced rational upscale factor displaySize/renderSize for each axis.
        const uint32_t gcdWidth  = computeGreatestCommonDivisor(displaySize.width, renderSize.width);
        const uint32_t gcdHeight = computeGreatestCommonDivisor(displaySize.height, renderSize.height);

        context->reducedFractionHrSize.width  = displaySize.width / gcdWidth;
        context->reducedFractionLrSize.width  = renderSize.width / gcdWidth;
        context->reducedFractionHrSize.height = displaySize.height / gcdHeight;
        context->reducedFractionLrSize.height = renderSize.height / gcdHeight;

        const uint32_t lutGroupsPerTile = offsetLutGroupsPerTile(context->pipelineFlags);
        context->offsetLutSize.width    = lutGroupsPerTile * context->reducedFractionHrSize.width;
        context->offsetLutSize.height   = context->reducedFractionHrSize.height;
    }
    else
    {
        // For x2 case
        context->reducedFractionHrSize.width  = 2;
        context->reducedFractionHrSize.height = 2;

        context->reducedFractionLrSize.width  = 1;
        context->reducedFractionLrSize.height = 1;
    }

    constexpr uint32_t inputTensorChannel    = 12;
    constexpr uint32_t feedbackTensorChannel = 4;
    // declare internal resources needed
    const FfxInternalResourceDescription PersistentInternalSurfaceDesc[] = {
        {FFX_NSS_RESOURCE_IDENTIFIER_PREPROCESS_INPUT_TENSOR,
         "NSS_PreprocessInputTensor",
         FFX_RESOURCE_TYPE_TENSOR,
         tensorUsage,
         tensorFormatSingleChannel,
         context->dataGraphSize.width,
         context->dataGraphSize.height,
         1,
         preprocessTensorFlags,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED},
         1,
         inputTensorChannel,
         4},

        {FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_1,
         "NSS_LumaDeriv_1",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         preProcessOutputImageUsage,
         FFX_SURFACE_FORMAT_R8G8B8A8_SNORM,
         context->dataGraphSize.width,
         context->dataGraphSize.height,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},

        {FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_2,
         "NSS_LumaDeriv_2",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         preProcessOutputImageUsage,
         FFX_SURFACE_FORMAT_R8G8B8A8_SNORM,
         context->dataGraphSize.width,
         context->dataGraphSize.height,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},

        {FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD,
         "NSS_DepthOffset",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         preProcessOutputImageUsage,
         depthOffsetFormat,
         context->dataGraphSize.width,
         context->dataGraphSize.height,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},

        {FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR,
         "NSS_FeedbackTensor",
         FFX_RESOURCE_TYPE_TENSOR,
         tensorUsage,
         tensorFormatSingleChannel,
         context->dataGraphSize.width,
         context->dataGraphSize.height,
         1,
         tensorResourceFlag,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED},
         1,
         feedbackTensorChannel,
         4},

        {FFX_NSS_RESOURCE_IDENTIFIER_KPN_TENSOR,
         "NSS_KpnCoefficientsTensor",
         FFX_RESOURCE_TYPE_TENSOR,
         tensorUsage,
         tensorFormatSingleChannel,
         context->kpnDimension.width,
         context->kpnDimension.height,
         1,
         kpnTensorFlags,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED},
         1,
         KpnTensorChannel,
         4},
        {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH_TM1,
         "NSS_reconstructed_prev_depth",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         FFX_RESOURCE_USAGE_UAV,
         FFX_SURFACE_FORMAT_R32_UINT,
         context->depthScatterSize.width,
         context->depthScatterSize.height,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
    };

    const bool manageHistory = (context->contextDescription.flags & FFX_NSS_CONTEXT_FLAG_MANAGE_HISTORY) != 0;

    // Persistent internal surfaces and
    // - optional dynamic LUT if context->useDynamicOffsetLut is true.
    // - optional disocclusion mask if context->useDisocclusionMaskPass is true.
    // - 2 optional history as ring buffer if manageHistory is enabled.
    constexpr uint32_t             MAX_INTERNAL_SURFACE_COUNT                      = FFX_COUNTOF(PersistentInternalSurfaceDesc) + 4;
    FfxInternalResourceDescription internalSurfaceDesc[MAX_INTERNAL_SURFACE_COUNT] = {};
    uint32_t                       internalSurfaceCount                            = 0;

    const auto addInternalSurface = [&](const FfxInternalResourceDescription& desc) {
        FFX_ASSERT(internalSurfaceCount < MAX_INTERNAL_SURFACE_COUNT);
        internalSurfaceDesc[internalSurfaceCount++] = desc;
    };

    for (uint32_t i = 0; i < FFX_ARRAY_ELEMENTS(PersistentInternalSurfaceDesc); ++i)
    {
        addInternalSurface(PersistentInternalSurfaceDesc[i]);
    }

    if (context->useDynamicOffsetLut)
    {
        addInternalSurface({FFX_NSS_RESOURCE_IDENTIFIER_OFFSET_LUT,
                            "NSS_OffsetLut",
                            FFX_RESOURCE_TYPE_TEXTURE2D,
                            FFX_RESOURCE_USAGE_UAV,
                            FFX_SURFACE_FORMAT_R32G32B32A32_UINT,
                            context->offsetLutSize.width,
                            context->offsetLutSize.height,
                            1,
                            FFX_RESOURCE_FLAGS_NONE,
                            {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}});
    }

    if (context->useDisocclusionMaskPass)
    {
        addInternalSurface({FFX_NSS_RESOURCE_IDENTIFIER_DISOCCLUSION_MASK_LQ,
                            "NSS_disocclusion_mask",
                            FFX_RESOURCE_TYPE_TEXTURE2D,
                            disocclusionMaskOutputImageUsage,
                            FFX_SURFACE_FORMAT_R8_UNORM,
                            context->depthScatterSize.width,
                            context->depthScatterSize.height,
                            1,
                            FFX_RESOURCE_FLAGS_NONE,
                            {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}});
    }

    if (manageHistory)
    {
        addInternalSurface({FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_1,
                            "NSS_upscaled_color_1",
                            FFX_RESOURCE_TYPE_TEXTURE2D,
                            postProcessOutputImageUsage,
                            FFX_SURFACE_FORMAT_R11G11B10_FLOAT,
                            displaySize.width,
                            displaySize.height,
                            1,
                            FFX_RESOURCE_FLAGS_NONE,
                            {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}});
        addInternalSurface({FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_2,
                            "NSS_upscaled_color_2",
                            FFX_RESOURCE_TYPE_TEXTURE2D,
                            postProcessOutputImageUsage,
                            FFX_SURFACE_FORMAT_R11G11B10_FLOAT,
                            displaySize.width,
                            displaySize.height,
                            1,
                            FFX_RESOURCE_FLAGS_NONE,
                            {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}});
    }

    // clear the SRV resources to NULL.
    memset(context->srvResources, 0, sizeof(context->srvResources));

    // Generally used resources by all presets
    for (uint32_t currentSurfaceIndex = 0; currentSurfaceIndex < internalSurfaceCount; ++currentSurfaceIndex)
    {
        FFX_VALIDATE(createResourceFromDescription(context, &internalSurfaceDesc[currentSurfaceIndex]));
    }

    // copy resources to uavResrouces list
    memcpy(context->uavResources, context->srvResources, sizeof(context->srvResources));

    // The external resources should not be created internally, just registered.
    for (uint32_t i = 0; i < FFX_COUNTOF(externalCommonResources); ++i)
    {
        constexpr FfxResourceInternal emptyResource = {};
        FFX_ASSERT(memcmp(&context->srvResources[externalCommonResources[i]], &emptyResource, sizeof(FfxResourceInternal)) == 0);
        FFX_ASSERT(memcmp(&context->uavResources[externalCommonResources[i]], &emptyResource, sizeof(FfxResourceInternal)) == 0);
    }

    if (!manageHistory)
    {
        for (uint32_t i = 0; i < FFX_COUNTOF(externalCondtinalResources); ++i)
        {
            constexpr FfxResourceInternal emptyResource = {};
            FFX_ASSERT(memcmp(&context->srvResources[externalCondtinalResources[i]], &emptyResource, sizeof(FfxResourceInternal)) == 0);
            FFX_ASSERT(memcmp(&context->uavResources[externalCondtinalResources[i]], &emptyResource, sizeof(FfxResourceInternal)) == 0);
        }
    }

    constexpr uint32_t     MAX_DATA_GRAPH_TENSOR_INFO                      = FFX_COUNTOF(srvTensorBindingTable) + FFX_COUNTOF(uavTensorBindingTable);
    FfxDataGraphTensorInfo dataGraphTensorInfo[MAX_DATA_GRAPH_TENSOR_INFO] = {};
    uint32_t               dataGraphTensorInfoCount                        = 0;
    FFX_VALIDATE(
        buildDataGraphTensorInfo(internalSurfaceDesc, internalSurfaceCount, dataGraphTensorInfo, MAX_DATA_GRAPH_TENSOR_INFO, &dataGraphTensorInfoCount));

    // avoid compiling pipelines on first render
    {
        errorCode = createPipelineStates(context, dataGraphTensorInfo, dataGraphTensorInfoCount);
        FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);
    }
    return FFX_OK;
}

static FfxErrorCode nssRelease(FfxNssContext_Private* context)
{
    FFX_ASSERT(context);

    releaseJitterAnalyzer(context);

    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineNssDepthScatter, context->effectContextId);
    if (context->useDisocclusionMaskPass)
    {
        ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineNssDisocclusionMask, context->effectContextId);
    }
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineNssPreprocess, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineNssDataGraph, context->effectContextId);
    if (context->useDynamicOffsetLut)
    {
        ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineNssGenerateOffsetLut, context->effectContextId);
    }
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineNssPostprocess, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineNssDebugView, context->effectContextId);

    // unregister resources not created internally
    for (uint32_t i = 0; i < FFX_COUNTOF(externalCommonResources); ++i)
    {
        context->srvResources[externalCommonResources[i]] = {FFX_NSS_RESOURCE_IDENTIFIER_NULL};
        context->uavResources[externalCommonResources[i]] = {FFX_NSS_RESOURCE_IDENTIFIER_NULL};
    }

    const bool manageHistory = (context->contextDescription.flags & FFX_NSS_CONTEXT_FLAG_MANAGE_HISTORY) != 0;
    if (!manageHistory)
    {
        for (uint32_t i = 0; i < FFX_COUNTOF(externalCondtinalResources); ++i)
        {
            context->srvResources[externalCondtinalResources[i]] = {FFX_NSS_RESOURCE_IDENTIFIER_NULL};
            context->uavResources[externalCondtinalResources[i]] = {FFX_NSS_RESOURCE_IDENTIFIER_NULL};
        }
    }

    // INPUT_EXPOSURE is caller-owned and must not be released as an internal resource.
    context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_INPUT_EXPOSURE] = {FFX_NSS_RESOURCE_IDENTIFIER_NULL};

    // release internal resources
    for (int32_t currentResourceIndex = 0; currentResourceIndex < FFX_NSS_RESOURCE_IDENTIFIER_COUNT; ++currentResourceIndex)
    {
        ffxSafeReleaseResource(&context->contextDescription.backendInterface, context->srvResources[currentResourceIndex], context->effectContextId);
    }

    // Destroy the context
    FFX_VALIDATE(context->contextDescription.backendInterface.fpDestroyBackendContext(&context->contextDescription.backendInterface, context->effectContextId));
    context->backendContextCreated = false;

    return FFX_OK;
}

static void setupDeviceDepthToViewSpaceDepthParams(FfxNssContext_Private* context, const FfxNssDispatchDescription* params)
{
    const bool bInverted = (context->contextDescription.flags & FFX_NSS_CONTEXT_FLAG_DEPTH_INVERTED) == FFX_NSS_CONTEXT_FLAG_DEPTH_INVERTED;
    const bool bInfinite = (context->contextDescription.flags & FFX_NSS_CONTEXT_FLAG_DEPTH_INFINITE) == FFX_NSS_CONTEXT_FLAG_DEPTH_INFINITE;

    // make sure it has no impact if near and far plane values are swapped in dispatch params
    // the flags "inverted" and "infinite" will decide what transform to use
    float fMin = FFX_MINIMUM(params->cameraNear, params->cameraFar);
    float fMax = FFX_MAXIMUM(params->cameraNear, params->cameraFar);

    if (bInverted)
    {
        float tmp = fMin;
        fMin      = fMax;
        fMax      = tmp;
    }

    // a 0 0 0   x
    // 0 b 0 0   y
    // 0 0 c d   z
    // 0 0 e 0   1

    const float fQ = fMax / (fMin - fMax);
    const float d  = -1.0f;  // for clarity

    const float matrix_elem_c[2][2] = {
        fQ,                   // non reversed, non infinite
        -1.0f - FLT_EPSILON,  // non reversed, infinite
        fQ,                   // reversed, non infinite
        0.0f + FLT_EPSILON    // reversed, infinite
    };

    const float matrix_elem_e[2][2] = {
        fQ * fMin,            // non reversed, non infinite
        -fMin - FLT_EPSILON,  // non reversed, infinite
        fQ * fMin,            // reversed, non infinite
        fMax,                 // reversed, infinite
    };

    context->constants._DeviceToViewDepth[0] = d * matrix_elem_c[bInverted][bInfinite];
    context->constants._DeviceToViewDepth[1] = matrix_elem_e[bInverted][bInfinite];

    // revert x and y coords
    const float aspect      = params->renderSize.width / float(params->renderSize.height);
    const float cotHalfFovY = cosf(0.5f * params->cameraFovAngleVertical) / sinf(0.5f * params->cameraFovAngleVertical);
    const float a           = cotHalfFovY / aspect;
    const float b           = cotHalfFovY;

    context->constants._DeviceToViewDepth[2] = (1.0f / a);
    context->constants._DeviceToViewDepth[3] = (1.0f / b);
}

static void computeJitterTileOffset(float jx, float jy, float sx, float sy, int mx, int my, FfxInt32x2& jitterTileOffset)
{
    // Project base and jittered LR pixel centers to HR index space
    int base_hr_x = static_cast<int>(std::floor(0.5f * sx));
    int base_hr_y = static_cast<int>(std::floor(0.5f * sy));

    int jittered_hr_x = static_cast<int>(std::floor((jx + 0.5f) * sx));
    int jittered_hr_y = static_cast<int>(std::floor((jy + 0.5f) * sy));

    int dx_offset = (jittered_hr_x - base_hr_x) % mx;
    if (dx_offset < 0)
        dx_offset += mx;
    int dy_offset = (jittered_hr_y - base_hr_y) % my;
    if (dy_offset < 0)
        dy_offset += my;

    jitterTileOffset[0] = dx_offset;
    jitterTileOffset[1] = dy_offset;
}

static bool NeedResetHistory(FfxNssContext_Private* context, const FfxNssDispatchDescription* params)
{
    /* Assume the application accounts for scenarios requiring a reset (eg. due to changes in render size,
	 * camera transitions, etc.), and correctly configures this through the reset parameter.
	 */
    return context->firstExecution || params->reset;
}

static float ComputeDepthClipRequiredSepScale(NssConstants& constants, const FfxNssDispatchDescription* params)
{
    FFX_ASSERT(params);
    const float half_viewport = sqrtf(float(params->renderSize.width * params->renderSize.width + params->renderSize.height * params->renderSize.height));
    const float cornerViewDirLen =
        sqrtf(constants._DeviceToViewDepth[2] * constants._DeviceToViewDepth[2] + constants._DeviceToViewDepth[3] * constants._DeviceToViewDepth[3] + 1.0f);

    return 1.37e-05f * cornerViewDirLen * half_viewport;
}

static float ComputeDepthClipPower(const FfxNssDispatchDescription* params)
{
    FFX_ASSERT(params);
    // Using the 1080x1920 as the reference viewport, where sqrt(1080^2 + 1920^2) = 2202.9071700822983f
    constexpr float REFERENCE_VIEWPORT_LENGTH = 2202.9071700822983f;
    const float     half_viewport = sqrtf(float(params->renderSize.width * params->renderSize.width + params->renderSize.height * params->renderSize.height));
    const float     resolutionFactor = ffxMin(ffxMax(half_viewport / REFERENCE_VIEWPORT_LENGTH, 0.0f), 1.0f);

    return 1.0f + (2.0f * resolutionFactor);
}

void UpdateNssJitterConstants(NssConstants& constants, const FfxNssDispatchDescription& params, bool resetHistory, bool cancelMotionJitter)
{
    const float jitter[4] = {params.jitterOffset.x,
                             params.jitterOffset.y,
                             params.jitterOffset.x / static_cast<float>(params.renderSize.width),
                             params.jitterOffset.y / static_cast<float>(params.renderSize.height)};
    for (uint32_t i = 0; i < 4; ++i)
    {
        constants._JitterOffsetTm1[i] = resetHistory ? jitter[i] : constants._JitterOffset[i];
        constants._JitterOffset[i]    = jitter[i];
    }

    // NSS samples at pixel + jitter, so raster positions are p - jitter.
    // Backward jittered motion therefore contains currentJitter - previousJitter.
    for (uint32_t i = 0; i < 2; ++i)
    {
        constants._JitterCancellation[i] = cancelMotionJitter ? constants._JitterOffset[i] - constants._JitterOffsetTm1[i] : 0.0f;
    }
}

static FfxErrorCode setupConstantBuffer(FfxNssContext_Private* context, const FfxNssDispatchDescription* params)
{
    NssConstants& constants        = context->constants;
    const bool    needResetHistory = NeedResetHistory(context, params);
    if (needResetHistory)
    {
        memset(&constants, 0, sizeof(constants));
    }

    // compute params to enable device depth to view space depth computation in shader
    setupDeviceDepthToViewSpaceDepthParams(context, params);

    constants._InputDims[0]    = params->renderSize.width;
    constants._InputDims[1]    = params->renderSize.height;
    constants._InvInputDims[0] = 1.f / static_cast<float>(params->renderSize.width);
    constants._InvInputDims[1] = 1.f / static_cast<float>(params->renderSize.height);

    constants._OutputDims[0]    = params->upscaleSize.width;
    constants._OutputDims[1]    = params->upscaleSize.height;
    constants._InvOutputDims[0] = 1.f / static_cast<float>(params->upscaleSize.width);
    constants._InvOutputDims[1] = 1.f / static_cast<float>(params->upscaleSize.height);

    constants._InputTensorSize[0]    = context->dataGraphSize.width;
    constants._InputTensorSize[1]    = context->dataGraphSize.height;
    constants._InputTensorSizeRcp[0] = 1.0f / static_cast<float>(context->dataGraphSize.width);
    constants._InputTensorSizeRcp[1] = 1.0f / static_cast<float>(context->dataGraphSize.height);

    UpdateNssJitterConstants(
        constants, *params, needResetHistory, (context->contextDescription.flags & FFX_NSS_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) != 0);

    // Setup scale factor (.xy scale, .zw inv_scale)
    constants._ScaleFactor[0] = constants._OutputDims[0] / FfxFloat32(constants._InputDims[0]);
    constants._ScaleFactor[1] = constants._OutputDims[1] / FfxFloat32(constants._InputDims[1]);
    constants._ScaleFactor[2] = constants._InputDims[0] / FfxFloat32(constants._OutputDims[0]);
    constants._ScaleFactor[3] = constants._InputDims[1] / FfxFloat32(constants._OutputDims[1]);

    // Setup motion vector scale.
    constants._MotionVectorScale[0] = params->motionVectorScale.x;
    constants._MotionVectorScale[1] = params->motionVectorScale.y;

    constants._PaddingScale[0] = constants._InputDims[0] / static_cast<float>(context->dataGraphSize.width);
    constants._PaddingScale[1] = constants._InputDims[1] / static_cast<float>(context->dataGraphSize.height);

    constants._DepthTm1Size[0]    = context->depthScatterSize.width;
    constants._DepthTm1Size[1]    = context->depthScatterSize.height;
    constants._InvDepthTm1Size[0] = 1.f / static_cast<float>(context->depthScatterSize.width);
    constants._InvDepthTm1Size[1] = 1.f / static_cast<float>(context->depthScatterSize.height);

    constants._KpnDimension[0] = context->kpnDimension.width;
    constants._KpnDimension[1] = context->kpnDimension.height;

    constants._DepthClipRequiredSepScale = ComputeDepthClipRequiredSepScale(constants, params);
    constants._DepthClipPower            = ComputeDepthClipPower(params);

    // KPN scale for QUALITY mode tap-selection: kpnDims / paddedDims (= dataGraphSize)
    constants._KpnScale[0] = (float)context->kpnDimension.width / (float)context->dataGraphSize.width;
    constants._KpnScale[1] = (float)context->kpnDimension.height / (float)context->dataGraphSize.height;

    constants._DebugViewMode =
        std::clamp(params->debugViewMode, static_cast<uint32_t>(NSS_DEBUG_VIEW_MODE_ALL), static_cast<uint32_t>(NSS_DEBUG_VIEW_MODE_TENSOR_WARP_FEEDBACK));

    const float noHistoryReset = needResetHistory ? 0.0f : 1.0f;
    const float exposure       = (params->exposure <= 0.0f) ? float(std::exp(2.0)) : params->exposure;
    const float invExposure    = 1.f / exposure;

    const FfxUInt32x2 indexModulo        = {context->reducedFractionHrSize.width, context->reducedFractionHrSize.height};
    const FfxUInt32x2 reducedInputModulo = {context->reducedFractionLrSize.width, context->reducedFractionLrSize.height};

    FfxInt32x2 jitterTileOffset = {};
    if (!context->useDynamicOffsetLut)
    {
        computeJitterTileOffset(constants._JitterOffset[0],
                                constants._JitterOffset[1],
                                constants._ScaleFactor[0],
                                constants._ScaleFactor[1],
                                indexModulo[0],
                                indexModulo[1],
                                jitterTileOffset);
    }

    constants._Exposure[0]           = exposure;
    constants._Exposure[1]           = invExposure;
    constants._IndexModulo[0]        = indexModulo[0];
    constants._IndexModulo[1]        = indexModulo[1];
    constants._ReducedInputModulo[0] = reducedInputModulo[0];
    constants._ReducedInputModulo[1] = reducedInputModulo[1];
    constants._LutOffset[0]          = jitterTileOffset[0];
    constants._LutOffset[1]          = jitterTileOffset[1];
    constants._NotHistoryReset       = noHistoryReset;

    // initialize constantBuffers data
    return context->contextDescription.backendInterface.fpStageConstantBufferDataFunc(&context->contextDescription.backendInterface,
                                                                                      &context->constants,
                                                                                      sizeof(context->constants),
                                                                                      &context->constantBuffers[FFX_NSS_CONSTANTBUFFER_IDENTIFIER_NSS]);
}

static FfxErrorCode scheduleDispatch(FfxNssContext_Private*           context,
                                     const FfxNssDispatchDescription* params,
                                     const FfxPipelineState*          pipeline,
                                     uint32_t                         dispatchX,
                                     uint32_t                         dispatchY,
                                     const char*                      debugName)
{
    FfxGpuJobDescription dispatchJob = {FFX_GPU_JOB_COMPUTE};
    if (debugName != nullptr)
    {
#ifdef FFX_DEBUG
        strncpy(dispatchJob.jobLabel, debugName, FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.jobLabel[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvTextureCount; ++currentShaderResourceViewIndex)
    {
        const uint32_t            currentResourceId = pipeline->srvTextureBindings[currentShaderResourceViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource   = context->srvResources[currentResourceId];
        dispatchJob.computeJobDescriptor.srvTextures[currentShaderResourceViewIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(dispatchJob.computeJobDescriptor.srvTextures[currentShaderResourceViewIndex].name,
                pipeline->srvTextureBindings[currentShaderResourceViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.computeJobDescriptor.srvTextures[currentShaderResourceViewIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentUnorderedAccessViewIndex = 0; currentUnorderedAccessViewIndex < pipeline->uavTextureCount; ++currentUnorderedAccessViewIndex)
    {
        const uint32_t currentResourceId = pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].resourceIdentifier;
#ifdef FFX_DEBUG
        strncpy(dispatchJob.computeJobDescriptor.uavTextureNames[currentUnorderedAccessViewIndex],
                pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.computeJobDescriptor.uavTextureNames[currentUnorderedAccessViewIndex][FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif

        {
            const FfxResourceInternal currentResource                                              = context->uavResources[currentResourceId];
            dispatchJob.computeJobDescriptor.uavTextures[currentUnorderedAccessViewIndex].resource = currentResource;
            dispatchJob.computeJobDescriptor.uavTextures[currentUnorderedAccessViewIndex].mip      = 0;
        }
    }

    // SRV buffers
    for (uint32_t index = 0; index < pipeline->srvBufferCount; ++index)
    {
        const uint32_t            currentResourceId                 = pipeline->srvBufferBindings[index].resourceIdentifier;
        const FfxResourceInternal currentResource                   = context->srvResources[currentResourceId];
        dispatchJob.computeJobDescriptor.srvBuffers[index].resource = currentResource;
    }

    // UAV buffers
    for (uint32_t index = 0; index < pipeline->uavBufferCount; ++index)
    {
        const uint32_t            currentResourceId                 = pipeline->uavBufferBindings[index].resourceIdentifier;
        const FfxResourceInternal currentResource                   = context->uavResources[currentResourceId];
        dispatchJob.computeJobDescriptor.uavBuffers[index].resource = currentResource;
    }

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->srvTensorCount; ++currentTensorIndex)
    {
        const uint32_t            currentResourceId                              = pipeline->srvTensorBindings[currentTensorIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                                = context->srvResources[currentResourceId];
        dispatchJob.computeJobDescriptor.srvTensors[currentTensorIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(dispatchJob.computeJobDescriptor.srvTensors[currentTensorIndex].name,
                pipeline->srvTensorBindings[currentTensorIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.computeJobDescriptor.srvTensors[currentTensorIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->uavTensorCount; ++currentTensorIndex)
    {
        const uint32_t            currentResourceId                              = pipeline->uavTensorBindings[currentTensorIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                                = context->uavResources[currentResourceId];
        dispatchJob.computeJobDescriptor.uavTensors[currentTensorIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(dispatchJob.computeJobDescriptor.uavTensors[currentTensorIndex].name,
                pipeline->uavTensorBindings[currentTensorIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.computeJobDescriptor.uavTensors[currentTensorIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    dispatchJob.computeJobDescriptor.dimensions[0] = dispatchX;
    dispatchJob.computeJobDescriptor.dimensions[1] = dispatchY;
    dispatchJob.computeJobDescriptor.dimensions[2] = 1;
    dispatchJob.computeJobDescriptor.pipeline      = *pipeline;

    for (uint32_t currentRootConstantIndex = 0; currentRootConstantIndex < pipeline->constCount; ++currentRootConstantIndex)
    {
#ifdef FFX_DEBUG
        strncpy(dispatchJob.computeJobDescriptor.cbNames[currentRootConstantIndex],
                pipeline->constantBufferBindings[currentRootConstantIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.computeJobDescriptor.cbNames[currentRootConstantIndex][FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
        dispatchJob.computeJobDescriptor.cbs[currentRootConstantIndex] =
            context->constantBuffers[pipeline->constantBufferBindings[currentRootConstantIndex].resourceIdentifier];
    }

    return context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &dispatchJob);
}

static FfxErrorCode scheduleFragment(FfxNssContext_Private*           context,
                                     const FfxNssDispatchDescription* params,
                                     const FfxPipelineState*          pipeline,
                                     uint32_t                         width,
                                     uint32_t                         height,
                                     const char*                      debugName)
{
    FFX_UNUSED(params);

    FfxGpuJobDescription fragmentJob = {FFX_GPU_JOB_FRAGMENT};
    if (debugName != nullptr)
    {
#ifdef FFX_DEBUG
        strncpy(fragmentJob.jobLabel, debugName, FFX_RESOURCE_NAME_SIZE - 1);
        fragmentJob.jobLabel[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvTextureCount; ++currentShaderResourceViewIndex)
    {
        const uint32_t            currentResourceId = pipeline->srvTextureBindings[currentShaderResourceViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource   = context->srvResources[currentResourceId];
        fragmentJob.fragmentJobDescriptor.srvTextures[currentShaderResourceViewIndex].resource = currentResource;
    }

    for (uint32_t currentUnorderedAccessViewIndex = 0; currentUnorderedAccessViewIndex < pipeline->uavTextureCount; ++currentUnorderedAccessViewIndex)
    {
        const uint32_t            currentResourceId = pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource   = context->uavResources[currentResourceId];
        fragmentJob.fragmentJobDescriptor.uavTextures[currentUnorderedAccessViewIndex].resource = currentResource;
        fragmentJob.fragmentJobDescriptor.uavTextures[currentUnorderedAccessViewIndex].mip      = 0;
    }

    // SRV buffers
    for (uint32_t index = 0; index < pipeline->srvBufferCount; ++index)
    {
        const uint32_t            currentResourceId                  = pipeline->srvBufferBindings[index].resourceIdentifier;
        const FfxResourceInternal currentResource                    = context->srvResources[currentResourceId];
        fragmentJob.fragmentJobDescriptor.srvBuffers[index].resource = currentResource;
    }

    // UAV buffers
    for (uint32_t index = 0; index < pipeline->uavBufferCount; ++index)
    {
        const uint32_t            currentResourceId                  = pipeline->uavBufferBindings[index].resourceIdentifier;
        const FfxResourceInternal currentResource                    = context->uavResources[currentResourceId];
        fragmentJob.fragmentJobDescriptor.uavBuffers[index].resource = currentResource;
    }

    for (uint32_t currentRenderTargetIndex = 0; currentRenderTargetIndex < pipeline->rtCount; ++currentRenderTargetIndex)
    {
        const uint32_t            currentResourceId                                     = pipeline->rtBindings[currentRenderTargetIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                                       = context->uavResources[currentResourceId];
        fragmentJob.fragmentJobDescriptor.rtTextures[currentRenderTargetIndex].resource = currentResource;
    }

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->srvTensorCount; ++currentTensorIndex)
    {
        const uint32_t            currentResourceId                               = pipeline->srvTensorBindings[currentTensorIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                                 = context->srvResources[currentResourceId];
        fragmentJob.fragmentJobDescriptor.srvTensors[currentTensorIndex].resource = currentResource;
    }

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->uavTensorCount; ++currentTensorIndex)
    {
        const uint32_t            currentResourceId                               = pipeline->uavTensorBindings[currentTensorIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                                 = context->uavResources[currentResourceId];
        fragmentJob.fragmentJobDescriptor.uavTensors[currentTensorIndex].resource = currentResource;
    }

    fragmentJob.fragmentJobDescriptor.viewport[0] = width;
    fragmentJob.fragmentJobDescriptor.viewport[1] = height;
    fragmentJob.fragmentJobDescriptor.pipeline    = *pipeline;

    for (uint32_t currentRootConstantIndex = 0; currentRootConstantIndex < pipeline->constCount; ++currentRootConstantIndex)
    {
#ifdef FFX_DEBUG
        strncpy(fragmentJob.fragmentJobDescriptor.cbNames[currentRootConstantIndex],
                pipeline->constantBufferBindings[currentRootConstantIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        fragmentJob.fragmentJobDescriptor.cbNames[currentRootConstantIndex][FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
        fragmentJob.fragmentJobDescriptor.cbs[currentRootConstantIndex] =
            context->constantBuffers[pipeline->constantBufferBindings[currentRootConstantIndex].resourceIdentifier];
    }

    return context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &fragmentJob);
}

static FfxErrorCode scheduleDataGraph(FfxNssContext_Private*           context,
                                      const FfxNssDispatchDescription* params,
                                      FfxPipelineState*                pipeline,
                                      const char*                      debugName)
{
    FfxGpuJobDescription dataGraphJob = {FFX_GPU_JOB_DATA_GRAPH};
    if (debugName != nullptr)
    {
#ifdef FFX_DEBUG
        strncpy(dataGraphJob.jobLabel, debugName, FFX_RESOURCE_NAME_SIZE - 1);
        dataGraphJob.jobLabel[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->srvTensorCount; ++currentTensorIndex)
    {
        const uint32_t            currentResourceId                                  = pipeline->srvTensorBindings[currentTensorIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                                    = context->srvResources[currentResourceId];
        dataGraphJob.dataGraphJobDescription.srvTensors[currentTensorIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(dataGraphJob.dataGraphJobDescription.srvTensors[currentTensorIndex].name,
                pipeline->srvTensorBindings[currentTensorIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dataGraphJob.dataGraphJobDescription.srvTensors[currentTensorIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->uavTensorCount; ++currentTensorIndex)
    {
        const uint32_t            currentResourceId                                  = pipeline->uavTensorBindings[currentTensorIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                                    = context->uavResources[currentResourceId];
        dataGraphJob.dataGraphJobDescription.uavTensors[currentTensorIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(dataGraphJob.dataGraphJobDescription.uavTensors[currentTensorIndex].name,
                pipeline->uavTensorBindings[currentTensorIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dataGraphJob.dataGraphJobDescription.uavTensors[currentTensorIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    dataGraphJob.dataGraphJobDescription.pipeline = *pipeline;

    return context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &dataGraphJob);
}

/// @cond
struct NssDispatchBackendScope
{
    FfxNssContext_Private* context;
    FfxCommandList         commandList;
    bool                   cleanupAttempted = false;
    bool                   completed        = false;

    ~NssDispatchBackendScope() noexcept
    {
        if (completed)
        {
            return;
        }

        context->requiresRecreation = true;
        if (!cleanupAttempted)
        {
            const FfxErrorCode cleanupError = context->contextDescription.backendInterface.fpUnregisterResources(
                &context->contextDescription.backendInterface, commandList, context->effectContextId);
            if (cleanupError != FFX_OK)
            {
                logErrorWithCallback(cleanupError, "failed to abort NSS dispatch", context->contextDescription.fpMessage);
            }
        }
    }

    FfxErrorCode Complete() noexcept
    {
        cleanupAttempted                = true;
        const FfxErrorCode cleanupError = context->contextDescription.backendInterface.fpUnregisterResources(
            &context->contextDescription.backendInterface, commandList, context->effectContextId);
        completed = cleanupError == FFX_OK;
        return cleanupError;
    }
};
/// @endcond

static FfxErrorCode nssDispatch(FfxNssContext_Private* context, const FfxNssDispatchDescription* params, NssDispatchTrace& trace)
{
    FFX_ASSERT(context);
    FFX_ASSERT(params);
    trace.stage = "setup";

    if ((params->flags & FFX_NSS_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING) == FFX_NSS_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING)
    {
        nssDebugCheckDispatch(context, params);
    }
    else
    {
        releaseJitterAnalyzer(context);
    }

    // validate the required textures are provided
    FFX_RETURN_ON_ERROR(params->color.resource != NULL, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(params->depth.resource != NULL, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(params->motionVectors.resource != NULL, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(params->output.resource != NULL, FFX_ERROR_INVALID_POINTER);
    const bool hasDebugView = (params->flags & FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW) == FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW;
    if (hasDebugView)
    {
        FFX_RETURN_ON_ERROR(params->debugViews.resource != NULL, FFX_ERROR_INVALID_POINTER);
    }

    const bool manageHistory = (context->contextDescription.flags & FFX_NSS_CONTEXT_FLAG_MANAGE_HISTORY) != 0;
    if (!manageHistory)
    {
        FFX_RETURN_ON_ERROR(params->outputTm1.resource != NULL, FFX_ERROR_INVALID_POINTER);
    }

    // When the caller opts into reading exposure from a texture, a valid texture is required each dispatch.
    const bool useExposureTexture = (context->contextDescription.flags & FFX_NSS_CONTEXT_FLAG_EXPOSURE_TEXTURE) != 0;
    if (useExposureTexture)
    {
        FFX_RETURN_ON_ERROR(params->exposureTexture.resource != NULL, FFX_ERROR_INVALID_POINTER);
    }

    if ((params->flags & FFX_NSS_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING) == FFX_NSS_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING)
    {
        analyzeJitterDispatch(context, params);
    }

    // take a short cut to the command list
    FfxCommandList          commandList = params->commandList;
    NssDispatchBackendScope backendScope{context, commandList};

    FfxGpuJobDescription clearJob = {FFX_GPU_JOB_CLEAR_FLOAT};

    const float clearValuesToZeroFloat[]{0.f, 0.f, 0.f, 0.f};
    memcpy(clearJob.clearJobDescriptor.color, clearValuesToZeroFloat, 4 * sizeof(float));

    const bool resetAccumulation = NeedResetHistory(context, params);
    trace.stage                  = "reset_history";
    if (resetAccumulation)
    {
        constexpr uint32_t always_clear_resources[] = {FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_1,
                                                       FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_2,
                                                       FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD,
                                                       FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR};

        for (uint32_t i = 0; i < FFX_COUNTOF(always_clear_resources); ++i)
        {
            clearJob.clearJobDescriptor.target = context->srvResources[always_clear_resources[i]];
            FFX_VALIDATE(context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob));
        }

        if (manageHistory)
        {
            constexpr uint32_t history_resources[] = {FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_1, FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_2};
            for (uint32_t i = 0; i < FFX_COUNTOF(history_resources); ++i)
            {
                clearJob.clearJobDescriptor.target = context->srvResources[history_resources[i]];
                FFX_VALIDATE(context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob));
            }
        }
    }

    // Prepare per frame descriptor tables
    const bool isOddFrame = !!(context->resourceFrameIndex & 1);

    const uint32_t lumaDerivSrvResourceIndex = isOddFrame ? FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_2 : FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_1;
    const uint32_t lumaDerivUavResourceIndex = isOddFrame ? FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_1 : FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_2;
    const uint32_t uavHistoryResourceIndex   = isOddFrame ? FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_1 : FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_2;
    const uint32_t srvHistoryResourceIndex   = isOddFrame ? FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_2 : FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_1;
    const bool     preProcessUseFragment     = (context->pipelineFlags & NSS_SHADER_PERMUTATION_PRE_PROCESS_COMPUTE) == 0;

    // Setup the resources for pre-process stage
    trace.stage = "resources";
    {
        // Input: Depth tm1
        // This is already setup in nssCreate(), nothing to do here.

        // Input: Depth
        FFX_VALIDATE(context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                                     &params->depth,
                                                                                     context->effectContextId,
                                                                                     &context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH]));

        // Input: Color
        FFX_VALIDATE(context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                                     &params->color,
                                                                                     context->effectContextId,
                                                                                     &context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_INPUT_COLOR]));

        // Input: Motion vector
        FFX_VALIDATE(context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                                     &params->motionVectors,
                                                                                     context->effectContextId,
                                                                                     &context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS]));

        // Input: History
        if (manageHistory)
        {
            // History will use the upscaled output generated by last frame.
            context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_UPSCALED_COLOR] = context->srvResources[srvHistoryResourceIndex];
        }
        else
        {
            FFX_VALIDATE(
                context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                                &params->outputTm1,
                                                                                context->effectContextId,
                                                                                &context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_UPSCALED_COLOR]));
        }

        // Input: Exposure texture.
        if (useExposureTexture)
        {
            FFX_VALIDATE(context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                                         &params->exposureTexture,
                                                                                         context->effectContextId,
                                                                                         &context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_INPUT_EXPOSURE]));
        }
        // Input: Feedback tm1
        // This is already setup in nssCreate(), nothing to do here.

        // Input: LumaDeriv tm1
        context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_TM1] = context->srvResources[lumaDerivSrvResourceIndex];

        // Output: LumaDeriv, consumed by next frame
        context->uavResources[FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV] = context->uavResources[lumaDerivUavResourceIndex];

        // Current-frame LumaDeriv as an SRV alias (debug view reads the value pre-process wrote this frame).
        context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV] = context->srvResources[lumaDerivUavResourceIndex];

        // Output: data graph input tensor
        // This is already setup in nssCreate(), nothing to do here.

        // Output: Depth offset, consumed by post-process stage
        // This is already setup in nssCreate(), nothing to do here.
    }

    // Setup the resources for data graph stage
    {
        // Input: data graph input tensor, outputted by pre-process stage.
        // This is already setup in nssCreate(), nothing to do here.

        // Output: Feedback tensor, consumed by next frame.
        // This is already setup in nssCreate(), nothing to do here.

        // Output: KPN coefficients tensor, consumed by post-process stage
        // This is already setup in nssCreate(), nothing to do here.
    }

    // Setup the resources for post-process stage
    {
        // Input: KPN coefficients tensor (KPN_TENSOR), outputted by data graph stage
        // This is already setup in nssCreate(), nothing to do here.

        // Input: Temporal feedback = current frame's feedback output by data graph.
        // This is already setup in nssCreate(), nothing to do here.

        // Input: Offset LUT is baked in shader code for exact x2, or generated into OFFSET_LUT for general ratios.

        // Input: Depth offset, outputted by pre-process stage
        // This is already setup in nssCreate(), nothing to do here.

        // Input: Color, already setup in pre-process stage, nothing to do here.
        // Input: Motion vector, already setup in pre-process stage, nothing to do here.
        // Input: History, already setup in pre-process stage, nothing to do here.

        if (manageHistory)
        {
            // Output: history buffer for next frame use
            context->uavResources[FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_UPSCALED_COLOR] = context->uavResources[uavHistoryResourceIndex];
        }

        // Output: Upscaled output
        FFX_VALIDATE(context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                                     &params->output,
                                                                                     context->effectContextId,
                                                                                     &context->uavResources[FFX_NSS_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT]));
        context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT] = context->uavResources[FFX_NSS_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT];
    }

    trace.stage = "constants";
    FFX_VALIDATE(setupConstantBuffer(context, params));

    NssCpuClock::time_point stageStart{};
    if (trace.verbose)
    {
        stageStart = NssCpuClock::now();
    }

    // Depth scatter
    trace.stage = "preprocess";
    {
        FfxGpuJobDescription clearJob     = {FFX_GPU_JOB_CLEAR_UINT};
        const bool           reverseDepth = (context->pipelineFlags & NSS_SHADER_PERMUTATION_REVERSE_Z) != 0;
        if (reverseDepth)
        {
            const uint32_t clearValuesToMinInt[]{0x0, 0x0, 0x0, 0x0};
            memcpy(clearJob.clearUintJobDescriptor.color, clearValuesToMinInt, 4 * sizeof(uint32_t));
        }
        else
        {
            const uint32_t clearValuesToMaxInt[]{0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
            memcpy(clearJob.clearUintJobDescriptor.color, clearValuesToMaxInt, 4 * sizeof(uint32_t));
        }

        clearJob.clearUintJobDescriptor.target = context->uavResources[FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH_TM1];
        FFX_VALIDATE(context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob));

        const int32_t dispatchX = FFX_DIVIDE_ROUNDING_UP(context->depthScatterSize.width, FFX_NSS_THREAD_GROUP_WIDTH);
        const int32_t dispatchY = FFX_DIVIDE_ROUNDING_UP(context->depthScatterSize.height, FFX_NSS_THREAD_GROUP_HEIGHT);
        FFX_VALIDATE(scheduleDispatch(context, params, &context->pipelineNssDepthScatter, dispatchX, dispatchY, "DepthScatter"));
    }

    if (context->useDisocclusionMaskPass)
    {
        if (preProcessUseFragment)
        {
            FFX_VALIDATE(scheduleFragment(
                context, params, &context->pipelineNssDisocclusionMask, context->depthScatterSize.width, context->depthScatterSize.height, "DisocclusionMask"));
        }
        else
        {
            const int32_t dispatchX = FFX_DIVIDE_ROUNDING_UP(context->depthScatterSize.width, FFX_NSS_THREAD_GROUP_WIDTH);
            const int32_t dispatchY = FFX_DIVIDE_ROUNDING_UP(context->depthScatterSize.height, FFX_NSS_THREAD_GROUP_HEIGHT);
            FFX_VALIDATE(scheduleDispatch(context, params, &context->pipelineNssDisocclusionMask, dispatchX, dispatchY, "DisocclusionMask"));
        }
    }

    if (preProcessUseFragment)
    {
        FFX_VALIDATE(
            scheduleFragment(context, params, &context->pipelineNssPreprocess, context->dataGraphSize.width, context->dataGraphSize.height, "Preprocess"));
    }
    else
    {
        const int32_t dispatchSrcX = FFX_DIVIDE_ROUNDING_UP(context->dataGraphSize.width, FFX_NSS_THREAD_GROUP_WIDTH);
        const int32_t dispatchSrcY = FFX_DIVIDE_ROUNDING_UP(context->dataGraphSize.height, FFX_NSS_THREAD_GROUP_HEIGHT);
        FFX_VALIDATE(scheduleDispatch(context, params, &context->pipelineNssPreprocess, dispatchSrcX, dispatchSrcY, "Preprocess"));
    }

    if (trace.verbose)
    {
        const auto now              = NssCpuClock::now();
        trace.timings.preScheduleUs = elapsedMicroseconds(stageStart, now);
        stageStart                  = now;
    }
    trace.stage = "data_graph";
    FFX_VALIDATE(scheduleDataGraph(context, params, &context->pipelineNssDataGraph, "DataGraph"));
    if (trace.verbose)
    {
        const auto now                = NssCpuClock::now();
        trace.timings.graphScheduleUs = elapsedMicroseconds(stageStart, now);
        stageStart                    = now;
    }

    trace.stage = "postprocess";
    if (context->useDynamicOffsetLut)
    {
        const uint32_t tileCount    = context->reducedFractionHrSize.width * context->reducedFractionHrSize.height;
        const uint32_t dispatchLutX = FFX_DIVIDE_ROUNDING_UP(tileCount, 64u);
        FFX_VALIDATE(scheduleDispatch(context, params, &context->pipelineNssGenerateOffsetLut, dispatchLutX, 1, "GenerateOffsetLut"));
    }

    const bool postProcessUseFragment = (context->pipelineFlags & NSS_SHADER_PERMUTATION_POST_PROCESS_COMPUTE) == 0;
    if (postProcessUseFragment)
    {
        FFX_VALIDATE(scheduleFragment(context, params, &context->pipelineNssPostprocess, params->upscaleSize.width, params->upscaleSize.height, "Postprocess"));
    }
    else
    {
        const int32_t dispatchDstX = FFX_DIVIDE_ROUNDING_UP(params->upscaleSize.width, FFX_NSS_THREAD_GROUP_WIDTH);
        const int32_t dispatchDstY = FFX_DIVIDE_ROUNDING_UP(params->upscaleSize.height, FFX_NSS_THREAD_GROUP_HEIGHT);
        FFX_VALIDATE(scheduleDispatch(context, params, &context->pipelineNssPostprocess, dispatchDstX, dispatchDstY, "Postprocess"));
    }

    if (trace.verbose)
    {
        trace.timings.postScheduleUs = elapsedMicroseconds(stageStart, NssCpuClock::now());
    }

    if (hasDebugView)
    {
        trace.stage = "debug_view";
        FFX_VALIDATE(context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                                     &params->debugViews,
                                                                                     context->effectContextId,
                                                                                     &context->uavResources[FFX_NSS_RESOURCE_IDENTIFIER_DEBUG_VIEWS]));
        context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_DEBUG_VIEWS] = context->uavResources[FFX_NSS_RESOURCE_IDENTIFIER_DEBUG_VIEWS];

        // Assume the debug view compute/fragment choice is same as the post process.
        if (postProcessUseFragment)
        {
            FFX_VALIDATE(scheduleFragment(context, params, &context->pipelineNssDebugView, params->upscaleSize.width, params->upscaleSize.height, "DebugView"));
        }
        else
        {
            const int32_t dispatchDbgX = FFX_DIVIDE_ROUNDING_UP(params->upscaleSize.width, FFX_NSS_THREAD_GROUP_WIDTH);
            const int32_t dispatchDbgY = FFX_DIVIDE_ROUNDING_UP(params->upscaleSize.height, FFX_NSS_THREAD_GROUP_HEIGHT);
            FFX_VALIDATE(scheduleDispatch(context, params, &context->pipelineNssDebugView, dispatchDbgX, dispatchDbgY, "DebugView"));
        }
    }

    trace.stage = "record_commands";
    FFX_VALIDATE(
        context->contextDescription.backendInterface.fpExecuteGpuJobs(&context->contextDescription.backendInterface, commandList, context->effectContextId));

    // release dynamic resources
    trace.stage = "unregister_resources";
    FFX_VALIDATE(backendScope.Complete());

    // NSS_MAX_QUEUED_FRAMES must be an even number.
    FFX_STATIC_ASSERT((NSS_MAX_QUEUED_FRAMES & 1) == 0);
    context->resourceFrameIndex = (context->resourceFrameIndex + 1) % NSS_MAX_QUEUED_FRAMES;
    context->firstExecution     = false;
    return FFX_OK;
}

FfxErrorCode ffxNssContextCreate(FfxNssContext* context, const FfxNssContextDescription* contextDescription)
{
    // check pointers are valid.
    if (!context)
    {
        return ffxLogError(FFX_ERROR_INVALID_POINTER, "context creation requires a context");
    }
    // zero context memory
    memset(context, 0, sizeof(FfxNssContext));

    if (!contextDescription)
    {
        return ffxLogError(FFX_ERROR_INVALID_POINTER, "context creation requires a description");
    }

    const FfxInterface& backend          = contextDescription->backendInterface;
    constexpr uint32_t  allComputeFlags  = FFX_NSS_CONTEXT_FLAG_PRE_PROCESS_COMPUTE | FFX_NSS_CONTEXT_FLAG_POST_PROCESS_COMPUTE;
    const bool          requiresGraphics = (contextDescription->flags & allComputeFlags) != allComputeFlags;
    const struct
    {
        bool        present;
        const char* name;
    } requiredCallbacks[] = {
        {backend.fpCreateBackendContext != nullptr, "fpCreateBackendContext"},
        {backend.fpDestroyBackendContext != nullptr, "fpDestroyBackendContext"},
        {backend.fpGetDeviceCapabilities != nullptr, "fpGetDeviceCapabilities"},
        {backend.fpSetMessageCallback != nullptr, "fpSetMessageCallback"},
        {backend.fpCreateResource != nullptr, "fpCreateResource"},
        {backend.fpDestroyResource != nullptr, "fpDestroyResource"},
        {backend.fpCreateComputePipeline != nullptr, "fpCreateComputePipeline"},
        {!requiresGraphics || backend.fpCreateGraphicsPipeline != nullptr, "fpCreateGraphicsPipeline"},
        {backend.fpCreateDataGraphPipeline != nullptr, "fpCreateDataGraphPipeline"},
        {backend.fpDestroyPipeline != nullptr, "fpDestroyPipeline"},
        {backend.fpRegisterResource != nullptr, "fpRegisterResource"},
        {backend.fpUnregisterResources != nullptr, "fpUnregisterResources"},
        {backend.fpStageConstantBufferDataFunc != nullptr, "fpStageConstantBufferDataFunc"},
        {backend.fpScheduleGpuJob != nullptr, "fpScheduleGpuJob"},
        {backend.fpExecuteGpuJobs != nullptr, "fpExecuteGpuJobs"},
    };
    for (const auto& callback : requiredCallbacks)
    {
        if (!callback.present)
        {
            char message[NSS_LOG_MESSAGE_CAPACITY];
            std::snprintf(message, sizeof(message), "context creation requires %s", callback.name);
            return logErrorWithCallback(FFX_ERROR_INCOMPLETE_INTERFACE, message, contextDescription->fpMessage);
        }
    }

    if (contextDescription->qualityMode != FFX_NSS_SHADER_QUALITY_MODE_QUALITY && contextDescription->qualityMode != FFX_NSS_SHADER_QUALITY_MODE_BALANCED &&
        contextDescription->qualityMode != FFX_NSS_SHADER_QUALITY_MODE_PERFORMANCE)
    {
        return logErrorWithCallback(FFX_ERROR_INVALID_ENUM, "context creation validates qualityMode", contextDescription->fpMessage);
    }
    if (!contextDescription->renderSize.width)
    {
        return logErrorWithCallback(FFX_ERROR_INVALID_ALIGNMENT, "context creation validates renderSize.width", contextDescription->fpMessage);
    }
    if (!contextDescription->renderSize.height)
    {
        return logErrorWithCallback(FFX_ERROR_INVALID_ALIGNMENT, "context creation validates renderSize.height", contextDescription->fpMessage);
    }
    if (!contextDescription->upscaleSize.width)
    {
        return logErrorWithCallback(FFX_ERROR_INVALID_ALIGNMENT, "context creation validates upscaleSize.width", contextDescription->fpMessage);
    }
    if (!contextDescription->upscaleSize.height)
    {
        return logErrorWithCallback(FFX_ERROR_INVALID_ALIGNMENT, "context creation validates upscaleSize.height", contextDescription->fpMessage);
    }

    // With single-step mirror mapping in preprocess shader, we require the aligned tensor extent
    // to stay within 2x render extent per axis. For align-up padding this is guaranteed when
    // render size is at least half the alignment.
    constexpr uint32_t kMinRenderSizeForMirror = (FFX_NSS_RESOURCE_ALIGNMENT + 1u) / 2u;
    if (contextDescription->renderSize.width < kMinRenderSizeForMirror)
    {
        return logErrorWithCallback(FFX_ERROR_INVALID_ARGUMENT, "context creation validates renderSize.width", contextDescription->fpMessage);
    }
    if (contextDescription->renderSize.height < kMinRenderSizeForMirror)
    {
        return logErrorWithCallback(FFX_ERROR_INVALID_ARGUMENT, "context creation validates renderSize.height", contextDescription->fpMessage);
    }

    // if a scratch buffer is declared, then we must have a size
    if (contextDescription->backendInterface.scratchBuffer)
    {
        if (!contextDescription->backendInterface.scratchBufferSize)
        {
            return logErrorWithCallback(FFX_ERROR_INCOMPLETE_INTERFACE, "context creation requires scratchBufferSize", contextDescription->fpMessage);
        }
    }

    // ensure the context is large enough for the internal context.
    FFX_STATIC_ASSERT(sizeof(FfxNssContext) >= sizeof(FfxNssContext_Private));

    // create the context.
    ffxLogSdkVersion(contextDescription->fpMessage);
    logNssVersion(contextDescription->fpMessage);
    FfxNssContext_Private* contextPrivate = (FfxNssContext_Private*)(context);
    const FfxErrorCode     errorCode      = nssCreate(contextPrivate, contextDescription);
    if (errorCode != FFX_OK)
    {
        if (!contextPrivate->unsupportedDevice)
        {
            logErrorWithCallback(errorCode, "NSS context creation failed", contextDescription->fpMessage);
        }
        if (contextPrivate->backendContextCreated)
        {
            const FfxErrorCode releaseError = nssRelease(contextPrivate);
            if (releaseError != FFX_OK)
            {
                logErrorWithCallback(releaseError, "failed to release an incomplete NSS context", contextDescription->fpMessage);
            }
        }
        contextPrivate->device = nullptr;
        return errorCode;
    }
    nssLogContextCreate(contextPrivate);
    return errorCode;
}

FfxErrorCode ffxNssContextGetGpuMemoryUsage(FfxNssContext* context, FfxEffectMemoryUsage* vramUsage)
{
    if (!context)
    {
        return ffxLogError(FFX_ERROR_INVALID_POINTER, "memory query requires a context");
    }
    FfxNssContext_Private* contextPrivate = reinterpret_cast<FfxNssContext_Private*>(context);
    if (!vramUsage)
    {
        return logErrorWithCallback(FFX_ERROR_INVALID_POINTER, "memory query requires vramUsage", contextPrivate->contextDescription.fpMessage);
    }

    if (!contextPrivate->device)
    {
        return logErrorWithCallback(FFX_ERROR_NULL_DEVICE, "memory query requires a device", contextPrivate->contextDescription.fpMessage);
    }
    if (!contextPrivate->contextDescription.backendInterface.fpGetEffectGpuMemoryUsage)
    {
        return logErrorWithCallback(
            FFX_ERROR_INCOMPLETE_INTERFACE, "memory query requires fpGetEffectGpuMemoryUsage", contextPrivate->contextDescription.fpMessage);
    }

    const FfxErrorCode errorCode = contextPrivate->contextDescription.backendInterface.fpGetEffectGpuMemoryUsage(
        &contextPrivate->contextDescription.backendInterface, contextPrivate->effectContextId, vramUsage);
    if (errorCode != FFX_OK)
    {
        return logErrorWithCallback(errorCode, "memory query", contextPrivate->contextDescription.fpMessage);
    }

    return FFX_OK;
}

FfxErrorCode ffxNssContextDestroy(FfxNssContext* context)
{
    if (!context)
    {
        return ffxLogError(FFX_ERROR_INVALID_POINTER, "context destruction requires a context");
    }

    // destroy the context.
    FfxNssContext_Private* contextPrivate = (FfxNssContext_Private*)(context);
    if (!contextPrivate->backendContextCreated)
    {
        return logErrorWithCallback(
            FFX_ERROR_INVALID_ARGUMENT, "context destruction requires a live NSS context", contextPrivate->contextDescription.fpMessage);
    }
    const FfxErrorCode errorCode = nssRelease(contextPrivate);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);
    contextPrivate->device = nullptr;
    printMessage(contextPrivate, FFX_MESSAGE_TYPE_INFO, "event=context_destroy");
    return FFX_OK;
}

FfxErrorCode ffxNssContextDispatch(FfxNssContext* context, const FfxNssDispatchDescription* dispatchParams)
{
    if (!context)
    {
        return ffxLogError(FFX_ERROR_INVALID_POINTER, "dispatch requires a context");
    }

    FfxNssContext_Private* contextPrivate = reinterpret_cast<FfxNssContext_Private*>(context);
    FFX_ASSERT(contextPrivate);
    ++contextPrivate->diagnostics.dispatchId;
    const bool verbose     = dispatchParams && (dispatchParams->flags & FFX_NSS_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING) != 0;
    const auto reportError = [contextPrivate, verbose](FfxErrorCode errorCode, const char* reason, const char* stage = "validation") {
        return logDispatchError(contextPrivate, errorCode, reason, verbose, stage);
    };
    if (!dispatchParams)
    {
        return reportError(FFX_ERROR_INVALID_POINTER, "dispatch requires parameters");
    }
    if (contextPrivate->requiresRecreation)
    {
        return reportError(FFX_ERROR_BACKEND_API_ERROR, "context requires recreation after a backend dispatch failure", "context_state");
    }
    if (verbose)
    {
        logDispatchBegin(contextPrivate, dispatchParams);
    }

    // validate zero sizes.
    if (!dispatchParams->renderSize.width)
    {
        return reportError(FFX_ERROR_INVALID_ARGUMENT, "dispatch validates renderSize.width");
    }
    if (!dispatchParams->renderSize.height)
    {
        return reportError(FFX_ERROR_INVALID_ARGUMENT, "dispatch validates renderSize.height");
    }
    if (!dispatchParams->upscaleSize.width)
    {
        return reportError(FFX_ERROR_INVALID_ARGUMENT, "dispatch validates upscaleSize.width");
    }
    if (!dispatchParams->upscaleSize.height)
    {
        return reportError(FFX_ERROR_INVALID_ARGUMENT, "dispatch validates upscaleSize.height");
    }

    // validate that renderSize/upscaleSize match the size declared at context creation.
    if (dispatchParams->renderSize.width != contextPrivate->contextDescription.renderSize.width)
    {
        return reportError(FFX_ERROR_OUT_OF_RANGE, "dispatch validates renderSize.width");
    }
    if (dispatchParams->renderSize.height != contextPrivate->contextDescription.renderSize.height)
    {
        return reportError(FFX_ERROR_OUT_OF_RANGE, "dispatch validates renderSize.height");
    }
    if (dispatchParams->upscaleSize.width != contextPrivate->contextDescription.upscaleSize.width)
    {
        return reportError(FFX_ERROR_OUT_OF_RANGE, "dispatch validates upscaleSize.width");
    }
    if (dispatchParams->upscaleSize.height != contextPrivate->contextDescription.upscaleSize.height)
    {
        return reportError(FFX_ERROR_OUT_OF_RANGE, "dispatch validates upscaleSize.height");
    }
    if (!contextPrivate->device)
    {
        return reportError(FFX_ERROR_NULL_DEVICE, "dispatch requires a device");
    }
    if (!dispatchParams->commandList)
    {
        return reportError(FFX_ERROR_INVALID_POINTER, "dispatch requires a command list");
    }
    if (!dispatchParams->color.resource)
    {
        return reportError(FFX_ERROR_INVALID_POINTER, "dispatch requires a color resource");
    }
    if (!dispatchParams->depth.resource)
    {
        return reportError(FFX_ERROR_INVALID_POINTER, "dispatch requires a depth resource");
    }
    if (!dispatchParams->motionVectors.resource)
    {
        return reportError(FFX_ERROR_INVALID_POINTER, "dispatch requires motion vectors");
    }
    if (!dispatchParams->output.resource)
    {
        return reportError(FFX_ERROR_INVALID_POINTER, "dispatch requires an output resource");
    }
    if ((dispatchParams->flags & FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW) && !dispatchParams->debugViews.resource)
    {
        return reportError(FFX_ERROR_INVALID_POINTER, "dispatch requires debug views");
    }
    if ((contextPrivate->contextDescription.flags & FFX_NSS_CONTEXT_FLAG_MANAGE_HISTORY) == 0 && !dispatchParams->outputTm1.resource)
    {
        return reportError(FFX_ERROR_INVALID_POINTER, "dispatch requires output history");
    }

    // dispatch the NSS passes.
    NssDispatchTrace        trace{"validation", {}, verbose};
    NssCpuClock::time_point dispatchStart{};
    if (verbose)
    {
        dispatchStart = NssCpuClock::now();
    }
    const FfxErrorCode errorCode = nssDispatch(contextPrivate, dispatchParams, trace);
    if (errorCode != FFX_OK)
    {
        return reportError(errorCode, "NSS dispatch", trace.stage);
    }
    contextPrivate->diagnostics.lastError = {};
    if (verbose)
    {
        const double dispatchUs = elapsedMicroseconds(dispatchStart, NssCpuClock::now());
        char         message[NSS_LOG_MESSAGE_CAPACITY];
        std::snprintf(message,
                      sizeof(message),
                      "event=cpu_profile cpu_pre_schedule_us=%.6g cpu_graph_schedule_us=%.6g cpu_post_schedule_us=%.6g cpu_dispatch_us=%.6g",
                      trace.timings.preScheduleUs,
                      trace.timings.graphScheduleUs,
                      trace.timings.postScheduleUs,
                      dispatchUs);
        printMessage(contextPrivate, FFX_MESSAGE_TYPE_INFO, message);
        printMessage(contextPrivate, FFX_MESSAGE_TYPE_INFO, "event=dispatch_end stage=record_commands result=recorded error_code=0 error_name=FFX_OK");
    }
    return errorCode;
}

int32_t ffxNssGetJitterPhaseCount(int32_t renderWidth, int32_t displayWidth)
{
    const float   basePhaseCount   = 8.0f;
    const int32_t jitterPhaseCount = int32_t(basePhaseCount * pow((float(displayWidth) / renderWidth), 2.0f));
    return jitterPhaseCount;
}

// Calculate halton number for index and base.
static float halton(int32_t index, int32_t base)
{
    float f = 1.0f, result = 0.0f;

    for (int32_t currentIndex = index; currentIndex > 0;)
    {
        f /= (float)base;
        result       = result + f * (float)(currentIndex % base);
        currentIndex = (uint32_t)(floorf((float)(currentIndex) / (float)(base)));
    }

    return result;
}

FfxErrorCode ffxNssGetJitterOffset(float* outX, float* outY, int32_t index, int32_t phaseCount)
{
    if (!outX)
    {
        return ffxLogError(FFX_ERROR_INVALID_POINTER, "jitter offset requires outX");
    }
    if (!outY)
    {
        return ffxLogError(FFX_ERROR_INVALID_POINTER, "jitter offset requires outY");
    }
    if (phaseCount <= 0)
    {
        return ffxLogError(FFX_ERROR_INVALID_ARGUMENT, "jitter offset requires a positive phase count");
    }

    const float x = halton((index % phaseCount) + 1, 2) - 0.5f;
    const float y = halton((index % phaseCount) + 1, 3) - 0.5f;

    *outX = x;
    *outY = y;
    return FFX_OK;
}

FFX_API bool ffxNssResourceIsNull(const FfxResource& resource)
{
    return resource.resource == NULL;
}
