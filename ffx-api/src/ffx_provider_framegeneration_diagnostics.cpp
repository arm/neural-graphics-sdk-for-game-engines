/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "ffx_provider_framegeneration_diagnostics.h"

#include "ffx_provider.h"
#include "ffx_logging.h"

#include <ffx_api/ffx_framegeneration.hpp>
#include <FidelityFX/host/ffx_frameinterpolation.h>

#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>

static constexpr const char* ffxLogComponentName()
{
    return "NFRU";
}

namespace
{
    constexpr size_t NFRU_LOG_MESSAGE_CAPACITY = 1024;

    std::atomic<uint64_t> g_nextNfruLogContextId{1};
    std::atomic<uint32_t> g_loggedNfruCapabilityFailures{0};

    bool sameStaticString(const char* lhs, const char* rhs)
    {
        return lhs == rhs || (lhs && rhs && std::strcmp(lhs, rhs) == 0);
    }

    const char* getPreOfName(const FfxNfruResolutionPlan& plan)
    {
        return plan.needsOfColourDownsample ? "of-colour" : "none";
    }

    const char* getPreWarpName(const FfxNfruResolutionPlan& plan)
    {
        if (plan.needsAllInputDownsample)
        {
            return "all-input";
        }
        if (plan.needsColourDownsample)
        {
            return "colour-only";
        }
        return "none";
    }

    const char* getFiStageName(FfxNfruFiStage stage)
    {
        switch (stage)
        {
        case FfxNfruFiStage::Validation:
            return "fi_validation";
        case FfxNfruFiStage::PipelineRefresh:
            return "fi_pipeline_refresh";
        case FfxNfruFiStage::Preprocess:
            return "fi_preprocess";
        case FfxNfruFiStage::DataGraph:
            return "fi_data_graph";
        case FfxNfruFiStage::Postprocess:
            return "fi_postprocess";
        case FfxNfruFiStage::RecordCommands:
            return "record_commands";
        case FfxNfruFiStage::Unregister:
            return "unregister";
        case FfxNfruFiStage::None:
        default:
            return "frame_interpolation";
        }
    }

    const char* getOfStageName(FfxNfruOfStage stage)
    {
        switch (stage)
        {
        case FfxNfruOfStage::Validation:
            return "of_validation";
        case FfxNfruOfStage::RegisterResources:
            return "of_register";
        case FfxNfruOfStage::ClearHistory:
            return "of_clear_history";
        case FfxNfruOfStage::MotionVectorHints:
            return "of_mv_hints";
        case FfxNfruOfStage::DataGraph:
            return "of_data_graph";
        case FfxNfruOfStage::RecordCommands:
            return "record_commands";
        case FfxNfruOfStage::Unregister:
            return "unregister";
        case FfxNfruOfStage::None:
        default:
            return "optical_flow";
        }
    }
}  // namespace

NfruDiagnostics NfruDiagnostics::Create(ffxApiMessage callback) noexcept
{
    NfruDiagnostics diagnostics{};
    diagnostics.contextId_ = g_nextNfruLogContextId.fetch_add(1, std::memory_order_relaxed);
    diagnostics.callback_  = callback;
    return diagnostics;
}

bool NfruDiagnostics::HasError() const noexcept
{
    return lastError_.reasonCode != nullptr;
}

void NfruDiagnostics::ClearError() noexcept
{
    lastError_ = {};
}

void NfruDiagnostics::LogVersions(ffxApiMessage callback) noexcept
{
    ffxLogSdkVersion();
    ffxLogNfruVersion();
    if (!callback)
    {
        return;
    }

    char message[192];
    std::snprintf(message, sizeof(message), "event=versions sdk_version=%d.%d.%d", FFX_SDK_VERSION_MAJOR, FFX_SDK_VERSION_MINOR, FFX_SDK_VERSION_PATCH);
    callback(FFX_API_MESSAGE_TYPE_INFO, message);
    std::snprintf(message,
                  sizeof(message),
                  "event=versions nfru_component_version=%d.%d.%d",
                  FFX_FRAMEINTERPOLATION_VERSION_MAJOR,
                  FFX_FRAMEINTERPOLATION_VERSION_MINOR,
                  FFX_FRAMEINTERPOLATION_VERSION_PATCH);
    callback(FFX_API_MESSAGE_TYPE_INFO, message);
}

void NfruDiagnostics::LogResolutionValidationError(ffxApiMessage callback, FfxNfruResolutionValidationError error) noexcept
{
    const char* message = "Frame generation dimensions are invalid.";
    switch (error)
    {
    case FfxNfruResolutionValidationError::DisplayWidth:
        message = "Frame generation requires a non-zero display width.";
        break;
    case FfxNfruResolutionValidationError::DisplayHeight:
        message = "Frame generation requires a non-zero display height.";
        break;
    case FfxNfruResolutionValidationError::RenderWidth:
        message = "Frame generation render width is below the minimum supported size.";
        break;
    case FfxNfruResolutionValidationError::RenderHeight:
        message = "Frame generation render height is below the minimum supported size.";
        break;
    case FfxNfruResolutionValidationError::MvDepthWidth:
        message = "Frame generation mvDepthSize.width is below the minimum supported size; only {0, 0} selects renderSize.";
        break;
    case FfxNfruResolutionValidationError::MvDepthHeight:
        message = "Frame generation mvDepthSize.height is below the minimum supported size; only {0, 0} selects renderSize.";
        break;
    case FfxNfruResolutionValidationError::MvDepthAspectRatio:
        message = "Frame generation mvDepthSize must have the same aspect ratio as renderSize.";
        break;
    case FfxNfruResolutionValidationError::DerivedWidth:
        message = "Frame generation derived NFRU lane width is not representable.";
        break;
    case FfxNfruResolutionValidationError::ShaderDimensionOutOfRange:
        message = "Frame generation input dimensions must fit the signed 32-bit shader range.";
        break;
    case FfxNfruResolutionValidationError::DerivedDimensionOutOfRange:
        message = "Frame generation derived NFRU dimensions exceed the signed shader range.";
        break;
    default:
        break;
    }
    ffxProviderLogMessage(callback, FFX_API_MESSAGE_TYPE_ERROR, message);
}

void NfruDiagnostics::LogUnsupportedBackBufferFormat(ffxApiMessage callback) noexcept
{
    ffxProviderLogMessage(callback,
                          FFX_API_MESSAGE_TYPE_ERROR,
                          "unsupported optical-flow backbuffer format; expected R8G8B8A8_UNORM, B8G8R8A8_UNORM, R11G11B10_FLOAT, or R8_UNORM");
}

void NfruDiagnostics::PrintMessage(FfxMsgType type, const char* message) const noexcept
{
    char contextualMessage[NFRU_LOG_MESSAGE_CAPACITY];
    std::snprintf(contextualMessage, sizeof(contextualMessage), "ctx_id=%" PRIu64 " %s", contextId_, message ? message : "");
    ffxProviderLogMessage(callback_, static_cast<uint32_t>(type), contextualMessage);
}

void NfruDiagnostics::PrintFrameMessage(uint64_t frameId, FfxMsgType type, const char* message) const noexcept
{
    char contextualMessage[NFRU_LOG_MESSAGE_CAPACITY];
    std::snprintf(contextualMessage, sizeof(contextualMessage), "frame_id=%" PRIu64 " %s", frameId, message ? message : "");
    PrintMessage(type, contextualMessage);
}

ffxReturnCode_t NfruDiagnostics::LogContextCreationError(const char*           source,
                                                         const char*           stage,
                                                         FfxErrorCode          errorCode,
                                                         const char*           reasonCode,
                                                         const char*           reason,
                                                         ffxReturnCode_t       returnCode,
                                                         NfruCapabilityFailure capabilityFailure) noexcept
{
    lastError_             = {errorCode, "create", stage, reasonCode};
    const uint32_t failure = static_cast<uint32_t>(capabilityFailure);
    if (failure != 0)
    {
        const uint32_t previousFailures = g_loggedNfruCapabilityFailures.fetch_or(failure, std::memory_order_relaxed);
        if ((previousFailures & failure) != 0)
        {
            return returnCode;
        }
    }

    char message[NFRU_LOG_MESSAGE_CAPACITY];
    std::snprintf(message,
                  sizeof(message),
                  "event=context_error phase=create source=%s stage=%s error_code=%d error_name=%s reason_code=%s reason=\"%s\" "
                  "fallback=application_required",
                  source,
                  stage,
                  errorCode,
                  ffxGetErrorCodeName(errorCode),
                  reasonCode,
                  reason);
    PrintMessage(FFX_MESSAGE_TYPE_ERROR, message);
    return returnCode;
}

ffxReturnCode_t NfruDiagnostics::LogRuntimeError(uint64_t        frameId,
                                                 const char*     phase,
                                                 const char*     source,
                                                 const char*     stage,
                                                 FfxErrorCode    errorCode,
                                                 const char*     reasonCode,
                                                 const char*     reason,
                                                 bool            verbose,
                                                 ffxReturnCode_t returnCode) noexcept
{
    const bool repeated = lastError_.code == errorCode && sameStaticString(lastError_.phase, phase) && sameStaticString(lastError_.stage, stage) &&
                          sameStaticString(lastError_.reasonCode, reasonCode);
    lastError_ = {errorCode, phase, stage, reasonCode};
    if (verbose || !repeated)
    {
        char message[NFRU_LOG_MESSAGE_CAPACITY];
        std::snprintf(message,
                      sizeof(message),
                      "event=dispatch_error phase=%s source=%s stage=%s error_code=%d error_name=%s reason_code=%s reason=\"%s\" "
                      "fallback=application_required",
                      phase,
                      source,
                      stage,
                      errorCode,
                      ffxGetErrorCodeName(errorCode),
                      reasonCode,
                      reason);
        PrintFrameMessage(frameId, FFX_MESSAGE_TYPE_ERROR, message);
    }
    return returnCode;
}

ffxReturnCode_t NfruDiagnostics::LogOpticalFlowError(uint64_t frameId, FfxNfruOfStage stage, FfxErrorCode errorCode, bool verbose) noexcept
{
    const bool validationFailure = stage == FfxNfruOfStage::Validation;
    return LogRuntimeError(frameId,
                           "dispatch",
                           "optical_flow",
                           getOfStageName(stage),
                           errorCode,
                           validationFailure ? "optical_flow_validation_failed" : "optical_flow_dispatch_failed",
                           validationFailure ? "optical flow dispatch inputs failed validation" : "optical flow failed to record commands",
                           verbose,
                           validationFailure ? FFX_API_RETURN_ERROR_PARAMETER : FFX_API_RETURN_ERROR_RUNTIME_ERROR);
}

ffxReturnCode_t NfruDiagnostics::LogFrameInterpolationError(uint64_t frameId, FfxNfruFiStage stage, FfxErrorCode errorCode, bool verbose) noexcept
{
    const bool validationFailure = stage == FfxNfruFiStage::Validation;
    return LogRuntimeError(frameId,
                           "dispatch",
                           "frame_interpolation",
                           getFiStageName(stage),
                           errorCode,
                           validationFailure ? "frame_interpolation_validation_failed" : "frame_interpolation_dispatch_failed",
                           validationFailure ? "frame interpolation dispatch inputs failed validation" : "frame interpolation failed to record commands",
                           verbose,
                           validationFailure ? FFX_API_RETURN_ERROR_PARAMETER : FFX_API_RETURN_ERROR_RUNTIME_ERROR);
}

void NfruDiagnostics::LogContextCreated(const ffxApiCreateContextDescFrameGeneration& desc,
                                        const NfruLogContextInfo&                     info,
                                        FfxOpticalFlowGridSize                        opticalFlowGrid) const noexcept
{
    const float scaleX = static_cast<float>(desc.displaySize.width) / desc.renderSize.width;
    const float scaleY = static_cast<float>(desc.displaySize.height) / desc.renderSize.height;
    char        message[NFRU_LOG_MESSAGE_CAPACITY];
    std::snprintf(message,
                  sizeof(message),
                  "event=context_create render=%ux%u display=%ux%u render_to_display_scale=%.4fx%.4f generated_frames_per_rendered=1 "
                  "frame_rate_multiplier=2 quality_mode=not_configurable backbuffer_format=%s format_id=%u",
                  desc.renderSize.width,
                  desc.renderSize.height,
                  desc.displaySize.width,
                  desc.displaySize.height,
                  scaleX,
                  scaleY,
                  ffxGetSurfaceFormatName(ConvertEnum<FfxSurfaceFormat>(desc.backBufferFormat)),
                  desc.backBufferFormat);
    PrintMessage(FFX_MESSAGE_TYPE_INFO, message);

    const FfxNfruResolutionPlan& plan = *info.resolutionPlan;
    std::snprintf(message,
                  sizeof(message),
                  "event=context_pipeline context_flags=0x%08x of_grid=%ux%u pre_of=%s pre_warp=%s mv_depth_input=%ux%u "
                  "of_input=%ux%u mv_depth_lane=%ux%u flow_lane=%ux%u tensor=%ux%u params_gate=%ux%u warp=%s fill_holes=%s "
                  "preprocess=%s postprocess=%s debug_view=%s mv_hints=%u precision=%s fp16_fallback=%s depth_history=%s color_history=%s",
                  info.contextFlags,
                  static_cast<unsigned>(opticalFlowGrid),
                  static_cast<unsigned>(opticalFlowGrid),
                  getPreOfName(plan),
                  getPreWarpName(plan),
                  plan.mvDepthInput.width,
                  plan.mvDepthInput.height,
                  plan.ofInputColour.width,
                  plan.ofInputColour.height,
                  plan.mvDepthLane.width,
                  plan.mvDepthLane.height,
                  plan.flowLane.width,
                  plan.flowLane.height,
                  plan.paddedTensorLane.width,
                  plan.paddedTensorLane.height,
                  plan.paramsLane.width,
                  plan.paramsLane.height,
                  (info.contextFlags & FFX_API_FG_CONTEXT_FLAG_WARP_COMPUTE) ? "compute" : "fragment",
                  (info.contextFlags & FFX_API_FG_CONTEXT_FLAG_FILL_HOLES_COMPUTE) ? "compute" : "fragment",
                  (info.contextFlags & FFX_API_FG_CONTEXT_FLAG_PRE_PROCESS_COMPUTE) ? "compute" : "fragment",
                  (info.contextFlags & FFX_API_FG_CONTEXT_FLAG_POSTPROCESS_COMPUTE) ? "compute" : "fragment",
                  (info.contextFlags & FFX_API_FG_CONTEXT_FLAG_DEBUG_VIEW_COMPUTE) ? "compute" : "fragment",
                  static_cast<unsigned>((info.contextFlags & FFX_API_FG_CONTEXT_FLAG_ENABLE_MV_HINTS) != 0),
                  info.fp16Supported ? "fp16" : "fp32",
                  info.fp16Supported ? "none" : "required_features_unsupported",
                  info.usesInternalDepthHistory ? "managed" : "application",
                  (info.contextFlags & FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR) ? "managed" : "application");
    PrintMessage(FFX_MESSAGE_TYPE_INFO, message);
}

void NfruDiagnostics::LogContextDestroyed(FfxErrorCode errorCode, const char* source) const noexcept
{
    char message[NFRU_LOG_MESSAGE_CAPACITY];
    std::snprintf(message,
                  sizeof(message),
                  "event=context_destroy result=%s source=%s stage=teardown error_code=%d error_name=%s fallback=%s",
                  errorCode == FFX_OK ? "success" : "failed",
                  source,
                  errorCode,
                  ffxGetErrorCodeName(errorCode),
                  errorCode == FFX_OK ? "none" : "application_required");
    PrintMessage(errorCode == FFX_OK ? FFX_MESSAGE_TYPE_INFO : FFX_MESSAGE_TYPE_ERROR, message);
}

void NfruDiagnostics::LogFrameGenerationDisabledOnDestroy() const noexcept
{
    PrintMessage(FFX_MESSAGE_TYPE_INFO, "event=frame_generation_state state=disabled reason=context_destroy");
}

void NfruDiagnostics::LogFrameGenerationState(const ffxApiConfigureDescFrameGeneration& desc, bool enabled) const noexcept
{
    char message[NFRU_LOG_MESSAGE_CAPACITY];
    std::snprintf(message,
                  sizeof(message),
                  "event=frame_generation_state state=%s mode=%s reason=configure debug_view=%u debug_logging=%u async=%u",
                  enabled ? "enabled" : "disabled",
                  (desc.flags & FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY) ? "manual" : "swapchain",
                  static_cast<unsigned>((desc.flags & FFX_API_FG_DISPATCH_FLAG_DRAW_DEBUG_VIEW) != 0),
                  static_cast<unsigned>((desc.flags & FFX_API_FG_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING) != 0),
                  static_cast<unsigned>(desc.allowAsyncWorkloads));
    PrintFrameMessage(desc.frameID, FFX_MESSAGE_TYPE_INFO, message);
}

void NfruDiagnostics::LogPrepareBegin(const ffxApiDispatchDescFrameGenerationPrepare& desc,
                                      const NfruLogContextInfo&                       info,
                                      FfxSurfaceFormat                                colorHistoryFormat) const noexcept
{
    constexpr float jitterMagnitudeLimit = 0.5f;
    constexpr float jitterTolerance      = 0.000001f;
    if (std::fabs(desc.jitterOffset.x) > jitterMagnitudeLimit + jitterTolerance || std::fabs(desc.jitterOffset.y) > jitterMagnitudeLimit + jitterTolerance)
    {
        PrintFrameMessage(desc.frameID, FFX_MESSAGE_TYPE_WARNING, "jitterOffset contains value outside of expected range [-0.5, 0.5]");
    }

    char message[NFRU_LOG_MESSAGE_CAPACITY];
    std::snprintf(message,
                  sizeof(message),
                  "event=prepare_begin jitter=%.6gx%.6g mv_scale=%.6gx%.6g frame_time_delta_ms=%.6g fov_y=%.6g near=%.6g far=%.6g "
                  "view_space_to_meters=%.6g mv_similarity=%.6g mv_noise=%.6g mv_relative=%.6g",
                  desc.jitterOffset.x,
                  desc.jitterOffset.y,
                  desc.motionVectorScale.x,
                  desc.motionVectorScale.y,
                  desc.frameTimeDelta,
                  desc.cameraFovAngleVertical,
                  desc.cameraNear,
                  desc.cameraFar,
                  desc.viewSpaceToMetersFactor,
                  desc.mvSimilarityThreshold,
                  desc.mvSimilarityNoiseThreshold,
                  desc.mvSimilarityRelativeThreshold);
    PrintFrameMessage(desc.frameID, FFX_MESSAGE_TYPE_INFO, message);

    std::snprintf(message,
                  sizeof(message),
                  "event=prepare_resources role=current depth=%ux%u/%s depth_format_id=%u motion_vectors=%ux%u/%s motion_format_id=%u",
                  desc.depth.description.width,
                  desc.depth.description.height,
                  ffxGetSurfaceFormatName(ConvertEnum<FfxSurfaceFormat>(desc.depth.description.format)),
                  desc.depth.description.format,
                  desc.motionVectors.description.width,
                  desc.motionVectors.description.height,
                  ffxGetSurfaceFormatName(ConvertEnum<FfxSurfaceFormat>(desc.motionVectors.description.format)),
                  desc.motionVectors.description.format);
    PrintFrameMessage(desc.frameID, FFX_MESSAGE_TYPE_INFO, message);

    const bool managedDepth = info.usesInternalDepthHistory;
    const bool managedColor = (info.contextFlags & FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR) != 0;
    std::snprintf(message,
                  sizeof(message),
                  "event=prepare_resources role=history depth=%ux%u/%s depth_format_id=%u depth_source=%s color=%ux%u/%s "
                  "color_format_id=%u color_source=%s",
                  managedDepth ? info.resolutionPlan->mvDepthLane.width : desc.depthTm1.description.width,
                  managedDepth ? info.resolutionPlan->mvDepthLane.height : desc.depthTm1.description.height,
                  managedDepth ? "R32_FLOAT" : ffxGetSurfaceFormatName(ConvertEnum<FfxSurfaceFormat>(desc.depthTm1.description.format)),
                  managedDepth ? static_cast<unsigned>(FFX_SURFACE_FORMAT_R32_FLOAT) : desc.depthTm1.description.format,
                  managedDepth ? "managed" : "application",
                  managedColor ? info.resolutionPlan->fullResColour.width : desc.colorTm1.description.width,
                  managedColor ? info.resolutionPlan->fullResColour.height : desc.colorTm1.description.height,
                  ffxGetSurfaceFormatName(colorHistoryFormat),
                  static_cast<unsigned>(colorHistoryFormat),
                  managedColor ? "managed" : "application");
    PrintFrameMessage(desc.frameID, FFX_MESSAGE_TYPE_INFO, message);
}

void NfruDiagnostics::LogPrepareCompleted(uint64_t frameId, double cpuPrepareUs) const noexcept
{
    char message[NFRU_LOG_MESSAGE_CAPACITY];
    std::snprintf(message, sizeof(message), "event=prepare_end result=prepared error_code=0 error_name=FFX_OK cpu_prepare_us=%.6g", cpuPrepareUs);
    PrintFrameMessage(frameId, FFX_MESSAGE_TYPE_INFO, message);
}

void NfruDiagnostics::LogDispatchBegin(const ffxApiDispatchDescFrameGeneration& desc,
                                       const NfruLogContextInfo&                info,
                                       bool                                     effectiveReset,
                                       const char*                              resetReason) const noexcept
{
    char message[NFRU_LOG_MESSAGE_CAPACITY];
    std::snprintf(message,
                  sizeof(message),
                  "event=dispatch_begin reset_requested=%u generated_frames=%u flags=0x%08x debug_view=%u",
                  static_cast<unsigned>(desc.reset),
                  desc.numGeneratedFrames,
                  info.frameGenerationFlags,
                  static_cast<unsigned>((info.frameGenerationFlags & FFX_API_FG_DISPATCH_FLAG_DRAW_DEBUG_VIEW) != 0));
    PrintFrameMessage(desc.frameID, FFX_MESSAGE_TYPE_INFO, message);

    const FfxApiResourceDescription& present = desc.presentColor.description;
    const FfxApiResourceDescription& output  = desc.outputs[0].description;
    std::snprintf(message,
                  sizeof(message),
                  "event=dispatch_resources present_color=%ux%u/%s present_format_id=%u output=%ux%u/%s output_format_id=%u "
                  "optical_flow=%ux%u/R16G16_FLOAT",
                  present.width,
                  present.height,
                  ffxGetSurfaceFormatName(ConvertEnum<FfxSurfaceFormat>(present.format)),
                  present.format,
                  output.width,
                  output.height,
                  ffxGetSurfaceFormatName(ConvertEnum<FfxSurfaceFormat>(output.format)),
                  output.format,
                  info.opticalFlowSize.width,
                  info.opticalFlowSize.height);
    PrintFrameMessage(desc.frameID, FFX_MESSAGE_TYPE_INFO, message);

    const FfxNfruResolutionPlan& plan = *info.resolutionPlan;
    std::snprintf(message,
                  sizeof(message),
                  "event=dispatch_pipeline path=%s pre_of=%s pre_warp=%s mv_hints=%u warp=%s fill_holes=%s preprocess=%s "
                  "data_graph=enabled inpaint=compute postprocess=%s debug_view=%s effective_reset=%u reset_reason=%s",
                  effectiveReset ? "reset_copy" : "interpolate",
                  getPreOfName(plan),
                  getPreWarpName(plan),
                  static_cast<unsigned>((info.contextFlags & FFX_API_FG_CONTEXT_FLAG_ENABLE_MV_HINTS) != 0),
                  (info.contextFlags & FFX_API_FG_CONTEXT_FLAG_WARP_COMPUTE) ? "compute" : "fragment",
                  (info.contextFlags & FFX_API_FG_CONTEXT_FLAG_FILL_HOLES_COMPUTE) ? "compute" : "fragment",
                  (info.contextFlags & FFX_API_FG_CONTEXT_FLAG_PRE_PROCESS_COMPUTE) ? "compute" : "fragment",
                  (info.contextFlags & FFX_API_FG_CONTEXT_FLAG_POSTPROCESS_COMPUTE) ? "compute" : "fragment",
                  (info.frameGenerationFlags & FFX_API_FG_DISPATCH_FLAG_DRAW_DEBUG_VIEW) ? "on" : "off",
                  static_cast<unsigned>(effectiveReset),
                  resetReason);
    PrintFrameMessage(desc.frameID, FFX_MESSAGE_TYPE_INFO, message);
}

void NfruDiagnostics::LogDispatchCompleted(uint64_t frameId, const NfruCpuProfile& profile) const noexcept
{
    char message[NFRU_LOG_MESSAGE_CAPACITY];
    std::snprintf(message,
                  sizeof(message),
                  "event=cpu_profile cpu_prepare_us=%.6g cpu_of_colour_dispatch_us=%.6g cpu_optical_flow_dispatch_us=%.6g "
                  "cpu_fi_pre_schedule_us=%.6g cpu_fi_graph_schedule_us=%.6g cpu_fi_post_schedule_us=%.6g "
                  "cpu_fi_dispatch_us=%.6g cpu_dispatch_us=%.6g",
                  profile.prepareUs,
                  profile.opticalFlowColourUs,
                  profile.opticalFlowUs,
                  profile.frameInterpolationPreScheduleUs,
                  profile.frameInterpolationGraphScheduleUs,
                  profile.frameInterpolationPostScheduleUs,
                  profile.frameInterpolationDispatchUs,
                  profile.dispatchUs);
    PrintFrameMessage(frameId, FFX_MESSAGE_TYPE_INFO, message);
    PrintFrameMessage(frameId, FFX_MESSAGE_TYPE_INFO, "event=dispatch_end stage=record_commands result=recorded error_code=0 error_name=FFX_OK");
}

void NfruDiagnostics::LogFrameSizeMismatch(
    uint64_t frameId, uint32_t presentWidth, uint32_t presentHeight, uint32_t outputWidth, uint32_t outputHeight) const noexcept
{
    char message[192] = {};
    std::snprintf(message,
                  sizeof(message),
                  "Frame generation input and output sizes must match: presentColor is %u x %u, outputs[0] is %u x %u.",
                  presentWidth,
                  presentHeight,
                  outputWidth,
                  outputHeight);
    PrintFrameMessage(frameId, FFX_MESSAGE_TYPE_WARNING, message);
}
