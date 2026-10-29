/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>

#include <ffx_api/ffx_framegeneration.h>
#include <FidelityFX/host/ffx_frameinterpolation_resolution.h>
#include <FidelityFX/host/ffx_types.h>

#include "ffx_frameinterpolation_trace.h"
#include "ffx_opticalflow_trace.h"

enum class NfruCapabilityFailure : uint32_t
{
    None                                  = 0,
    MissingDataGraphOpticalFlow           = 1u << 0,
    UnsupportedOpticalFlowGrid            = 1u << 1,
    OpticalFlowResolutionOutOfRange       = 1u << 2,
    FrameInterpolationPipelineUnsupported = 1u << 3,
};

struct NfruLogContextInfo
{
    const FfxNfruResolutionPlan* resolutionPlan;
    FfxDimensions2D              opticalFlowSize;
    uint32_t                     contextFlags;
    uint32_t                     frameGenerationFlags;
    bool                         usesInternalDepthHistory;
    bool                         fp16Supported;
};

struct NfruCpuProfile
{
    double prepareUs;
    double opticalFlowColourUs;
    double opticalFlowUs;
    double frameInterpolationPreScheduleUs;
    double frameInterpolationGraphScheduleUs;
    double frameInterpolationPostScheduleUs;
    double frameInterpolationDispatchUs;
    double dispatchUs;
};

class NfruDiagnostics
{
public:
    static NfruDiagnostics Create(ffxApiMessage callback) noexcept;

    bool HasError() const noexcept;
    void ClearError() noexcept;

    static void LogVersions(ffxApiMessage callback) noexcept;
    static void LogResolutionValidationError(ffxApiMessage callback, FfxNfruResolutionValidationError error) noexcept;
    static void LogUnsupportedBackBufferFormat(ffxApiMessage callback) noexcept;

    ffxReturnCode_t LogContextCreationError(const char*           source,
                                            const char*           stage,
                                            FfxErrorCode          errorCode,
                                            const char*           reasonCode,
                                            const char*           reason,
                                            ffxReturnCode_t       returnCode,
                                            NfruCapabilityFailure capabilityFailure = NfruCapabilityFailure::None) noexcept;

    ffxReturnCode_t LogRuntimeError(uint64_t        frameId,
                                    const char*     phase,
                                    const char*     source,
                                    const char*     stage,
                                    FfxErrorCode    errorCode,
                                    const char*     reasonCode,
                                    const char*     reason,
                                    bool            verbose,
                                    ffxReturnCode_t returnCode = FFX_API_RETURN_ERROR_RUNTIME_ERROR) noexcept;

    ffxReturnCode_t LogOpticalFlowError(uint64_t frameId, FfxNfruOfStage stage, FfxErrorCode errorCode, bool verbose) noexcept;
    ffxReturnCode_t LogFrameInterpolationError(uint64_t frameId, FfxNfruFiStage stage, FfxErrorCode errorCode, bool verbose) noexcept;

    void LogContextCreated(const ffxApiCreateContextDescFrameGeneration& desc,
                           const NfruLogContextInfo&                     info,
                           FfxOpticalFlowGridSize                        opticalFlowGrid) const noexcept;
    void LogContextDestroyed(FfxErrorCode errorCode, const char* source) const noexcept;
    void LogFrameGenerationDisabledOnDestroy() const noexcept;
    void LogFrameGenerationState(const ffxApiConfigureDescFrameGeneration& desc, bool enabled) const noexcept;
    void LogPrepareBegin(const ffxApiDispatchDescFrameGenerationPrepare& desc,
                         const NfruLogContextInfo&                       info,
                         FfxSurfaceFormat                                colorHistoryFormat) const noexcept;
    void LogPrepareCompleted(uint64_t frameId, double cpuPrepareUs) const noexcept;
    void LogDispatchBegin(const ffxApiDispatchDescFrameGeneration& desc,
                          const NfruLogContextInfo&                info,
                          bool                                     effectiveReset,
                          const char*                              resetReason) const noexcept;
    void LogDispatchCompleted(uint64_t frameId, const NfruCpuProfile& profile) const noexcept;
    void LogFrameSizeMismatch(uint64_t frameId, uint32_t presentWidth, uint32_t presentHeight, uint32_t outputWidth, uint32_t outputHeight) const noexcept;

private:
    struct ErrorSignature
    {
        FfxErrorCode code;
        const char*  phase;
        const char*  stage;
        const char*  reasonCode;
    };

    void PrintMessage(FfxMsgType type, const char* message) const noexcept;
    void PrintFrameMessage(uint64_t frameId, FfxMsgType type, const char* message) const noexcept;

    uint64_t       contextId_;
    ffxApiMessage  callback_;
    ErrorSignature lastError_;
};
