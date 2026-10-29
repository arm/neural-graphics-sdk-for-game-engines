/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <FidelityFX/host/ffx_frameinterpolation.h>

enum class FfxNfruFiStage
{
    None,
    Validation,
    PipelineRefresh,
    Preprocess,
    DataGraph,
    Postprocess,
    RecordCommands,
    Unregister,
};

struct FfxNfruFrameInterpolationTrace
{
    FfxNfruFiStage failedStage;
    bool           collectTimings;
    double         preScheduleUs;
    double         graphScheduleUs;
    double         postScheduleUs;
};

FFX_API void ffxLogNfruVersion(FfxLogMessageCallback callback = nullptr) noexcept;

FFX_API FfxErrorCode ffxFrameInterpolationContextCreateForFrameGeneration(FfxFrameInterpolationContext*            context,
                                                                          FfxFrameInterpolationContextDescription* contextDescription);
FFX_API FfxErrorCode ffxFrameInterpolationContextDestroyForFrameGeneration(FfxFrameInterpolationContext* context);
FFX_API FfxErrorCode ffxFrameInterpolationPrepareForFrameGeneration(FfxFrameInterpolationContext*                  context,
                                                                    const FfxFrameInterpolationPrepareDescription* params);
FFX_API FfxErrorCode ffxFrameInterpolationDispatchWithTrace(FfxFrameInterpolationContext*                   context,
                                                            const FfxFrameInterpolationDispatchDescription* params,
                                                            FfxNfruFrameInterpolationTrace*                 trace);
FFX_API FfxErrorCode ffxFrameInterpolationDispatchDownsampleOfColourForFrameGeneration(FfxFrameInterpolationContext* context,
                                                                                       FfxCommandList                commandList,
                                                                                       FfxResource                   currentColour);
