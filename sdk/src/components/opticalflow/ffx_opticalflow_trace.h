/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "FidelityFX/host/ffx_opticalflow.h"

enum class FfxNfruOfStage
{
    None,
    Validation,
    RegisterResources,
    ClearHistory,
    MotionVectorHints,
    DataGraph,
    RecordCommands,
    Unregister,
};

struct FfxNfruOpticalFlowTrace
{
    FfxNfruOfStage failedStage;
};

FFX_API FfxErrorCode ffxOpticalFlowContextCreateForFrameGeneration(FfxOpticalFlowContext* context, FfxOpticalFlowContextDescription* contextDescription);
FFX_API FfxErrorCode ffxOpticalFlowContextCreateForFrameGenerationWithDepthInput(FfxOpticalFlowContext*            context,
                                                                                 FfxOpticalFlowContextDescription* contextDescription,
                                                                                 FfxDimensions2D                   mvDepthInputSize);
FFX_API FfxErrorCode ffxOpticalFlowContextDestroyForFrameGeneration(FfxOpticalFlowContext* context);
FFX_API FfxErrorCode ffxOpticalFlowGetSharedResourceDescriptionsForFrameGeneration(FfxOpticalFlowContext*                    context,
                                                                                   FfxOpticalFlowSharedResourceDescriptions* sharedResources);
FFX_API FfxErrorCode ffxOpticalFlowContextDispatchWithTrace(FfxOpticalFlowContext*             context,
                                                            FfxOpticalFlowDispatchDescription* dispatchParams,
                                                            FfxNfruOpticalFlowTrace*           trace);
