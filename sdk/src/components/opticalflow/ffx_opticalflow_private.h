// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#pragma once

#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_NULL                    0
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COLOR                   1
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_COLOR              2
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_DEPTH                   3
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_DEPTH              4
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COMPUTED_MOTION_VECTORS 5
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_RESULT                  6
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_DEPTH_1                 7
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_DEPTH_2                 8
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COLOR_1                 9
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COLOR_2                 10
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COUNT                   11

#define FFX_OPTICALFLOW_CONSTANTBUFFER_IDENTIFIER 0
#define FFX_OPTICALFLOW_CONSTANTBUFFER_COUNT      1

struct FfxPipelineState;

typedef enum OpticalFlowShaderPermutationOptions
{
    OPTICALFLOW_SHADER_PERMUTATION_DEPTH_INVERTED    = (1 << 0),
    OPTICALFLOW_SHADER_PERMUTATION_MV_HINTS_FRAGMENT = (1 << 1),
} OpticalFlowShaderPermutationOptions;

typedef struct OpticalFlowComputeMVHintsConstants
{
    float   motion_matrix_m1p1[16];
    int32_t output_dims[2];
    int32_t depth_size[2];
    float   depth_size_rcp[2];
    int32_t color_size[2];
} OpticalFlowComputeMVHintsConstants;

// The private implementation of the arm data graph optical flow context.
typedef struct OpticalFlowContext_Private
{
    FfxOpticalFlowContextDescription   contextDescription;
    FfxUInt32                          effectContextId;
    FfxDevice                          device;
    FfxPipelineState                   pipelineOpticalflow;
    FfxPipelineState                   pipelineComputeMVHints;
    FfxPipelineState                   pipelineFragmentMVHints;
    bool                               useMVHintsFragment;
    FfxResourceInternal                srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COUNT];
    FfxResourceInternal                uavResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COUNT];
    FfxConstantBuffer                  constantBuffers[FFX_OPTICALFLOW_CONSTANTBUFFER_COUNT];
    FfxDimensions2D                    dimensions;
    OpticalFlowComputeMVHintsConstants computeMVHintsConstants;
    FfxFloat32x4x4                     lastFrameViewProjection;
    FfxDimensions2D                    opticalFlowSize;

    bool     firstExecution;
    uint32_t resourceFrameIndex;
} OpticalFlowContext_Private;
