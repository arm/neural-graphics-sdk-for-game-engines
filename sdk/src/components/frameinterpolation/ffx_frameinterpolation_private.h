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
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#pragma once

#include <FidelityFX/gpu/frameinterpolation/ffx_frameinterpolation_resources.h>
#include <FidelityFX/host/ffx_types.h>

/// An enumeration of all the permutations that can be passed to the FSR3 algorithm.
///
/// FSR3 features are organized through a set of pre-defined compile
/// permutation options that need to be specified. Which shader blob
/// is returned for pipeline creation will be determined by what combination
/// of shader permutations are enabled.
///
/// @ingroup ffxFrameGeneration
typedef enum FfxFrameInterpolationShaderPermutationOptions
{
    FRAMEINTERPOLATION_SHADER_PERMUTATION_LOW_RES_MOTION_VECTORS  = (1 << 0),
    FRAMEINTERPOLATION_SHADER_PERMUTATION_JITTER_MOTION_VECTORS   = (1 << 1),
    FRAMEINTERPOLATION_SHADER_PERMUTATION_DEPTH_INVERTED          = (1 << 2),   ///< Indicates input resources were generated with inverted depth
    FRAMEINTERPOLATION_SHADER_PERMUTATION_FORCE_WAVE64            = (1 << 3),   ///< doesn't map to a define, selects different table
    FRAMEINTERPOLATION_SHADER_PERMUTATION_ALLOW_FP16              = (1 << 4),   ///< Enables fast math computations where possible
    FRAMEINTERPOLATION_SHADER_PERMUTATION_ENABLE_DATA_GRAPH_FI    = (1 << 5),   ///< Use weight gen shader for device not support ML extensions for Vulkan
    FRAMEINTERPOLATION_SHADER_PERMUTATION_WARP_FRAGMENT           = (1 << 6),   ///< Run warp stages (MV TP1 + flow TM1) as fragment shaders
    FRAMEINTERPOLATION_SHADER_PERMUTATION_FILL_HOLES_FRAGMENT     = (1 << 7),   ///< Run fill holes stages (TP1 + TM1) as fragment shaders
    FRAMEINTERPOLATION_SHADER_PERMUTATION_PRE_PROCESS_FRAGMENT    = (1 << 8),   ///< Run preprocess as fragment shader when supported
    FRAMEINTERPOLATION_SHADER_PERMUTATION_POSTPROCESS_FRAGMENT    = (1 << 9),   ///< Run postprocess as fragment shader
    FRAMEINTERPOLATION_SHADER_PERMUTATION_DEBUG_VIEW_FRAGMENT     = (1 << 11),  ///< Run debug view as fragment shader
    FRAMEINTERPOLATION_SHADER_PERMUTATION_COMPUTE_SUPPORT_TENSOR  = (1 << 12),  ///< Indicates that tensor ops are supported in compute shader
    FRAMEINTERPOLATION_SHADER_PERMUTATION_FRAGMENT_SUPPORT_TENSOR = (1 << 13),  ///< Indicates that tensor ops are supported in fragment shader
    FRAMEINTERPOLATION_SHADER_PERMUTATION_MANAGE_PREVIOUS_DEPTH   = (1 << 14),  ///< Indicates that the sdk manages depth history internally
    FRAMEINTERPOLATION_SHADER_PERMUTATION_MANAGE_PREVIOUS_COLOR   = (1 << 15),  ///< Indicates that the sdk manages color history internally
} FfxInterpolationShaderPermutationOptions;

struct FfxPipelineState;
struct FfxResource;

typedef struct FrameInterpolationConstants
{
    FfxFloat32x4x4 _MotionTM1ToTP1;
    FfxFloat32x4x4 _MotionTP1ToTM1;

    FfxFloat32x4 _QuantParamsSINT;   ///< Quant parameters between float and sint. .xy for quantize, .zw for dequantize
    FfxFloat32x4 _QuantParamsSNORM;  ///< Quant parameters between float and snorm. .xy for quantize, .zw for dequantize

    float _DeviceToViewDepth[4];

    int32_t _RenderSize[2];
    int32_t _DisplaySize[2];
    int32_t _OfSize[2];

    float _RenderSizeRcp[2];
    float _DisplaySizeRcp[2];
    float _OfSizeRcp[2];

    float _Jitter[2];
    float _MotionVectorScale[2];
    float _MvSimilarityThresholds[2];  // x: similarity threshold (eps), y: noise threshold (tau)

    int32_t  _InputTensorSize[2];
    float    _InputTensorSizeRcp[2];
    int      _Reset;
    float    _Timestep;
    uint32_t _RandomSeed;
    uint32_t _pad0;  // std140 padding for ivec2 alignment

    int32_t _MvDepthLaneSize[2];
    float   _MvDepthLaneSizeRcp[2];
    int32_t _FlowLaneSize[2];
    float   _FlowLaneSizeRcp[2];
    float   _OfGridSizeRcp;  ///< Reciprocal of optical flow block/grid size (e.g. 0.25 for 4x4 blocks).
} FrameInterpolationConstants;

typedef struct FrameInterpolationRenderDescription
{
    FfxDimensions2D renderSize;
    FfxDimensions2D upscaleSize;

    float cameraNear;
    float cameraFar;
    float cameraFovAngleVertical;
    float viewSpaceToMetersFactor;

    FfxFloatCoords2D motionVectorScale;  ///< The scale factor to apply to motion vectors.
} FrameInterpolationRenderDescription;

// FfxFsr3Context_Private
// The private implementation of the FSR3 context.
typedef struct FfxArmFrameInterpolationContext_Private
{
    FfxFrameInterpolationContextDescription contextDescription;
    FfxUInt32                               effectContextId;
    FrameInterpolationRenderDescription     renderDescription;
    FrameInterpolationConstants             constants;
    FfxDevice                               device;

    // Canonical resolution lane model: cap internal resource dimensions
    FfxDimensions2D ofInputColour;             // Capped OF input colour resolution (≤1080p height)
    FfxDimensions2D mvDepthLane;               // MV/depth processing lane (capped at 540p height)
    FfxDimensions2D flowLane;                  // Optical flow lane (capped at 270p height)
    FfxDimensions2D tensorLane;                // Neural network tensor lane (capped at 270p height)
    bool            requiresColourDownsample;  // true when displaySize.height > 1080

    // FrameInterpolation Pipelines
    FfxPipelineState pipelineDebugView;

    // NFRU Pipeline States
    FfxPipelineState pipelineDownsampleOfColour;  // OF colour downsample (conditional, only when requiresColourDownsample)
    FfxPipelineState pipelineInitWarp;
    FfxPipelineState pipelineWarpFlow;
    FfxPipelineState pipelineWarpFlowTM1;
    FfxPipelineState pipelineFillHoles;
    FfxPipelineState pipelineFillHolesTM1;
    FfxPipelineState pipelinePreprocess;
    FfxPipelineState pipelineNFRUInterpolation;
    FfxPipelineState pipelinePostprocess;

    FfxConstantBuffer constantBuffers[FFX_FRAMEINTERPOLATION_CONSTANTBUFFER_COUNT];

    // 2 arrays of resources, as e.g. FFX_FSR3_RESOURCE_IDENTIFIER_LOCK_STATUS will use different resources when bound as SRV vs when bound as UAV
    FfxResourceInternal srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNT];
    FfxResourceInternal uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNT];

    bool firstExecution;
    bool refreshPipelineStates;

    FfxDataGraphTensorInfo dataGraphTensorInfo[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNT];
    uint32_t               dataGraphTensorInfoCount;

    uint32_t pipelineFlags;
    uint32_t dataGraphWidth;
    uint32_t dataGraphHeight;

    uint64_t       previousFrameID;
    uint64_t       dispatchCount;
    FfxFloat32x4x4 lastFrameViewProjection;

    bool enableDataGraphFI = true;
} FfxArmFrameInterpolationContext_Private;
