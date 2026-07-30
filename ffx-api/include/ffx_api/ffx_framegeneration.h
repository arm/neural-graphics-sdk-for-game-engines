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

#include "ffx_api.h"
#include "ffx_api_types.h"

/// @ingroup ffxFrameGeneration
#define FFX_API_EFFECT_ID_FRAMEGENERATION 0x00020000u

#if defined(__cplusplus)
extern "C" {
#endif

/// @ingroup ffxFrameGeneration
enum FfxApiCreateContextFramegenerationFlags
{
    FFX_API_FG_CONTEXT_FLAG_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS =
        (1 << 0),  ///< A bit indicating if the motion vectors are rendered at display resolution.
    FFX_API_FG_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION =
        (1 << 1),                                              ///< A bit indicating that the motion vectors have the jittering pattern applied to them.
    FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED = (1 << 2),  ///< A bit indicating that the input depth buffer data provided is inverted [1..0].
    FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INFINITE = (1 << 3),  ///< A bit indicating that the input depth buffer data provided is using an infinite far plane.
    FFX_API_FG_CONTEXT_FLAG_ENABLE_HIGH_DYNAMIC_RANGE =
        (1 << 4),  ///< A bit indicating if the input color data provided to all inputs is using a high-dynamic range.
    FFX_API_FG_CONTEXT_FLAG_WARP_FRAGMENT        = (1 << 5),  ///< A bit indicating that warp stages (MV TP1 + flow TM1) should run as fragment jobs
    FFX_API_FG_CONTEXT_FLAG_FILL_HOLES_FRAGMENT  = (1 << 6),  ///< A bit indicating that fill holes stages (TP1 + TM1) should run as fragment jobs
    FFX_API_FG_CONTEXT_FLAG_PRE_PROCESS_FRAGMENT = (1 << 7),  ///< A bit indicating that the pre-process stage should run as fragment job when supported
    FFX_API_FG_CONTEXT_FLAG_POSTPROCESS_FRAGMENT = (1 << 8),  ///< A bit indicating that the postprocess stage should run as fragment job when supported
    FFX_API_FG_CONTEXT_FLAG_DEBUG_VIEW_FRAGMENT  = (1 << 9),  ///< A bit indicating that the debug view stage should run as fragment job when supported
    FFX_API_FG_CONTEXT_FLAG_ALL_STAGES_FRAGMENT  = FFX_API_FG_CONTEXT_FLAG_WARP_FRAGMENT | FFX_API_FG_CONTEXT_FLAG_FILL_HOLES_FRAGMENT |
                                                  FFX_API_FG_CONTEXT_FLAG_PRE_PROCESS_FRAGMENT | FFX_API_FG_CONTEXT_FLAG_POSTPROCESS_FRAGMENT |
                                                  FFX_API_FG_CONTEXT_FLAG_DEBUG_VIEW_FRAGMENT,  ///< All stages run as fragment jobs when supported
    FFX_API_FG_CONTEXT_FLAG_ALL_STAGES_COMPUTE = 0,                                             ///< All stages run as compute jobs (no fragment bits set)
    FFX_API_FG_CONTEXT_FLAG_MV_HINTS_FRAGMENT  = (1 << 11),  ///< A bit indicating that motion vector hints will be generated in a fragment shader

    FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH = (1 << 17),  ///< A bit indicating that the SDK manages the previous depth internally.
    FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR = (1 << 18),  ///< A bit indicating that the SDK manages the previous color internally.
};

/// @ingroup ffxFrameGeneration
enum FfxApiDispatchFramegenerationFlags
{
    FFX_API_FG_DISPATCH_FLAG_DRAW_DEBUG_VIEW =
        (1 << 0),  ///< A bit indicating that the generated output resource will contain debug views with relevant information.
    FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY =
        (1 << 1),  ///< A bit indicating that the context should only run frame interpolation and not modify the swapchain.
    FFX_API_FG_DISPATCH_FLAG_DUMP_GENERATED_FRAME = (1 << 2),  ///< A bit indicating that the generated frame will be dumped to disk.
};

/// @ingroup ffxFrameGeneration
#define FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION 0x00020001u

/// @ingroup ffxFrameGeneration
struct ffxApiCreateContextDescFrameGeneration
{
    ffxCreateContextDescHeader header;
    uint32_t                   flags;                  ///< A combination of zero or more values from FfxApiCreateContextFramegenerationFlags.
    struct FfxApiDimensions2D  displaySize;            ///< The resolution at which both rendered and generated frames will be displayed.
    struct FfxApiDimensions2D  renderSize;             ///< The maximum rendering resolution.
    FfxApiFloat32x4x4          initialViewProjection;  ///< The initial view-projection matrix used to initialize internal state.
    uint32_t                   backBufferFormat;       ///< The surface format for the backbuffer. One of the values from FfxApiSurfaceFormat.
    ffxApiMessage              fpMessage;              ///< A pointer to a function that can receive messages from the runtime. May be null.
};

/// @ingroup ffxFrameGeneration
#define FFX_API_CALLBACK_DESC_TYPE_FRAMEGENERATION_PRESENT 0x00020005u

/// @ingroup ffxFrameGeneration
struct ffxApiCallbackDescFrameGenerationPresent
{
    ffxDispatchDescHeader header;
    void*                 device;                 ///< The device passed in (from a backend description) during context creation.
    void*                 commandList;            ///< A command list that will be executed before presentation.
    struct FfxApiResource currentBackBuffer;      ///< Backbuffer image either rendered or generated.
    struct FfxApiResource currentUI;              ///< UI image for composition if passed. Otherwise empty.
    struct FfxApiResource outputSwapChainBuffer;  ///< Output image that will be presented.
    bool                  isGeneratedFrame;       ///< true if this frame is generated, false if rendered.
    uint64_t
        frameID;  ///< Identifier used to select internal resources when async support is enabled. Must increment by exactly one (1) for each frame. Any non-exactly-one difference will reset the frame generation logic.
};

/// @ingroup ffxFrameGeneration
#define FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION 0x00020003u

/// @ingroup ffxFrameGeneration
struct ffxApiDispatchDescFrameGeneration
{
    ffxDispatchDescHeader header;
    void*                 commandList;         ///< The command list on which to register render commands.
    struct FfxApiResource presentColor;        ///< The current presentation color, this will be used as source data.
    struct FfxApiResource outputs[4];          ///< Destination targets (1 for each frame in numGeneratedFrames).
    uint32_t              numGeneratedFrames;  ///< The number of frames to generate from the passed in color target.
    bool                  reset;               ///< A boolean value which when set to true, indicates the camera has moved discontinuously.
    uint64_t
        frameID;  ///< Identifier used to select internal resources when async support is enabled. Must increment by exactly one (1) for each frame. Any non-exactly-one difference will reset the frame generation logic.
};

typedef ffxReturnCode_t (*FfxApiCallbackFrameGenerationPresentFunc)(ffxApiCallbackDescFrameGenerationPresent* params, void* pUserCtx);
typedef ffxReturnCode_t (*FfxApiDispatchFrameGenerationFunc)(ffxApiDispatchDescFrameGeneration* params, void* pUserCtx);

/// @ingroup ffxFrameGeneration
#define FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION 0x00020002u

/// @ingroup ffxFrameGeneration
struct ffxApiConfigureDescFrameGeneration
{
    ffxConfigureDescHeader                   header;
    void*                                    swapChain;                           ///< The swapchain to use with frame generation.
    FfxApiCallbackFrameGenerationPresentFunc presentCallback;                     ///< A UI composition callback to call when finalizing the frame image.
    void*                                    presentCallbackUserContext;          ///< A pointer to be passed to the UI composition callback.
    FfxApiDispatchFrameGenerationFunc        frameGenerationCallback;             ///< The frame generation callback to use to generate a frame.
    void*                                    frameGenerationCallbackUserContext;  ///< A pointer to be passed to the frame generation callback.
    bool                                     frameGenerationEnabled;  ///< Sets the state of frame generation. Set to false to disable frame generation.
    bool     allowAsyncWorkloads;   ///< Sets the state of async workloads. Set to true to enable generation work on async compute.
    uint32_t flags;                 ///< Zero or combination of flags from FfxApiDispatchFrameGenerationFlags.
    bool     onlyPresentGenerated;  ///< Set to true to only present generated frames.
    uint64_t
                frameID;  ///< Identifier used to select internal resources when async support is enabled. Must increment by exactly one (1) for each frame. Any non-exactly-one difference will reset the frame generation logic.
    const char* dumpGeneratedFramePath;  ///< Path to dump generated frames if the corresponding flag is set.
};

/// @ingroup ffxFrameGeneration
#define FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE 0x00020004u

/// @ingroup ffxFrameGeneration
struct ffxApiDispatchDescFrameGenerationPrepare
{
    ffxDispatchDescHeader header;
    uint64_t
                               frameID;  ///< Identifier used to select internal resources when async support is enabled. Must increment by exactly one (1) for each frame. Any non-exactly-one difference will reset the frame generation logic.
    void*                      commandList;        ///< A command list to record frame generation commands into.
    struct FfxApiFloatCoords2D jitterOffset;       ///< The subpixel jitter offset applied to the camera.
    struct FfxApiFloatCoords2D motionVectorScale;  ///< The scale factor to apply to motion vectors.
    float
        mvSimilarityThreshold;  ///< Motion-vector similarity threshold (epsilon) used by dynamic-mask classification. <= 0 falls back to the legacy default (0.3).
    float
        mvSimilarityNoiseThreshold;  ///< Motion-vector noise threshold (tau) used by dynamic-mask classification. <= 0 falls back to the legacy default (1.0).

    float                 frameTimeDelta;           ///< Time elapsed in milliseconds since the last frame.
    float                 cameraNear;               ///< The distance to the near plane of the camera.
    float                 cameraFar;                ///< The distance to the far plane of the camera. This is used only used in case of non infinite depth.
    float                 cameraFovAngleVertical;   ///< The camera angle field of view in the vertical direction (expressed in radians).
    float                 viewSpaceToMetersFactor;  ///< The scale factor to convert view space units to meters
    struct FfxApiResource depth;                    ///< The depth buffer data
    struct FfxApiResource motionVectors;            ///< The motion vector data
    struct FfxApiResource
        depthTm1;  ///< The depth buffer data from the previous frame. Required when FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH is not set; ignored otherwise.
    struct FfxApiResource
        colorTm1;  ///< The rendered output from the previous frame. Required when FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR is not set; ignored otherwise.

    FfxApiFloat32x4x4 viewProjection;  ///< This frame's ViewProjection matrix (column major)
};

/// @ingroup ffxFrameGeneration
#define FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION_KEYVALUE 0x00020006u

/// @ingroup ffxFrameGeneration
struct ffxApiConfigureDescFrameGenerationKeyValue
{
    ffxConfigureDescHeader header;
    uint64_t               key;  ///< Configuration key, member of the FfxApiConfigureFrameGenerationKey enumeration.
    uint64_t               u64;  ///< Integer value or enum value to set.
    void*                  ptr;  ///< Pointer to set or pointer to value to set.
};

/// @ingroup ffxFrameGeneration
#define FFX_API_QUERY_DESC_TYPE_FG_GETPIPELINESTAGEINFO 0x00020007u  ///< header type for <c><i>ffxApiQueryDescFgGetPipelineStageInfo</i></c>.
/// @ingroup ffxFrameGeneration
struct ffxApiQueryDescFgGetPipelineStageInfo
{
    ffxQueryDescHeader header;
    bool*              pOutWarpUsesFragment;  ///< A pointer to a <c>bool</c> which will be set to true if warp stages run as fragment jobs. May be null.
    bool* pOutFillHolesUsesFragment;          ///< A pointer to a <c>bool</c> which will be set to true if fill holes stages run as fragment jobs. May be null.
    bool* pOutPreProcessUsesFragment;   ///< A pointer to a <c>bool</c> which will be set to true if the pre-process stage runs as a fragment job. May be null.
    bool* pOutPostprocessUsesFragment;  ///< A pointer to a <c>bool</c> which will be set to true if the postprocess stage runs as a fragment job. May be null.
    bool* pOutDebugViewUsesFragment;    ///< A pointer to a <c>bool</c> which will be set to true if the debug view stage runs as a fragment job. May be null.
};

#if defined(__cplusplus)
}  // extern "C"
#endif
