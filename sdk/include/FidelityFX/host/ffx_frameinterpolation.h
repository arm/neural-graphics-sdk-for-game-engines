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

#pragma once

// Include the interface for the backend of the Frameinterpolation API.
#include <FidelityFX/host/ffx_interface.h>

/// NFRU component major version.
///
/// @ingroup ffxFrameGeneration
#define FFX_FRAMEINTERPOLATION_VERSION_MAJOR (2)

/// NFRU component minor version.
///
/// @ingroup ffxFrameGeneration
#define FFX_FRAMEINTERPOLATION_VERSION_MINOR (0)

/// NFRU component patch version.
///
/// @ingroup ffxFrameGeneration
#define FFX_FRAMEINTERPOLATION_VERSION_PATCH (0)

/// FidelityFX Frame Interpolation context count
///
/// Defines the number of internal effect contexts required by Frame Interpolation
///
/// @ingroup ffxFrameInterpolation
#define FFX_FRAMEINTERPOLATION_CONTEXT_COUNT (1)

/// The size of the context specified in 32bit values.
///
/// @ingroup ffxFrameGeneration
#define FFX_FRAMEINTERPOLATION_CONTEXT_SIZE (FFX_SDK_DEFAULT_CONTEXT_SIZE)

#if defined(__cplusplus)
extern "C" {
#endif  // #if defined(__cplusplus)

/// An enumeration of all the passes which constitute the FSR3 algorithm.
///
/// FSR3 is implemented as a composite of several compute passes each
/// computing a key part of the final result. Each call to the
/// <c><i>FfxFsr3ScheduleGpuJobFunc</i></c> callback function will
/// correspond to a single pass included in <c><i>FfxFsr3Pass</i></c>. For a
/// more comprehensive description of each pass, please refer to the FSR3
/// reference documentation.
///
/// Please note in some cases e.g.: <c><i>FFX_FSR3_PASS_ACCUMULATE</i></c>
/// and <c><i>FFX_FSR3_PASS_ACCUMULATE_SHARPEN</i></c> either one pass or the
/// other will be used (they are mutually exclusive). The choice of which will
/// depend on the way the <c><i>FfxFsr3Context</i></c> is created and the
/// precise contents of <c><i>FfxFsr3DispatchParamters</i></c> each time a call
/// is made to <c><i>ffxFsr3ContextDispatch</i></c>.
///
/// @ingroup ffxFrameGeneration
typedef enum FfxFrameInterpolationPass
{
    FFX_FRAMEINTERPOLATION_PASS_INIT_WARP,                // 00_init_warp
    FFX_FRAMEINTERPOLATION_PASS_DOWNSAMPLE_OF_COLOUR,     // 01_downsample_of_colour
    FFX_FRAMEINTERPOLATION_PASS_DOWNSAMPLE_ALL_INPUT,     // 02_downsample_all_input
    FFX_FRAMEINTERPOLATION_PASS_DOWNSAMPLE_COLOUR,        // 03_downsample_colour
    FFX_FRAMEINTERPOLATION_PASS_WARP_MV_TP1,              // 10_warp_flow
    FFX_FRAMEINTERPOLATION_PASS_WARP_FLOW_TM1,            // 11_warp_flow_tm1
    FFX_FRAMEINTERPOLATION_PASS_FILL_HOLES_TM1,           // 21_fill_holes_tm1
    FFX_FRAMEINTERPOLATION_PASS_PREPROCESS,               // 30_preprocess
    FFX_FRAMEINTERPOLATION_PASS_NFRU_INTERPOLATION,       // 40_data_graph_weight_gen
    FFX_FRAMEINTERPOLATION_PASS_DOWNSAMPLE_INPAINT_SEED,  // 31_downsample_inpaint (seed -> level 0)
    FFX_FRAMEINTERPOLATION_PASS_DOWNSAMPLE_INPAINT_MIP,   // 31_downsample_inpaint (level i-1 -> i)
    FFX_FRAMEINTERPOLATION_PASS_INPAINT,                  // 50_inpaint (inpaint resolver)
    FFX_FRAMEINTERPOLATION_PASS_POSTPROCESS,              // 50_postprocess
    FFX_FRAMEINTERPOLATION_PASS_DEBUG_VIEW,
    FFX_FRAMEINTERPOLATION_PASS_COUNT  ///< The number of passes performed by FrameInterpolation.
} FfxFrameInterpolationPass;

// forward declarations
struct FfxFrameInterpolationContext;

/// An enumeration of bit flags used when creating a
/// <c><i>FfxFrameInterpolationContext</i></c>. See <c><i>FfxFrameInterpolationContextDescription</i></c>.
///
/// @ingroup ffxFrameGeneration
typedef enum FfxFrameInterpolationInitializationFlagBits
{

    FFX_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED = (1 << 0),  ///< A bit indicating that the input depth buffer data provided is inverted [1..0].
    FFX_FG_CONTEXT_FLAG_ENABLE_DEPTH_INFINITE = (1 << 1),  ///< A bit indicating that the input depth buffer data provided is using an infinite far plane.
    FFX_ARM_FRAMEINTERPOLATION_ENABLE_TEXTURE1D_USAGE             = (1 << 2),  ///< A bit indicating that the backend should use 1D textures.
    FFX_FG_CONTEXT_FLAG_ENABLE_HIGH_DYNAMIC_RANGE                 = (1 << 3),  ///< A bit indicating that HDR values are present in the imaging pipeline.
    FFX_FG_CONTEXT_FLAG_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS  = (1 << 4),  ///< A bit indicating if the motion vectors are rendered at display resolution.
    FFX_FG_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION = (1 << 5),
    FFX_FG_CONTEXT_FLAG_WARP_COMPUTE =
        (1 << 6),  ///< A bit indicating that warp stages (MV TP1 + flow TM1) should run as compute jobs. When unset, fragment jobs are used.
    FFX_FG_CONTEXT_FLAG_FILL_HOLES_COMPUTE =
        (1 << 7),  ///< A bit indicating that fill holes stages (TP1 + TM1) should run as compute jobs. When unset, fragment jobs are used.
    FFX_FG_CONTEXT_FLAG_PRE_PROCESS_COMPUTE =
        (1 << 8),  ///< A bit indicating that the preprocess stage should run as a compute job. When unset, the fragment job is used.
    FFX_FG_CONTEXT_FLAG_POSTPROCESS_COMPUTE =
        (1 << 9),  ///< A bit indicating that the postprocess stage should run as a compute job. When unset, the fragment job is used.
    FFX_FG_CONTEXT_FLAG_DEBUG_VIEW_COMPUTE =
        (1 << 11),  ///< A bit indicating that the debug view stage should run as a compute job. When unset, the fragment job is used.
    FFX_FG_CONTEXT_FLAG_ALL_STAGES_COMPUTE = FFX_FG_CONTEXT_FLAG_WARP_COMPUTE | FFX_FG_CONTEXT_FLAG_FILL_HOLES_COMPUTE |
                                             FFX_FG_CONTEXT_FLAG_PRE_PROCESS_COMPUTE | FFX_FG_CONTEXT_FLAG_POSTPROCESS_COMPUTE |
                                             FFX_FG_CONTEXT_FLAG_DEBUG_VIEW_COMPUTE,  ///< Convenience: set all selectable stages to compute.
    FFX_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH = (1 << 15),                            ///< A bit indicating that the SDK manages the previous depth internally.
    FFX_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR = (1 << 16),                            ///< A bit indicating that the SDK manages the previous color internally.
} FfxFrameInterpolationInitializationFlagBits;

/// Pass a string message
///
/// Used for debug messages.
///
/// @param [in] type                       The type of message.
/// @param [in] message                    A string message to pass.
///
///
/// @ingroup ffxFi
typedef void (*FfxFiMessage)(FfxMsgType type, const char* message);

/// A structure encapsulating the parameters required to initialize
/// FidelityFX Frameinterpolation upscaling.
///
/// @ingroup ffxFrameGeneration
typedef struct FfxFrameInterpolationContextDescription
{
    uint32_t               flags;          ///< A collection of <c><i>FfxFrameInterpolationInitializationFlagBits</i></c>.
    FfxDimensions2D        maxRenderSize;  ///< Maximum render size; dimensions must fit the signed 32-bit shader range.
    FfxDimensions2D        displaySize;    ///< Presentation resolution; dimensions must fit the signed 32-bit shader range.
    FfxSurfaceFormat       backBufferFormat;
    FfxInterface           backendInterface;       ///< A set of pointers to the backend implementation for FidelityFX SDK
    FfxFloat32x4x4         initialViewProjection;  ///< The initial view-projection matrix used to initialize internal state.
    FfxDimensions2D        opticalFlowSize;        ///< The optical flow size.
    FfxOpticalFlowGridSize opticalFlowGridSize;    ///< Optical flow block/grid size (1, 2, 4 or 8). Used to convert flow vectors to UV space on the GPU.
    FfxFiMessage           fpMessage;              ///< A pointer to a function that can receive messages from the runtime.
    FfxDimensions2D
        mvDepthSize;  ///< Shared input depth/motion-vector resolution. {0, 0} defaults to maxRenderSize. Dimensions must fit the signed 32-bit shader range and have the same aspect ratio as maxRenderSize.
} FfxFrameInterpolationContextDescription;

/// A structure encapsulating the FidelityFX Super Resolution 2 context.
///
/// This sets up an object which contains all persistent internal data and
/// resources that are required by FSR3.
///
/// The <c><i>FfxFsr3Context</i></c> object should have a lifetime matching
/// your use of FSR3. Before destroying the FSR3 context care should be taken
/// to ensure the GPU is not accessing the resources created or used by FSR3.
/// It is therefore recommended that the GPU is idle before destroying the
/// FSR3 context.
///
/// @ingroup ffxFrameGeneration
typedef struct FfxFrameInterpolationContext
{
    uint32_t data[FFX_FRAMEINTERPOLATION_CONTEXT_SIZE];  ///< An opaque set of <c>uint32_t</c> which contain the data for the context.
} FfxFrameInterpolationContext;

/// Create a FidelityFX Super Resolution 2 context from the parameters
/// programmed to the <c><i>FfxFsr3CreateParams</i></c> structure.
///
/// The context structure is the main object used to interact with the FSR3
/// API, and is responsible for the management of the internal resources used
/// by the FSR3 algorithm. When this API is called, multiple calls will be
/// made via the pointers contained in the <c><i>callbacks</i></c> structure.
/// These callbacks will attempt to retreive the device capabilities, and
/// create the internal resources, and pipelines required by FSR3's
/// frame-to-frame function. Depending on the precise configuration used when
/// creating the <c><i>FfxFsr3Context</i></c> a different set of resources and
/// pipelines might be requested via the callback functions.
///
/// The flags included in the <c><i>flags</i></c> field of
/// <c><i>FfxFsr3Context</i></c> how match the configuration of your
/// application as well as the intended use of FSR3. It is important that these
/// flags are set correctly (as well as a correct programmed
/// <c><i>FfxFsr3DispatchDescription</i></c>) to ensure correct operation. It is
/// recommended to consult the overview documentation for further details on
/// how FSR3 should be integerated into an application.
///
/// When the <c><i>FfxFsr3Context</i></c> is created, you should use the
/// <c><i>ffxFsr3ContextDispatch</i></c> function each frame where FSR3
/// upscaling should be applied. See the documentation of
/// <c><i>ffxFsr3ContextDispatch</i></c> for more details.
///
/// The <c><i>FfxFsr3Context</i></c> should be destroyed when use of it is
/// completed, typically when an application is unloaded or FSR3 upscaling is
/// disabled by a user. To destroy the FSR3 context you should call
/// <c><i>ffxFsr3ContextDestroy</i></c>.
///
/// @param [out] context                A pointer to a <c><i>FfxFsr3Context</i></c> structure to populate.
/// @param [in]  contextDescription     A pointer to a <c><i>FfxFsr3ContextDescription</i></c> structure.
///
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_CODE_NULL_POINTER         The operation failed because either <c><i>context</i></c> or <c><i>contextDescription</i></c> was <c><i>NULL</i></c>.
/// @retval
/// FFX_ERROR_INCOMPLETE_INTERFACE      The operation failed because the <c><i>FfxFsr3ContextDescription.callbacks</i></c>  was not fully specified.
/// @retval
/// FFX_ERROR_BACKEND_API_ERROR         The operation failed because of an error returned from the backend.
///
/// @ingroup ffxFrameGeneration
FFX_API FfxErrorCode ffxFrameInterpolationContextCreate(FfxFrameInterpolationContext* context, FfxFrameInterpolationContextDescription* contextDescription);

FFX_API FfxErrorCode ffxFrameInterpolationContextGetGpuMemoryUsage(FfxFrameInterpolationContext* pContext, FfxEffectMemoryUsage* vramUsage);

typedef struct FfxFrameInterpolationPrepareDescription
{
    FfxCommandList   commandList;   ///< The <c><i>FfxCommandList</i></c> to record frame interpolation commands into.
    FfxFloatCoords2D jitterOffset;  ///< Sample-position offset in maxRenderSize pixels, matching the upscaler for this frameID; not raster displacement.
    FfxFloatCoords2D
          motionVectorScale;  ///< Converts stored motion to mvDepthSize pixel offsets (displaySize pixels when the display-resolution motion-vector flag is set).
    float mvSimilarityThreshold;  ///< Motion-vector similarity threshold (epsilon) used by dynamic-mask classification. <= 0 falls back to the default (0.25).
    float mvSimilarityNoiseThreshold;     ///< Motion-vector noise threshold (tau) used by dynamic-mask classification. <= 0 falls back to the default (0.125).
    float mvSimilarityRelativeThreshold;  ///< Relative motion-vector threshold; <= 0 uses 0.002, while NaN and infinities are rejected.

    float frameTimeDelta;
    float cameraNear;
    float cameraFar;
    float viewSpaceToMetersFactor;
    float cameraFovAngleVertical;

    FfxResource depth;           ///< The depth buffer data
    FfxResource motionVectors;   ///< The motion vector data
    FfxResource depthTm1;        ///< The depth buffer data from the previous frame (read)
    FfxResource depthTm1Next;    ///< The depth buffer write target for this frame; becomes depth_tm1 in the next iteration (written by warp pass)
    FfxResource colorTm1;        ///< The color buffer data from the previous frame (read by the in-flight interpolation pass)
    FfxResource colorTm1Backup;  ///< Ping-pong backup of @ref colorTm1, written this frame so async interpolation can keep reading @ref colorTm1
    uint64_t    frameID;

    FfxFloat32x4x4 viewProjection;  ///< This frame's ViewProjection matrix (column major)
} FfxFrameInterpolationPrepareDescription;

FFX_API FfxErrorCode ffxFrameInterpolationPrepare(FfxFrameInterpolationContext* context, const FfxFrameInterpolationPrepareDescription* params);

typedef enum FfxFrameInterpolationDispatchFlags
{
    FFX_FRAMEINTERPOLATION_DISPATCH_DRAW_DEBUG_VIEW =
        (1 << 0),  ///< A bit indicating that the interpolated output resource will contain debug views with relevant information.
} FfxFrameInterpolationDispatchFlags;

typedef struct FfxFrameInterpolationDispatchDescription
{
    uint32_t       flags;              ///< combination of FfxFrameInterpolationDispatchFlags
    FfxCommandList commandList;        ///< The <c><i>FfxCommandList</i></c> to record frame interpolation commands into.
    FfxResource    currentBackBuffer;  ///< The current presentation color.
    FfxResource    output;             ///< The output resource where to store the interpolated result.

    FfxResource      opticalFlowVector;      ///< The optical flow motion vectors (see example computation in the FfxOpticalFlow effect)
    FfxDimensions2D  opticalFlowBufferSize;  ///< The optical flow motion vector resource dimensions
    FfxFloatCoords2D opticalFlowScale;       ///< The optical flow motion vector scale factor, used to scale resoure values into [0.0,1.0] range.

    float cameraNear;               ///< The distance to the near plane of the camera.
    float cameraFar;                ///< The distance to the far plane of the camera. This is used only used in case of non infinite depth.
    float cameraFovAngleVertical;   ///< The camera angle field of view in the vertical direction (expressed in radians).
    float viewSpaceToMetersFactor;  ///< The unit to scale view space coordinates to meters.

    float frameTimeDelta;  ///< The time elapsed since the last frame (expressed in milliseconds).
    bool  reset;           ///< A boolean value which when set to true, indicates the camera has moved discontinuously.

    uint64_t
        frameID;  ///< Identifier used to select internal resources when async support is enabled. Must increment by exactly one (1) for each frame. Any non-exactly-one difference will reset the frame generation logic.

} FfxFrameInterpolationDispatchDescription;

FFX_API FfxErrorCode ffxFrameInterpolationDispatch(FfxFrameInterpolationContext* context, const FfxFrameInterpolationDispatchDescription* params);

/// Dispatch the OF colour downsample pass (pre-OF step for >1080p inputs).
/// Must be called before optical flow dispatch when the resolution plan selects OF-colour downsampling.
/// The downsampled colour is stored internally; retrieve it with ffxFrameInterpolationGetDownsampledOfColourP1().
/// No-op when the context was created with displaySize.height <= 1080.
///
/// @param [in] context         A pointer to an initialised FfxFrameInterpolationContext.
/// @param [in] commandList     The command list to record the dispatch on.
/// @param [in] currentColour   The full-resolution colour resource to downsample.
///
/// @retval FFX_OK              The operation completed successfully.
/// @ingroup ffxFrameGeneration
FFX_API FfxErrorCode ffxFrameInterpolationDispatchDownsampleOfColour(FfxFrameInterpolationContext* context,
                                                                     FfxCommandList                commandList,
                                                                     FfxResource                   currentColour);

/// Get whether the context requires colour downsampling before OF (displaySize.height > 1080).
///
/// @param [in]  context        A pointer to an initialised FfxFrameInterpolationContext.
/// @param [out] pRequired      Receives true if downsample is required.
///
/// @retval FFX_OK              The operation completed successfully.
/// @ingroup ffxFrameGeneration
FFX_API FfxErrorCode ffxFrameInterpolationGetRequiresOfColourDownsample(FfxFrameInterpolationContext* context, bool* pRequired);

/// Get the downsampled P1 (current frame) internal colour resource.
/// Only valid after ffxFrameInterpolationDispatchDownsampleOfColour() has been called.
/// Returns a null resource when the resolution plan does not select OF-colour downsampling.
///
/// @param [in]  context        A pointer to an initialised FfxFrameInterpolationContext.
/// @param [out] pResource      Receives the FfxResource handle for the downsampled colour.
///
/// @retval FFX_OK              The operation completed successfully.
/// @ingroup ffxFrameGeneration
FFX_API FfxErrorCode ffxFrameInterpolationGetDownsampledOfColourP1(FfxFrameInterpolationContext* context, FfxResource* pResource);

/// Get the downsampled M1 (previous frame) internal colour resource.
/// After the ping-pong swap at the end of each dispatch, M1 holds the
/// previous frame's downsampled colour — suitable for use as the OF's
/// colour reference (colorTm1) at >1080p.
/// Returns a null resource when the resolution plan does not select OF-colour downsampling.
///
/// @param [in]  context        A pointer to an initialised FfxFrameInterpolationContext.
/// @param [out] pResource      Receives the FfxResource handle for the previous frame's downsampled colour.
///
/// @retval FFX_OK              The operation completed successfully.
/// @ingroup ffxFrameGeneration
FFX_API FfxErrorCode ffxFrameInterpolationGetDownsampledOfColourM1(FfxFrameInterpolationContext* context, FfxResource* pResource);

/// Destroy the FidelityFX Super Resolution context.
///
/// @param [out] context                A pointer to a <c><i>FfxFsr3Context</i></c> structure to destroy.
///
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_CODE_NULL_POINTER         The operation failed because either <c><i>context</i></c> was <c><i>NULL</i></c>.
///
/// @ingroup ffxFrameGeneration
FFX_API FfxErrorCode ffxFrameInterpolationContextDestroy(FfxFrameInterpolationContext* context);

/// Queries the effect version number.
///
/// @returns
/// The SDK version the effect was built with.
///
/// @ingroup ffxFrameGeneration
FFX_API FfxVersionNumber ffxFrameInterpolationGetEffectVersion();

#if defined(__cplusplus)
}
#endif  // #if defined(__cplusplus)
