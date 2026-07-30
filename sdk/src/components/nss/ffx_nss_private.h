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
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#pragma once
#include "FidelityFX/gpu/nss/ffx_nss_resources.h"

enum NssScalePresetMode
{
    NSS_SCALE_PRESET_NONE = 0,
    NSS_SCALE_PRESET_X2,
    NSS_SCALE_PRESET_COUNT,
};

/// Bit position of the 2-bit shader quality mode field within NssShaderPermutationOptions.
/// Extract the quality mode with: (flags >> NSS_SHADER_PERMUTATION_QUALITY_MODE_SHIFT) & NSS_SHADER_QUALITY_MODE_MASK
static constexpr uint32_t NSS_SHADER_QUALITY_MODE_MASK              = 3;
static constexpr uint32_t NSS_SHADER_PERMUTATION_QUALITY_MODE_SHIFT = 10;

/// An enumeration of all the permutations that can be passed to the NSS algorithm.
///
/// NSS features are organized through a set of pre-defined compile
/// permutation options that need to be specified. Which shader blob
/// is returned for pipeline creation will be determined by what combination
/// of shader permutations are enabled.
///
/// @ingroup ffxNss
typedef enum NssShaderPermutationOptions : uint32_t
{
    NSS_SHADER_PERMUTATION_QUANTIZED             = (1 << 0),
    NSS_SHADER_PERMUTATION_REVERSE_Z             = (1 << 1),
    NSS_SHADER_PERMUTATION_RESAMPLE_BICUBIC      = (1 << 2),
    NSS_SHADER_PERMUTATION_STATIC                = (1 << 3),
    NSS_SHADER_PERMUTATION_ALLOW_16BIT           = (1 << 4),
    NSS_SHADER_PERMUTATION_SCALE_PRESET_X2       = (1 << 5),
    NSS_SHADER_PERMUTATION_MANAGE_HISTORY        = (1 << 7),
    NSS_SHADER_PERMUTATION_PRE_PROCESS_FRAGMENT  = (1 << 8),
    NSS_SHADER_PERMUTATION_POST_PROCESS_FRAGMENT = (1 << 9),
    // Bits 10-11: 2-bit shader quality mode field
    NSS_SHADER_PERMUTATION_QUALITY_MODE_MASK       = (NSS_SHADER_QUALITY_MODE_MASK << NSS_SHADER_PERMUTATION_QUALITY_MODE_SHIFT),
    NSS_SHADER_PERMUTATION_COMPUTE_SUPPORT_TENSOR  = (1 << 12),
    NSS_SHADER_PERMUTATION_FRAGMENT_SUPPORT_TENSOR = (1 << 13),
} NssShaderPermutationOptions;

/// Debug view display modes for the NSS debug view pass.
/// The debug view is a 4x4 tile grid; mode 0 shows all tiles, modes 1-16 show a
/// single tile fullscreen. Only NSS_DEBUG_VIEW_MODE_ALL is consumed by the shader,
/// the named single-tile values document the mode -> tile mapping.
typedef enum NssDebugViewMode : uint32_t
{
    NSS_DEBUG_VIEW_MODE_ALL                      = 0,   ///< Show all 16 tiles in a 4x4 grid.
    NSS_DEBUG_VIEW_MODE_HISTORY                  = 1,   ///< row0,col0 — history color (previous upscaled output).
    NSS_DEBUG_VIEW_MODE_DEPTH                    = 2,   ///< row0,col1 — current-frame input depth.
    NSS_DEBUG_VIEW_MODE_RECONSTRUCTED_PREV_DEPTH = 3,   ///< row0,col2 — reconstructed previous-frame depth.
    NSS_DEBUG_VIEW_MODE_NEAREST_OFFSET           = 4,   ///< row0,col3 — nearest depth offset.
    NSS_DEBUG_VIEW_MODE_LOW_RES_COLOR            = 5,   ///< row1,col0 — low-res jittered input colour (texture).
    NSS_DEBUG_VIEW_MODE_MOTION_VECTOR            = 6,   ///< row1,col1 — input motion vectors (texture).
    NSS_DEBUG_VIEW_MODE_LUMA_DERIV_TM1           = 7,   ///< row1,col2 — previous-frame luma derivative.
    NSS_DEBUG_VIEW_MODE_TEMPORAL_FEEDBACK        = 8,   ///< row1,col3 — temporal feedback tensor texture.
    NSS_DEBUG_VIEW_MODE_LR_WARPED_HISTORY        = 9,   ///< row2,col0 — lr_warped_history (preprocess input tensor).
    NSS_DEBUG_VIEW_MODE_DISOCCLUSION_MASK        = 10,  ///< row2,col1 — disocclusion mask.
    NSS_DEBUG_VIEW_MODE_LUMA_DERIV_T             = 11,  ///< row2,col2 — current-frame luma derivative (SRV).
    NSS_DEBUG_VIEW_MODE_DEPTH_DILATED            = 12,  ///< row2,col3 — depth-dilated value used to compute the disocclusion mask.
    NSS_DEBUG_VIEW_MODE_TENSOR_COLOR             = 13,  ///< row3,col0 — colour feature (preprocess input tensor).
    NSS_DEBUG_VIEW_MODE_TENSOR_MOTION_DETECTOR   = 14,  ///< row3,col1 — motion-detector feature (input tensor).
    NSS_DEBUG_VIEW_MODE_TENSOR_LUMA_INSTABILITY  = 15,  ///< row3,col2 — luma-derivative/instability feature (input tensor).
    NSS_DEBUG_VIEW_MODE_TENSOR_WARP_FEEDBACK     = 16,  ///< row3,col3 — warped feedback RGB feature (input tensor).
} NssDebugViewMode;

/// Constants for NSS dispatches.
///
/// These constants are updated each frame and passed to the NSS preprocess
/// and postprocess compute shaders.
///
/// @ingroup ffxNss
typedef struct NssConstants
{
    FfxFloat32x4 _DeviceToViewDepth;
    FfxFloat32x4 _JitterOffset;     ///<  Current frame's jitter offset. .xy = pixels, .zw = uvs
    FfxFloat32x4 _JitterOffsetTm1;  ///<  Last frame's jitter offset. .xy = pixels, .zw = uvs
    FfxFloat32x4 _ScaleFactor;      ///<  Upscale factor. .xy = scale, .zw = inverse scale

    FfxUInt32x2 _OutputDims;  ///< Upscaled image dimensions (width, height)
    FfxUInt32x2 _InputDims;   ///< Rendered image dimensions (width, height)

    FfxFloat32x2 _InvOutputDims;  ///< Inverse upscaled image dimensions (width, height)
    FfxFloat32x2 _InvInputDims;   ///< Inverse rendered image dimensions (width, height)

    FfxUInt32x2  _DepthTm1Size;     ///< The depth tm1 size
    FfxFloat32x2 _InvDepthTm1Size;  ///< The inverse of depth tm1 size

    FfxUInt32x2  _InputTensorSize;     ///< The input tensor size (width, height)
    FfxFloat32x2 _InputTensorSizeRcp;  ///< The inv of input tensor size.

    FfxUInt32x2  _KpnDimension;       ///< The KPN coefficients tensor dimension
    FfxFloat32x2 _MotionVectorScale;  ///< .x = motion vector scale.x, .y = motion vector scale.y

    FfxFloat32x2 _PaddingScale;  ///< input / padded_preprocess.
    FfxFloat32   _DepthClipRequiredSepScale;
    FfxFloat32   _DepthClipPower;

    FfxFloat32x2 _KpnScale;         ///< kpnDims / paddedDims (= 0.25 when KPN is at 1/4 of dataGraphSize)
    FfxUInt32    _DebugViewMode;    ///< Debug view mode (see NssDebugViewMode). 0 = all tiles, 1-16 = single tile fullscreen.
    FfxFloat32   _NotHistoryReset;  ///< 1.0 if history is valid, 0.0 if history needs reset

    FfxFloat32x2 _Exposure;     ///< .x = exposure, .y = inverse exposure
    FfxUInt32x2  _IndexModulo;  ///< Reduced output/input numerator per axis; also dynamic LUT tile modulo.

    FfxUInt32x2 _ReducedInputModulo;  ///< Reduced output/input denominator per axis, used by dynamic LUT generation.
    FfxUInt32x2 _LutOffset;           ///< Jittered offset inside the modulo tile. .x = offset.x, .y = offset.y
} NssConstants;

struct FfxPipelineState;

/// FfxNssContext_Private
/// The private implementation of the NSS context.
///
/// @ingroup ffxNss
typedef struct FfxNssContext_Private
{
    FfxNssContextDescription contextDescription;  ///< The description used to create this context
    FfxUInt32                effectContextId;
    NssConstants             constants;  ///< The constants used for the current dispatch. Setup and stored in host side.
    FfxDevice                device;
    FfxPipelineState         pipelineNssDepthScatter;                          ///< The pipeline state for the NSS depth scatter pass.
    FfxPipelineState         pipelineNssDisocclusionMask;                      ///< The pipeline state for the NSS disocclusion mask pass.
    FfxPipelineState         pipelineNssPreprocess;                            ///< The pipeline state for the NSS preprocess pass.
    FfxPipelineState         pipelineNssDataGraph;                             ///< The pipeline state for the NSS data graph pass.
    FfxPipelineState         pipelineNssGenerateOffsetLut;                     ///< The pipeline state for the NSS dynamic offset LUT generation pass.
    FfxPipelineState         pipelineNssPostprocess;                           ///< The pipeline state for the NSS postprocess pass.
    FfxPipelineState         pipelineNssDebugView;                             ///< The pipeline state for the NSS debug view pass.
    FfxConstantBuffer        constantBuffers[FFX_NSS_CONSTANTBUFFER_COUNT];    ///< Pointer to constant data in staging ring buffer and data size.
    FfxResourceInternal      srvResources[FFX_NSS_RESOURCE_IDENTIFIER_COUNT];  ///< SRV resource table.
    FfxResourceInternal      uavResources[FFX_NSS_RESOURCE_IDENTIFIER_COUNT];  ///< UAV resource table.
    FfxConstantBuffer        rootConstant;

    bool            firstExecution;
    uint32_t        resourceFrameIndex;
    uint32_t        pipelineFlags;
    FfxDimensions2D depthScatterSize;
    FfxDimensions2D processSize;
    FfxDimensions2D dataGraphSize;
    FfxDimensions2D kpnDimension;
    FfxDimensions2D offsetLutSize;
    FfxDimensions2D reducedFractionLrSize;
    FfxDimensions2D reducedFractionHrSize;
    bool            useDynamicOffsetLut;
    bool            useDisocclusionMaskPass;
} FfxNssContext_Private;
