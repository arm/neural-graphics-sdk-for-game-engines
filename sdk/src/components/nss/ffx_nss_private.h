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
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#pragma once
#include "FidelityFX/gpu/nss/ffx_nss_resources.h"

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
    NSS_SHADER_PERMUTATION_QUANTIZED                      = (1 << 0),
    NSS_SHADER_PERMUTATION_REVERSE_Z                      = (1 << 1),
    NSS_SHADER_PERMUTATION_RESAMPLE_BICUBIC               = (1 << 2),
    NSS_SHADER_PERMUTATION_STATIC                         = (1 << 3),
    NSS_SHADER_PERMUTATION_ALIAS_OUTPUT_TENSORS_AS_IMAGES = (1 << 4),
    NSS_SHADER_PERMUTATION_ALLOW_16BIT                    = (1 << 5),
    NSS_SHADER_PERMUTATION_SCALE_PRESET_MODE_X1_3         = (1 << 6),
    NSS_SHADER_PERMUTATION_SCALE_PRESET_MODE_X1_5         = (1 << 7),
    NSS_SHADER_PERMUTATION_SCALE_PRESET_MODE_X2           = (1 << 8),
} NssShaderPermutationOptions;

/// 32bits constants for NSS dispatches.
///
/// These constants are used when the shader permutation option
/// "NSS_SHADER_PERMUTATION_ALLOW_16BIT" is not enabled.
///
/// @ingroup ffxNss
typedef struct NssConstants32bitParameters
{
    FfxFloat32x4 _QuantParamsSNORM;    ///< Quant parameters between float and snorm. .xy for quantize, .zw for dequantize
    FfxFloat32x4 _QuantParamsSINT;     ///< Quant parameters between float and sint. .xy for quantize, .zw for dequantize
    FfxFloat32x4 _MotionDisThreshPad;  ///< .x = motion vector thresholds, .y = disocclusion threshold, .z = disocclusion scale
    FfxFloat32x2 _Exposure;            ///< .x = exposure, .y = inverse exposure
    FfxUInt32x2  _IndexModulo;         ///< Euqal to {2, 2}. Hardcode to use 2x2 tile size.
    FfxUInt32x2  _LutOffset;           ///< Jittered offset in 2x2 tile. .x = offset.x, .y = offset.y
    FfxFloat32   _NotHistoryReset;     ///< 1.0 if history is valid, 0.0 if history needs reset
} NssConstants32bitParameters;

/// 16bits constants for NSS dispatches.
///
/// These constants are used when the shader permutation option
/// "NSS_SHADER_PERMUTATION_ALLOW_16BIT" is enabled.
///
/// The definition of member variables is the same as
/// "NssConstants32bitParameters".
///
/// @ingroup ffxNss
typedef struct NssConstants16bitParameters
{
    FfxUInt32x2 _QuantParamsSNORM;
    FfxUInt32x2 _QuantParamsSINT;
    FfxUInt32x2 _MotionDisThreshPad;
    FfxUInt32   _Exposure;
    FfxUInt32   _IndexModulo;
    FfxUInt32   _LutOffset;
    FfxUInt32   _NotHistoryReset;
} NssConstants16bitParameters;

/// Constants for NSS dispatches.
///
/// These constants are updated each frame and passed to the NSS preprocess
/// and postprocess compute shaders.
///
/// Part of the constants are 32bit floats, others are either 16bit packed floats
/// or 32bit floats depending on shader permutation options
/// "NSS_SHADER_PERMUTATION_ALLOW_16BIT"
///
/// @ingroup ffxNss
typedef struct NssConstants
{
    FfxFloat32x4 _DeviceToViewDepth;
    FfxFloat32x4 _JitterOffset;     ///<  Current frame's jitter offset. .xy = pixels, .zw = uvs
    FfxFloat32x4 _JitterOffsetTm1;  ///<  Last frame's jitter offset. .xy = pixels, .zw = uvs
    FfxFloat32x4 _ScaleFactor;      ///<  Upscale factor. .xy = scale, .zw = inverse scale

    FfxUInt32x2  _OutputDims;         ///< Upscaled image dimensions (width, height)
    FfxUInt32x2  _InputDims;          ///< Rendered image dimensions (width, height)
    FfxFloat32x2 _InvOutputDims;      ///< Inverse upscaled image dimensions (width, height)
    FfxFloat32x2 _InvInputDims;       ///< Inverse rendered image dimensions (width, height)
    FfxFloat32x2 _MotionVectorScale;  ///< .x = motion vector scale.x, .y = motion vector scale.y
    FfxUInt32x2  _UnpaddedInputDims;  ///< Unpadded rendered image dimensions (width, height)

    union
    {
        NssConstants16bitParameters _16bit;
        NssConstants32bitParameters _32bit;
    } dynamicPrecision;  ///< Union of 16bit and 32bit precision constant parameters
} NssConstants;

struct FfxDeviceCapabilities;
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
    FfxDeviceCapabilities    deviceCapabilities;
    FfxPipelineState         pipelineNssMirrorPadding;                         ///< The pipeline state for the NSS mirror padding pass.
    FfxPipelineState         pipelineNssPreprocess;                            ///< The pipeline state for the NSS preprocess pass.
    FfxPipelineState         pipelineNssDataGraph;                             ///< The pipeline state for the NSS data graph pass.
    FfxPipelineState         pipelineNssPostprocess;                           ///< The pipeline state for the NSS postprocess pass.
    FfxPipelineState         pipelineNssDebugView;                             ///< The pipeline state for the NSS debug view pass.
    FfxConstantBuffer        constantBuffers[FFX_NSS_CONSTANTBUFFER_COUNT];    ///< Pointer to constant data in staging ring buffer and data size.
    FfxResourceInternal      srvResources[FFX_NSS_RESOURCE_IDENTIFIER_COUNT];  ///< SRV resource table.
    FfxResourceInternal      uavResources[FFX_NSS_RESOURCE_IDENTIFIER_COUNT];  ///< UAV resource table.
    FfxConstantBuffer        rootConstant;

    bool     firstExecution;
    uint32_t resourceFrameIndex;
    bool     hasPaddingPass;
    uint32_t paddedInputWidth;
    uint32_t paddedInputHeight;
    uint32_t paddedOutputWidth;
    uint32_t paddedOutputHeight;
} FfxNssContext_Private;
