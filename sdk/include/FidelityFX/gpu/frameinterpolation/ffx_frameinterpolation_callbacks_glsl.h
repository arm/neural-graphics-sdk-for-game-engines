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

#include "ffx_frameinterpolation_resources.h"

#if defined(FFX_GPU)
#include "ffx_core.h"
#include "ffx_frameinterpolation_common.h"

#define COUNTER_SPD                          0
#define COUNTER_FRAME_INDEX_SINCE_LAST_RESET 1

#define tensor_t    int8_t
#define tensorVec_t int8_t4

struct InputTensorElement_t
{
    int8_t4 rgbM1MV_rP1MV;  // rgb_m1_warp_t_mv.rgb, rgb_p1_warp_t_mv.r
    int8_t4 gbP1MV_rgM1OF;  // rgb_p1_warp_t_mv.gb, rgb_m1_warp_t_flow.rg
    int8_t4 bM1OF_rgbP1OF;  // rgb_m1_warp_t_flow.b, rgb_p1_warp_t_flow.rgb
    int8_t4 depth_disMask;  // depth_m1_warp_t_mv_norm, depth_p1_warp_t_mv_norm, dis_mask_m1, dis_mask_p1
};

///////////////////////////////////////////////
// declare CBs and CB accessors
///////////////////////////////////////////////
#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION, std140) uniform cbFI_t
{
    FfxFloat32x4x4 _MotionTM1ToTP1;
    FfxFloat32x4x4 _MotionTP1ToTM1;

    FfxFloat32x4 _QuantParamsSINT;   ///< Quant parameters between float and sint. .xy for quantize, .zw for dequantize
    FfxFloat32x4 _QuantParamsSNORM;  ///< Quant parameters between float and snorm. .xy for quantize, .zw for dequantize

    FfxFloat32x4 _DeviceToViewDepth;

    FfxInt32x2 _RenderSize;
    FfxInt32x2 _DisplaySize;

    FfxFloat32x2 _DisplaySizeRcp;

    FfxFloat32x2 _Jitter;
    FfxFloat32x2 _MotionVectorScale;
    FfxFloat32x2 _JitterCancellation;
    FfxFloat32x3 _MvSimilarityThresholds;

    FfxInt32x2 _InputTensorSize;
    FfxInt32   _Reset;
    FfxFloat32 _Timestep;
    FfxUInt32  _RandomSeed;
    FfxUInt32  _UseReducedMotionInput;

    FfxInt32x2   _MvDepthLaneSize;
    FfxFloat32x2 _MvDepthLaneSizeRcp;
    FfxInt32x2   _FlowLaneSize;
    FfxFloat32x2 _FlowLaneSizeRcp;
    FfxFloat32x2 _ParamsLaneSizeRcp;
    FfxFloat32x4 _QuantParamsGateSNORM;
    FfxFloat32x2 _OfInputSizeRcp;
    FfxFloat32   _DepthClipScale;
    FfxFloat32   _DepthClipPower;
}
cbFI;

FfxFloat32x4x4 GetMotionTM1ToTP1()
{
    return cbFI._MotionTM1ToTP1;
}

FfxFloat32x4x4 GetMotionTP1ToTM1()
{
    return cbFI._MotionTP1ToTM1;
}

FfxFloat32x2 Jitter()
{
    return cbFI._Jitter;
}

FFX_MIN16_F2 MotionVectorScale()
{
    return FFX_MIN16_F2(cbFI._MotionVectorScale);
}

FfxFloat32x2 CancelMotionVectorJitter(FfxFloat32x2 scaledMotion)
{
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS
    return scaledMotion - cbFI._JitterCancellation;
#else
    return scaledMotion;
#endif
}

FFX_MIN16_F MvSimilarityThreshold()
{
    return FFX_MIN16_F(cbFI._MvSimilarityThresholds.x);
}

FFX_MIN16_F MvSimilarityNoiseThreshold()
{
    return FFX_MIN16_F(cbFI._MvSimilarityThresholds.y);
}

FFX_MIN16_F MvSimilarityRelativeThreshold()
{
    return FFX_MIN16_F(cbFI._MvSimilarityThresholds.z);
}

FfxInt32x2 RenderSize()
{
    return cbFI._RenderSize;
}

FfxInt32x2 DisplaySize()
{
    return cbFI._DisplaySize;
}

FfxFloat32x2 DisplaySizeRcp()
{
    return cbFI._DisplaySizeRcp;
}

FfxInt32x2 InputTensorSize()
{
    return cbFI._InputTensorSize;
}

FfxBoolean Reset()
{
    return cbFI._Reset == 1;
}

FfxFloat32 Timestep()
{
    return cbFI._Timestep;
}

FfxUInt32 RandomSeed()
{
    return cbFI._RandomSeed;
}

FfxBoolean UseReducedMotionInput()
{
    return cbFI._UseReducedMotionInput != 0;
}

FfxInt32x2 MvDepthLaneSize()
{
    return cbFI._MvDepthLaneSize;
}

FfxFloat32x2 MvDepthLaneSizeRcp()
{
    return cbFI._MvDepthLaneSizeRcp;
}

FfxInt32x2 FlowLaneSize()
{
    return cbFI._FlowLaneSize;
}

FfxFloat32x2 FlowLaneSizeRcp()
{
    return cbFI._FlowLaneSizeRcp;
}

FfxFloat32x2 ParamsLaneSizeRcp()
{
    return cbFI._ParamsLaneSizeRcp;
}

FfxFloat32x4 DeviceToViewSpaceTransformFactors()
{
    return cbFI._DeviceToViewDepth;
}

FfxFloat32x2 OfInputSizeRcp()
{
    return cbFI._OfInputSizeRcp;
}

FfxFloat32 DepthClipScale()
{
    return cbFI._DepthClipScale;
}

FfxFloat32 DepthClipPower()
{
    return cbFI._DepthClipPower;
}

FFX_MIN16_F4 QuantParamsSINT()
{
    return FFX_MIN16_F4(cbFI._QuantParamsSINT);
}

FFX_MIN16_F4 QuantParamsSNORM()
{
    return FFX_MIN16_F4(cbFI._QuantParamsSNORM);
}

FFX_MIN16_F4 QuantParamsGateSNORM()
{
    return FFX_MIN16_F4(cbFI._QuantParamsGateSNORM);
}

FfxFloat32 ConvertFromDeviceDepthToViewSpace(FfxFloat32 fDeviceDepth)
{
    const FfxFloat32x4 deviceToViewDepth = DeviceToViewSpaceTransformFactors();
    return deviceToViewDepth[1] / (fDeviceDepth - deviceToViewDepth[0]);
}

#endif  // defined(FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION)

///////////////////////////////////////////////
// declare samplers
///////////////////////////////////////////////

layout(set = 0, binding = 1000) uniform sampler s_PointClamp;
layout(set = 0, binding = 1001) uniform sampler s_LinearClamp;
layout(set = 0, binding = 1002) uniform sampler s_PointBorder;
layout(set = 0, binding = 1003) uniform sampler s_LinearBorder;

FfxInt32x2 ClampTexelCoord(FfxInt32x2 iPxPos, texture2D texture)
{
    return clamp(iPxPos, FfxInt32x2(0), FfxInt32x2(textureSize(texture, 0)) - FfxInt32x2(1));
}

FfxInt32x2 ClampTexelCoord(FfxInt32x2 iPxPos, utexture2D texture)
{
    return clamp(iPxPos, FfxInt32x2(0), FfxInt32x2(textureSize(texture, 0)) - FfxInt32x2(1));
}

///////////////////////////////////////////////
// declare SRVs and SRV accessors
///////////////////////////////////////////////

// Arm NFRU SRVs

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_MOTION_TP1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_MOTION_TP1) uniform mediump texture2D r_motion_tp1;
#define _MotionTp1Texture sampler2D(r_motion_tp1, s_PointClamp)
FFX_MIN16_F2 LoadInputMotionTp1Raw(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return FFX_MIN16_F2(texelFetch(r_motion_tp1, ClampTexelCoord(iPxPos, r_motion_tp1), 0).xy);
}
FFX_MIN16_F2 LoadInputMotionTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS
    FfxFloat32x2 motion = texelFetch(r_motion_tp1, ClampTexelCoord(iPxPos, r_motion_tp1), 0).xy * FfxFloat32x2(MotionVectorScale());
    return FFX_MIN16_F2(CancelMotionVectorJitter(motion));
#else
    return LoadInputMotionTp1Raw(iPxPos) * MotionVectorScale();
#endif
}

FFX_MIN16_F2 LoadInputMotionTp1(FFX_PARAMETER_IN FfxFloat32x2 fUvPos)
{
    FfxInt32x2 iPxPos = ClampTexelCoord(FfxInt32x2(floor(fUvPos * FfxFloat32x2(textureSize(r_motion_tp1, 0)))), r_motion_tp1);
    return LoadInputMotionTp1(iPxPos);
}

#endif  // FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_MOTION_TP1

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TP1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TP1) uniform highp texture2D r_depth_tp1;
#define _DepthTp1Texture sampler2D(r_depth_tp1, s_PointBorder)

FfxFloat32 LoadInputDepthTp1Pixel(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    FfxFloat32 depth = texelFetch(r_depth_tp1, ClampTexelCoord(iPxPos, r_depth_tp1), 0).x;
    return depth;
}

#endif  // FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TP1

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_DEPTH_TP1_FEEDBACK)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_DEPTH_TP1_FEEDBACK) uniform highp texture2D r_depth_tm1_next;

FfxFloat32 LoadDepthTp1FeedbackPixel(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return texelFetch(r_depth_tm1_next, iPxPos, 0).x;
}

FfxFloat32x2 DepthTp1FeedbackSize()
{
    return FfxFloat32x2(textureSize(r_depth_tm1_next, 0));
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_FLOW_TM1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_FLOW_TM1) uniform highp texture2D r_flow_tm1;
FfxFloat32x2 LoadInputFlowTm1(FFX_PARAMETER_IN FfxFloat32x2 fUvPos)
{
    FfxInt32x2 iPxPos = ClampTexelCoord(FfxInt32x2(floor(fUvPos * FfxFloat32x2(textureSize(r_flow_tm1, 0)))), r_flow_tm1);
    return FfxFloat32x2(texelFetch(r_flow_tm1, iPxPos, 0).xy) * OfInputSizeRcp();
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TM1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TM1) uniform highp texture2D r_depth_tm1;
#define _DepthTm1Texture sampler2D(r_depth_tm1, s_PointBorder)
FfxFloat32 LoadInputDepthTm1(FFX_PARAMETER_IN FfxFloat32x2 fUvPos)
{
    FfxInt32x2 iPxPos = ClampTexelCoord(FfxInt32x2(floor(fUvPos * FfxFloat32x2(textureSize(r_depth_tm1, 0)))), r_depth_tm1);
    FfxFloat32 depth  = texelFetch(r_depth_tm1, iPxPos, 0).x;
    return depth;
}

FfxFloat32 LoadInputDepthTm1Pixel(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    FfxFloat32 depth = texelFetch(r_depth_tm1, ClampTexelCoord(iPxPos, r_depth_tm1), 0).x;
    return depth;
}

FfxFloat32x2 DepthTm1Size()
{
    return FfxFloat32x2(textureSize(r_depth_tm1, 0));
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DYNAMIC_MASK_TM1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DYNAMIC_MASK_TM1) uniform lowp texture2D r_dynamic_mask_tm1;
#define _DynamicMaskTm1 sampler2D(r_dynamic_mask_tm1, s_PointBorder)

FfxFloat32 LoadDynamicMaskTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return texelFetch(r_dynamic_mask_tm1, ClampTexelCoord(iPxPos, r_dynamic_mask_tm1), 0).x;
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DYNAMIC_MASK_TP1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DYNAMIC_MASK_TP1) uniform texture2D r_dynamic_mask_tp1;
FfxFloat32 LoadDynamicMaskTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return texelFetch(r_dynamic_mask_tp1, ClampTexelCoord(iPxPos, r_dynamic_mask_tp1), 0).x;
}

FfxInt32x2 DynamicMaskTp1Size()
{
    return FfxInt32x2(textureSize(r_dynamic_mask_tp1, 0));
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_MOTION_QDATA_TP1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_MOTION_QDATA_TP1) uniform utexture2D r_warped_motion_qdata_tp1;
FfxUInt32 LoadWarpedMotionQDataTP1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return texelFetch(r_warped_motion_qdata_tp1, iPxPos, 0).x;
}

FfxInt32x2 WarpedMotionQDataTP1Size()
{
    return FfxInt32x2(textureSize(r_warped_motion_qdata_tp1, 0));
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FLOW_QDATA_TM1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FLOW_QDATA_TM1) uniform utexture2D r_warped_flow_qdata_tm1;
FfxUInt32 LoadWarpedFlowQDataTM1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return texelFetch(r_warped_flow_qdata_tm1, iPxPos, 0).x;
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_FLOW_TM1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_FLOW_TM1) uniform mediump texture2D r_warped_filled_flow_tm1;
// bilinear flow sampling
FFX_MIN16_F2 SampleWarpedFilledFlowTM1(FFX_PARAMETER_IN FfxFloat32x2 fUv)
{
    return FFX_MIN16_F2(textureLod(sampler2D(r_warped_filled_flow_tm1, s_LinearClamp), fUv, 0.0).xy);
}
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREVIOUS_INTERPOLATION_SOURCE
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREVIOUS_INTERPOLATION_SOURCE) uniform mediump texture2D r_previous_interpolation_source;

FFX_MIN16_F3 SamplePreviousBackbuffer(FFX_PARAMETER_IN FfxFloat32x2 fUv)
{
    return FFX_MIN16_F3(textureLod(sampler2D(r_previous_interpolation_source, s_LinearClamp), fUv, 0.0).xyz);
}
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_CURRENT_INTERPOLATION_SOURCE
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_CURRENT_INTERPOLATION_SOURCE) uniform mediump texture2D r_current_interpolation_source;

FFX_MIN16_F3 LoadCurrentBackbuffer(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return FFX_MIN16_F3(texelFetch(r_current_interpolation_source, iPxPos, 0).rgb);
}
FFX_MIN16_F3 SampleCurrentBackbuffer(FFX_PARAMETER_IN FfxFloat32x2 fUv)
{
    return FFX_MIN16_F3(textureLod(sampler2D(r_current_interpolation_source, s_LinearClamp), fUv, 0.0).xyz);
}
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_OF_COLOUR_P1
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_OF_COLOUR_P1) uniform mediump texture2D r_of_colour_p1;
FFX_MIN16_F3 SampleOfColourP1(FFX_PARAMETER_IN FfxFloat32x2 fUv)
{
    return FFX_MIN16_F3(textureLod(sampler2D(r_of_colour_p1, s_LinearClamp), fUv, 0.0).rgb);
}
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREPROCESS_COLOUR_P1
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREPROCESS_COLOUR_P1) uniform mediump texture2D r_preprocess_colour_p1;
FfxInt32x2 PreprocessColourP1Size()
{
    return textureSize(r_preprocess_colour_p1, 0);
}
FFX_MIN16_F3 LoadPreprocessColourP1Pixel(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return FFX_MIN16_F3(texelFetch(r_preprocess_colour_p1, ClampTexelCoord(iPxPos, r_preprocess_colour_p1), 0).rgb);
}
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREPROCESS_COLOUR_M1
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREPROCESS_COLOUR_M1) uniform mediump texture2D r_preprocess_colour_m1;
FfxInt32x2 PreprocessColourM1Size()
{
    return textureSize(r_preprocess_colour_m1, 0);
}
FFX_MIN16_F3 LoadPreprocessColourM1Pixel(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return FFX_MIN16_F3(texelFetch(r_preprocess_colour_m1, ClampTexelCoord(iPxPos, r_preprocess_colour_m1), 0).rgb);
}
#endif

#if FFX_ARM_FRAMEINTERPOLATION_OPTION_ENABLE_DATA_GRAPH_FI && FFX_ARM_FRAMEINTERPOLATION_OPTION_SUPPORT_TENSOR
#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_IN_TENSOR)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_IN_TENSOR) uniform readonly tensorARM<tensor_t, 4> r_in_tensor;
tensorVec_t LoadInTensor(FFX_PARAMETER_IN FfxUInt32x2 coord, FFX_PARAMETER_IN FfxUInt32 channelOffset)
{
    tensor_t result[4];
    tensorReadARM(r_in_tensor, FfxUInt32[](0, coord.y, coord.x, channelOffset), result);
    return tensorVec_t(result[0], result[1], result[2], result[3]);
}

tensorVec_t LoadInTensorData0(FFX_PARAMETER_IN FfxUInt32x2 coord)
{
    tensor_t result[4];
    tensorReadARM(r_in_tensor, FfxUInt32[](0, coord.y, coord.x, 0), result);
    return tensorVec_t(result[0], result[1], result[2], result[3]);
}

tensorVec_t LoadInTensorData1(FFX_PARAMETER_IN FfxUInt32x2 coord)
{
    tensor_t result[4];
    tensorReadARM(r_in_tensor, FfxUInt32[](0, coord.y, coord.x, 4), result);
    return tensorVec_t(result[0], result[1], result[2], result[3]);
}

tensorVec_t LoadInTensorData2(FFX_PARAMETER_IN FfxUInt32x2 coord)
{
    tensor_t result[4];
    tensorReadARM(r_in_tensor, FfxUInt32[](0, coord.y, coord.x, 8), result);
    return tensorVec_t(result[0], result[1], result[2], result[3]);
}

tensorVec_t LoadInTensorData3(FFX_PARAMETER_IN FfxUInt32x2 coord)
{
    tensor_t result[4];
    tensorReadARM(r_in_tensor, FfxUInt32[](0, coord.y, coord.x, 12), result);
    return tensorVec_t(result[0], result[1], result[2], result[3]);
}
#endif
#else
#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_IN_TENSOR)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_IN_TENSOR, std430) readonly buffer InputTensorBuffer_t
{
    InputTensorElement_t elementData[];
}
r_in_tensor;

int8_t4 LoadInTensor(FFX_PARAMETER_IN FfxUInt32x2 coord, FFX_PARAMETER_IN FfxUInt32 channelOffset)
{
    FfxUInt32 idx = coord.y * InputTensorSize().x + coord.x;
    if (channelOffset == 0)
        return r_in_tensor.elementData[idx].rgbM1MV_rP1MV;
    else if (channelOffset == 4)
        return r_in_tensor.elementData[idx].gbP1MV_rgM1OF;
    else if (channelOffset == 8)
        return r_in_tensor.elementData[idx].bM1OF_rgbP1OF;
    else  // channelOffset == 12
        return r_in_tensor.elementData[idx].depth_disMask;
}

int8_t4 LoadInTensorData0(FFX_PARAMETER_IN FfxUInt32x2 coord)
{
    FfxUInt32 idx = coord.y * InputTensorSize().x + coord.x;
    return r_in_tensor.elementData[idx].rgbM1MV_rP1MV;
}

int8_t4 LoadInTensorData1(FFX_PARAMETER_IN FfxUInt32x2 coord)
{
    FfxUInt32 idx = coord.y * InputTensorSize().x + coord.x;
    return r_in_tensor.elementData[idx].gbP1MV_rgM1OF;
}

int8_t4 LoadInTensorData2(FFX_PARAMETER_IN FfxUInt32x2 coord)
{
    FfxUInt32 idx = coord.y * InputTensorSize().x + coord.x;
    return r_in_tensor.elementData[idx].bM1OF_rgbP1OF;
}

int8_t4 LoadInTensorData3(FFX_PARAMETER_IN FfxUInt32x2 coord)
{
    FfxUInt32 idx = coord.y * InputTensorSize().x + coord.x;
    return r_in_tensor.elementData[idx].depth_disMask;
}
#endif
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_OUT_PARAMS_TENSOR
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_OUT_PARAMS_TENSOR) uniform lowp texture2D r_out_params_tensor;

FfxFloat32x4 SampleOutParamsTensor(FFX_PARAMETER_IN FfxFloat32x2 UV)
{
    FfxFloat32x4 weights = textureLod(sampler2D(r_out_params_tensor, s_LinearClamp), UV, 0.0);
    return weights;
}
#endif

// inpaint SRVs
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_OUT_PARAMS_GATE_TENSOR
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_OUT_PARAMS_GATE_TENSOR) uniform lowp texture2D r_out_params_gate_tensor;
FfxFloat32 SampleOutParamsGateTensor(FFX_PARAMETER_IN FfxFloat32x2 UV)
{
    return textureLod(sampler2D(r_out_params_gate_tensor, s_LinearClamp), UV, 0.0).x;
}
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_SEED
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_SEED) uniform highp texture2D r_inpaint_seed;
FfxFloat32x4 SampleInpaintSeed(FFX_PARAMETER_IN FfxFloat32x2 uv)
{
    return textureLod(sampler2D(r_inpaint_seed, s_LinearClamp), uv, 0.0);
}
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_SOURCE
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_SOURCE) uniform highp texture2D r_inpaint_source;
FfxFloat32x4 SampleInpaintSource(FFX_PARAMETER_IN FfxFloat32x2 uv)
{
    return textureLod(sampler2D(r_inpaint_source, s_LinearClamp), uv, 0.0);
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_LEVEL_0) && defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_LEVEL_1) && \
    defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_LEVEL_2) && defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_LEVEL_3) && \
    defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_LEVEL_4)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_LEVEL_0) uniform highp texture2D r_inpaint_level_0;
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_LEVEL_1) uniform highp texture2D r_inpaint_level_1;
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_LEVEL_2) uniform highp texture2D r_inpaint_level_2;
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_LEVEL_3) uniform highp texture2D r_inpaint_level_3;
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_LEVEL_4) uniform highp texture2D r_inpaint_level_4;
FfxFloat32x4 SampleInpaintLevel(FFX_PARAMETER_IN FfxFloat32x2 uv, FFX_PARAMETER_IN FfxInt32 level)
{
    if (level == 0)
        return textureLod(sampler2D(r_inpaint_level_0, s_LinearClamp), uv, 0.0);
    if (level == 1)
        return textureLod(sampler2D(r_inpaint_level_1, s_LinearClamp), uv, 0.0);
    if (level == 2)
        return textureLod(sampler2D(r_inpaint_level_2, s_LinearClamp), uv, 0.0);
    if (level == 3)
        return textureLod(sampler2D(r_inpaint_level_3, s_LinearClamp), uv, 0.0);
    return textureLod(sampler2D(r_inpaint_level_4, s_LinearClamp), uv, 0.0);
}
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_RESULT
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_RESULT) uniform highp texture2D r_inpaint_result;
FfxFloat32x4 SampleInpaintResult(FFX_PARAMETER_IN FfxFloat32x2 uv)
{
    return textureLod(sampler2D(r_inpaint_result, s_LinearClamp), uv, 0.0);
}
#endif

// inpaint UAVs / render targets
#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_INPAINT_SEED)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_INPAINT_SEED, rgba8) uniform image2D rw_inpaint_seed;
void StoreInpaintSeed(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x4 value)
{
    imageStore(rw_inpaint_seed, iPxPos, value);
}
#elif defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_INPAINT_SEED)
layout(location = FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_INPAINT_SEED) out highp FfxFloat32x4 rw_inpaint_seed;
void StoreInpaintSeed(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x4 value)
{
    rw_inpaint_seed = value;
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_INPAINT_DESTINATION)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_INPAINT_DESTINATION, rgba8) uniform image2D rw_inpaint_destination;
void StoreInpaintDestination(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x4 value)
{
    imageStore(rw_inpaint_destination, iPxPos, value);
}
FfxInt32x2 InpaintDestinationSize()
{
    return imageSize(rw_inpaint_destination);
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_INPAINT_RESULT)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_INPAINT_RESULT, rgba8) writeonly uniform image2D rw_inpaint_result;
void StoreInpaintResult(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x4 value)
{
    imageStore(rw_inpaint_result, iPxPos, value);
}
#endif

// Arm NFRU UAVs

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_PREPROCESS_COLOUR_P1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_PREPROCESS_COLOUR_P1, OUTPUT_IMG_FORMAT) uniform mediump image2D rw_preprocess_colour_p1;
void StorePreprocessColourP1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F3 val)
{
    imageStore(rw_preprocess_colour_p1, iPxPos, FFX_MIN16_F4(val, 1.0f));
}
#elif defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_PREPROCESS_COLOUR_P1)
layout(location = FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_PREPROCESS_COLOUR_P1) out mediump FfxFloat32x4 rw_preprocess_colour_p1;
void StorePreprocessColourP1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F3 val)
{
    rw_preprocess_colour_p1 = FfxFloat32x4(val, 1.0f);
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_DEPTH_TM1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_DEPTH_TM1, r32f) uniform image2D rw_depth_tm1;
void StoreDepthTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32 depth)
{
    imageStore(rw_depth_tm1, iPxPos, FfxFloat32x4(depth, 0.0f, 0.0f, 0.0f));
}
#elif defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DEPTH_TM1)
layout(location = FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DEPTH_TM1) out highp FfxFloat32x4 rw_depth_tm1;
void StoreDepthTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32 depth)
{
    rw_depth_tm1 = FfxFloat32x4(depth, 0.0f, 0.0f, 0.0f);
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_DEPTH_TM1_NEXT)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_DEPTH_TM1_NEXT, r32f) uniform image2D rw_depth_tm1_next;
void StoreDepthTm1Next(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32 depth)
{
    imageStore(rw_depth_tm1_next, iPxPos, FfxFloat32x4(depth, 0.0f, 0.0f, 0.0f));
}
#elif defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DEPTH_TM1_NEXT)
layout(location = FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DEPTH_TM1_NEXT) out highp FfxFloat32x4 rw_depth_tm1_next;
void StoreDepthTm1Next(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32 depth)
{
    rw_depth_tm1_next = FfxFloat32x4(depth, 0.0f, 0.0f, 0.0f);
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_DYNAMIC_MASK_TP1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_DYNAMIC_MASK_TP1, r8) uniform highp image2D rw_dynamic_mask_tp1;
void StoreDynamicMaskTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32 mask)
{
    imageStore(rw_dynamic_mask_tp1, iPxPos, FfxFloat32x4(mask, 0.0f, 0.0f, 1.0f));
}
#elif defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DYNAMIC_MASK_TP1)
layout(location = FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DYNAMIC_MASK_TP1) out highp FfxFloat32x4 rw_dynamic_mask_tp1;
void StoreDynamicMaskTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32 mask)
{
    rw_dynamic_mask_tp1 = FfxFloat32x4(mask, 0.0f, 0.0f, 1.0f);
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_MOTION_QDATA_TP1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_MOTION_QDATA_TP1, r32ui) uniform uimage2D rw_warped_motion_qdata_tp1;
void UpdateWarpMotionQDataTp1(FFX_PARAMETER_IN FFX_MIN16_I2 iPxSample, FFX_PARAMETER_IN FfxUInt32 qdata)
{
    imageAtomicMax(rw_warped_motion_qdata_tp1, iPxSample, qdata);
}
void InitWarpClearQDataTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    imageStore(rw_warped_motion_qdata_tp1, iPxPos, uvec4(0));
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_FLOW_QDATA_TM1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_FLOW_QDATA_TM1, r32ui) uniform uimage2D rw_warped_flow_qdata_tm1;
void UpdateWarpFlowQDataTm1(FFX_PARAMETER_IN FfxInt32x2 iPxSample, FFX_PARAMETER_IN FfxUInt32 qdata)
{
    imageAtomicMax(rw_warped_flow_qdata_tm1, iPxSample, qdata);
}
void InitWarpClearQDataTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    imageStore(rw_warped_flow_qdata_tm1, iPxPos, uvec4(0));
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_OF_COLOUR_P1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_OF_COLOUR_P1, rgba8) uniform mediump image2D rw_of_colour_p1;
void StoreOfColourP1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F3 val)
{
    imageStore(rw_of_colour_p1, iPxPos, FFX_MIN16_F4(val, 1.0f));
}
#elif defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_OF_COLOUR_P1)
layout(location = FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_OF_COLOUR_P1) out mediump FfxFloat32x4 rw_of_colour_p1;
void StoreOfColourP1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F3 val)
{
    rw_of_colour_p1 = FfxFloat32x4(val, 1.0f);
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_REDUCED_MOTION_P1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_REDUCED_MOTION_P1, rg16f) uniform mediump image2D rw_reduced_motion_p1;
void StoreReducedMotionP1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F2 val)
{
    imageStore(rw_reduced_motion_p1, iPxPos, FFX_MIN16_F4(val, 0.0f, 0.0f));
}
#elif defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_REDUCED_MOTION_P1)
layout(location = FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_REDUCED_MOTION_P1) out mediump FfxFloat32x4 rw_reduced_motion_p1;
void StoreReducedMotionP1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F2 val)
{
    rw_reduced_motion_p1 = FfxFloat32x4(val, 0.0f, 0.0f);
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_FILLED_FLOW_TM1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_FILLED_FLOW_TM1, rg16f) uniform image2D rw_warped_filled_flow_tm1;
void StoreWarpFilledFlowTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F2 val)
{
    imageStore(rw_warped_filled_flow_tm1, iPxPos, FFX_MIN16_F4(val, 0.0f, 0.0f));
}
#elif defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_WARP_FILLED_FLOW_TM1)
layout(location = FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_WARP_FILLED_FLOW_TM1) out mediump FfxFloat32x4 rw_warped_filled_flow_tm1;
void StoreWarpFilledFlowTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F2 val)
{
    rw_warped_filled_flow_tm1 = FfxFloat32x4(val, 0.0f, 0.0f);
}
#endif

#if FFX_ARM_FRAMEINTERPOLATION_OPTION_ENABLE_DATA_GRAPH_FI && FFX_ARM_FRAMEINTERPOLATION_OPTION_SUPPORT_TENSOR
#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_IN_TENSOR)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_IN_TENSOR) uniform writeonly tensorARM<tensor_t, 4> rw_in_tensor;
void StoreInTensor(FFX_PARAMETER_IN FfxUInt32x2 coord, FFX_PARAMETER_IN InputTensorElement_t te)
{
    const uint tensorOperandsNonTemporalARM = 0x1U;

    int8_t t0[16] = {
        te.rgbM1MV_rP1MV.x,
        te.rgbM1MV_rP1MV.y,
        te.rgbM1MV_rP1MV.z,
        te.rgbM1MV_rP1MV.w,
        te.gbP1MV_rgM1OF.x,
        te.gbP1MV_rgM1OF.y,
        te.gbP1MV_rgM1OF.z,
        te.gbP1MV_rgM1OF.w,
        te.bM1OF_rgbP1OF.x,
        te.bM1OF_rgbP1OF.y,
        te.bM1OF_rgbP1OF.z,
        te.bM1OF_rgbP1OF.w,
        te.depth_disMask.x,
        te.depth_disMask.y,
        te.depth_disMask.z,
        te.depth_disMask.w,
    };

    tensorWriteARM(rw_in_tensor, FfxUInt32[](0, coord.y, coord.x, 0), t0, tensorOperandsNonTemporalARM);
}
#endif
#else
#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_IN_TENSOR)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_IN_TENSOR, std430) writeonly buffer InputTensorBuffer_t
{
    InputTensorElement_t data[];
}
rw_in_tensor;

void StoreInTensor(FFX_PARAMETER_IN FfxUInt32x2 coord, FFX_PARAMETER_IN InputTensorElement_t te)
{
    FfxUInt32 idx          = coord.y * InputTensorSize().x + coord.x;
    rw_in_tensor.data[idx] = te;
}
#endif
#endif  //FFX_ARM_FRAMEINTERPOLATION_OPTION_ENABLE_DATA_GRAPH_FI

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_OUTPUT
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_OUTPUT, OUTPUT_IMG_FORMAT) uniform image2D rw_output;
void StoreFrameinterpolationOutput(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F4 val)
{
    imageStore(rw_output, iPxPos, val);
}
#elif defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_OUTPUT)
layout(location = FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_OUTPUT) out mediump FfxFloat32x4 rw_output;
void StoreFrameinterpolationOutput(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F4 val)
{
    rw_output = FfxFloat32x4(val);
}
#endif

// StorePreviousInterpolationSource: writes the current backbuffer pixel into the
// ping-pong backup buffer that becomes colorTm1 on the next frame. Compiled in
// either a compute (UAV) or fragment (render-target) variant depending on which
// binding macro the host shader defined.
#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_PREVIOUS_INTERPOLATION_SOURCE)
layout(set     = 0,
       binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_PREVIOUS_INTERPOLATION_SOURCE,
       OUTPUT_IMG_FORMAT) uniform image2D rw_previous_interpolation_source;
void StorePreviousInterpolationSource(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F4 val)
{
    imageStore(rw_previous_interpolation_source, iPxPos, val);
}
#elif defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_PREVIOUS_INTERPOLATION_SOURCE)
layout(location = FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_PREVIOUS_INTERPOLATION_SOURCE) out mediump FfxFloat32x4 rw_previous_interpolation_source;
void StorePreviousInterpolationSource(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F4 val)
{
    // iPxPos is implicit in the fragment-shader variant (gl_FragCoord drives the write).
    rw_previous_interpolation_source = FfxFloat32x4(val);
}
#endif

#if !FFX_ARM_FRAMEINTERPOLATION_OPTION_ENABLE_DATA_GRAPH_FI
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_OUT_PARAMS_TENSOR
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_OUT_PARAMS_TENSOR, rgba16f) uniform image2D rw_out_params_tensor;
void StoreOutParamsTensor(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F4 val)
{
    imageStore(rw_out_params_tensor, iPxPos, val);
}
#endif
#endif  //!FFX_ARM_FRAMEINTERPOLATION_OPTION_ENABLE_DATA_GRAPH_FI

#endif  // #if defined(FFX_GPU)