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

#include "ffx_frameinterpolation_resources.h"

#if defined(FFX_GPU)
#include "ffx_core.h"
#include "ffx_frameinterpolation_common.h"

#define COUNTER_SPD                          0
#define COUNTER_FRAME_INDEX_SINCE_LAST_RESET 1

#define tensor_t    int8_t
#define tensorVec_t int8_t4

#ifndef NEW_DYNAMIC_MASK
#define NEW_DYNAMIC_MASK 1
#endif

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
    FfxInt32x2 _OfSize;

    FfxFloat32x2 _RenderSizeRcp;
    FfxFloat32x2 _DisplaySizeRcp;
    FfxFloat32x2 _OfSizeRcp;

    FfxFloat32x2 _Jitter;
    FfxFloat32x2 _MotionVectorScale;
    FfxFloat32x2 _MvSimilarityThresholds;

    FfxInt32x2   _InputTensorSize;
    FfxFloat32x2 _InputTensorSizeRcp;
    FfxInt32     _Reset;
    FfxFloat32   _Timestep;
    FfxUInt32    _RandomSeed;
    FfxUInt32    _pad0;

    FfxInt32x2   _MvDepthLaneSize;
    FfxFloat32x2 _MvDepthLaneSizeRcp;
    FfxInt32x2   _FlowLaneSize;
    FfxFloat32x2 _FlowLaneSizeRcp;
    FfxFloat32   _OfGridSizeRcp;
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

FFX_MIN16_F MvSimilarityThreshold()
{
    return FFX_MIN16_F(cbFI._MvSimilarityThresholds.x);
}

FFX_MIN16_F MvSimilarityNoiseThreshold()
{
    return FFX_MIN16_F(cbFI._MvSimilarityThresholds.y);
}

FfxInt32x2 RenderSize()
{
    return cbFI._RenderSize;
}

FfxFloat32x2 RenderSizeRcp()
{
    return cbFI._RenderSizeRcp;
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

FfxFloat32x2 InputTensorSizeRcp()
{
    return cbFI._InputTensorSizeRcp;
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

FfxFloat32x4 DeviceToViewSpaceTransformFactors()
{
    return cbFI._DeviceToViewDepth;
}

FfxInt32x2 GetOpticalFlowResolution()
{
    return cbFI._OfSize;
}

FfxFloat32x2 GetOpticalFlowInvSize()
{
    return cbFI._OfSizeRcp;
}

FfxFloat32 OfGridSizeRcp()
{
    return cbFI._OfGridSizeRcp;
}

FFX_MIN16_F4 QuantParamsSINT()
{
    return FFX_MIN16_F4(cbFI._QuantParamsSINT);
}

FFX_MIN16_F4 QuantParamsSNORM()
{
    return FFX_MIN16_F4(cbFI._QuantParamsSNORM);
}

FfxFloat32 ConvertFromDeviceDepthToViewSpace(FfxFloat32 fDeviceDepth)
{
    const FfxFloat32x4 deviceToViewDepth = DeviceToViewSpaceTransformFactors();
    return deviceToViewDepth[1] / (fDeviceDepth - deviceToViewDepth[0]);
}

// TODO_ARM do we need to swizzel xy like the reference one?
FfxFloat32x2 ComputeNdc(FfxFloat32x2 fPxPos, FfxInt32x2 iSize)
{
    return fPxPos / FfxFloat32x2(iSize) * FfxFloat32x2(2.0f, -2.0f) + FfxFloat32x2(-1.0f, 1.0f);
}

FfxFloat32x3 GetViewSpacePosition(FfxInt32x2 iViewportPos, FfxInt32x2 iViewportSize, FfxFloat32 fDeviceDepth)
{
    const FfxFloat32x4 fDeviceToViewDepth = DeviceToViewSpaceTransformFactors();

    const FfxFloat32 Z = ConvertFromDeviceDepthToViewSpace(fDeviceDepth);

    const FfxFloat32x2 fNdcPos = ComputeNdc(iViewportPos, iViewportSize);
    const FfxFloat32   X       = fDeviceToViewDepth[2] * fNdcPos.x * Z;
    const FfxFloat32   Y       = fDeviceToViewDepth[3] * fNdcPos.y * Z;

    return FfxFloat32x3(X, Y, Z);
}
#endif  // defined(FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION)

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_CB_INPAINTING_PYRAMID)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_CB_INPAINTING_PYRAMID, std140) uniform cbInpaintingPyramid_t
{
    FfxUInt32   mips;
    FfxUInt32   numWorkGroups;
    FfxUInt32x2 workGroupOffset;
}
cbInpaintingPyramid;

FfxUInt32 NumMips()
{
    return cbInpaintingPyramid.mips;
}
FfxUInt32 NumWorkGroups()
{
    return cbInpaintingPyramid.numWorkGroups;
}
FfxUInt32x2 WorkGroupOffset()
{
    return cbInpaintingPyramid.workGroupOffset;
}

#endif  // defined(FFX_ARM_FRAMEINTERPOLATION_BIND_CB_INPAINTING_PYRAMID)

///////////////////////////////////////////////
// declare samplers
///////////////////////////////////////////////

layout(set = 0, binding = 1000) uniform sampler s_PointClamp;
layout(set = 0, binding = 1001) uniform sampler s_LinearClamp;
layout(set = 0, binding = 1002) uniform sampler s_PointBorder;
layout(set = 0, binding = 1003) uniform sampler s_LinearBorder;

///////////////////////////////////////////////
// declare SRVs and SRV accessors
///////////////////////////////////////////////

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_OUTPUT
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_OUTPUT) uniform texture2D r_output;

FfxFloat32x4 LoadFrameInterpolationOutput(FFX_PARAMETER_IN FfxInt32x2 iPxInput)
{
    return texelFetch(r_output, iPxInput, 0);
}
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PRESENT_BACKBUFFER
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PRESENT_BACKBUFFER) uniform texture2D r_present_backbuffer;

FfxFloat32x4 LoadPresentBackbuffer(FFX_PARAMETER_IN FfxInt32x2 iPxInput)
{
    return texelFetch(r_present_backbuffer, iPxInput, 0);
}
FfxFloat32x4 SamplePresentBackbuffer(FFX_PARAMETER_IN FfxFloat32x2 fUv)
{
    return textureLod(sampler2D(r_present_backbuffer, s_LinearClamp), fUv, 0.0);
}
#endif

// Arm NFRU SRVs

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_MOTION_TP1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_MOTION_TP1) uniform mediump texture2D r_motion_tp1;
FFX_MIN16_F2 LoadInputMotionTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    FFX_MIN16_F2 motion = FFX_MIN16_F2(texelFetch(r_motion_tp1, iPxPos, 0).xy) * MotionVectorScale();
#if FFX_FRAMEINTERPOLATION_OPTION_JITTERED_MOTION_VECTORS
#error ARM_TODO: Implement Jitter cancellation (Missing uniforms?)!
    motion -= MotionVectorJitterCancellation();
#endif
    // Return Motion in UV space
    return motion;
}

FFX_MIN16_F2 LoadInputMotionTp1(FFX_PARAMETER_IN FfxFloat32x2 fUvPos)
{
    FfxInt32x2 iPxPos = FfxInt32x2(floor(fUvPos * FfxFloat32x2(textureSize(r_motion_tp1, 0))));
    return LoadInputMotionTp1(iPxPos);
}

FFX_MIN16_F2 LoadInputMotionTP1WithOffset(FFX_PARAMETER_IN FfxFloat32x2 fUvPos, FfxInt32x2 iOffset)
{
    FfxInt32x2 iSize  = FfxInt32x2(textureSize(r_motion_tp1, 0));
    FfxInt32x2 iPxPos = FfxInt32x2(floor(fUvPos * FfxFloat32x2(iSize)));
    // Keep neighborhood fetch in-bounds after applying iOffset.
    FfxInt32x2   iSamplePx = clamp(iPxPos + iOffset, FfxInt32x2(0), iSize - FfxInt32x2(1));
    FFX_MIN16_F2 motion    = FFX_MIN16_F2(texelFetch(r_motion_tp1, iSamplePx, 0).xy) * MotionVectorScale();
#if FFX_FRAMEINTERPOLATION_OPTION_JITTERED_MOTION_VECTORS
#error ARM_TODO: Implement Jitter cancellation (Missing uniforms?)!
    motion -= MotionVectorJitterCancellation();
#endif
    // Return Motion in UV space
    return motion;
}

#endif  // FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_MOTION_TP1

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TP1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TP1) uniform texture2D r_depth_tp1;

FfxFloat32 LoadInputDepthTp1(FFX_PARAMETER_IN FfxFloat32x2 fUvPos)
{
    FfxInt32x2 iPxPos = FfxInt32x2(floor(fUvPos * FfxFloat32x2(textureSize(r_depth_tp1, 0))));
    FfxFloat32 depth  = texelFetch(r_depth_tp1, iPxPos, 0).x;
    return depth;
}

FfxFloat32 LoadInputDepthTp1WithOffset(FFX_PARAMETER_IN FfxFloat32x2 fUvPos, FfxInt32x2 iOffset)
{
    FfxInt32x2 iSize     = FfxInt32x2(textureSize(r_depth_tp1, 0));
    FfxInt32x2 iPxPos    = FfxInt32x2(floor(fUvPos * FfxFloat32x2(iSize)));
    FfxInt32x2 iSamplePx = clamp(iPxPos + iOffset, FfxInt32x2(0), iSize - FfxInt32x2(1));
    FfxFloat32 depth     = texelFetch(r_depth_tp1, iSamplePx, 0).x;
    return depth;
}

FfxFloat32 LoadInputDepthTp1Pixel(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    FfxFloat32 depth = texelFetch(r_depth_tp1, iPxPos, 0).x;
    return depth;
}

FfxFloat32x2 DepthTp1Size()
{
    return FfxFloat32x2(textureSize(r_depth_tp1, 0));
}

#endif  // FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TP1

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_FLOW_TM1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_FLOW_TM1) uniform mediump texture2D r_flow_tm1;
FFX_MIN16_F2 LoadInputFlowTm1(FFX_PARAMETER_IN FfxFloat32x2 fUvPos)
{
    FfxInt32x2 iPxPos = FfxInt32x2(floor(fUvPos * FfxFloat32x2(textureSize(r_flow_tm1, 0))));
    return FFX_MIN16_F2(texelFetch(r_flow_tm1, iPxPos, 0).xy) * FFX_MIN16_F2(GetOpticalFlowInvSize()) * FFX_MIN16_F(OfGridSizeRcp());
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TM1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TM1) uniform texture2D r_depth_tm1;
FfxFloat32 LoadInputDepthTm1(FFX_PARAMETER_IN FfxFloat32x2 fUvPos)
{
    FfxInt32x2 iPxPos = FfxInt32x2(floor(fUvPos * FfxFloat32x2(textureSize(r_depth_tm1, 0))));
    FfxFloat32 depth  = texelFetch(r_depth_tm1, iPxPos, 0).x;
    return depth;
}

FfxFloat32 LoadInputDepthTm1Pixel(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    FfxFloat32 depth = texelFetch(r_depth_tm1, iPxPos, 0).x;
    return depth;
}

FfxFloat32x2 DepthTm1Size()
{
    return FfxFloat32x2(textureSize(r_depth_tm1, 0));
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_HOLES_TM1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_HOLES_TM1) uniform texture2D r_mv_holes_tm1;
FfxFloat32 LoadInputHolesTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return texelFetch(r_mv_holes_tm1, iPxPos, 0).x;
}

FfxFloat32 LoadInputHolesTm1(FFX_PARAMETER_IN FfxFloat32x2 fUvPos)
{
    FfxInt32x2 iPxPos = FfxInt32x2(floor(fUvPos * FfxFloat32x2(textureSize(r_mv_holes_tm1, 0))));
    return LoadInputHolesTm1(iPxPos);
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_HOLES_TP1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_HOLES_TP1) uniform texture2D r_mv_holes_tp1;
FfxFloat32 LoadInputHolesTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return texelFetch(r_mv_holes_tp1, iPxPos, 0).x;
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DYNAMIC_MASK_TM1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DYNAMIC_MASK_TM1) uniform texture2D r_dynamic_mask_tm1;
FfxFloat32 LoadDynamicMaskTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return texelFetch(r_dynamic_mask_tm1, iPxPos, 0).x;
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DYNAMIC_MASK_TP1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DYNAMIC_MASK_TP1) uniform texture2D r_dynamic_mask_tp1;
FfxFloat32 LoadDynamicMaskTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return texelFetch(r_dynamic_mask_tp1, iPxPos, 0).x;
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_MOTION_QDATA_TP1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_MOTION_QDATA_TP1) uniform utexture2D r_warped_motion_qdata_tp1;
FfxUInt32 LoadWarpedMotionQDataTP1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return texelFetch(r_warped_motion_qdata_tp1, iPxPos, 0).x;
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FLOW_QDATA_TM1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FLOW_QDATA_TM1) uniform utexture2D r_warped_flow_qdata_tm1;
FfxUInt32 LoadWarpedFlowQDataTM1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return texelFetch(r_warped_flow_qdata_tm1, iPxPos, 0).x;
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_MOTION_QDATA_TM1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_MOTION_QDATA_TM1) uniform utexture2D r_warped_motion_qdata_tm1;
FfxUInt32 LoadWarpedMotionQDataTM1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return texelFetch(r_warped_motion_qdata_tm1, iPxPos, 0).x;
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_MOTION_TP1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_MOTION_TP1) uniform mediump texture2D r_warped_filled_motion_tp1;
FFX_MIN16_F2 LoadWarpedFilledMotionTP1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return FFX_MIN16_F2(texelFetch(r_warped_filled_motion_tp1, iPxPos, 0).xy);
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_FLOW_TM1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_FLOW_TM1) uniform mediump texture2D r_warped_filled_flow_tm1;
FFX_MIN16_F2 LoadWarpedFilledFlowTM1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return FFX_MIN16_F2(texelFetch(r_warped_filled_flow_tm1, iPxPos, 0).xy);
}
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREVIOUS_INTERPOLATION_SOURCE
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREVIOUS_INTERPOLATION_SOURCE) uniform mediump texture2D r_previous_interpolation_source;

FFX_MIN16_F3 LoadPreviousBackbuffer(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    return FFX_MIN16_F3(texelFetch(r_previous_interpolation_source, iPxPos, 0).rgb);
}
FFX_MIN16_F3 SamplePreviousBackbufferClampBorder(FFX_PARAMETER_IN FfxFloat32x2 fUv)
{
    return FFX_MIN16_F3(textureLod(sampler2D(r_previous_interpolation_source, s_LinearBorder), fUv, 0.0).xyz);
}
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
FFX_MIN16_F3 SampleCurrentBackbufferClampBorder(FFX_PARAMETER_IN FfxFloat32x2 fUv)
{
    return FFX_MIN16_F3(textureLod(sampler2D(r_current_interpolation_source, s_LinearBorder), fUv, 0.0).xyz);
}
FFX_MIN16_F3 SampleCurrentBackbuffer(FFX_PARAMETER_IN FfxFloat32x2 fUv)
{
    return FFX_MIN16_F3(textureLod(sampler2D(r_current_interpolation_source, s_LinearClamp), fUv, 0.0).xyz);
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

// Arm NFRU UAVs

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
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_DYNAMIC_MASK_TP1, r8) uniform image2D rw_dynamic_mask_tp1;
void StoreDynamicMaskTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxUInt8 mask)
{
    float mask_f = clamp(float(mask), 0.0f, 1.0f);
    imageStore(rw_dynamic_mask_tp1, iPxPos, FfxFloat32x4(mask_f, 0.0f, 0.0f, 1.0f));
}
#elif defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DYNAMIC_MASK_TP1)
layout(location = FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DYNAMIC_MASK_TP1) out lowp FfxFloat32x4 rw_dynamic_mask_tp1;
void StoreDynamicMaskTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxUInt8 mask)
{
    float mask_f        = clamp(float(mask), 0.0f, 1.0f);
    rw_dynamic_mask_tp1 = FfxFloat32x4(mask_f, 0.0f, 0.0f, 1.0f);
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
void UpdateWarpFlowQDataTm1(FFX_PARAMETER_IN FFX_MIN16_I2 iPxSample, FFX_PARAMETER_IN FfxUInt32 qdata)
{
    imageAtomicMax(rw_warped_flow_qdata_tm1, iPxSample, qdata);
}
void InitWarpClearQDataTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    imageStore(rw_warped_flow_qdata_tm1, iPxPos, uvec4(0));
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_HOLES_TP1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_HOLES_TP1, r8) uniform image2D rw_mv_holes_tp1;
void StoreHolesTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F2 nearestMotion)
{
    imageStore(rw_mv_holes_tp1,
               iPxPos + FfxInt32x2(floor(nearestMotion * FFX_MIN16_F2(MvDepthLaneSize()) * FFX_MIN16_F(1.0f - Timestep()))),
               FFX_MIN16_F4(1, 0, 0, 0));
}
void InitWarpClearHolesTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    imageStore(rw_mv_holes_tp1, iPxPos, vec4(0.0));
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_HOLES_TM1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_HOLES_TM1, r8) uniform image2D rw_mv_holes_tm1;
void StoreHolesTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F2 nearestMotion)
{
    imageStore(rw_mv_holes_tm1, iPxPos + FfxInt32x2(floor(nearestMotion * FFX_MIN16_F2(MvDepthLaneSize()))), FFX_MIN16_F4(1, 0, 0, 0));
}
void InitWarpClearHolesTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    imageStore(rw_mv_holes_tm1, iPxPos, vec4(0.0));
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_COLOUR_P1_INTERNAL)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_COLOUR_P1_INTERNAL, rgba8) uniform mediump image2D rw_colour_p1_internal;
void StoreColourP1Internal(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F3 val)
{
    imageStore(rw_colour_p1_internal, iPxPos, FFX_MIN16_F4(val, 1.0f));
}
#elif defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_COLOUR_P1_INTERNAL)
layout(location = FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_COLOUR_P1_INTERNAL) out mediump FfxFloat32x4 rw_colour_p1_internal;
void StoreColourP1Internal(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F3 val)
{
    rw_colour_p1_internal = FfxFloat32x4(val, 1.0f);
}
#endif

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_FILLED_MOTION_TP1)
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_FILLED_MOTION_TP1, rg16f) uniform image2D rw_warped_filled_motion_tp1;
void StoreWarpFilledMotionTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F2 val)
{
    imageStore(rw_warped_filled_motion_tp1, iPxPos, FFX_MIN16_F4(val, 0.0f, 0.0f));
}
#elif defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_WARP_FILLED_MOTION_TP1)
layout(location = FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_WARP_FILLED_MOTION_TP1) out mediump FfxFloat32x4 rw_warped_filled_motion_tp1;
void StoreWarpFilledMotionTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F2 val)
{
    rw_warped_filled_motion_tp1 = FfxFloat32x4(val, 0.0f, 0.0f);
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