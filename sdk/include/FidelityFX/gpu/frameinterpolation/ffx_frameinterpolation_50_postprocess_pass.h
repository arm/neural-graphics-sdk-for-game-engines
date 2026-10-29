/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef GPU_FFX_FRAMEINTERPOLATION_40_POSTPROCESS_PASS_H
#define GPU_FFX_FRAMEINTERPOLATION_40_POSTPROCESS_PASS_H

#define HALF_MIN 0.00006103515625HF

#include "ffx_frameinterpolation_callbacks_glsl.h"
#include "quant.h"

// methods

FFX_MIN16_F max4(FFX_MIN16_F4 x)
{
    return max(max(max(x.x, x.y), x.z), x.w);
}

FFX_MIN16_F2 LoadMotionQMVD(int32_t2 pixel, out bool dynamic_oob_hole)
{
    FfxUInt32 packed = LoadWarpedMotionQDataTP1(pixel);
    dynamic_oob_hole = (packed == 0x7FFFFFFFu);
    float2 motion    = dqFloatMv(packed);
    motion += mix(float2(0.0f), float2(0.001f), step(0.5f, float(packed)));
    motion *= float2(1.0f) / float2(MvDepthLaneSize());
    return FFX_MIN16_F2(motion) * FFX_MIN16_F(!dynamic_oob_hole);
}

FFX_MIN16_F4 Stable_Softmax(FFX_MIN16_F4 tensor)
{
    // Subtract the max value to help reduce overflow issues
    FFX_MIN16_F4 stable_ten     = tensor - max4(tensor);
    FFX_MIN16_F4 e_ten          = exp(stable_ten);
    FFX_MIN16_F  sum            = dot(e_ten, FFX_MIN16_F4(1.0HF));
    FFX_MIN16_F4 smax_activated = e_ten / (sum + HALF_MIN);
    return smax_activated;
}

float2 CroppedNetworkParamsUV(float2 uv)
{
    int32_t2 logical_dims   = max(int32_t2(1), MvDepthLaneSize() / 2);
    float2   logical_dims_f = float2(logical_dims);
    float2   params_pixel   = clamp(uv * logical_dims_f - 0.5f, float2(0.0f), logical_dims_f - 1.0f);
    return (params_pixel + 0.5f) * ParamsLaneSizeRcp();
}

void postprocess(int32_t2 output_pixel)
{
    //-------------------------------------------------------------------------
    // 1) Dispatch guard
    //-------------------------------------------------------------------------
    int32_t2 displaySize = DisplaySize();
    if (any(greaterThanEqual(output_pixel, displaySize)))
        return;

    //-------------------------------------------------------------------------
    // 2) Motion and OF sampling. Reprojection setup
    //-------------------------------------------------------------------------
    float2 uv = (float2(output_pixel.xy) + 0.5f) * DisplaySizeRcp();

    float2   mvDepthLanef = float2(MvDepthLaneSize());
    int32_t2 iMvCoord     = clamp(int32_t2(floor(uv * mvDepthLanef)), int32_t2(0), WarpedMotionQDataTP1Size() - int32_t2(1));

    // Read Output parameters - bilinear upsampling is done here. The graph params/gate are on the
    // params lane (padded preprocess lane / 2), so crop display UV to the logical half-resolution
    // content before sampling the physical padded image.
    float2       tensor_uv = CroppedNetworkParamsUV(uv);
    FFX_MIN16_F4 params    = Dequantize(FFX_MIN16_F4(SampleOutParamsTensor(tensor_uv)), QuantParamsSNORM().zw);

    // Read Vectors warped to "t" position - sampling nearest.
    FFX_MIN16_F2 flow_t_f30_p1    = SampleWarpedFilledFlowTM1(uv);
    bool         dynamic_oob_hole = false;  // Not used
    FFX_MIN16_F2 mv_t_f30_m1      = LoadMotionQMVD(iMvCoord, dynamic_oob_hole);

    // Calculate warped game mv positions
    float  timestep   = Timestep();
    float2 uv_m1_mv   = uv + float2(mv_t_f30_m1 * timestep);
    float2 uv_p1_mv   = uv - float2(mv_t_f30_m1 * (1.0f - timestep));
    float2 uv_m1_flow = uv - float2(flow_t_f30_p1 * timestep);
    float2 uv_p1_flow = uv + float2(flow_t_f30_p1 * (1.0f - timestep));

    bool uv_p1_mv_oob = (uv_p1_mv.x <= 0.0f || uv_p1_mv.x >= 1.0f) || (uv_p1_mv.y <= 0.0f || uv_p1_mv.y >= 1.0f);
    bool uv_m1_mv_oob = (uv_m1_mv.x <= 0.0f || uv_m1_mv.x >= 1.0f) || (uv_m1_mv.y <= 0.0f || uv_m1_mv.y >= 1.0f);

    //-------------------------------------------------------------------------
    // 3) Colour motion and OF warping
    //-------------------------------------------------------------------------
    FFX_MIN16_F3 rgb_m1_warp_t_mv   = SamplePreviousBackbuffer(uv_m1_mv);
    FFX_MIN16_F3 rgb_p1_warp_t_mv   = SampleCurrentBackbuffer(uv_p1_mv);
    FFX_MIN16_F3 rgb_m1_warp_t_flow = SamplePreviousBackbuffer(uv_m1_flow);
    FFX_MIN16_F3 rgb_p1_warp_t_flow = SampleCurrentBackbuffer(uv_p1_flow);

    //-------------------------------------------------------------------------
    // 4) Colour blending
    //-------------------------------------------------------------------------
    params = Stable_Softmax(params);

    FFX_MIN16_F3 rgb_t = FFX_MIN16_F3(0.HF);
    rgb_t += rgb_m1_warp_t_mv * params.x;
    rgb_t += rgb_p1_warp_t_mv * params.y;
    rgb_t += rgb_m1_warp_t_flow * params.z;
    rgb_t += rgb_p1_warp_t_flow * params.w;

    //-------------------------------------------------------------------------
    // 5) Inpaint holes
    //-------------------------------------------------------------------------
    // gated inpaint: only blend the resolved inpaint colour into actual holes (zero MV or both MV
    // warps OOB, excluding dynamic-OOB markers that the network handles itself)
    bool hole_to_inpaint =
        ((mv_t_f30_m1.x == FFX_MIN16_F(0.0f) && mv_t_f30_m1.y == FFX_MIN16_F(0.0f)) || (uv_m1_mv_oob && uv_p1_mv_oob)) && (!dynamic_oob_hole);
    if (hole_to_inpaint)
    {
        FFX_MIN16_F  gate    = ffxSaturate(Dequantize(FFX_MIN16_F(SampleOutParamsGateTensor(tensor_uv)), QuantParamsGateSNORM().zw));
        FFX_MIN16_F4 inpaint = FFX_MIN16_F4(SampleInpaintResult(uv));
        rgb_t                = ffxLerp(rgb_t, inpaint.rgb, inpaint.a * gate);
    }

    //-----------------------------------------------------------------------
    // 6) Output final colour
    //-------------------------------------------------------------------------
    StoreFrameinterpolationOutput(output_pixel, FFX_MIN16_F4(rgb_t, 1.0f));

    // When the SDK owns the previous-source ping-pong buffer, persist the current
    // backbuffer pixel so it can be sampled as colorTm1 on the next frame without
    // an extra copy job (see PREVIOUS_INTERPOLATION_SOURCE_BACKUP).
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_MANAGE_PREVIOUS_COLOR
    StorePreviousInterpolationSource(output_pixel, FFX_MIN16_F4(LoadCurrentBackbuffer(output_pixel), 1.0f));
#endif
}

#endif