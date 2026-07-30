// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#ifndef GPU_FFX_FRAMEINTERPOLATION_40_POSTPROCESS_PASS_H
#define GPU_FFX_FRAMEINTERPOLATION_40_POSTPROCESS_PASS_H

#define HALF_MIN 0.00006103515625HF

#include "ffx_frameinterpolation_callbacks_glsl.h"

// methods

FFX_MIN16_F max4(FFX_MIN16_F4 x)
{
    return max(max(max(x.x, x.y), x.z), x.w);
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

void postprocess(int32_t2 output_pixel)
{
    int32_t2 displaySize = DisplaySize();
    if (any(greaterThanEqual(output_pixel, displaySize)))
        return;

    float2 uv = (float2(output_pixel.xy) + 0.5f) * DisplaySizeRcp();

    float2   mvDepthLanef = float2(MvDepthLaneSize());
    float2   flowLanef    = float2(FlowLaneSize());
    float2   ofSizef      = float2(GetOpticalFlowResolution());
    int32_t2 iMvCoord     = int32_t2(floor(uv * mvDepthLanef));
    int32_t2 iFlowCoord   = int32_t2(floor(uv * flowLanef));

    // Read Output parameters - bilinear upsampling is done here
    float2       tensor_uv = uv * ofSizef * InputTensorSizeRcp();
    FFX_MIN16_F4 params    = Dequantize(FFX_MIN16_F4(SampleOutParamsTensor(tensor_uv)), QuantParamsSNORM().zw);

    // Read Vectors warped to "t" position - sampling nearest
    FFX_MIN16_F2 flow_t_f30_p1 = LoadWarpedFilledFlowTM1(iFlowCoord);
    FFX_MIN16_F2 mv_t_f30_m1   = LoadWarpedFilledMotionTP1(iMvCoord);

    // Calculate warped game mv positions
    float  timestep = Timestep();
    float2 uv_m1_mv = uv + float2(mv_t_f30_m1 * timestep);
    float2 uv_p1_mv = uv - float2(mv_t_f30_m1 * (1.0f - timestep));

    // Calculate warped optical flow positions
    float2 uv_m1_flow = uv - float2(flow_t_f30_p1 * timestep);
    float2 uv_p1_flow = uv + float2(flow_t_f30_p1 * (1.0f - timestep));

    // Warp colour using gather approach with warped vectors
    FFX_MIN16_F3 rgb_m1_warp_t_mv   = SamplePreviousBackbuffer(uv_m1_mv);
    FFX_MIN16_F3 rgb_p1_warp_t_mv   = SampleCurrentBackbuffer(uv_p1_mv);
    FFX_MIN16_F3 rgb_m1_warp_t_flow = SamplePreviousBackbuffer(uv_m1_flow);
    FFX_MIN16_F3 rgb_p1_warp_t_flow = SampleCurrentBackbuffer(uv_p1_flow);

    // Softmax network output to get blending coeffs
    params = Stable_Softmax(params);

    // Composite Output
    FFX_MIN16_F3 rgb_t = FFX_MIN16_F3(0.HF);
    rgb_t += rgb_m1_warp_t_mv * params.x;
    rgb_t += rgb_p1_warp_t_mv * params.y;
    rgb_t += rgb_m1_warp_t_flow * params.z;
    rgb_t += rgb_p1_warp_t_flow * params.w;

    // Write alpha composited output
    StoreFrameinterpolationOutput(output_pixel, FFX_MIN16_F4(rgb_t, 1.0f));

    // When the SDK owns the previous-source ping-pong buffer, persist the current
    // backbuffer pixel so it can be sampled as colorTm1 on the next frame without
    // an extra copy job (see PREVIOUS_INTERPOLATION_SOURCE_BACKUP).
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_MANAGE_PREVIOUS_COLOR
    StorePreviousInterpolationSource(output_pixel, FFX_MIN16_F4(LoadCurrentBackbuffer(output_pixel), 1.0f));
#endif
}

#endif