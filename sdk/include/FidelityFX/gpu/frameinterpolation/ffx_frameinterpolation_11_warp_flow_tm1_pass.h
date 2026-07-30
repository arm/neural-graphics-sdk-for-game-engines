// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#ifndef GPU_FFX_FRAMEINTERPOLATION_01_WARP_FLOW_TM1_PASS_H
#define GPU_FFX_FRAMEINTERPOLATION_01_WARP_FLOW_TM1_PASS_H

#include "ffx_frameinterpolation_callbacks_glsl.h"
// Must include this file after layouts are defined, due to containing hard-coded variable
#include "warp_flow_common.h"

void warp_flow_tm1(int32_t2 output_pixel)
{
    int32_t2 ofRes     = FlowLaneSize();
    float2   ofInvSize = FlowLaneSizeRcp();
    float    ofInvArea = ofInvSize.x * ofInvSize.y;
    if (any(greaterThanEqual(output_pixel, ofRes)))
        return;

    int32_t output_pixel_cont = output_pixel.x + output_pixel.y * ofRes.x;

    float2 uv = (float2(output_pixel.xy) + 0.5f) * float2(1.0 / ofRes.x, 1.0 / ofRes.y);

    // Load depth, normalise
    FFX_MIN16_F depth_m1 = FFX_MIN16_F(LoadInputDepthTm1(uv));

    // Need 1-near depth for warping
    depth_m1 = FFX_MIN16_F(InvertAndNormalizeDepth(depth_m1, output_pixel_cont, ofInvArea));

    // Load optical flow
    half2 flow_m1_f30_p1 = LoadInputFlowTm1(uv);

    // Perform depth-aware forward warp
    int32_t2 outCoord;
    uint32_t outData;
    Warp(output_pixel, flow_m1_f30_p1, depth_m1, ofRes, FFX_MIN16_F(Timestep()), FFX_MIN16_F(1.0), outCoord, outData);
    UpdateWarpFlowQDataTm1(FFX_MIN16_I2(outCoord), outData);
}

#endif