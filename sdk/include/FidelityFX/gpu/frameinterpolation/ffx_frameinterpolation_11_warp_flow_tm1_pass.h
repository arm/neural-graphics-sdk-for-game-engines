/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef GPU_FFX_FRAMEINTERPOLATION_01_WARP_FLOW_TM1_PASS_H
#define GPU_FFX_FRAMEINTERPOLATION_01_WARP_FLOW_TM1_PASS_H

#include "ffx_frameinterpolation_callbacks_glsl.h"
#include "quant.h"

FfxBoolean Warp(FfxInt32x2                   dtId,
                FfxFloat32x2                 motion,
                FfxFloat32                   depth,
                FfxInt32x2                   output_dims,
                FfxFloat32                   t,
                FfxFloat32                   s,
                FFX_PARAMETER_OUT FfxInt32x2 outCoord,
                FFX_PARAMETER_OUT FfxUInt32  outData)
{
    FfxFloat32x2 vector = motion * FfxFloat32x2(output_dims);
    FfxInt32x2   coord  = dtId + FfxInt32x2(floor(vector * t));

    if (!IsOnScreen(coord, output_dims))
        return false;

    outCoord = coord;
    outData  = qFloat(s * vector, depth);
    return true;
}

void warp_flow_tm1(int32_t2 output_pixel)
{
    //-------------------------------------------------------------------------
    // 1) Dispatch guard
    //-------------------------------------------------------------------------
    FfxInt32x2   ofRes     = FlowLaneSize();
    FfxFloat32x2 ofInvSize = FlowLaneSizeRcp();
    FfxFloat32   ofInvArea = ofInvSize.x * ofInvSize.y;
    if (any(greaterThanEqual(output_pixel, ofRes)))
        return;

    FfxInt32 output_pixel_cont = output_pixel.x + output_pixel.y * ofRes.x;

    FfxFloat32x2 uv = (FfxFloat32x2(output_pixel) + 0.5f) * ofInvSize;

    //-------------------------------------------------------------------------
    // 2) Load and normalize t-1 depth before scattering
    //-------------------------------------------------------------------------
    FfxFloat32 depth_m1 = LoadInputDepthTm1(uv);
    // Get 1-near depth for warping
    depth_m1 = InvertAndNormalizeDepth(depth_m1, output_pixel_cont, ofInvArea);

    //-------------------------------------------------------------------------
    // 3) Load optical flow and convert from pixels to UV.
    //-------------------------------------------------------------------------
    FfxFloat32x2 flow_m1_f30_p1 = LoadInputFlowTm1(uv);

    //-------------------------------------------------------------------------
    // 4) Depth-aware scatter to intermediate t frame
    //-------------------------------------------------------------------------
    FfxInt32x2 outCoord;
    FfxUInt32  outData;
    if (Warp(output_pixel, flow_m1_f30_p1, depth_m1, ofRes, Timestep(), 1.0f, outCoord, outData))
    {
        UpdateWarpFlowQDataTm1(outCoord, outData);
    }
}

#endif