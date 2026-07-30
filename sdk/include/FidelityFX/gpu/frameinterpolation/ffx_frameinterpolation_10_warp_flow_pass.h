// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#ifndef GPU_FFX_FRAMEINTERPOLATION_00_WARP_FLOW_PASS_H
#define GPU_FFX_FRAMEINTERPOLATION_00_WARP_FLOW_PASS_H

#include "ffx_frameinterpolation_callbacks_glsl.h"
// Must include this file after layouts are defined, due to containing hard-coded variable
#include "warp_flow_common.h"

// Returns the calculated camera motion in the same forward/backward convention as the input motion matrix.
half2 CalculateCameraMotion(float2 uv, float depth, mat4 MotionMat, FFX_PARAMETER_OUT half mask)
{
    float2 uvT    = float2(uv.x, 1.0f - uv.y);
    float4 clip   = float4(2.0f * uvT - 1.0f, depth, 1.0f);
    float4 reproj = MotionMat * clip;
    mask          = half(reproj.w < 0);
    float2 uvProj = ((reproj.xy / reproj.w) + 1.0f) * 0.5f;

    float2 velocity = uvT - uvProj;

    float basically_0_motion = 1e-5f;
    velocity                 = all(lessThan(abs(velocity), float2(basically_0_motion))) ? float2(0.0f) : velocity;
    velocity                 = any(isnan(velocity)) ? float2(0.0f) : velocity;
    velocity                 = any(isinf(velocity)) ? float2(0.0f) : velocity;

    return half2(-velocity.x, velocity.y);
}

FFX_MIN16_F2 sym_ceil(FFX_MIN16_F2 x)
{
    return sign(x) * ceil(abs(x));
}

FFX_MIN16_F2 quantise(FFX_MIN16_F2 x)
{
    const int32_t full_bits = _BITS_X + _BITS_Y;
    const int32_t max_int   = (1 << full_bits) - 1;
    FFX_MIN16_F   c         = FFX_MIN16_F(max_int) / FFX_MIN16_F(_MAX_VAL);
    int32_t2      q         = int32_t2(clamp(sym_ceil(x * c), FFX_MIN16_F(-_MAX_VAL), FFX_MIN16_F(_MAX_VAL)));
    return FFX_MIN16_F2(q) / c;
}

FFX_MIN16_F dynamic_mask_v5(FFX_MIN16_F2 a, FFX_MIN16_F2 b, FFX_MIN16_F eps, FFX_MIN16_F tau)
{
    a                   = quantise(a);
    b                   = quantise(b);
    FFX_MIN16_F diff    = length(a - b);
    FFX_MIN16_F max_len = max(max(length(a), length(b)), tau);
    return max_len < tau ? FFX_MIN16_F(0.0) : FFX_MIN16_F(diff >= eps);
}

FFX_MIN16_F dynamic_mask_v4(FFX_MIN16_F2 a, FFX_MIN16_F2 b, FFX_MIN16_F eps, FFX_MIN16_F tau)
{
    FFX_MIN16_F diff  = length(a - b);
    FFX_MIN16_F denom = max(max(length(a), length(b)), tau);
    return FFX_MIN16_F((diff / denom) >= eps);
}

void warp_flow(int32_t2 output_pixel)
{
    int32_t2 renderSize = MvDepthLaneSize();
    if (any(greaterThanEqual(output_pixel, renderSize)))
        return;

    int32_t output_pixel_cont = output_pixel.x + output_pixel.y * renderSize.x;

    float  renderSizeAreaRcp = 1.0 / float(renderSize.x * renderSize.y);
    float2 uv                = (float2(output_pixel.xy) + 0.5f) * MvDepthLaneSizeRcp();

    // Load depth, normalise
    // TODO: do we need to look at neighbours when downsampling?
    FFX_MIN16_F depth_p1 = FFX_MIN16_F(LoadInputDepthTp1(uv));
    // Write downsampled depth to depth_tm1_next BEFORE normalising (raw device depth)
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_MANAGE_PREVIOUS_DEPTH
    StoreDepthTm1Next(output_pixel, FfxFloat32(depth_p1));
#endif
    // Calculate camera motion vectors BEFORE normalising depth!
    half  cm_p1_f30_m1_mask;
    half2 cm_p1_f30_m1 = CalculateCameraMotion(uv, depth_p1, GetMotionTP1ToTM1(), cm_p1_f30_m1_mask);
    depth_p1           = FFX_MIN16_F(InvertAndNormalizeDepth(depth_p1, output_pixel_cont, renderSizeAreaRcp));

    // Load game rendered mv
    half2 mv_p1_f30_m1 = LoadInputMotionTp1(uv);

    // Perform depth-aware forward warp
    int32_t2 outCoord;
    uint32_t outData;
    Warp(output_pixel, mv_p1_f30_m1, depth_p1, renderSize, FFX_MIN16_F(1.0) - FFX_MIN16_F(Timestep()), FFX_MIN16_F(1.0), outCoord, outData);
    UpdateWarpMotionQDataTp1(FFX_MIN16_I2(outCoord), outData);

    /*---------------------------.
    | :: DD mask holes writes :: |
    '---------------------------*/

    const int32_t  iSampleCount = 9;
    FFX_MIN16_F    offset_depth[iSampleCount];
    const int32_t2 offsets[9] = {int32_t2(+0, +0),
                                 int32_t2(+1, +0),
                                 int32_t2(+0, +1),
                                 int32_t2(+0, -1),
                                 int32_t2(-1, +0),
                                 int32_t2(-1, +1),
                                 int32_t2(+1, +1),
                                 int32_t2(-1, -1),
                                 int32_t2(+1, -1)};

    // pull out the depth loads to allow SC to batch them
    offset_depth[0] = FFX_MIN16_F(LoadInputDepthTp1WithOffset(uv, offsets[0]).r);
    offset_depth[1] = FFX_MIN16_F(LoadInputDepthTp1WithOffset(uv, offsets[1]).r);
    offset_depth[2] = FFX_MIN16_F(LoadInputDepthTp1WithOffset(uv, offsets[2]).r);
    offset_depth[3] = FFX_MIN16_F(LoadInputDepthTp1WithOffset(uv, offsets[3]).r);
    offset_depth[4] = FFX_MIN16_F(LoadInputDepthTp1WithOffset(uv, offsets[4]).r);
    offset_depth[5] = FFX_MIN16_F(LoadInputDepthTp1WithOffset(uv, offsets[5]).r);
    offset_depth[6] = FFX_MIN16_F(LoadInputDepthTp1WithOffset(uv, offsets[6]).r);
    offset_depth[7] = FFX_MIN16_F(LoadInputDepthTp1WithOffset(uv, offsets[7]).r);
    offset_depth[8] = FFX_MIN16_F(LoadInputDepthTp1WithOffset(uv, offsets[8]).r);

    // find closest depth
    int32_t     nearest_offset_index = 0;
    FFX_MIN16_F nearest_depth        = offset_depth[nearest_offset_index];
#pragma unroll
    for (int32_t iSampleIndex = 1; iSampleIndex < iSampleCount; ++iSampleIndex)
    {
        FFX_MIN16_F tap = offset_depth[iSampleIndex];
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH
        if (tap > nearest_depth)
        {
#else
        if (tap < nearest_depth)
        {
#endif
            nearest_depth        = tap;
            nearest_offset_index = iSampleIndex;
        }
    }

    FFX_MIN16_F2 nearest_motion = LoadInputMotionTP1WithOffset(uv, offsets[nearest_offset_index]);
    StoreHolesTp1(output_pixel, nearest_motion);
    StoreHolesTm1(output_pixel, nearest_motion);

    /*-------------------------------.
    | :: Bidirectional scattering :: |
    '-------------------------------*/

    // Load depth. Don't invert here as we need it for calculating camera motion vectors
    FFX_MIN16_F depth_m1 = FFX_MIN16_F(LoadInputDepthTm1(uv));

    // Calculate camera motion vectors BEFORE normalising depth
    half         cm_m1_f30_p1_mask;
    FFX_MIN16_F2 cm_m1_f30_p1 = CalculateCameraMotion(uv, depth_m1, GetMotionTM1ToTP1(), cm_m1_f30_p1_mask);

    // Get feedback dynamic mask, and calculate dynamic for next frame
    // 1 == Invalid vector
    // 0 == Valid vector to scatter
    uint8_t dynamic_mask_m1 = uint8_t(LoadDynamicMaskTm1(output_pixel));
    uint8_t dynamic_mask_p1 = uint8_t(
#if NEW_DYNAMIC_MASK
        dynamic_mask_v5(
            cm_p1_f30_m1 * FFX_MIN16_F2(renderSize), mv_p1_f30_m1 * FFX_MIN16_F2(renderSize), MvSimilarityThreshold(), MvSimilarityNoiseThreshold()) >
        FFX_MIN16_F(0.0)
#else
        dynamic_mask_v4(cm_p1_f30_m1, mv_p1_f30_m1, MvSimilarityThreshold(), MvSimilarityNoiseThreshold()) > FFX_MIN16_F(0.0)
#endif
    );

    // Save dynamic mask to feedback for next set of frames
    StoreDynamicMaskTp1(output_pixel, dynamic_mask_p1 + uint8_t(cm_p1_f30_m1_mask));

    // Dont scatter invalid mvs
    if (dynamic_mask_m1 + uint8_t(cm_m1_f30_p1_mask) > 0)
        return;

    // Normalise depth before scattering
    depth_m1 = FFX_MIN16_F(InvertAndNormalizeDepth(depth_m1, output_pixel_cont, renderSizeAreaRcp));

    // Perform depth-aware forward warp
    Warp(output_pixel, cm_m1_f30_p1, depth_m1, renderSize, FFX_MIN16_F(Timestep()), FFX_MIN16_F(-1.0f), outCoord, outData);
    UpdateWarpMotionQDataTp1(FFX_MIN16_I2(outCoord), outData);
}

#endif