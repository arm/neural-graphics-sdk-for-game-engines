/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef GPU_FFX_FRAMEINTERPOLATION_10_WARP_FLOW_PASS_H
#define GPU_FFX_FRAMEINTERPOLATION_10_WARP_FLOW_PASS_H

#include "ffx_frameinterpolation_callbacks_glsl.h"
// Must include this file after layouts are defined, due to containing hard-coded variable
#include "quant.h"

//------------------------------------------------------------------------------

const float kBilinearWeightThreshold = 0.1f;

void Warp4Tap(int32_t2 dtId, float2 motion, float depth, float t, float s)
{
    float2   vector = motion * float2(MvDepthLaneSize());
    uint32_t qdata  = qFloat(s * vector, depth);

    float2 warp_uv = (float2(dtId) + 0.5f) * MvDepthLaneSizeRcp();
    warp_uv += motion * t;

    BilinearSamplingData bilinear = GetBilinearSamplingData(warp_uv, MvDepthLaneSize());
    for (int32_t i = 0; i < 4; ++i)
    {
        if (bilinear.fWeights[i] <= kBilinearWeightThreshold)
        {
            continue;
        }

        int32_t2 store_pos = bilinear.iBasePos + bilinear.iOffsets[i];
        if (!IsOnScreen(store_pos, MvDepthLaneSize()))
        {
            continue;
        }

        UpdateWarpMotionQDataTp1(FFX_MIN16_I2(store_pos), qdata);
    }
}

float LoadDilatedDepthTap3x3(int32_t2 pos, int32_t2 offset, int32_t2 depth_size, highp texture2D depth_texture, out int32_t2 clamped_sample_pos)
{
    int32_t2 sample_pos = pos + offset;
    clamped_sample_pos  = clamp(sample_pos, int32_t2(0), depth_size - int32_t2(1));
    float valid_sample  = float(all(equal(sample_pos, clamped_sample_pos)));
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH
    float non_valid_depth = 0.0f;  // OOB samples are considered to be at the far plane, which is 0.0 in inverted depth
#else
    float non_valid_depth = 1.0f;
#endif
    return mix(non_valid_depth, texelFetch(depth_texture, clamped_sample_pos, 0).x, valid_sample);
}

float LoadDilatedDepthTap3x3(int32_t2 pos, int32_t2 offset, int32_t2 depth_size, highp texture2D depth_texture)
{
    int32_t2 clamped_sample_pos;
    return LoadDilatedDepthTap3x3(pos, offset, depth_size, depth_texture, clamped_sample_pos);
}

// return the nearest depth in a 3x3 region around the given pixel
float LoadDilatedDepth3x3(int32_t2 pos, highp texture2D depth_texture)
{
    const int32_t  iSampleCount          = 9;
    const int32_t2 depth_size            = int32_t2(textureSize(depth_texture, 0));
    const int32_t2 offsets[iSampleCount] = {int32_t2(+0, +0),
                                            int32_t2(+1, +0),
                                            int32_t2(+0, +1),
                                            int32_t2(+0, -1),
                                            int32_t2(-1, +0),
                                            int32_t2(-1, +1),
                                            int32_t2(+1, +1),
                                            int32_t2(-1, -1),
                                            int32_t2(+1, -1)};

    float depth_taps[iSampleCount];

    // Pull out the depth loads to allow SC to batch them.
    depth_taps[0] = texelFetch(depth_texture, ClampTexelCoord(pos, depth_texture), 0).x;
    depth_taps[1] = LoadDilatedDepthTap3x3(pos, offsets[1], depth_size, depth_texture);
    depth_taps[2] = LoadDilatedDepthTap3x3(pos, offsets[2], depth_size, depth_texture);
    depth_taps[3] = LoadDilatedDepthTap3x3(pos, offsets[3], depth_size, depth_texture);
    depth_taps[4] = LoadDilatedDepthTap3x3(pos, offsets[4], depth_size, depth_texture);
    depth_taps[5] = LoadDilatedDepthTap3x3(pos, offsets[5], depth_size, depth_texture);
    depth_taps[6] = LoadDilatedDepthTap3x3(pos, offsets[6], depth_size, depth_texture);
    depth_taps[7] = LoadDilatedDepthTap3x3(pos, offsets[7], depth_size, depth_texture);
    depth_taps[8] = LoadDilatedDepthTap3x3(pos, offsets[8], depth_size, depth_texture);

    float best_depth = depth_taps[0];
#pragma unroll
    for (int32_t iSampleIndex = 1; iSampleIndex < iSampleCount; ++iSampleIndex)
    {
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH
        best_depth = max(best_depth, depth_taps[iSampleIndex]);
#else
        best_depth      = min(best_depth, depth_taps[iSampleIndex]);
#endif
    }

    return best_depth;
}

float LoadDilatedDepthAndMotion3x3(int32_t2          pos,
                                   highp texture2D   depth_texture,
                                   mediump texture2D motion_texture,
                                   out half2         out_motion,
                                   out int32_t2      out_sample_pos,
                                   out float         out_center_depth)
{
    const int32_t  iSampleCount          = 9;
    const int32_t2 depth_size            = int32_t2(textureSize(depth_texture, 0));
    const int32_t2 offsets[iSampleCount] = {int32_t2(+0, +0),
                                            int32_t2(+1, +0),
                                            int32_t2(+0, +1),
                                            int32_t2(+0, -1),
                                            int32_t2(-1, +0),
                                            int32_t2(-1, +1),
                                            int32_t2(+1, +1),
                                            int32_t2(-1, -1),
                                            int32_t2(+1, -1)};

    int32_t2 sample_positions[iSampleCount];
    float    depth_taps[iSampleCount];

    // Pull out the depth loads to allow SC to batch them.
    sample_positions[0] = ClampTexelCoord(pos, depth_texture);
    depth_taps[0]       = texelFetch(depth_texture, sample_positions[0], 0).x;
    depth_taps[1]       = LoadDilatedDepthTap3x3(pos, offsets[1], depth_size, depth_texture, sample_positions[1]);
    depth_taps[2]       = LoadDilatedDepthTap3x3(pos, offsets[2], depth_size, depth_texture, sample_positions[2]);
    depth_taps[3]       = LoadDilatedDepthTap3x3(pos, offsets[3], depth_size, depth_texture, sample_positions[3]);
    depth_taps[4]       = LoadDilatedDepthTap3x3(pos, offsets[4], depth_size, depth_texture, sample_positions[4]);
    depth_taps[5]       = LoadDilatedDepthTap3x3(pos, offsets[5], depth_size, depth_texture, sample_positions[5]);
    depth_taps[6]       = LoadDilatedDepthTap3x3(pos, offsets[6], depth_size, depth_texture, sample_positions[6]);
    depth_taps[7]       = LoadDilatedDepthTap3x3(pos, offsets[7], depth_size, depth_texture, sample_positions[7]);
    depth_taps[8]       = LoadDilatedDepthTap3x3(pos, offsets[8], depth_size, depth_texture, sample_positions[8]);

    float    best_depth = out_center_depth = depth_taps[0];
    int32_t2 best_sample_pos               = sample_positions[0];
#pragma unroll
    for (int32_t iSampleIndex = 1; iSampleIndex < iSampleCount; ++iSampleIndex)
    {
        float sample_depth = depth_taps[iSampleIndex];
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH
        int32_t use_tap = int32_t(1.0f - step(sample_depth, best_depth));
#else
        int32_t use_tap = int32_t(1.0f - step(best_depth, sample_depth));
#endif
        int32_t2 use_tap2 = int32_t2(use_tap);
        best_sample_pos   = best_sample_pos * (int32_t2(1) - use_tap2) + sample_positions[iSampleIndex] * use_tap2;
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH
        best_depth = max(best_depth, sample_depth);
#else
        best_depth      = min(best_depth, sample_depth);
#endif
    }

    float2   motion_scale      = MotionVectorScale();
    int32_t2 motion_size       = int32_t2(textureSize(motion_texture, 0));
    int32_t2 motion_sample_pos = ClampTexelCoord(int32_t2(floor((float2(best_sample_pos) + 0.5f) / float2(depth_size) * float2(motion_size))), motion_texture);
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH
    half valid_motion = half(1.0f - step(best_depth, 0.0f));
#else
    half valid_motion = half(1.0f - step(1.0f, best_depth));
#endif
    out_motion     = half2(CancelMotionVectorJitter(texelFetch(motion_texture, motion_sample_pos, 0).rg * motion_scale)) * valid_motion;
    out_sample_pos = best_sample_pos;
    return best_depth;
}

float dynamic_mask(float2 a, float2 b, float abs_eps, float q_margin, float rel_eps)
{
    float diff      = length(a - b);
    float mag       = max(length(a), length(b));
    float threshold = max(abs_eps, rel_eps * mag + q_margin);
    return float(diff >= threshold);
}

void WriteDynamicMask(int32_t2 output_pixel, uint8_t value)
{
    float mask = value > uint8_t(0) ? 1.0f : 0.0f;
    StoreDynamicMaskTp1(output_pixel, mask);
}

half2 CalculateCameraMotion(float2 uv, float depth, mat4 MotionMat, out half mask, out float depth_reproj)
{
    depth = max(depth, 1e-6f);  // Avoid issues reprojection of points at far plane

    float2 uvT    = float2(uv.x, 1.0f - uv.y);
    float4 clip   = float4(2.0f * uvT - 1.0f, depth, 1.0f);
    float4 reproj = MotionMat * clip;

    bool invalid = reproj.w <= 1e-6f || any(isnan(reproj)) || any(isinf(reproj));
    mask         = half(invalid);
    if (invalid)
    {
        depth_reproj = depth;
        return half2(0.0HF);
    }

    depth_reproj = reproj.z / reproj.w;

    float2 uvProj = ((reproj.xy / reproj.w) + 1.0f) * 0.5f;

    float2 velocity = uvT - uvProj;

    float basically_0_motion = 1e-5f;
    velocity                 = all(lessThan(abs(velocity), float2(basically_0_motion))) ? float2(0.0f) : velocity;
    velocity                 = any(isnan(velocity)) ? float2(0.0f) : velocity;
    velocity                 = any(isinf(velocity)) ? float2(0.0f) : velocity;

    return half2(-velocity.x, velocity.y);
}

void warp_flow(int32_t2 output_pixel)
{
    //-------------------------------------------------------------------------
    // 1) Dispatch guard
    //-------------------------------------------------------------------------
    int32_t2 mvDims = MvDepthLaneSize();
    if (any(greaterThanEqual(output_pixel, mvDims)))
        return;

    int32_t  output_pixel_cont = output_pixel.x + output_pixel.y * mvDims.x;
    float2   uv                = (float2(output_pixel.xy) + 0.5f) * MvDepthLaneSizeRcp();
    int32_t2 iDepthCoord       = int32_t2(floor(uv * float2(textureSize(_DepthTp1Texture, 0))));
    int32_t2 iDepthM1Coord     = int32_t2(floor(uv * float2(textureSize(r_depth_tm1, 0))));
    int32_t2 iDMCoord          = int32_t2(floor(uv * float2(textureSize(_DynamicMaskTm1, 0))));

    //-------------------------------------------------------------------------
    // 2) Dilate depth t+1 and rendered motion in 3x3 region
    //-------------------------------------------------------------------------
    half2    mv_p1_f30_m1;  // Dilated motion vectors
    int32_t2 mv_p1_f30_m1_sample_pos;
    half     cm_p1_f30_m1_mask;        // Invalid camera mv mask
    float    depth_dilated_p1_reproj;  // Ignore, not used
    float    depth_p1;

    float depth_dilated_p1 = LoadDilatedDepthAndMotion3x3(iDepthCoord,
                                                          r_depth_tp1,
                                                          r_motion_tp1,
                                                          mv_p1_f30_m1,
                                                          mv_p1_f30_m1_sample_pos,
                                                          depth_p1  // Non-dilated depth
    );
    // Write downsampled depth to depth_tm1_next BEFORE normalising (raw device depth)
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_MANAGE_PREVIOUS_DEPTH
    StoreDepthTm1Next(output_pixel, FfxFloat32(depth_p1));
#endif

    float2 mv_p1_f30_m1_sample_uv = (float2(mv_p1_f30_m1_sample_pos) + 0.5f) / float2(textureSize(_DepthTp1Texture, 0));
    half2  cm_p1_f30_m1 = CalculateCameraMotion(mv_p1_f30_m1_sample_uv, depth_dilated_p1, GetMotionTP1ToTM1(), cm_p1_f30_m1_mask, depth_dilated_p1_reproj);

    // Depending on the engine, there might not be mvs at the skybox.
    // Just in case, use generated camera motion vectors here
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH  // 0 is far plane
    int is_skybox = int(depth_dilated_p1 <= 0.001f);
#else
    int  is_skybox    = int(depth_dilated_p1 >= 0.999f);
#endif
    mv_p1_f30_m1 = is_skybox != 0 ? cm_p1_f30_m1 : mv_p1_f30_m1;

    //-------------------------------------------------------------------------
    // 4) Depth-aware scatter of t+1 data to intermediate t frame
    //-------------------------------------------------------------------------
    // Load and normalize p1 depth before scattering.
    float invMvArea  = MvDepthLaneSizeRcp().x * MvDepthLaneSizeRcp().y;
    depth_dilated_p1 = InvertAndNormalizeDepth(depth_dilated_p1, output_pixel_cont, invMvArea);
    Warp4Tap(output_pixel, float2(mv_p1_f30_m1), depth_dilated_p1, 1.0f - Timestep(), 1.0f);

    //-------------------------------------------------------------------------
    // Bidirectional scattering
    //-------------------------------------------------------------------------

    //-------------------------------------------------------------------------
    // 5) Dilate depth t-1 in 3x3 region
    //-------------------------------------------------------------------------
    float depth_dilated_m1 = LoadDilatedDepth3x3(iDepthM1Coord, r_depth_tm1);

    //-------------------------------------------------------------------------
    // 6) Calculate dilated camera motion for t-1 to t+1 reprojection
    //-------------------------------------------------------------------------
    half  cm_m1_f30_p1_mask;
    float depth_dilated_m1_preproj;
    half2 cm_m1_f30_p1 = CalculateCameraMotion(uv, depth_dilated_m1, GetMotionTM1ToTP1(), cm_m1_f30_p1_mask, depth_dilated_m1_preproj);

    //-------------------------------------------------------------------------
    // 7) Get feedback dynamic mask, and calculate dynamic for next frame
    //-------------------------------------------------------------------------
    // 1 == Invalid vector
    // 0 == Valid vector to scatter
    // uint8_t dynamic_mask_m1 = uint8_t(texelFetch(_DynamicMask, iDMCoord, 0).r);
    uint8_t dynamic_mask_m1 = uint8_t(round(LoadDynamicMaskTm1(iDMCoord) * 255.0f));

    uint8_t dynamic_mask_p1 = uint8_t(dynamic_mask(cm_p1_f30_m1 * float2(mvDims),
                                                   mv_p1_f30_m1 * float2(mvDims),
                                                   MvSimilarityThreshold(),
                                                   MvSimilarityNoiseThreshold(),
                                                   MvSimilarityRelativeThreshold()) > 0.0);

    // Save dynamic mask to feedback for the next frame pair.
    WriteDynamicMask(output_pixel, dynamic_mask_p1 + uint8_t(cm_p1_f30_m1_mask));

    //-------------------------------------------------------------------------
    // 8) Depth-aware scatter of t-1 data to intermediate t frame
    //-------------------------------------------------------------------------

    float2 uv_m1       = uv - (mv_p1_f30_m1 * Timestep());
    bool   dynamic_oob = uv_m1.x < 0.0f || uv_m1.y < 0.0f || uv_m1.x >= 1.0f || uv_m1.y >= 1.0f;

    if (dynamic_mask_p1 > 0 && dynamic_oob)
    {
        uint32_t qdata = 0x7FFFFFFF;
        UpdateWarpMotionQDataTp1(FFX_MIN16_I2(output_pixel), qdata);
    }

    // Do not scatter invalid motion vectors.
    if (dynamic_mask_m1 + uint8_t(cm_m1_f30_p1_mask) > 0)
    {
        return;
    }

    // Normalize reprojected depth before scattering.
    // Use reprojected depth for m1 data to ensure better depth comparison when scattering
    depth_dilated_m1_preproj = InvertAndNormalizeDepth(depth_dilated_m1_preproj, output_pixel_cont, invMvArea);
    Warp4Tap(output_pixel, float2(cm_m1_f30_p1), depth_dilated_m1_preproj, Timestep(), -1.0f);
}

#endif