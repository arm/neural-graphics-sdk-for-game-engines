/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#version 450
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_GOOGLE_include_directive : enable

#include "ffx_opticalflow_callbacks_glsl.h"

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform Uniforms {
    FfxFloat32x4x4 _MotionMatM1P1;     // 64 B
    FfxInt32x2     _OutputDims;        // 8 B  (270p - half of depth resolution)
    FfxInt32x2     _DepthDims;         // 8 B
    FfxFloat32x2   _InvDepthDims;      // 8 B
    FfxInt32x2     _ColorDims;         // 8 B
    FfxInt32x2     _MvDepthDims;       // 8 B
    FfxFloat32x2   _InvMvDepthDims;    // 8 B
}ubo;

layout(set = 0, binding = 1)        uniform texture2D r_depth_tm1; // 540p | R32_FLOAT 32bpp
layout(set = 0, binding = 2, rg16f) uniform writeonly mediump image2D rw_motion_vectors; // 270p (downsampled) | RG16F 32bpp

//------------------------------------------------------------------------------

// Returns the calculated camera motion in the same forward/backward convention as the input motion matrix.
FFX_MIN16_F2 CalculateCameraMotion(FfxFloat32x2 uv, FfxFloat32 depth, FfxFloat32x4x4 MotionMat)
{
    // Calculate "camera motion vectors" based on camera motion via ViewProjection mat. Fails for animated objects
    FfxFloat32x2 uvT = FfxFloat32x2(uv.x, 1.0HF - uv.y);
    FfxFloat32x4 clip = FfxFloat32x4(2.0f * uvT - 1.0f, depth, 1.0f);
    FfxFloat32x4 reproj = MotionMat * clip;
    FfxFloat32x2 uvProj = ((reproj.xy / reproj.w) + 1.0f) * 0.5f;

    FFX_MIN16_F2 velocity = FFX_MIN16_F2(uvT - uvProj);

    // Remove outliers
    FFX_MIN16_F basically_0_motion = 1e-5HF;
    velocity = all(lessThan(abs(velocity), FFX_MIN16_F2(basically_0_motion))) ? FFX_MIN16_F2(0.0HF) : velocity;
    velocity = any(isnan(velocity)) ? FFX_MIN16_F2(0.0HF) : velocity;
    velocity = any(isinf(velocity)) ? FFX_MIN16_F2(0.0HF) : velocity;

    // Invert direction to follow mv convention
    return FFX_MIN16_F2(-velocity.x, velocity.y);
}

void ComputeMVHints(FfxInt32x2 output_pixel)
{
    if (any(greaterThanEqual(output_pixel, ubo._OutputDims))) return;

    // Match pass 10's representative motion/depth pixel for each flow-lane hint.
    FfxInt32x2 mv_depth_pixel =
        (output_pixel * ubo._MvDepthDims + ubo._MvDepthDims / 2) / ubo._OutputDims;
    FfxFloat32x2 uv = (FfxFloat32x2(mv_depth_pixel) + 0.5f) * ubo._InvMvDepthDims;
    FfxInt32x2 depthDims  = ubo._DepthDims;
    FfxInt32x2 depth_pixel =
        clamp(FfxInt32x2(floor(uv * FfxFloat32x2(depthDims))), FfxInt32x2(0), depthDims - FfxInt32x2(1));

    const FfxInt32 iSampleCount = 9;
    FfxFloat32 depth_offset[iSampleCount];

    FfxFloat32x2 depth_uv  = (FfxFloat32x2(depth_pixel) + 0.5f) * ubo._InvDepthDims;
    FfxFloat32x4 gather_tl = textureGather(sampler2D(r_depth_tm1, s_PointClamp), depth_uv - ubo._InvDepthDims, 0);
    FfxFloat32x4 gather_br = textureGather(sampler2D(r_depth_tm1, s_PointClamp), depth_uv, 0);

    depth_offset[0] = gather_tl.y;
    depth_offset[1] = gather_br.z;
    depth_offset[2] = gather_br.x;
    depth_offset[3] = gather_tl.z;
    depth_offset[4] = gather_tl.x;
    depth_offset[5] = texelFetch(
        r_depth_tm1, clamp(depth_pixel + FfxInt32x2(-1, +1), FfxInt32x2(0), depthDims - FfxInt32x2(1)), 0).r;
    depth_offset[6] = gather_br.y;
    depth_offset[7] = gather_tl.w;
    depth_offset[8] = texelFetch(
        r_depth_tm1, clamp(depth_pixel + FfxInt32x2(+1, -1), FfxInt32x2(0), depthDims - FfxInt32x2(1)), 0).r;

    // Find closest depth among 9-tap neighbors
    FfxFloat32 nearest_depth = depth_offset[0];
    #pragma unroll
    for (FfxInt32 iSampleIndex = 1; iSampleIndex < iSampleCount; ++iSampleIndex) {
        FfxFloat32 tap = depth_offset[iSampleIndex];
#if DEPTH_INVERTED
        if (tap > nearest_depth) {
#else
        if (tap < nearest_depth) {
#endif
            nearest_depth = tap;
        }
    }

    FFX_MIN16_F2 motion = CalculateCameraMotion(clamp(uv, 0.0f, 1.0f), nearest_depth, ubo._MotionMatM1P1);

    // Convert from UV space to display pixel space according to DDK convention [From Optical Flow 3.2]
    motion *= FFX_MIN16_F2(ubo._ColorDims);

    imageStore(rw_motion_vectors, output_pixel, FFX_MIN16_F4(motion, 0.HF, 0.HF));
}


// entry-point
void main()
{
    FfxInt32x2 output_pixel = FfxInt32x2(gl_GlobalInvocationID.xy);

    ComputeMVHints(output_pixel);
}
