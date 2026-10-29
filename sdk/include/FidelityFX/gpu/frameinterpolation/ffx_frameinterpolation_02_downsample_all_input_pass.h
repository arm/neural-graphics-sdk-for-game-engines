/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "ffx_frameinterpolation_callbacks_glsl.h"
#include "ffx_frameinterpolation_common.h"

void SelectNearestDepthCoord(int32_t2 output_pixel, int32_t2 output_size, out float selected_depth, out float2 selected_uv)
{
    int32_t2 depth_size = int32_t2(textureSize(r_depth_tp1, 0));
    if (all(equal(depth_size, output_size)))
    {
        selected_depth = LoadInputDepthTp1Pixel(output_pixel);
        selected_uv    = (float2(output_pixel) + 0.5f) / float2(depth_size);
        return;
    }

    float2   depth_per_output = float2(depth_size) / float2(output_size);
    int32_t2 base             = clamp(int32_t2(floor(float2(output_pixel) * depth_per_output)), int32_t2(0), depth_size - int32_t2(1));
    int32_t2 exclusive_end    = min(depth_size, max(base + int32_t2(1), int32_t2(ceil(float2(output_pixel + int32_t2(1)) * depth_per_output))));

    int32_t2 selected_coord = base;
    selected_depth          = LoadInputDepthTp1Pixel(selected_coord);

    for (int32_t y = base.y; y < exclusive_end.y; ++y)
    {
        for (int32_t x = base.x; x < exclusive_end.x; ++x)
        {
            int32_t2 sample_coord = int32_t2(x, y);
            float    sample_depth = LoadInputDepthTp1Pixel(sample_coord);
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH
            if (sample_depth > selected_depth)
#else
            if (sample_depth < selected_depth)
#endif
            {
                selected_depth = sample_depth;
                selected_coord = sample_coord;
            }
        }
    }

    selected_uv = (float2(selected_coord) + 0.5f) / float2(depth_size);
}

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DEPTH_TM1_NEXT) && defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_REDUCED_MOTION_P1)
void downsample_all_input(int32_t2 output_pixel)
{
    int32_t2 output_size = MvDepthLaneSize();
    if (any(greaterThanEqual(output_pixel, output_size)))
    {
        return;
    }

    float  selected_depth;
    float2 selected_uv;
    SelectNearestDepthCoord(output_pixel, output_size, selected_depth, selected_uv);

    StorePreprocessColourP1(output_pixel, SampleOfColourP1(selected_uv));
    StoreDepthTm1Next(output_pixel, selected_depth);
    StoreReducedMotionP1(output_pixel, LoadInputMotionTp1Raw(FfxInt32x2(floor(selected_uv * FfxFloat32x2(textureSize(r_motion_tp1, 0))))));
}
#endif
