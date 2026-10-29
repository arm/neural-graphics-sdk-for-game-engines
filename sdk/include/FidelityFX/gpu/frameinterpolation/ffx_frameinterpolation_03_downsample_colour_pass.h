/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "ffx_frameinterpolation_02_downsample_all_input_pass.h"

void downsample_colour(int32_t2 output_pixel)
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
}
