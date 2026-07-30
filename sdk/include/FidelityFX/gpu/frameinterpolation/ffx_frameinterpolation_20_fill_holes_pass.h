// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#ifndef GPU_FRAMEINTERPOLATION_10_FILL_HOLES_PASS_H
#define GPU_FRAMEINTERPOLATION_10_FILL_HOLES_PASS_H

#include "ffx_frameinterpolation_callbacks_glsl.h"
uint32_t LoadPackedQDepth(int32_t2 coord)
{
    return LoadWarpedMotionQDataTP1(coord);
}

void StoreDilatedVector(int32_t2 coord, FFX_MIN16_F2 value)
{
    StoreWarpFilledMotionTp1(coord, value);
}
#include "fill_holes_common.h"

void fill_holes(int32_t2 output_pixel)
{
    int32_t2 renderSize = MvDepthLaneSize();
    if (any(greaterThanEqual(output_pixel, renderSize)))
        return;

    float2 invRenderSize = MvDepthLaneSizeRcp();
    FillHoles(output_pixel, renderSize, invRenderSize);
}

#endif