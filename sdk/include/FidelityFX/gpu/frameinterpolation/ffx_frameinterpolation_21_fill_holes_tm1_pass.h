/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef GPU_FRAMEINTERPOLATION_10_FILL_HOLES_PASS_H
#define GPU_FRAMEINTERPOLATION_10_FILL_HOLES_PASS_H

#include "ffx_frameinterpolation_callbacks_glsl.h"
uint32_t LoadPackedQDepth(int32_t2 coord)
{
    return LoadWarpedFlowQDataTM1(coord);
}

void StoreDilatedVector(int32_t2 coord, FFX_MIN16_F2 value)
{
    StoreWarpFilledFlowTm1(coord, value);
}

#define FFX_FRAMEINTERPOLATION_DEPTH_QDATA_TEXTURE r_warped_flow_qdata_tm1
#include "fill_holes_common.h"
#undef FFX_FRAMEINTERPOLATION_DEPTH_QDATA_TEXTURE

void fill_holes(int32_t2 output_pixel)
{
    //-------------------------------------------------------------------------
    // 1) Dispatch guard
    //-------------------------------------------------------------------------
    int32_t2 ofSize = FlowLaneSize();
    if (any(greaterThanEqual(output_pixel, ofSize)))
        return;

    //-------------------------------------------------------------------------
    // 2) Fill holes with nearest neighbour sampling in 3x3 window.
    //-------------------------------------------------------------------------
    float2 invOfSize = FlowLaneSizeRcp();
    FillHoles(output_pixel, ofSize, invOfSize);
}

#endif