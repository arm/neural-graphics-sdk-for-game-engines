/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "quant.h"

#ifndef FILL_HOLES_COMMON
#define FILL_HOLES_COMMON

/*
    To use, declare the following:
    FfxFloat32 LoadPackedDepth(FfxInt32x2 coord)
    {
        return dqFloatDepth(Load QData for coord);
    }
    FFX_MIN16_F2 LoadPackedMotion(FfxInt32x2 coord)
    {
        return dqFloatMv(Load QData for coord);
    }
    void StoreDilatedVector(FfxInt32x2 coord, FFX_MIN16_F2 value)
    {
        Store value;
    }
*/
void FillHoles(FfxInt32x2 output_pixel, FfxInt32x2 output_dims, FfxFloat32x2 inv_output_dims)
{
    /*
        Loosely based on:
        https://github.com/arm/accuracy-super-resolution-generic-library/blob/38697a58a6e7818ec9d28774bc073f537abb9178/
        include/gpu/fsr2/ffxm_fsr2_reconstruct_dilated_velocity_and_previous_depth.h#L59
    */
    const FfxInt32 iSampleCount = 9;
    FfxUInt32      depth_mvs_packed[iSampleCount];

    // pull out the depth loads to allow SC to batch them
    depth_mvs_packed[0] = LoadPackedQDepth(output_pixel + FfxInt32x2(0, 0));
    depth_mvs_packed[1] = LoadPackedQDepth(output_pixel + FfxInt32x2(1, 0));
    depth_mvs_packed[2] = LoadPackedQDepth(output_pixel + FfxInt32x2(0, 1));
    depth_mvs_packed[3] = LoadPackedQDepth(output_pixel + FfxInt32x2(0, -1));
    depth_mvs_packed[4] = LoadPackedQDepth(output_pixel + FfxInt32x2(-1, 0));
    depth_mvs_packed[5] = LoadPackedQDepth(output_pixel + FfxInt32x2(-1, 1));
    depth_mvs_packed[6] = LoadPackedQDepth(output_pixel + FfxInt32x2(1, 1));
    depth_mvs_packed[7] = LoadPackedQDepth(output_pixel + FfxInt32x2(-1, -1));
    depth_mvs_packed[8] = LoadPackedQDepth(output_pixel + FfxInt32x2(1, -1));

    // find closest depth
    FfxUInt32 nearest_packed_depth_mv = depth_mvs_packed[0];
#pragma unroll
    for (FfxInt32 iSampleIndex = 1; iSampleIndex < iSampleCount; ++iSampleIndex)
    {
        FfxUInt32 tap = depth_mvs_packed[iSampleIndex];
        if (tap > nearest_packed_depth_mv)
        {
            nearest_packed_depth_mv = tap;
        }
    }

    // Unpack / Dequantize motion
    FfxFloat32x2 motion = dqFloatMv(nearest_packed_depth_mv);
    motion *= FfxFloat32x2(inv_output_dims);

    // Write dilated vector
    StoreDilatedVector(output_pixel, FFX_MIN16_F2(motion));
}

#endif  // FILL_HOLES_COMMON
