/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "quant.h"

#ifndef FILL_HOLES_COMMON
#define FILL_HOLES_COMMON

#ifndef FFX_FRAMEINTERPOLATION_DEPTH_QDATA_TEXTURE
#error "FFX_FRAMEINTERPOLATION_DEPTH_QDATA_TEXTURE must be defined before including fill_holes_common.h"
#endif

#define _DepthQDataTex usampler2D(FFX_FRAMEINTERPOLATION_DEPTH_QDATA_TEXTURE, s_PointClamp)

/*
    To use, declare the following:
    uint32_t LoadPackedQDepth(FfxInt32x2 coord)
    {
        return LoadWarpedFlowQDataTM1(coord);
    }
*/
void FillHoles(FfxInt32x2 output_pixel, FfxInt32x2 output_dims, FfxFloat32x2 inv_output_dims)
{
    // Loosely based on FidelityFX FSR2's reconstructed velocity/depth dilation.
    const FfxInt32 iSampleCount = 9;
    // 3x3 neighborhood layout:
    // (-1,-1) (0,-1) (1,-1)
    // (-1, 0) (0, 0) (1, 0)
    // (-1, 1) (0, 1) (1, 1)
    FfxUInt32 depth_mvs_packed[iSampleCount];

    // Normalized pixel coordinates for texture sampling
    FfxFloat32x2 normalizedCoord = (FfxFloat32x2(output_pixel) + 0.5f) / FfxFloat32x2(output_dims);
    FfxFloat32x2 texelSize       = 1.0f / FfxFloat32x2(output_dims);

    // Gather around (x-1, y-1): covers (-1,-1), (0,-1), (-1,0), (0,0)
    FfxUInt32x4 gather1 = textureGather(_DepthQDataTex, normalizedCoord - texelSize, 0);
    // GLSL gather component mapping at pixel centers:
    // x = (u0, v1), y = (u1, v1), z = (u1, v0), w = (u0, v0)
    depth_mvs_packed[0] = gather1.y;  // (0, 0)
    depth_mvs_packed[3] = gather1.z;  // (0,-1)
    depth_mvs_packed[4] = gather1.x;  // (-1,0)
    depth_mvs_packed[7] = gather1.w;  // (-1,-1)

    // Gather around (x, y): covers (0,0), (1,0), (0,1), (1,1)
    FfxUInt32x4 gather2 = textureGather(_DepthQDataTex, normalizedCoord, 0);
    depth_mvs_packed[1] = gather2.z;  // (1,0)
    depth_mvs_packed[2] = gather2.x;  // (0,1)
    depth_mvs_packed[6] = gather2.y;  // (1,1)

    // Two additional texelFetch calls for the remaining corner pixels
    depth_mvs_packed[5] = LoadPackedQDepth(ClampTexelCoord(output_pixel + FfxInt32x2(-1, 1), FFX_FRAMEINTERPOLATION_DEPTH_QDATA_TEXTURE));  // (-1, 1)
    depth_mvs_packed[8] = LoadPackedQDepth(ClampTexelCoord(output_pixel + FfxInt32x2(1, -1), FFX_FRAMEINTERPOLATION_DEPTH_QDATA_TEXTURE));  // (1, -1)

    // Find closest depth
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

#undef _DepthQDataTex
