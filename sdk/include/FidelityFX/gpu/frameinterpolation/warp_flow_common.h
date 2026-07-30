/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "quant.h"

#ifndef WARP_FLOW_COMMON
#define WARP_FLOW_COMMON

void Warp(FfxInt32x2                   dtId,
          half2                        motion,
          FfxFloat32                   depth,
          FfxInt32x2                   output_dims,
          FFX_MIN16_F                  t,
          FFX_MIN16_F                  s,
          FFX_PARAMETER_OUT FfxInt32x2 outCoord,
          FFX_PARAMETER_OUT FfxUInt32  outData)
{
    // Scatter depth and motion vector

    // Calculate target forward warp uv location
    FFX_MIN16_F2 vector = FFX_MIN16_F2(motion) * FFX_MIN16_F2(output_dims);

    // Calculate target forward warp texel location
    // Scale by intermediate timestep
    FfxInt32x2 coord = dtId + FfxInt32x2(floor(vector * t));

    if (!IsOnScreen(coord, output_dims))
        return;

    // Pack data for scattering. This packing has the layout:
    // (MSB->LSB): depth | exponent | signX | signY | mantissaX | mantissaY
    FfxUInt32 qdata = qFloat(s * vector, depth);

    outCoord = coord;
    outData  = qdata;
}

#endif  // WARP_FLOW_COMMON
