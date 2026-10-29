/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "ffx_frameinterpolation_callbacks_glsl.h"
#include "ffx_frameinterpolation_common.h"

// Resolves the inpaint mip pyramid into a single flow-lane inpaint result
// via a multi-scale weighted blend. Pure colour-domain logic — no depth
// convention dependency. Operates on the flow lane.
#define FFX_FRAMEINTERPOLATION_INPAINT_MAX_SAMPLES 5

void inpaint_resolve(FfxInt32x2 output_pixel)
{
    //-------------------------------------------------------------------------
    // 1) Dispatch guard
    //-------------------------------------------------------------------------
    FfxInt32x2 flow_dims = FlowLaneSize();
    if (any(greaterThanEqual(output_pixel, flow_dims)))
    {
        return;
    }

    FfxFloat32x2 uv = (FfxFloat32x2(output_pixel) + 0.5f) * FlowLaneSizeRcp();

    //-------------------------------------------------------------------------
    // 2) Inpaint hole with weighted average of inpaint mip chain samples
    //-------------------------------------------------------------------------

    // weight = 1 / 2^i
    FfxFloat32 weights[FFX_FRAMEINTERPOLATION_INPAINT_MAX_SAMPLES];
    weights[0] = 1.0f;
    weights[1] = 0.5f;
    weights[2] = 0.25f;
    weights[3] = 0.125f;
    weights[4] = 0.0625f;

    FfxFloat32x3 rgb_inpaint = FfxFloat32x3(0.0f);
    FfxFloat32   weight_sum  = 0.0f;

#pragma unroll
    for (FfxInt32 i = 0; i < FFX_FRAMEINTERPOLATION_INPAINT_MAX_SAMPLES; ++i)
    {
        FfxFloat32x4 s = SampleInpaintLevel(uv, i);
        rgb_inpaint += s.rgb * weights[i];
        weight_sum += s.a * weights[i];
    }

    if (weight_sum > 0.0f)
    {
        rgb_inpaint /= weight_sum;
    }

    // Write out inpaint result, encoding hole/validity in alpha channel
    FfxFloat32 inpaint_valid = weight_sum > 0.0f ? 1.0f : 0.0f;
    StoreInpaintResult(output_pixel, FfxFloat32x4(rgb_inpaint * inpaint_valid, inpaint_valid));
}
