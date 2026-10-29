/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "ffx_frameinterpolation_callbacks_glsl.h"
#include "ffx_frameinterpolation_common.h"

// Define FFX_FRAMEINTERPOLATION_INPAINT_FROM_SEED in the seed→level-0 entry.

void downsample_inpaint(FfxInt32x2 coord)
{
    FfxInt32x2 output_dims = InpaintDestinationSize();
    if (any(greaterThanEqual(coord, output_dims)))
    {
        return;
    }

    FfxFloat32x2 uv = (FfxFloat32x2(coord) + 0.5f) / FfxFloat32x2(output_dims);

#ifdef FFX_FRAMEINTERPOLATION_INPAINT_FROM_SEED
    FfxFloat32x4 inpaint_sample = SampleInpaintSeed(uv);
#else
    FfxFloat32x4 inpaint_sample = SampleInpaintSource(uv);
#endif

    // Validity-weighted un-premultiply (guarded against the a==0 hole to avoid NaN).
    FfxFloat32   sample_valid = inpaint_sample.a > 0.0f ? 1.0f : 0.0f;
    FfxFloat32x3 rgb          = inpaint_sample.a > 0.0f ? inpaint_sample.rgb / inpaint_sample.a : FfxFloat32x3(0.0f);

    StoreInpaintDestination(coord, FfxFloat32x4(rgb * sample_valid, sample_valid));
}
