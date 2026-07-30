// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "ffx_frameinterpolation_callbacks_glsl.h"
#include "ffx_frameinterpolation_common.h"

/// Downsample full-resolution colour to ofInputColour size for OF input.
/// Uses nearest-neighbor sampling with coordinate scaling (matching reference 01_downsample_of_colour.frag).
void downsample_of_colour(int32_t2 output_pixel)
{
    int32_t2 output_dims = int32_t2(RenderSize());  // ofInputColour passed via RenderSize for this dispatch
    if (any(greaterThanEqual(output_pixel, output_dims)))
    {
        return;
    }

    int32_t2 input_dims = DisplaySize();
    if (all(equal(input_dims, output_dims)))
    {
        // No-op passthrough when sizes match (H <= 1080)
        StoreColourP1Internal(output_pixel, LoadCurrentBackbuffer(output_pixel));
        return;
    }

    // Nearest-neighbor downsample with coordinate mapping
    FfxFloat32x2 scale       = FfxFloat32x2(input_dims) / FfxFloat32x2(output_dims);
    int32_t2     input_pixel = int32_t2(round((FfxFloat32x2(output_pixel) + 0.5f) * scale - 0.5f));
    input_pixel              = clamp(input_pixel, int32_t2(0), input_dims - int32_t2(1));
    StoreColourP1Internal(output_pixel, LoadCurrentBackbuffer(input_pixel));
}
