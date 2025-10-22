/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef GPU_NSS_POSTPROCESS_H
#define GPU_NSS_POSTPROCESS_H

void Postprocess(int32_t2 output_pixel)
{
    if (any(greaterThanEqual(output_pixel, OutputDims())))
        return;

    float2   uv          = (float2(output_pixel) + 0.5) * InvOutputDims();
    int32_t2 input_pixel = int32_t2(uv * InputDims());

    //-------------------------------------------------------------------------
    // 1) Warp history
    //-------------------------------------------------------------------------
    half  onscreen;
    half3 history = LoadWarpedHistory(uv, input_pixel, onscreen);

    //-------------------------------------------------------------------------
    // 2) KPN filter → col
    //-------------------------------------------------------------------------
    half4 col_to_accum;
    half3 colour = LoadAndFilterColour(output_pixel, uv, col_to_accum);

    // -------------------------------------------------------------------------
    // 3) Load temporal parameters
    //-------------------------------------------------------------------------
    half theta, alpha;
    LoadTemporalParameters(uv, theta, alpha);

    //-------------------------------------------------------------------------
    // 3) Rectify history, force reset when offscreen
    //-------------------------------------------------------------------------
    half3 rectified = lerp(colour, history, theta * onscreen);

    //-------------------------------------------------------------------------
    // 3) Accumulate new sample
    //-------------------------------------------------------------------------
    half3 accumulated = lerp(Tonemap(rectified), Tonemap(col_to_accum.rgb), alpha * col_to_accum.a);

    //-------------------------------------------------------------------------
    // 4) Inverse tonemap + exposure and write output
    //-------------------------------------------------------------------------
    half3 out_linear = InverseTonemap(accumulated) * InvExposure();
    WriteUpsampledColour(output_pixel, out_linear);
}
#endif  // GPU_NSS_POSTPROCESS_H