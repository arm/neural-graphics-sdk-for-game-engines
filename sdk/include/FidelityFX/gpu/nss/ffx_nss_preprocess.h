/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef GPU_NSS_PREPROCESS_H
#define GPU_NSS_PREPROCESS_H

void Preprocess(int32_t2 input_pixel)
{
    if (any(greaterThanEqual(input_pixel, InputDims())))
        return;

    float2 uv = (float2(input_pixel) + 0.5f) * InvInputDims();

    //-------------------------------------------------------------------------
    // 1) Dilate depth, find nearest pixel coordinate
    //-------------------------------------------------------------------------
    float    depth_dilated        = float(0.f);
    int32_t2 nearest_pixel_offset = int32_t2(0);
    FindNearestDepth(input_pixel, RenderSize(), depth_dilated, nearest_pixel_offset);

    //-------------------------------------------------------------------------
    // 2) Load motion vectors
    //-------------------------------------------------------------------------
    half2 motion = LoadMotion(input_pixel + nearest_pixel_offset);

    // Suppress very small motion - no value in resampling here
    half2 motion_pix = motion * half2(RenderSize());
    motion *= half(dot(motion_pix, motion_pix) > MotionWarpThresh());

    // calculate sample position(s) for everything in `tm1` frame
    float2 reproj_uv       = uv - float2(motion);
    float2 unjitter_tm1_uv = reproj_uv - JitterOffsetTm1Uv();

    //-------------------------------------------------------------------------
    // 3) Calculate depth-based disocclusion mask
    //-------------------------------------------------------------------------
    half disocclusion_mask = half(ComputeDepthClip(unjitter_tm1_uv, depth_dilated));

    // Scale disocclusion mask on static frames to let network know this is happening under
    // static conditions, reduces jitter differences across frames causing false flags
    half dm_scale     = dot(motion_pix, motion_pix) > MotionDisThresh() ? half(1.0f) : DisocclusionScale();
    disocclusion_mask = disocclusion_mask * dm_scale;

    //-------------------------------------------------------------------------
    // 4) Downsample + warp history buffer
    //-------------------------------------------------------------------------
    half3 warped_history = WarpHistory(reproj_uv);

    //-------------------------------------------------------------------------
    // 5) Read current low-res / jittered / aliased colour
    //-------------------------------------------------------------------------
    half3 jittered_colour = LoadColour(input_pixel);

    //-------------------------------------------------------------------------
    // 6) Calculate derivative of `luma`
    //    helps identifying high-frequency flicker due to jitter
    //-------------------------------------------------------------------------
    half2 luma_derivative = CalculateLumaDerivative(reproj_uv, jittered_colour, disocclusion_mask);

    //-------------------------------------------------------------------------
    // 7) Warp temporal feedback
    //-------------------------------------------------------------------------
    half4 temporal_feedback = WarpFeedback(reproj_uv);

    //-------------------------------------------------------------------------
    // 8) Convert dilated depth coord to a position offset
    //-------------------------------------------------------------------------
    uint8_t enc_depth_offset = EncodeNearestDepthCoord(nearest_pixel_offset);

    //-------------------------------------------------------------------------
    // 9) Write Outputs
    //-------------------------------------------------------------------------
    // Consumed by NE
    WriteToTensor(input_pixel,
                  jittered_colour,    // 3ch
                  warped_history,     // 3ch
                  disocclusion_mask,  // 1ch
                  luma_derivative.x,  // 1ch
                  temporal_feedback   // 4ch
    );                                // total: 12ch

    // Consumed by post process and frame t+1
    WriteNearestDepthOffset(input_pixel, enc_depth_offset);

    // Consumed at frame t+1
    WriteLumaDerivative(input_pixel, luma_derivative);
}

#endif  // GPU_NSS_PREPROCESS_H