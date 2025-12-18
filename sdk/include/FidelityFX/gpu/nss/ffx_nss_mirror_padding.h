/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef GPU_NSS_MIRROR_PADDING_H
#define GPU_NSS_MIRROR_PADDING_H

void ApplyMirrorPadding(int32_t2 input_pixel)
{
    if (any(greaterThanEqual(input_pixel, InputDims())))
    {
        return;
    }

    // calculate input_pixel with mirror padding
    int32_t2 unpadded_input_pixel = input_pixel;
    if (unpadded_input_pixel.x >= UnpaddedInputDims().x)
    {
        unpadded_input_pixel.x = (UnpaddedInputDims().x - 1) - (unpadded_input_pixel.x - UnpaddedInputDims().x);
    }
    if (unpadded_input_pixel.y >= UnpaddedInputDims().y)
    {
        unpadded_input_pixel.y = (UnpaddedInputDims().y - 1) - (unpadded_input_pixel.y - UnpaddedInputDims().y);
    }
    float2 unpadded_uv = (float2(unpadded_input_pixel) + 0.5) / float2(UnpaddedInputDims());

    StorePaddedColor(input_pixel, LoadUnpaddedColor(unpadded_uv));
    StorePaddedDepth(input_pixel, LoadUnpaddedDepth(unpadded_uv));
    StorePaddedMotion(input_pixel, LoadUnpaddedMotion(unpadded_uv));
    StorePaddedDepthTm1(input_pixel, LoadUnpaddedDepthTm1(unpadded_uv));
}

#endif