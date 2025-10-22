// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#ifndef GPU_NSS_DEBUG_VIEW_H
#define GPU_NSS_DEBUG_VIEW_H
struct FfxDebugViewport
{
    uint32_t2 offset;
    uint32_t2 size;
};

// Macro to cull and draw debug viewport
#define DRAW_VIEWPORT(function, pos, vp, data)              \
    {                                                       \
        int32_t2 posInViewport = getPosInViewport(pos, vp); \
        function(posInViewport, data);                      \
    }

int32_t2 getPosInViewport(int32_t2 pos, FfxDebugViewport vp)
{
    // the dispatch is w.r.t. to input sizes
    float2 uv = (float2(pos) + 0.5) * InvInputDims();

    return int32_t2(vp.offset + uv * vp.size);
}

void DebugView(int32_t2 inputPos)
{
    float2   uv        = (float2(inputPos) + 0.5) / InputDims();
    int32_t2 outputPos = int32_t2(uv * OutputDims());

    //Motion
    float    depthDilated       = float(0.f);
    int32_t2 nearestPixelOffset = int32_t2(0);
    FindNearestDepth(inputPos, RenderSize(), depthDilated, nearestPixelOffset);
    half4 motion = half4(LoadMotion(inputPos + nearestPixelOffset), 0.0HF, 1.0HF);

    //preprocess Tensor
    PreprocessTensorElement preprocessTensor         = LoadPreprocessTensor(inputPos);
    half4                   preprocessWarpedHistory  = half4(preprocessTensor.wh_rgb_col_r.rgb, 1.0HF);
    half4                   preprocessJitteredColour = half4(preprocessTensor.wh_rgb_col_r.a, preprocessTensor.col_gb_dm_fback_r.rgb);
    half4                   preprocessFeedback       = half4(preprocessTensor.col_gb_dm_fback_r.a, preprocessTensor.fback_gba_ld.rgb);
    half4                   preprocessDisoLuma       = half4(preprocessTensor.col_gb_dm_fback_r.b, preprocessTensor.fback_gba_ld.a, 0.0HF, 1.0HF);

    //Postprocess raw data
    half4 k0, k1, k2, k3;
    LoadKPNRaw(uv, k0, k1, k2, k3);

#if SCALE_MODE == SCALE_2_0X
    //Postprocess KPNWeight
    int16_t2   tiledIdx   = (int16_t2(outputPos) + LutOffset()) % int16_t2(IndexModulo());
    int16_t    lutIdx     = tiledIdx.y * int16_t(IndexModulo()) + tiledIdx.x;
    KernelTile lut        = kernelLUT[lutIdx];
    half4      kpnWeights = clamp(LoadKPNWeight(uv, lutIdx), half4(EPS), half4(1.HF));
#else
    half4 kpnWeights = half4(0.0HF);  //No vec4 kpn weights for this method.
#endif
    //Temporal parameters
    half theta, alpha;
    LoadTemporalParameters(uv, theta, alpha);
    half4 tempParam = half4(theta, alpha, 0.0HF, 1.0HF);

    //Output
    half4 upscaledOutput = LoadUpscaledOutput(outputPos);

#define VIEWPORT_GRID_SIZE_X 4
#define VIEWPORT_GRID_SIZE_Y 3

    float2 fViewportScale = float2(1.0f / VIEWPORT_GRID_SIZE_X, 1.0f / VIEWPORT_GRID_SIZE_Y);

    uint32_t2 iViewportSize = uint32_t2(GetDebugViewDimensions() * fViewportScale);

    // compute grid [y][x] for easier placement of viewports
    FfxDebugViewport vp[VIEWPORT_GRID_SIZE_Y][VIEWPORT_GRID_SIZE_X];
    for (int32_t y = 0; y < VIEWPORT_GRID_SIZE_Y; y++)
    {
        for (int32_t x = 0; x < VIEWPORT_GRID_SIZE_X; x++)
        {
            vp[y][x].offset = iViewportSize * uint32_t2(x, y);
            vp[y][x].size   = iViewportSize;
        }
    }

    DRAW_VIEWPORT(StoreDebugView, inputPos, vp[0][0], preprocessWarpedHistory);
    DRAW_VIEWPORT(StoreDebugView, inputPos, vp[0][1], preprocessJitteredColour);
    DRAW_VIEWPORT(StoreDebugView, inputPos, vp[0][2], preprocessFeedback);
    DRAW_VIEWPORT(StoreDebugView, inputPos, vp[0][3], preprocessDisoLuma);

    DRAW_VIEWPORT(StoreDebugView, inputPos, vp[1][0], k0);
    DRAW_VIEWPORT(StoreDebugView, inputPos, vp[1][1], k1);
    DRAW_VIEWPORT(StoreDebugView, inputPos, vp[1][2], k2);
    DRAW_VIEWPORT(StoreDebugView, inputPos, vp[1][3], k3);

    DRAW_VIEWPORT(StoreDebugView, inputPos, vp[2][0], motion);
    DRAW_VIEWPORT(StoreDebugView, inputPos, vp[2][1], kpnWeights);
    DRAW_VIEWPORT(StoreDebugView, inputPos, vp[2][2], tempParam);
    DRAW_VIEWPORT(StoreDebugView, inputPos, vp[2][3], upscaledOutput);
}
#endif  // GPU_NSS_DEBUG_VIEW_H
