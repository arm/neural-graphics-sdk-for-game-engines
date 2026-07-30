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
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#ifndef FFX_ARM_FRAMEINTERPOLATION_DEBUG_VIEW_H
#define FFX_ARM_FRAMEINTERPOLATION_DEBUG_VIEW_H

struct DequantizedTensorElement_t
{
    FFX_MIN16_F4 rgbM1MV_rP1MV;
    FFX_MIN16_F4 gbP1MV_rgM1OF;
    FFX_MIN16_F4 bM1OF_rgbP1OF;
    FFX_MIN16_F4 depth_disMask;
};

// Strength of the dynamic-mask overlay tint (0 = invisible, 1 = fully replaces
// the postprocess colour where the mask is saturated).
#define FFX_FRAMEINTERPOLATION_DEBUG_MASK_OVERLAY_STRENGTH FFX_MIN16_F(0.4hf)

// Colour used to visualize the dynamic mask TP1 region.
FFX_MIN16_F3 DynamicMaskOverlayColor()
{
    return FFX_MIN16_F3(0.5hf, 0.5hf, 0.0hf);  // yellow
}

// Load a single dynamic mask TP1 texel (clamped to the mask bounds).
FFX_MIN16_F LoadDynamicMaskTp1(FfxFloat32x2 uv)
{
    FfxInt32x2 iMaskSize = MvDepthLaneSize();
    FfxInt32x2 maskCoord = clamp(FfxInt32x2(uv * FfxFloat32x2(iMaskSize)), FfxInt32x2(0, 0), iMaskSize - FfxInt32x2(1, 1));
    return FFX_MIN16_F(LoadDynamicMaskTp1(maskCoord));
}

void computeDebugView(FfxInt32x2 outputPos)
{
#define VIEWPORT_GRID_SIZE_X 3
#define VIEWPORT_GRID_SIZE_Y 3

    FfxInt32x2 iDisplaySize  = FfxInt32x2(DisplaySize());
    FfxInt32x2 iViewportSize = iDisplaySize / FfxInt32x2(VIEWPORT_GRID_SIZE_X, VIEWPORT_GRID_SIZE_Y);

    // Determine which grid cell this output pixel belongs to
    FfxInt32 gridX = outputPos.x / iViewportSize.x;
    FfxInt32 gridY = outputPos.y / iViewportSize.y;

    // Out of grid (handles non-divisible display sizes)
    if (gridX >= VIEWPORT_GRID_SIZE_X || gridY >= VIEWPORT_GRID_SIZE_Y)
    {
        StoreFrameinterpolationOutput(outputPos, FFX_MIN16_F4(0.0hf, 0.0hf, 0.0hf, 1.0hf));
        return;
    }

    // Position within the viewport
    FfxInt32x2 posInViewport = outputPos - FfxInt32x2(gridX, gridY) * iViewportSize;

    // Map viewport position to tensor coordinate via UV
    FfxFloat32x2 uv          = (FfxFloat32x2(posInViewport) + 0.5f) / FfxFloat32x2(iViewportSize);
    FfxInt32x2   iTensorSize = FfxInt32x2(InputTensorSize());
    FfxInt32x2   tensorCoord = clamp(FfxInt32x2(uv * FfxFloat32x2(iTensorSize)), FfxInt32x2(0, 0), iTensorSize - FfxInt32x2(1, 1));

    // Load and dequantize tensor data at the computed coordinate
    DequantizedTensorElement_t inTensor;
    inTensor.rgbM1MV_rP1MV = Dequantize(LoadInTensor(tensorCoord, 0), FFX_MIN16_F2(1.0hf / QuantParamsSINT().x, QuantParamsSINT().y));
    inTensor.gbP1MV_rgM1OF = Dequantize(LoadInTensor(tensorCoord, 4), FFX_MIN16_F2(1.0hf / QuantParamsSINT().x, QuantParamsSINT().y));
    inTensor.bM1OF_rgbP1OF = Dequantize(LoadInTensor(tensorCoord, 8), FFX_MIN16_F2(1.0hf / QuantParamsSINT().x, QuantParamsSINT().y));
    inTensor.depth_disMask = Dequantize(LoadInTensor(tensorCoord, 12), FFX_MIN16_F2(1.0hf / QuantParamsSINT().x, QuantParamsSINT().y));

    // Select visualization data based on grid cell
    FFX_MIN16_F4 color;
    FfxInt32     cellIndex = gridY * VIEWPORT_GRID_SIZE_X + gridX;
    switch (cellIndex)
    {
    case 0:
        color = FFX_MIN16_F4(inTensor.depth_disMask.x, 0.0hf, 0.0hf, 1.0hf);
        break;  // warped depth tm1
    case 1:
        color = FFX_MIN16_F4(inTensor.depth_disMask.y, 0.0hf, 0.0hf, 1.0hf);
        break;  // warped depth tp1
    case 2:
        color = FFX_MIN16_F4(inTensor.depth_disMask.z, 0.0hf, 0.0hf, 1.0hf);
        break;  // disocclusion mask tm1
    case 3:
        color = FFX_MIN16_F4(inTensor.depth_disMask.w, 0.0hf, 0.0hf, 1.0hf);
        break;  // disocclusion mask tp1
    case 4:
        color = FFX_MIN16_F4(inTensor.rgbM1MV_rP1MV.xyz, 1.0hf);
        break;  // motion warped tm1 color
    case 5:
        color = FFX_MIN16_F4(inTensor.rgbM1MV_rP1MV.w, inTensor.gbP1MV_rgM1OF.xy, 1.0hf);
        break;  // motion warped tp1 color
    case 6:
        color = FFX_MIN16_F4(inTensor.gbP1MV_rgM1OF.zw, inTensor.bM1OF_rgbP1OF.x, 1.0hf);
        break;  // OF warped tm1 color
    case 7:
        color = FFX_MIN16_F4(inTensor.bM1OF_rgbP1OF.yzw, 1.0hf);
        break;  // OF warped tp1 color
    case 8:
    {
        // 9th cell: dynamic mask TP1 blended onto the current-frame (TP1) colour.
        FFX_MIN16_F3 colorTp1 = SampleCurrentBackbuffer(uv);
        FFX_MIN16_F  mask     = LoadDynamicMaskTp1(uv);
        FFX_MIN16_F3 overlay  = DynamicMaskOverlayColor();
        color.rgb             = mix(colorTp1, overlay, FFX_FRAMEINTERPOLATION_DEBUG_MASK_OVERLAY_STRENGTH * mask);
        color.a               = 1.0hf;
        break;  // motion vector magnitude
    }
    default:
        color = FFX_MIN16_F4(0.0hf, 0.0hf, 0.0hf, 1.0hf);
        break;  // unused cell
    }

    StoreFrameinterpolationOutput(outputPos, color);
}

#endif  // FFX_ARM_FRAMEINTERPOLATION_DEBUG_VIEW_H
