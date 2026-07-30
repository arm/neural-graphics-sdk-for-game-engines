// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
#version 460
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_GOOGLE_include_directive : enable

//------------------------------------------------------------------------------

#if FFX_ARM_FRAMEINTERPOLATION_OPTION_ENABLE_DATA_GRAPH_FI
// Shouldn't access this function.
void main()
{
    return;
}
#else

//-----------------------------------------------------------------------------
// Input: SRV bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_IN_TENSOR           0

//-----------------------------------------------------------------------------
// Output: UAV bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_OUT_PARAMS_TENSOR   1

#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION   2

#include "frameinterpolation/ffx_frameinterpolation_callbacks_glsl.h"
#include "frameinterpolation/ffx_frameinterpolation_common.h"

// Reverse of WriteToInputTensor()
// Input:  tensor : InputTensorElement_t
// Output: the original fields used to write it

void DecodeFromInputTensor(
    in  InputTensorElement_t tensor,
    out FFX_MIN16_F3 rgb_mv_m1,
    out FFX_MIN16_F3 rgb_mv_p1,
    out FFX_MIN16_F3 rgb_mv2_m1,
    out FFX_MIN16_F3 rgb_of_m1,
    out FFX_MIN16_F3 rgb_of_p1,
    out FFX_MIN16_F2 dis_mask,
    out FFX_MIN16_F  qDepth   // what WriteToInputTensor stored from `depth`
)
{
    rgb_mv_m1 = tensor.rgbM1MV_rP1MV.rgb;

    rgb_mv_p1 = FFX_MIN16_F3(
        tensor.rgbM1MV_rP1MV.a,
        tensor.gbP1MV_rgM1OF.r,
        tensor.gbP1MV_rgM1OF.g
    );

    rgb_of_m1 = FFX_MIN16_F3(
        tensor.gbP1MV_rgM1OF.b,
        tensor.gbP1MV_rgM1OF.a,
        tensor.bM1OF_rgbP1OF.r
    );

    rgb_mv2_m1 = FFX_MIN16_F3(0); //TODO: how to set this value

    rgb_of_p1 = tensor.bM1OF_rgbP1OF.gba;

    qDepth = FFX_MIN16_F(tensor.depth_disMask.r); //TODO: check this value

    dis_mask = FFX_MIN16_F2(tensor.depth_disMask.ba);
}

FFX_MIN16_F CalcDistance(FFX_MIN16_F3 a, FFX_MIN16_F3 b)
{
    return abs(a.x-b.x)+abs(a.y-b.y)+abs(a.z-b.z);
}


#ifndef FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS
#define FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS layout (local_size_x = FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH, local_size_y = FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT, local_size_z = FFX_FRAMEINTERPOLATION_THREAD_GROUP_DEPTH) in;
#endif // #ifndef FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS

FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS
void main()
{
    FfxInt32x2 output_pixel = FfxInt32x2(gl_GlobalInvocationID.xy);
    FfxInt32x2 renderSize = RenderSize();
    if (any(greaterThanEqual(output_pixel, renderSize)))
    {
        return;
    }

    InputTensorElement_t tensor;
    tensor.rgbM1MV_rP1MV = LoadInTensorData0(FfxUInt32x2(output_pixel));
    tensor.gbP1MV_rgM1OF = LoadInTensorData1(FfxUInt32x2(output_pixel));
    tensor.bM1OF_rgbP1OF = LoadInTensorData2(FfxUInt32x2(output_pixel));
    tensor.depth_disMask = LoadInTensorData3(FfxUInt32x2(output_pixel));

    FFX_MIN16_F3 rgb_mv_m1;
    FFX_MIN16_F3 rgb_mv_p1;
    FFX_MIN16_F3 rgb_mv2_m1;
    FFX_MIN16_F3 rgb_of_m1;
    FFX_MIN16_F3 rgb_of_p1;
    FFX_MIN16_F2 dis_mask;
    FFX_MIN16_F  qDepth;

    // Decode back into original components
    DecodeFromInputTensor(
        tensor,
        rgb_mv_m1,
        rgb_mv_p1,
        rgb_mv2_m1,
        rgb_of_m1,
        rgb_of_p1,
        dis_mask,
        qDepth
    );

    // Compute distances & consistencies
    FFX_MIN16_F dPrev_MV_OF   = CalcDistance(rgb_mv_m1, rgb_of_m1);
    FFX_MIN16_F dNext_MV_OF   = CalcDistance(rgb_mv_p1, rgb_of_p1);
    FFX_MIN16_F dPrev_MV_MV2  = CalcDistance(rgb_mv_m1, rgb_mv2_m1);
    FFX_MIN16_F dCross        = CalcDistance(FFX_MIN16_F3(rgb_mv_m1+rgb_of_m1) * FFX_MIN16_F(0.5f), FFX_MIN16_F3(rgb_mv_p1+rgb_of_p1) * FFX_MIN16_F(0.5f));

    FFX_MIN16_F cPrev = FFX_MIN16_F(1.0f) / (FFX_MIN16_F(1e-3) + FFX_MIN16_F(0.5) * dPrev_MV_OF + FFX_MIN16_F(0.5) * dPrev_MV_MV2);
    FFX_MIN16_F cNext = FFX_MIN16_F(1.0f) / (FFX_MIN16_F(1e-3) + dNext_MV_OF);

    // Disocclusion / Invalid-Motion Penalty
    FFX_MIN16_F m = clamp(max(dis_mask.x, dis_mask.y), FFX_MIN16_F(0.0f), FFX_MIN16_F(1.0f));
    FFX_MIN16_F trustMask = FFX_MIN16_F(1.0f) - m;

    // Depth-Based Temporal Trust
    const FFX_MIN16_F kNear = FFX_MIN16_F(0.15f);
    const FFX_MIN16_F kFar  = FFX_MIN16_F(0.85f);
    float tDepth = smoothstep(kNear, kFar, clamp(qDepth, FFX_MIN16_F(0.0f), FFX_MIN16_F(1.0f)));

    // Direction Split (Prev vs Next)
    float confPrev = cPrev * trustMask * tDepth;
    float confNext = cNext * trustMask * tDepth;

    float crossBoost = 1.0 / (1e-3 + dCross);
    confPrev *= (1.0 + 0.5 * cPrev * crossBoost);
    confNext *= (1.0 + 0.5 * cNext * crossBoost);

    float dirSum = max(confPrev + confNext, 1e-8);
    float Sprev  = confPrev / dirSum;
    float Snext  = confNext / dirSum;

    // Method Split (MV vs OF)
    // For OF
    FFX_MIN16_F3 prevAvg = FFX_MIN16_F3(rgb_mv_m1 + rgb_of_m1) * FFX_MIN16_F(0.5f);
    float ePrevMV  = 1.0 / (1e-3 + CalcDistance(rgb_mv_m1, prevAvg));
    float ePrevOF  = 1.0 / (1e-3 + CalcDistance(rgb_of_m1, prevAvg));
    float mv2Agree = 1.0 / (1e-3 + dPrev_MV_MV2);
    ePrevMV *= (1.0 + 0.25 * mv2Agree);
    float Pmv = ePrevMV / (ePrevMV + ePrevOF + 1e-8);
    float Pof = ePrevOF / (ePrevMV + ePrevOF + 1e-8);
    // For MVs
    FFX_MIN16_F3 nextAvg = FFX_MIN16_F3(rgb_mv_p1 + rgb_of_p1) * FFX_MIN16_F(0.5f);
    float eNextMV  = 1.0 / (1e-3 + CalcDistance(rgb_mv_p1, nextAvg));
    float eNextOF  = 1.0 / (1e-3 + CalcDistance(rgb_of_p1, nextAvg));
    float eNextSum = max(eNextMV + eNextOF, 1e-8);
    float Nmv  = eNextMV / eNextSum;  // "next MV" share
    float Nof  = eNextOF / eNextSum;  // "next OF" share

    // Final Weights & Safety Rails
    float wPrevMV = Sprev * Pmv;
    float wPrevOF = Sprev * Pof;
    float wNextMV = Snext * Nmv;
    float wNextOF = Snext * Nof;

    // disocclusion hard fail
    if (m > 0.9)
    {
        wPrevMV = wPrevOF = wNextMV = 0.0;
        wNextOF = 1.0;
    }

    float sumW = wPrevMV + wPrevOF + wNextMV + wNextOF;
    // fallback
    if (sumW < 1e-6)
    {
        wPrevMV = 0.0; wPrevOF = 0.5;
        wNextMV = 0.0; wNextOF = 0.5;
        sumW = 1.0;
    }

    // Normalize weights
    float inv = 1.0 / sumW;
    wPrevMV *= inv;
    wPrevOF *= inv;
    wNextMV *= inv;
    wNextOF *= inv;

    FFX_MIN16_F4 weights = FFX_MIN16_F4(wPrevMV, wNextMV, wPrevOF, wNextOF);

    // Store the weights
    StoreOutParamsTensor(output_pixel, weights);
}

#endif

