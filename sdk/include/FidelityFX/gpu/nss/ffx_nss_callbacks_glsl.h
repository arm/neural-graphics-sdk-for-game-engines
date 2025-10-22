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

#ifndef GPU_NSS_FFX_NSS_CALLBACKS_GLSL_H
#define GPU_NSS_FFX_NSS_CALLBACKS_GLSL_H

#include "ffx_nss_resources.h"

#if defined(FFX_GPU)

// Scale modes support fast LUT path
// Need to define before including ffx_nss_common_glsl.h
#define SCALE_1_3X 1
#define SCALE_1_5X 2
#define SCALE_2_0X 3

#if SCALE_PRESET_MODE == SCALE_1_3X || SCALE_PRESET_MODE == SCALE_1_5X || SCALE_PRESET_MODE == SCALE_2_0X
#define SCALE_MODE SCALE_PRESET_MODE
#else
#define SCALE_MODE 0  //No LUT method.
#endif

#include "ffx_core.h"
#include "ffx_nss_common_glsl.h"

#if QUANTIZED
#define tensor_t    int8_t
#define tensorVec_t int8_t4
#else
#define tensor_t    float
#define tensorVec_t float4
#endif

#if REVERSE_Z
#define INVERTED_DEPTH 1
#endif

#ifdef INVERTED_DEPTH
#define MAX_DEPTH 0.f
#else
#define MAX_DEPTH 1.f
#endif

// Preprocess tensor element
struct PreprocessTensorElement
{
    FfxFloat32x4 wh_rgb_col_r;       // warped_history.rgb, jittered_colour.r
    FfxFloat32x4 col_gb_dm_fback_r;  // jittered_colour.gb, disocclusion mask, feedback.r
    FfxFloat32x4 fback_gba_ld;       // feedback.gba, luma derivative
};

// Preprocess tensor representation which considers quantization
struct PreprocessTensorElementInternal
{
    tensorVec_t wh_rgb_col_r;       // warped_history.rgb, jittered_colour.r
    tensorVec_t col_gb_dm_fback_r;  // jittered_colour.gb, disocclusion mask, feedback.r
    tensorVec_t fback_gba_ld;       // feedback.gba, luma derivative
};

struct TensorElement
{
    int8_t4 wh_rgb_col_r;       // warped_history.rgb, jittered_colour.r
    int8_t4 col_gb_dm_fback_r;  // jittered_colour.gb, disocclusion mask, feedback.r
    int8_t4 fback_gba_ld;       // feedback.gba, luma derivative
};

struct BilinearSamplingData
{
    int32_t2 iOffsets[4];
    float    fWeights[4];
    int32_t2 iBasePos;
    float2   fQuadCenterUv;
};

BilinearSamplingData GetBilinearSamplingData(float2 fUv, int32_t2 iSize)
{
    /*
        Closely based on:
        https://github.com/arm/accuracy-super-resolution-generic-library/blob/
        38697a58a6e7818ec9d28774bc073f537abb9178/include/gpu/fsr2/ffxm_fsr2_common.h#L548
    */

    BilinearSamplingData data;

    float2 fPxSample   = (fUv * iSize) - float2(0.5f, 0.5f);
    data.iBasePos      = int32_t2(floor(fPxSample));
    data.fQuadCenterUv = (fPxSample + 0.5f) / float2(iSize);
    float2 fPxFrac     = fract(fPxSample);

    data.iOffsets[0] = int32_t2(0, 0);
    data.iOffsets[2] = int32_t2(1, 0);
    data.iOffsets[1] = int32_t2(0, 1);
    data.iOffsets[3] = int32_t2(1, 1);

    data.fWeights[0] = (1.f - fPxFrac.x) * (1.f - fPxFrac.y);
    data.fWeights[1] = (fPxFrac.x) * (1.f - fPxFrac.y);
    data.fWeights[2] = (1.f - fPxFrac.x) * (fPxFrac.y);
    data.fWeights[3] = (fPxFrac.x) * (fPxFrac.y);

    return data;
}

///////////////////////////////////////////////
// declare CBs and CB accessors
///////////////////////////////////////////////
#if defined(NSS_BIND_CB_NSS)
layout(set = 0, binding = NSS_BIND_CB_NSS, std140) uniform cbNSS_t
{
    // ─────────────── 32bit precision objects ───────────────
    float4   _DeviceToViewDepth;  //  16 B
    float4   _JitterOffset;       //  16 B (.xy = pixels, .zw = uvs)
    float4   _JitterOffsetTm1;    //  16 B (.xy = pixels, .zw = uvs)
    float4   _ScaleFactor;        //  16 B (.xy = scale, .zw = inv scale)
    int32_t2 _OutputDims;         //   8 B
    int32_t2 _InputDims;          //   8 B
    float2   _InvOutputDims;      //   8 B
    float2   _InvInputDims;       //   8 B
    float2   _MotionVectorScale;  //   8 B
    float2   _Padding0;           //  8 B (padding to 16 byte boundary)

    // ───────────────  16bit precision objects  ────────────────
    half4    _QuantParamsSNORM;    //   8 B  (.xy for quantize, .zw for dequantize)
    half4    _QuantParamsSINT;     //   8 B  (.xy for quantize, .zw for dequantize)
    half4    _MotionDisThreshPad;  //   8 B  (.xyzw = motion/disocclusion thresholds)
    half2    _Exposure;            //   4 B  (.x = exposure, .y = 1/exp)
    int16_t2 _IndexModulo;         //   4 B
    int16_t2 _LutOffset;           //   4 B
    half     _NotHistoryReset;     //   2 B
}
cbNSS;

//--------------------------------------------------------------------------------
// Accessors
//--------------------------------------------------------------------------------

float4 DeviceToViewDepth()
{
    return cbNSS._DeviceToViewDepth;
}

float2 JitterOffsetPixel()
{
    return cbNSS._JitterOffset.xy;
}

float2 JitterOffsetUv()
{
    return cbNSS._JitterOffset.zw;
}

float2 JitterOffsetTm1Pixel()
{
    return cbNSS._JitterOffsetTm1.xy;
}

float2 JitterOffsetTm1Uv()
{
    return cbNSS._JitterOffsetTm1.zw;
}

float4 ScaleFactor()
{
    return cbNSS._ScaleFactor;
}

int32_t2 OutputDims()
{
    return cbNSS._OutputDims;
}

int32_t2 InputDims()
{
    return cbNSS._InputDims;
}

int32_t2 RenderSize()
{
    return InputDims();
}

float2 InvOutputDims()
{
    return cbNSS._InvOutputDims;
}

float2 InvInputDims()
{
    return cbNSS._InvInputDims;
}

float2 MotionVectorScale()
{
    return cbNSS._MotionVectorScale.xy;
}

half4 QuantParamsSNORM()
{
    return cbNSS._QuantParamsSNORM;
}

half4 QuantParamsSINT()
{
    return cbNSS._QuantParamsSINT;
}

half2 InputQuantParams()
{
    return cbNSS._QuantParamsSINT.xy;
}

#if ALIAS_OUTPUT_TENSORS_AS_IMAGES
half2 FeedbackQuantParams()
{
    return cbNSS._QuantParamsSNORM.zw;
}
#else
half2 FeedbackQuantParams()
{
    return cbNSS._QuantParamsSINT.zw;
}
#endif

half Exposure()
{
    return cbNSS._Exposure.x;
}

half InvExposure()
{
    return cbNSS._Exposure.y;
}

half MotionWarpThresh()
{
    return cbNSS._MotionDisThreshPad.x;
}

half MotionThresh()
{
    return MotionWarpThresh();
}

half MotionDisThresh()
{
    return cbNSS._MotionDisThreshPad.y;
}

half DisocclusionScale()
{
    return cbNSS._MotionDisThreshPad.z;
}

float2 InvScale()
{
    return cbNSS._ScaleFactor.zw;
}

int16_t2 IndexModulo()
{
    return cbNSS._IndexModulo;
}

#if ALIAS_OUTPUT_TENSORS_AS_IMAGES
half2 K0QuantParams()
{
    return cbNSS._QuantParamsSNORM.zw;
}

half2 K1QuantParams()
{
    return cbNSS._QuantParamsSNORM.zw;
}

half2 K2QuantParams()
{
    return cbNSS._QuantParamsSNORM.zw;
}

half2 K3QuantParams()
{
    return cbNSS._QuantParamsSNORM.zw;
}

half2 TemporalQuantParams()
{
    return cbNSS._QuantParamsSNORM.zw;
}
#else
half2 K0QuantParams()
{
    return cbNSS._QuantParamsSINT.zw;
}

half2 K1QuantParams()
{
    return cbNSS._QuantParamsSINT.zw;
}

half2 K2QuantParams()
{
    return cbNSS._QuantParamsSINT.zw;
}

half2 K3QuantParams()
{
    return cbNSS._QuantParamsSINT.zw;
}

half2 TemporalQuantParams()
{
    return cbNSS._QuantParamsSINT.zw;
}
#endif

int16_t2 LutOffset()
{
    return cbNSS._LutOffset;
}

half NotHistoryReset()
{
    return cbNSS._NotHistoryReset;
}

float GetViewSpaceDepth(float fDeviceDepth)
{
    /*
        Closely based on:
        https://github.com/arm/accuracy-super-resolution-generic-library/blob/
        38697a58a6e7818ec9d28774bc073f537abb9178/include/gpu/fsr2/ffxm_fsr2_common.h#L462

        `fDeviceToViewDepth` / `_DeviceToViewDepth` details found in:
        https://github.com/GPUOpen-Effects/FidelityFX-FSR2/blob/
        1680d1edd5c034f88ebbbb793d8b88f8842cf804/src/ffx-fsr2-api/ffx_fsr2.cpp#L663
    */

    const float4 fDeviceToViewDepth = DeviceToViewDepth();

    return (fDeviceToViewDepth[1] / (fDeviceDepth - fDeviceToViewDepth[0]));
}

#endif  // #if defined(NSS_BIND_CB_NSS)

layout(set = 0, binding = 1000) uniform sampler s_PointClamp;
layout(set = 0, binding = 1001) uniform sampler s_LinearClamp;

//=========================================================================
// Helper functions for reading tensors
//=========================================================================
#if QUANTIZED
#define DECLARE_SAMPLE_TENSOR(TENSORNAME, tensor_variable)                                                 \
    half4 Load##TENSORNAME##Quad(int32_t2 coord, half2 quant_params)                                       \
    {                                                                                                      \
        tensor_t a[4];                                                                                     \
        tensorReadARM(tensor_variable, uint[](0, coord.y, coord.x, 0), a);                                 \
        tensorVec_t v = tensorVec_t(a[0], a[1], a[2], a[3]);                                               \
        return Dequantize(v, quant_params);                                                                \
    }                                                                                                      \
    half4 Sample##TENSORNAME##Tensor(float2 uv, half2 quant_params)                                        \
    {                                                                                                      \
        uint32_t2 dims  = uint32_t2(tensorSizeARM(tensor_variable, 2), tensorSizeARM(tensor_variable, 1)); \
        float2    coord = uv * float2(dims) - 0.5;                                                         \
        uint32_t2 f     = min(dims - 1U, uint32_t2(max(float2(0.0), floor(coord))));                       \
        uint32_t2 c     = min(dims - 1U, uint32_t2(max(float2(0.0), ceil(coord))));                        \
        float2    frac  = fract(coord);                                                                    \
        float4    c00   = Load##TENSORNAME##Quad(int32_t2(f), quant_params);                               \
        float4    c01   = Load##TENSORNAME##Quad(int32_t2(f.x, c.y), quant_params);                        \
        float4    c10   = Load##TENSORNAME##Quad(int32_t2(c.x, f.y), quant_params);                        \
        float4    c11   = Load##TENSORNAME##Quad(int32_t2(c), quant_params);                               \
        float4    c0    = mix(c00, c01, frac.y);                                                           \
        float4    c1    = mix(c10, c11, frac.y);                                                           \
        return half4(mix(c0, c1, frac.x));                                                                 \
    }
#else
#define DECLARE_SAMPLE_TENSOR(TENSORNAME, tensor_variable)                                                 \
    half4 Load##TENSORNAME##Quad(int32_t2 coord, half2 quant_params)                                       \
    {                                                                                                      \
        tensor_t a[4];                                                                                     \
        tensorReadARM(tensor_variable, uint[](0, coord.y, coord.x, 0), a);                                 \
        tensorVec_t v = tensorVec_t(a[0], a[1], a[2], a[3]);                                               \
        return v;                                                                                          \
    }                                                                                                      \
    half4 Sample##TENSORNAME##Tensor(float2 uv, half2 quant_params)                                        \
    {                                                                                                      \
        uint32_t2 dims  = uint32_t2(tensorSizeARM(tensor_variable, 2), tensorSizeARM(tensor_variable, 1)); \
        float2    coord = uv * float2(dims) - 0.5;                                                         \
        uint32_t2 f     = min(dims - 1U, uint32_t2(max(float2(0.0), floor(coord))));                       \
        uint32_t2 c     = min(dims - 1U, uint32_t2(max(float2(0.0), ceil(coord))));                        \
        float2    frac  = fract(coord);                                                                    \
        float4    c00   = Load##TENSORNAME##Quad(int32_t2(f), quant_params);                               \
        float4    c01   = Load##TENSORNAME##Quad(int32_t2(f.x, c.y), quant_params);                        \
        float4    c10   = Load##TENSORNAME##Quad(int32_t2(c.x, f.y), quant_params);                        \
        float4    c11   = Load##TENSORNAME##Quad(int32_t2(c), quant_params);                               \
        float4    c0    = mix(c00, c01, frac.y);                                                           \
        float4    c1    = mix(c10, c11, frac.y);                                                           \
        return half4(mix(c0, c1, frac.x));                                                                 \
    }
#endif

//=========================================================================
// Common Resources for both pre-process and post-process
//=========================================================================

//-------------------------------------------------------------------------
// Input: color‐jittered
//-------------------------------------------------------------------------
#if defined(NSS_BIND_SRV_INPUT_COLOR_JITTERED)
layout(set = 0, binding = NSS_BIND_SRV_INPUT_COLOR_JITTERED) uniform mediump texture2D r_input_color_jittered;
#define _ColourTex sampler2D(r_input_color_jittered, s_LinearClamp)

FfxFloat32x4 LoadInputColorJittered(int32_t2 iPxPos)
{
    return texelFetch(r_input_color_jittered, iPxPos, 0);
}

FfxFloat32x4 SampleInputColorJittered(FfxFloat32x2 fUV)
{
    return textureLod(sampler2D(r_input_color_jittered, s_LinearClamp), fUV, 0);
}

half3 LoadColour(int32_t2 pixel)
{
    return Tonemap(SafeColour(half3(texelFetch(_ColourTex, pixel, 0).rgb) * Exposure()));
}

#endif

//-------------------------------------------------------------------------
// Input: motion‐vectors
//-------------------------------------------------------------------------
#if defined(NSS_BIND_SRV_INPUT_MOTION_VECTORS)
layout(set = 0, binding = NSS_BIND_SRV_INPUT_MOTION_VECTORS) uniform mediump texture2D r_input_motion_vectors;
#define _InputMotionTex sampler2D(r_input_motion_vectors, s_LinearClamp)

half2 LoadMotion(int32_t2 pixel)
{
    return half2(texelFetch(_InputMotionTex, pixel, 0).rg * MotionVectorScale());
}

#endif  // #if defined(NSS_BIND_SRV_INPUT_MOTION_VECTORS)

//-------------------------------------------------------------------------
// Input: Prev‐upscaled‐color
//-------------------------------------------------------------------------
#if defined(NSS_BIND_SRV_HISTORY_UPSCALED_COLOR)
layout(set = 0, binding = NSS_BIND_SRV_HISTORY_UPSCALED_COLOR) uniform mediump texture2D r_prev_upscaled_color;
#define _HistoryTex sampler2D(r_prev_upscaled_color, s_LinearClamp)

half3 LoadHistory(float2 uv)
{
    return half3(textureLod(_HistoryTex, uv, 0).rgb);
}

half3 WarpHistory(float2 uv)
{
    return Tonemap(SafeColour(half3(textureLod(_HistoryTex, uv, 0).rgb) * Exposure()));
}

half3 LoadHistoryCatmull(float2 uv)
{
    //------------------------------------------------------------------------------------
    // 1) Compute Catmull–Rom weights
    //------------------------------------------------------------------------------------
    float2 scaledUV  = uv * OutputDims();
    float2 baseFloor = floor(scaledUV - 0.5) + 0.5;

    half2 f  = half2(scaledUV - baseFloor);
    half2 f2 = f * f;
    half2 f3 = f2 * f;

    // Catmull–Rom basis
    half2 w0 = f2 - 0.5HF * (f3 + f);
    half2 w1 = 1.5HF * f3 - 2.5HF * f2 + 1.0HF;
    half2 w3 = 0.5HF * (f3 - f2);
    half2 w2 = (1.0HF - w0) - w1 - w3;  // = 1 - (w0 + w1 + w3)

    // Combine w1 and w2 for center axis
    half2 w12 = w1 + w2;
    half  wx0 = w0.x, wy0 = w0.y;
    half  wx1 = w12.x, wy1 = w12.y;
    half  wx2 = w3.x, wy2 = w3.y;

    // Final weights for the cross sample layout
    half wUp     = wx1 * wy0;  // center in X, up in Y
    half wDown   = wx1 * wy2;  // center in X, down in Y
    half wLeft   = wx0 * wy1;  // left   in X, center in Y
    half wRight  = wx2 * wy1;  // right  in X, center in Y
    half wCenter = wx1 * wy1;  // center in X, center in Y

    // Fractional offsets for the center
    half dx = w2.x / wx1;
    half dy = w2.y / wy1;

    //------------------------------------------------------------------------------------
    // 2) Gather the 5 taps
    //------------------------------------------------------------------------------------
    half4 left   = half4(LoadHistory((baseFloor + float2(-1.0, dy)) * InvOutputDims()), 1.HF);
    half4 up     = half4(LoadHistory((baseFloor + float2(dx, -1.0)) * InvOutputDims()), 1.HF);
    half4 center = half4(LoadHistory((baseFloor + float2(dx, dy)) * InvOutputDims()), 1.HF);
    half4 right  = half4(LoadHistory((baseFloor + float2(2.0, dy)) * InvOutputDims()), 1.HF);
    half4 down   = half4(LoadHistory((baseFloor + float2(dx, 2.0)) * InvOutputDims()), 1.HF);

    //------------------------------------------------------------------------------------
    // 3) Accumulate and track min/max
    //------------------------------------------------------------------------------------
    half4 accum = up * wUp + left * wLeft + center * wCenter + right * wRight + down * wDown;
    half3 cmin3 = min(up.rgb, min(left.rgb, min(center.rgb, min(right.rgb, down.rgb))));
    half3 cmax3 = max(up.rgb, max(left.rgb, max(center.rgb, max(right.rgb, down.rgb))));

    //------------------------------------------------------------------------------------
    // 4) Final color
    //------------------------------------------------------------------------------------
    half3 color = accum.rgb * rcp(accum.w);

    // dering in the case where we have negative values, we don't do this all the time
    // as it can impose unnecessary blurring on the output
    return any(lessThan(color, half3(0.HF))) ? clamp(color, cmin3, cmax3) : color;
}

FfxFloat32x3 SampleHistory(FfxFloat32x2 fUV)
{
#if RESAMPLE_BICUBIC
    FfxFloat32x3 colour = FfxFloat32x3(0.f);

    CatmullRomSamples samples = GetBicubic2DCatmullRomSamples(fUV, FfxFloat32x2(OutputDims().x, OutputDims().y), InvOutputDims());
    for (FfxUInt32 i = 0U; i < samples.Count; i++)
    {
        FfxFloat32x2 sampleUV = samples.UV[i];
        colour += textureLod(sampler2D(r_prev_upscaled_color, s_LinearClamp), sampleUV, 0).rgb * samples.Weight[i];
    }
    // defend against negative values
    return max(FfxFloat32x3(0.f), colour * samples.FinalMultiplier);

#else   // bilinear
    return textureLod(sampler2D(r_prev_upscaled_color, s_LinearClamp), fUV, 0.0).rgb;
#endif  // RESAMPLE_BICUBIC
}
#endif  // #if defined(NSS_BIND_SRV_HISTORY_UPSCALED_COLOR)

//=========================================================================
// Resources for pre-process
//=========================================================================

//-------------------------------------------------------------------------
// Input: DepthOffset tm1 (aka NearestDepthCoordTm1Tex)
//-------------------------------------------------------------------------
#if defined(NSS_BIND_SRV_NEAREST_DEPTH_COORD_TM1)
layout(set = 0, binding = NSS_BIND_SRV_NEAREST_DEPTH_COORD_TM1) uniform highp texture2D r_input_nearest_depth_coord_tm1;
#define _NearestDepthCoordTm1Tex sampler2D(r_input_nearest_depth_coord_tm1, s_LinearClamp)

int32_t2 LoadDepthNearestDepthOffsetTm1(int32_t2 pixel)
{
    //FIXME: `* 255` only needed because scenario runner doesn't support R8_SINT/R8_UINT
    half    encNorm = half(texelFetch(_NearestDepthCoordTm1Tex, pixel, 0).r);
    int32_t code    = int32_t(encNorm * 255.0 + 0.5);

    // 3. map back to {-1,0,1}²
    return DecodeNearestDepthCoord(code);
}

#endif

//-------------------------------------------------------------------------
// Input: Prev‐feedback‐tensor (SRV Buffer)
// We want to bilinearly sample this tensor:
//  1. when it is bound as an image, we can use a sampler.
//  2. when it is bound as a tensorARM or buffer, we need to manually load the quad positions and lerp.
// Additionally, the model can be quantized, which results in 2 variants per permutation, resulting in 6 permutations.
//-------------------------------------------------------------------------
#if defined(NSS_BIND_SRV_FEEDBACK_TM1_TENSOR)
#if ALIAS_OUTPUT_TENSORS_AS_IMAGES
layout(set = 0, binding = NSS_BIND_SRV_FEEDBACK_TM1_TENSOR) uniform lowp texture2D r_prev_feedback_tensor;
#define _FeedbackTensor sampler2D(r_prev_feedback_tensor, s_LinearClamp)

half4 WarpFeedback(float2 uv)
{
    return Dequantize(half4(textureLod(_FeedbackTensor, uv, 0)), FeedbackQuantParams()) * NotHistoryReset();
}

#else
// --- bind as GL ARM tensor, manual load via tensorReadARM ---
layout(set = 0, binding = NSS_BIND_SRV_FEEDBACK_TM1_TENSOR) uniform readonly tensorARM<tensor_t, 4> r_prev_feedback_tensor;
DECLARE_SAMPLE_TENSOR(Feedback, r_prev_feedback_tensor)

half4 WarpFeedback(float2 uv)
{
    return SampleFeedbackTensor(uv, FeedbackQuantParams()) * NotHistoryReset();
}

#endif
#endif

//-------------------------------------------------------------------------
// Input: Prev‐luma‐deriv
//-------------------------------------------------------------------------
#if defined(NSS_BIND_SRV_LUMA_DERIV_TM1)
layout(set = 0, binding = NSS_BIND_SRV_LUMA_DERIV_TM1) uniform lowp texture2D r_prev_luma_deriv;
#define _LumaDerivTm1Tex sampler2D(r_prev_luma_deriv, s_LinearClamp)

half2 WarpLumaDerivative(float2 uv)
{
    return half2(textureLod(_LumaDerivTm1Tex, uv, 0).rg);
}

half2 CalculateLumaDerivative(float2 reproj_uv, half3 jittered_colour, half disocclusion_mask)
{
    const half DIS_THRESH      = 0.01HF;
    const half DERIV_MIN       = 0.05HF;
    const half DERIV_MAX       = 0.3HF;
    const half DERIV_POW       = 1.5HF;
    const half DERIV_ALPHA     = 0.1HF;
    const half DERIV_MAX_R     = rcp(DERIV_MAX);
    const half DERIV_MAX_POW_R = rcp(pow(DERIV_MAX, DERIV_POW));

    //--------------------------------------------------------------------
    // 1.  Fetch history (luma + derivative)
    //--------------------------------------------------------------------
    half2 h              = WarpLumaDerivative(reproj_uv);
    half  luma_tm1       = h.y;
    half  derivative_tm1 = h.x;

    //--------------------------------------------------------------------
    // 2.  Current luma & raw derivative
    //--------------------------------------------------------------------
    half luma_t       = Luminance(jittered_colour);
    half derivative_t = abs(luma_t - luma_tm1);

    //--------------------------------------------------------------------
    // 3.  Soft-clip & normalize
    //--------------------------------------------------------------------
    // Clip to `DERIV_MAX` which is ~typical max value,
    // allows for better precision allocation when normalized
    half clipped = min(derivative_t, DERIV_MAX);

    // Discard values less than `DERIV_MIN` to reduce ghosting
    clipped *= step(DERIV_MIN, derivative_t);

    // Normalize with soft-clip
    // x^1.5  =  x * sqrt(x) | NOTE: only works because `DERIV_POW=1.5`
    half curved = clipped * sqrt(clipped) * DERIV_MAX_POW_R;

    //--------------------------------------------------------------------
    // 4.  Temporal accumulation
    //--------------------------------------------------------------------
    // Accumulate the new derivative into the history.
    // We apply an adaptive alpha scaling, to ensure that if a derivative converges to a high value
    // it becomes more difficult to reset that value, this provides temporally stable convergence
    half alpha_scale = mix(DERIV_ALPHA, DERIV_ALPHA * 0.1HF, clamp(derivative_tm1, half(0.HF), DERIV_MAX) * DERIV_MAX_R);

    half derivative = mix(derivative_tm1, curved, alpha_scale);

    //--------------------------------------------------------------------
    // 5.  Remove disoccluded pixels
    //--------------------------------------------------------------------
    derivative *= step(disocclusion_mask, DIS_THRESH);

    // .x -> derivative for current frame, .y -> luma of current frame
    return half2(derivative, luma_t);
}
#endif  // #if defined(NSS_BIND_SRV_LUMA_DERIV_TM1)

//-------------------------------------------------------------------------
// Input: Prev‐depth
//-------------------------------------------------------------------------
#if defined(NSS_BIND_SRV_INPUT_DEPTH_TM1)
layout(set = 0, binding = NSS_BIND_SRV_INPUT_DEPTH_TM1) uniform highp texture2D r_prev_depth;
#define _DepthTm1Tex sampler2D(r_prev_depth, s_LinearClamp)

FfxFloat32 LoadPrevDepth(int32_t2 iPxPos)
{
    return texelFetch(r_prev_depth, iPxPos, 0).r;
}

FfxFloat32 SamplePrevDepth(FfxFloat32x2 fUV)
{
    return textureLod(sampler2D(r_prev_depth, s_LinearClamp), fUV, 0.0).r;
}

// declaration
// int32_t2 LoadDepthNearestDepthOffsetTm1(int32_t2 pixel);

void GatherReconstructedPreviousDepthRQuad(float2 fUV, inout float4 depthQuad)
{
    int32_t2 offset    = LoadDepthNearestDepthOffsetTm1(int32_t2(fUV * InputDims()));
    float2   offset_uv = float2(offset) * InvInputDims();
    depthQuad          = textureGather(_DepthTm1Tex, fUV + offset_uv, 0).wzxy;
}

FfxFloat32x2 ComputeNdc(FfxFloat32x2 fPxPos, int32_t2 iSize)
{
    /*
        Closely based on:
        https://github.com/arm/accuracy-super-resolution-generic-library/blob/
        38697a58a6e7818ec9d28774bc073f537abb9178/include/gpu/fsr2/ffxm_fsr2_common.h#L457
    */

    return fPxPos.yx / FfxFloat32x2(iSize.yx) * FfxFloat32x2(2.0f, -2.0f) + FfxFloat32x2(-1.0f, 1.0f);
}

float3 GetViewSpacePosition(int32_t2 iViewportPos, int32_t2 iViewportSize, float fDeviceDepth)
{
    /*
        Closely based on:
        https://github.com/arm/accuracy-super-resolution-generic-library/blob/
        38697a58a6e7818ec9d28774bc073f537abb9178/include/gpu/fsr2/ffxm_fsr2_common.h#L475
    */

    const float4 fDeviceToViewDepth = DeviceToViewDepth();

    const float Z = GetViewSpaceDepth(fDeviceDepth);

    const float2 fNdcPos = ComputeNdc(iViewportPos, iViewportSize);
    const float  X       = fDeviceToViewDepth[2] * fNdcPos.x * Z;
    const float  Y       = fDeviceToViewDepth[3] * fNdcPos.y * Z;

    return float3(X, Y, Z);
}

float ComputeDepthClip(float2 fUvSample, float fCurrentDepthSample)
{
    /*
        Closely based on:
        https://github.com/arm/accuracy-super-resolution-generic-library/blob/
        38697a58a6e7818ec9d28774bc073f537abb9178/include/gpu/fsr2/ffxm_fsr2_depth_clip.h#L36
    */

    const float          fReconstructedDepthBilinearWeightThreshold = 0.1f;
    float                fCurrentDepthViewSpace                     = GetViewSpaceDepth(fCurrentDepthSample);
    BilinearSamplingData bilinearInfo                               = GetBilinearSamplingData(fUvSample, RenderSize());

    float fDepth     = 0.0f;
    float fWeightSum = 0.0f;

    float4 fPrevDepthSamples;
    GatherReconstructedPreviousDepthRQuad(bilinearInfo.fQuadCenterUv, fPrevDepthSamples);

    for (int32_t iSampleIndex = 0; iSampleIndex < 4; iSampleIndex++)
    {
        const int32_t2 iOffset    = bilinearInfo.iOffsets[iSampleIndex];
        const int32_t2 iSamplePos = bilinearInfo.iBasePos + iOffset;

        const float fWeight  = bilinearInfo.fWeights[iSampleIndex];
        const bool  onscreen = IsOnScreen(iSamplePos, RenderSize());
        fWeightSum += onscreen ? 0.f : fWeight;
        if (onscreen)
        {
            if (fWeight > fReconstructedDepthBilinearWeightThreshold)
            {
                const float fPrevDepthSample           = fPrevDepthSamples[iSampleIndex];
                const float fPrevNearestDepthViewSpace = GetViewSpaceDepth(fPrevDepthSample);
                const float fDepthDiff                 = fCurrentDepthViewSpace - fPrevNearestDepthViewSpace;

                if (fDepthDiff > 0.0f)
                {
#ifdef INVERTED_DEPTH
                    const float fPlaneDepth = min(fPrevDepthSample, fCurrentDepthSample);
#else
                    const float fPlaneDepth = max(fPrevDepthSample, fCurrentDepthSample);
#endif

                    const float3 fCenter = GetViewSpacePosition(int32_t2(RenderSize() * 0.5f), RenderSize(), fPlaneDepth);
                    const float3 fCorner = GetViewSpacePosition(int32_t2(0, 0), RenderSize(), fPlaneDepth);

                    const float fHalfViewportWidth = length(float2(RenderSize()));
                    const float fDepthThreshold    = max(fCurrentDepthViewSpace, fPrevNearestDepthViewSpace);

                    const float Ksep                     = 1.37e-05f;
                    const float Kfov                     = length(fCorner) / length(fCenter);
                    const float fRequiredDepthSeparation = Ksep * Kfov * fHalfViewportWidth * fDepthThreshold;

                    const float fResolutionFactor = saturate(length(float2(RenderSize())) / length(float2(1920.0f, 1080.0f)));
                    const float fPower            = lerp(1.0f, 3.0f, fResolutionFactor);
                    fDepth += pow(saturate(float(fRequiredDepthSeparation / fDepthDiff)), fPower) * fWeight;
                    fWeightSum += fWeight;
                }
            }
        }
    }

    return (fWeightSum > 0) ? saturate(1.0f - fDepth / fWeightSum) : 0.0f;
}

#endif  // #if defined(NSS_BIND_SRV_INPUT_DEPTH_TM1)

//-------------------------------------------------------------------------
// Input: Input‐depth
//-------------------------------------------------------------------------
#if defined(NSS_BIND_SRV_INPUT_DEPTH)
layout(set = 0, binding = NSS_BIND_SRV_INPUT_DEPTH) uniform highp texture2D r_input_depth;
#define _DepthTex sampler2D(r_input_depth, s_PointClamp)

FfxFloat32 LoadInputDepth(int32_t2 iPxPos)
{
    return texelFetch(r_input_depth, iPxPos, 0).r;
}

FfxFloat32 SampleInputDepth(FfxFloat32x2 fUV)
{
    return textureLod(sampler2D(r_input_depth, s_PointClamp), fUV, 0.0).r;
}

// motion vector dilation code adapted from Unity's TAA implementation
highp FfxFloat32x2 GetClosestFragment(highp FfxFloat32x2 uv, highp FfxFloat32 depth)
{
    highp FfxFloat32x2 k = InvOutputDims();  // output texel size

    highp FfxFloat32x4 neighborhood = FfxFloat32x4(textureLod(sampler2D(r_input_depth, s_PointClamp), uv - k, 0).x,
                                                   textureLod(sampler2D(r_input_depth, s_PointClamp), uv + FfxFloat32x2(k.x, -k.y), 0).x,
                                                   textureLod(sampler2D(r_input_depth, s_PointClamp), uv + FfxFloat32x2(-k.x, k.y), 0).x,
                                                   textureLod(sampler2D(r_input_depth, s_PointClamp), uv + k, 0).x);

#ifdef REVERSE_Z
#define COMPARE_DEPTH(a, b) step(b, a)
#else
#define COMPARE_DEPTH(a, b) step(a, b)
#endif

    highp FfxFloat32x3 result = FfxFloat32x3(0.f, 0.f, depth);
    result                    = mix(result, FfxFloat32x3(-1.f, -1.f, neighborhood.x), COMPARE_DEPTH(neighborhood.x, result.z));
    result                    = mix(result, FfxFloat32x3(1.f, -1.f, neighborhood.y), COMPARE_DEPTH(neighborhood.y, result.z));
    result                    = mix(result, FfxFloat32x3(-1.f, 1.f, neighborhood.z), COMPARE_DEPTH(neighborhood.z, result.z));
    result                    = mix(result, FfxFloat32x3(1.f, 1.f, neighborhood.w), COMPARE_DEPTH(neighborhood.w, result.z));

    return (uv + result.xy * k);
}

void FindNearestDepth(int32_t2 iPxPos, int32_t2 iPxSize, out float fNearestDepth, out int32_t2 fNearestDepthOffset)
{
    /*
        Closely based on:
        https://github.com/arm/accuracy-super-resolution-generic-library/blob/38697a58a6e7818ec9d28774bc073f537abb9178/
        include/gpu/fsr2/ffxm_fsr2_reconstruct_dilated_velocity_and_previous_depth.h#L59
    */

    int32_t       iSampleIndex = 0;
    const int32_t iSampleCount = 9;
    // x, y
    const int32_t2 iSampleOffsets[iSampleCount] = {
        int32_t2(+0, +0).yx,
        int32_t2(+1, +0).yx,
        int32_t2(+0, +1).yx,
        int32_t2(+0, -1).yx,
        int32_t2(-1, +0).yx,
        int32_t2(-1, +1).yx,
        int32_t2(+1, +1).yx,
        int32_t2(-1, -1).yx,
        int32_t2(+1, -1).yx,
    };

    // optimize by using textureGather (with textureGather, we can use 4 sample count instead of 9)
    // pull out the depth loads to allow SC to batch them
    float depth[9];
    depth[0] = float(texelFetchOffset(_DepthTex, iPxPos, 0, int32_t2(+0, +0).yx).r);
    depth[1] = float(texelFetchOffset(_DepthTex, iPxPos, 0, int32_t2(+1, +0).yx).r);
    depth[2] = float(texelFetchOffset(_DepthTex, iPxPos, 0, int32_t2(+0, +1).yx).r);
    depth[3] = float(texelFetchOffset(_DepthTex, iPxPos, 0, int32_t2(+0, -1).yx).r);
    depth[4] = float(texelFetchOffset(_DepthTex, iPxPos, 0, int32_t2(-1, +0).yx).r);
    depth[5] = float(texelFetchOffset(_DepthTex, iPxPos, 0, int32_t2(-1, +1).yx).r);
    depth[6] = float(texelFetchOffset(_DepthTex, iPxPos, 0, int32_t2(+1, +1).yx).r);
    depth[7] = float(texelFetchOffset(_DepthTex, iPxPos, 0, int32_t2(-1, -1).yx).r);
    depth[8] = float(texelFetchOffset(_DepthTex, iPxPos, 0, int32_t2(+1, -1).yx).r);

    // find closest depth
    fNearestDepth       = depth[0];
    fNearestDepthOffset = iSampleOffsets[0];
#pragma unroll
    for (iSampleIndex = 1; iSampleIndex < iSampleCount; ++iSampleIndex)
    {
        int32_t2 iPos = iPxPos + iSampleOffsets[iSampleIndex];
        if (IsOnScreen(iPos, iPxSize))
        {
            float fNdDepth = depth[iSampleIndex];
#ifdef INVERTED_DEPTH
            if (fNdDepth > fNearestDepth)
            {
#else
            if (fNdDepth < fNearestDepth)
            {
#endif
                fNearestDepth       = fNdDepth;
                fNearestDepthOffset = iSampleOffsets[iSampleIndex];
            }
        }
    }
}

#endif  // #if defined(NSS_BIND_SRV_INPUT_DEPTH)

//-------------------------------------------------------------------------
// Output: Luma‑deriv (RWTexture2D UAV) – R8G8_UNORM
//-------------------------------------------------------------------------
#if defined(NSS_BIND_UAV_LUMA_DERIV)
layout(set = 0, binding = NSS_BIND_UAV_LUMA_DERIV, rg8) uniform writeonly lowp image2D rw_luma_deriv;

void StoreLumaDeriv(int32_t2 iPxPos, FfxFloat32x2 v)
{
    imageStore(rw_luma_deriv, iPxPos, FfxFloat32x4(v, 0.0, 0.0));
}

void WriteLumaDerivative(int32_t2 pixel, half2 derivative)
{
    imageStore(rw_luma_deriv, pixel, half4(derivative, half2(0.f, 1.f)));
}

#endif  // NSS_BIND_UAV_LUMA_DERIV

//-------------------------------------------------------------------------
// Output: Depth Offset (aka NearestDepthCoordOut)
//-------------------------------------------------------------------------
#if defined(NSS_BIND_UAV_NEAREST_DEPTH_COORD)
layout(set = 0, binding = NSS_BIND_UAV_NEAREST_DEPTH_COORD, r8) uniform mediump image2D rw_nearest_depth_coord_out;

void WriteNearestDepthOffset(int32_t2 pixel, uint8_t offset)
{
    //FIXME: `/ 255` only needed because scenario runner doesn't support `R8_SINT`/`R8_UINT`
    half enc_norm = half(offset) / 255.HF;
    imageStore(rw_nearest_depth_coord_out, pixel, half4(enc_norm, 0.HF, 0.HF, 1.HF));
}
#endif

//-------------------------------------------------------------------------
// Ouput: Preprocessed‐tensor (RWBuffer UAV)
//-------------------------------------------------------------------------
#if defined(NSS_BIND_UAV_PREPROCESS_INPUT_TENSOR)

layout(set = 0, binding = NSS_BIND_UAV_PREPROCESS_INPUT_TENSOR) uniform tensorARM<tensor_t, 4> rw_preprocessed_tensor;
#define _PreprocessTensor rw_preprocessed_tensor

void WriteToTensor(int32_t2 outputPixel, half3 input_colour, half3 history, half disocclusion_mask, half luma_derivative, half4 temporal_feedback)
{
    TensorElement te;
    te.wh_rgb_col_r      = Quantize(half4(history.rgb, input_colour.r), InputQuantParams());
    te.col_gb_dm_fback_r = Quantize(half4(input_colour.gb, disocclusion_mask, temporal_feedback.r), InputQuantParams());
    te.fback_gba_ld      = Quantize(half4(temporal_feedback.gba, luma_derivative), InputQuantParams());

    // currently we have to split this into two calls to tensorWriteARM() since malisc only supports a limited number of data sizes.
    // See https://jira.arm.com/browse/MPGCOMP-18646
    int8_t t0[8] = {te.wh_rgb_col_r.x,
                    te.wh_rgb_col_r.y,
                    te.wh_rgb_col_r.z,
                    te.wh_rgb_col_r.w,
                    te.col_gb_dm_fback_r.x,
                    te.col_gb_dm_fback_r.y,
                    te.col_gb_dm_fback_r.z,
                    te.col_gb_dm_fback_r.w};
    tensorWriteARM(_PreprocessTensor, uint[](0, outputPixel.y, outputPixel.x, 0), t0);

    int8_t t1[4] = {te.fback_gba_ld.x, te.fback_gba_ld.y, te.fback_gba_ld.z, te.fback_gba_ld.w};
    tensorWriteARM(_PreprocessTensor, uint[](0, outputPixel.y, outputPixel.x, 8), t1);
}

PreprocessTensorElement LoadPreprocessTensor(int32_t2 coord)
{
    tensor_t wh_rgb_col_r[4];       // warped_history.rgb, jittered_colour.r
    tensor_t col_gb_dm_fback_r[4];  // jittered_colour.gb, disocclusion mask, feedback.r
    tensor_t fback_gba_ld[4];       // feedback.gba, luma derivative

    tensorReadARM(_PreprocessTensor, uint32_t[](0, coord.y, coord.x, 0), wh_rgb_col_r);
    tensorReadARM(_PreprocessTensor, uint32_t[](0, coord.y, coord.x, 4), col_gb_dm_fback_r);
    tensorReadARM(_PreprocessTensor, uint32_t[](0, coord.y, coord.x, 8), fback_gba_ld);

    PreprocessTensorElementInternal f;
    f.wh_rgb_col_r      = tensorVec_t(wh_rgb_col_r[0], wh_rgb_col_r[1], wh_rgb_col_r[2], wh_rgb_col_r[3]);
    f.col_gb_dm_fback_r = tensorVec_t(col_gb_dm_fback_r[0], col_gb_dm_fback_r[1], col_gb_dm_fback_r[2], col_gb_dm_fback_r[3]);
    f.fback_gba_ld      = tensorVec_t(fback_gba_ld[0], fback_gba_ld[1], fback_gba_ld[2], fback_gba_ld[3]);

#if QUANTIZED
    PreprocessTensorElement f_dequantized;
    f_dequantized.wh_rgb_col_r      = Dequantize(f.wh_rgb_col_r, QuantParamsSINT().zw);
    f_dequantized.col_gb_dm_fback_r = Dequantize(f.col_gb_dm_fback_r, QuantParamsSINT().zw);
    f_dequantized.fback_gba_ld      = Dequantize(f.fback_gba_ld, QuantParamsSINT().zw);
    return f_dequantized;
#endif

    // When we are not quantized, we can return the internal representation directly.
    return PreprocessTensorElement(f.wh_rgb_col_r, f.col_gb_dm_fback_r, f.fback_gba_ld);
}

#endif  // NSS_BIND_UAV_PREPROCESSED_TENSOR

//=========================================================================
// Resources for post-process
//=========================================================================

//-------------------------------------------------------------------------
// Input: Data Graph Output Tensors 0
//    There are 5 tensors.
//-------------------------------------------------------------------------
#if defined(NSS_BIND_SRV_K0_TENSOR) && defined(NSS_BIND_SRV_K1_TENSOR) && defined(NSS_BIND_SRV_K2_TENSOR) && defined(NSS_BIND_SRV_K3_TENSOR) && \
    defined(NSS_BIND_SRV_K4_TENSOR)
#define NSS_USE_DATA_GRAPH_OUTPUT_TENSORS 1
#endif

#if defined(NSS_USE_DATA_GRAPH_OUTPUT_TENSORS)

#if ALIAS_OUTPUT_TENSORS_AS_IMAGES
layout(set = 0, binding = NSS_BIND_SRV_K0_TENSOR) uniform lowp texture2D r_coefficients_k0_tensor;
layout(set = 0, binding = NSS_BIND_SRV_K1_TENSOR) uniform lowp texture2D r_coefficients_k1_tensor;
layout(set = 0, binding = NSS_BIND_SRV_K2_TENSOR) uniform lowp texture2D r_coefficients_k2_tensor;
layout(set = 0, binding = NSS_BIND_SRV_K3_TENSOR) uniform lowp texture2D r_coefficients_k3_tensor;
layout(set = 0, binding = NSS_BIND_SRV_K4_TENSOR) uniform lowp texture2D r_coefficients_k4_tensor;
#define _K0Tensor       sampler2D(r_coefficients_k0_tensor, s_LinearClamp)
#define _K1Tensor       sampler2D(r_coefficients_k1_tensor, s_LinearClamp)
#define _K2Tensor       sampler2D(r_coefficients_k2_tensor, s_LinearClamp)
#define _K3Tensor       sampler2D(r_coefficients_k3_tensor, s_LinearClamp)
#define _TemporalTensor sampler2D(r_coefficients_k4_tensor, s_LinearClamp)

// --- alias as image, so we can hardware-sample ---
half4 LoadKPNWeight(float2 uv, int16_t lut_idx)
{
    // Load 4 kernel slices (each with 4 taps)
    half4 k0 = Dequantize(half4(textureLod(_K0Tensor, uv, 0)), K0QuantParams());
    half4 k1 = Dequantize(half4(textureLod(_K1Tensor, uv, 0)), K1QuantParams());
    half4 k2 = Dequantize(half4(textureLod(_K2Tensor, uv, 0)), K2QuantParams());
    half4 k3 = Dequantize(half4(textureLod(_K3Tensor, uv, 0)), K3QuantParams());

    // Precomputed swizzle patterns for KernelTile
    half4 p0 = half4(k0.x, k2.x, k0.z, k2.z);
    half4 p1 = half4(k1.x, k3.x, k1.z, k3.z);
    half4 p2 = half4(k0.y, k2.y, k0.w, k2.w);
    half4 p3 = half4(k1.y, k3.y, k1.w, k3.w);

    // Return the correct pattern for this tile
    return (lut_idx == 0) ? p0 : (lut_idx == 1) ? p1 : (lut_idx == 2) ? p2 : p3;
}

void LoadKPNRaw(float2 uv, out half4 k0, out half4 k1, out half4 k2, out half4 k3)
{
    // Load 4 kernel slices (each with 4 taps)
    k0 = clamp(Dequantize(half4(textureLod(_K0Tensor, uv, 0)), K0QuantParams()), half4(EPS), half4(1.HF));
    k1 = clamp(Dequantize(half4(textureLod(_K1Tensor, uv, 0)), K1QuantParams()), half4(EPS), half4(1.HF));
    k2 = clamp(Dequantize(half4(textureLod(_K2Tensor, uv, 0)), K2QuantParams()), half4(EPS), half4(1.HF));
    k3 = clamp(Dequantize(half4(textureLod(_K3Tensor, uv, 0)), K3QuantParams()), half4(EPS), half4(1.HF));
}

void LoadTemporalParameters(float2 uv, out half theta, out half alpha)
{
    half2 tp = Dequantize(half2(textureLod(_TemporalTensor, uv, 0).xy), TemporalQuantParams());
    theta    = tp.x * NotHistoryReset();  // {0 <= x <= 1}
    alpha    = tp.y * 0.35HF + 0.05HF;    // { 0.05 <= x <= 0.4}
}

#else
layout(set = 0, binding = NSS_BIND_SRV_K0_TENSOR) uniform readonly tensorARM<tensor_t, 4> r_coefficients_k0_tensor;
layout(set = 0, binding = NSS_BIND_SRV_K1_TENSOR) uniform readonly tensorARM<tensor_t, 4> r_coefficients_k1_tensor;
layout(set = 0, binding = NSS_BIND_SRV_K2_TENSOR) uniform readonly tensorARM<tensor_t, 4> r_coefficients_k2_tensor;
layout(set = 0, binding = NSS_BIND_SRV_K3_TENSOR) uniform readonly tensorARM<tensor_t, 4> r_coefficients_k3_tensor;
layout(set = 0, binding = NSS_BIND_SRV_K4_TENSOR) uniform readonly tensorARM<tensor_t, 4> r_coefficients_k4_tensor;
DECLARE_SAMPLE_TENSOR(K0, r_coefficients_k0_tensor)
DECLARE_SAMPLE_TENSOR(K1, r_coefficients_k1_tensor)
DECLARE_SAMPLE_TENSOR(K2, r_coefficients_k2_tensor)
DECLARE_SAMPLE_TENSOR(K3, r_coefficients_k3_tensor)
DECLARE_SAMPLE_TENSOR(K4, r_coefficients_k4_tensor)

half4 LoadKPNWeight(float2 uv, int16_t lut_idx)
{
    // Load 4 kernel slices (each with 4 taps)
    half4 k0 = SampleK0Tensor(uv, K0QuantParams());
    half4 k1 = SampleK1Tensor(uv, K1QuantParams());
    half4 k2 = SampleK2Tensor(uv, K2QuantParams());
    half4 k3 = SampleK3Tensor(uv, K3QuantParams());

    // Precomputed swizzle patterns for KernelTile
    half4 p0 = half4(k0.x, k2.x, k0.z, k2.z);
    half4 p1 = half4(k1.x, k3.x, k1.z, k3.z);
    half4 p2 = half4(k0.y, k2.y, k0.w, k2.w);
    half4 p3 = half4(k1.y, k3.y, k1.w, k3.w);

    // Return the correct pattern for this tile
    return (lut_idx == 0) ? p0 : (lut_idx == 1) ? p1 : (lut_idx == 2) ? p2 : p3;
}

void LoadKPNRaw(float2 uv, out half4 k0, out half4 k1, out half4 k2, out half4 k3)
{
    // Load 4 kernel slices (each with 4 taps)
    k0 = clamp(SampleK0Tensor(uv, K0QuantParams()), half4(EPS), half4(1.HF));
    k1 = clamp(SampleK1Tensor(uv, K1QuantParams()), half4(EPS), half4(1.HF));
    k2 = clamp(SampleK2Tensor(uv, K2QuantParams()), half4(EPS), half4(1.HF));
    k3 = clamp(SampleK3Tensor(uv, K3QuantParams()), half4(EPS), half4(1.HF));
}

void LoadTemporalParameters(float2 uv, out half theta, out half alpha)
{
    half2 tp = SampleK4Tensor(uv, TemporalQuantParams()).xy;
    theta    = tp.x * NotHistoryReset();  // {0 <= x <= 1}
    alpha    = tp.y * 0.35HF + 0.05HF;    // { 0.05 <= x <= 0.4}
}

#endif  //#if ALIAS_OUTPUT_TENSORS_AS_IMAGES

#if SCALE_MODE == SCALE_2_0X
/*
    Optimised special case pattern for applying 4x4 kernel to
    sparse jitter-aware 2x2 upsampled image
*/
half3 LoadAndFilterColour(int32_t2 output_pixel, float2 uv, out half4 col_to_accum)
{
    //-------------------------------------------------------------------
    // 1. Compute indexes, load correct pattern from LUT for given thread
    //-------------------------------------------------------------------
    float2 out_tex = float2(output_pixel) + 0.5f;

    // Compute the LUT index for this pixel
    int16_t2   tiled_idx = (int16_t2(output_pixel) + LutOffset()) % int16_t2(IndexModulo());
    int16_t    lut_idx   = tiled_idx.y * int16_t(IndexModulo()) + tiled_idx.x;
    KernelTile lut       = kernelLUT[lut_idx];

    //------------------------------------------------------------------
    // 2. Apply KPN
    //------------------------------------------------------------------
    // Dequantize the kernel weights
    half4 kpn_weights = clamp(LoadKPNWeight(uv, lut_idx), half4(EPS), half4(1.HF));

    // Calculate tap locations
    int16_t4 tap_x = clamp(int16_t4(floor((float4(out_tex.x) + float4(lut.dx)) * InvScale().x)), int16_t4(0), int16_t4(InputDims().x - 1));
    int16_t4 tap_y = clamp(int16_t4(floor((float4(out_tex.y) + float4(lut.dy)) * InvScale().y)), int16_t4(0), int16_t4(InputDims().y - 1));

    // Gather taps
    half4x4 interm;
    interm[0] = half4(SafeColour(half3(texelFetch(_ColourTex, int16_t2(tap_x[0], tap_y[0]), 0).rgb) * half3(Exposure())), 1.HF);
    interm[1] = half4(SafeColour(half3(texelFetch(_ColourTex, int16_t2(tap_x[1], tap_y[1]), 0).rgb) * half3(Exposure())), 1.HF);
    interm[2] = half4(SafeColour(half3(texelFetch(_ColourTex, int16_t2(tap_x[2], tap_y[2]), 0).rgb) * half3(Exposure())), 1.HF);
    interm[3] = half4(SafeColour(half3(texelFetch(_ColourTex, int16_t2(tap_x[3], tap_y[3]), 0).rgb) * half3(Exposure())), 1.HF);

    // Special case: grab the accumulation pixel, when it corresponds to current thread
    half match   = half(lut.dx[CENTER_TAP] == 0 && lut.dy[CENTER_TAP] == 0);
    col_to_accum = interm[CENTER_TAP] * match;

    // Apply filter
    half4 out_colour = interm * kpn_weights;

    return half3(out_colour.rgb * rcp(out_colour.w));
}
#else
/*
    Common metod for applying KPN to arbitrary scale factors.
*/
half3 LoadAndFilterColour(int32_t2 output_pixel, float2 uv, out half4 col_to_accum)
{
    float weight_sum = 0.0f;
    half3 out_colour = half3(0.HF);
    col_to_accum     = half4(0.HF);

    half4x4 kpn_weights;
    LoadKPNRaw(uv, kpn_weights[0], kpn_weights[1], kpn_weights[2], kpn_weights[3]);

    for (int x = 0; x < 4; x++)
    {
        for (int y = 0; y < 4; y++)
        {
            half     weight       = kpn_weights[x][y];
            int32_t2 tap_location = output_pixel + int32_t2((x - 1), (y - 1));

            // Figure out which pixel in the input image corresponds to this location in output space.
            // There might not be one though, depending on the jitter and upscaling ratio etc.
            // For each of x, y (independently), the input sample points are at locations:
            //   j + n + 0.5,  in the input space, where j = jitter offset in input pixels and n is any integer from 0 to input image size
            //   sj + sn + 0.5s = sn + (j + 0.5)*s, in the OUTPUT space, where s = upscaling ratio (e.g. 2.0)
            //
            //  We need to find if there is a value of n which corresponds to our desired output pixel location (o)
            //   floor(sn + (j + 0.5)*s) = o
            //   o                     <= sn + (j + 0.5)*s   < o + 1
            //   o - (j + 0.5)*s       <= sn                 < o + 1 - (j + 0.5)*s
            //   (o - (j + 0.5)*s) / s <= n                  < (o + 1 - (j + 0.5)*s) / s
            //   o/s - (j + 0.5)       <= n                  < (o + 1) / s - (j + 0.5)
            int32_t2 input_pixel = int32_t2(-1);
            half2    lower       = half2(tap_location) * half2(InvScale()) - half2((JitterOffsetPixel() + 0.5f));
            half2    upper       = half2(tap_location + int32_t2(1)) * half2(InvScale()) - half2((JitterOffsetPixel() + 0.5f));
            int32_t2 candidate   = int32_t2(floor(upper));
            if (lower.x < half2(candidate).x && lower.y < half2(candidate).y)
            {
                input_pixel = candidate;
            }

            if (input_pixel.x >= 0 && input_pixel.y >= 0)
            {
                input_pixel     = clamp(input_pixel, int32_t2(0), int32_t2(InputDims() - 1));
                half3 tap_color = half3(texelFetch(_ColourTex, int16_t2(input_pixel), 0).rgb);

                out_colour += tap_color * weight;
                weight_sum += weight;

                if (x == 1 && y == 1)
                {
                    // Centre tap - there is a sample point exactly on this output pixel
                    col_to_accum = half4(tap_color, 1.HF);
                }
            }
        }
    }

    out_colour = SafeColour(out_colour * half3(Exposure()) / half3(weight_sum + EPS));
    col_to_accum.rgb *= half3(Exposure());

    return out_colour;
}

#endif  //#if SCALE_MODE == SCALE_2_0X

#endif

//-------------------------------------------------------------------------
// Input: Depth Offset (aka NearestDepthCoordTex)
//-------------------------------------------------------------------------
#if defined(NSS_BIND_SRV_NEAREST_DEPTH_COORD)
layout(set = 0, binding = NSS_BIND_SRV_NEAREST_DEPTH_COORD) uniform highp texture2D r_input_nearest_depth_coord;
#define _NearestDepthCoordTex sampler2D(r_input_nearest_depth_coord, s_LinearClamp)

int32_t2 LoadNearestDepthOffset(int32_t2 pixel)
{
    //FIXME: `* 255` only needed because scenario runner doesn't support `R8_SINT`/`R8_UINT`
    half    encNorm = half(texelFetch(_NearestDepthCoordTex, pixel, 0).r);
    int32_t code    = int32_t(encNorm * 255.0 + 0.5);

    // 3. map back to {-1,0,1}²
    return DecodeNearestDepthCoord(code);
}

#if defined(NSS_BIND_SRV_HISTORY_UPSCALED_COLOR) && defined(NSS_BIND_SRV_INPUT_MOTION_VECTORS)
half3 LoadWarpedHistory(float2 uv, int32_t2 input_pixel, out half onscreen)
{
    // Dilate motion vectors with previously calculated nearest depth coordinate
    int32_t2 nearest_offset = LoadNearestDepthOffset(input_pixel);
    half2    motion         = LoadMotion(input_pixel + nearest_offset);

    // Suppress very small motion - no need to resample
    half2 motion_pix = motion * half2(OutputDims());
    motion *= half(dot(motion_pix, motion_pix) > MotionThresh());

    // UV coordinates in previous frame to resample history
    float2 reproj_uv = uv - float2(motion);

    // Mask to flag whether the motion vector is resampling from valid location onscreen
    onscreen = half(all(greaterThanEqual(reproj_uv, float2(0.0))) && all(lessThan(reproj_uv, float2(1.0))));

#ifdef HISTORY_CATMULL
    half3 warped_history = LoadHistoryCatmull(reproj_uv);
#else
    half3 warped_history = LoadHistory(reproj_uv);
#endif

    return SafeColour(warped_history * Exposure());
}
#endif

#endif

//-------------------------------------------------------------------------
// Input: KernelLUT (baked into ffx_nss_common_glsl.h)
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
// Output: Upscaled‑output (RWTexture2D UAV)
//-------------------------------------------------------------------------
#if defined(NSS_BIND_UAV_UPSCALED_OUTPUT)
layout(set = 0, binding = NSS_BIND_UAV_UPSCALED_OUTPUT, OUTPUT_IMG_FORMAT) uniform mediump image2D rw_upscaled_output;

half4 LoadUpscaledOutput(int32_t2 iPxPos)
{
    return half4(imageLoad(rw_upscaled_output, iPxPos));
}

void StoreUpscaledOutput(int32_t2 iPxPos, FfxFloat32x4 v)
{
    imageStore(rw_upscaled_output, iPxPos, v);
}

half3 SimulateR11G11B10Precision(half3 rgb)
{
    const float   epsilon  = 1e-12;
    const int32_t r_m_bits = 6, g_m_bits = 6, b_m_bits = 5;
    const int32_t exp_bits = 5;
    const float   bias     = float((1 << (exp_bits - 1)) - 1);  // 15.0

    float3 val = max(float3(rgb), float3(epsilon));  // Clamp to avoid denormals

    float3 exp_unclamped = floor(log2(val));
    float3 exp_clipped   = clamp(exp_unclamped, -bias, bias + 1.0);  // Allow max valid exp

    float3 mant = val / exp2(exp_clipped) - 1.0;

    // Scale mantissa to simulate quantization
    float r_scale = float(1 << r_m_bits);
    float g_scale = float(1 << g_m_bits);
    float b_scale = float(1 << b_m_bits);

    float r_exp = exp_clipped.r;
    float g_exp = exp_clipped.g;
    float b_exp = exp_clipped.b;

    float exp_max = (1 << exp_bits) - 1;

    r_exp = clamp(r_exp + bias, 0.f, exp_max) - bias;
    g_exp = clamp(g_exp + bias, 0.f, exp_max) - bias;
    b_exp = clamp(b_exp + bias, 0.f, exp_max) - bias;

    float r_mant = clamp(round(mant.r * r_scale), 0.f, r_scale - 1) / r_scale;
    float g_mant = clamp(round(mant.g * g_scale), 0.f, g_scale - 1) / g_scale;
    float b_mant = clamp(round(mant.b * b_scale), 0.f, b_scale - 1) / b_scale;

    // Reconstruct quantized float: (1 + mantissa) * 2^exp
    float r_out = (1.0 + r_mant) * exp2(r_exp);
    float g_out = (1.0 + g_mant) * exp2(g_exp);
    float b_out = (1.0 + b_mant) * exp2(b_exp);

    return half3(r_out, g_out, b_out);
}
// #define SIMULATE_R11G11B10_BEFORE_WRITE 1
void WriteUpsampledColour(int32_t2 pixel, half3 colour)
{
    half3 to_write = SafeColour(colour);
#ifdef SIMULATE_R11G11B10_BEFORE_WRITE
    to_write = SimulateR11G11B10Precision(to_write);
#endif
    // Write with alpha = 1.0
    imageStore(rw_upscaled_output, pixel, half4(to_write, 1.0));
}

#endif  // #if defined(NSS_BIND_UAV_UPSCALED_OUTPUT)

//-------------------------------------------------------------------------
// Debug views (RWTexture2D UAV)
//-------------------------------------------------------------------------
#if defined(NSS_BIND_UAV_DEBUG_VIEWS)
layout(set = 0, binding = NSS_BIND_UAV_DEBUG_VIEWS, OUTPUT_IMG_FORMAT) uniform mediump image2D rw_debug_views;

float4 LoadDebugView(int32_t2 iPxPos)
{
    return imageLoad(rw_debug_views, iPxPos);
}

void StoreDebugView(int32_t2 iPxPos, float4 v)
{
    imageStore(rw_debug_views, iPxPos, v);
}

float2 GetDebugViewDimensions()
{
    return imageSize(rw_debug_views);
}
#endif  // #if defined(NSS_BIND_UAV_DEBUG_VIEWS)

#endif  // #if defined(FFX_GPU)

#endif  // GPU_NSS_FFX_NSS_CALLBACKS_GLSL_H