/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#if !defined(FFX_NSS_COMMON_GLSL_H)
#define FFX_NSS_COMMON_GLSL_H

#if defined(FFX_GPU)

#define FLT_MIN            1.175494351e-38
#define MEDIUMP_FLT_MAX    65504.f
#define MEDIUMP_FLT_MIN    0.00006103515625
#define FLT_EPS            MEDIUMP_FLT_MIN
#define SaturateMediump(x) min(x, MEDIUMP_FLT_MAX)

// clamp HDR value within a safe range
vec3 SafeHDR(vec3 c)
{
    return vec3(clamp(c, 0.f, MEDIUMP_FLT_MAX));
}

// optimised bicubic catmull rom implementation adapted from TAAStandalone.usf in UE4
void Bicubic2DCatmullRom(in vec2 uv, in vec2 size, in vec2 invSize, out vec2 samples[3], out vec2 weights[3])
{
    uv *= size;

    vec2 tc = floor(uv - 0.5) + 0.5;
    vec2 f  = uv - tc;
    vec2 f2 = f * f;
    vec2 f3 = f2 * f;

    vec2 w0 = f2 - 0.5 * (f3 + f);
    vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.f;
    vec2 w3 = 0.5 * (f3 - f2);
    vec2 w2 = 1.f - w0 - w1 - w3;

    weights[0] = w0;
    weights[1] = w1 + w2;
    weights[2] = w3;

    samples[0] = tc - 1.f;
    samples[1] = tc + w2 / weights[1];
    samples[2] = tc + 2.f;

    samples[0] *= invSize;
    samples[1] *= invSize;
    samples[2] *= invSize;
}

#define BICUBIC_CATMULL_ROM_SAMPLES 5U

struct CatmullRomSamples
{
    // constant number of samples (BICUBIC_CATMULL_ROM_SAMPLES)
    uint Count;

    // bilinear sampling UV coordinates of the samples
    vec2 UV[BICUBIC_CATMULL_ROM_SAMPLES];

    // weights of the samples
    float Weight[BICUBIC_CATMULL_ROM_SAMPLES];

    // final multiplier (it is faster to multiply 3 RGB values than reweights the 5 weights)
    float FinalMultiplier;
};

CatmullRomSamples GetBicubic2DCatmullRomSamples(vec2 uv, vec2 size, in vec2 invSize)
{
    vec2 weights[3];
    vec2 samples[3];
    Bicubic2DCatmullRom(uv, size, invSize, samples, weights);

    CatmullRomSamples crSamples;
    crSamples.Count = BICUBIC_CATMULL_ROM_SAMPLES;

    // optimized by removing corner samples
    crSamples.UV[0] = vec2(samples[1].x, samples[0].y);
    crSamples.UV[1] = vec2(samples[0].x, samples[1].y);
    crSamples.UV[2] = vec2(samples[1].x, samples[1].y);
    crSamples.UV[3] = vec2(samples[2].x, samples[1].y);
    crSamples.UV[4] = vec2(samples[1].x, samples[2].y);

    crSamples.Weight[0] = weights[1].x * weights[0].y;
    crSamples.Weight[1] = weights[0].x * weights[1].y;
    crSamples.Weight[2] = weights[1].x * weights[1].y;
    crSamples.Weight[3] = weights[2].x * weights[1].y;
    crSamples.Weight[4] = weights[1].x * weights[2].y;

    // reweight after removing the corners
    float cornerWeights;
    cornerWeights = crSamples.Weight[0];
    cornerWeights += crSamples.Weight[1];
    cornerWeights += crSamples.Weight[2];
    cornerWeights += crSamples.Weight[3];
    cornerWeights += crSamples.Weight[4];
    crSamples.FinalMultiplier = 1.f / cornerWeights;

    return crSamples;
}

float sigmoid(float x)
{
    // sigmoid(x) = 1 / (1 + exp(-x))
    return (1.f / (1.f + exp(-x)));
}

vec2 sigmoid(vec2 x)
{
    // sigmoid(x) = 1 / (1 + exp(-x))
    return (1.f / (1.f + exp(-x)));
}

vec3 sigmoid(vec3 x)
{
    // sigmoid(x) = 1 / (1 + exp(-x))
    return (1.f / (1.f + exp(-x)));
}

vec4 sigmoid(vec4 x)
{
    // sigmoid(x) = 1 / (1 + exp(-x))
    return (1.f / (1.f + exp(-x)));
}

vec3 clip_aabb(vec3 minc, vec3 maxc, vec3 history)
{
    vec3  center    = 0.5f * (maxc + minc);
    vec3  extents   = 0.5f * (maxc - minc);
    vec3  offset    = history - center;
    vec3  ts        = abs(extents / (offset + FLT_EPS));
    float t         = clamp(min(min(ts.r, ts.g), ts.b), 0.0, 1.0);
    vec3  rectified = center + offset * t;
    return rectified;
}

vec3 RGBToYCoCg(vec3 Rgb)
{
    vec3 YCoCg;
    YCoCg.y   = Rgb.r - Rgb.b;
    float tmp = Rgb.b + YCoCg.y / 2.0;
    YCoCg.z   = Rgb.g - tmp;
    YCoCg.x   = tmp + YCoCg.z / 2.0;
    return YCoCg;
}

float RGBToY(vec3 Rgb)
{
    return 0.25 * Rgb.r + 0.5 * Rgb.g + 0.25 * Rgb.b;
}

vec3 YCoCgToRGB(vec3 YCoCg)
{
    vec3  Rgb;
    float tmp = YCoCg.x - YCoCg.z / 2.0;
    Rgb.g     = YCoCg.z + tmp;
    Rgb.b     = tmp - YCoCg.y / 2.0;
    Rgb.r     = Rgb.b + YCoCg.y;
    return Rgb;
}

FfxFloat32 GetViewSpaceDepth(FfxFloat32 fDeviceDepth, FfxFloat32x4 fDeviceToViewDepth)
{
    // fDeviceToViewDepth details found in ffx_nss.cpp
    return (fDeviceToViewDepth[1] / (fDeviceDepth - fDeviceToViewDepth[0]));
}

#define lerp mix

#if FFX_HALF
// --- RCP functions for float16 types ---
half rcp(half x)
{
    return half(1.HF) / x;
}
half2 rcp(half2 x)
{
    return half2(1.HF) / x;
}
half3 rcp(half3 x)
{
    return half3(1.HF) / x;
}
half4 rcp(half4 x)
{
    return half4(1.HF) / x;
}

// --- Saturate functions for float16 types ---
half saturate(half x)
{
    return clamp(x, half(0.HF), half(1.HF));
}
half2 saturate(half2 x)
{
    return clamp(x, half2(0.HF), half2(1.HF));
}
half3 saturate(half3 x)
{
    return clamp(x, half3(0.HF), half3(1.HF));
}
half4 saturate(half4 x)
{
    return clamp(x, half4(0.HF), half4(1.HF));
}
#endif

// --- RCP functions for float32 types ---
float rcp(float x)
{
    return float(1.0f) / x;
}
float2 rcp(float2 x)
{
    return float2(1.0f) / x;
}
float3 rcp(float3 x)
{
    return float3(1.0f) / x;
}
float4 rcp(float4 x)
{
    return float4(1.0f) / x;
}

// --- Saturate functions for float32 types ---
float saturate(float x)
{
    return clamp(x, 0.f, 1.f);
}
float2 saturate(float2 x)
{
    return clamp(x, float2(0.f), float2(1.f));
}
float3 saturate(float3 x)
{
    return clamp(x, float3(0.f), float3(1.f));
}
float4 saturate(float4 x)
{
    return clamp(x, float4(0.f), float4(1.f));
}

#define MAX_FP16 65504.HF
#define EPS      1e-7HF

#define TONEMAP_KARIS
// #define TONEMAP_REINHARD

#if defined(TONEMAP_KARIS)
half3 Tonemap(half3 x)
{
    // Karis tonemapper
    // http://graphicrants.blogspot.com/2013/12/tone-mapping.html
    x = clamp(x, half3(0.HF), half3(MAX_FP16));
    return x * rcp(half3(1.HF) + max(max(x.r, x.g), x.b));
}

half3 InverseTonemap(half3 x)
{
    // Karis tonemapper inverse
    // http://graphicrants.blogspot.com/2013/12/tone-mapping.html
    x = clamp(x, half3(0.HF), half3(1.HF - EPS));
    return x * rcp(half3(1.HF) - max(max(x.r, x.g), x.b));
}
#elif defined(TONEMAP_REINHARD)
vec3 Tonemap(vec3 colour)
{
    // 1 / (1 + colour)
    return colour / (vec3(1.f) + colour);
}

vec3 TonemapExp(vec3 colour, float exponent)
{
    // 1 / (1 + colour)
    colour = colour * exp(exponent);
    return colour / (vec3(1.f) + colour);
}

vec3 InverseTonemap(vec3 colour)
{
    // 1 / (1 - colour)
    return colour / (vec3(1.f) - colour);
}

vec4 Tonemap(vec4 colour)
{
    // 1 / (1 + colour)
    return colour / (vec4(1.f) + colour);
}

vec4 InverseTonemap(vec4 colour)
{
    // 1 / (1 - colour)
    return colour / (vec4(1.f) - colour);
}

#if FFX_HALF
half3 Tonemap(half3 colour)
{
    // 1 / (1 + colour)
    return colour / (half3(1.f) + colour);
}

half3 TonemapExp(half3 colour, half exponent)
{
    // 1 / (1 + colour)
    colour = colour * exp(exponent);
    return colour / (half3(1.f) + colour);
}

half3 InverseTonemap(half3 colour)
{
    // 1 / (1 - colour)
    return colour / (half3(1.f) - colour);
}

half4 Tonemap(half4 colour)
{
    // 1 / (1 + colour)
    return colour / (half4(1.f) + colour);
}

half4 InverseTonemap(half4 colour)
{
    // 1 / (1 - colour)
    return colour / (half4(1.f) - colour);
}
#endif
#endif

bool IsOnScreen(int32_t2 pos, int32_t2 size)
{
    return all(lessThan(uint32_t2(pos), uint32_t2(size)));
}

half Luminance(half3 rgb)
{
    // ITU-R BT.709: `0.2126 * R + 0.7152 * G + 0.0722 * B`
    return dot(rgb, half3(0.2126, 0.7152, 0.0722));
}

// Quantize/Dequantize
// ──────────────────────────────────────────────────────────────────────────────────────────
// all expect .x = scale, .y = zero point, dequantize expect to receive: .x = rcp(scale)

half Dequantize(half i, half2 quant_params)
{
    return (i - quant_params.y) * quant_params.x;
}

half2 Dequantize(half2 i, half2 quant_params)
{
    return (i - quant_params.y) * quant_params.x;
}

half3 Dequantize(half3 i, half2 quant_params)
{
    return (i - quant_params.y) * quant_params.x;
}

half4 Dequantize(half4 i, half2 quant_params)
{
    return (i - quant_params.y) * quant_params.x;
}

int8_t Quantize(half f, half2 quant_params)
{
    return int8_t(clamp(round(f * quant_params.x + quant_params.y), half(-128.HF), half(127.HF)));
}

int8_t2 Quantize(half2 f, half2 quant_params)
{
    return int8_t2(clamp(round(f * quant_params.x + quant_params.y), half2(-128.HF), half2(127.HF)));
}

int8_t3 Quantize(half3 f, half2 quant_params)
{
    return int8_t3(clamp(round(f * quant_params.x + quant_params.y), half3(-128.HF), half3(127.HF)));
}

int8_t4 Quantize(half4 f, half2 quant_params)
{
    return int8_t4(clamp(round(f * quant_params.x + quant_params.y), half4(-128.HF), half4(127.HF)));
}

half3 SafeColour(half3 x)
{
    return clamp(x, half3(0.HF), half3(MAX_FP16));
}

// Encode/Decode
// ──────────────────────────────────────────────────────────────────────────────────────────
// Note: both encode/decode methods are currently bound to 3x3 windows, they should be
//       expandable in future if needed. The most likely to need this would be the jitter
//       encoding, where 3x3 may not be enough for larger than 3x3 scale factors.

uint8_t EncodeNearestDepthCoord(int32_t2 o)
{
    // o ∈ {-1, 0, 1}²
    o = clamp(o, ivec2(-1), ivec2(1));
    return uint8_t((o.y + 1) << 2 | (o.x + 1));  // 0-15
}

int32_t2 DecodeNearestDepthCoord(int32_t code)
{
    int32_t x = int32_t(code & 0x3) - 1;         // bits 0-1
    int32_t y = int32_t((code >> 2) & 0x3) - 1;  // bits 2-3
    return int32_t2(x, y);
}

struct KernelTile
{
    int16_t4 dx;
    int16_t4 dy;
};

// Define actual scale value based on mode
#if SCALE_MODE == 0
//No LUT method.

#elif SCALE_MODE == SCALE_1_3X
// x1.3 not supported yet

#elif SCALE_MODE == SCALE_1_5X
// x1.5 not supported yet

#elif SCALE_MODE == SCALE_2_0X

#define CENTER_TAP   0
#define NUM_PATTERNS 4

const KernelTile kernelLUT[NUM_PATTERNS] = {{// Pattern 0:
                                             // Taps:  0,  2,  8, 10
                                             // Grid:
                                             //   [●  ·  ●  ·]
                                             //   [·  ·  ·  ·]
                                             //   [●  ·  ●  ·]
                                             //   [·  ·  ·  ·]
                                             int16_t4(-1, +1, -1, +1),
                                             int16_t4(-1, -1, +1, +1)},
                                            {// Pattern 1:
                                             // Taps:  1,  3,  9, 11
                                             // Grid:
                                             //   [·  ●  ·  ●]
                                             //   [·  ·  ·  ·]
                                             //   [·  ●  ·  ●]
                                             //   [·  ·  ·  ·]
                                             int16_t4(0, +2, 0, +2),
                                             int16_t4(-1, -1, +1, +1)},
                                            {// Pattern 2:
                                             // Taps:  4,  6, 12, 14
                                             // Grid:
                                             //   [·  ·  ·  ·]
                                             //   [●  ·  ●  ·]
                                             //   [·  ·  ·  ·]
                                             //   [●  ·  ●  ·]
                                             int16_t4(-1, +1, -1, +1),
                                             int16_t4(0, 0, +2, +2)},
                                            {// Pattern 3:
                                             // Taps:  5,  7, 13, 15
                                             // Grid:
                                             //   [·  ·  ·  ·]
                                             //   [·  ●  ·  ●]
                                             //   [·  ·  ·  ·]
                                             //   [·  ●  ·  ●]
                                             int16_t4(0, +2, 0, +2),  // center-aligned
                                             int16_t4(0, 0, +2, +2)}};

#else
#error "Unsupported SCALE_MODE"
#endif

#endif  // #if defined(FFX_GPU)

#endif  //!defined(FFX_NSS_COMMON_HLSL_H)
