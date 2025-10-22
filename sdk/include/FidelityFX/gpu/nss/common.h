/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef GPU_NSS_COMMON_H
#define GPU_NSS_COMMON_H

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
#endif  // GPU_NSS_COMMON_H
