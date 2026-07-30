// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#ifndef GPU_FFX_FRAMEINTERPOLATION_20_PREPROCESS_PASS_H
#define GPU_FFX_FRAMEINTERPOLATION_20_PREPROCESS_PASS_H

#include "ffx_frameinterpolation_callbacks_glsl.h"
#include "warp_flow_common.h"

float SingleTapDepthClip(float current_depth, float previous_depth)
{
    /*
        Closely based on:
        https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/f4c1da8e92f3fe563b5c28c44e6267ce6b6b8eb2/
        Kits/FidelityFX/framegeneration/fsr3/include/gpu/frameinterpolation/ffx_frameinterpolation_disocclusion_mask.h#L65
    */

    const float2 render_size            = float2(RenderSize());
    const float  fCurrentDepthViewSpace = ConvertFromDeviceDepthToViewSpace(current_depth);

    const float fPrevDepthSample           = previous_depth;
    const float fPrevNearestDepthViewSpace = ConvertFromDeviceDepthToViewSpace(fPrevDepthSample);

    const float fDepthDiff = fCurrentDepthViewSpace - fPrevNearestDepthViewSpace;
    if (fDepthDiff <= 0)
        return 0;

    const float fPlaneDepth = max(fPrevDepthSample, current_depth);

    const float3 fCenter = GetViewSpacePosition(int32_t2(render_size * 0.5f), int32_t2(render_size), fPlaneDepth);
    const float3 fCorner = GetViewSpacePosition(int32_t2(0, 0), int32_t2(render_size), fPlaneDepth);

    const float fHalfViewportWidth = length(float2(render_size));
    const float fDepthThreshold    = max(fCurrentDepthViewSpace, fPrevNearestDepthViewSpace);

    const float Ksep                     = 1.37e-05f;
    const float Kfov                     = length(fCorner) / length(fCenter);
    const float fRequiredDepthSeparation = Ksep * Kfov * fHalfViewportWidth * fDepthThreshold;

    const float fResolutionFactor = ffxSaturate(length(float2(render_size)) / length(float2(1080.0f, 1920.0f)));
    const float fPower            = ffxLerp(1.0f, 3.0f, fResolutionFactor);
    const float fDepth            = pow(ffxSaturate(fRequiredDepthSeparation / fDepthDiff), fPower);

    return 1.0 - fDepth;
}

// Transform tp1 depth values to tm1 frame's screen space
float reproject_depth_fwd(float2 uv, float depth)
{
    float2 uvT    = float2(uv.x, 1.0f - uv.y);
    float4 clip   = float4(2.0f * uvT - 1.0f, depth, 1.0f);
    float4 reproj = FFX_TRANSFORM_VECTOR(GetMotionTP1ToTM1(), clip);
    return reproj.z / reproj.w;
}

// Transform tm1 depth values to tp1 frame's screen space
float reproject_depth_bwd(float2 uv, float depth)
{
    float2 uvT    = float2(uv.x, 1.0f - uv.y);
    float4 clip   = float4(2.0f * uvT - 1.0f, depth, 1.0f);
    float4 reproj = FFX_TRANSFORM_VECTOR(GetMotionTM1ToTP1(), clip);
    return reproj.z / reproj.w;
}

FFX_MIN16_F2 SingleTapCalculateDisocclusionMask(float depth_m1_warp_t_mv, float depth_p1_warp_t_mv, float2 uv_m1_mv, float2 uv_p1_mv)
{
    // Reproject/transform depth to respective opposite depth screen space
    float depth_m1_warp_t_mv_transform_p1 = reproject_depth_bwd(uv_m1_mv, depth_m1_warp_t_mv);
    float depth_p1_warp_t_mv_transform_m1 = reproject_depth_fwd(uv_p1_mv, depth_p1_warp_t_mv);

    float dis_mask_m1 = SingleTapDepthClip(depth_p1_warp_t_mv, depth_m1_warp_t_mv_transform_p1);
    float dis_mask_p1 = SingleTapDepthClip(depth_m1_warp_t_mv, depth_p1_warp_t_mv_transform_m1);

    return FFX_MIN16_F2(dis_mask_m1, dis_mask_p1);
}

uint32_t hash32_fast(uint32_t v)
{
    v = (v ^ 61u) ^ (v >> 16);
    v *= 9u;
    v ^= v >> 4;
    v *= 0x27d4eb2du;
    v ^= v >> 15;
    return v;
}

half hash_to_half(uint32_t x, uint32_t y)
{
    uint32_t v = x + 0x9e3779b9u * y + RandomSeed();
    uint32_t h = hash32_fast(v);
    return half(uintBitsToFloat((h >> 9) | 0x3f800000u) - 1.0f);
}

void WriteToInputTensor(int32_t2     outputPixel,
                        FFX_MIN16_F3 rgb_m1_warp_t_mv,
                        FFX_MIN16_F3 rgb_p1_warp_t_mv,
                        FFX_MIN16_F3 rgb_m1_warp_t_flow,
                        FFX_MIN16_F3 rgb_p1_warp_t_flow,
                        FFX_MIN16_F2 dis_mask,
                        float2       depth)
{
    InputTensorElement_t te;

    int8_t zero = int8_t(0);

    te.rgbM1MV_rP1MV = Quantize(FFX_MIN16_F4(rgb_m1_warp_t_mv.rgb, rgb_p1_warp_t_mv.r), QuantParamsSINT().xy);      //rgbM1MV_rP1MV
    te.gbP1MV_rgM1OF = Quantize(FFX_MIN16_F4(rgb_p1_warp_t_mv.gb, rgb_m1_warp_t_flow.rg), QuantParamsSINT().xy);    //gbP1MV_rgM1OF
    te.bM1OF_rgbP1OF = Quantize(FFX_MIN16_F4(rgb_m1_warp_t_flow.b, rgb_p1_warp_t_flow.rgb), QuantParamsSINT().xy);  //bM1OF_rgbP1OF
    te.depth_disMask = Quantize(FFX_MIN16_F4(depth, dis_mask), QuantParamsSINT().xy);                               //depth_disMask

    StoreInTensor(outputPixel, te);
}

void preprocess(int32_t2 input_pixel)
{
    int32_t2 padded_output_pixel = input_pixel;
    int32_t2 renderSize          = RenderSize();
    float2   render_size         = float2(renderSize);
    int32_t2 mvDepthLane         = MvDepthLaneSize();
    int32_t2 flowLane            = FlowLaneSize();
    if (any(greaterThanEqual(padded_output_pixel, InputTensorSize())))
    {
        return;
    }

    int32_t2 output_pixel = padded_output_pixel;
    if (output_pixel.x >= flowLane.x)
    {
        output_pixel.x = (flowLane.x - 1) - (output_pixel.x - flowLane.x);
    }
    if (output_pixel.y >= flowLane.y)
    {
        output_pixel.y = (flowLane.y - 1) - (output_pixel.y - flowLane.y);
    }
    float2 invFlowLane = FlowLaneSizeRcp();
    float2 uv          = ((output_pixel.xy) + 0.5f) * invFlowLane;

    int32_t2 iMvCoord   = int32_t2(floor(uv * float2(mvDepthLane)));
    int32_t2 iFlowCoord = int32_t2(floor(uv * float2(flowLane)));

    // Read Vectors warped to "t" position - sampling nearest
    FFX_MIN16_F2 flow_t_f30_p1 = LoadWarpedFilledFlowTM1(iFlowCoord);
    FFX_MIN16_F2 mv_t_f30_m1   = LoadWarpedFilledMotionTP1(iMvCoord);

    // Calculate warped game mv positions
    float  timestep = Timestep();
    float2 uv_m1_mv = uv + float2(mv_t_f30_m1 * timestep);
    float2 uv_p1_mv = uv - float2(mv_t_f30_m1 * (1.0f - timestep));

    // Calculate warped optical flow positions
    float2 uv_m1_flow = uv - float2(flow_t_f30_p1 * timestep);
    float2 uv_p1_flow = uv + float2(flow_t_f30_p1 * (1.0f - timestep));

    // Calculate OOB areas
    bool uv_p1_mv_oob   = (uv_p1_mv.x <= 0.0 || uv_p1_mv.x >= 1.0) || (uv_p1_mv.y <= 0.0 || uv_p1_mv.y >= 1.0);
    bool uv_m1_mv_oob   = (uv_m1_mv.x <= 0.0 || uv_m1_mv.x >= 1.0) || (uv_m1_mv.y <= 0.0 || uv_m1_mv.y >= 1.0);
    bool uv_m1_flow_oob = (uv_m1_flow.x <= 0.0 || uv_m1_flow.x >= 1.0) || (uv_m1_flow.y <= 0.0 || uv_m1_flow.y >= 1.0);
    bool uv_p1_flow_oob = (uv_p1_flow.x <= 0.0 || uv_p1_flow.x >= 1.0) || (uv_p1_flow.y <= 0.0 || uv_p1_flow.y >= 1.0);

    // Calculate double dissoclusion mask
    float depth_p1_warp_m1_warp_t = LoadInputHolesTm1(uv_m1_mv);
    float hole_mask_m1            = float(depth_p1_warp_m1_warp_t != 0.0f) * (1.0 - float(uv_p1_mv_oob));

    float depth_p1_warp_t = LoadInputHolesTp1(iMvCoord);
    float hole_mask_t     = depth_p1_warp_t * (1.0 - float(uv_p1_mv_oob));

    half dd_mask = half((hole_mask_t - hole_mask_m1) < 0.0f);

    // Warp colour using gather approach with warped vectors

    FFX_MIN16_F3 rgb_m1_warp_t_mv   = FFX_MIN16_F3(0.0f);
    FFX_MIN16_F3 rgb_p1_warp_t_mv   = FFX_MIN16_F3(0.0f);
    FFX_MIN16_F3 rgb_m1_warp_t_flow = FFX_MIN16_F3(0.0f);
    FFX_MIN16_F3 rgb_p1_warp_t_flow = FFX_MIN16_F3(0.0f);

    if (!uv_m1_mv_oob)
        rgb_m1_warp_t_mv = LoadPreviousBackbuffer(int32_t2(uv_m1_mv * DisplaySize()));
    if (!uv_p1_mv_oob)
        rgb_p1_warp_t_mv = LoadCurrentBackbuffer(int32_t2(uv_p1_mv * DisplaySize()));
    if (!uv_m1_flow_oob)
        rgb_m1_warp_t_flow = LoadPreviousBackbuffer(int32_t2(uv_m1_flow * DisplaySize()));
    if (!uv_p1_flow_oob)
        rgb_p1_warp_t_flow = LoadCurrentBackbuffer(int32_t2(uv_p1_flow * DisplaySize()));

    // Randomize OOB RGB values to prevent network from learning spurious correlations
    // Uses spatial hash (y*10000+z) combined with random_seed for temporal variation
    uint32_t seed_base = uint32_t(output_pixel.x) * 10000u + uint32_t(output_pixel.y);

#if !(DISABLE_RANDOM_OOB)
    if (uv_m1_mv_oob)
    {
        rgb_m1_warp_t_mv = FFX_MIN16_F3(hash_to_half(seed_base, 0u), hash_to_half(seed_base, 1u), hash_to_half(seed_base, 2u));
    }
    if (uv_p1_mv_oob)
    {
        rgb_p1_warp_t_mv = FFX_MIN16_F3(hash_to_half(seed_base, 3u), hash_to_half(seed_base, 4u), hash_to_half(seed_base, 5u));
    }
    if (uv_m1_flow_oob)
    {
        rgb_m1_warp_t_flow = FFX_MIN16_F3(hash_to_half(seed_base, 6u), hash_to_half(seed_base, 7u), hash_to_half(seed_base, 8u));
    }
    if (uv_p1_flow_oob)
    {
        rgb_p1_warp_t_flow = FFX_MIN16_F3(hash_to_half(seed_base, 9u), hash_to_half(seed_base, 10u), hash_to_half(seed_base, 11u));
    }
#endif

    // Warp depth
    float2   depth_tm1_size     = DepthTm1Size();
    float2   depth_tp1_size     = DepthTp1Size();
    int32_t2 depth_m1_pixel     = int32_t2(floor(uv_m1_mv * depth_tm1_size));
    int32_t2 depth_p1_pixel     = int32_t2(floor(uv_p1_mv * depth_tp1_size));
    float    depth_m1_warp_t_mv = LoadInputDepthTm1Pixel(depth_m1_pixel);
    float    depth_p1_warp_t_mv = LoadInputDepthTp1Pixel(depth_p1_pixel);

    // OOB depth should be near plane value
    if (!IsOnScreen(depth_m1_pixel, int32_t2(depth_tm1_size)))
    {
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH
        depth_m1_warp_t_mv = 1.0f;
#else
        depth_m1_warp_t_mv = 0.0f;
#endif
    }

    if (!IsOnScreen(depth_p1_pixel, int32_t2(depth_tp1_size)))
    {
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH
        depth_p1_warp_t_mv = 1.0f;
#else
        depth_p1_warp_t_mv = 0.0f;
#endif
    }

    // Normalize Depth
    float  depth_m1_warp_t_mv_norm = NormalizeDepth(depth_m1_warp_t_mv);
    float  depth_p1_warp_t_mv_norm = NormalizeDepth(depth_p1_warp_t_mv);
    float2 depth                   = float2(depth_m1_warp_t_mv_norm, depth_p1_warp_t_mv_norm);

    // Calculate disocclusion mask
    FFX_MIN16_F2 dis_mask = SingleTapCalculateDisocclusionMask(depth_m1_warp_t_mv, depth_p1_warp_t_mv, uv_m1_mv, uv_p1_mv);

    // Encode double disocclusion mask as dis_mask.x = dis_mask.y = 1
    dis_mask.x = ffxSaturate(dis_mask.x + dd_mask);
    dis_mask.y = ffxSaturate(dis_mask.y + dd_mask);

    // Write Tensor
    WriteToInputTensor(output_pixel, rgb_m1_warp_t_mv, rgb_p1_warp_t_mv, rgb_m1_warp_t_flow, rgb_p1_warp_t_flow, dis_mask, depth);
}

#endif