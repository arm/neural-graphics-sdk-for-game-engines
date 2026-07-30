// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#version 460
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_GOOGLE_include_directive : require

#if FFX_HALF
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#endif

#define NSS_BIND_SRV_INPUT_MOTION_VECTORS            0   // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS
#define NSS_BIND_SRV_INPUT_DEPTH                     1   // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH
#define NSS_BIND_SRV_INPUT_DEPTH_TM1                 2   // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH_TM1

#define NSS_BIND_RENDER_TARGET_DISOCCLUSION_MASK_LQ  0   // FFX_NSS_RESOURCE_IDENTIFIER_DISOCCLUSION_MASK_LQ

#define NSS_BIND_CB_NSS                              4

#include "nss/ffx_nss_disocclusion_mask_lq.h"

void main()
{
    DisocclusionMaskLq(int32_t2(gl_FragCoord.xy));
}
