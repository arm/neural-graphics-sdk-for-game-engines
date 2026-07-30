// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#version 460
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_GOOGLE_include_directive : require
#if NSS_SUPPORT_TENSOR
#extension GL_ARM_tensors : require
#endif

#if FFX_HALF
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#endif

// TM1 means Time Minus 1, i.e. previous frame

//----------------------------------------------------------------------------- 
// Input: SRV bindings
//----------------------------------------------------------------------------- 
#define NSS_BIND_SRV_INPUT_COLOR_JITTERED               0   // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_COLOR
#define NSS_BIND_SRV_INPUT_DEPTH                        1   // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH
#define NSS_BIND_SRV_INPUT_MOTION_VECTORS               2   // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS
#define NSS_BIND_SRV_HISTORY_UPSCALED_COLOR             3   // FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_UPSCALED_COLOR
#define NSS_BIND_SRV_FEEDBACK_TENSOR                    4   // FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR
#define NSS_BIND_SRV_INPUT_DEPTH_TM1                    5   // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH_TM1
#define NSS_BIND_SRV_LUMA_DERIV_TM1                     6   // FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_TM1
#define NSS_BIND_SRV_DISOCCLUSION_MASK_LQ               7   // FFX_NSS_RESOURCE_IDENTIFIER_DISOCCLUSION_MASK_LQ

//----------------------------------------------------------------------------- 
// Output: RT bindings + conditional tensor or buffer aliasing for tensor write
//----------------------------------------------------------------------------- 
#define NSS_BIND_PREPROCESS_INPUT_TENSOR                8   // rw_preprocessed_tensor (FFX_NSS_RESOURCE_IDENTIFIER_PREPROCESS_INPUT_TENSOR)
#define NSS_BIND_RENDER_TARGET_LUMA_DERIV               0   // rw_luma_deriv (FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV)
#define NSS_BIND_RENDER_TARGET_NEAREST_DEPTH_COORD      1   // rw_nearest_depth_coord_out (FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD)

#define NSS_BIND_CB_NSS                                 10

#define NSS_PREPROCESS 1

#include "nss/ffx_nss_preprocess.h"

void main()
{
    rw_luma_deriv = vec4(float4(EmptyDerivativeStateForStorage()));
    rw_nearest_depth_coord_out = vec4(0.0, 0.0, 0.0, 1.0);
    Preprocess(int32_t2(gl_FragCoord.xy));
}
