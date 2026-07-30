// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

//-----------------------------------------------------------------------------
// Input: SRV bindings
//-----------------------------------------------------------------------------
#define NSS_BIND_SRV_INPUT_COLOR_JITTERED               0    // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_COLOR
#define NSS_BIND_SRV_INPUT_MOTION_VECTORS               1    // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS
#define NSS_BIND_SRV_HISTORY_UPSCALED_COLOR             2    // FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_UPSCALED_COLOR
#define NSS_BIND_KPN_TENSOR                             3    // FFX_NSS_RESOURCE_IDENTIFIER_KPN_TENSOR
#define NSS_BIND_SRV_FEEDBACK_TENSOR                    4    // FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR
#define NSS_BIND_SRV_NEAREST_DEPTH_COORD                5    // FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD
#define NSS_BIND_SRV_OFFSET_LUT                         6   // FFX_NSS_RESOURCE_IDENTIFIER_OFFSET_LUT

//-----------------------------------------------------------------------------
// Output: UAV bindings
//-----------------------------------------------------------------------------
#define NSS_BIND_UAV_UPSCALED_OUTPUT                    7    // FFX_NSS_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT
#define NSS_BIND_UAV_HISTORY_UPSCALED_COLOR             8   // FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_UPSCALED_COLOR

#define NSS_BIND_CB_NSS                                 9

#define NSS_POSTPROCESS 1

#include "nss/ffx_nss_postprocess.h"

#ifndef FFX_NSS_NUM_THREADS
#define FFX_NSS_NUM_THREADS layout (local_size_x = FFX_NSS_THREAD_GROUP_WIDTH, local_size_y = FFX_NSS_THREAD_GROUP_HEIGHT, local_size_z = FFX_NSS_THREAD_GROUP_DEPTH) in;
#endif // FFX_NSS_NUM_THREADS

FFX_NSS_NUM_THREADS
void main()
{
    Postprocess(int32_t2(gl_GlobalInvocationID.xy));
}
