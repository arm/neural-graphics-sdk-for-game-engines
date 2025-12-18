// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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


#define NSS_BIND_SRV_UNPADDED_COLOR     0  // FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_COLOR
#define NSS_BIND_SRV_UNPADDED_DEPTH     1  // FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_DEPTH
#define NSS_BIND_SRV_UNPADDED_MOTION    2  // FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_MOTION
#define NSS_BIND_SRV_UNPADDED_DEPTH_TM1 3  // FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_DEPTH_TM1
#define NSS_BIND_UAV_PADDED_COLOR     4  // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_COLOR
#define NSS_BIND_UAV_PADDED_DEPTH     5  // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH
#define NSS_BIND_UAV_PADDED_MOTION    6  // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS
#define NSS_BIND_UAV_PADDED_DEPTH_TM1 7  // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH_TM1

#define NSS_BIND_CB_NSS                                 8

#include "nss/ffx_nss_callbacks_glsl.h"
#include "nss/ffx_nss_mirror_padding.h"

#ifndef FFX_NSS_THREAD_GROUP_WIDTH
#define FFX_NSS_THREAD_GROUP_WIDTH 16
#endif // FFX_NSS_THREAD_GROUP_WIDTH
#ifndef FFX_NSS_THREAD_GROUP_HEIGHT
#define FFX_NSS_THREAD_GROUP_HEIGHT 16
#endif // FFX_NSS_THREAD_GROUP_HEIGHT
#ifndef FFX_NSS_THREAD_GROUP_DEPTH
#define FFX_NSS_THREAD_GROUP_DEPTH 1
#endif // FFX_NSS_THREAD_GROUP_DEPTH
#ifndef FFX_NSS_NUM_THREADS
#define FFX_NSS_NUM_THREADS layout (local_size_x = FFX_NSS_THREAD_GROUP_WIDTH, local_size_y = FFX_NSS_THREAD_GROUP_HEIGHT, local_size_z = FFX_NSS_THREAD_GROUP_DEPTH) in;
#endif // FFX_NSS_NUM_THREADS

FFX_NSS_NUM_THREADS
void main()
{
    ApplyMirrorPadding(int32_t2(gl_GlobalInvocationID.xy));
}
