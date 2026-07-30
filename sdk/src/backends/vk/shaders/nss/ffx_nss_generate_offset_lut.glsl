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

#define NSS_BIND_CB_NSS 11
#define NSS_BIND_UAV_OFFSET_LUT 0

#include "nss/ffx_nss_generate_offset_lut.h"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main()
{
    GenerateOffsetLut(int32_t(gl_GlobalInvocationID.x));
}
