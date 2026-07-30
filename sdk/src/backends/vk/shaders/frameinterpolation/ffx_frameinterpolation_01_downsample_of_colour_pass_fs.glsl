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

// SRV bindings
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_CURRENT_INTERPOLATION_SOURCE 0

// RT bindings
#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_COLOUR_P1_INTERNAL 0

// CB binding
#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION 1

#include "frameinterpolation/ffx_frameinterpolation_01_downsample_of_colour_pass.h"

void main()
{
    downsample_of_colour(int32_t2(gl_FragCoord.xy));
}
