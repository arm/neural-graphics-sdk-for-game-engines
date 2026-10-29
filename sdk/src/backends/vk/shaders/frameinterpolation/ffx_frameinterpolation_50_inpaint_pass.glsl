/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */
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
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_LEVEL_0 0
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_LEVEL_1 1
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_LEVEL_2 2
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_LEVEL_3 3
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_LEVEL_4 4

// UAV bindings
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_INPAINT_RESULT 5

// CB binding
#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION 6

#include "frameinterpolation/ffx_frameinterpolation_50_inpaint_pass.h"

layout(local_size_x = FFX_FRAMEINTERPOLATION_INPAINT_GROUP_WIDTH,
       local_size_y = FFX_FRAMEINTERPOLATION_INPAINT_GROUP_HEIGHT,
       local_size_z = FFX_FRAMEINTERPOLATION_INPAINT_GROUP_DEPTH) in;
void main()
{
    inpaint_resolve(int32_t2(gl_GlobalInvocationID.xy));
}
