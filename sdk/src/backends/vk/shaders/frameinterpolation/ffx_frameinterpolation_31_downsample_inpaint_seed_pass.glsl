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

// Seed -> inpaint logical level 0.
#define FFX_FRAMEINTERPOLATION_INPAINT_FROM_SEED 1

// SRV bindings
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPAINT_SEED 0

// UAV bindings
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_INPAINT_DESTINATION 1

// CB binding
#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION 2

#include "frameinterpolation/ffx_frameinterpolation_31_downsample_inpaint_pass.h"

layout(local_size_x = FFX_FRAMEINTERPOLATION_INPAINT_GROUP_WIDTH,
       local_size_y = FFX_FRAMEINTERPOLATION_INPAINT_GROUP_HEIGHT,
       local_size_z = FFX_FRAMEINTERPOLATION_INPAINT_GROUP_DEPTH) in;
void main()
{
    downsample_inpaint(int32_t2(gl_GlobalInvocationID.xy));
}
