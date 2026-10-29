/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#version 450
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_GOOGLE_include_directive : enable

#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_OF_COLOUR_P1                       0
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TP1                    1
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_MOTION_TP1                   2

#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_PREPROCESS_COLOUR_P1     0
#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DEPTH_TM1_NEXT           1
#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_REDUCED_MOTION_P1        2

#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION                  3

#include "frameinterpolation/ffx_frameinterpolation_02_downsample_all_input_pass.h"

void main()
{
    downsample_all_input(int32_t2(gl_FragCoord.xy));
}
