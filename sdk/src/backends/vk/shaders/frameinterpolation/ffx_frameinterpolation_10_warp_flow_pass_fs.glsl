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

//-----------------------------------------------------------------------------
// Input: SRV bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_MOTION_TP1            0
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TP1             1
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TM1             2
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DYNAMIC_MASK_TM1      3

//-----------------------------------------------------------------------------
// Output: RT bindings + UAV bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_MOTION_QDATA_TP1       4
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_HOLES_TP1                   5
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_HOLES_TM1                   6
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_MANAGE_PREVIOUS_DEPTH
#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DEPTH_TM1_NEXT    0
#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DYNAMIC_MASK_TP1  1
#else
#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DYNAMIC_MASK_TP1  0
#endif

#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION           8

#include "frameinterpolation/ffx_frameinterpolation_10_warp_flow_pass.h"


// entry-point
void main()
{
    warp_flow(int32_t2(gl_FragCoord.xy));
}