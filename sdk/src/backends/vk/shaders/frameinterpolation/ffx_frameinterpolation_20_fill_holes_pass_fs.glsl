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
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_MOTION_QDATA_TP1            0

//-----------------------------------------------------------------------------
// Output: RT bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_WARP_FILLED_MOTION_TP1 0

#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION                1

#include "frameinterpolation/ffx_frameinterpolation_20_fill_holes_pass.h"

// entry-point
void main()
{
    fill_holes(int32_t2(gl_FragCoord.xy));
}
