// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#version 450
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
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_OUT_PARAMS_TENSOR               0
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_MOTION_TP1          1
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_FLOW_TM1            2
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_MOTION_TM1          3
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREVIOUS_INTERPOLATION_SOURCE   4
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_CURRENT_INTERPOLATION_SOURCE    5

//-----------------------------------------------------------------------------
// Input: RT bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_OUTPUT                0
// Additional render target used when the SDK (not the application) owns the
// previous-interpolation-source ping-pong buffer; the postprocess pass writes it
// here so it can be sampled as colorTm1 next frame.
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_MANAGE_PREVIOUS_COLOR
#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_PREVIOUS_INTERPOLATION_SOURCE     1
#endif

#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION               6


#include "frameinterpolation/ffx_frameinterpolation_50_postprocess_pass.h"


// entry-point
void main()
{
    postprocess(int32_t2(gl_FragCoord.xy));
}