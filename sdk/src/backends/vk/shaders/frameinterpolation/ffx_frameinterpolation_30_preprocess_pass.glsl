/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_SUPPORT_TENSOR
#extension GL_ARM_tensors : require
#endif

//-----------------------------------------------------------------------------
// Input: SRV bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_MOTION_QDATA_TP1          0
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_FLOW_TM1            1
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREPROCESS_COLOUR_M1            2
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREPROCESS_COLOUR_P1            3
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_DEPTH_TP1_FEEDBACK               4
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TM1                 5

//-----------------------------------------------------------------------------
// Output: UAV bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_IN_TENSOR                       8
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_INPAINT_SEED                    10

#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION               9

#include "frameinterpolation/ffx_frameinterpolation_30_preprocess_pass.h"

//------------------------------------------------------------------------------

layout(local_size_x = FFX_FRAMEINTERPOLATION_PREPROCESS_THREAD_GROUP_WIDTH,
       local_size_y = FFX_FRAMEINTERPOLATION_PREPROCESS_THREAD_GROUP_HEIGHT,
       local_size_z = FFX_FRAMEINTERPOLATION_PREPROCESS_THREAD_GROUP_DEPTH) in;
void main()
{
    preprocess(int32_t2(gl_GlobalInvocationID.xy));
}