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
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_SUPPORT_TENSOR
#extension GL_ARM_tensors : require
#endif

//-----------------------------------------------------------------------------
// Input: SRV bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_MOTION_TP1          0
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_FLOW_TM1            1
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREVIOUS_INTERPOLATION_SOURCE   2
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_CURRENT_INTERPOLATION_SOURCE    3
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TP1                 4
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TM1                 5
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_HOLES_TP1                 6
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_HOLES_TM1                 7

//-----------------------------------------------------------------------------
// Output: UAV bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_IN_TENSOR                       8

#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION               9

#include "frameinterpolation/ffx_frameinterpolation_30_preprocess_pass.h"

//------------------------------------------------------------------------------

#ifndef FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS
#define FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS layout (local_size_x = FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH, local_size_y = FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT, local_size_z = FFX_FRAMEINTERPOLATION_THREAD_GROUP_DEPTH) in;
#endif // #ifndef FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS

FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS
void main()
{
    preprocess(int32_t2(gl_GlobalInvocationID.xy));
}