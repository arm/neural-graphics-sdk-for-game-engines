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
// Output: UAV bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_MOTION_QDATA_TP1   0
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_FLOW_QDATA_TM1    1
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_HOLES_TP1               2
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_HOLES_TM1               3

#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION       4

#include "frameinterpolation/ffx_frameinterpolation_00_init_warp_pass.h"

// entry-point
#ifndef FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS
#define FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS layout (local_size_x = FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH, local_size_y = FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT, local_size_z = FFX_FRAMEINTERPOLATION_THREAD_GROUP_DEPTH) in;
#endif // #ifndef FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS

FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS
void main()
{
    init_warp(int32_t2(gl_GlobalInvocationID.xy));
}
