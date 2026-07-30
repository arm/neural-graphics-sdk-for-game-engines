// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#version 460
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_GOOGLE_include_directive : require
#if NSS_SUPPORT_TENSOR
#extension GL_ARM_tensors : require
#endif

#if FFX_HALF
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#endif

//-----------------------------------------------------------------------------
// Input: SRV bindings
//-----------------------------------------------------------------------------
#define NSS_BIND_SRV_INPUT_COLOR_JITTERED    0    // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_COLOR
#define NSS_BIND_SRV_INPUT_MOTION_VECTORS    1    // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS
#define NSS_BIND_SRV_HISTORY_UPSCALED_COLOR  2    // FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_UPSCALED_COLOR
#define NSS_BIND_SRV_INPUT_DEPTH             3    // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH
#define NSS_BIND_SRV_INPUT_DEPTH_TM1         4    // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH_TM1 (r_prev_depth, R32UI)
#define NSS_BIND_SRV_LUMA_DERIV_TM1          5    // FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_TM1
#define NSS_BIND_SRV_NEAREST_DEPTH_COORD     6    // FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD
#define NSS_BIND_SRV_FEEDBACK_TENSOR         7    // FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR (4-ch SNORM texture)
#define NSS_BIND_SRV_DISOCCLUSION_MASK_LQ    8    // FFX_NSS_RESOURCE_IDENTIFIER_DISOCCLUSION_MASK_LQ
#define NSS_BIND_SRV_LUMA_DERIV              9    // FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV (current-frame luma deriv)

//-----------------------------------------------------------------------------
// Input: preprocess-input tensor (buffer/tensor alias)
//-----------------------------------------------------------------------------
#define NSS_BIND_PREPROCESS_INPUT_TENSOR     10   // FFX_NSS_RESOURCE_IDENTIFIER_PREPROCESS_INPUT_TENSOR

//-----------------------------------------------------------------------------
// Output: UAV binding (compute path)
//-----------------------------------------------------------------------------
#define NSS_BIND_UAV_DEBUG_VIEWS             11   // FFX_NSS_RESOURCE_IDENTIFIER_DEBUG_VIEWS

#define NSS_BIND_CB_NSS                      12

#include "nss/ffx_nss_debug_view.h"

#ifndef FFX_NSS_NUM_THREADS
#define FFX_NSS_NUM_THREADS layout (local_size_x = FFX_NSS_THREAD_GROUP_WIDTH, local_size_y = FFX_NSS_THREAD_GROUP_HEIGHT, local_size_z = FFX_NSS_THREAD_GROUP_DEPTH) in;
#endif  // FFX_NSS_NUM_THREADS

FFX_NSS_NUM_THREADS
void main()
{
    DebugViewCS(int32_t2(gl_GlobalInvocationID.xy));
}
