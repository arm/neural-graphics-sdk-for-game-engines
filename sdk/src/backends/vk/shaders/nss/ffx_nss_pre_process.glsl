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

// TM1 means Time Minus 1, i.e. previous frame

//-----------------------------------------------------------------------------
// Input: SRV bindings
//-----------------------------------------------------------------------------
#define NSS_BIND_SRV_INPUT_COLOR_JITTERED               0   // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_COLOR
#define NSS_BIND_SRV_INPUT_DEPTH                        1   // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH
#define NSS_BIND_SRV_INPUT_MOTION_VECTORS               2   // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS
#define NSS_BIND_SRV_HISTORY_UPSCALED_COLOR             3   // FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_UPSCALED_COLOR
#define NSS_BIND_SRV_FEEDBACK_TENSOR                    4   // FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR
#define NSS_BIND_SRV_INPUT_DEPTH_TM1                    5   // FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH_TM1
#define NSS_BIND_SRV_LUMA_DERIV_TM1                     6   // FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_TM1
#define NSS_BIND_SRV_DISOCCLUSION_MASK_LQ               7   // FFX_NSS_RESOURCE_IDENTIFIER_DISOCCLUSION_MASK_LQ


//-----------------------------------------------------------------------------
// Output: UAV bindings
//-----------------------------------------------------------------------------
// layout (set=1, binding=0)       uniform writeonly tensorARM<int8_t, 4> _PreprocessTensor;         // 540p  | 12ch 96bpp 
// layout (set=1, binding=1, rgba8_snorm) uniform writeonly lowp image2D  _PreProcessLumaDerivOut;   // 540p  | R8G8B8A8_SNORM 32bpp
// layout (set=1, binding=3, r8)   uniform writeonly lowp image2D         _NearestDepthCoordOut;     // 540p  | R8 8bpp (TODO: update to `R8_UINT` when scenario runner supports it)
#define NSS_BIND_PREPROCESS_INPUT_TENSOR                8   // rw_preprocessed_tensor (FFX_NSS_RESOURCE_IDENTIFIER_PREPROCESS_INPUT_TENSOR)
#define NSS_BIND_UAV_LUMA_DERIV                         9   // rw_luma_deriv (FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV)
#define NSS_BIND_UAV_NEAREST_DEPTH_COORD                10  // rw_nearest_depth_coord_tm1 (FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD)
#define NSS_BIND_UAV_DEPTH_TM1                          11

#define NSS_BIND_CB_NSS                                 12

#define NSS_PREPROCESS 1

#include "nss/ffx_nss_preprocess.h"

#ifndef FFX_NSS_NUM_THREADS
#define FFX_NSS_NUM_THREADS layout (local_size_x = FFX_NSS_THREAD_GROUP_WIDTH, local_size_y = FFX_NSS_THREAD_GROUP_HEIGHT, local_size_z = FFX_NSS_THREAD_GROUP_DEPTH) in;
#endif // FFX_NSS_NUM_THREADS

FFX_NSS_NUM_THREADS
void main()
{
    Preprocess(int32_t2(gl_GlobalInvocationID.xy));
}
