// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#define FFX_CPU
#include <FidelityFX/host/ffx_util.h>
#include "ffx_frameinterpolation_shaderblobs.h"
#include "frameinterpolation/ffx_frameinterpolation_private.h"

// Data Graph
#include <nfru_v1_int8.h>

// Normal permutations
#include <ffx_frameinterpolation_00_init_warp_pass_permutations.h>
#include <ffx_frameinterpolation_01_downsample_of_colour_pass_fs_permutations.h>
#include <ffx_frameinterpolation_10_warp_flow_pass_permutations.h>
#include <ffx_frameinterpolation_10_warp_flow_pass_fs_permutations.h>
#include <ffx_frameinterpolation_11_warp_flow_tm1_pass_permutations.h>
#include <ffx_frameinterpolation_11_warp_flow_tm1_pass_fs_permutations.h>
#include <ffx_frameinterpolation_20_fill_holes_pass_permutations.h>
#include <ffx_frameinterpolation_20_fill_holes_pass_fs_permutations.h>
#include <ffx_frameinterpolation_21_fill_holes_tm1_pass_permutations.h>
#include <ffx_frameinterpolation_21_fill_holes_tm1_pass_fs_permutations.h>
#include <ffx_frameinterpolation_30_preprocess_pass_permutations.h>
#include <ffx_frameinterpolation_30_preprocess_pass_fs_permutations.h>
#include <ffx_frameinterpolation_50_postprocess_pass_permutations.h>
#include <ffx_frameinterpolation_50_postprocess_pass_fs_permutations.h>
#include <ffx_frameinterpolation_debug_view_pass_permutations.h>
#include <ffx_frameinterpolation_debug_view_pass_fs_permutations.h>
#include <ffx_frameinterpolation_vs_permutations.h>

// 16bit permutations
#include <ffx_frameinterpolation_00_init_warp_pass_16bit_permutations.h>
#include <ffx_frameinterpolation_01_downsample_of_colour_pass_fs_16bit_permutations.h>
#include <ffx_frameinterpolation_10_warp_flow_pass_16bit_permutations.h>
#include <ffx_frameinterpolation_10_warp_flow_pass_fs_16bit_permutations.h>
#include <ffx_frameinterpolation_11_warp_flow_tm1_pass_16bit_permutations.h>
#include <ffx_frameinterpolation_11_warp_flow_tm1_pass_fs_16bit_permutations.h>
#include <ffx_frameinterpolation_20_fill_holes_pass_16bit_permutations.h>
#include <ffx_frameinterpolation_20_fill_holes_pass_fs_16bit_permutations.h>
#include <ffx_frameinterpolation_21_fill_holes_tm1_pass_16bit_permutations.h>
#include <ffx_frameinterpolation_21_fill_holes_tm1_pass_fs_16bit_permutations.h>
#include <ffx_frameinterpolation_30_preprocess_pass_16bit_permutations.h>
#include <ffx_frameinterpolation_30_preprocess_pass_fs_16bit_permutations.h>
#include <ffx_frameinterpolation_50_postprocess_pass_16bit_permutations.h>
#include <ffx_frameinterpolation_50_postprocess_pass_fs_16bit_permutations.h>
#include <ffx_frameinterpolation_debug_view_pass_16bit_permutations.h>
#include <ffx_frameinterpolation_debug_view_pass_fs_16bit_permutations.h>
#include <ffx_frameinterpolation_vs_16bit_permutations.h>
#include <string.h>  // for memset

#if defined(POPULATE_PERMUTATION_KEY)
#undef POPULATE_PERMUTATION_KEY
#endif  // #if defined(POPULATE_PERMUTATION_KEY)
#define POPULATE_PERMUTATION_KEY(options, key)                                                                                                               \
    key.index                                                    = 0;                                                                                        \
    key.FFX_ARM_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS = FFX_CONTAINS_FLAG(options, FRAMEINTERPOLATION_SHADER_PERMUTATION_LOW_RES_MOTION_VECTORS); \
    key.FFX_ARM_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS  = FFX_CONTAINS_FLAG(options, FRAMEINTERPOLATION_SHADER_PERMUTATION_JITTER_MOTION_VECTORS);  \
    key.FFX_ARM_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH         = FFX_CONTAINS_FLAG(options, FRAMEINTERPOLATION_SHADER_PERMUTATION_DEPTH_INVERTED);         \
    key.FFX_ARM_FRAMEINTERPOLATION_OPTION_ENABLE_DATA_GRAPH_FI   = FFX_CONTAINS_FLAG(options, FRAMEINTERPOLATION_SHADER_PERMUTATION_ENABLE_DATA_GRAPH_FI);   \
    key.FFX_ARM_FRAMEINTERPOLATION_OPTION_SUPPORT_TENSOR         = 0;                                                                                        \
    key.FFX_ARM_FRAMEINTERPOLATION_OPTION_MANAGE_PREVIOUS_DEPTH  = FFX_CONTAINS_FLAG(options, FRAMEINTERPOLATION_SHADER_PERMUTATION_MANAGE_PREVIOUS_DEPTH);  \
    key.FFX_ARM_FRAMEINTERPOLATION_OPTION_MANAGE_PREVIOUS_COLOR  = FFX_CONTAINS_FLAG(options, FRAMEINTERPOLATION_SHADER_PERMUTATION_MANAGE_PREVIOUS_COLOR);

// Init warp (compute only, no fragment variant)
static FfxShaderBlob armFrameInterpolationGetInitWarpPassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    ffx_frameinterpolation_00_init_warp_pass_PermutationKey key;
    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_00_init_warp_pass_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_00_init_warp_pass_16bit_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_00_init_warp_pass_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_00_init_warp_pass_PermutationInfo, tableIndex);
    }
}

// Downsample OF colour (fragment-only)
static FfxShaderBlob armFrameInterpolationGetDownsampleOfColourPassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    ffx_frameinterpolation_01_downsample_of_colour_pass_fs_PermutationKey key;
    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_01_downsample_of_colour_pass_fs_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_01_downsample_of_colour_pass_fs_16bit_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_01_downsample_of_colour_pass_fs_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_01_downsample_of_colour_pass_fs_PermutationInfo, tableIndex);
    }
}

// Warp motion TP1
static FfxShaderBlob armFrameInterpolationGetWarpFlowPassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    ffx_frameinterpolation_10_warp_flow_pass_PermutationKey key;
    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    const bool useFragmentJob =
        (permutationOptions & FRAMEINTERPOLATION_SHADER_PERMUTATION_WARP_FRAGMENT) == FRAMEINTERPOLATION_SHADER_PERMUTATION_WARP_FRAGMENT;
    if (useFragmentJob)
    {
        if (is16bit)
        {
            const int32_t tableIndex = g_ffx_frameinterpolation_10_warp_flow_pass_fs_16bit_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_10_warp_flow_pass_fs_16bit_PermutationInfo, tableIndex);
        }
        else
        {
            const int32_t tableIndex = g_ffx_frameinterpolation_10_warp_flow_pass_fs_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_10_warp_flow_pass_fs_PermutationInfo, tableIndex);
        }
    }

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_10_warp_flow_pass_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_10_warp_flow_pass_16bit_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_10_warp_flow_pass_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_10_warp_flow_pass_PermutationInfo, tableIndex);
    }
}

// Warp flow TM1
static FfxShaderBlob armFrameInterpolationGetWarpFlowTM1PassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    ffx_frameinterpolation_11_warp_flow_tm1_pass_PermutationKey key;
    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    const bool useFragmentJob =
        (permutationOptions & FRAMEINTERPOLATION_SHADER_PERMUTATION_WARP_FRAGMENT) == FRAMEINTERPOLATION_SHADER_PERMUTATION_WARP_FRAGMENT;
    if (useFragmentJob)
    {
        if (is16bit)
        {
            const int32_t tableIndex = g_ffx_frameinterpolation_11_warp_flow_tm1_pass_fs_16bit_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_11_warp_flow_tm1_pass_fs_16bit_PermutationInfo, tableIndex);
        }
        else
        {
            const int32_t tableIndex = g_ffx_frameinterpolation_11_warp_flow_tm1_pass_fs_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_11_warp_flow_tm1_pass_fs_PermutationInfo, tableIndex);
        }
    }

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_11_warp_flow_tm1_pass_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_11_warp_flow_tm1_pass_16bit_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_11_warp_flow_tm1_pass_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_11_warp_flow_tm1_pass_PermutationInfo, tableIndex);
    }
}

// Fill motion hole TP1
static FfxShaderBlob armFrameInterpolationGetFillHolesPassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    ffx_frameinterpolation_20_fill_holes_pass_PermutationKey key;
    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    const bool useFragmentJob =
        (permutationOptions & FRAMEINTERPOLATION_SHADER_PERMUTATION_FILL_HOLES_FRAGMENT) == FRAMEINTERPOLATION_SHADER_PERMUTATION_FILL_HOLES_FRAGMENT;
    if (useFragmentJob)
    {
        if (is16bit)
        {
            const int32_t tableIndex = g_ffx_frameinterpolation_20_fill_holes_pass_fs_16bit_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_20_fill_holes_pass_fs_16bit_PermutationInfo, tableIndex);
        }
        else
        {
            const int32_t tableIndex = g_ffx_frameinterpolation_20_fill_holes_pass_fs_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_20_fill_holes_pass_fs_PermutationInfo, tableIndex);
        }
    }

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_20_fill_holes_pass_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_20_fill_holes_pass_16bit_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_20_fill_holes_pass_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_20_fill_holes_pass_PermutationInfo, tableIndex);
    }
}

// Fill flow TM1
static FfxShaderBlob armFrameInterpolationGetFillHolesTM1PassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    ffx_frameinterpolation_21_fill_holes_tm1_pass_PermutationKey key;
    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    const bool useFragmentJob =
        (permutationOptions & FRAMEINTERPOLATION_SHADER_PERMUTATION_FILL_HOLES_FRAGMENT) == FRAMEINTERPOLATION_SHADER_PERMUTATION_FILL_HOLES_FRAGMENT;
    if (useFragmentJob)
    {
        if (is16bit)
        {
            const int32_t tableIndex = g_ffx_frameinterpolation_21_fill_holes_tm1_pass_fs_16bit_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_21_fill_holes_tm1_pass_fs_16bit_PermutationInfo, tableIndex);
        }
        else
        {
            const int32_t tableIndex = g_ffx_frameinterpolation_21_fill_holes_tm1_pass_fs_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_21_fill_holes_tm1_pass_fs_PermutationInfo, tableIndex);
        }
    }

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_21_fill_holes_tm1_pass_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_21_fill_holes_tm1_pass_16bit_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_21_fill_holes_tm1_pass_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_21_fill_holes_tm1_pass_PermutationInfo, tableIndex);
    }
}

static FfxShaderBlob armFrameInterpolationGetPreprocessPassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    ffx_frameinterpolation_30_preprocess_pass_PermutationKey key;
    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    const bool useFragmentJob =
        (permutationOptions & FRAMEINTERPOLATION_SHADER_PERMUTATION_PRE_PROCESS_FRAGMENT) == FRAMEINTERPOLATION_SHADER_PERMUTATION_PRE_PROCESS_FRAGMENT;
    const auto supportTensorBit =
        useFragmentJob ? FRAMEINTERPOLATION_SHADER_PERMUTATION_FRAGMENT_SUPPORT_TENSOR : FRAMEINTERPOLATION_SHADER_PERMUTATION_COMPUTE_SUPPORT_TENSOR;
    key.FFX_ARM_FRAMEINTERPOLATION_OPTION_SUPPORT_TENSOR = FFX_CONTAINS_FLAG(permutationOptions, supportTensorBit);
    if (useFragmentJob)
    {
        if (is16bit)
        {
            const int32_t tableIndex = g_ffx_frameinterpolation_30_preprocess_pass_fs_16bit_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_30_preprocess_pass_fs_16bit_PermutationInfo, tableIndex);
        }
        else
        {
            const int32_t tableIndex = g_ffx_frameinterpolation_30_preprocess_pass_fs_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_30_preprocess_pass_fs_PermutationInfo, tableIndex);
        }
    }

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_30_preprocess_pass_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_30_preprocess_pass_16bit_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_30_preprocess_pass_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_30_preprocess_pass_PermutationInfo, tableIndex);
    }
}

static FfxShaderBlob armFrameInterpolationGetPostprocessPassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    ffx_frameinterpolation_50_postprocess_pass_PermutationKey key;
    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    const bool useFragmentJob =
        (permutationOptions & FRAMEINTERPOLATION_SHADER_PERMUTATION_POSTPROCESS_FRAGMENT) == FRAMEINTERPOLATION_SHADER_PERMUTATION_POSTPROCESS_FRAGMENT;
    if (useFragmentJob)
    {
        if (is16bit)
        {
            const int32_t tableIndex = g_ffx_frameinterpolation_50_postprocess_pass_fs_16bit_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_50_postprocess_pass_fs_16bit_PermutationInfo, tableIndex);
        }
        else
        {
            const int32_t tableIndex = g_ffx_frameinterpolation_50_postprocess_pass_fs_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_50_postprocess_pass_fs_PermutationInfo, tableIndex);
        }
    }

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_50_postprocess_pass_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_50_postprocess_pass_16bit_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_50_postprocess_pass_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_50_postprocess_pass_PermutationInfo, tableIndex);
    }
}

static FfxShaderBlob armFrameInterpolationGetDebugViewPassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    ffx_frameinterpolation_debug_view_pass_PermutationKey key;
    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    const bool useFragmentJob =
        (permutationOptions & FRAMEINTERPOLATION_SHADER_PERMUTATION_DEBUG_VIEW_FRAGMENT) == FRAMEINTERPOLATION_SHADER_PERMUTATION_DEBUG_VIEW_FRAGMENT;
    const auto supportTensorBit =
        useFragmentJob ? FRAMEINTERPOLATION_SHADER_PERMUTATION_FRAGMENT_SUPPORT_TENSOR : FRAMEINTERPOLATION_SHADER_PERMUTATION_COMPUTE_SUPPORT_TENSOR;
    key.FFX_ARM_FRAMEINTERPOLATION_OPTION_SUPPORT_TENSOR = FFX_CONTAINS_FLAG(permutationOptions, supportTensorBit);
    if (useFragmentJob)
    {
        if (is16bit)
        {
            const int32_t tableIndex = g_ffx_frameinterpolation_debug_view_pass_fs_16bit_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_debug_view_pass_fs_16bit_PermutationInfo, tableIndex);
        }
        else
        {
            const int32_t tableIndex = g_ffx_frameinterpolation_debug_view_pass_fs_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_debug_view_pass_fs_PermutationInfo, tableIndex);
        }
    }

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_debug_view_pass_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_debug_view_pass_16bit_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_debug_view_pass_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_debug_view_pass_PermutationInfo, tableIndex);
    }
}

static FfxShaderBlob armFrameInterpolationGetGeneralVertexPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    ffx_frameinterpolation_vs_PermutationKey key;
    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_vs_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_vs_16bit_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_frameinterpolation_vs_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_frameinterpolation_vs_PermutationInfo, tableIndex);
    }
}

FfxErrorCode frameInterpolationGetPermutationBlobByIndex(
    FfxFrameInterpolationPass passId, uint32_t permutationOptions, FfxShaderBlob* outBlob, FfxShaderBlob* outVertBlob, FfxDataGraphBlob* outDataGraphBlob)
{
    bool is16bit = FFX_CONTAINS_FLAG(permutationOptions, FRAMEINTERPOLATION_SHADER_PERMUTATION_ALLOW_FP16);

    if (outVertBlob)
    {
        FfxShaderBlob blob = armFrameInterpolationGetGeneralVertexPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outVertBlob, &blob, sizeof(FfxShaderBlob));
    }

    switch (passId)
    {
    case FFX_FRAMEINTERPOLATION_PASS_INIT_WARP:
    {
        FfxShaderBlob blob = armFrameInterpolationGetInitWarpPassPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }

    case FFX_FRAMEINTERPOLATION_PASS_DOWNSAMPLE_OF_COLOUR:
    {
        FfxShaderBlob blob = armFrameInterpolationGetDownsampleOfColourPassPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }

    case FFX_FRAMEINTERPOLATION_PASS_WARP_MV_TP1:
    {
        FfxShaderBlob blob = armFrameInterpolationGetWarpFlowPassPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }

    case FFX_FRAMEINTERPOLATION_PASS_FILL_HOLES_TP1:
    {
        FfxShaderBlob blob = armFrameInterpolationGetFillHolesPassPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }

    case FFX_FRAMEINTERPOLATION_PASS_WARP_FLOW_TM1:
    {
        FfxShaderBlob blob = armFrameInterpolationGetWarpFlowTM1PassPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }

    case FFX_FRAMEINTERPOLATION_PASS_FILL_HOLES_TM1:
    {
        FfxShaderBlob blob = armFrameInterpolationGetFillHolesTM1PassPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }

    case FFX_FRAMEINTERPOLATION_PASS_PREPROCESS:
    {
        FfxShaderBlob blob = armFrameInterpolationGetPreprocessPassPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }

    case FFX_FRAMEINTERPOLATION_PASS_POSTPROCESS:
    {
        FfxShaderBlob blob = armFrameInterpolationGetPostprocessPassPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }
    case FFX_FRAMEINTERPOLATION_PASS_NFRU_INTERPOLATION:
    {
        memcpy(outDataGraphBlob, &g_nfru_v1_int8_Info, sizeof(FfxDataGraphBlob));
        return FFX_OK;
    }
    case FFX_FRAMEINTERPOLATION_PASS_DEBUG_VIEW:
    {
        FfxShaderBlob blob = armFrameInterpolationGetDebugViewPassPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }
    default:
        FFX_ASSERT_FAIL("Should never reach here.");
        break;
    }

    // return an empty blob
    memset(outBlob, 0, sizeof(FfxShaderBlob));
    return FFX_OK;
}
