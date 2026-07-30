// Copyright  © 2023 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "ffx_nss_shaderblobs.h"
#include "nss/ffx_nss_private.h"
#include <FidelityFX/host/ffx_util.h>

#include <nss_v1_0_1_high_int8.h>
#include <nss_v1_0_1_mid_low_int8.h>

#include <ffx_nss_pre_process_16bit_permutations.h>
#include <ffx_nss_pre_process_permutations.h>
#include <ffx_nss_pre_process_fs_16bit_permutations.h>
#include <ffx_nss_pre_process_fs_permutations.h>

#include <ffx_nss_post_process_16bit_permutations.h>
#include <ffx_nss_post_process_permutations.h>
#include <ffx_nss_post_process_fs_16bit_permutations.h>
#include <ffx_nss_post_process_fs_permutations.h>

#include <ffx_nss_depth_scatter_16bit_permutations.h>
#include <ffx_nss_depth_scatter_permutations.h>

#include <ffx_nss_disocclusion_mask_lq_16bit_permutations.h>
#include <ffx_nss_disocclusion_mask_lq_permutations.h>
#include <ffx_nss_disocclusion_mask_lq_fs_16bit_permutations.h>
#include <ffx_nss_disocclusion_mask_lq_fs_permutations.h>

#include <ffx_nss_generate_offset_lut_16bit_permutations.h>
#include <ffx_nss_generate_offset_lut_permutations.h>

#include <ffx_nss_debug_view_16bit_permutations.h>
#include <ffx_nss_debug_view_permutations.h>
#include <ffx_nss_debug_view_fs_16bit_permutations.h>
#include <ffx_nss_debug_view_fs_permutations.h>

#include <ffx_nss_vs_16bit_permutations.h>
#include <ffx_nss_vs_permutations.h>

#include <string.h>  // for memset

#if defined(POPULATE_PERMUTATION_KEY)
#undef POPULATE_PERMUTATION_KEY
#endif  // #if defined(POPULATE_PERMUTATION_KEY)
#define POPULATE_PERMUTATION_KEY(options, key)                                                                             \
    key.index                   = 0;                                                                                       \
    key.REVERSE_Z               = FFX_CONTAINS_FLAG(options, NSS_SHADER_PERMUTATION_REVERSE_Z);                            \
    key.RESAMPLE_BICUBIC        = FFX_CONTAINS_FLAG(options, NSS_SHADER_PERMUTATION_RESAMPLE_BICUBIC);                     \
    key.MANAGE_HISTORY          = FFX_CONTAINS_FLAG(options, NSS_SHADER_PERMUTATION_MANAGE_HISTORY);                       \
    key.NSS_SHADER_QUALITY_MODE = ((options) >> NSS_SHADER_PERMUTATION_QUALITY_MODE_SHIFT) & NSS_SHADER_QUALITY_MODE_MASK; \
    key.NSS_SUPPORT_TENSOR      = 0;

static FfxShaderBlob nssGetDepthScatterPassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    ffx_nss_depth_scatter_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);
    // Depth scatter is always compute job
    key.NSS_SUPPORT_TENSOR = FFX_CONTAINS_FLAG(permutationOptions, NSS_SHADER_PERMUTATION_COMPUTE_SUPPORT_TENSOR);

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_nss_depth_scatter_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_depth_scatter_16bit_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_nss_depth_scatter_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_depth_scatter_PermutationInfo, tableIndex);
    }
}

static FfxShaderBlob nssGetPreprocessPassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    const bool useFragmentJob   = (permutationOptions & NSS_SHADER_PERMUTATION_PRE_PROCESS_FRAGMENT) == NSS_SHADER_PERMUTATION_PRE_PROCESS_FRAGMENT;
    const auto supportTensorBit = useFragmentJob ? NSS_SHADER_PERMUTATION_FRAGMENT_SUPPORT_TENSOR : NSS_SHADER_PERMUTATION_COMPUTE_SUPPORT_TENSOR;
    ffx_nss_pre_process_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);
    key.NSS_SUPPORT_TENSOR = FFX_CONTAINS_FLAG(permutationOptions, supportTensorBit);

    if (useFragmentJob)
    {
        if (is16bit)
        {
            const int32_t tableIndex = g_ffx_nss_pre_process_fs_16bit_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_pre_process_fs_16bit_PermutationInfo, tableIndex);
        }
        else
        {
            const int32_t tableIndex = g_ffx_nss_pre_process_fs_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_pre_process_fs_PermutationInfo, tableIndex);
        }
    }

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_nss_pre_process_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_pre_process_16bit_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_nss_pre_process_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_pre_process_PermutationInfo, tableIndex);
    }
}

static FfxShaderBlob nssGetDisocclusionMaskLqPassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    const bool useFragmentJob   = (permutationOptions & NSS_SHADER_PERMUTATION_PRE_PROCESS_FRAGMENT) == NSS_SHADER_PERMUTATION_PRE_PROCESS_FRAGMENT;
    const auto supportTensorBit = useFragmentJob ? NSS_SHADER_PERMUTATION_FRAGMENT_SUPPORT_TENSOR : NSS_SHADER_PERMUTATION_COMPUTE_SUPPORT_TENSOR;
    ffx_nss_disocclusion_mask_lq_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);
    key.NSS_SUPPORT_TENSOR = FFX_CONTAINS_FLAG(permutationOptions, supportTensorBit);

    if (useFragmentJob)
    {
        if (is16bit)
        {
            const int32_t tableIndex = g_ffx_nss_disocclusion_mask_lq_fs_16bit_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_disocclusion_mask_lq_fs_16bit_PermutationInfo, tableIndex);
        }

        const int32_t tableIndex = g_ffx_nss_disocclusion_mask_lq_fs_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_disocclusion_mask_lq_fs_PermutationInfo, tableIndex);
    }

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_nss_disocclusion_mask_lq_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_disocclusion_mask_lq_16bit_PermutationInfo, tableIndex);
    }

    const int32_t tableIndex = g_ffx_nss_disocclusion_mask_lq_IndirectionTable[key.index];
    return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_disocclusion_mask_lq_PermutationInfo, tableIndex);
}

static FfxShaderBlob nssGetPostprocessPassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    const bool useFragmentJob   = (permutationOptions & NSS_SHADER_PERMUTATION_POST_PROCESS_FRAGMENT) == NSS_SHADER_PERMUTATION_POST_PROCESS_FRAGMENT;
    const auto supportTensorBit = useFragmentJob ? NSS_SHADER_PERMUTATION_FRAGMENT_SUPPORT_TENSOR : NSS_SHADER_PERMUTATION_COMPUTE_SUPPORT_TENSOR;
    ffx_nss_post_process_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);
    key.NSS_SUPPORT_TENSOR = FFX_CONTAINS_FLAG(permutationOptions, supportTensorBit);

    key.SCALE_PRESET_MODE = FFX_CONTAINS_FLAG(permutationOptions, NSS_SHADER_PERMUTATION_SCALE_PRESET_X2) ? NSS_SCALE_PRESET_X2 : NSS_SCALE_PRESET_NONE;
    FFX_ASSERT(key.SCALE_PRESET_MODE < NSS_SCALE_PRESET_COUNT);

    if (useFragmentJob)
    {
        if (is16bit)
        {
            const int32_t tableIndex = g_ffx_nss_post_process_fs_16bit_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_post_process_fs_16bit_PermutationInfo, tableIndex);
        }
        else
        {
            const int32_t tableIndex = g_ffx_nss_post_process_fs_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_post_process_fs_PermutationInfo, tableIndex);
        }
    }

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_nss_post_process_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_post_process_16bit_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_nss_post_process_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_post_process_PermutationInfo, tableIndex);
    }
}

static FfxShaderBlob nssGetGenerateOffsetLutPassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    ffx_nss_generate_offset_lut_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);
    key.SCALE_PRESET_MODE = FFX_CONTAINS_FLAG(permutationOptions, NSS_SHADER_PERMUTATION_SCALE_PRESET_X2) ? NSS_SCALE_PRESET_X2 : NSS_SCALE_PRESET_NONE;
    FFX_ASSERT(key.SCALE_PRESET_MODE < NSS_SCALE_PRESET_COUNT);

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_nss_generate_offset_lut_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_generate_offset_lut_16bit_PermutationInfo, tableIndex);
    }

    const int32_t tableIndex = g_ffx_nss_generate_offset_lut_IndirectionTable[key.index];
    return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_generate_offset_lut_PermutationInfo, tableIndex);
}

static FfxShaderBlob nssGetDebugViewPassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    const bool useFragmentJob   = (permutationOptions & NSS_SHADER_PERMUTATION_POST_PROCESS_FRAGMENT) == NSS_SHADER_PERMUTATION_POST_PROCESS_FRAGMENT;
    const auto supportTensorBit = useFragmentJob ? NSS_SHADER_PERMUTATION_FRAGMENT_SUPPORT_TENSOR : NSS_SHADER_PERMUTATION_COMPUTE_SUPPORT_TENSOR;

    ffx_nss_debug_view_PermutationKey key;
    POPULATE_PERMUTATION_KEY(permutationOptions, key);
    key.NSS_SUPPORT_TENSOR = FFX_CONTAINS_FLAG(permutationOptions, supportTensorBit);

    if (useFragmentJob)
    {
        if (is16bit)
        {
            const int32_t tableIndex = g_ffx_nss_debug_view_fs_16bit_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_debug_view_fs_16bit_PermutationInfo, tableIndex);
        }
        else
        {
            const int32_t tableIndex = g_ffx_nss_debug_view_fs_IndirectionTable[key.index];
            return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_debug_view_fs_PermutationInfo, tableIndex);
        }
    }

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_nss_debug_view_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_debug_view_16bit_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_nss_debug_view_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_debug_view_PermutationInfo, tableIndex);
    }
}

static FfxShaderBlob nssGetGeneralVertexPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    ffx_nss_vs_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_nss_vs_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_vs_16bit_PermutationInfo, tableIndex);
    }

    const int32_t tableIndex = g_ffx_nss_vs_IndirectionTable[key.index];
    return POPULATE_SHADER_BLOB_FFX(g_ffx_nss_vs_PermutationInfo, tableIndex);
}

FfxErrorCode nssGetPermutationBlobByIndex(
    FfxNssPass passId, uint32_t permutationOptions, FfxShaderBlob* outShaderBlob, FfxShaderBlob* outVertBlob, FfxDataGraphBlob* outDataGraphBlob)
{
    const bool is16bit = FFX_CONTAINS_FLAG(permutationOptions, NSS_SHADER_PERMUTATION_ALLOW_16BIT);
    const auto shaderQuality =
        static_cast<FfxNssShaderQualityMode>(((permutationOptions) >> NSS_SHADER_PERMUTATION_QUALITY_MODE_SHIFT) & NSS_SHADER_QUALITY_MODE_MASK);

    if (outVertBlob)
    {
        FfxShaderBlob blob = nssGetGeneralVertexPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outVertBlob, &blob, sizeof(FfxShaderBlob));
    }

    switch (passId)
    {
    case FFX_NSS_PASS_DEPTH_SCATTER:
    {
        FfxShaderBlob blob = nssGetDepthScatterPassPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outShaderBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }

    case FFX_NSS_PASS_PREPROCESS:
    {
        FfxShaderBlob blob = nssGetPreprocessPassPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outShaderBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }

    case FFX_NSS_PASS_DISOCCLUSION_MASK:
    {
        FfxShaderBlob blob = nssGetDisocclusionMaskLqPassPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outShaderBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }

    case FFX_NSS_PASS_DATA_GRAPH:
    {
        // Select the v1.0 VGF based on quality mode: quality mode uses the HQ graph,
        // other modes use the LQ (sparse KPN16) graph.
        const bool isQualityMode = shaderQuality == FFX_NSS_SHADER_QUALITY_MODE_QUALITY;
        if (isQualityMode)
        {
            memcpy(outDataGraphBlob, &g_nss_v1_0_1_high_int8_Info, sizeof(FfxDataGraphBlob));
        }
        else
        {
            memcpy(outDataGraphBlob, &g_nss_v1_0_1_mid_low_int8_Info, sizeof(FfxDataGraphBlob));
        }
        return FFX_OK;
    }

    case FFX_NSS_PASS_POSTPROCESS:
    {
        FfxShaderBlob blob = nssGetPostprocessPassPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outShaderBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }

    case FFX_NSS_PASS_GENERATE_OFFSET_LUT:
    {
        FfxShaderBlob blob = nssGetGenerateOffsetLutPassPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outShaderBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }

    case FFX_NSS_PASS_DEBUG_VIEW:
    {
        FfxShaderBlob blob = nssGetDebugViewPassPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outShaderBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }

    default:
        FFX_ASSERT_FAIL("Should never reach here.");
        break;
    }

    return FFX_OK;
}
