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
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "ffx_nss_shaderblobs.h"
#include "nss/ffx_nss_private.h"
#include <FidelityFX/host/ffx_util.h>

#include <nss_v0_1_1_int8.h>

#include <ffx_nss_mirror_padding_16bit_permutations.h>
#include <ffx_nss_mirror_padding_permutations.h>

#include <ffx_nss_pre_process_16bit_permutations.h>
#include <ffx_nss_pre_process_permutations.h>

#include <ffx_nss_post_process_16bit_permutations.h>
#include <ffx_nss_post_process_permutations.h>

#include <ffx_nss_debug_view_16bit_permutations.h>
#include <ffx_nss_debug_view_permutations.h>

#include <string.h>  // for memset

#if defined(POPULATE_PERMUTATION_KEY)
#undef POPULATE_PERMUTATION_KEY
#endif  // #if defined(POPULATE_PERMUTATION_KEY)
#define POPULATE_PERMUTATION_KEY(options, key)                                                                \
    key.index                          = 0;                                                                   \
    key.REVERSE_Z                      = FFX_CONTAINS_FLAG(options, NSS_SHADER_PERMUTATION_REVERSE_Z);        \
    key.RESAMPLE_BICUBIC               = FFX_CONTAINS_FLAG(options, NSS_SHADER_PERMUTATION_RESAMPLE_BICUBIC); \
    key.ALIAS_OUTPUT_TENSORS_AS_IMAGES = FFX_CONTAINS_FLAG(options, NSS_SHADER_PERMUTATION_ALIAS_OUTPUT_TENSORS_AS_IMAGES);

static FfxShaderBlob nssGetMirrorPaddingPassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    ffx_nss_mirror_padding_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_nss_mirror_padding_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX_TENSOR(g_ffx_nss_mirror_padding_16bit_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_nss_mirror_padding_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX_TENSOR(g_ffx_nss_mirror_padding_PermutationInfo, tableIndex);
    }
}

static FfxShaderBlob nssGetPreprocessPassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    ffx_nss_pre_process_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_nss_pre_process_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX_TENSOR(g_ffx_nss_pre_process_16bit_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_nss_pre_process_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX_TENSOR(g_ffx_nss_pre_process_PermutationInfo, tableIndex);
    }
}

static FfxShaderBlob nssGetPostprocessPassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    ffx_nss_post_process_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    // Match with difinitions in ffx_nss_private.h
    // #define SCALE_1_3X 1
    // #define SCALE_1_5X 2
    // #define SCALE_2_0X 3
    if ((permutationOptions & NSS_SHADER_PERMUTATION_SCALE_PRESET_MODE_X1_3) == NSS_SHADER_PERMUTATION_SCALE_PRESET_MODE_X1_3)
    {
        key.SCALE_PRESET_MODE = 1;
    }
    else if ((permutationOptions & NSS_SHADER_PERMUTATION_SCALE_PRESET_MODE_X1_5) == NSS_SHADER_PERMUTATION_SCALE_PRESET_MODE_X1_5)
    {
        key.SCALE_PRESET_MODE = 2;
    }
    else if ((permutationOptions & NSS_SHADER_PERMUTATION_SCALE_PRESET_MODE_X2) == NSS_SHADER_PERMUTATION_SCALE_PRESET_MODE_X2)
    {
        key.SCALE_PRESET_MODE = 3;
    }
    else
    {
        key.SCALE_PRESET_MODE = 0;  // No LUT method.
    }

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_nss_post_process_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX_TENSOR(g_ffx_nss_post_process_16bit_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_nss_post_process_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX_TENSOR(g_ffx_nss_post_process_PermutationInfo, tableIndex);
    }
}

static FfxShaderBlob nssGetDebugViewPassPermutationBlobByIndex(uint32_t permutationOptions, bool is16bit)
{
    ffx_nss_debug_view_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    // Match with difinitions in ffx_nss_private.h
    // #define SCALE_1_3X 1
    // #define SCALE_1_5X 2
    // #define SCALE_2_0X 3
    if ((permutationOptions & NSS_SHADER_PERMUTATION_SCALE_PRESET_MODE_X1_3) == NSS_SHADER_PERMUTATION_SCALE_PRESET_MODE_X1_3)
    {
        key.SCALE_PRESET_MODE = 1;
    }
    else if ((permutationOptions & NSS_SHADER_PERMUTATION_SCALE_PRESET_MODE_X1_5) == NSS_SHADER_PERMUTATION_SCALE_PRESET_MODE_X1_5)
    {
        key.SCALE_PRESET_MODE = 2;
    }
    else if ((permutationOptions & NSS_SHADER_PERMUTATION_SCALE_PRESET_MODE_X2) == NSS_SHADER_PERMUTATION_SCALE_PRESET_MODE_X2)
    {
        key.SCALE_PRESET_MODE = 3;
    }
    else
    {
        key.SCALE_PRESET_MODE = 0;  // No LUT method.
    }

    if (is16bit)
    {
        const int32_t tableIndex = g_ffx_nss_debug_view_16bit_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX_TENSOR(g_ffx_nss_debug_view_16bit_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_nss_debug_view_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX_TENSOR(g_ffx_nss_debug_view_PermutationInfo, tableIndex);
    }
}

FfxErrorCode nssGetPermutationBlobByIndex(FfxNssPass passId, uint32_t permutationOptions, FfxShaderBlob* outShaderBlob, FfxDataGraphBlob* outDataGraphBlob)
{
    const bool is16bit = FFX_CONTAINS_FLAG(permutationOptions, NSS_SHADER_PERMUTATION_ALLOW_16BIT);

    switch (passId)
    {
    case FFX_NSS_PASS_MIRROR_PADDING:
    {
        FfxShaderBlob blob = nssGetMirrorPaddingPassPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outShaderBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }

    case FFX_NSS_PASS_PREPROCESS:
    {
        FfxShaderBlob blob = nssGetPreprocessPassPermutationBlobByIndex(permutationOptions, is16bit);
        memcpy(outShaderBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }

    case FFX_NSS_PASS_DATA_GRAPH:
    {
        memcpy(outDataGraphBlob, &g_nss_v0_1_1_int8_Info, sizeof(FfxDataGraphBlob));
        return FFX_OK;
    }

    case FFX_NSS_PASS_POSTPROCESS:
    {
        FfxShaderBlob blob = nssGetPostprocessPassPermutationBlobByIndex(permutationOptions, is16bit);
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
