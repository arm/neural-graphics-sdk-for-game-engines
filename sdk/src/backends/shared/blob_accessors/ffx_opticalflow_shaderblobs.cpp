// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include <FidelityFX/host/ffx_util.h>
#include "ffx_opticalflow_shaderblobs.h"
#include "opticalflow/ffx_opticalflow_private.h"

#include <ffx_opticalflow_mv_hints_permutations.h>
#include <ffx_opticalflow_mv_hints_fs_permutations.h>
#include <ffx_opticalflow_mv_hints_vs_permutations.h>
#include <string.h>  // for memset

#if defined(POPULATE_PERMUTATION_KEY)
#undef POPULATE_PERMUTATION_KEY
#endif  // #if defined(POPULATE_PERMUTATION_KEY)
#define POPULATE_PERMUTATION_KEY(options, key) \
    key.index          = 0;                    \
    key.DEPTH_INVERTED = FFX_CONTAINS_FLAG(options, OPTICALFLOW_SHADER_PERMUTATION_DEPTH_INVERTED);

static FfxShaderBlob opticalflowGetComputeMotionFromDepthPassPermutationBlobByIndex(uint32_t permutationOptions)
{
    ffx_opticalflow_mv_hints_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    const bool useFragmentJob = FFX_CONTAINS_FLAG(permutationOptions, OPTICALFLOW_SHADER_PERMUTATION_MV_HINTS_FRAGMENT);
    if (useFragmentJob)
    {
        const int32_t tableIndex = g_ffx_opticalflow_mv_hints_fs_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_mv_hints_fs_PermutationInfo, tableIndex);
    }

    {
        const int32_t tableIndex = g_ffx_opticalflow_mv_hints_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_mv_hints_PermutationInfo, tableIndex);
    }
}

static FfxShaderBlob opticalflowGetVertexPermutationBlobByIndex(uint32_t permutationOptions)
{
    ffx_opticalflow_mv_hints_vs_PermutationKey key;
    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    const int32_t tableIndex = g_ffx_opticalflow_mv_hints_vs_IndirectionTable[key.index];
    return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_mv_hints_vs_PermutationInfo, tableIndex);
}

FfxErrorCode opticalflowGetPermutationBlobByIndex(FfxOpticalflowPass passId, uint32_t permutationOptions, FfxShaderBlob* outBlob, FfxShaderBlob* outVertBlob)
{
    if (outVertBlob)
    {
        FfxShaderBlob blob = opticalflowGetVertexPermutationBlobByIndex(permutationOptions);
        memcpy(outVertBlob, &blob, sizeof(FfxShaderBlob));
    }

    switch (passId)
    {
    case FFX_OPTICALFLOW_PASS_COMPUTE_MOTION_FROM_DEPTH:
    {
        FfxShaderBlob blob = opticalflowGetComputeMotionFromDepthPassPermutationBlobByIndex(permutationOptions);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }

    default:
        FFX_ASSERT_MESSAGE(false, "Not implemented");
        break;
    }

    // return an empty blob
    memset(outBlob, 0, sizeof(FfxShaderBlob));
    return FFX_OK;
}

FfxErrorCode opticalflowIsWave64(uint32_t permutationOptions, bool& isWave64)
{
    isWave64 = false;
    return FFX_OK;
}