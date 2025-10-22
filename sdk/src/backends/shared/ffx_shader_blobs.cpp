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

#include "ffx_shader_blobs.h"

#if defined(FFX_NSS) || defined(FFX_ALL)
#include "blob_accessors/ffx_nss_shaderblobs.h"
#endif  // #if defined(FFX_NSS) || defined(FFX_ALL)

#include <string.h>  // for memset

FfxErrorCode ffxGetPermutationBlobByIndex(
    FfxEffect effectId, FfxPass passId, uint32_t permutationOptions, FfxShaderBlob* outBlob, FfxShaderBlob* outVertBlob, FfxDataGraphBlob* outDataGraphBlob)
{
    switch (effectId)
    {
#if defined(FFX_NSS) || defined(FFX_ALL)
    case FFX_EFFECT_NSS:
        return nssGetPermutationBlobByIndex((FfxNssPass)passId, permutationOptions, outBlob, outDataGraphBlob);
#endif  // #if defined(FFX_NSS) || defined(FFX_ALL)

    default:
        FFX_ASSERT_MESSAGE(false, "Not implemented");
        break;
    }

    // return an empty blob
    memset(outBlob, 0, sizeof(FfxShaderBlob));
    return FFX_OK;
}

FfxErrorCode ffxIsWave64(FfxEffect effectId, uint32_t permutationOptions, bool& isWave64)
{
    switch (effectId)
    {
#if defined(FFX_NSS) || defined(FFX_ALL)
    case FFX_EFFECT_NSS:
        return FFX_OK;
#endif  // #if defined(FFX_NSS) || defined(FFX_ALL)

    default:
        FFX_ASSERT_MESSAGE(false, "Not implemented");
        isWave64 = false;
        break;
    }

    return FFX_ERROR_BACKEND_API_ERROR;
}
