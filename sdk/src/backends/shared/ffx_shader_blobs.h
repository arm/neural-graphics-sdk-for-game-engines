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

#pragma once

#include <FidelityFX/host/ffx_interface.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif  // #if defined(__cplusplus)

struct FfxShaderBlob;

// Get a shader blob for the specified effect, pass, and permutation index.
FfxErrorCode ffxGetPermutationBlobByIndex(FfxEffect         effectId,
                                          FfxPass           passId,
                                          uint32_t          permutationOptions,
                                          FfxShaderBlob*    outBlob,
                                          FfxShaderBlob*    outVertBlob      = nullptr,
                                          FfxDataGraphBlob* outDataGraphBlob = nullptr);

// Check is Wave64 is requested on this permutation
FfxErrorCode ffxIsWave64(FfxEffect effectId, uint32_t permutationOptions, bool& isWave64);

#if defined(__cplusplus)
}
#endif  // #if defined(__cplusplus)
