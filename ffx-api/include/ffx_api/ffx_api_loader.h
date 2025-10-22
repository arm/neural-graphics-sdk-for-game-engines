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
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "ffx_api.h"
#ifdef _WIN32
#include <windows.h>
typedef HMODULE FfxModuleHandle;
#else
#include <dlfcn.h>
typedef void* FfxModuleHandle;
#endif

typedef struct ffxFunctions
{
    PfnFfxCreateContext  CreateContext;
    PfnFfxDestroyContext DestroyContext;
    PfnFfxConfigure      Configure;
    PfnFfxQuery          Query;
    PfnFfxDispatch       Dispatch;
} ffxFunctions;

inline static void* ffxGetProcAddress(FfxModuleHandle module, const char* name)
{
#ifdef _WIN32
    return (void*)GetProcAddress(module, name);
#else
    return dlsym(module, name);
#endif
}

static inline void ffxLoadFunctions(ffxFunctions* pOutFunctions, FfxModuleHandle module)
{
    pOutFunctions->CreateContext  = (PfnFfxCreateContext)ffxGetProcAddress(module, "ffxCreateContext");
    pOutFunctions->DestroyContext = (PfnFfxDestroyContext)ffxGetProcAddress(module, "ffxDestroyContext");
    pOutFunctions->Configure      = (PfnFfxConfigure)ffxGetProcAddress(module, "ffxConfigure");
    pOutFunctions->Query          = (PfnFfxQuery)ffxGetProcAddress(module, "ffxQuery");
    pOutFunctions->Dispatch       = (PfnFfxDispatch)ffxGetProcAddress(module, "ffxDispatch");
}
