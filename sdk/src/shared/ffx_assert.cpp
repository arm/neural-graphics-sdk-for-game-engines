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
/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include <FidelityFX/host/ffx_assert.h>
#include "ffx_logging.h"

#include <stdio.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

static constexpr const char* ffxLogComponentName()
{
    return "Assert";
}
static const uint32_t MAX_ASSERT_BUFFER_SIZE = 512;

static FfxAssertCallback s_assertCallback;

// set the printing callback function
void ffxAssertSetPrintingCallback(FfxAssertCallback callback)
{
    s_assertCallback = callback;
    return;
}

// implementation of assert reporting
bool ffxAssertReport(const char* file, int32_t line, const char* condition, const char* message)
{
    if (!file)
    {
        return true;
    }

    // Keep assertion reporting allocation-free so it remains usable in failure paths.
    char        tempBuf[MAX_ASSERT_BUFFER_SIZE] = {};
    const char* detail                          = message ? message : (condition ? condition : "unspecified assertion");
    snprintf(tempBuf, sizeof(tempBuf), "%s(%d): ASSERTION FAILED. %s", file, line, detail);

    if (s_assertCallback)
    {
        s_assertCallback(tempBuf);
    }
    else
    {
#ifdef _WIN32
        // For Visual Studio Debugging on Windows
        OutputDebugStringA(tempBuf);
#endif
        ffxLogMessage(ffxLogComponentName(), FFX_MESSAGE_TYPE_ERROR, tempBuf);
    }

    return true;
}
