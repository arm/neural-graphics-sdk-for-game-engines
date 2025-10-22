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

#include "pch.hpp"

std::string  WCharToUTF8(const std::wstring& wstr);
std::wstring UTF8ToWChar(const std::string& str);

#if !_WIN32

#include <cstdio>
#include <string>
#include <locale>
#include <codecvt>

// Converts std::wstring (UTF-16/32) to UTF-8 std::string
inline std::string WStringToUtf8(const std::wstring& wstr)
{
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
    return conv.to_bytes(wstr);
}

// Linux replacement for _wfopen_s
inline int _wfopen_s(FILE** file, const wchar_t* filename, const wchar_t* mode)
{
    if (!file)
        return -1;
    std::string filename_utf8 = WStringToUtf8(filename);
    std::string mode_utf8     = WStringToUtf8(mode);
    *file                     = fopen(filename_utf8.c_str(), mode_utf8.c_str());
    return *file ? 0 : errno;
}

#endif