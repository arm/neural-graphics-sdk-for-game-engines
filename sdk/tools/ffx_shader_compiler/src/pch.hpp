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

#define NOMINMAX
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <atlcomcli.h>
#include <dxcapi.h>
#include <d3dcompiler.h>
#include <d3d12shader.h>
#else
// On Linux, define Windows types and macros as needed
#include <cstdint>

typedef int      BOOL;
typedef uint8_t  BYTE;
typedef uint32_t DWORD;
typedef uint32_t UINT;
typedef int      HRESULT;
typedef void*    HMODULE;

// Minimal stub for CComPtr (does nothing, just for compilation)
template <typename T>
class CComPtr
{
public:
    CComPtr()
        : ptr(nullptr)
    {
    }
    CComPtr(T* p)
        : ptr(p)
    {
    }
    T* operator->()
    {
        return ptr;
    }
    T** operator&()
    {
        return &ptr;
    }
    operator bool() const
    {
        return ptr != nullptr;
    }
    T* ptr;
};

// LoadLibrary stub (returns nullptr)
inline HMODULE LoadLibraryA(const char*)
{
    return nullptr;
}
inline HMODULE LoadLibraryW(const wchar_t*)
{
    return nullptr;
}

// Other Windows-specific macros
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif
#endif
#include <assert.h>
#include <stdio.h>
#include <exception>
#include <string>
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <iomanip>
#include <bitset>
#include <iostream>
#include <algorithm>
#include <fstream>

#include <filesystem>
namespace fs = std::filesystem;

#include <process.hpp>
namespace tpl = TinyProcessLib;
