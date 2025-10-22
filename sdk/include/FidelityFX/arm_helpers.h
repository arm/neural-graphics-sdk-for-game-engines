/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#ifdef __clang__
#include <wchar.h>

inline int wcscpy_s(wchar_t* dest, size_t destsz, const wchar_t* src)
{
    PRAGMA_DISABLE_DEPRECATION_WARNINGS
    if (wcslen(src) < destsz)
    {
        wcscpy(dest, src);
        return 0;
    }
    else
        return 1;
    PRAGMA_ENABLE_DEPRECATION_WARNINGS
}

template <size_t size>
inline int wcscpy_s(wchar_t (&dest)[size], const wchar_t* src)
{
    return wcscpy_s(dest, size, src);
}

#endif
