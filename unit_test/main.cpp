/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "framework/test_framework.h"
#include "framework/vk_test_utils.h"

#include <array>
#include <filesystem>
#include <iostream>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace
{
    std::filesystem::path GetExecutablePath(const std::string& fallbackPath)
    {
#if defined(_WIN32)
        std::array<wchar_t, 32768> pathBuffer{};
        const DWORD                pathLength = GetModuleFileNameW(nullptr, pathBuffer.data(), static_cast<DWORD>(pathBuffer.size()));
        if (pathLength != 0 && pathLength < pathBuffer.size())
        {
            return std::filesystem::path(pathBuffer.data());
        }
#elif defined(__linux__)
        std::error_code error;
        const auto      executablePath = std::filesystem::read_symlink("/proc/self/exe", error);
        if (!error)
        {
            return executablePath;
        }
#endif
        return fallbackPath;
    }
}  // namespace

int main(int argc, char** argv)
{
    const unit_test::RunnerOptions options        = unit_test::ParseOptions(argc, argv);
    const std::filesystem::path    executablePath = GetExecutablePath(options.executablePath);
    unit_test::vk::SetShaderDirectory((executablePath.parent_path() / "shader").string());

    if (options.showHelp)
    {
        unit_test::PrintHelp();
        return 0;
    }

    if (options.hasError)
    {
        std::cerr << "Error: " << options.errorMessage << '\n';
        unit_test::PrintHelp();
        return 1;
    }

    if (!options.internalRunName.empty())
    {
        return unit_test::RunInternal(options.internalRunName);
    }

    return unit_test::RunAll(options);
}
