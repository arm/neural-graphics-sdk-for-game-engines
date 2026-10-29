/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdio>
#include <cstdint>
#include <string>

namespace arm::capture
{
    // Creates the directory containing a file path. Dataset layout remains owned
    // by the provider/payload; this module only performs portable filesystem work.
    bool EnsureParentDirectory(const std::string& filePath);

    // Creates an output root and rejects an existing directory so a capture
    // cannot accidentally append to a previous dataset.
    bool EnsureUniqueOutputDirectory(std::string& outputDirectory);

    // Opens a file for writing after ensuring its parent directory exists. The
    // caller owns the returned handle and must close it with std::fclose.
    FILE* OpenFileForWrite(const std::string& filePath, const char* mode);

    // Writes a complete binary artifact after ensuring its parent directory exists.
    bool WriteBinaryFile(const std::string& filePath, const void* data, uint64_t size);

}  // namespace arm::capture
