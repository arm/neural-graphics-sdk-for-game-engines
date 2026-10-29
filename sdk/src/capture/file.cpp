/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "file.h"
#include "log.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <new>
#include <random>
#include <system_error>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

namespace arm::capture
{
    namespace
    {
        bool IsPathSeparator(char character)
        {
            return character == '/' || character == '\\';
        }

        std::string GenerateUniqueDirectorySuffix()
        {
            std::random_device random;
            char               suffix[17];
            std::snprintf(suffix, sizeof(suffix), "%08x%08x", static_cast<unsigned int>(random()), static_cast<unsigned int>(random()));
            return suffix;
        }

        bool CreateDirectoryIfNeeded(const char* path)
        {
#ifdef _WIN32
            if (_mkdir(path) == 0 || errno == EEXIST)
#else
            if (mkdir(path, 0755) == 0 || errno == EEXIST)
#endif
                return true;

            ARM_CAPTURE_LOG_ERROR("CAPTURE", "failed to create directory %s: %s", path, std::strerror(errno));
            return false;
        }
    }  // namespace

    bool EnsureParentDirectory(const std::string& filePath)
    {
        const std::string::size_type separator = filePath.find_last_of("\\/");
        if (separator == std::string::npos || separator == 0)
            return true;

        std::string directory = filePath.substr(0, separator);
#ifdef _WIN32
        std::string::size_type firstDirectorySeparator = 1;
        if (directory.size() >= 2 && IsPathSeparator(directory[0]) && IsPathSeparator(directory[1]))
        {
            const std::string::size_type serverEnd = directory.find_first_of("\\/", 2);
            if (serverEnd == std::string::npos)
                return true;
            const std::string::size_type shareEnd = directory.find_first_of("\\/", serverEnd + 1);
            if (shareEnd == std::string::npos)
                return true;
            firstDirectorySeparator = shareEnd + 1;
        }
#else
        constexpr std::string::size_type firstDirectorySeparator = 1;
#endif
        for (std::string::size_type index = firstDirectorySeparator; index < directory.size(); ++index)
        {
            if (!IsPathSeparator(directory[index]) || (index == 2 && directory[1] == ':'))
                continue;

            const char saved   = directory[index];
            directory[index]   = '\0';
            const bool created = CreateDirectoryIfNeeded(directory.c_str());
            directory[index]   = saved;
            if (!created)
                return false;
        }
        return CreateDirectoryIfNeeded(directory.c_str());
    }

    bool EnsureUniqueOutputDirectory(std::string& outputDirectory)
    {
        if (!EnsureParentDirectory(outputDirectory))
            return false;

        std::error_code error;
        if (std::filesystem::create_directory(outputDirectory, error))
            return true;
        if (error == std::errc::file_exists)
            ARM_CAPTURE_LOG_ERROR("CAPTURE", "output directory already exists: %s; choose a new capture path", outputDirectory.c_str());
        else if (error)
            ARM_CAPTURE_LOG_ERROR("CAPTURE", "failed to create output directory %s: %s", outputDirectory.c_str(), error.message().c_str());
        return false;
    }

    FILE* OpenFileForWrite(const std::string& filePath, const char* mode)
    {
        if (!mode || !EnsureParentDirectory(filePath))
            return nullptr;

        FILE* file = std::fopen(filePath.c_str(), mode);
        if (!file)
            ARM_CAPTURE_LOG_ERROR("CAPTURE", "failed to open %s: %s", filePath.c_str(), std::strerror(errno));
        return file;
    }

    bool WriteBinaryFile(const std::string& filePath, const void* data, uint64_t size)
    {
        if (!data || size == 0 || !EnsureParentDirectory(filePath))
            return false;
        if (size > std::numeric_limits<size_t>::max())
        {
            ARM_CAPTURE_LOG_ERROR("CAPTURE", "binary payload is too large for this platform: %s", filePath.c_str());
            return false;
        }
        FILE* file = OpenFileForWrite(filePath, "wb");
        if (!file)
            return false;

        const size_t requested   = static_cast<size_t>(size);
        const size_t written     = std::fwrite(data, 1, requested, file);
        const int    closeResult = std::fclose(file);
        if (written == requested && closeResult == 0)
            return true;

        ARM_CAPTURE_LOG_ERROR("CAPTURE", "failed to write %s", filePath.c_str());
        return false;
    }

}  // namespace arm::capture
