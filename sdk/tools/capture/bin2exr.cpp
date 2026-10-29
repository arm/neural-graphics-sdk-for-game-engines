/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include <capture/payload.h>
#include <capture/format.h>

#include <json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{
    using json   = nlohmann::json;
    namespace fs = std::filesystem;

    constexpr uintmax_t kMaxMetadataBytes = 16u * 1024u * 1024u;

    enum class ConvertResult
    {
        Success,
        MetadataError,
        ConversionError
    };

    // Bin2Exr intentionally accepts ASCII-only capture roots and stream directories so
    // the legacy narrow path/I/O pipeline never depends on Windows code-page conversion exceptions.
    bool IsAscii(const std::string& text)
    {
        return std::all_of(text.begin(), text.end(), [](unsigned char character) { return character <= 0x7f; });
    }

    bool IsAsciiPath(const fs::path& path)
    {
#if defined(_WIN32)
        return std::all_of(path.native().begin(), path.native().end(), [](wchar_t character) { return static_cast<uint32_t>(character) <= 0x7f; });
#else
        return IsAscii(path.native());
#endif
    }

    bool IsWithinRoot(const fs::path& root, const fs::path& candidate, std::string& diagnostic)
    {
        std::error_code rootError;
        const fs::path  canonicalRoot = fs::weakly_canonical(root, rootError);
        if (rootError)
        {
            diagnostic = "Cannot resolve capture directory: " + rootError.message();
            return false;
        }
        std::error_code candidateError;
        const fs::path  canonicalCandidate = fs::weakly_canonical(candidate, candidateError);
        if (candidateError)
        {
            diagnostic = "Cannot resolve candidate path: " + candidateError.message();
            return false;
        }

        auto rootPart      = canonicalRoot.begin();
        auto candidatePart = canonicalCandidate.begin();
        for (; rootPart != canonicalRoot.end() && candidatePart != canonicalCandidate.end(); ++rootPart, ++candidatePart)
        {
            if (*rootPart != *candidatePart)
                return false;
        }
        return rootPart == canonicalRoot.end();
    }

    bool ReadUint32(const json& value, uint32_t& result)
    {
        if (value.is_number_unsigned())
        {
            const uint64_t number = value.get<uint64_t>();
            if (number > std::numeric_limits<uint32_t>::max())
                return false;
            result = static_cast<uint32_t>(number);
            return true;
        }
        if (value.is_number_integer())
        {
            const int64_t number = value.get<int64_t>();
            if (number < 0 || static_cast<uint64_t>(number) > std::numeric_limits<uint32_t>::max())
                return false;
            result = static_cast<uint32_t>(number);
            return true;
        }
        return false;
    }

    bool ReadFloat(const json& value, float& result)
    {
        if (!value.is_number())
            return false;
        const double number = value.get<double>();
        if (!std::isfinite(number) || number > std::numeric_limits<float>::max() || number < -std::numeric_limits<float>::max())
            return false;
        result = static_cast<float>(number);
        return std::isfinite(result);
    }

    bool Convert(const fs::path&         datasetRoot,
                 const std::string&      relativePath,
                 uint32_t                format,
                 uint32_t                width,
                 uint32_t                height,
                 arm::capture::ImageKind kind,
                 float                   motionBias,
                 float                   motionScale,
                 bool                    depthIsD24)
    {
        const fs::path input  = datasetRoot / relativePath;
        fs::path       output = input;
        output.replace_extension(".exr");
        std::string pathDiagnostic;
        if (!IsWithinRoot(datasetRoot, input, pathDiagnostic) || !IsWithinRoot(datasetRoot, output, pathDiagnostic))
        {
            std::cerr << "Refusing path outside capture directory (" << pathDiagnostic << ")\n";
            return false;
        }
        std::error_code       statusError;
        const fs::file_status inputStatus = fs::status(input, statusError);
        if (statusError)
        {
            std::cerr << "Cannot query public raw image (" << statusError.message() << ")\n";
            return false;
        }
        if (inputStatus.type() == fs::file_type::not_found)
        {
            std::cerr << "Missing public raw image\n";
            return false;
        }
        if (inputStatus.type() != fs::file_type::regular)
        {
            std::cerr << "Public raw image is not a regular file\n";
            return false;
        }
        uint64_t expectedSize = 0;
        if (!arm::capture::TryGetPackedImageSize(format, width, height, expectedSize))
        {
            std::cerr << "Raw image dimensions are too large\n";
            return false;
        }
        std::error_code sizeError;
        const uintmax_t actualSize = fs::file_size(input, sizeError);
        if (sizeError)
        {
            std::cerr << "Cannot query raw image size (" << sizeError.message() << ")\n";
            return false;
        }
        if (actualSize != expectedSize)
        {
            std::cerr << "Raw image size does not match metadata\n";
            return false;
        }
        if (!arm::capture::ConvertRawImageToExr(input.string(), output.string(), format, width, height, kind, motionBias, motionScale, depthIsD24))
        {
            std::cerr << "Cannot convert raw image\n";
            return false;
        }
        return true;
    }

    ConvertResult ConvertDataset(const fs::path& root, const json& metadata, bool depthIsD24)
    {
        if (!metadata.is_object() || !metadata.contains("TargetResolution") || !metadata.at("TargetResolution").is_object() ||
            !metadata.at("TargetResolution").contains("X") || !metadata.at("TargetResolution").contains("Y") || !metadata.contains("Frames") ||
            !metadata.at("Frames").is_array())
        {
            std::cerr << "0000.json has no usable resolution or frame metadata\n";
            return ConvertResult::MetadataError;
        }

        const json& targetResolution = metadata.at("TargetResolution");
        uint32_t    targetWidth      = 0;
        uint32_t    targetHeight     = 0;
        if (!ReadUint32(targetResolution.at("X"), targetWidth) || !ReadUint32(targetResolution.at("Y"), targetHeight))
        {
            std::cerr << "0000.json has out-of-range target resolution metadata\n";
            return ConvertResult::MetadataError;
        }
        if (targetWidth == 0 || targetHeight == 0 || metadata.at("Frames").empty())
        {
            std::cerr << "0000.json has an empty resolution or frame list\n";
            return ConvertResult::MetadataError;
        }

        if (!metadata.contains("Streams") || !metadata.at("Streams").is_object() || metadata.at("Streams").empty() || metadata.at("Streams").size() > 32)
        {
            std::cerr << "0000.json has no stream descriptors; recapture with the updated capture writer\n";
            return ConvertResult::MetadataError;
        }

        float      motionBias     = 0.5f;
        float      motionScale    = 2.0f;
        const bool hasMotionBias  = metadata.contains("MotionVectorUnormBias");
        const bool hasMotionScale = metadata.contains("MotionVectorUnormScale");
        if ((hasMotionBias && !metadata.at("MotionVectorUnormBias").is_number()) || (hasMotionScale && !metadata.at("MotionVectorUnormScale").is_number()))
        {
            std::cerr << "0000.json has invalid motion-vector conversion metadata\n";
            return ConvertResult::MetadataError;
        }
        if (hasMotionBias)
            if (!ReadFloat(metadata.at("MotionVectorUnormBias"), motionBias))
            {
                std::cerr << "0000.json has invalid motion-vector conversion metadata\n";
                return ConvertResult::MetadataError;
            }
        if (hasMotionScale)
            if (!ReadFloat(metadata.at("MotionVectorUnormScale"), motionScale))
            {
                std::cerr << "0000.json has invalid motion-vector conversion metadata\n";
                return ConvertResult::MetadataError;
            }
        if (!std::isfinite(motionBias) || !std::isfinite(motionScale) || motionScale == 0.0f)
        {
            std::cerr << "0000.json has invalid motion-vector conversion metadata\n";
            return ConvertResult::MetadataError;
        }

        struct Stream
        {
            std::string             directory;
            uint32_t                format = 0;
            uint32_t                width  = 0;
            uint32_t                height = 0;
            arm::capture::ImageKind kind;
        };

        std::vector<Stream> streams;
        for (const auto& [name, descriptor] : metadata.at("Streams").items())
        {
            if (!descriptor.is_object() || !descriptor.contains("directory") || !descriptor.contains("kind") || !descriptor.contains("format") ||
                !descriptor.contains("width") || !descriptor.contains("height"))
            {
                std::cerr << "Invalid stream descriptor: " << name << '\n';
                return ConvertResult::MetadataError;
            }
            Stream stream;
            if (!descriptor.at("directory").is_string() || !descriptor.at("kind").is_string() || !ReadUint32(descriptor.at("format"), stream.format) ||
                !ReadUint32(descriptor.at("width"), stream.width) || !ReadUint32(descriptor.at("height"), stream.height))
            {
                std::cerr << "Invalid stream descriptor: " << name << '\n';
                return ConvertResult::MetadataError;
            }
            const std::string directory = descriptor.at("directory").get<std::string>();
            const std::string kind      = descriptor.at("kind").get<std::string>();
            if (directory.empty() || stream.width == 0 || stream.height == 0 || !arm::capture::ParseImageKind(kind, stream.kind))
            {
                std::cerr << "Invalid stream descriptor: " << name << '\n';
                return ConvertResult::MetadataError;
            }
            if (!IsAscii(directory))
            {
                std::cerr << "Unsupported non-ASCII stream directory: " << name << '\n';
                return ConvertResult::ConversionError;
            }
            stream.directory = directory;
            streams.push_back(std::move(stream));
        }

        const bool isNfruDataset = std::any_of(streams.begin(), streams.end(), [](const Stream& stream) {
            return stream.kind == arm::capture::ImageKind::GroundTruth || stream.kind == arm::capture::ImageKind::OpticalFlow;
        });

        bool                  success = true;
        std::vector<uint32_t> frameNumbers;
        frameNumbers.reserve(metadata.at("Frames").size());
        for (size_t index = 0; index < metadata.at("Frames").size(); ++index)
        {
            const json& frame = metadata.at("Frames").at(index);
            if (!frame.is_object() || !frame.contains("Frame"))
            {
                std::cerr << "0000.json contains an invalid frame entry at index " << index << '\n';
                success = false;
                continue;
            }
            uint32_t frameNumber = 0;
            if (!ReadUint32(frame.at("Frame"), frameNumber))
            {
                std::cerr << "0000.json contains an out-of-range frame number at index " << index << '\n';
                success = false;
                continue;
            }
            frameNumbers.push_back(frameNumber);
        }
        std::sort(frameNumbers.begin(), frameNumbers.end());
        for (size_t index = 0; index < frameNumbers.size(); ++index)
        {
            if (frameNumbers[index] != index)
            {
                std::cerr << "0000.json contains a non-contiguous or duplicate frame timeline\n";
                success = false;
                break;
            }
        }
        if (!success)
            return ConvertResult::MetadataError;
        for (const Stream& stream : streams)
        {
            const fs::path streamPath = fs::path(stream.directory);
            if (streamPath.is_absolute())
            {
                std::cerr << "Refusing absolute stream directory\n";
                success = false;
                continue;
            }

            const fs::path directory = root / streamPath;
            std::string    pathDiagnostic;
            if (!IsWithinRoot(root, directory, pathDiagnostic))
            {
                std::cerr << "Refusing stream directory outside capture directory (" << pathDiagnostic << ")\n";
                success = false;
                continue;
            }
            const bool      isNfruGenerated  = stream.kind == arm::capture::ImageKind::Output || stream.kind == arm::capture::ImageKind::OpticalFlow;
            const bool      hasExpectedFiles = !isNfruDataset || std::any_of(frameNumbers.begin(), frameNumbers.end(), [isNfruGenerated](uint32_t frameNumber) {
                return ((frameNumber % 2u) == 1u) == isNfruGenerated;
            });
            std::error_code directoryStatusError;
            const fs::file_status directoryStatus = fs::status(directory, directoryStatusError);
            if (directoryStatusError)
            {
                std::cerr << "Cannot query public stream directory (" << directoryStatusError.message() << ")\n";
                success = false;
                continue;
            }
            if (directoryStatus.type() == fs::file_type::not_found && hasExpectedFiles)
            {
                std::cerr << "Missing public stream directory\n";
                success = false;
                continue;
            }
            if (directoryStatus.type() != fs::file_type::directory && directoryStatus.type() != fs::file_type::not_found)
            {
                std::cerr << "Public stream path is not a directory\n";
                success = false;
                continue;
            }
            if (!hasExpectedFiles)
                continue;
            for (const uint32_t frameNumber : frameNumbers)
            {
                if (isNfruDataset && ((frameNumber % 2u) == 1u) != isNfruGenerated)
                    continue;
                char filename[16];
                std::snprintf(filename, sizeof(filename), "%04u.bin", frameNumber);
                const fs::path        expected = directory / filename;
                std::error_code       expectedStatusError;
                const fs::file_status expectedStatus = fs::status(expected, expectedStatusError);
                if (expectedStatusError)
                {
                    std::cerr << "Cannot query expected raw image (" << expectedStatusError.message() << ")\n";
                    success = false;
                }
                else if (expectedStatus.type() != fs::file_type::regular)
                {
                    std::cerr << "Missing expected raw image\n";
                    success = false;
                }
            }
            std::error_code        iteratorError;
            fs::directory_iterator iterator(directory, iteratorError);
            if (iteratorError)
            {
                std::cerr << "Cannot enumerate stream directory (" << iteratorError.message() << ")\n";
                success = false;
                continue;
            }
            const fs::directory_iterator end;
            while (iterator != end)
            {
                const fs::directory_entry entry = *iterator;
                std::error_code           entryStatusError;
                const fs::file_status     entryStatus = entry.status(entryStatusError);
                if (entryStatusError)
                {
                    std::cerr << "Cannot query stream directory entry (" << entryStatusError.message() << ")\n";
                    success = false;
                }
                else if (entryStatus.type() == fs::file_type::regular && entry.path().extension() == ".bin")
                {
                    if (!IsAsciiPath(entry.path()))
                    {
                        std::cerr << "Unsupported non-ASCII raw image path\n";
                        success = false;
                    }
                    else
                    {
                        std::string entryDiagnostic;
                        if (!IsWithinRoot(root, entry.path(), entryDiagnostic))
                        {
                            std::cerr << "Refusing stream file outside capture directory (" << entryDiagnostic << ")\n";
                            success = false;
                        }
                        else
                        {
                            std::error_code entryCanonicalError;
                            const fs::path  canonicalEntry = fs::weakly_canonical(entry.path(), entryCanonicalError);
                            if (entryCanonicalError)
                            {
                                std::cerr << "Cannot resolve stream file path (" << entryCanonicalError.message() << ")\n";
                                success = false;
                            }
                            else
                            {
                                std::error_code rootCanonicalError;
                                const fs::path  canonicalRoot = fs::weakly_canonical(root, rootCanonicalError);
                                if (rootCanonicalError)
                                {
                                    std::cerr << "Cannot resolve capture directory (" << rootCanonicalError.message() << ")\n";
                                    success = false;
                                }
                                else
                                {
                                    std::error_code relativeError;
                                    const fs::path  relativePath = fs::relative(canonicalEntry, canonicalRoot, relativeError);
                                    if (relativeError)
                                    {
                                        std::cerr << "Cannot resolve stream file path (" << relativeError.message() << ")\n";
                                        success = false;
                                    }
                                    else if (!IsAsciiPath(canonicalEntry) || !IsAsciiPath(relativePath))
                                    {
                                        std::cerr << "Unsupported non-ASCII raw image path\n";
                                        success = false;
                                    }
                                    else
                                    {
                                        success &= Convert(root,
                                                           relativePath.generic_string(),
                                                           stream.format,
                                                           stream.width,
                                                           stream.height,
                                                           stream.kind,
                                                           motionBias,
                                                           motionScale,
                                                           depthIsD24);
                                    }
                                }
                            }
                        }
                    }
                }
                std::error_code incrementError;
                iterator.increment(incrementError);
                if (incrementError)
                {
                    std::cerr << "Cannot advance stream directory iterator (" << incrementError.message() << ")\n";
                    success = false;
                    break;
                }
            }
        }
        return success ? ConvertResult::Success : ConvertResult::ConversionError;
    }
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2 && (argc != 3 || std::string(argv[2]) != "--depth-is-d24"))
    {
        std::cerr << "Usage: Bin2Exr <capture-directory> [--depth-is-d24]\n";
        return 2;
    }
    if (!IsAscii(argv[1]))
    {
        std::cerr << "Unsupported non-ASCII capture directory\n";
        return 1;
    }

    const fs::path  root         = argv[1];
    const bool      depthIsD24   = argc == 3;
    const fs::path  metadataPath = root / "0000.json";
    std::error_code metadataError;
    const uintmax_t metadataBytes = fs::file_size(metadataPath, metadataError);
    if (metadataError || metadataBytes > kMaxMetadataBytes)
    {
        std::cerr << "Capture metadata is missing or exceeds the converter's 16 MiB resource limit\n";
        return 2;
    }
    std::ifstream stream(metadataPath);
    if (!stream)
    {
        std::cerr << "Cannot open capture metadata: " << metadataPath.string() << "\n";
        return 2;
    }
    const json metadata = json::parse(stream, nullptr, false);
    if (metadata.is_discarded())
    {
        std::cerr << "Cannot parse capture metadata\n";
        return 2;
    }
    const ConvertResult result = ConvertDataset(root, metadata, depthIsD24);
    return result == ConvertResult::Success ? 0 : result == ConvertResult::MetadataError ? 2 : 1;
}
