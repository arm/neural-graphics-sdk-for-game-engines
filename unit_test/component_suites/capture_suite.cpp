/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "framework/test_framework.h"
#include "framework/vk_test_utils.h"

#include "capture/common.h"
#include "capture/allocator.h"
#include "capture/base.h"
#include "capture/execution.h"
#include "capture/file.h"
#include "capture/format.h"
#include "capture/json.hpp"
#include "capture/payload.h"
#include "capture/readback.h"
#include "capture/staging_pool.h"
#include "capture/transaction.h"
#include "capture/utils.h"
#include "capture/worker.h"
#include "capture/framegenerations.h"
#include "capture/neuralssuperresolution.h"

static constexpr const char* ffxLogComponentName()
{
    return "CaptureTests";
}

#include <FidelityFX/host/backends/vk/vk_wrapper.h>
#include <FidelityFX/gpu/frameinterpolation/ffx_frameinterpolation_resources.h>
#include <ffx_api/ffx_framegeneration.h>
#include <ffx_api/ffx_api_types.h>
#include <ffx_api/ffx_nss.h>
#include <tinyexr/tinyexr.h>

#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace
{
    class OutputDirectory
    {
    public:
        explicit OutputDirectory(const char* name)
            : path_(std::filesystem::current_path() / "payload_serialization_test_output" / name)
        {
            std::error_code error;
            std::filesystem::remove_all(path_, error);
        }

        ~OutputDirectory()
        {
            std::error_code error;
            std::filesystem::remove_all(path_, error);
        }

        std::filesystem::path File(const char* name) const
        {
            return path_ / name;
        }

    private:
        std::filesystem::path path_;
    };

    class ExrReader
    {
    public:
        ExrReader()
        {
            InitEXRHeader(&header_);
            InitEXRImage(&image_);
        }

        ~ExrReader()
        {
            FreeEXRImage(&image_);
            FreeEXRHeader(&header_);
        }

        bool Load(const std::filesystem::path& path)
        {
            const std::string pathString = path.string();
            const char*       error      = nullptr;
            if (ParseEXRVersionFromFile(&version_, pathString.c_str()) != TINYEXR_SUCCESS ||
                ParseEXRHeaderFromFile(&header_, &version_, pathString.c_str(), &error) != TINYEXR_SUCCESS ||
                LoadEXRImageFromFile(&image_, &header_, pathString.c_str(), &error) != TINYEXR_SUCCESS)
            {
                if (error)
                    FreeEXRErrorMessage(error);
                return false;
            }
            return true;
        }

        int ChannelIndex(const char* name) const
        {
            for (int index = 0; index < header_.num_channels; ++index)
            {
                if (std::strcmp(header_.channels[index].name, name) == 0)
                    return index;
            }
            return -1;
        }

        float FloatPixel(const char* channel, uint32_t pixel) const
        {
            const int index = ChannelIndex(channel);
            if (index < 0 || header_.pixel_types[index] != TINYEXR_PIXELTYPE_FLOAT)
                return 0.0f;
            return reinterpret_cast<const float*>(image_.images[index])[pixel];
        }

        bool HasFloatChannels(uint32_t count) const
        {
            if (header_.num_channels != static_cast<int>(count))
                return false;
            for (uint32_t index = 0; index < count; ++index)
            {
                if (header_.pixel_types[index] != TINYEXR_PIXELTYPE_FLOAT)
                    return false;
            }
            return true;
        }

        int Width() const
        {
            return image_.width;
        }

        int Height() const
        {
            return image_.height;
        }

    private:
        EXRVersion version_{};
        EXRHeader  header_{};
        EXRImage   image_{};
    };

    bool CloseEnough(float actual, float expected)
    {
        return std::fabs(actual - expected) < 0.0001f;
    }

    std::vector<uint8_t> ReadBytes(const std::filesystem::path& path)
    {
        std::ifstream input(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }
}  // namespace

namespace
{
    constexpr uint64_t kMiB = 1024ull * 1024;
}

UNIT_TEST_CASE(capture, emulated_framerate_handles_invalid_frame_time)
{
    using arm::capture::CalculateEmulatedFramerate;

    UNIT_TEST_CHECK(CalculateEmulatedFramerate(16.6f, 1, 240) == 60);
    UNIT_TEST_CHECK(CalculateEmulatedFramerate(0.0f, 30, 90) == 60);
    UNIT_TEST_CHECK(CalculateEmulatedFramerate(-1.0f, 30, 90) == 60);
    UNIT_TEST_CHECK(CalculateEmulatedFramerate(std::numeric_limits<float>::quiet_NaN(), 30, 90) == 60);
    UNIT_TEST_CHECK(CalculateEmulatedFramerate(std::numeric_limits<float>::infinity(), 30, 90) == 60);
    UNIT_TEST_CHECK(CalculateEmulatedFramerate(std::numeric_limits<float>::denorm_min(), 30, 90) == 90);
    UNIT_TEST_CHECK(CalculateEmulatedFramerate(1000.0f, 30, 90) == 30);
    return true;
}

UNIT_TEST_CASE(capture, performance_toggle_parses_explicit_values)
{
    using arm::capture::ParseCapturePerformanceEnabled;

    UNIT_TEST_CHECK(ParseCapturePerformanceEnabled("1"));
    UNIT_TEST_CHECK(ParseCapturePerformanceEnabled("true"));
    UNIT_TEST_CHECK(ParseCapturePerformanceEnabled("YES"));
    UNIT_TEST_CHECK(ParseCapturePerformanceEnabled("On"));
    UNIT_TEST_CHECK(!ParseCapturePerformanceEnabled(nullptr));
    UNIT_TEST_CHECK(!ParseCapturePerformanceEnabled(""));
    UNIT_TEST_CHECK(!ParseCapturePerformanceEnabled("0"));
    UNIT_TEST_CHECK(!ParseCapturePerformanceEnabled("false"));
    UNIT_TEST_CHECK(!ParseCapturePerformanceEnabled("enabled"));
    return true;
}

// Capture runtime foundation: arithmetic, resource descriptors, transactions, and metadata bookkeeping.
UNIT_TEST_CASE(capture, foundation_alignment)
{
    using namespace arm::capture::detail;

    UNIT_TEST_CHECK(AlignUp(0, 1) == 0);
    UNIT_TEST_CHECK(AlignUp(kMiB + 1, 256) == kMiB + 256);
    UNIT_TEST_CHECK(AlignUp(std::numeric_limits<uint64_t>::max() - 1, 4) == 0);
    UNIT_TEST_CHECK(StagingAllocationUnitSize(1) == kMiB);
    UNIT_TEST_CHECK(StagingAllocationUnitSize(256) == kMiB);
    UNIT_TEST_CHECK(StagingAllocationUnitSize(3) == 0);
    return true;
}

UNIT_TEST_CASE(capture, foundation_slot_bytes)
{
    using namespace arm::capture::detail;

    UNIT_TEST_CHECK(SlotBytesForAttachments({1}, {4}, 256, kMiB) == kMiB);
    UNIT_TEST_CHECK(SlotBytesForAttachments({1, 1, 1, 1, 1}, {4, 4, 4, 4, 4}, kMiB, kMiB) == 5 * kMiB);
    UNIT_TEST_CHECK(SlotBytesForAttachments({kMiB, 1, kMiB}, {4, 4, 4}, kMiB, kMiB) == 3 * kMiB);
    UNIT_TEST_CHECK(SlotBytesForAttachments({}, {}, 256, kMiB) == 0);
    UNIT_TEST_CHECK(SlotBytesForAttachments({1}, {4}, 0, kMiB) == 0);
    UNIT_TEST_CHECK(SlotBytesForAttachments({0}, {4}, 256, kMiB) == 0);
    UNIT_TEST_CHECK(SlotBytesForAttachments({std::numeric_limits<uint64_t>::max(), 1}, {4, 4}, 1, kMiB) == 0);
    UNIT_TEST_CHECK(SlotBytesForAttachments({1, 1}, {4, 12}, 16, 16) == 64);
    return true;
}

UNIT_TEST_CASE(capture, foundation_resource_bytes)
{
    FfxResource resource{};
    resource.resource           = reinterpret_cast<void*>(1);
    resource.description.type   = FFX_RESOURCE_TYPE_TEXTURE2D;
    resource.description.width  = 2;
    resource.description.height = 3;
    resource.description.format = static_cast<FfxSurfaceFormat>(FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT);
    uint64_t total              = 0;

    UNIT_TEST_CHECK(arm::capture::AddResourceBytes(resource, total));
    UNIT_TEST_CHECK(total == 2u * 3u * 4u);

    resource.description.format = static_cast<FfxSurfaceFormat>(FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT);
    UNIT_TEST_CHECK(arm::capture::AddResourceBytes(resource, total));
    UNIT_TEST_CHECK(total == (2u * 3u * 4u) + (2u * 3u * 16u));

    total                       = 0;
    resource.description.format = static_cast<FfxSurfaceFormat>(FFX_API_SURFACE_FORMAT_R32G32_FLOAT);
    UNIT_TEST_CHECK(arm::capture::AddResourceBytes(resource, total));
    UNIT_TEST_CHECK(total == 2u * 3u * 8u);

    resource.description.format = static_cast<FfxSurfaceFormat>(FFX_API_SURFACE_FORMAT_R32G32_TYPELESS);
    UNIT_TEST_CHECK(arm::capture::AddResourceBytes(resource, total));
    UNIT_TEST_CHECK(total == 2u * (2u * 3u * 8u));

    resource.description.format = static_cast<FfxSurfaceFormat>(FFX_API_SURFACE_FORMAT_UNKNOWN);
    UNIT_TEST_CHECK(!arm::capture::AddResourceBytes(resource, total));
    return true;
}

UNIT_TEST_CASE(capture, foundation_attachment_plan_requirements)
{
    arm::capture::CaptureAttachmentPlan plan;
    plan.inputCount = 2;
    for (const uint32_t width : {2u, 3u, 5u})
    {
        FfxResource resource{};
        resource.resource           = reinterpret_cast<void*>(1);
        resource.description.type   = FFX_RESOURCE_TYPE_TEXTURE2D;
        resource.description.width  = width;
        resource.description.height = 1;
        resource.description.format = static_cast<FfxSurfaceFormat>(FFX_API_SURFACE_FORMAT_R8_UINT);
        plan.attachments.push_back({resource, {}});
    }

    arm::capture::SlotRequirements requirements;
    UNIT_TEST_CHECK(arm::capture::BuildSlotRequirements(plan, requirements));
    UNIT_TEST_CHECK(requirements.attachmentBytes == std::vector<uint64_t>({2, 3, 5}));
    UNIT_TEST_CHECK(requirements.attachmentAlignments == std::vector<uint64_t>({4, 4, 4}));

    plan.inputCount = plan.attachments.size() + 1;
    UNIT_TEST_CHECK(!arm::capture::BuildSlotRequirements(plan, requirements));
    UNIT_TEST_CHECK(requirements.attachmentBytes.empty());
    UNIT_TEST_CHECK(requirements.attachmentAlignments.empty());
    return true;
}

UNIT_TEST_CASE(capture, foundation_format_bytes_per_pixel)
{
    const std::array formats = {
        FFX_API_SURFACE_FORMAT_UNKNOWN,
        FFX_API_SURFACE_FORMAT_R32G32B32A32_TYPELESS,
        FFX_API_SURFACE_FORMAT_R32G32B32A32_UINT,
        FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT,
        FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,
        FFX_API_SURFACE_FORMAT_R32G32B32_FLOAT,
        FFX_API_SURFACE_FORMAT_R32G32_FLOAT,
        FFX_API_SURFACE_FORMAT_R8_UINT,
        FFX_API_SURFACE_FORMAT_R8_SINT,
        FFX_API_SURFACE_FORMAT_R32_UINT,
        FFX_API_SURFACE_FORMAT_R8G8B8A8_TYPELESS,
        FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM,
        FFX_API_SURFACE_FORMAT_R8G8B8A8_SNORM,
        FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB,
        FFX_API_SURFACE_FORMAT_B8G8R8A8_TYPELESS,
        FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM,
        FFX_API_SURFACE_FORMAT_B8G8R8A8_SRGB,
        FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT,
        FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM,
        FFX_API_SURFACE_FORMAT_R16G16_FLOAT,
        FFX_API_SURFACE_FORMAT_R16G16_UINT,
        FFX_API_SURFACE_FORMAT_R16G16_SINT,
        FFX_API_SURFACE_FORMAT_R16_FLOAT,
        FFX_API_SURFACE_FORMAT_R16_UINT,
        FFX_API_SURFACE_FORMAT_R16_UNORM,
        FFX_API_SURFACE_FORMAT_R16_SNORM,
        FFX_API_SURFACE_FORMAT_R8_UNORM,
        FFX_API_SURFACE_FORMAT_R8_SNORM,
        FFX_API_SURFACE_FORMAT_R8G8_UNORM,
        FFX_API_SURFACE_FORMAT_R8G8_UINT,
        FFX_API_SURFACE_FORMAT_R8G8_SINT,
        FFX_API_SURFACE_FORMAT_R32_FLOAT,
        FFX_API_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP,
        FFX_API_SURFACE_FORMAT_R16G16B16A16_TYPELESS,
        FFX_API_SURFACE_FORMAT_R32G32_TYPELESS,
        FFX_API_SURFACE_FORMAT_R10G10B10A2_TYPELESS,
        FFX_API_SURFACE_FORMAT_R16G16_TYPELESS,
        FFX_API_SURFACE_FORMAT_R16_TYPELESS,
        FFX_API_SURFACE_FORMAT_R8_TYPELESS,
        FFX_API_SURFACE_FORMAT_R8G8_TYPELESS,
        FFX_API_SURFACE_FORMAT_R32_TYPELESS,
    };
    const std::array<uint32_t, formats.size()> expectedBytesPerPixel = {
        0, 16, 16, 16, 8, 12, 8, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 1, 1, 2, 2, 2, 4, 4, 8, 8, 4, 4, 2, 1, 2, 4,
    };
    static_assert(formats.size() == static_cast<size_t>(FFX_API_SURFACE_FORMAT_R32_TYPELESS) + 1);

    for (size_t index = 0; index < formats.size(); ++index)
        UNIT_TEST_CHECK(arm::capture::GetFormatBytesPerPixel(formats[index]) == expectedBytesPerPixel[index]);
    return true;
}

UNIT_TEST_CASE(capture, readback_descriptor)
{
    FfxResource image{};
    image.resource           = reinterpret_cast<void*>(1);
    image.state              = FFX_RESOURCE_STATE_PIXEL_READ;
    image.description.type   = FFX_RESOURCE_TYPE_TEXTURE2D;
    image.description.format = static_cast<FfxSurfaceFormat>(FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM);
    image.description.width  = 4;
    image.description.height = 2;

    arm::capture::vk::ImageReadbackDesc imageDesc{};
    UNIT_TEST_CHECK(arm::capture::vk::BuildImageReadbackDesc(image, imageDesc));
    UNIT_TEST_CHECK(imageDesc.image == reinterpret_cast<VkImage>(image.resource));
    UNIT_TEST_CHECK(imageDesc.currentLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    UNIT_TEST_CHECK(imageDesc.extent.width == 4);
    UNIT_TEST_CHECK(imageDesc.extent.height == 2);
    UNIT_TEST_CHECK(imageDesc.byteSize == 32);

    image.description.format = static_cast<FfxSurfaceFormat>(FFX_API_SURFACE_FORMAT_R32G32_FLOAT);
    UNIT_TEST_CHECK(arm::capture::vk::BuildImageReadbackDesc(image, imageDesc));
    UNIT_TEST_CHECK(imageDesc.byteSize == 4u * 2u * 8u);

    image.state = FFX_RESOURCE_STATE_GENERIC_READ;
    UNIT_TEST_CHECK(arm::capture::vk::BuildImageReadbackDesc(image, imageDesc));
    UNIT_TEST_CHECK(imageDesc.currentLayout == VK_IMAGE_LAYOUT_GENERAL);

    image.state = FFX_RESOURCE_STATE_DATA_GRAPH_TEXTURE_READ;
    UNIT_TEST_CHECK(arm::capture::vk::BuildImageReadbackDesc(image, imageDesc));
    UNIT_TEST_CHECK(imageDesc.currentLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    image.state = FFX_RESOURCE_STATE_DATA_GRAPH_TEXTURE_WRITE;
    UNIT_TEST_CHECK(arm::capture::vk::BuildImageReadbackDesc(image, imageDesc));
    UNIT_TEST_CHECK(imageDesc.currentLayout == VK_IMAGE_LAYOUT_GENERAL);

    image.description.usage = FFX_RESOURCE_USAGE_DEPTHTARGET;
    UNIT_TEST_CHECK(arm::capture::vk::BuildImageReadbackDesc(image, imageDesc));
    UNIT_TEST_CHECK(imageDesc.aspect == VK_IMAGE_ASPECT_DEPTH_BIT);

    FfxResource buffer{};
    buffer.resource         = reinterpret_cast<void*>(2);
    buffer.state            = FFX_RESOURCE_STATE_COMPUTE_UAV;
    buffer.description.type = FFX_RESOURCE_TYPE_BUFFER;
    buffer.description.size = 128;
    arm::capture::vk::BufferReadbackDesc bufferDesc{};
    UNIT_TEST_CHECK(arm::capture::vk::BuildBufferReadbackDesc(buffer, bufferDesc));
    UNIT_TEST_CHECK(bufferDesc.buffer == reinterpret_cast<VkBuffer>(buffer.resource));
    UNIT_TEST_CHECK(bufferDesc.byteSize == 128);
    UNIT_TEST_CHECK(bufferDesc.currentState == FFX_RESOURCE_STATE_COMPUTE_UAV);

    const VkBufferMemoryBarrier2 restore = arm::capture::vk::MakeBufferReadbackRestoreBarrier(bufferDesc.buffer);
    UNIT_TEST_CHECK(restore.srcStageMask == VK_PIPELINE_STAGE_2_TRANSFER_BIT);
    UNIT_TEST_CHECK(restore.srcAccessMask == VK_ACCESS_2_TRANSFER_READ_BIT);
    UNIT_TEST_CHECK(restore.dstStageMask == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    UNIT_TEST_CHECK(restore.dstAccessMask == (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT));
    UNIT_TEST_CHECK(restore.buffer == bufferDesc.buffer);
    UNIT_TEST_CHECK(restore.size == VK_WHOLE_SIZE);

    buffer.description.size = 126;
    UNIT_TEST_CHECK(!arm::capture::vk::BuildBufferReadbackDesc(buffer, bufferDesc));

    buffer.description.size = 0;
    UNIT_TEST_CHECK(!arm::capture::vk::BuildBufferReadbackDesc(buffer, bufferDesc));

    constexpr FfxResourceStates kInvalidCombinedBufferState =
        static_cast<FfxResourceStates>(FFX_RESOURCE_STATE_COMMON | FFX_RESOURCE_STATE_COMPUTE_UAV | FFX_RESOURCE_STATE_PIXEL_UAV);
    buffer.description.size = 128;
    buffer.state            = kInvalidCombinedBufferState;
    UNIT_TEST_CHECK(!arm::capture::vk::BuildBufferReadbackDesc(buffer, bufferDesc));

    buffer.state = FFX_RESOURCE_STATE_DATA_GRAPH_TEXTURE_READ;
    UNIT_TEST_CHECK(!arm::capture::vk::BuildBufferReadbackDesc(buffer, bufferDesc));
    buffer.state = FFX_RESOURCE_STATE_DATA_GRAPH_TEXTURE_WRITE;
    UNIT_TEST_CHECK(!arm::capture::vk::BuildBufferReadbackDesc(buffer, bufferDesc));

    image.description.width  = std::numeric_limits<uint32_t>::max();
    image.description.height = std::numeric_limits<uint32_t>::max();
    UNIT_TEST_CHECK(!arm::capture::vk::BuildImageReadbackDesc(image, imageDesc));
    return true;
}

UNIT_TEST_CASE(capture, transaction_invalid)
{
    arm::capture::Transaction transaction{};
    auto                      metadata    = std::make_unique<arm::capture::FrameMeta>();
    const FfxCommandList      commandList = reinterpret_cast<FfxCommandList>(1);

    UNIT_TEST_CHECK(!arm::capture::BeginTransaction(arm::capture::InvalidSession, 1024, commandList, std::move(metadata), transaction));
    UNIT_TEST_CHECK(!transaction.active);
    UNIT_TEST_CHECK(!arm::capture::CommitTransaction(transaction));
    arm::capture::CancelTransaction(transaction);

    FfxResource resource{};
    UNIT_TEST_CHECK(!arm::capture::AddAttachment(transaction, resource, std::make_unique<arm::capture::NormalPayload>("x", "x.exr")));
    UNIT_TEST_CHECK(!arm::capture::AddAttachment(transaction, resource, std::unique_ptr<arm::capture::Payload>{}));
    uint32_t slotIndex = UINT32_MAX;
    UNIT_TEST_CHECK(!arm::capture::detail::GetWorker().AcquireSlot(arm::capture::InvalidSession, 1024, slotIndex));
    return true;
}

// A failed Abort must leave the transaction active so the caller cannot mistake it for a drained transaction.
UNIT_TEST_CASE(capture, transaction_cancel_failure_preserves_active_state)
{
    arm::capture::Transaction transaction{};
    transaction.active      = true;
    transaction.session     = arm::capture::InvalidSession;
    transaction.commandList = nullptr;

    UNIT_TEST_CHECK(!arm::capture::CancelTransaction(transaction));
    UNIT_TEST_CHECK(transaction.active);
    return true;
}

UNIT_TEST_CASE(capture, metadata_json_count)
{
    enum class JsonOperation
    {
        None,
        Open,
        Write,
        Flush,
        Tell,
        Seek,
        Close,
    };

    class FaultInjectingJsonFileOps final : public arm::capture::JsonFileOps
    {
    public:
        FaultInjectingJsonFileOps(JsonOperation failureOperation = JsonOperation::None, uint32_t failOnCall = 0)
            : failureOperation(failureOperation)
            , failOnCall(failOnCall)
        {
        }

        FILE* Open(const std::string&) override
        {
            ++openCount;
            if (ShouldFail(JsonOperation::Open, openCount))
                return nullptr;
            return std::tmpfile();
        }

        int Write(FILE* file, const char* format, va_list args) override
        {
            ++writeCount;
            if (ShouldFail(JsonOperation::Write, writeCount))
                return -1;
            return std::vfprintf(file, format, args);
        }

        int Flush(FILE* file) override
        {
            ++flushCount;
            if (ShouldFail(JsonOperation::Flush, flushCount))
                return -1;
            return std::fflush(file);
        }

        long Tell(FILE* file) override
        {
            ++tellCount;
            if (ShouldFail(JsonOperation::Tell, tellCount))
                return -1;
            return std::ftell(file);
        }

        int Seek(FILE* file, long offset, int origin) override
        {
            ++seekCount;
            if (ShouldFail(JsonOperation::Seek, seekCount))
                return -1;
            return std::fseek(file, offset, origin);
        }

        int Close(FILE* file) override
        {
            ++closeCount;
            const int result = std::fclose(file);
            return ShouldFail(JsonOperation::Close, closeCount) ? -1 : result;
        }

        bool ShouldFail(JsonOperation operation, uint32_t call) const
        {
            return failureOperation == operation && failOnCall == call;
        }

        JsonOperation failureOperation = JsonOperation::None;
        uint32_t      failOnCall       = 0;
        uint32_t      openCount        = 0;
        uint32_t      writeCount       = 0;
        uint32_t      flushCount       = 0;
        uint32_t      tellCount        = 0;
        uint32_t      seekCount        = 0;
        uint32_t      closeCount       = 0;
    };

    class TestJsonWriter final : public arm::capture::JsonWriter
    {
    public:
        TestJsonWriter() = default;

        explicit TestJsonWriter(arm::capture::JsonFileOps& fileOps)
            : JsonWriter(fileOps)
        {
        }

        bool Write(const std::filesystem::path& path)
        {
            if (!Open(path.string()))
                return false;
            Writef("{\"Count\": ");
            SetCountOffset();
            Writef("         0,\"Frames\":[");
            BeginEntry();
            Writef("{\"Frame\":0}");
            CommitEntry();
            BeginEntry();
            Writef("{\"Frame\":1}");
            CommitEntry();
            Writef("]}");
            return BackfillCountAndClose();
        }
    };

    OutputDirectory output("json");
    const auto      path = output.File("nested/metadata.json");
    TestJsonWriter  writer;
    UNIT_TEST_CHECK(writer.Write(path));
    UNIT_TEST_CHECK(!writer.IsOpen());
    UNIT_TEST_CHECK(writer.GetWrittenEntryCount() == 2);

    const std::vector<uint8_t> bytes = ReadBytes(path);
    const std::string          json(bytes.begin(), bytes.end());
    UNIT_TEST_CHECK(json.find("\"Count\":          2") != std::string::npos);
    UNIT_TEST_CHECK(json.find("\"Frame\":1") != std::string::npos);

    const auto blockingParent = output.File("blocking_parent");
    {
        std::ofstream blocker(blockingParent);
        UNIT_TEST_CHECK(blocker.good());
    }
    TestJsonWriter blockedWriter;
    UNIT_TEST_CHECK(!blockedWriter.Write(blockingParent / "metadata.json"));

    {
        FaultInjectingJsonFileOps fileOps(JsonOperation::Open, 1);
        TestJsonWriter            writerWithFailure(fileOps);
        UNIT_TEST_CHECK(!writerWithFailure.Write(output.File("injected_open_failure.json")));
        UNIT_TEST_CHECK(!writerWithFailure.IsOpen());
        UNIT_TEST_CHECK(fileOps.openCount == 1);
        UNIT_TEST_CHECK(fileOps.writeCount == 0);
        UNIT_TEST_CHECK(fileOps.closeCount == 0);
    }

    const auto verifyFailure =
        [&](JsonOperation operation, uint32_t failOnCall, uint32_t expectedWrites, uint32_t expectedFlushes, uint32_t expectedTells, uint32_t expectedSeeks) {
            FaultInjectingJsonFileOps fileOps(operation, failOnCall);
            TestJsonWriter            writerWithFailure(fileOps);
            UNIT_TEST_CHECK(!writerWithFailure.Write(output.File("injected_failure.json")));
            UNIT_TEST_CHECK(!writerWithFailure.IsOpen());
            UNIT_TEST_CHECK(writerWithFailure.GetWrittenEntryCount() <= 2);
            UNIT_TEST_CHECK(fileOps.openCount == 1);
            UNIT_TEST_CHECK(fileOps.writeCount == expectedWrites);
            UNIT_TEST_CHECK(fileOps.flushCount == expectedFlushes);
            UNIT_TEST_CHECK(fileOps.tellCount == expectedTells);
            UNIT_TEST_CHECK(fileOps.seekCount == expectedSeeks);
            UNIT_TEST_CHECK(fileOps.closeCount == 1);
            return true;
        };

    UNIT_TEST_CHECK(verifyFailure(JsonOperation::Write, 1, 1, 0, 0, 0));
    UNIT_TEST_CHECK(verifyFailure(JsonOperation::Flush, 1, 3, 1, 1, 0));
    UNIT_TEST_CHECK(verifyFailure(JsonOperation::Flush, 3, 6, 3, 1, 0));
    UNIT_TEST_CHECK(verifyFailure(JsonOperation::Flush, 4, 7, 4, 1, 1));
    UNIT_TEST_CHECK(verifyFailure(JsonOperation::Tell, 1, 1, 0, 1, 0));
    UNIT_TEST_CHECK(verifyFailure(JsonOperation::Seek, 1, 6, 3, 1, 1));
    UNIT_TEST_CHECK(verifyFailure(JsonOperation::Close, 1, 7, 4, 1, 1));

    {
        FaultInjectingJsonFileOps fileOps;
        TestJsonWriter            writerWithFailure(fileOps);
        UNIT_TEST_CHECK(writerWithFailure.Write(output.File("injected_success.json")));
        UNIT_TEST_CHECK(fileOps.openCount == 1);
        UNIT_TEST_CHECK(fileOps.writeCount == 7);
        UNIT_TEST_CHECK(fileOps.flushCount == 4);
        UNIT_TEST_CHECK(fileOps.tellCount == 1);
        UNIT_TEST_CHECK(fileOps.seekCount == 1);
        UNIT_TEST_CHECK(fileOps.closeCount == 1);
    }

    return true;
}

// Payload serialization: raw, EXR, depth, motion-vector, and invalid-input behavior.
UNIT_TEST_CASE(capture, payload_binary_write)
{
    OutputDirectory                 output("binary");
    const std::filesystem::path     path  = output.File("nested/raw/payload.bin");
    const std::array<uint8_t, 5>    bytes = {0x00, 0x7f, 0x80, 0xab, 0xff};
    arm::capture::BinaryPayload     payload("rawCapture", path.string());
    const arm::capture::PayloadView view = {bytes.data(), bytes.size(), FFX_API_SURFACE_FORMAT_R8_UINT, {5, 1}};

    UNIT_TEST_CHECK(std::strcmp(payload.GetName(), "rawCapture") == 0);
    UNIT_TEST_CHECK(payload.Write(view));
    UNIT_TEST_CHECK(std::filesystem::is_regular_file(path));
    UNIT_TEST_CHECK(ReadBytes(path) == std::vector<uint8_t>(bytes.begin(), bytes.end()));

    UNIT_TEST_CHECK(!payload.Write({nullptr, 0, FFX_API_SURFACE_FORMAT_R8_UINT, {5, 1}}));
    UNIT_TEST_CHECK(ReadBytes(path) == std::vector<uint8_t>(bytes.begin(), bytes.end()));

    const auto blockingParent = output.File("blocking_parent");
    {
        std::ofstream blocker(blockingParent);
        UNIT_TEST_CHECK(blocker.good());
    }
    arm::capture::BinaryPayload blockedPayload("blockedCapture", (blockingParent / "payload.bin").string());
    UNIT_TEST_CHECK(!blockedPayload.Write(view));

    return true;
}

UNIT_TEST_CASE(capture, payload_binary_write_rejects_unrepresentable_size)
{
#if SIZE_MAX < UINT64_MAX
    const OutputDirectory output("payload_binary_unrepresentable_size");
    const uint8_t         value = 0;
    UNIT_TEST_CHECK(!arm::capture::WriteBinaryFile(output.File("oversized.bin").string(), &value, static_cast<uint64_t>(SIZE_MAX) + 1));
#endif
    return true;
}

UNIT_TEST_CASE(capture, payload_exr_rgba)
{
    OutputDirectory              output("normal");
    const std::filesystem::path  path = output.File("nested/color.exr");
    const std::array<uint8_t, 8> rgba = {0, 64, 128, 255, 255, 128, 64, 0};
    arm::capture::NormalPayload  payload("color", path.string());

    payload.Write({rgba.data(), rgba.size(), FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM, {2, 1}});
    UNIT_TEST_CHECK(std::filesystem::is_regular_file(path));

    ExrReader image;
    UNIT_TEST_CHECK(image.Load(path));
    UNIT_TEST_CHECK(image.Width() == 2);
    UNIT_TEST_CHECK(image.Height() == 1);
    UNIT_TEST_CHECK(image.HasFloatChannels(4));
    UNIT_TEST_CHECK(CloseEnough(image.FloatPixel("R", 0), 0.0f));
    UNIT_TEST_CHECK(CloseEnough(image.FloatPixel("G", 0), 64.0f / 255.0f));
    UNIT_TEST_CHECK(CloseEnough(image.FloatPixel("B", 0), 128.0f / 255.0f));
    UNIT_TEST_CHECK(CloseEnough(image.FloatPixel("A", 0), 1.0f));
    UNIT_TEST_CHECK(CloseEnough(image.FloatPixel("R", 1), 1.0f));
    UNIT_TEST_CHECK(CloseEnough(image.FloatPixel("A", 1), 0.0f));
    return true;
}

UNIT_TEST_CASE(capture, payload_depth_motion)
{
    OutputDirectory               output("depth_and_scaled");
    const std::filesystem::path   depthPath  = output.File("depth/depth.exr");
    const std::array<uint16_t, 2> depthBytes = {0, 65535};
    arm::capture::DepthPayload    depth("depth", depthPath.string());
    const std::filesystem::path   motionPath  = output.File("motion/motion.exr");
    const std::array<uint8_t, 4>  motionBytes = {0, 255, 255, 0};
    arm::capture::ScaledPayload   motion("motion", motionPath.string(), {0.5f, 2.0f});

    depth.Write({depthBytes.data(), sizeof(depthBytes), FFX_API_SURFACE_FORMAT_R16_UNORM, {2, 1}});
    motion.Write({motionBytes.data(), motionBytes.size(), FFX_API_SURFACE_FORMAT_R8G8_UNORM, {2, 1}});

    ExrReader depthImage;
    UNIT_TEST_CHECK(depthImage.Load(depthPath));
    UNIT_TEST_CHECK(depthImage.Width() == 2);
    UNIT_TEST_CHECK(depthImage.Height() == 1);
    UNIT_TEST_CHECK(depthImage.HasFloatChannels(1));
    UNIT_TEST_CHECK(CloseEnough(depthImage.FloatPixel("R", 0), 0.0f));
    UNIT_TEST_CHECK(CloseEnough(depthImage.FloatPixel("R", 1), 1.0f));

    ExrReader motionImage;
    UNIT_TEST_CHECK(motionImage.Load(motionPath));
    UNIT_TEST_CHECK(motionImage.Width() == 2);
    UNIT_TEST_CHECK(motionImage.Height() == 1);
    UNIT_TEST_CHECK(motionImage.HasFloatChannels(3));
    UNIT_TEST_CHECK(CloseEnough(motionImage.FloatPixel("R", 0), -1.0f));
    UNIT_TEST_CHECK(CloseEnough(motionImage.FloatPixel("G", 0), 1.0f));
    UNIT_TEST_CHECK(CloseEnough(motionImage.FloatPixel("B", 0), 0.0f));
    UNIT_TEST_CHECK(CloseEnough(motionImage.FloatPixel("R", 1), 1.0f));
    UNIT_TEST_CHECK(CloseEnough(motionImage.FloatPixel("G", 1), -1.0f));
    return true;
}

UNIT_TEST_CASE(capture, payload_motion_non_default_conversion)
{
    OutputDirectory              output("non_default_motion");
    const std::filesystem::path  motionPath  = output.File("motion.exr");
    const std::array<uint8_t, 2> motionBytes = {128, 64};
    arm::capture::ScaledPayload  motion("motion", motionPath.string(), {0.25f, 4.0f});

    UNIT_TEST_CHECK(motion.Write({motionBytes.data(), motionBytes.size(), FFX_API_SURFACE_FORMAT_R8G8_UNORM, {1, 1}}));

    ExrReader motionImage;
    UNIT_TEST_CHECK(motionImage.Load(motionPath));
    UNIT_TEST_CHECK(CloseEnough(motionImage.FloatPixel("R", 0), (128.0f / 255.0f - 0.25f) * 4.0f));
    UNIT_TEST_CHECK(CloseEnough(motionImage.FloatPixel("G", 0), (64.0f / 255.0f - 0.25f) * 4.0f));
    return true;
}

UNIT_TEST_CASE(capture, payload_depth_d24)
{
    OutputDirectory               output("r32_uint_depth");
    const std::filesystem::path   inputPath          = output.File("depth.bin");
    const std::filesystem::path   exrPath            = output.File("depth.exr");
    const std::filesystem::path   rawExrPath         = output.File("depth_d24.exr");
    const std::filesystem::path   oversizedInputPath = output.File("depth_oversized.bin");
    const std::filesystem::path   oversizedExrPath   = output.File("depth_oversized.exr");
    const std::array<uint32_t, 2> depth              = {0x00000000u, 0x00ffffffu};

    std::filesystem::create_directories(inputPath.parent_path());
    {
        std::ofstream input(inputPath, std::ios::binary);
        input.write(reinterpret_cast<const char*>(depth.data()), sizeof(depth));
    }

    arm::capture::DepthPayload payload("depth", exrPath.string());
    payload.Write({depth.data(), sizeof(depth), FFX_API_SURFACE_FORMAT_R32_UINT, {2, 1}});
    UNIT_TEST_CHECK(!std::filesystem::exists(exrPath));

    UNIT_TEST_CHECK(arm::capture::ConvertRawImageToExr(
        inputPath.string(), rawExrPath.string(), FFX_API_SURFACE_FORMAT_R32_UINT, 2, 1, arm::capture::ImageKind::Depth, 0.5f, 2.0f, true));
    UNIT_TEST_CHECK(std::filesystem::is_regular_file(rawExrPath));
    {
        std::ofstream input(oversizedInputPath, std::ios::binary);
        input.write(reinterpret_cast<const char*>(depth.data()), sizeof(depth));
        const uint8_t trailingByte = 0xff;
        input.write(reinterpret_cast<const char*>(&trailingByte), sizeof(trailingByte));
    }
    UNIT_TEST_CHECK(arm::capture::ConvertRawImageToExr(
        oversizedInputPath.string(), oversizedExrPath.string(), FFX_API_SURFACE_FORMAT_R32_UINT, 2, 1, arm::capture::ImageKind::Depth, 0.5f, 2.0f, true));
    UNIT_TEST_CHECK(std::filesystem::is_regular_file(oversizedExrPath));

    ExrReader image;
    UNIT_TEST_CHECK(image.Load(rawExrPath));
    UNIT_TEST_CHECK(image.Width() == 2);
    UNIT_TEST_CHECK(image.Height() == 1);
    UNIT_TEST_CHECK(CloseEnough(image.FloatPixel("R", 0), 0.0f));
    UNIT_TEST_CHECK(CloseEnough(image.FloatPixel("R", 1), 1.0f));
    return true;
}

UNIT_TEST_CASE(capture, payload_r32g32_raw_conversion)
{
    OutputDirectory             output("r32g32_raw");
    const std::filesystem::path inputPath     = output.File("motion.bin");
    const std::filesystem::path outputPath    = output.File("motion.exr");
    const std::filesystem::path truncatedPath = output.File("truncated.bin");
    const std::filesystem::path truncatedExr  = output.File("truncated.exr");
    const std::array<float, 2>  motion        = {-0.25f, 0.75f};
    const std::array<float, 1>  truncated     = {motion[0]};

    std::filesystem::create_directories(inputPath.parent_path());
    {
        std::ofstream input(inputPath, std::ios::binary);
        input.write(reinterpret_cast<const char*>(motion.data()), sizeof(motion));
    }
    {
        std::ofstream input(truncatedPath, std::ios::binary);
        input.write(reinterpret_cast<const char*>(truncated.data()), sizeof(truncated));
    }

    UNIT_TEST_CHECK(arm::capture::ConvertRawImageToExr(
        inputPath.string(), outputPath.string(), FFX_API_SURFACE_FORMAT_R32G32_FLOAT, 1, 1, arm::capture::ImageKind::Motion));
    UNIT_TEST_CHECK(!arm::capture::ConvertRawImageToExr(
        truncatedPath.string(), truncatedExr.string(), FFX_API_SURFACE_FORMAT_R32G32_FLOAT, 1, 1, arm::capture::ImageKind::Motion));
    UNIT_TEST_CHECK(std::filesystem::is_regular_file(outputPath));
    UNIT_TEST_CHECK(!std::filesystem::exists(truncatedExr));

    ExrReader image;
    UNIT_TEST_CHECK(image.Load(outputPath));
    UNIT_TEST_CHECK(image.Width() == 1);
    UNIT_TEST_CHECK(image.Height() == 1);
    UNIT_TEST_CHECK(image.HasFloatChannels(3));
    UNIT_TEST_CHECK(CloseEnough(image.FloatPixel("R", 0), motion[0]));
    UNIT_TEST_CHECK(CloseEnough(image.FloatPixel("G", 0), motion[1]));
    UNIT_TEST_CHECK(CloseEnough(image.FloatPixel("B", 0), 0.0f));
    return true;
}

UNIT_TEST_CASE(capture, payload_optical_flow_raw_conversion)
{
    OutputDirectory             output("optical_flow_raw");
    const std::filesystem::path inputPath  = output.File("optical_flow.bin");
    const std::filesystem::path outputPath = output.File("optical_flow.exr");
    const std::array<float, 2>  flow       = {-0.25f, 0.75f};

    std::filesystem::create_directories(inputPath.parent_path());
    {
        std::ofstream input(inputPath, std::ios::binary);
        input.write(reinterpret_cast<const char*>(flow.data()), sizeof(flow));
    }

    UNIT_TEST_CHECK(arm::capture::ConvertRawImageToExr(
        inputPath.string(), outputPath.string(), FFX_API_SURFACE_FORMAT_R32G32_FLOAT, 1, 1, arm::capture::ImageKind::OpticalFlow));

    ExrReader image;
    UNIT_TEST_CHECK(image.Load(outputPath));
    UNIT_TEST_CHECK(image.Width() == 1);
    UNIT_TEST_CHECK(image.Height() == 1);
    UNIT_TEST_CHECK(image.HasFloatChannels(3));
    UNIT_TEST_CHECK(CloseEnough(image.FloatPixel("R", 0), flow[0]));
    UNIT_TEST_CHECK(CloseEnough(image.FloatPixel("G", 0), flow[1]));
    UNIT_TEST_CHECK(CloseEnough(image.FloatPixel("B", 0), 0.0f));
    return true;
}

UNIT_TEST_CASE(capture, payload_invalid_view)
{
    OutputDirectory              output("invalid");
    const std::array<uint8_t, 4> bytes         = {1, 2, 3, 4};
    const auto                   unknownPath   = output.File("unknown.exr");
    const auto                   shortPath     = output.File("short.exr");
    const auto                   oversizedPath = output.File("oversized.exr");
    const auto                   zeroPath      = output.File("zero.exr");

    arm::capture::NormalPayload unknown("unknown", unknownPath.string());
    arm::capture::NormalPayload shortPayload("short", shortPath.string());
    arm::capture::NormalPayload oversized("oversized", oversizedPath.string());
    arm::capture::NormalPayload zero("zero", zeroPath.string());
    unknown.Write({bytes.data(), bytes.size(), FFX_API_SURFACE_FORMAT_UNKNOWN, {1, 1}});
    shortPayload.Write({bytes.data(), 3, FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM, {1, 1}});
    UNIT_TEST_CHECK(oversized.Write({bytes.data(), 5, FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM, {1, 1}}));
    zero.Write({bytes.data(), bytes.size(), FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM, {0, 1}});

    UNIT_TEST_CHECK(!std::filesystem::exists(unknownPath));
    UNIT_TEST_CHECK(!std::filesystem::exists(shortPath));
    UNIT_TEST_CHECK(std::filesystem::is_regular_file(oversizedPath));
    UNIT_TEST_CHECK(!std::filesystem::exists(zeroPath));
    return true;
}

UNIT_TEST_CASE(capture, path_unc_parent)
{
#if defined(_WIN32)
    UNIT_TEST_CHECK(arm::capture::EnsureParentDirectory(R"(\\capture-server\dataset-share\payload.bin)"));
#endif
    return true;
}

UNIT_TEST_CASE(capture, path_builder_rejects_truncation)
{
    std::string path;
    UNIT_TEST_CHECK(arm::capture::TryBuildCapturePath(path, "capture", "x2/colour/0000", 0, ".exr"));
    UNIT_TEST_CHECK(path == "capture/x2/colour/0000/0000.exr");

    const std::string commonPrefix(640, 'a');
    std::string       firstPath;
    std::string       secondPath;
    UNIT_TEST_CHECK(!arm::capture::TryBuildCapturePath(firstPath, commonPrefix + "first", "x2/colour/0000", 0, ".exr"));
    UNIT_TEST_CHECK(!arm::capture::TryBuildCapturePath(secondPath, commonPrefix + "second", "x2/colour/0000", 0, ".exr"));
    UNIT_TEST_CHECK(firstPath.empty());
    UNIT_TEST_CHECK(secondPath.empty());
    return true;
}

UNIT_TEST_CASE(capture, payload_factory_kind)
{
    arm::capture::Config config{};
    config.motionVectorUnormBias  = 0.5f;
    config.motionVectorUnormScale = 2.0f;

    UNIT_TEST_CHECK(dynamic_cast<arm::capture::NormalPayload*>(
                        arm::capture::CreateCapturePayload(config, {"color", arm::capture::ImageKind::Color, "color.exr"}).get()) != nullptr);
    UNIT_TEST_CHECK(dynamic_cast<arm::capture::DepthPayload*>(
                        arm::capture::CreateCapturePayload(config, {"depth", arm::capture::ImageKind::Depth, "depth.exr"}).get()) != nullptr);
    UNIT_TEST_CHECK(dynamic_cast<arm::capture::ScaledPayload*>(
                        arm::capture::CreateCapturePayload(config, {"motion", arm::capture::ImageKind::Motion, "motion.exr"}).get()) != nullptr);
    UNIT_TEST_CHECK(dynamic_cast<arm::capture::NormalPayload*>(
                        arm::capture::CreateCapturePayload(config, {"output", arm::capture::ImageKind::Output, "output.exr"}).get()) != nullptr);

    config.imageOutputEncoding = arm::capture::ImageOutputEncoding::Raw;
    UNIT_TEST_CHECK(dynamic_cast<arm::capture::BinaryPayload*>(
                        arm::capture::CreateCapturePayload(config, {"color", arm::capture::ImageKind::Color, "color.bin"}).get()) != nullptr);
    return true;
}

namespace
{
    using namespace arm::capture;
    using namespace arm::capture::detail;

    class FakeMetadataSink final : public FrameMetadataSink
    {
    public:
        explicit FakeMetadataSink(bool completeResult = true, bool finalizeResult = true)
            : completeResult(completeResult)
            , finalizeResult(finalizeResult)
        {
        }

        bool CompleteFrame(std::unique_ptr<FrameMeta> frame) override
        {
            completedDatasetIndices.push_back(frame->datasetIndex);
            ++completeCount;
            return completeResult;
        }

        bool FinalizeMetadata() override
        {
            ++finalizeCount;
            return finalizeResult;
        }

        bool                  completeResult;
        bool                  finalizeResult;
        std::vector<uint32_t> completedDatasetIndices;
        std::atomic<uint32_t> finalizeCount{0};
        std::atomic<uint32_t> completeCount{0};
    };

    class FailingPayload final : public Payload
    {
    public:
        const char* GetName() const override
        {
            return "failing";
        }

        bool Write(const PayloadView&) override
        {
            return false;
        }
    };

    class TestStagingPageSink final : public StagingPageSink
    {
    public:
        explicit TestStagingPageSink(std::function<void(VkBuffer, VkDeviceMemory, VkDeviceSize)> publish = {})
            : publish_(std::move(publish))
        {
        }

        void PublishPage(VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize bytes) override
        {
            if (publish_)
                publish_(buffer, memory, bytes);
        }

        void MarkReleaseUnsafe() override
        {
            releaseSafe_.store(false, std::memory_order_release);
        }

        bool IsReleaseSafe() const override
        {
            return releaseSafe_.load(std::memory_order_acquire);
        }

    private:
        std::function<void(VkBuffer, VkDeviceMemory, VkDeviceSize)> publish_;
        std::atomic_bool                                            releaseSafe_{true};
    };

    VkDeviceContext MakeDeviceContext(const unit_test::vk::VulkanContext& context)
    {
        VkDeviceContext deviceContext{};
        deviceContext.vkDevice              = context.device;
        deviceContext.vkPhysicalDevice      = context.physicalDevice;
        deviceContext.vkInstance            = context.instance;
        deviceContext.vkDeviceProcAddr      = context.getDeviceProcAddr;
        deviceContext.vkGetInstanceProcAddr = context.getInstanceProcAddr;
        return deviceContext;
    }

    bool InitializeCaptureContext(unit_test::vk::VulkanContext& context)
    {
        std::string error;
        return unit_test::vk::InitializeVulkanContext(context, &error) && InitVulkanWrapper(MakeDeviceContext(context));
    }

    SessionDescriptor MakeDescriptor(const unit_test::vk::VulkanContext& context, const std::shared_ptr<FakeMetadataSink>& sink)
    {
        SessionDescriptor descriptor(nullptr, 1024, sink);
        descriptor.device         = context.device;
        descriptor.physicalDevice = context.physicalDevice;
        return descriptor;
    }

    class QueryFailureWorkerPlatform final : public WorkerPlatform
    {
    public:
        QueryFailureWorkerPlatform(VkResult                             terminalResult,
                                   VkDevice                             device,
                                   VkPhysicalDevice                     physicalDevice,
                                   std::shared_ptr<StagingPagePlatform> stagingPlatform = nullptr)
            : terminalResult_(terminalResult)
            , device_(device)
            , physicalDevice_(physicalDevice)
            , stagingPlatform_(std::move(stagingPlatform))
        {
        }

        bool SupportsCapture(VkDevice, VkPhysicalDevice) const override
        {
            return captureSupported.load(std::memory_order_acquire);
        }

        VkQueryPool CreateTimestampQueryPool(VkDevice device, uint32_t queryCount) override
        {
            if (failNextQueryPoolCreation.exchange(false, std::memory_order_acq_rel))
                return VK_NULL_HANDLE;

            VkQueryPoolCreateInfo createInfo{};
            createInfo.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            createInfo.queryType  = VK_QUERY_TYPE_TIMESTAMP;
            createInfo.queryCount = queryCount;

            VkQueryPool queryPool = VK_NULL_HANDLE;
            auto        create    = reinterpret_cast<PFN_vkCreateQueryPool>(vk::GetDeviceProc(device, "vkCreateQueryPool"));
            if (create == nullptr || create(device, &createInfo, nullptr, &queryPool) != VK_SUCCESS)
                return VK_NULL_HANDLE;
            ++createQueryPoolCalls;
            queryPools.push_back(queryPool);
            return queryPool;
        }

        void DestroyQueryPool(VkDevice device, VkQueryPool queryPool) override
        {
            ++destroyQueryPoolCalls;
            if (blockQueryPoolDestroy)
            {
                std::unique_lock<std::mutex> lock(queryPoolMutex);
                queryPoolDestroyEntered = true;
                queryPoolCv.notify_all();
                queryPoolCv.wait(lock, [this] { return allowQueryPoolDestroy; });
            }
            auto destroy = reinterpret_cast<PFN_vkDestroyQueryPool>(vk::GetDeviceProc(device, "vkDestroyQueryPool"));
            if (destroy != nullptr && queryPool != VK_NULL_HANDLE)
                destroy(device, queryPool, nullptr);
        }

        void WaitForQueryPoolDestroy() const
        {
            std::unique_lock<std::mutex> lock(queryPoolMutex);
            queryPoolCv.wait(lock, [this] { return queryPoolDestroyEntered; });
        }

        void AllowQueryPoolDestroy() const
        {
            {
                std::lock_guard<std::mutex> lock(queryPoolMutex);
                allowQueryPoolDestroy = true;
            }
            queryPoolCv.notify_all();
        }

        VkResult GetQueryPoolResults(
            VkDevice, VkQueryPool queryPool, uint32_t firstQuery, uint32_t, size_t dataSize, void* data, VkDeviceSize, VkQueryResultFlags) const override
        {
            const uint32_t call = ++queryCalls;
            if (call == 1)
                return VK_NOT_READY;
            if (blockInsideQuery.load(std::memory_order_acquire))
            {
                std::unique_lock<std::mutex> lock(queryMutex);
                queryEntered = true;
                queryCv.notify_all();
                queryCv.wait(lock, [this] { return allowQueryReturn; });
            }
            if (blockQueryResults.load(std::memory_order_acquire))
                return VK_NOT_READY;
            if (blockFirstQueryPool.load(std::memory_order_acquire))
            {
                VkQueryPool expected = VK_NULL_HANDLE;
                firstQueryPool.compare_exchange_strong(expected, queryPool, std::memory_order_acq_rel);
                if (queryPool == firstQueryPool.load(std::memory_order_acquire))
                    return VK_NOT_READY;
            }
            if (terminalResult_ == VK_SUCCESS && data != nullptr && dataSize >= 2 * sizeof(uint64_t))
                static_cast<uint64_t*>(data)[1] = 1;
            return terminalResult_;
        }

        void WaitForQueryCall() const
        {
            std::unique_lock<std::mutex> lock(queryMutex);
            queryCv.wait(lock, [this] { return queryEntered; });
        }

        void AllowQueryReturn() const
        {
            {
                std::lock_guard<std::mutex> lock(queryMutex);
                allowQueryReturn = true;
            }
            queryCv.notify_all();
        }

        std::shared_ptr<StagingPool> CreateStagingPool(VkDevice, VkPhysicalDevice, const StagingPoolConfig& config) override
        {
            ++createPoolCalls;
            lastStagingPool = StagingPool::Create(device_, physicalDevice_, config, stagingPlatform_);
            return lastStagingPool;
        }

        VkResult                             terminalResult_;
        VkDevice                             device_;
        VkPhysicalDevice                     physicalDevice_;
        std::shared_ptr<StagingPagePlatform> stagingPlatform_;
        mutable std::atomic<uint32_t>        createPoolCalls{0};
        mutable std::atomic<uint32_t>        createQueryPoolCalls{0};
        mutable std::atomic<uint32_t>        queryCalls{0};
        mutable std::atomic<uint32_t>        destroyQueryPoolCalls{0};
        std::atomic_bool                     captureSupported{true};
        std::atomic_bool                     failNextQueryPoolCreation{false};
        std::vector<VkQueryPool>             queryPools;
        std::atomic_bool                     blockFirstQueryPool{false};
        std::atomic_bool                     blockQueryResults{false};
        std::atomic_bool                     blockInsideQuery{false};
        mutable std::atomic<VkQueryPool>     firstQueryPool{VK_NULL_HANDLE};
        std::shared_ptr<StagingPool>         lastStagingPool;
        bool                                 blockQueryPoolDestroy = false;
        mutable std::mutex                   queryPoolMutex;
        mutable std::condition_variable      queryPoolCv;
        mutable bool                         queryPoolDestroyEntered = false;
        mutable bool                         allowQueryPoolDestroy   = false;
        mutable std::mutex                   queryMutex;
        mutable std::condition_variable      queryCv;
        mutable bool                         queryEntered     = false;
        mutable bool                         allowQueryReturn = false;
    };

    class TrackingStagingPagePlatform final : public StagingPagePlatform
    {
    public:
        enum class Failure
        {
            None,
            Allocate,
            Bind,
            AllocateDeviceLost,
            BindDeviceLost,
            BlockAllocate,
        };

        explicit TrackingStagingPagePlatform(Failure      failure            = Failure::None,
                                             VkDeviceSize hostVisibleBytes   = 64 * kMiB,
                                             VkDeviceSize maxAllocationBytes = 64 * kMiB)
            : failure_(failure)
            , hostVisibleBytes_(hostVisibleBytes)
            , maxAllocationBytes_(maxAllocationBytes)
        {
        }

        void GetPhysicalDeviceMemoryProperties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* properties) const override
        {
            *properties                              = {};
            properties->memoryTypeCount              = 1;
            properties->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            properties->memoryTypes[0].heapIndex     = 0;
            properties->memoryHeapCount              = 1;
            properties->memoryHeaps[0].size          = hostVisibleBytes_;
        }

        void GetPhysicalDeviceProperties2(VkPhysicalDevice, VkPhysicalDeviceProperties2* properties) const override
        {
            auto* maintenance3 = reinterpret_cast<VkPhysicalDeviceMaintenance3Properties*>(properties->pNext);
            if (maintenance3 != nullptr)
                maintenance3->maxMemoryAllocationSize = maxAllocationBytes_;
            properties->properties.limits.optimalBufferCopyOffsetAlignment = 256;
            properties->properties.limits.minMemoryMapAlignment            = 256;
        }

        VkResult CreateBuffer(VkDevice, const VkBufferCreateInfo* createInfo, VkBuffer* buffer) const override
        {
            *buffer          = reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(++nextHandle_));
            lastBufferBytes_ = createInfo->size;
            ++createBufferCalls;
            return VK_SUCCESS;
        }

        void GetBufferMemoryRequirements(VkDevice, VkBuffer, VkMemoryRequirements* requirements) const override
        {
            requirements->size           = memoryRequirementBytes == 0 ? lastBufferBytes_ : memoryRequirementBytes;
            requirements->alignment      = kMiB;
            requirements->memoryTypeBits = 1;
        }

        VkResult AllocateMemory(VkDevice, const VkMemoryAllocateInfo*, VkDeviceMemory* memory) const override
        {
            ++allocateMemoryCalls;
            if (failure_ == Failure::Allocate || failure_ == Failure::AllocateDeviceLost)
                return failure_ == Failure::AllocateDeviceLost ? VK_ERROR_DEVICE_LOST : VK_ERROR_OUT_OF_DEVICE_MEMORY;
            if (failure_ == Failure::BlockAllocate)
            {
                std::unique_lock<std::mutex> lock(blockMutex_);
                allocationEntered_ = true;
                blockCv_.notify_all();
                blockCv_.wait(lock, [this] { return allowAllocation_; });
            }
            *memory = reinterpret_cast<VkDeviceMemory>(static_cast<uintptr_t>(++nextHandle_));
            return VK_SUCCESS;
        }

        void WaitForAllocation() const
        {
            std::unique_lock<std::mutex> lock(blockMutex_);
            blockCv_.wait(lock, [this] { return allocationEntered_; });
        }

        void AllowAllocation() const
        {
            {
                std::lock_guard<std::mutex> lock(blockMutex_);
                allowAllocation_ = true;
            }
            blockCv_.notify_all();
        }

        VkResult BindBufferMemory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) const override
        {
            ++bindBufferCalls;
            if (failure_ == Failure::Bind || failure_ == Failure::BindDeviceLost)
                return failure_ == Failure::BindDeviceLost ? VK_ERROR_DEVICE_LOST : VK_ERROR_MEMORY_MAP_FAILED;
            return VK_SUCCESS;
        }

        VkResult MapMemory(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, void**) const override
        {
            return VK_ERROR_MEMORY_MAP_FAILED;
        }

        void UnmapMemory(VkDevice, VkDeviceMemory) const override
        {
        }

        void DestroyBuffer(VkDevice, VkBuffer) const override
        {
            ++destroyBufferCalls;
        }

        void FreeMemory(VkDevice, VkDeviceMemory) const override
        {
            ++freeMemoryCalls;
        }

        Failure                         failure_;
        VkDeviceSize                    hostVisibleBytes_;
        VkDeviceSize                    maxAllocationBytes_;
        VkDeviceSize                    memoryRequirementBytes = 0;
        mutable uint64_t                nextHandle_            = 0;
        mutable VkDeviceSize            lastBufferBytes_       = kMiB;
        mutable uint32_t                createBufferCalls      = 0;
        mutable uint32_t                allocateMemoryCalls    = 0;
        mutable uint32_t                bindBufferCalls        = 0;
        mutable uint32_t                destroyBufferCalls     = 0;
        mutable uint32_t                freeMemoryCalls        = 0;
        mutable std::mutex              blockMutex_;
        mutable std::condition_variable blockCv_;
        mutable bool                    allocationEntered_ = false;
        mutable bool                    allowAllocation_   = false;
    };

    class TrackingVulkanStagingPagePlatform final : public StagingPagePlatform
    {
    public:
        TrackingVulkanStagingPagePlatform()
            : delegate_(CreateDefaultStagingPagePlatform())
        {
        }

        void GetPhysicalDeviceMemoryProperties(VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties* properties) const override
        {
            delegate_->GetPhysicalDeviceMemoryProperties(physicalDevice, properties);
        }

        void GetPhysicalDeviceProperties2(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2* properties) const override
        {
            delegate_->GetPhysicalDeviceProperties2(physicalDevice, properties);
        }

        VkResult CreateBuffer(VkDevice device, const VkBufferCreateInfo* createInfo, VkBuffer* buffer) const override
        {
            ++createBufferCalls;
            return delegate_->CreateBuffer(device, createInfo, buffer);
        }

        void GetBufferMemoryRequirements(VkDevice device, VkBuffer buffer, VkMemoryRequirements* requirements) const override
        {
            delegate_->GetBufferMemoryRequirements(device, buffer, requirements);
        }

        VkResult AllocateMemory(VkDevice device, const VkMemoryAllocateInfo* allocateInfo, VkDeviceMemory* memory) const override
        {
            return delegate_->AllocateMemory(device, allocateInfo, memory);
        }

        VkResult BindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset) const override
        {
            return delegate_->BindBufferMemory(device, buffer, memory, offset);
        }

        VkResult MapMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, void** data) const override
        {
            ++mapMemoryCalls;
            if (mapResult != VK_SUCCESS)
                return mapResult;
            return delegate_->MapMemory(device, memory, offset, size, data);
        }

        void UnmapMemory(VkDevice device, VkDeviceMemory memory) const override
        {
            ++unmapMemoryCalls;
            delegate_->UnmapMemory(device, memory);
        }

        void DestroyBuffer(VkDevice device, VkBuffer buffer) const override
        {
            ++destroyBufferCalls;
            delegate_->DestroyBuffer(device, buffer);
        }

        void FreeMemory(VkDevice device, VkDeviceMemory memory) const override
        {
            ++freeMemoryCalls;
            delegate_->FreeMemory(device, memory);
        }

        std::shared_ptr<StagingPagePlatform> delegate_;
        mutable std::atomic<uint32_t>        createBufferCalls{0};
        mutable std::atomic<uint32_t>        destroyBufferCalls{0};
        mutable std::atomic<uint32_t>        freeMemoryCalls{0};
        mutable std::atomic<uint32_t>        mapMemoryCalls{0};
        mutable std::atomic<uint32_t>        unmapMemoryCalls{0};
        VkResult                             mapResult = VK_SUCCESS;
    };

    FfxResource gNfruCaptureOpticalFlow{};

    FfxResource GetNfruCaptureResource(FfxInterface*, FfxResourceInternal)
    {
        return gNfruCaptureOpticalFlow;
    }

    VkDeviceSize GetHostVisibleHeapBytesForTest(VkPhysicalDevice physicalDevice)
    {
        VkPhysicalDeviceMemoryProperties properties{};
        VulkanWrapper().vkGetPhysicalDeviceMemoryProperties(physicalDevice, &properties);

        bool hostVisibleHeap[VK_MAX_MEMORY_HEAPS] = {};
        for (uint32_t index = 0; index < properties.memoryTypeCount; ++index)
        {
            if ((properties.memoryTypes[index].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
                hostVisibleHeap[properties.memoryTypes[index].heapIndex] = true;
        }

        VkDeviceSize bytes = 0;
        for (uint32_t index = 0; index < properties.memoryHeapCount; ++index)
        {
            if (hostVisibleHeap[index] && bytes <= std::numeric_limits<VkDeviceSize>::max() - properties.memoryHeaps[index].size)
                bytes += properties.memoryHeaps[index].size;
        }
        return bytes;
    }
}  // namespace

// Runtime resource management: staging-pool sharing, worker sessions, and finalization.
UNIT_TEST_CASE(capture, allocator_page_create)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    const VkDeviceSize alignment = arm::capture::GetStagingPoolOffsetAlignment(context.physicalDevice);
    const VkDeviceSize unitSize  = arm::capture::detail::StagingAllocationUnitSize(alignment);
    UNIT_TEST_CHECK(unitSize != 0);

    arm::capture::StagingPoolConfig config{};
    config.hostVisibleMemoryRatio = 0.125f;

    VkBuffer       pageBuffer = VK_NULL_HANDLE;
    VkDeviceMemory pageMemory = VK_NULL_HANDLE;
    VkDeviceSize   pageBytes  = 0;
    {
        const auto pageSink =
            std::make_shared<TestStagingPageSink>([&pageBuffer, &pageMemory, &pageBytes](VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize bytes) {
                pageBuffer = buffer;
                pageMemory = memory;
                pageBytes  = bytes;
            });
        arm::capture::StagingPageAllocatorConfig allocatorConfig{unitSize, config.hostVisibleMemoryRatio};
        auto allocator = std::make_unique<arm::capture::StagingPageAllocator>(context.device, context.physicalDevice, allocatorConfig, pageSink.get());

        UNIT_TEST_CHECK(allocator->CanAllocateBlock(unitSize));
        UNIT_TEST_CHECK(allocator->EnsurePageForBlock(unitSize));
        UNIT_TEST_CHECK(pageBuffer != VK_NULL_HANDLE);
        UNIT_TEST_CHECK(pageMemory != VK_NULL_HANDLE);
        UNIT_TEST_CHECK(pageBytes >= unitSize);
    }

    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, allocator_failed_page_cleanup)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    const VkDeviceSize              unitSize = arm::capture::detail::StagingAllocationUnitSize(kMiB);
    arm::capture::StagingPoolConfig config{};
    config.hostVisibleMemoryRatio = 1.0f;

    for (const TrackingStagingPagePlatform::Failure failure : {TrackingStagingPagePlatform::Failure::Allocate, TrackingStagingPagePlatform::Failure::Bind})
    {
        auto platform = std::make_shared<TrackingStagingPagePlatform>(failure);
        auto pageSink = std::make_shared<TestStagingPageSink>();
        {
            arm::capture::StagingPageAllocatorConfig allocatorConfig{unitSize, config.hostVisibleMemoryRatio};
            auto                                     allocator =
                std::make_unique<arm::capture::StagingPageAllocator>(context.device, context.physicalDevice, allocatorConfig, pageSink.get(), platform);
            UNIT_TEST_CHECK(!allocator->EnsurePageForBlock(unitSize));
        }

        UNIT_TEST_CHECK(platform->destroyBufferCalls == 1);
        UNIT_TEST_CHECK(platform->freeMemoryCalls == (failure == TrackingStagingPagePlatform::Failure::Bind ? 1u : 0u));
    }

    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, allocator_rejects_actual_requirement_over_remaining_budget)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    const VkDeviceSize unitSize      = arm::capture::detail::StagingAllocationUnitSize(kMiB);
    auto               platform      = std::make_shared<TrackingStagingPagePlatform>(TrackingStagingPagePlatform::Failure::None, 2 * kMiB, 64 * kMiB);
    platform->memoryRequirementBytes = 3 * kMiB;
    auto                                     pageSink = std::make_shared<TestStagingPageSink>();
    arm::capture::StagingPageAllocatorConfig allocatorConfig{unitSize, 1.0f};
    auto allocator = std::make_unique<arm::capture::StagingPageAllocator>(context.device, context.physicalDevice, allocatorConfig, pageSink.get(), platform);

    UNIT_TEST_CHECK(!allocator->EnsurePageForBlock(unitSize));
    UNIT_TEST_CHECK(platform->createBufferCalls == 1);
    UNIT_TEST_CHECK(platform->allocateMemoryCalls == 0);
    UNIT_TEST_CHECK(platform->destroyBufferCalls == 1);
    UNIT_TEST_CHECK(platform->freeMemoryCalls == 0);

    platform->memoryRequirementBytes = 0;
    UNIT_TEST_CHECK(allocator->EnsurePageForBlock(unitSize));
    UNIT_TEST_CHECK(platform->allocateMemoryCalls == 1);

    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, allocator_rejects_actual_requirement_over_allocation_cap)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    const VkDeviceSize unitSize      = arm::capture::detail::StagingAllocationUnitSize(kMiB);
    auto               platform      = std::make_shared<TrackingStagingPagePlatform>(TrackingStagingPagePlatform::Failure::None, 8 * kMiB, 2 * kMiB);
    platform->memoryRequirementBytes = 3 * kMiB;
    auto                                     pageSink = std::make_shared<TestStagingPageSink>();
    arm::capture::StagingPageAllocatorConfig allocatorConfig{unitSize, 1.0f};
    auto allocator = std::make_unique<arm::capture::StagingPageAllocator>(context.device, context.physicalDevice, allocatorConfig, pageSink.get(), platform);

    UNIT_TEST_CHECK(!allocator->EnsurePageForBlock(unitSize));
    UNIT_TEST_CHECK(platform->createBufferCalls == 1);
    UNIT_TEST_CHECK(platform->allocateMemoryCalls == 0);
    UNIT_TEST_CHECK(platform->destroyBufferCalls == 1);
    UNIT_TEST_CHECK(platform->freeMemoryCalls == 0);

    platform->memoryRequirementBytes = 0;
    UNIT_TEST_CHECK(allocator->EnsurePageForBlock(unitSize));
    UNIT_TEST_CHECK(platform->allocateMemoryCalls == 1);

    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, allocator_device_lost_abandons_partial_page)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    const VkDeviceSize              unitSize = arm::capture::detail::StagingAllocationUnitSize(kMiB);
    arm::capture::StagingPoolConfig config{};
    config.hostVisibleMemoryRatio = 1.0f;

    for (const TrackingStagingPagePlatform::Failure failure :
         {TrackingStagingPagePlatform::Failure::AllocateDeviceLost, TrackingStagingPagePlatform::Failure::BindDeviceLost})
    {
        auto                                     platform = std::make_shared<TrackingStagingPagePlatform>(failure);
        auto                                     pageSink = std::make_shared<TestStagingPageSink>();
        arm::capture::StagingPageAllocatorConfig allocatorConfig{unitSize, config.hostVisibleMemoryRatio};
        auto                                     allocator =
            std::make_unique<arm::capture::StagingPageAllocator>(context.device, context.physicalDevice, allocatorConfig, pageSink.get(), platform);
        UNIT_TEST_CHECK(!allocator->EnsurePageForBlock(unitSize));

        UNIT_TEST_CHECK(platform->destroyBufferCalls == 0);
        UNIT_TEST_CHECK(platform->freeMemoryCalls == 0);
    }

    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, pool_acquire_return)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    arm::capture::StagingPoolConfig config{};
    config.hostVisibleMemoryRatio                           = 0.125f;
    auto                                       pagePlatform = std::make_shared<TrackingStagingPagePlatform>();
    std::shared_ptr<arm::capture::StagingPool> pool         = arm::capture::StagingPool::Create(context.device, context.physicalDevice, config, pagePlatform);
    UNIT_TEST_CHECK(pool != nullptr);
    const std::weak_ptr<arm::capture::StagingPool> weakPool = pool;

    const VkDeviceSize unitSize = arm::capture::detail::StagingAllocationUnitSize(pool->OffsetAlignment());
    UNIT_TEST_CHECK(unitSize != 0);
    UNIT_TEST_CHECK(pool->Supports(unitSize));

    arm::capture::StagingBlock first = pool->Acquire(unitSize);
    UNIT_TEST_CHECK(first);
    UNIT_TEST_CHECK(first.buffer != VK_NULL_HANDLE);
    UNIT_TEST_CHECK(first.memory != VK_NULL_HANDLE);
    UNIT_TEST_CHECK(first.size == unitSize);
    UNIT_TEST_CHECK(first.unitCount == 1);

    const VkDeviceSize unsupportedSize = std::numeric_limits<VkDeviceSize>::max();
    UNIT_TEST_CHECK(!pool->Supports(unsupportedSize));
    UNIT_TEST_CHECK(!pool->Acquire(unsupportedSize));

    const VkBuffer     firstBuffer = first.buffer;
    const VkDeviceSize firstOffset = first.offset;
    pool->Return(first);
    UNIT_TEST_CHECK(!first);

    arm::capture::StagingBlock second = pool->Acquire(unitSize);
    UNIT_TEST_CHECK(second);
    UNIT_TEST_CHECK(second.buffer == firstBuffer);
    UNIT_TEST_CHECK(second.offset == firstOffset);
    pool->Return(second);
    UNIT_TEST_CHECK(!second);

    pool.reset();
    UNIT_TEST_CHECK(weakPool.expired());
    UNIT_TEST_CHECK(pagePlatform->destroyBufferCalls == 1);
    UNIT_TEST_CHECK(pagePlatform->freeMemoryCalls == 1);
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, pool_rejects_block_with_unrepresentable_unit_count)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    constexpr VkDeviceSize hugeBlockBytes = 4ull * 1024 * 1024 * 1024 * 1024 * 1024;
    auto pagePlatform = std::make_shared<TrackingStagingPagePlatform>(TrackingStagingPagePlatform::Failure::None, hugeBlockBytes, hugeBlockBytes);
    arm::capture::StagingPoolConfig config{};
    config.hostVisibleMemoryRatio                   = 1.0f;
    std::shared_ptr<arm::capture::StagingPool> pool = arm::capture::StagingPool::Create(context.device, context.physicalDevice, config, pagePlatform);
    UNIT_TEST_CHECK(pool != nullptr);

    const VkDeviceSize unitSize = arm::capture::detail::StagingAllocationUnitSize(pool->OffsetAlignment());
    UNIT_TEST_CHECK(unitSize != 0);
    UNIT_TEST_CHECK(hugeBlockBytes / unitSize > std::numeric_limits<uint32_t>::max());

    StagingBlock block = pool->Acquire(unitSize);
    UNIT_TEST_CHECK(block);
    const uint32_t pageCreateCalls = pagePlatform->createBufferCalls;

    UNIT_TEST_CHECK(!pool->Supports(hugeBlockBytes));
    UNIT_TEST_CHECK(!pool->Acquire(hugeBlockBytes));
    UNIT_TEST_CHECK(pagePlatform->createBufferCalls == pageCreateCalls);

    pool->Return(block);
    pool.reset();
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, worker_releases_pages_after_normal_session)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    auto                         pagePlatform = std::make_shared<TrackingStagingPagePlatform>();
    auto                         platform = std::make_shared<QueryFailureWorkerPlatform>(VK_NOT_READY, context.device, context.physicalDevice, pagePlatform);
    arm::capture::detail::Worker worker(platform);
    auto                         sink = std::make_shared<FakeMetadataSink>();

    arm::capture::SessionDescriptor descriptor(nullptr, 1024, sink);
    descriptor.device                         = context.device;
    descriptor.physicalDevice                 = context.physicalDevice;
    const arm::capture::SessionHandle session = worker.Create(descriptor);
    UNIT_TEST_CHECK(session != arm::capture::InvalidSession);

    uint32_t slotIndex = UINT32_MAX;
    UNIT_TEST_CHECK(worker.AcquireSlot(session, 1024, slotIndex));
    worker.Close(session);
    UNIT_TEST_CHECK(worker.Wait(session, 1000));
    worker.Destroy(session);

    UNIT_TEST_CHECK(platform->destroyQueryPoolCalls == 1);
    UNIT_TEST_CHECK(pagePlatform->createBufferCalls == 1);
    UNIT_TEST_CHECK(pagePlatform->destroyBufferCalls == 1);
    UNIT_TEST_CHECK(pagePlatform->freeMemoryCalls == 1);

    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, worker_waits_for_query_pool_destroy)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    auto platform                   = std::make_shared<QueryFailureWorkerPlatform>(VK_NOT_READY, context.device, context.physicalDevice);
    platform->blockQueryPoolDestroy = true;
    arm::capture::detail::Worker worker(platform);
    auto                         sink = std::make_shared<FakeMetadataSink>();

    arm::capture::SessionDescriptor descriptor(nullptr, 1024, sink);
    descriptor.device                         = context.device;
    descriptor.physicalDevice                 = context.physicalDevice;
    const arm::capture::SessionHandle session = worker.Create(descriptor);
    UNIT_TEST_CHECK(session != arm::capture::InvalidSession);

    worker.Close(session);
    platform->WaitForQueryPoolDestroy();
    UNIT_TEST_CHECK(!worker.Wait(session, 10));
    platform->AllowQueryPoolDestroy();
    UNIT_TEST_CHECK(worker.Wait(session, 1000));
    worker.Destroy(session);
    UNIT_TEST_CHECK(platform->destroyQueryPoolCalls == 1);

    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, worker_releases_shared_pool_after_last_session)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    auto                         pagePlatform = std::make_shared<TrackingStagingPagePlatform>();
    auto                         platform = std::make_shared<QueryFailureWorkerPlatform>(VK_NOT_READY, context.device, context.physicalDevice, pagePlatform);
    arm::capture::detail::Worker worker(platform);
    auto                         sink = std::make_shared<FakeMetadataSink>();

    arm::capture::SessionDescriptor descriptor(nullptr, 1024, sink);
    descriptor.device                        = context.device;
    descriptor.physicalDevice                = context.physicalDevice;
    const arm::capture::SessionHandle first  = worker.Create(descriptor);
    const arm::capture::SessionHandle second = worker.Create(descriptor);
    UNIT_TEST_CHECK(first != arm::capture::InvalidSession);
    UNIT_TEST_CHECK(second != arm::capture::InvalidSession);

    uint32_t firstSlot = UINT32_MAX;
    UNIT_TEST_CHECK(worker.AcquireSlot(first, 1024, firstSlot));
    worker.Close(first);
    UNIT_TEST_CHECK(worker.Wait(first, 1000));
    worker.Destroy(first);
    UNIT_TEST_CHECK(pagePlatform->destroyBufferCalls == 0);
    UNIT_TEST_CHECK(pagePlatform->freeMemoryCalls == 0);

    worker.Close(second);
    UNIT_TEST_CHECK(worker.Wait(second, 1000));
    worker.Destroy(second);
    UNIT_TEST_CHECK(pagePlatform->destroyBufferCalls == 1);
    UNIT_TEST_CHECK(pagePlatform->freeMemoryCalls == 1);

    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, worker_parallel_sessions_share_pool)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    auto                         pagePlatform = std::make_shared<TrackingStagingPagePlatform>();
    auto                         platform = std::make_shared<QueryFailureWorkerPlatform>(VK_NOT_READY, context.device, context.physicalDevice, pagePlatform);
    arm::capture::detail::Worker worker(platform);
    auto                         firstSink  = std::make_shared<FakeMetadataSink>();
    auto                         secondSink = std::make_shared<FakeMetadataSink>();

    const arm::capture::SessionHandle first  = worker.Create(MakeDescriptor(context, firstSink));
    const arm::capture::SessionHandle second = worker.Create(MakeDescriptor(context, secondSink));
    UNIT_TEST_CHECK(first != arm::capture::InvalidSession);
    UNIT_TEST_CHECK(second != arm::capture::InvalidSession);

    std::atomic_bool firstReady{false};
    std::atomic_bool secondReady{false};
    std::atomic_bool start{false};
    std::atomic_bool firstSucceeded{false};
    std::atomic_bool secondSucceeded{false};

    std::thread firstThread([&] {
        firstReady.store(true, std::memory_order_release);
        while (!start.load(std::memory_order_acquire))
            std::this_thread::yield();

        uint32_t   slotIndex = UINT32_MAX;
        const bool acquired  = worker.AcquireSlot(first, 1024, slotIndex);
        if (acquired)
        {
            worker.Close(first);
            firstSucceeded.store(worker.Wait(first, 1000), std::memory_order_release);
            worker.Destroy(first);
        }
    });
    std::thread secondThread([&] {
        secondReady.store(true, std::memory_order_release);
        while (!start.load(std::memory_order_acquire))
            std::this_thread::yield();

        uint32_t   slotIndex = UINT32_MAX;
        const bool acquired  = worker.AcquireSlot(second, 1024, slotIndex);
        if (acquired)
        {
            worker.Close(second);
            secondSucceeded.store(worker.Wait(second, 1000), std::memory_order_release);
            worker.Destroy(second);
        }
    });

    while (!firstReady.load(std::memory_order_acquire) || !secondReady.load(std::memory_order_acquire))
        std::this_thread::yield();
    start.store(true, std::memory_order_release);
    firstThread.join();
    secondThread.join();

    UNIT_TEST_CHECK(firstSucceeded.load(std::memory_order_acquire));
    UNIT_TEST_CHECK(secondSucceeded.load(std::memory_order_acquire));
    UNIT_TEST_CHECK(platform->createPoolCalls == 1);
    UNIT_TEST_CHECK(pagePlatform->createBufferCalls == 1);
    UNIT_TEST_CHECK(pagePlatform->destroyBufferCalls == 1);
    UNIT_TEST_CHECK(pagePlatform->freeMemoryCalls == 1);

    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, pool_budget_exhaust)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    constexpr VkDeviceSize targetBudgetUnits = 4;
    const VkDeviceSize     hostVisibleBytes  = GetHostVisibleHeapBytesForTest(context.physicalDevice);
    const VkDeviceSize     alignment         = arm::capture::GetStagingPoolOffsetAlignment(context.physicalDevice);
    const VkDeviceSize     unitSize          = arm::capture::detail::StagingAllocationUnitSize(alignment);
    UNIT_TEST_CHECK(hostVisibleBytes != 0);
    UNIT_TEST_CHECK(unitSize != 0);

    const double targetBudgetBytes = static_cast<double>(targetBudgetUnits * unitSize);
    const float  budgetRatio       = static_cast<float>(targetBudgetBytes / static_cast<double>(hostVisibleBytes));
    UNIT_TEST_CHECK(budgetRatio > 0.0f);
    UNIT_TEST_CHECK(budgetRatio <= 1.0f);

    arm::capture::StagingPoolConfig config{};
    config.hostVisibleMemoryRatio                   = budgetRatio;
    std::shared_ptr<arm::capture::StagingPool> pool = arm::capture::StagingPool::Create(context.device, context.physicalDevice, config);
    UNIT_TEST_CHECK(pool != nullptr);
    UNIT_TEST_CHECK(pool->Supports(unitSize));

    std::vector<arm::capture::StagingBlock> blocks;
    bool                                    exhausted = false;
    for (VkDeviceSize index = 0; index < targetBudgetUnits + 2; ++index)
    {
        arm::capture::StagingBlock block = pool->Acquire(unitSize);
        if (!block)
        {
            exhausted = true;
            break;
        }
        blocks.push_back(block);
    }

    UNIT_TEST_CHECK(exhausted);
    UNIT_TEST_CHECK(blocks.size() >= 2);
    UNIT_TEST_CHECK(blocks.size() <= targetBudgetUnits + 1);
    for (arm::capture::StagingBlock& block : blocks)
        pool->Return(block);

    arm::capture::StagingBlock recovered = pool->Acquire(unitSize);
    UNIT_TEST_CHECK(recovered);
    pool->Return(recovered);

    pool.reset();
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, worker_rotates_pending_query_polling_across_sessions)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    unit_test::vk::Buffer buffer{};
    std::string           error;
    UNIT_TEST_CHECK(unit_test::vk::CreateBuffer(context,
                                                64,
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                buffer,
                                                &error));
    auto platform = std::make_shared<QueryFailureWorkerPlatform>(VK_SUCCESS, context.device, context.physicalDevice);
    platform->blockFirstQueryPool.store(true, std::memory_order_release);
    Worker              worker(platform);
    OutputDirectory     output("session_poll");
    auto                firstSink  = std::make_shared<FakeMetadataSink>();
    auto                secondSink = std::make_shared<FakeMetadataSink>();
    const SessionHandle first      = worker.Create(MakeDescriptor(context, firstSink));
    const SessionHandle second     = worker.Create(MakeDescriptor(context, secondSink));
    UNIT_TEST_CHECK(first != InvalidSession);
    UNIT_TEST_CHECK(second != InvalidSession);

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType                   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size                    = 64;
    bufferCreateInfo.usage                   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const FfxResourceDescription description = ffxGetBufferResourceDescriptionVK(buffer.buffer, bufferCreateInfo, FFX_RESOURCE_USAGE_READ_ONLY);
    const FfxResource            source = ffxGetResourceVK(reinterpret_cast<void*>(buffer.buffer), description, "session_poll", FFX_RESOURCE_STATE_COPY_SRC);

    uint32_t firstSlot = UINT32_MAX;
    UNIT_TEST_CHECK(worker.AcquireSlot(first, 64, firstSlot));
    VkCommandBuffer firstCommandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, firstCommandBuffer, &error));
    UNIT_TEST_CHECK(worker.RecordBuffer(
        first, firstSlot, firstCommandBuffer, source, std::make_unique<BinaryPayload>("first", output.File("session_poll_first.bin").string())));
    auto firstFrameMeta = std::make_unique<FrameMeta>();
    UNIT_TEST_CHECK(worker.Submit(first, firstSlot, firstCommandBuffer, &firstFrameMeta));
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(context, firstCommandBuffer, &error));

    uint32_t secondSlot = UINT32_MAX;
    UNIT_TEST_CHECK(worker.AcquireSlot(second, 64, secondSlot));
    VkCommandBuffer secondCommandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, secondCommandBuffer, &error));
    UNIT_TEST_CHECK(worker.RecordBuffer(
        second, secondSlot, secondCommandBuffer, source, std::make_unique<BinaryPayload>("second", output.File("session_poll_second.bin").string())));
    auto secondFrameMeta = std::make_unique<FrameMeta>();
    UNIT_TEST_CHECK(worker.Submit(second, secondSlot, secondCommandBuffer, &secondFrameMeta));
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(context, secondCommandBuffer, &error));

    const auto queryObservedDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (platform->firstQueryPool.load(std::memory_order_acquire) == VK_NULL_HANDLE && std::chrono::steady_clock::now() < queryObservedDeadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const VkQueryPool firstPolledQueryPool = platform->firstQueryPool.load(std::memory_order_acquire);
    UNIT_TEST_CHECK(firstPolledQueryPool != VK_NULL_HANDLE);
    UNIT_TEST_CHECK(platform->queryPools.size() == 2);

    const SessionHandle blockedSession = firstPolledQueryPool == platform->queryPools[0] ? first : second;
    const SessionHandle readySession   = blockedSession == first ? second : first;
    worker.Close(blockedSession);
    worker.Close(readySession);

    const bool readySessionFinished = worker.Wait(readySession, 1000);
    platform->blockFirstQueryPool.store(false, std::memory_order_release);
    const bool blockedSessionFinished = worker.Wait(blockedSession, 5000);
    UNIT_TEST_CHECK(readySessionFinished);
    UNIT_TEST_CHECK(blockedSessionFinished);

    worker.Destroy(first);
    worker.Destroy(second);
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, query_set_rotates_one_shot_pools)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    auto     platform = std::make_shared<QueryFailureWorkerPlatform>(VK_SUCCESS, context.device, context.physicalDevice);
    QuerySet querySet;
    UNIT_TEST_CHECK(querySet.Initialize(platform, context.device, 500));

    std::vector<CompletionQuery> queries;
    queries.reserve(501);
    for (uint32_t index = 0; index < 501; ++index)
    {
        const std::optional<CompletionQuery> query = querySet.Acquire();
        UNIT_TEST_CHECK(query.has_value());
        queries.push_back(*query);
    }

    UNIT_TEST_CHECK(platform->createQueryPoolCalls == 2);
    UNIT_TEST_CHECK(queries[0].index == 0);
    UNIT_TEST_CHECK(queries[499].index == 499);
    UNIT_TEST_CHECK(queries[500].index == 0);
    UNIT_TEST_CHECK(queries[0].pool != queries[500].pool);

    uint64_t       resultValues[2]    = {};
    const VkResult firstResult        = platform->GetQueryPoolResults(context.device,
                                                               queries[0].pool,
                                                               queries[0].index,
                                                               1,
                                                               sizeof(resultValues),
                                                               resultValues,
                                                               sizeof(resultValues),
                                                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    const VkResult oldPoolReadyResult = platform->GetQueryPoolResults(context.device,
                                                                      queries[0].pool,
                                                                      queries[0].index,
                                                                      1,
                                                                      sizeof(resultValues),
                                                                      resultValues,
                                                                      sizeof(resultValues),
                                                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    const VkResult newPoolResult      = platform->GetQueryPoolResults(context.device,
                                                                 queries[500].pool,
                                                                 queries[500].index,
                                                                 1,
                                                                 sizeof(resultValues),
                                                                 resultValues,
                                                                 sizeof(resultValues),
                                                                 VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    UNIT_TEST_CHECK(firstResult == VK_NOT_READY);
    UNIT_TEST_CHECK(oldPoolReadyResult == VK_SUCCESS);
    UNIT_TEST_CHECK(newPoolResult == VK_SUCCESS);

    for (uint32_t index = 0; index < 500; ++index)
        querySet.ReleaseQuery(queries[index]);
    UNIT_TEST_CHECK(platform->destroyQueryPoolCalls == 1);

    querySet.ReleaseQuery(queries[500]);
    UNIT_TEST_CHECK(platform->destroyQueryPoolCalls == 1);

    querySet.ReleaseAll();
    UNIT_TEST_CHECK(platform->destroyQueryPoolCalls == 2);
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, query_set_destroys_pools_when_scope_ends)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    const auto platform = std::make_shared<QueryFailureWorkerPlatform>(VK_SUCCESS, context.device, context.physicalDevice);
    {
        QuerySet querySet;
        UNIT_TEST_CHECK(querySet.Initialize(platform, context.device, 4));
        UNIT_TEST_CHECK(platform->destroyQueryPoolCalls == 0);
    }
    UNIT_TEST_CHECK(platform->destroyQueryPoolCalls == 1);

    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, query_set_reclaims_full_current_pool_across_rotations)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    auto     platform = std::make_shared<QueryFailureWorkerPlatform>(VK_SUCCESS, context.device, context.physicalDevice);
    QuerySet querySet;
    UNIT_TEST_CHECK(querySet.Initialize(platform, context.device, 4));

    constexpr uint32_t rotationCount = 32;
    for (uint32_t rotation = 0; rotation < rotationCount; ++rotation)
    {
        CompletionQuery queries[4];
        for (CompletionQuery& query : queries)
        {
            const std::optional<CompletionQuery> acquired = querySet.Acquire();
            UNIT_TEST_CHECK(acquired.has_value());
            query = *acquired;
        }
        for (const CompletionQuery& query : queries)
            querySet.ReleaseQuery(query);

        UNIT_TEST_CHECK(platform->createQueryPoolCalls == rotation + 1);
        UNIT_TEST_CHECK(platform->destroyQueryPoolCalls == rotation + 1);
    }

    querySet.ReleaseAll();
    UNIT_TEST_CHECK(platform->destroyQueryPoolCalls == rotationCount);
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, pool_share_device)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));
    {
        Worker worker;

        auto                firstSink  = std::make_shared<FakeMetadataSink>();
        auto                secondSink = std::make_shared<FakeMetadataSink>();
        const SessionHandle first      = worker.Create(MakeDescriptor(context, firstSink));
        const SessionHandle second     = worker.Create(MakeDescriptor(context, secondSink));

        UNIT_TEST_CHECK(first != InvalidSession);
        UNIT_TEST_CHECK(second != InvalidSession);
        UNIT_TEST_CHECK(first != second);

        worker.Close(first);
        worker.Close(second);
        UNIT_TEST_CHECK(worker.Wait(first, 5000));
        UNIT_TEST_CHECK(worker.Wait(second, 5000));
        UNIT_TEST_CHECK(firstSink->finalizeCount == 1);
        UNIT_TEST_CHECK(secondSink->finalizeCount == 1);

        worker.Destroy(first);
        worker.Destroy(second);
    }
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, pool_cleanup_last)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));
    {
        Worker worker;

        auto                firstSink = std::make_shared<FakeMetadataSink>();
        const SessionHandle first     = worker.Create(MakeDescriptor(context, firstSink));
        UNIT_TEST_CHECK(first != InvalidSession);

        worker.Close(first);
        UNIT_TEST_CHECK(worker.Wait(first, 5000));
        worker.Destroy(first);

        auto                secondSink = std::make_shared<FakeMetadataSink>();
        const SessionHandle second     = worker.Create(MakeDescriptor(context, secondSink));
        UNIT_TEST_CHECK(second != InvalidSession);

        worker.Close(second);
        UNIT_TEST_CHECK(worker.Wait(second, 5000));
        worker.Destroy(second);
        UNIT_TEST_CHECK(secondSink->finalizeCount == 1);
    }
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

// Worker runtime: record a real Vulkan image readback, submit it, consume GPU completion, and finalize the frame.
UNIT_TEST_CASE(capture, worker_image_consume)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    unit_test::vk::Image image{};
    std::string          error;
    UNIT_TEST_CHECK(
        unit_test::vk::CreateImage2D(context, 16, 16, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, image, &error));
    UNIT_TEST_CHECK(unit_test::vk::ClearImage(context, image, {0.25f, 0.0f, 0.0f, 1.0f}, &error));

    const auto          sink   = std::make_shared<FakeMetadataSink>();
    const auto          output = OutputDirectory("worker_image_consume");
    Worker              worker;
    const SessionHandle session = worker.Create(MakeDescriptor(context, sink));
    UNIT_TEST_CHECK(session != InvalidSession);

    uint32_t slotIndex = UINT32_MAX;
    UNIT_TEST_CHECK(worker.AcquireSlot(session, 16 * 16, slotIndex));

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, commandBuffer, &error));
    const FfxResource source    = unit_test::vk::MakeFfxImageResource(image, FFX_RESOURCE_USAGE_READ_ONLY, FFX_RESOURCE_STATE_GENERIC_UAV, "worker_image");
    auto              payload   = std::make_unique<BinaryPayload>("worker_image", output.File("image.bin").string());
    auto              frameMeta = std::make_unique<FrameMeta>();
    UNIT_TEST_CHECK(worker.Record(session, slotIndex, commandBuffer, source, std::move(payload)));
    UNIT_TEST_CHECK(worker.Submit(session, slotIndex, commandBuffer, &frameMeta));
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(context, commandBuffer, &error));

    worker.Close(session);
    UNIT_TEST_CHECK(worker.Wait(session, 5000));
    UNIT_TEST_CHECK(sink->completeCount == 1);
    UNIT_TEST_CHECK(sink->finalizeCount == 1);
    UNIT_TEST_CHECK(std::filesystem::is_regular_file(output.File("image.bin")));
    worker.Destroy(session);

    unit_test::vk::DestroyImage(context, image);
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

// A reserved completion query must remain pending until its command buffer has executed.
UNIT_TEST_CASE(capture, worker_does_not_consume_unsubmitted_completion_query)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    unit_test::vk::Buffer buffer{};
    std::string           error;
    UNIT_TEST_CHECK(unit_test::vk::CreateBuffer(context,
                                                64,
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                buffer,
                                                &error));

    auto            platform = std::make_shared<QueryFailureWorkerPlatform>(VK_SUCCESS, context.device, context.physicalDevice);
    Worker          worker(platform, 1);
    OutputDirectory output("unsubmitted_query_generation");
    auto            sink = std::make_shared<FakeMetadataSink>();

    const SessionHandle session = worker.Create(MakeDescriptor(context, sink));
    UNIT_TEST_CHECK(session != InvalidSession);

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType                   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size                    = 64;
    bufferCreateInfo.usage                   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const FfxResourceDescription description = ffxGetBufferResourceDescriptionVK(buffer.buffer, bufferCreateInfo, FFX_RESOURCE_USAGE_READ_ONLY);
    const FfxResource            source =
        ffxGetResourceVK(reinterpret_cast<void*>(buffer.buffer), description, "unsubmitted_query_generation", FFX_RESOURCE_STATE_COPY_SRC);

    uint32_t firstSlot = UINT32_MAX;
    UNIT_TEST_CHECK(worker.AcquireSlot(session, 64, firstSlot));
    UNIT_TEST_CHECK(firstSlot == 0);

    VkCommandBuffer firstCommandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, firstCommandBuffer, &error));
    auto firstPayload   = std::make_unique<BinaryPayload>("first", output.File("unsubmitted_query_generation_first.bin").string());
    auto firstFrameMeta = std::make_unique<FrameMeta>();
    UNIT_TEST_CHECK(worker.RecordBuffer(session, firstSlot, firstCommandBuffer, source, std::move(firstPayload)));
    UNIT_TEST_CHECK(worker.Submit(session, firstSlot, firstCommandBuffer, &firstFrameMeta));
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(context, firstCommandBuffer, &error));

    const auto firstDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (sink->completeCount.load(std::memory_order_acquire) != 1 && std::chrono::steady_clock::now() < firstDeadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const bool firstCompleted = sink->completeCount.load(std::memory_order_acquire) == 1;

    uint32_t secondSlot = UINT32_MAX;
    UNIT_TEST_CHECK(worker.AcquireSlot(session, 64, secondSlot));
    UNIT_TEST_CHECK(secondSlot == 1);

    VkCommandBuffer reusedCommandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, reusedCommandBuffer, &error));
    auto reusedPayload   = std::make_unique<BinaryPayload>("second", output.File("unsubmitted_query_generation_second.bin").string());
    auto reusedFrameMeta = std::make_unique<FrameMeta>();
    UNIT_TEST_CHECK(worker.RecordBuffer(session, secondSlot, reusedCommandBuffer, source, std::move(reusedPayload)));
    const uint32_t queryCallsBeforeSubmit = platform->queryCalls.load(std::memory_order_acquire);
    platform->blockQueryResults.store(true, std::memory_order_release);
    UNIT_TEST_CHECK(worker.Submit(session, secondSlot, reusedCommandBuffer, &reusedFrameMeta));

    const auto queryPolledDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (platform->queryCalls.load(std::memory_order_acquire) == queryCallsBeforeSubmit && std::chrono::steady_clock::now() < queryPolledDeadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const bool queryPolled               = platform->queryCalls.load(std::memory_order_acquire) > queryCallsBeforeSubmit;
    const bool consumedBeforeQueueSubmit = sink->completeCount.load(std::memory_order_acquire) != 1;

    // Submit before teardown even when the assertion below fails.
    platform->blockQueryResults.store(false, std::memory_order_release);
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(context, reusedCommandBuffer, &error));
    worker.Close(session);
    UNIT_TEST_CHECK(worker.Wait(session, 5000));
    worker.Destroy(session);

    unit_test::vk::DestroyBuffer(context, buffer);
    unit_test::vk::DestroyVulkanContext(context);

    UNIT_TEST_CHECK(firstCompleted);
    UNIT_TEST_CHECK(queryPolled);
    if (consumedBeforeQueueSubmit)
        UNIT_TEST_FAIL("worker consumed an unsubmitted query generation before its command buffer was submitted");
    return true;
}

UNIT_TEST_CASE(capture, worker_shutdown_detach_stops_unsubmitted_query_processing)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    unit_test::vk::Buffer buffer{};
    std::string           error;
    UNIT_TEST_CHECK(unit_test::vk::CreateBuffer(context,
                                                64,
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                buffer,
                                                &error));

    const auto pagePlatform = std::make_shared<TrackingVulkanStagingPagePlatform>();
    const auto platform     = std::make_shared<QueryFailureWorkerPlatform>(VK_SUCCESS, context.device, context.physicalDevice, pagePlatform);
    const auto sink         = std::make_shared<FakeMetadataSink>();
    const auto detached     = std::make_shared<std::atomic_bool>(false);
    {
        Worker            worker(platform, 1);
        SessionDescriptor descriptor = MakeDescriptor(context, sink);
        descriptor.shutdownDetached  = detached;
        const SessionHandle session  = worker.Create(descriptor);
        UNIT_TEST_CHECK(session != InvalidSession);

        uint32_t slotIndex = UINT32_MAX;
        UNIT_TEST_CHECK(worker.AcquireSlot(session, 64, slotIndex));

        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, commandBuffer, &error));
        VkBufferCreateInfo bufferCreateInfo{};
        bufferCreateInfo.sType                   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size                    = 64;
        bufferCreateInfo.usage                   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        const FfxResourceDescription description = ffxGetBufferResourceDescriptionVK(buffer.buffer, bufferCreateInfo, FFX_RESOURCE_USAGE_READ_ONLY);
        const FfxResource source    = ffxGetResourceVK(reinterpret_cast<void*>(buffer.buffer), description, "shutdown_detach", FFX_RESOURCE_STATE_COPY_SRC);
        auto              payload   = std::make_unique<BinaryPayload>("shutdown_detach", "shutdown_detach.bin");
        auto              frameMeta = std::make_unique<FrameMeta>();
        UNIT_TEST_CHECK(worker.RecordBuffer(session, slotIndex, commandBuffer, source, std::move(payload)));

        platform->blockInsideQuery.store(true, std::memory_order_release);
        UNIT_TEST_CHECK(worker.Submit(session, slotIndex, commandBuffer, &frameMeta));
        worker.Close(session);
        platform->WaitForQueryCall();

        detached->store(true, std::memory_order_release);
        UNIT_TEST_CHECK(!worker.Wait(session, 10));
        platform->AllowQueryReturn();
        UNIT_TEST_CHECK(worker.Wait(session, 1000));
        const uint32_t queryCallsAfterDetach = platform->queryCalls.load(std::memory_order_acquire);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        UNIT_TEST_CHECK(platform->queryCalls.load(std::memory_order_acquire) == queryCallsAfterDetach);
        UNIT_TEST_CHECK(sink->completeCount == 0);
        UNIT_TEST_CHECK(sink->finalizeCount == 0);
        UNIT_TEST_CHECK(platform->destroyQueryPoolCalls == 0);
        UNIT_TEST_CHECK(pagePlatform->destroyBufferCalls == 0);
        UNIT_TEST_CHECK(pagePlatform->freeMemoryCalls == 0);
    }

    UNIT_TEST_CHECK(pagePlatform->destroyBufferCalls == 0);
    UNIT_TEST_CHECK(pagePlatform->freeMemoryCalls == 0);
    unit_test::vk::DestroyBuffer(context, buffer);
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, worker_payload_write_failure_fails_session)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    unit_test::vk::Image image{};
    std::string          error;
    UNIT_TEST_CHECK(
        unit_test::vk::CreateImage2D(context, 16, 16, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, image, &error));
    UNIT_TEST_CHECK(unit_test::vk::ClearImage(context, image, {0.25f, 0.0f, 0.0f, 1.0f}, &error));

    const auto        sink = std::make_shared<FakeMetadataSink>();
    Worker            worker;
    std::atomic_bool  failureNotified{false};
    SessionDescriptor descriptor = MakeDescriptor(context, sink);
    descriptor.onFailure         = [&failureNotified]() { failureNotified.store(true, std::memory_order_release); };
    const SessionHandle session  = worker.Create(descriptor);
    UNIT_TEST_CHECK(session != InvalidSession);

    uint32_t slotIndex = UINT32_MAX;
    UNIT_TEST_CHECK(worker.AcquireSlot(session, 16 * 16, slotIndex));

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, commandBuffer, &error));
    const FfxResource source    = unit_test::vk::MakeFfxImageResource(image, FFX_RESOURCE_USAGE_READ_ONLY, FFX_RESOURCE_STATE_GENERIC_UAV, "worker_image");
    auto              payload   = std::make_unique<FailingPayload>();
    auto              frameMeta = std::make_unique<FrameMeta>();
    UNIT_TEST_CHECK(worker.Record(session, slotIndex, commandBuffer, source, std::move(payload)));
    UNIT_TEST_CHECK(worker.Submit(session, slotIndex, commandBuffer, &frameMeta));
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(context, commandBuffer, &error));

    worker.Close(session);
    UNIT_TEST_CHECK(worker.Wait(session, 5000));
    UNIT_TEST_CHECK(failureNotified.load(std::memory_order_acquire));
    UNIT_TEST_CHECK(sink->completeCount == 0);
    UNIT_TEST_CHECK(sink->finalizeCount == 1);
    worker.Destroy(session);

    unit_test::vk::DestroyImage(context, image);
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, worker_metadata_write_failure_fails_session)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    unit_test::vk::Image image{};
    std::string          error;
    UNIT_TEST_CHECK(
        unit_test::vk::CreateImage2D(context, 16, 16, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, image, &error));
    UNIT_TEST_CHECK(unit_test::vk::ClearImage(context, image, {0.25f, 0.0f, 0.0f, 1.0f}, &error));

    const auto        sink   = std::make_shared<FakeMetadataSink>(false);
    const auto        output = OutputDirectory("worker_metadata_write_failure");
    Worker            worker;
    std::atomic_bool  failureNotified{false};
    SessionDescriptor descriptor = MakeDescriptor(context, sink);
    descriptor.onFailure         = [&failureNotified]() { failureNotified.store(true, std::memory_order_release); };
    const SessionHandle session  = worker.Create(descriptor);
    UNIT_TEST_CHECK(session != InvalidSession);

    uint32_t slotIndex = UINT32_MAX;
    UNIT_TEST_CHECK(worker.AcquireSlot(session, 16 * 16, slotIndex));

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, commandBuffer, &error));
    const FfxResource source    = unit_test::vk::MakeFfxImageResource(image, FFX_RESOURCE_USAGE_READ_ONLY, FFX_RESOURCE_STATE_GENERIC_UAV, "worker_image");
    auto              payload   = std::make_unique<BinaryPayload>("worker_image", output.File("image.bin").string());
    auto              frameMeta = std::make_unique<FrameMeta>();
    UNIT_TEST_CHECK(worker.Record(session, slotIndex, commandBuffer, source, std::move(payload)));
    UNIT_TEST_CHECK(worker.Submit(session, slotIndex, commandBuffer, &frameMeta));
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(context, commandBuffer, &error));

    worker.Close(session);
    UNIT_TEST_CHECK(worker.Wait(session, 5000));
    UNIT_TEST_CHECK(failureNotified.load(std::memory_order_acquire));
    UNIT_TEST_CHECK(sink->completeCount == 1);
    UNIT_TEST_CHECK(sink->finalizeCount == 1);
    worker.Destroy(session);

    unit_test::vk::DestroyImage(context, image);
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, worker_metadata_finalize_failure_fails_session)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    const auto        sink = std::make_shared<FakeMetadataSink>(true, false);
    Worker            worker;
    std::atomic_bool  failureNotified{false};
    SessionDescriptor descriptor = MakeDescriptor(context, sink);
    descriptor.onFailure         = [&failureNotified]() { failureNotified.store(true, std::memory_order_release); };
    const SessionHandle session  = worker.Create(descriptor);
    UNIT_TEST_CHECK(session != InvalidSession);

    worker.Close(session);
    UNIT_TEST_CHECK(worker.Wait(session, 5000));
    UNIT_TEST_CHECK(failureNotified.load(std::memory_order_acquire));
    UNIT_TEST_CHECK(sink->completeCount == 0);
    UNIT_TEST_CHECK(sink->finalizeCount == 1);
    worker.Destroy(session);

    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

// Worker runtime: record a real Vulkan buffer readback through the same completion and drain path.
UNIT_TEST_CASE(capture, worker_buffer_consume)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    unit_test::vk::Buffer buffer{};
    std::string           error;
    UNIT_TEST_CHECK(unit_test::vk::CreateBuffer(context,
                                                64,
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                buffer,
                                                &error));
    std::array<uint8_t, 64> expectedBytes{};
    expectedBytes.fill(0x5a);
    void* mappedBuffer = nullptr;
    UNIT_TEST_CHECK(VulkanWrapper().vkMapMemory(context.device, buffer.memory, 0, expectedBytes.size(), 0, &mappedBuffer) == VK_SUCCESS);
    std::memcpy(mappedBuffer, expectedBytes.data(), expectedBytes.size());
    VulkanWrapper().vkUnmapMemory(context.device, buffer.memory);

    const auto          sink   = std::make_shared<FakeMetadataSink>();
    const auto          output = OutputDirectory("worker_buffer_consume");
    Worker              worker;
    const SessionHandle session = worker.Create(MakeDescriptor(context, sink));
    UNIT_TEST_CHECK(session != InvalidSession);

    uint32_t slotIndex = UINT32_MAX;
    UNIT_TEST_CHECK(worker.AcquireSlot(session, 64, slotIndex));

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, commandBuffer, &error));
    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType                   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size                    = 64;
    bufferCreateInfo.usage                   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const FfxResourceDescription description = ffxGetBufferResourceDescriptionVK(buffer.buffer, bufferCreateInfo, FFX_RESOURCE_USAGE_READ_ONLY);
    const FfxResource source    = ffxGetResourceVK(reinterpret_cast<void*>(buffer.buffer), description, "worker_buffer", FFX_RESOURCE_STATE_COMPUTE_READ);
    auto              payload   = std::make_unique<BinaryPayload>("worker_buffer", output.File("buffer.bin").string());
    auto              frameMeta = std::make_unique<FrameMeta>();
    UNIT_TEST_CHECK(worker.RecordBuffer(session, slotIndex, commandBuffer, source, std::move(payload)));
    VulkanWrapper().vkCmdFillBuffer(commandBuffer, buffer.buffer, 0, 64, 0u);
    UNIT_TEST_CHECK(worker.Submit(session, slotIndex, commandBuffer, &frameMeta));
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(context, commandBuffer, &error));

    worker.Close(session);
    UNIT_TEST_CHECK(worker.Wait(session, 5000));
    UNIT_TEST_CHECK(sink->completeCount == 1);
    UNIT_TEST_CHECK(sink->finalizeCount == 1);
    UNIT_TEST_CHECK(std::filesystem::is_regular_file(output.File("buffer.bin")));
    std::array<uint8_t, 64> capturedBytes{};
    {
        std::ifstream capturedFile(output.File("buffer.bin"), std::ios::binary);
        UNIT_TEST_CHECK(capturedFile.read(reinterpret_cast<char*>(capturedBytes.data()), capturedBytes.size()));
    }
    UNIT_TEST_CHECK(capturedBytes == expectedBytes);
    worker.Destroy(session);

    unit_test::vk::DestroyBuffer(context, buffer);
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, worker_preserves_dataset_indices_when_completion_is_out_of_order)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    unit_test::vk::Buffer buffer{};
    std::string           error;
    UNIT_TEST_CHECK(unit_test::vk::CreateBuffer(context,
                                                64,
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                buffer,
                                                &error));

    const auto          platform = std::make_shared<QueryFailureWorkerPlatform>(VK_SUCCESS, context.device, context.physicalDevice);
    const auto          sink     = std::make_shared<FakeMetadataSink>();
    Worker              worker(platform, 1);
    const SessionHandle session = worker.Create(MakeDescriptor(context, sink));
    UNIT_TEST_CHECK(session != InvalidSession);

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType                   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size                    = 64;
    bufferCreateInfo.usage                   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const FfxResourceDescription description = ffxGetBufferResourceDescriptionVK(buffer.buffer, bufferCreateInfo, FFX_RESOURCE_USAGE_READ_ONLY);
    const FfxResource            source = ffxGetResourceVK(reinterpret_cast<void*>(buffer.buffer), description, "out_of_order", FFX_RESOURCE_STATE_COPY_SRC);
    OutputDirectory              output("out_of_order_completion");

    uint32_t firstSlot = UINT32_MAX;
    UNIT_TEST_CHECK(worker.AcquireSlot(session, 64, firstSlot));
    VkCommandBuffer firstCommandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, firstCommandBuffer, &error));
    UNIT_TEST_CHECK(
        worker.RecordBuffer(session, firstSlot, firstCommandBuffer, source, std::make_unique<BinaryPayload>("first", output.File("first.bin").string())));
    auto firstFrameMeta          = std::make_unique<FrameMeta>();
    firstFrameMeta->datasetIndex = 0;

    uint32_t secondSlot = UINT32_MAX;
    UNIT_TEST_CHECK(worker.AcquireSlot(session, 64, secondSlot));
    VkCommandBuffer secondCommandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, secondCommandBuffer, &error));
    UNIT_TEST_CHECK(
        worker.RecordBuffer(session, secondSlot, secondCommandBuffer, source, std::make_unique<BinaryPayload>("second", output.File("second.bin").string())));
    auto secondFrameMeta          = std::make_unique<FrameMeta>();
    secondFrameMeta->datasetIndex = 1;

    UNIT_TEST_CHECK(platform->queryPools.size() == 2);
    platform->queryCalls.store(1, std::memory_order_release);
    platform->firstQueryPool.store(platform->queryPools[0], std::memory_order_release);
    platform->blockFirstQueryPool.store(true, std::memory_order_release);
    platform->blockInsideQuery.store(true, std::memory_order_release);

    UNIT_TEST_CHECK(worker.Submit(session, firstSlot, firstCommandBuffer, &firstFrameMeta));
    platform->WaitForQueryCall();
    UNIT_TEST_CHECK(worker.Submit(session, secondSlot, secondCommandBuffer, &secondFrameMeta));
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(context, firstCommandBuffer, &error));
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(context, secondCommandBuffer, &error));
    platform->AllowQueryReturn();

    const auto secondCompletedDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (sink->completeCount.load(std::memory_order_acquire) == 0 && std::chrono::steady_clock::now() < secondCompletedDeadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    UNIT_TEST_CHECK(sink->completeCount == 1);
    UNIT_TEST_CHECK(sink->completedDatasetIndices[0] == 1);

    platform->blockFirstQueryPool.store(false, std::memory_order_release);
    worker.Close(session);
    UNIT_TEST_CHECK(worker.Wait(session, 5000));
    UNIT_TEST_CHECK(sink->completedDatasetIndices.size() == 2);
    UNIT_TEST_CHECK(sink->completedDatasetIndices[1] == 0);
    worker.Destroy(session);

    unit_test::vk::DestroyBuffer(context, buffer);
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

// Worker runtime: abort an acquired slot before any GPU readback is recorded and drain it safely.
UNIT_TEST_CASE(capture, worker_abort_empty)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    const auto          sink = std::make_shared<FakeMetadataSink>();
    Worker              worker;
    const SessionHandle session = worker.Create(MakeDescriptor(context, sink));
    UNIT_TEST_CHECK(session != InvalidSession);

    uint32_t slotIndex = UINT32_MAX;
    UNIT_TEST_CHECK(worker.AcquireSlot(session, 64, slotIndex));
    UNIT_TEST_CHECK(worker.Abort(session, slotIndex, VK_NULL_HANDLE));

    worker.Close(session);
    UNIT_TEST_CHECK(worker.Wait(session, 5000));
    UNIT_TEST_CHECK(sink->completeCount == 0);
    UNIT_TEST_CHECK(sink->finalizeCount == 1);
    worker.Destroy(session);

    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

// A cancelled copy keeps its staging block occupied until the reserved completion query completes.
UNIT_TEST_CASE(capture, worker_abort_transaction_keeps_recorded_staging_until_completion)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    unit_test::vk::Buffer buffer{};
    std::string           error;
    UNIT_TEST_CHECK(unit_test::vk::CreateBuffer(context,
                                                64,
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                buffer,
                                                &error));

    const auto          sink   = std::make_shared<FakeMetadataSink>();
    const auto          output = OutputDirectory("worker_abort_unsubmitted");
    Worker              worker;
    const SessionHandle session = worker.Create(MakeDescriptor(context, sink));
    UNIT_TEST_CHECK(session != InvalidSession);

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType                   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size                    = 64;
    bufferCreateInfo.usage                   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const FfxResourceDescription description = ffxGetBufferResourceDescriptionVK(buffer.buffer, bufferCreateInfo, FFX_RESOURCE_USAGE_READ_ONLY);
    const FfxResource source = ffxGetResourceVK(reinterpret_cast<void*>(buffer.buffer), description, "worker_abort_unsubmitted", FFX_RESOURCE_STATE_COPY_SRC);

    uint32_t slotIndex = UINT32_MAX;
    UNIT_TEST_CHECK(worker.AcquireSlot(session, 64, slotIndex));
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, commandBuffer, &error));
    UNIT_TEST_CHECK(
        worker.RecordBuffer(session, slotIndex, commandBuffer, source, std::make_unique<BinaryPayload>("payload", output.File("payload.bin").string())));

    UNIT_TEST_CHECK(worker.Abort(session, slotIndex, commandBuffer));
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(context, commandBuffer, &error));
    worker.Close(session);
    UNIT_TEST_CHECK(worker.Wait(session, 5000));
    UNIT_TEST_CHECK(sink->completeCount == 0);
    UNIT_TEST_CHECK(sink->finalizeCount == 1);
    worker.Destroy(session);

    unit_test::vk::DestroyBuffer(context, buffer);
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, worker_rejects_unsupported_capture_capability)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    const auto platform = std::make_shared<QueryFailureWorkerPlatform>(VK_SUCCESS, context.device, context.physicalDevice);
    platform->captureSupported.store(false, std::memory_order_release);
    Worker     worker(platform);
    const auto sink = std::make_shared<FakeMetadataSink>();

    UNIT_TEST_CHECK(worker.Create(MakeDescriptor(context, sink)) == InvalidSession);
    UNIT_TEST_CHECK(platform->createQueryPoolCalls == 0);
    UNIT_TEST_CHECK(platform->createPoolCalls == 0);

    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

// Query reservation must fail before a transaction can record any capture copy.
UNIT_TEST_CASE(capture, worker_query_reservation_failure_does_not_acquire_staging)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    unit_test::vk::Buffer buffer{};
    std::string           error;
    UNIT_TEST_CHECK(unit_test::vk::CreateBuffer(context,
                                                64,
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                buffer,
                                                &error));

    const auto          pagePlatform = std::make_shared<TrackingVulkanStagingPagePlatform>();
    const auto          platform     = std::make_shared<QueryFailureWorkerPlatform>(VK_SUCCESS, context.device, context.physicalDevice, pagePlatform);
    Worker              worker(platform, 1);
    auto                sink = std::make_shared<FakeMetadataSink>();
    OutputDirectory     output("worker_query_reservation_failure");
    const SessionHandle session = worker.Create(MakeDescriptor(context, sink));
    UNIT_TEST_CHECK(session != InvalidSession);

    // Keep the first slot's completion query pending while exercising the
    // query-pool allocation failure. Without this synchronization, the worker
    // can recycle the query before the second AcquireSlot call on faster
    // platforms, making the test timing-dependent.
    platform->blockInsideQuery.store(true, std::memory_order_release);
    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType                   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size                    = 64;
    bufferCreateInfo.usage                   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const FfxResourceDescription description = ffxGetBufferResourceDescriptionVK(buffer.buffer, bufferCreateInfo, FFX_RESOURCE_USAGE_READ_ONLY);
    const FfxResource            source =
        ffxGetResourceVK(reinterpret_cast<void*>(buffer.buffer), description, "worker_query_reservation_failure", FFX_RESOURCE_STATE_COPY_SRC);

    uint32_t slotIndex = UINT32_MAX;
    UNIT_TEST_CHECK(worker.AcquireSlot(session, 64, slotIndex));
    VkCommandBuffer firstCommandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, firstCommandBuffer, &error));
    UNIT_TEST_CHECK(
        worker.RecordBuffer(session, slotIndex, firstCommandBuffer, source, std::make_unique<BinaryPayload>("first", output.File("first.bin").string())));
    UNIT_TEST_CHECK(worker.Abort(session, slotIndex, firstCommandBuffer));
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(context, firstCommandBuffer, &error));
    platform->WaitForQueryCall();
    const uint32_t createBufferCallsAfterFirstSlot = pagePlatform->createBufferCalls;

    platform->failNextQueryPoolCreation.store(true, std::memory_order_release);
    slotIndex = UINT32_MAX;
    UNIT_TEST_CHECK(!worker.AcquireSlot(session, 64, slotIndex));
    UNIT_TEST_CHECK(slotIndex == UINT32_MAX);
    UNIT_TEST_CHECK(pagePlatform->createBufferCalls == createBufferCallsAfterFirstSlot);

    UNIT_TEST_CHECK(worker.AcquireSlot(session, 64, slotIndex));
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, commandBuffer, &error));

    UNIT_TEST_CHECK(
        worker.RecordBuffer(session, slotIndex, commandBuffer, source, std::make_unique<BinaryPayload>("payload", output.File("payload.bin").string())));
    platform->failNextQueryPoolCreation.store(true, std::memory_order_release);
    UNIT_TEST_CHECK(worker.Abort(session, slotIndex, commandBuffer));
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(context, commandBuffer, &error));

    platform->AllowQueryReturn();
    platform->blockInsideQuery.store(false, std::memory_order_release);
    worker.Close(session);
    UNIT_TEST_CHECK(worker.Wait(session, 5000));
    worker.Destroy(session);

    unit_test::vk::DestroyBuffer(context, buffer);
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

// A cancelled copy keeps its staging block occupied until the reserved completion query completes.
UNIT_TEST_CASE(capture, worker_abort_does_not_recycle_recorded_staging)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    unit_test::vk::Buffer buffer{};
    std::string           error;
    UNIT_TEST_CHECK(unit_test::vk::CreateBuffer(context,
                                                64,
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                buffer,
                                                &error));

    const auto pagePlatform = std::make_shared<TrackingVulkanStagingPagePlatform>();
    const auto platform     = std::make_shared<QueryFailureWorkerPlatform>(VK_SUCCESS, context.device, context.physicalDevice, pagePlatform);
    platform->blockFirstQueryPool.store(true, std::memory_order_release);
    Worker              worker(platform, 1);
    auto                sink = std::make_shared<FakeMetadataSink>();
    OutputDirectory     output("worker_abort_staging_lifetime");
    const SessionHandle session = worker.Create(MakeDescriptor(context, sink));
    UNIT_TEST_CHECK(session != InvalidSession);
    UNIT_TEST_CHECK(platform->lastStagingPool != nullptr);

    const VkDeviceSize alignment  = platform->lastStagingPool->OffsetAlignment();
    const VkDeviceSize unitSize   = arm::capture::detail::StagingAllocationUnitSize(alignment);
    const VkDeviceSize blockBytes = arm::capture::detail::SlotBytesForAttachments({64}, {4}, alignment, unitSize);
    UNIT_TEST_CHECK(unitSize != 0);
    UNIT_TEST_CHECK(blockBytes != 0);

    std::vector<StagingBlock> reservedBlocks;
    for (;;)
    {
        StagingBlock block = platform->lastStagingPool->Acquire(blockBytes);
        if (!block)
            break;
        reservedBlocks.push_back(block);
    }
    UNIT_TEST_CHECK(reservedBlocks.size() >= 2);
    platform->lastStagingPool->Return(reservedBlocks.back());
    reservedBlocks.pop_back();

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType                   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size                    = 64;
    bufferCreateInfo.usage                   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const FfxResourceDescription description = ffxGetBufferResourceDescriptionVK(buffer.buffer, bufferCreateInfo, FFX_RESOURCE_USAGE_READ_ONLY);
    const FfxResource            source =
        ffxGetResourceVK(reinterpret_cast<void*>(buffer.buffer), description, "worker_abort_staging_lifetime", FFX_RESOURCE_STATE_COPY_SRC);

    uint32_t slotIndex = UINT32_MAX;
    UNIT_TEST_CHECK(worker.AcquireSlot(session, 64, slotIndex));
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, commandBuffer, &error));
    UNIT_TEST_CHECK(
        worker.RecordBuffer(session, slotIndex, commandBuffer, source, std::make_unique<BinaryPayload>("payload", output.File("payload.bin").string())));
    platform->failNextQueryPoolCreation.store(true, std::memory_order_release);
    UNIT_TEST_CHECK(worker.Abort(session, slotIndex, commandBuffer));
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(context, commandBuffer, &error));

    StagingBlock recycled = platform->lastStagingPool->Acquire(blockBytes);
    UNIT_TEST_CHECK(!recycled);

    platform->blockFirstQueryPool.store(false, std::memory_order_release);
    worker.Close(session);
    UNIT_TEST_CHECK(worker.Wait(session, 5000));
    worker.Destroy(session);

    for (StagingBlock& block : reservedBlocks)
        platform->lastStagingPool->Return(block);

    unit_test::vk::DestroyBuffer(context, buffer);
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, worker_allocation_failure_returns_false)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    auto                         pagePlatform = std::make_shared<TrackingStagingPagePlatform>(TrackingStagingPagePlatform::Failure::Allocate);
    auto                         platform = std::make_shared<QueryFailureWorkerPlatform>(VK_NOT_READY, context.device, context.physicalDevice, pagePlatform);
    arm::capture::detail::Worker worker(platform, 1);
    auto                         sink = std::make_shared<FakeMetadataSink>();

    arm::capture::SessionDescriptor descriptor(nullptr, 1024, sink);
    descriptor.device                         = context.device;
    descriptor.physicalDevice                 = context.physicalDevice;
    const arm::capture::SessionHandle session = worker.Create(descriptor);
    UNIT_TEST_CHECK(session != arm::capture::InvalidSession);
    uint32_t slotIndex = UINT32_MAX;
    UNIT_TEST_CHECK(!worker.AcquireSlot(session, 1024, slotIndex));
    UNIT_TEST_CHECK(slotIndex == UINT32_MAX);
    UNIT_TEST_CHECK(pagePlatform->createBufferCalls == 1);
    UNIT_TEST_CHECK(pagePlatform->allocateMemoryCalls == 1);
    UNIT_TEST_CHECK(pagePlatform->destroyBufferCalls == 1);
    UNIT_TEST_CHECK(pagePlatform->freeMemoryCalls == 0);
    UNIT_TEST_CHECK(platform->createQueryPoolCalls == 1);
    UNIT_TEST_CHECK(platform->destroyQueryPoolCalls == 1);
    UNIT_TEST_CHECK(sink->completeCount == 0);
    UNIT_TEST_CHECK(sink->finalizeCount == 0);

    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

// Worker runtime: preserve VK_NOT_READY, then discard all pending work after a terminal query failure.
UNIT_TEST_CASE(capture, worker_query_device_lost_discards_pending)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    unit_test::vk::Buffer buffer{};
    std::string           error;
    UNIT_TEST_CHECK(unit_test::vk::CreateBuffer(context,
                                                64,
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                buffer,
                                                &error));

    const auto sink         = std::make_shared<FakeMetadataSink>();
    const auto pagePlatform = std::make_shared<TrackingVulkanStagingPagePlatform>();
    const auto platform     = std::make_shared<QueryFailureWorkerPlatform>(VK_ERROR_DEVICE_LOST, context.device, context.physicalDevice, pagePlatform);
    {
        Worker            worker(platform);
        std::atomic_bool  failureNotified{false};
        SessionDescriptor descriptor = MakeDescriptor(context, sink);
        descriptor.onFailure         = [&failureNotified]() { failureNotified.store(true, std::memory_order_release); };
        const SessionHandle session  = worker.Create(descriptor);
        UNIT_TEST_CHECK(session != InvalidSession);

        uint32_t slotIndex = UINT32_MAX;
        UNIT_TEST_CHECK(worker.AcquireSlot(session, 64, slotIndex));

        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, commandBuffer, &error));
        VkBufferCreateInfo bufferCreateInfo{};
        bufferCreateInfo.sType                   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size                    = 64;
        bufferCreateInfo.usage                   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        const FfxResourceDescription description = ffxGetBufferResourceDescriptionVK(buffer.buffer, bufferCreateInfo, FFX_RESOURCE_USAGE_READ_ONLY);
        const FfxResource source  = ffxGetResourceVK(reinterpret_cast<void*>(buffer.buffer), description, "worker_query_failure", FFX_RESOURCE_STATE_COPY_SRC);
        auto              payload = std::make_unique<BinaryPayload>("worker_query_failure", "worker_query_failure.bin");
        auto              frameMeta = std::make_unique<FrameMeta>();
        UNIT_TEST_CHECK(worker.RecordBuffer(session, slotIndex, commandBuffer, source, std::move(payload)));
        UNIT_TEST_CHECK(worker.Submit(session, slotIndex, commandBuffer, &frameMeta));
        UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(context, commandBuffer, &error));

        worker.Close(session);
        UNIT_TEST_CHECK(worker.Wait(session, 5000));
        UNIT_TEST_CHECK(platform->queryCalls >= 2);
        UNIT_TEST_CHECK(sink->completeCount == 0);
        UNIT_TEST_CHECK(sink->finalizeCount == 1);
        UNIT_TEST_CHECK(failureNotified.load(std::memory_order_acquire));
        worker.Destroy(session);
        UNIT_TEST_CHECK(platform->destroyQueryPoolCalls == 0);
        UNIT_TEST_CHECK(pagePlatform->destroyBufferCalls == 0);
        UNIT_TEST_CHECK(pagePlatform->freeMemoryCalls == 0);

        const SessionHandle replacement = worker.Create(MakeDescriptor(context, sink));
        UNIT_TEST_CHECK(replacement != InvalidSession);
        uint32_t replacementSlot = UINT32_MAX;
        UNIT_TEST_CHECK(worker.AcquireSlot(replacement, 64, replacementSlot));
        worker.Close(replacement);
        UNIT_TEST_CHECK(worker.Wait(replacement, 5000));
        worker.Destroy(replacement);
        UNIT_TEST_CHECK(platform->createPoolCalls == 2);
        UNIT_TEST_CHECK(pagePlatform->destroyBufferCalls == 1);
        UNIT_TEST_CHECK(pagePlatform->freeMemoryCalls == 1);
    }

    unit_test::vk::DestroyBuffer(context, buffer);
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, worker_query_error_abandons_pages)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    unit_test::vk::Buffer buffer{};
    std::string           error;
    UNIT_TEST_CHECK(unit_test::vk::CreateBuffer(context,
                                                64,
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                buffer,
                                                &error));

    const auto        sink         = std::make_shared<FakeMetadataSink>();
    const auto        pagePlatform = std::make_shared<TrackingVulkanStagingPagePlatform>();
    const auto        platform     = std::make_shared<QueryFailureWorkerPlatform>(VK_ERROR_UNKNOWN, context.device, context.physicalDevice, pagePlatform);
    Worker            worker(platform, 1);
    SessionDescriptor descriptor = MakeDescriptor(context, sink);
    std::atomic_bool  failureNotified{false};
    descriptor.onFailure        = [&failureNotified]() { failureNotified.store(true, std::memory_order_release); };
    const SessionHandle session = worker.Create(descriptor);
    UNIT_TEST_CHECK(session != InvalidSession);
    UNIT_TEST_CHECK(platform->lastStagingPool != nullptr);

    const VkDeviceSize alignment  = platform->lastStagingPool->OffsetAlignment();
    const VkDeviceSize unitSize   = arm::capture::detail::StagingAllocationUnitSize(alignment);
    const VkDeviceSize blockBytes = arm::capture::detail::SlotBytesForAttachments({64}, {4}, alignment, unitSize);
    UNIT_TEST_CHECK(unitSize != 0);
    UNIT_TEST_CHECK(blockBytes != 0);

    std::vector<StagingBlock> reservedBlocks;
    for (;;)
    {
        StagingBlock block = platform->lastStagingPool->Acquire(blockBytes);
        if (!block)
            break;
        reservedBlocks.push_back(block);
    }
    UNIT_TEST_CHECK(reservedBlocks.size() >= 2);
    platform->lastStagingPool->Return(reservedBlocks.back());
    reservedBlocks.pop_back();

    const uint32_t destroyBufferCallsBeforeQueryError = pagePlatform->destroyBufferCalls;
    const uint32_t freeMemoryCallsBeforeQueryError    = pagePlatform->freeMemoryCalls;

    uint32_t slotIndex = UINT32_MAX;
    UNIT_TEST_CHECK(worker.AcquireSlot(session, 64, slotIndex));
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, commandBuffer, &error));
    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType                   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size                    = 64;
    bufferCreateInfo.usage                   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const FfxResourceDescription description = ffxGetBufferResourceDescriptionVK(buffer.buffer, bufferCreateInfo, FFX_RESOURCE_USAGE_READ_ONLY);
    const FfxResource source    = ffxGetResourceVK(reinterpret_cast<void*>(buffer.buffer), description, "worker_query_error", FFX_RESOURCE_STATE_COPY_SRC);
    auto              payload   = std::make_unique<BinaryPayload>("worker_query_error", "worker_query_error.bin");
    auto              frameMeta = std::make_unique<FrameMeta>();
    UNIT_TEST_CHECK(worker.RecordBuffer(session, slotIndex, commandBuffer, source, std::move(payload)));
    UNIT_TEST_CHECK(worker.Submit(session, slotIndex, commandBuffer, &frameMeta));
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(context, commandBuffer, &error));
    worker.Close(session);
    UNIT_TEST_CHECK(worker.Wait(session, 5000));

    StagingBlock recycled = platform->lastStagingPool->Acquire(blockBytes);
    UNIT_TEST_CHECK(!recycled);
    UNIT_TEST_CHECK(platform->createQueryPoolCalls == 1);
    UNIT_TEST_CHECK(platform->destroyQueryPoolCalls == 0);
    UNIT_TEST_CHECK(failureNotified.load(std::memory_order_acquire));
    worker.Destroy(session);
    UNIT_TEST_CHECK(pagePlatform->destroyBufferCalls == destroyBufferCallsBeforeQueryError);
    UNIT_TEST_CHECK(pagePlatform->freeMemoryCalls == freeMemoryCallsBeforeQueryError);

    for (StagingBlock& block : reservedBlocks)
        platform->lastStagingPool->Return(block);

    unit_test::vk::DestroyBuffer(context, buffer);
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, worker_map_device_lost_abandons_pages)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    unit_test::vk::Buffer buffer{};
    std::string           error;
    UNIT_TEST_CHECK(unit_test::vk::CreateBuffer(context,
                                                64,
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                buffer,
                                                &error));

    const auto sink            = std::make_shared<FakeMetadataSink>();
    const auto pagePlatform    = std::make_shared<TrackingVulkanStagingPagePlatform>();
    pagePlatform->mapResult    = VK_ERROR_DEVICE_LOST;
    const auto        platform = std::make_shared<QueryFailureWorkerPlatform>(VK_SUCCESS, context.device, context.physicalDevice, pagePlatform);
    Worker            worker(platform);
    std::atomic_bool  failureNotified{false};
    SessionDescriptor descriptor = MakeDescriptor(context, sink);
    descriptor.onFailure         = [&failureNotified]() { failureNotified.store(true, std::memory_order_release); };
    const SessionHandle session  = worker.Create(descriptor);
    UNIT_TEST_CHECK(session != InvalidSession);

    uint32_t slotIndex = UINT32_MAX;
    UNIT_TEST_CHECK(worker.AcquireSlot(session, 64, slotIndex));

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, commandBuffer, &error));
    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType                   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size                    = 64;
    bufferCreateInfo.usage                   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const FfxResourceDescription description = ffxGetBufferResourceDescriptionVK(buffer.buffer, bufferCreateInfo, FFX_RESOURCE_USAGE_READ_ONLY);
    const FfxResource source    = ffxGetResourceVK(reinterpret_cast<void*>(buffer.buffer), description, "worker_map_failure", FFX_RESOURCE_STATE_COPY_SRC);
    auto              payload   = std::make_unique<BinaryPayload>("worker_map_failure", "worker_map_failure.bin");
    auto              frameMeta = std::make_unique<FrameMeta>();
    UNIT_TEST_CHECK(worker.RecordBuffer(session, slotIndex, commandBuffer, source, std::move(payload)));
    UNIT_TEST_CHECK(worker.Submit(session, slotIndex, commandBuffer, &frameMeta));
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(context, commandBuffer, &error));

    worker.Close(session);
    UNIT_TEST_CHECK(worker.Wait(session, 5000));
    UNIT_TEST_CHECK(failureNotified.load(std::memory_order_acquire));
    UNIT_TEST_CHECK(sink->completeCount == 0);
    worker.Destroy(session);
    UNIT_TEST_CHECK(pagePlatform->destroyBufferCalls == 0);
    UNIT_TEST_CHECK(pagePlatform->freeMemoryCalls == 0);

    unit_test::vk::DestroyBuffer(context, buffer);
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

namespace
{
    inline constexpr const char* kAndroidTestPropertyPrefix    = "debug.ngsdk.test.";
    inline constexpr const char* kDesktopTestEnvironmentPrefix = "TEST_CAPTURE_";

    class TestCaptureController final : public arm::capture::ControllerBase
    {
    public:
        explicit TestCaptureController(const bool enableEnvironment = false)
            : ControllerBase(enableEnvironment ? ARM_CAPTURE_PLATFORM_PREFIX(kAndroidTestPropertyPrefix, kDesktopTestEnvironmentPrefix) : nullptr,
                             "TEST-CAPTURE")
        {
        }

        using ControllerBase::Configure;

        bool AdvanceToConfig(uint32_t frameIndex)
        {
            return configInput_.AdvanceConfig(frameIndex);
        }

        bool AdvanceFrameForTest(uint32_t frameIndex)
        {
            bool result = false;
            while (dispatchFrameIndex_ <= frameIndex)
                result = AdvanceFrameThenCheck();
            return result;
        }

        bool AdvanceDispatchWhileSessionPreparationIsPending(uint32_t frameIndex)
        {
            if (!AdvanceFrameForTest(frameIndex) || sessionPreparationPending_)
                return false;
            ++capturedDispatches_;
            return true;
        }

        void CompleteSessionPreparationForTest()
        {
            sessionPreparationPending_ = false;
        }

        uint32_t CapturedDispatchesForTest() const
        {
            return capturedDispatches_;
        }

        bool HasPendingConfig(uint32_t frameIndex = 0)
        {
            return configInput_.HasPendingConfig(frameIndex);
        }

        void EnableEnvironmentPollingForTest()
        {
            configInput_.EnableEnvironmentPolling(0);
        }

        bool AdvanceConfig(uint32_t frameIndex)
        {
            return configInput_.AdvanceConfig(frameIndex);
        }

        const arm::capture::Config& ActiveConfig() const
        {
            return configInput_.GetActiveConfig();
        }

        bool StartForTest(const char* component)
        {
            return execution_.StartCapture(component);
        }

        bool StartForTest(const arm::capture::Config& config)
        {
            return StartCapture(config);
        }

        bool CompleteForTest()
        {
            return RequestCaptureDrain(arm::capture::CaptureResult::Success);
        }

        bool CompleteForTest(arm::capture::CaptureResult result)
        {
            return execution_.RequestDrain(result) && execution_.TryFinalizeDrain();
        }

    protected:
        bool OnCaptureBegin(const arm::capture::Config&) override
        {
            sessionPreparationPending_ = true;
            capturedDispatches_        = 0;
            return true;
        }

        void OnCaptureStop() override
        {
            sessionPreparationPending_ = false;
        }

    private:
        bool     sessionPreparationPending_ = false;
        uint32_t capturedDispatches_        = 0;
    };

    struct TestEnvScope : unit_test::ScopedCaptureEnvironment
    {
        TestEnvScope(
            const char* dir, const char* frames, const char* revision, const char* start = nullptr, const char* scale = nullptr, const char* bias = nullptr)
        {
            const auto setVariable = [](const char* name, const char* value) {
                if (!unit_test::ScopedCaptureEnvironment::SetVariable(name, value))
                    unit_test::MarkTestFailed("failed to set capture test variable: " + std::string(name));
            };
#if defined(__ANDROID__)
            if (dir)
                setVariable("debug.ngsdk.test.dir", dir);
            if (frames)
                setVariable("debug.ngsdk.test.frames", frames);
            if (revision)
                setVariable("debug.ngsdk.test.revision", revision);
            if (start)
                setVariable("debug.ngsdk.test.start", start);
            if (scale)
                setVariable("debug.ngsdk.test.mv_unorm_scale", scale);
            if (bias)
                setVariable("debug.ngsdk.test.mv_unorm_bias", bias);
#else
            if (dir)
                setVariable("TEST_CAPTURE_DIR", dir);
            if (frames)
                setVariable("TEST_CAPTURE_FRAME_COUNT", frames);
            if (revision)
                setVariable("TEST_CAPTURE_CONFIG_REVISION", revision);
            if (start)
                setVariable("TEST_CAPTURE_START_FRAME", start);
            if (scale)
                setVariable("TEST_CAPTURE_MV_UNORM_SCALE", scale);
            if (bias)
                setVariable("TEST_CAPTURE_MV_UNORM_BIAS", bias);
#endif
        }
    };

    ffxApiConfigureDescCapture MakeBeginCapture(const char* outputDirectory, uint32_t startDispatch, uint32_t frameCount)
    {
        ffxApiConfigureDescCapture desc{};
        desc.header.type            = FFX_API_CONFIGURE_DESC_TYPE_CAPTURE;
        desc.operation              = FFX_API_CAPTURE_BEGIN;
        desc.outputDirectory        = outputDirectory;
        desc.startDispatch          = startDispatch;
        desc.frameCount             = frameCount;
        desc.motionVectorUnormBias  = 0.25f;
        desc.motionVectorUnormScale = 4.0f;
        return desc;
    }

}  // namespace

// Runtime capture control: configuration validation and execution state transitions.
UNIT_TEST_CASE(capture, config_valid_begin)
{
    arm::capture::Execution             execution;
    ffxApiConfigureDescCapture          configure = MakeBeginCapture("nss_capture_test", 7, 3);
    std::optional<arm::capture::Config> applied;
    bool                                cancel = true;

    UNIT_TEST_CHECK(arm::capture::ConfigureCapture(execution, configure, [&applied, &cancel](const std::optional<arm::capture::Config>& config, bool isCancel) {
        applied = config;
        cancel  = isCancel;
        return true;
    }));
    UNIT_TEST_CHECK(applied.has_value());
    UNIT_TEST_CHECK(applied->outputDirectory == "nss_capture_test");
    UNIT_TEST_CHECK(applied->startFrame == 7);
    UNIT_TEST_CHECK(applied->maxFrameCount == 3);
    UNIT_TEST_CHECK(applied->motionVectorUnormBias == 0.25f);
    UNIT_TEST_CHECK(applied->motionVectorUnormScale == 4.0f);
    UNIT_TEST_CHECK(!cancel);
    return true;
}

UNIT_TEST_CASE(capture, config_copies_output_directory)
{
    TestCaptureController      controller;
    char                       outputDirectory[] = "capture-copied";
    ffxApiConfigureDescCapture configure         = MakeBeginCapture(outputDirectory, 0, 1);

    UNIT_TEST_CHECK(controller.Configure(configure) == FFX_API_RETURN_OK);
    std::strcpy(outputDirectory, "caller-mutated");
    UNIT_TEST_CHECK(controller.AdvanceToConfig(0));
    UNIT_TEST_CHECK(controller.ActiveConfig().outputDirectory == "capture-copied");
    return true;
}

UNIT_TEST_CASE(capture, config_preserves_final_output_directory)
{
    OutputDirectory            output("config_preserves_final_output_directory");
    const std::string          outputPath = output.File("final-dataset-root").string();
    TestCaptureController      controller;
    ffxApiConfigureDescCapture configure = MakeBeginCapture(outputPath.c_str(), 0, 1);

    UNIT_TEST_CHECK(controller.Configure(configure) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(controller.AdvanceToConfig(0));
    UNIT_TEST_CHECK(controller.StartForTest(controller.ActiveConfig()));
    UNIT_TEST_CHECK(controller.ActiveConfig().outputDirectory == outputPath);
    return true;
}

UNIT_TEST_CASE(capture, invalid_output_directory_fails_capture_start)
{
    OutputDirectory             output("invalid_output_directory");
    const std::filesystem::path parent = output.File("parent");
    std::filesystem::create_directories(parent);
    const std::filesystem::path blocker = parent / "blocker";
    {
        std::ofstream file(blocker);
        UNIT_TEST_CHECK(file.good());
    }

    TestCaptureController      controller;
    FfxApiCaptureStatus        status{};
    const std::string          invalidOutput = (blocker / "child").string();
    ffxApiConfigureDescCapture configure     = MakeBeginCapture(invalidOutput.c_str(), 0, 1);
    UNIT_TEST_CHECK(controller.Configure(configure) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(!controller.AdvanceFrameForTest(0));
    controller.GetStatus(status, 0);
    UNIT_TEST_CHECK(status.phase == FFX_API_CAPTURE_PHASE_IDLE);
    UNIT_TEST_CHECK(status.result == FFX_API_CAPTURE_RESULT_FAILED);
    return true;
}

UNIT_TEST_CASE(capture, failed_execution_start_reports_failure)
{
    OutputDirectory       output("failed_execution_start_reports_failure");
    const std::string     outputPath = output.File("dataset").string();
    TestCaptureController controller(true);
    arm::capture::Config  config{};
    FfxApiCaptureStatus   status{};

    config.component       = "TEST-CAPTURE";
    config.outputDirectory = outputPath;
    config.maxFrameCount   = 1;

    UNIT_TEST_CHECK(controller.StartForTest("already-capturing"));
    UNIT_TEST_CHECK(!controller.StartForTest(config));
    controller.GetStatus(status, 0);
    UNIT_TEST_CHECK(status.result == FFX_API_CAPTURE_RESULT_FAILED);

    return true;
}

UNIT_TEST_CASE(capture, config_accepted_future_begin_resets_terminal_result)
{
    OutputDirectory            output("config_accepted_future_begin_resets_terminal_result");
    const std::string          firstOutputPath  = output.File("first-dataset").string();
    const std::string          secondOutputPath = output.File("second-dataset").string();
    TestCaptureController      controller;
    ffxApiConfigureDescCapture firstBegin  = MakeBeginCapture(firstOutputPath.c_str(), 0, 1);
    ffxApiConfigureDescCapture secondBegin = MakeBeginCapture(secondOutputPath.c_str(), 100, 1);
    FfxApiCaptureStatus        status{};

    UNIT_TEST_CHECK(controller.Configure(firstBegin) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(controller.AdvanceToConfig(0));
    UNIT_TEST_CHECK(controller.StartForTest(controller.ActiveConfig()));
    UNIT_TEST_CHECK(controller.CompleteForTest(arm::capture::CaptureResult::Success));
    controller.GetStatus(status, 1);
    UNIT_TEST_CHECK(status.phase == FFX_API_CAPTURE_PHASE_IDLE);
    UNIT_TEST_CHECK(status.result == FFX_API_CAPTURE_RESULT_SUCCESS);

    UNIT_TEST_CHECK(controller.Configure(secondBegin) == FFX_API_RETURN_OK);
    controller.GetStatus(status, 0);
    UNIT_TEST_CHECK(status.phase == FFX_API_CAPTURE_PHASE_IDLE);
    UNIT_TEST_CHECK(status.result == FFX_API_CAPTURE_RESULT_NONE);
    return true;
}

UNIT_TEST_CASE(capture, config_rejected_future_begin_preserves_terminal_result)
{
    OutputDirectory            output("config_rejected_future_begin_preserves_terminal_result");
    const std::string          outputPath = output.File("dataset").string();
    TestCaptureController      controller;
    ffxApiConfigureDescCapture firstBegin   = MakeBeginCapture(outputPath.c_str(), 0, 1);
    ffxApiConfigureDescCapture invalidBegin = MakeBeginCapture("", 100, 1);
    FfxApiCaptureStatus        status{};

    UNIT_TEST_CHECK(controller.Configure(firstBegin) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(controller.AdvanceToConfig(0));
    UNIT_TEST_CHECK(controller.StartForTest(controller.ActiveConfig()));
    UNIT_TEST_CHECK(controller.CompleteForTest(arm::capture::CaptureResult::Success));

    UNIT_TEST_CHECK(controller.Configure(invalidBegin) == FFX_API_RETURN_ERROR_PARAMETER);
    controller.GetStatus(status, 1);
    UNIT_TEST_CHECK(status.phase == FFX_API_CAPTURE_PHASE_IDLE);
    UNIT_TEST_CHECK(status.result == FFX_API_CAPTURE_RESULT_SUCCESS);
    return true;
}

UNIT_TEST_CASE(capture, expired_queued_begin_reports_failure)
{
    OutputDirectory            output("expired_queued_begin_reports_failure");
    const std::string          outputPath = output.File("dataset").string();
    TestCaptureController      controller;
    ffxApiConfigureDescCapture begin = MakeBeginCapture(outputPath.c_str(), 0, 1);
    FfxApiCaptureStatus        status{};

    UNIT_TEST_CHECK(controller.Configure(begin) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(!controller.AdvanceFrameForTest(1));
    controller.GetStatus(status, 0);
    UNIT_TEST_CHECK(status.phase == FFX_API_CAPTURE_PHASE_IDLE);
    UNIT_TEST_CHECK(status.result == FFX_API_CAPTURE_RESULT_FAILED);
    return true;
}

UNIT_TEST_CASE(capture, pending_session_preparation_cannot_extend_dispatch_range)
{
    OutputDirectory            output("pending_session_preparation_cannot_extend_dispatch_range");
    const std::string          outputPath = output.File("dataset").string();
    TestCaptureController      controller;
    ffxApiConfigureDescCapture begin = MakeBeginCapture(outputPath.c_str(), 7, 2);
    FfxApiCaptureStatus        status{};

    UNIT_TEST_CHECK(controller.Configure(begin) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(!controller.AdvanceDispatchWhileSessionPreparationIsPending(7));
    UNIT_TEST_CHECK(!controller.AdvanceDispatchWhileSessionPreparationIsPending(8));
    UNIT_TEST_CHECK(!controller.AdvanceDispatchWhileSessionPreparationIsPending(9));
    controller.CompleteSessionPreparationForTest();
    UNIT_TEST_CHECK(!controller.AdvanceDispatchWhileSessionPreparationIsPending(10));
    UNIT_TEST_CHECK(controller.CapturedDispatchesForTest() == 0);

    controller.GetStatus(status, 0);
    UNIT_TEST_CHECK(status.phase == FFX_API_CAPTURE_PHASE_IDLE);
    UNIT_TEST_CHECK(status.result == FFX_API_CAPTURE_RESULT_FAILED);
    UNIT_TEST_CHECK(status.capturedFrames == 0);
    UNIT_TEST_CHECK(status.requestedFrames == 2);
    return true;
}

UNIT_TEST_CASE(capture, output_directory_collision_is_rejected)
{
    OutputDirectory             output("output_directory_reservation");
    const std::filesystem::path requested     = output.File("dataset");
    const std::string           requestedPath = requested.string();
    TestCaptureController       firstController;
    TestCaptureController       secondController;
    ffxApiConfigureDescCapture  firstConfigure  = MakeBeginCapture(requestedPath.c_str(), 0, 1);
    ffxApiConfigureDescCapture  secondConfigure = MakeBeginCapture(requestedPath.c_str(), 0, 1);

    UNIT_TEST_CHECK(firstController.Configure(firstConfigure) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(firstController.AdvanceToConfig(0));
    UNIT_TEST_CHECK(firstController.StartForTest(firstController.ActiveConfig()));
    UNIT_TEST_CHECK(firstController.ActiveConfig().outputDirectory == requestedPath);
    UNIT_TEST_CHECK(secondController.Configure(secondConfigure) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(secondController.AdvanceToConfig(0));
    UNIT_TEST_CHECK(!secondController.StartForTest(secondController.ActiveConfig()));
    UNIT_TEST_CHECK(std::filesystem::is_directory(firstController.ActiveConfig().outputDirectory));
    UNIT_TEST_CHECK(secondController.ActiveConfig().outputDirectory == requestedPath);
    return true;
}

UNIT_TEST_CASE(capture, output_directory_concurrent_creation_is_atomic)
{
    OutputDirectory             output("output_directory_concurrent_creation");
    const std::filesystem::path requested     = output.File("dataset");
    const std::string           requestedPath = requested.string();
    std::array<std::string, 2>  paths{requestedPath, requestedPath};
    std::array<bool, 2>         created{};
    std::atomic_uint32_t        ready{0};
    std::atomic<bool>           start{false};

    const auto create = [&paths, &created, &ready, &start](size_t index) {
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire))
            std::this_thread::yield();
        created[index] = arm::capture::EnsureUniqueOutputDirectory(paths[index]);
    };
    std::thread first(create, 0);
    std::thread second(create, 1);
    while (ready.load(std::memory_order_acquire) != paths.size())
        std::this_thread::yield();
    start.store(true, std::memory_order_release);
    first.join();
    second.join();

    UNIT_TEST_CHECK(created[0] != created[1]);
    UNIT_TEST_CHECK(std::filesystem::is_directory(requestedPath));
    return true;
}

UNIT_TEST_CASE(capture, config_invalid_operation)
{
    arm::capture::Execution    execution;
    ffxApiConfigureDescCapture configure = MakeBeginCapture("nss_capture_test", 0, 1);
    configure.operation                  = static_cast<FfxApiCaptureOperation>(99);

    UNIT_TEST_CHECK(!arm::capture::ConfigureCapture(execution, configure, [](const std::optional<arm::capture::Config>&, bool) { return true; }));
    return true;
}

UNIT_TEST_CASE(capture, config_invalid_settings)
{
    TestCaptureController controller(true);

    ffxApiConfigureDescCapture emptyDirectory = MakeBeginCapture("", 0, 1);
    UNIT_TEST_CHECK(controller.Configure(emptyDirectory) == FFX_API_RETURN_ERROR_PARAMETER);

    ffxApiConfigureDescCapture zeroFrames = MakeBeginCapture("capture", 0, 0);
    UNIT_TEST_CHECK(controller.Configure(zeroFrames) == FFX_API_RETURN_ERROR_PARAMETER);

    ffxApiConfigureDescCapture nonFiniteBias = MakeBeginCapture("capture", 0, 1);
    nonFiniteBias.motionVectorUnormBias      = std::numeric_limits<float>::quiet_NaN();
    nonFiniteBias.motionVectorUnormScale     = 2.0f;
    UNIT_TEST_CHECK(controller.Configure(nonFiniteBias) == FFX_API_RETURN_ERROR_PARAMETER);

    ffxApiConfigureDescCapture nonFiniteScale = MakeBeginCapture("capture", 0, 1);
    nonFiniteScale.motionVectorUnormScale     = std::numeric_limits<float>::infinity();
    UNIT_TEST_CHECK(controller.Configure(nonFiniteScale) == FFX_API_RETURN_ERROR_PARAMETER);
    ffxApiConfigureDescCapture zeroScale = MakeBeginCapture("capture", 0, 1);
    zeroScale.motionVectorUnormScale     = 0.0f;
    UNIT_TEST_CHECK(controller.Configure(zeroScale) == FFX_API_RETURN_ERROR_PARAMETER);
    ffxApiConfigureDescCapture overflowingScale = MakeBeginCapture("capture", 0, 1);
    overflowingScale.motionVectorUnormBias      = -1.0f;
    overflowingScale.motionVectorUnormScale     = std::numeric_limits<float>::max();
    UNIT_TEST_CHECK(controller.Configure(overflowingScale) == FFX_API_RETURN_ERROR_PARAMETER);
    return true;
}

UNIT_TEST_CASE(capture, config_requires_environment_revision)
{
#if !FFX_CAPTURE_ENABLED || defined(__ANDROID__)
    return true;
#else
#ifdef _WIN32
    UNIT_TEST_CHECK(_putenv_s("TEST_CAPTURE_DIR", "environment-capture") == 0);
    UNIT_TEST_CHECK(_putenv_s("TEST_CAPTURE_FRAME_COUNT", "1") == 0);
    UNIT_TEST_CHECK(_putenv_s("TEST_CAPTURE_CONFIG_REVISION", "") == 0);
#else
    setenv("TEST_CAPTURE_DIR", "environment-capture", 1);
    setenv("TEST_CAPTURE_FRAME_COUNT", "1", 1);
    unsetenv("TEST_CAPTURE_CONFIG_REVISION");
#endif

    EnvironmentConfigReader reader("TEST_CAPTURE_", "TEST-CAPTURE");
    UNIT_TEST_CHECK(!reader.ReadConfig().has_value());

#ifdef _WIN32
    _putenv_s("TEST_CAPTURE_DIR", "");
    _putenv_s("TEST_CAPTURE_FRAME_COUNT", "");
    _putenv_s("TEST_CAPTURE_CONFIG_REVISION", "");
#else
    unsetenv("TEST_CAPTURE_DIR");
    unsetenv("TEST_CAPTURE_FRAME_COUNT");
    unsetenv("TEST_CAPTURE_CONFIG_REVISION");
#endif
    return true;
#endif
}

UNIT_TEST_CASE(capture, config_api_begin_supersedes_pending_environment_capture)
{
#if FFX_CAPTURE_ENABLED && !defined(__ANDROID__)
    TestEnvScope scope("environment-capture", "1", "10");

    TestCaptureController controller(true);
    UNIT_TEST_CHECK(controller.HasPendingConfig(0));
    const ffxApiConfigureDescCapture apiBegin = MakeBeginCapture("api-capture", 0, 1);
    UNIT_TEST_CHECK(controller.Configure(apiBegin) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(controller.AdvanceToConfig(0));
    UNIT_TEST_CHECK(!controller.HasPendingConfig(0));
#else
    return true;
#endif
    return true;
}

UNIT_TEST_CASE(capture, config_api_begin_supersedes_environment_published_between_polls)
{
#if FFX_CAPTURE_ENABLED
    TestEnvScope          first("environment-first", "1", "1");
    TestCaptureController controller(true);
    UNIT_TEST_CHECK(controller.HasPendingConfig(0));
    UNIT_TEST_CHECK(controller.AdvanceToConfig(0));

    TestEnvScope                     second("environment-second", "1", "2");
    const ffxApiConfigureDescCapture apiBegin = MakeBeginCapture("api-capture", 0, 1);
    UNIT_TEST_CHECK(controller.Configure(apiBegin) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(controller.AdvanceToConfig(0));
    UNIT_TEST_CHECK(!controller.HasPendingConfig(30));
#else
    return true;
#endif
    return true;
}

UNIT_TEST_CASE(capture, config_missing_revision_is_ignored_until_supplied)
{
    TestEnvScope          scopeWithoutRev("environment-capture", "1", nullptr);
    TestCaptureController idleController(true);
    UNIT_TEST_CHECK(!idleController.HasPendingConfig(0));

    TestEnvScope          scopeWithRev("environment-capture", "1", "1");
    TestCaptureController activeController(true);
    UNIT_TEST_CHECK(activeController.HasPendingConfig(0));
    return true;
}

UNIT_TEST_CASE(capture, config_revisions_must_increase)
{
    TestEnvScope scope("environment-capture", "1", "10");
#if defined(__ANDROID__)
    EnvironmentConfigReader reader("debug.ngsdk.test.", "TEST-CAPTURE");
    const char*             revisionName = "debug.ngsdk.test.revision";
#else
    EnvironmentConfigReader reader("TEST_CAPTURE_", "TEST-CAPTURE");
    const char* revisionName = "TEST_CAPTURE_CONFIG_REVISION";
#endif
    UNIT_TEST_CHECK(reader.ReadConfig().has_value());

    UNIT_TEST_CHECK(scope.SetVariable(revisionName, "9"));
    UNIT_TEST_CHECK(!reader.ReadConfig().has_value());

    UNIT_TEST_CHECK(scope.SetVariable(revisionName, "10"));
    UNIT_TEST_CHECK(!reader.ReadConfig().has_value());

    UNIT_TEST_CHECK(scope.SetVariable(revisionName, "11"));
    UNIT_TEST_CHECK(reader.ReadConfig().has_value());
    return true;
}

UNIT_TEST_CASE(capture, config_invalid_environment_start_frame_is_rejected)
{
#if defined(__ANDROID__)
    MarkTestSkipped("uses desktop environment variables; Android property coverage is provided separately");
    return true;
#else
    TestEnvScope scope(nullptr, nullptr, nullptr);
#ifdef _WIN32
    UNIT_TEST_CHECK(_putenv_s("TEST_CAPTURE_DIR", "environment-capture") == 0);
    UNIT_TEST_CHECK(_putenv_s("TEST_CAPTURE_FRAME_COUNT", "1") == 0);
    UNIT_TEST_CHECK(_putenv_s("TEST_CAPTURE_START_FRAME", "not-a-frame") == 0);
    UNIT_TEST_CHECK(_putenv_s("TEST_CAPTURE_MV_UNORM_SCALE", "0") == 0);
    UNIT_TEST_CHECK(_putenv_s("TEST_CAPTURE_CONFIG_REVISION", "2") == 0);
#else
    setenv("TEST_CAPTURE_DIR", "environment-capture", 1);
    setenv("TEST_CAPTURE_FRAME_COUNT", "1", 1);
    setenv("TEST_CAPTURE_START_FRAME", "not-a-frame", 1);
    setenv("TEST_CAPTURE_MV_UNORM_SCALE", "0", 1);
    setenv("TEST_CAPTURE_CONFIG_REVISION", "2", 1);
#endif

    TestCaptureController controller(true);
    UNIT_TEST_CHECK(!controller.HasPendingConfig(0));
    UNIT_TEST_CHECK(!controller.AdvanceToConfig(0));

#ifdef _WIN32
    UNIT_TEST_CHECK(_putenv_s("TEST_CAPTURE_START_FRAME", " -1") == 0);
    UNIT_TEST_CHECK(_putenv_s("TEST_CAPTURE_CONFIG_REVISION", "3") == 0);
#else
    setenv("TEST_CAPTURE_START_FRAME", " -1", 1);
    setenv("TEST_CAPTURE_CONFIG_REVISION", "3", 1);
#endif
    TestCaptureController whitespaceNegativeController(true);
    UNIT_TEST_CHECK(!whitespaceNegativeController.HasPendingConfig(0));

#ifdef _WIN32
    UNIT_TEST_CHECK(_putenv_s("TEST_CAPTURE_START_FRAME", "0") == 0);
    UNIT_TEST_CHECK(_putenv_s("TEST_CAPTURE_CONFIG_REVISION", "4") == 0);
#else
    setenv("TEST_CAPTURE_START_FRAME", "0", 1);
    setenv("TEST_CAPTURE_CONFIG_REVISION", "4", 1);
#endif
    TestCaptureController zeroScaleEnvironmentController(true);
    UNIT_TEST_CHECK(!zeroScaleEnvironmentController.HasPendingConfig(0));

#ifdef _WIN32
    _putenv_s("TEST_CAPTURE_DIR", "");
    _putenv_s("TEST_CAPTURE_FRAME_COUNT", "");
    _putenv_s("TEST_CAPTURE_START_FRAME", "");
    _putenv_s("TEST_CAPTURE_MV_UNORM_SCALE", "");
    _putenv_s("TEST_CAPTURE_MV_UNORM_BIAS", "");
    _putenv_s("TEST_CAPTURE_CONFIG_REVISION", "");
#else
    unsetenv("TEST_CAPTURE_DIR");
    unsetenv("TEST_CAPTURE_FRAME_COUNT");
    unsetenv("TEST_CAPTURE_START_FRAME");
    unsetenv("TEST_CAPTURE_MV_UNORM_SCALE");
    unsetenv("TEST_CAPTURE_MV_UNORM_BIAS");
    unsetenv("TEST_CAPTURE_CONFIG_REVISION");
#endif
    return true;
#endif
}

UNIT_TEST_CASE(capture, environment_bias_overflow_is_rejected_on_platform)
{
    TestEnvScope          scope("environment-capture", "1", "1", nullptr, "2", "3.4e38");
    TestCaptureController controller(true);
    UNIT_TEST_CHECK(!controller.HasPendingConfig(0));
    {
        TestEnvScope updatedScope("environment-capture", "1", "2", nullptr, "4", "0.25");
        UNIT_TEST_CHECK(!controller.HasPendingConfig(30));
    }
    return true;
}

UNIT_TEST_CASE(capture, android_capture_configuration_is_property_only)
{
#if defined(__ANDROID__)
    UNIT_TEST_CHECK(__system_property_set("debug.ngsdk.test.dir", "property-capture") == 0);
    UNIT_TEST_CHECK(__system_property_set("debug.ngsdk.test.frames", "2") == 0);
    UNIT_TEST_CHECK(__system_property_set("debug.ngsdk.test.revision", "1") == 0);
    setenv("debug.ngsdk.test.DIR", "environment-capture", 1);
    setenv("debug.ngsdk.test.FRAME_COUNT", "99", 1);
    setenv("debug.ngsdk.test.CONFIG_REVISION", "2", 1);

    EnvironmentConfigReader     reader("debug.ngsdk.test.", "TEST-CAPTURE");
    const std::optional<Config> propertyConfig = reader.ReadConfig();
    UNIT_TEST_CHECK(propertyConfig.has_value());
    UNIT_TEST_CHECK(propertyConfig->outputDirectory == "property-capture");
    UNIT_TEST_CHECK(propertyConfig->maxFrameCount == 2);

    UNIT_TEST_CHECK(__system_property_set("debug.ngsdk.test.dir", "") == 0);
    UNIT_TEST_CHECK(__system_property_set("debug.ngsdk.test.frames", "") == 0);
    UNIT_TEST_CHECK(__system_property_set("debug.ngsdk.test.revision", "2") == 0);
    EnvironmentConfigReader noFallbackReader("debug.ngsdk.test.", "TEST-CAPTURE");
    UNIT_TEST_CHECK(!noFallbackReader.ReadConfig().has_value());

    unsetenv("debug.ngsdk.test.DIR");
    unsetenv("debug.ngsdk.test.FRAME_COUNT");
    unsetenv("debug.ngsdk.test.CONFIG_REVISION");
    __system_property_set("debug.ngsdk.test.dir", "");
    __system_property_set("debug.ngsdk.test.frames", "");
    __system_property_set("debug.ngsdk.test.revision", "");
#endif
    return true;
}

UNIT_TEST_CASE(capture, environment_revisions_queue_independent_requests)
{
    TestEnvScope          initial("environment-first", "1", "1", "100");
    TestCaptureController controller(true);
    UNIT_TEST_CHECK(controller.HasPendingConfig(0));

    {
        TestEnvScope newest("environment-newest", "1", "2", "200");
        UNIT_TEST_CHECK(controller.HasPendingConfig(30));
    }
    UNIT_TEST_CHECK(controller.AdvanceToConfig(100));
    UNIT_TEST_CHECK(controller.ActiveConfig().outputDirectory == "environment-first");
    UNIT_TEST_CHECK(controller.AdvanceToConfig(200));
    UNIT_TEST_CHECK(controller.ActiveConfig().outputDirectory == "environment-newest");
    return true;
}

UNIT_TEST_CASE(capture, environment_revisions_bound_pending_requests)
{
    TestEnvScope          environmentScope(nullptr, nullptr, nullptr);
    TestCaptureController controller(true);
    controller.EnableEnvironmentPollingForTest();
    for (uint32_t revision = 1; revision <= 17; ++revision)
    {
        const std::string directory     = "environment-" + std::to_string(revision);
        const std::string revisionValue = std::to_string(revision);
        TestEnvScope      scope(directory.c_str(), "1", revisionValue.c_str(), revisionValue.c_str());
        UNIT_TEST_CHECK(controller.HasPendingConfig(revision * 30));
    }
    for (uint32_t revision = 1; revision <= 16; ++revision)
    {
        UNIT_TEST_CHECK(controller.AdvanceToConfig(revision));
        UNIT_TEST_CHECK(controller.ActiveConfig().outputDirectory == "environment-" + std::to_string(revision));
    }
    UNIT_TEST_CHECK(!controller.AdvanceConfig(17));
    return true;
}

UNIT_TEST_CASE(capture, environment_configuration_value_length_is_bounded)
{
#if !FFX_CAPTURE_ENABLED || defined(__ANDROID__)
    return true;
#else
    const std::string acceptedDirectory(arm::capture::kMaxCaptureOutputDirectoryLength, 'd');
    const std::string rejectedDirectory(arm::capture::kMaxCaptureOutputDirectoryLength + 1, 'd');
    arm::capture::EnvironmentConfigReader reader("TEST_CAPTURE_", "TEST-CAPTURE");
    {
        TestEnvScope scope(acceptedDirectory.c_str(), "1", "1");
        UNIT_TEST_CHECK(reader.ReadConfig().has_value());
    }
    {
        TestEnvScope scope(rejectedDirectory.c_str(), "1", "2");
        UNIT_TEST_CHECK(!reader.ReadConfig().has_value());
    }
    const std::string acceptedBias(4094, '0');
    const std::string rejectedBias(4095, '0');
    {
        TestEnvScope scope("environment-capture", "1", "3", nullptr, "2", (acceptedBias + ".5").c_str());
        UNIT_TEST_CHECK(reader.ReadConfig().has_value());
    }
    {
        TestEnvScope scope("environment-capture", "1", "4", nullptr, "2", (rejectedBias + ".5").c_str());
        UNIT_TEST_CHECK(!reader.ReadConfig().has_value());
    }
    return true;
#endif
}

UNIT_TEST_CASE(capture, config_invalid_api_begin_preserves_environment_configuration)
{
    TestEnvScope          scope("environment-capture", "1", "1");
    TestCaptureController controller(true);
    auto                  invalidBegin = MakeBeginCapture("", 0, 1);
    UNIT_TEST_CHECK(controller.Configure(invalidBegin) == FFX_API_RETURN_ERROR_PARAMETER);
    UNIT_TEST_CHECK(controller.HasPendingConfig());
    UNIT_TEST_CHECK(controller.AdvanceToConfig(0));
    UNIT_TEST_CHECK(controller.ActiveConfig().outputDirectory == "environment-capture");
    return true;
}

UNIT_TEST_CASE(capture, config_api_begin_output_directory_length_is_bounded)
{
    const std::string acceptedDirectory(arm::capture::kMaxCaptureOutputDirectoryLength, 'a');
    const std::string rejectedDirectory(arm::capture::kMaxCaptureOutputDirectoryLength + 1, 'a');

    TestCaptureController acceptedController;
    UNIT_TEST_CHECK(acceptedController.Configure(MakeBeginCapture(acceptedDirectory.c_str(), 0, 1)) == FFX_API_RETURN_OK);

    TestCaptureController rejectedController;
    UNIT_TEST_CHECK(rejectedController.Configure(MakeBeginCapture(rejectedDirectory.c_str(), 0, 1)) == FFX_API_RETURN_ERROR_PARAMETER);
    return true;
}

UNIT_TEST_CASE(capture, config_state_machine_rejects_overlapping_operations)
{
    TestCaptureController controller;

    auto       begin     = MakeBeginCapture("capture-state-machine", 0, 1);
    const auto nextBegin = MakeBeginCapture("capture-state-machine-next", 0, 1);
    UNIT_TEST_CHECK(controller.Configure(nextBegin) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(controller.Configure(begin) == FFX_API_RETURN_ERROR_PARAMETER);

    ffxApiConfigureDescCapture cancel = begin;
    cancel.operation                  = FFX_API_CAPTURE_CANCEL;
    UNIT_TEST_CHECK(controller.Configure(cancel) == FFX_API_RETURN_ERROR_PARAMETER);

    UNIT_TEST_CHECK(controller.AdvanceToConfig(0));
    UNIT_TEST_CHECK(controller.StartForTest("TEST-CAPTURE"));
    UNIT_TEST_CHECK(controller.Configure(begin) == FFX_API_RETURN_ERROR_PARAMETER);
    UNIT_TEST_CHECK(controller.Configure(cancel) == FFX_API_RETURN_OK);

    UNIT_TEST_CHECK(controller.Configure(begin) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(controller.Configure(cancel) == FFX_API_RETURN_ERROR_PARAMETER);
    return true;
}

UNIT_TEST_CASE(capture, cancel_selects_api_input_mode_for_future_requests)
{
    TestEnvScope          scope("environment-capture", "1", "10");
    TestCaptureController controller(true);
    UNIT_TEST_CHECK(controller.HasPendingConfig(0));
    UNIT_TEST_CHECK(controller.AdvanceToConfig(0));
    UNIT_TEST_CHECK(controller.StartForTest("TEST-CAPTURE"));

    auto cancel      = MakeBeginCapture("api-capture", 0, 1);
    cancel.operation = FFX_API_CAPTURE_CANCEL;
    UNIT_TEST_CHECK(controller.Configure(cancel) == FFX_API_RETURN_OK);

#if defined(__ANDROID__)
    scope.SetVariable("debug.ngsdk.test.frames", "1");
    scope.SetVariable("debug.ngsdk.test.dir", "environment-capture-next");
    scope.SetVariable("debug.ngsdk.test.revision", "2");
#else
    scope.SetVariable("TEST_CAPTURE_FRAME_COUNT", "1");
    scope.SetVariable("TEST_CAPTURE_DIR", "environment-capture-next");
    scope.SetVariable("TEST_CAPTURE_CONFIG_REVISION", "2");
#endif
    UNIT_TEST_CHECK(!controller.HasPendingConfig(60));
    return true;
}

UNIT_TEST_CASE(capture, environment_polling_requires_startup_configuration)
{
#if !FFX_CAPTURE_ENABLED || defined(__ANDROID__)
    return true;
#else
    OutputDirectory output("environment_polling_requires_startup_configuration");
    const std::string apiOutput = output.File("api-capture").string();
#ifdef _WIN32
    _putenv_s("TEST_CAPTURE_DIR", "");
    _putenv_s("TEST_CAPTURE_FRAME_COUNT", "");
    _putenv_s("TEST_CAPTURE_START_FRAME", "");
    _putenv_s("TEST_CAPTURE_CONFIG_REVISION", "");
#else
    unsetenv("TEST_CAPTURE_DIR");
    unsetenv("TEST_CAPTURE_FRAME_COUNT");
    unsetenv("TEST_CAPTURE_START_FRAME");
    unsetenv("TEST_CAPTURE_CONFIG_REVISION");
#endif
    TestCaptureController controller(true);

#ifdef _WIN32
    _putenv_s("TEST_CAPTURE_FRAME_COUNT", "1");
    _putenv_s("TEST_CAPTURE_DIR", "environment-capture-next");
    _putenv_s("TEST_CAPTURE_CONFIG_REVISION", "2");
#else
    setenv("TEST_CAPTURE_FRAME_COUNT", "1", 1);
    setenv("TEST_CAPTURE_DIR", "environment-capture-next", 1);
    setenv("TEST_CAPTURE_CONFIG_REVISION", "2", 1);
#endif
    UNIT_TEST_CHECK(!controller.HasPendingConfig(30));

    const auto begin = MakeBeginCapture(apiOutput.c_str(), 0, 1);
    UNIT_TEST_CHECK(controller.Configure(begin) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(controller.AdvanceFrameForTest(0));
    UNIT_TEST_CHECK(controller.CompleteForTest());
    UNIT_TEST_CHECK(!controller.HasPendingConfig(30));

#ifdef _WIN32
    _putenv_s("TEST_CAPTURE_DIR", "");
    _putenv_s("TEST_CAPTURE_FRAME_COUNT", "");
    _putenv_s("TEST_CAPTURE_CONFIG_REVISION", "");
    _putenv_s("TEST_CAPTURE_START_FRAME", "");
#else
    unsetenv("TEST_CAPTURE_DIR");
    unsetenv("TEST_CAPTURE_FRAME_COUNT");
    unsetenv("TEST_CAPTURE_CONFIG_REVISION");
    unsetenv("TEST_CAPTURE_START_FRAME");
#endif
    return true;
#endif
}

UNIT_TEST_CASE(capture, execution_drain)
{
    arm::capture::Execution execution;

    UNIT_TEST_CHECK(execution.GetState() == arm::capture::ExecutionPhase::Idle);
    UNIT_TEST_CHECK(execution.StartCapture("capture"));
    UNIT_TEST_CHECK(execution.GetState() == arm::capture::ExecutionPhase::Capturing);
    UNIT_TEST_CHECK(!execution.StartCapture("capture-again"));

    UNIT_TEST_CHECK(execution.RequestDrain(arm::capture::CaptureResult::Success));
    UNIT_TEST_CHECK(execution.GetState() == arm::capture::ExecutionPhase::Draining);
    UNIT_TEST_CHECK(execution.TryFinalizeDrain());
    UNIT_TEST_CHECK(execution.GetState() == arm::capture::ExecutionPhase::Idle);
    UNIT_TEST_CHECK(execution.GetResult() == arm::capture::CaptureResult::Success);
    UNIT_TEST_CHECK(!execution.TryFinalizeDrain());

    arm::capture::Execution abortedExecution;
    UNIT_TEST_CHECK(abortedExecution.StartCapture("capture-abort"));
    UNIT_TEST_CHECK(abortedExecution.RequestDrain(arm::capture::CaptureResult::Cancelled));
    UNIT_TEST_CHECK(abortedExecution.TryFinalizeDrain());
    UNIT_TEST_CHECK(abortedExecution.GetState() == arm::capture::ExecutionPhase::Idle);
    UNIT_TEST_CHECK(abortedExecution.GetResult() == arm::capture::CaptureResult::Cancelled);

    arm::capture::Execution failedExecution;
    UNIT_TEST_CHECK(failedExecution.StartCapture("capture-failed"));
    UNIT_TEST_CHECK(failedExecution.RequestDrain(arm::capture::CaptureResult::Failed));
    UNIT_TEST_CHECK(failedExecution.GetState() == arm::capture::ExecutionPhase::Draining);
    UNIT_TEST_CHECK(failedExecution.TryFinalizeDrain());
    UNIT_TEST_CHECK(failedExecution.GetState() == arm::capture::ExecutionPhase::Idle);
    UNIT_TEST_CHECK(failedExecution.GetResult() == arm::capture::CaptureResult::Failed);

    FfxApiCaptureStatus  status{};
    arm::capture::Config config{};
    config.maxFrameCount = 4;
    arm::capture::FillCaptureStatus(failedExecution, config, status, 3);
    UNIT_TEST_CHECK(status.phase == FFX_API_CAPTURE_PHASE_IDLE);
    UNIT_TEST_CHECK(status.result == FFX_API_CAPTURE_RESULT_FAILED);
    UNIT_TEST_CHECK(status.requestedFrames == 4);
    UNIT_TEST_CHECK(status.capturedFrames == 3);
    return true;
}

UNIT_TEST_CASE(capture, execution_invalid_drain_preserves_capture)
{
    arm::capture::Execution execution;
    UNIT_TEST_CHECK(execution.StartCapture("capture-invalid-drain"));
    UNIT_TEST_CHECK(!execution.RequestDrain(arm::capture::CaptureResult::None));
    UNIT_TEST_CHECK(execution.GetState() == arm::capture::ExecutionPhase::Capturing);
    UNIT_TEST_CHECK(execution.GetResult() == arm::capture::CaptureResult::None);
    return true;
}

UNIT_TEST_CASE(capture, execution_shutdown_after_immediate_drain_preserves_cancelled_result)
{
    arm::capture::Execution execution;
    UNIT_TEST_CHECK(execution.StartCapture("capture-shutdown-immediate"));
    UNIT_TEST_CHECK(execution.RequestDrain(arm::capture::CaptureResult::Cancelled));
    UNIT_TEST_CHECK(execution.TryFinalizeDrain());
    UNIT_TEST_CHECK(execution.GetState() == arm::capture::ExecutionPhase::Idle);
    UNIT_TEST_CHECK(execution.GetResult() == arm::capture::CaptureResult::Cancelled);

    execution.DrainAndDestroyForShutdown(1);
    UNIT_TEST_CHECK(execution.GetState() == arm::capture::ExecutionPhase::Idle);
    UNIT_TEST_CHECK(execution.GetResult() == arm::capture::CaptureResult::Cancelled);
    return true;
}

UNIT_TEST_CASE(capture, execution_shutdown_rejects_active_transaction)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    auto                    sink = std::make_shared<FakeMetadataSink>();
    arm::capture::Execution execution;
    UNIT_TEST_CHECK(execution.StartCapture("capture-shutdown-active"));
    UNIT_TEST_CHECK(execution.EnsureSession(MakeDescriptor(context, sink)));

    std::string     error;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, commandBuffer, &error));
    UNIT_TEST_CHECK(execution.BeginTransaction(64, reinterpret_cast<FfxCommandList>(commandBuffer), std::make_unique<arm::capture::FrameMeta>()));

    UNIT_TEST_CHECK(!execution.DrainAndDestroyForShutdown(1));
    UNIT_TEST_CHECK(execution.GetState() == arm::capture::ExecutionPhase::Capturing);
    UNIT_TEST_CHECK(execution.GetResult() == arm::capture::CaptureResult::None);

    (void)execution.CancelTransaction();
    (void)execution.RequestDrain(arm::capture::CaptureResult::Cancelled);
    (void)execution.TryFinalizeDrain();
    UNIT_TEST_CHECK(execution.DrainAndDestroyForShutdown(1000));
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, execution_reuses_after_terminal_results)
{
    for (const arm::capture::CaptureResult result :
         {arm::capture::CaptureResult::Success, arm::capture::CaptureResult::Cancelled, arm::capture::CaptureResult::Failed})
    {
        arm::capture::Execution execution;
        UNIT_TEST_CHECK(execution.StartCapture("capture-reuse"));
        UNIT_TEST_CHECK(execution.RequestDrain(result));
        UNIT_TEST_CHECK(execution.TryFinalizeDrain());
        UNIT_TEST_CHECK(execution.GetState() == arm::capture::ExecutionPhase::Idle);
        UNIT_TEST_CHECK(execution.GetResult() == result);

        UNIT_TEST_CHECK(execution.StartCapture("capture-reuse-again"));
        UNIT_TEST_CHECK(execution.GetState() == arm::capture::ExecutionPhase::Capturing);
        UNIT_TEST_CHECK(execution.GetResult() == arm::capture::CaptureResult::None);
    }
    return true;
}

UNIT_TEST_CASE(capture, execution_final_status_is_stable_until_next_begin)
{
    arm::capture::Execution execution;
    arm::capture::Config    config{};
    config.maxFrameCount = 4;

    UNIT_TEST_CHECK(execution.StartCapture("capture-stable-status"));
    UNIT_TEST_CHECK(execution.RequestDrain(arm::capture::CaptureResult::Cancelled));
    UNIT_TEST_CHECK(execution.TryFinalizeDrain());

    FfxApiCaptureStatus first{};
    FfxApiCaptureStatus second{};
    arm::capture::FillCaptureStatus(execution, config, first, 2);
    arm::capture::FillCaptureStatus(execution, config, second, 2);
    UNIT_TEST_CHECK(first.phase == FFX_API_CAPTURE_PHASE_IDLE);
    UNIT_TEST_CHECK(first.result == FFX_API_CAPTURE_RESULT_CANCELLED);
    UNIT_TEST_CHECK(first.capturedFrames == 2);
    UNIT_TEST_CHECK(first.requestedFrames == 4);
    UNIT_TEST_CHECK(std::memcmp(&first, &second, sizeof(first)) == 0);

    UNIT_TEST_CHECK(execution.StartCapture("capture-stable-status-again"));
    FfxApiCaptureStatus reset{};
    arm::capture::FillCaptureStatus(execution, config, reset, 0);
    UNIT_TEST_CHECK(reset.phase == FFX_API_CAPTURE_PHASE_CAPTURING);
    UNIT_TEST_CHECK(reset.result == FFX_API_CAPTURE_RESULT_NONE);
    return true;
}

UNIT_TEST_CASE(capture, execution_session_io_failure_status)
{
    unit_test::vk::VulkanContext context{};
    UNIT_TEST_CHECK(InitializeCaptureContext(context));

    unit_test::vk::Image image{};
    std::string          error;
    UNIT_TEST_CHECK(
        unit_test::vk::CreateImage2D(context, 16, 16, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, image, &error));
    UNIT_TEST_CHECK(unit_test::vk::ClearImage(context, image, {0.25f, 0.0f, 0.0f, 1.0f}, &error));

    const auto              output = OutputDirectory("execution_session_io_failure");
    const auto              sink   = std::make_shared<FakeMetadataSink>(false);
    arm::capture::Execution execution;
    UNIT_TEST_CHECK(execution.StartCapture("capture-io-failure"));

    arm::capture::SessionDescriptor descriptor = MakeDescriptor(context, sink);
    UNIT_TEST_CHECK(execution.EnsureSession(descriptor));

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    UNIT_TEST_CHECK(unit_test::vk::BeginCommandBuffer(context, commandBuffer, &error));
    const FfxResource source = unit_test::vk::MakeFfxImageResource(image, FFX_RESOURCE_USAGE_READ_ONLY, FFX_RESOURCE_STATE_GENERIC_UAV, "execution_image");
    UNIT_TEST_CHECK(execution.BeginTransaction(16 * 16, reinterpret_cast<FfxCommandList>(commandBuffer), std::make_unique<arm::capture::FrameMeta>()));
    UNIT_TEST_CHECK(execution.AddAttachment(source, std::make_unique<arm::capture::BinaryPayload>("execution_image", output.File("image.bin").string())));
    UNIT_TEST_CHECK(execution.CommitTransaction());
    UNIT_TEST_CHECK(unit_test::vk::SubmitCommandBuffer(context, commandBuffer, &error));

    UNIT_TEST_CHECK(execution.RequestDrain(arm::capture::CaptureResult::Success));
    bool finalized = false;
    for (uint32_t attempt = 0; attempt < 5000 && !finalized; ++attempt)
    {
        finalized = execution.TryFinalizeDrain();
        if (!finalized)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    UNIT_TEST_CHECK(finalized);

    FfxApiCaptureStatus  status{};
    arm::capture::Config config{};
    config.maxFrameCount = 1;
    arm::capture::FillCaptureStatus(execution, config, status, 0);
    UNIT_TEST_CHECK(status.phase == FFX_API_CAPTURE_PHASE_IDLE);
    UNIT_TEST_CHECK(status.result == FFX_API_CAPTURE_RESULT_FAILED);

    unit_test::vk::DestroyImage(context, image);
    unit_test::vk::DestroyVulkanContext(context);
    return true;
}

UNIT_TEST_CASE(capture, execution_failed_result_dominates_concurrent_drain)
{
    std::atomic<arm::capture::CaptureResult> result{arm::capture::CaptureResult::None};
    std::atomic_bool                         start{false};

    std::thread successDrain([&] {
        while (!start.load(std::memory_order_acquire))
            std::this_thread::yield();
        for (uint32_t iteration = 0; iteration < 1000; ++iteration)
            arm::capture::SetCaptureResult(result, arm::capture::CaptureResult::Success);
    });
    std::thread failedDrain([&] {
        while (!start.load(std::memory_order_acquire))
            std::this_thread::yield();
        arm::capture::SetCaptureResult(result, arm::capture::CaptureResult::Failed);
    });

    start.store(true, std::memory_order_release);
    successDrain.join();
    failedDrain.join();

    UNIT_TEST_CHECK(result.load(std::memory_order_acquire) == arm::capture::CaptureResult::Failed);
    return true;
}

UNIT_TEST_CASE(capture, execution_failed_result_dominates_concurrent_shutdown)
{
    std::atomic<arm::capture::CaptureResult> result{arm::capture::CaptureResult::None};
    std::atomic_bool                         start{false};

    std::thread shutdown([&] {
        while (!start.load(std::memory_order_acquire))
            std::this_thread::yield();
        for (uint32_t iteration = 0; iteration < 1000; ++iteration)
            arm::capture::SetCaptureResult(result, arm::capture::CaptureResult::Cancelled);
    });
    std::thread failure([&] {
        while (!start.load(std::memory_order_acquire))
            std::this_thread::yield();
        arm::capture::SetCaptureResult(result, arm::capture::CaptureResult::Failed);
    });

    start.store(true, std::memory_order_release);
    shutdown.join();
    failure.join();

    UNIT_TEST_CHECK(result.load(std::memory_order_acquire) == arm::capture::CaptureResult::Failed);
    return true;
}

// Provider integration: the NSS controller must produce a complete one-frame dataset through the runtime.
UNIT_TEST_CASE(capture, nss_capture_frame)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!InitializeCaptureContext(vkContext))
        UNIT_TEST_FAIL("Vulkan unavailable for NSS capture smoke test.");

    constexpr uint32_t kRenderWidth    = 32;
    constexpr uint32_t kRenderHeight   = 32;
    constexpr uint32_t kUpscaleWidth   = 64;
    constexpr uint32_t kUpscaleHeight  = 64;
    constexpr uint32_t kExposureWidth  = 2;
    constexpr uint32_t kExposureHeight = 3;
    constexpr uint32_t kCaptureWindow  = 500;

    unit_test::vk::Image colorImage{};
    unit_test::vk::Image depthImage{};
    unit_test::vk::Image motionImage{};
    unit_test::vk::Image outputImage{};
    unit_test::vk::Image exposureImage{};
    bool                 success       = false;
    std::string          failureReason = "NSS capture smoke test failed.";

    do
    {
        const VkImageUsageFlags inputUsage =
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        const VkImageUsageFlags outputUsage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

        if (!unit_test::vk::CreateImage2D(vkContext, kRenderWidth, kRenderHeight, VK_FORMAT_B10G11R11_UFLOAT_PACK32, inputUsage, colorImage, &errorMessage) ||
            !unit_test::vk::CreateImage2D(vkContext, kRenderWidth, kRenderHeight, VK_FORMAT_R32_SFLOAT, inputUsage, depthImage, &errorMessage) ||
            !unit_test::vk::CreateImage2D(vkContext, kRenderWidth, kRenderHeight, VK_FORMAT_R16G16_SFLOAT, inputUsage, motionImage, &errorMessage) ||
            !unit_test::vk::CreateImage2D(
                vkContext, kUpscaleWidth, kUpscaleHeight, VK_FORMAT_B10G11R11_UFLOAT_PACK32, outputUsage, outputImage, &errorMessage) ||
            !unit_test::vk::CreateImage2D(vkContext, kExposureWidth, kExposureHeight, VK_FORMAT_R32_SFLOAT, inputUsage, exposureImage, &errorMessage))
        {
            failureReason = errorMessage.empty() ? "failed to create NSS capture images" : errorMessage;
            break;
        }

        VkDeviceContext                                  deviceContext = MakeDeviceContext(vkContext);
        arm::capture::neuralssuperresolution::Controller controller(&deviceContext,
                                                                    FFX_API_NSS_CONTEXT_FLAG_QUANTIZED | FFX_API_NSS_CONTEXT_FLAG_DEPTH_INFINITE |
                                                                        FFX_API_NSS_CONTEXT_FLAG_EXPOSURE_TEXTURE |
                                                                        FFX_API_NSS_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION);
        OutputDirectory                                  output("nss_controller_capture");
        const std::filesystem::path                      outputBase       = output.File("dataset");
        const std::string                                outputBaseString = outputBase.string();

        ffxApiConfigureDescCapture configure{};
        configure.header.type            = FFX_API_CONFIGURE_DESC_TYPE_CAPTURE;
        configure.operation              = FFX_API_CAPTURE_BEGIN;
        configure.outputDirectory        = outputBaseString.c_str();
        configure.startDispatch          = 0;
        configure.frameCount             = kCaptureWindow;
        configure.motionVectorUnormBias  = 0.25f;
        configure.motionVectorUnormScale = 4.0f;
        if (controller.Configure(configure) != FFX_API_RETURN_OK)
        {
            failureReason = "NSS capture controller rejected begin configuration";
            break;
        }

        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        if (!unit_test::vk::BeginCommandBuffer(vkContext, commandBuffer, &errorMessage))
        {
            failureReason = errorMessage.empty() ? "failed to begin NSS capture command buffer" : errorMessage;
            break;
        }

        ffxApiDispatchDescNss dispatch{};
        dispatch.header.type   = FFX_API_DISPATCH_DESC_TYPE_NSS;
        dispatch.commandList   = commandBuffer;
        dispatch.color         = unit_test::vk::MakeApiImageResource(colorImage, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        dispatch.depth         = unit_test::vk::MakeApiImageResource(depthImage, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        dispatch.motionVectors = unit_test::vk::MakeApiImageResource(motionImage, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        dispatch.output        = unit_test::vk::MakeApiImageResource(outputImage, FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch.outputTm1     = unit_test::vk::MakeApiImageResource(outputImage, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        dispatch.exposureTexture =
            unit_test::vk::MakeApiImageResource(exposureImage, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        dispatch.renderSize             = {kRenderWidth, kRenderHeight};
        dispatch.upscaleSize            = {kUpscaleWidth, kUpscaleHeight};
        dispatch.jitterOffset           = {0.0f, 0.0f};
        dispatch.cameraNear             = 0.1f;
        dispatch.cameraFar              = -3.0f;
        dispatch.cameraFovAngleVertical = 1.2f;
        dispatch.exposure               = 1.0f;
        dispatch.motionVectorScale      = {static_cast<float>(kRenderWidth), static_cast<float>(kRenderHeight)};
        dispatch.frameTimeDelta         = std::numeric_limits<float>::quiet_NaN();
        dispatch.reset                  = true;

        FfxNssContext nssContext{};
        if (!controller.BeginDispatchCapture(dispatch, nssContext))
        {
            failureReason = "NSS controller failed BeginDispatchCapture";
            break;
        }
        if (!controller.ShouldResetHistoryForDispatch() || !controller.CompleteDispatchCapture(dispatch, nssContext))
        {
            failureReason = "NSS controller failed first-frame reset check or CompleteDispatchCapture";
            break;
        }

        if (!unit_test::vk::SubmitCommandBuffer(vkContext, commandBuffer, &errorMessage))
        {
            failureReason = errorMessage.empty() ? "failed to submit NSS capture command buffer" : errorMessage;
            break;
        }
        controller.Shutdown();

        const std::filesystem::path dataset = outputBase;
        if (!std::filesystem::is_regular_file(dataset / "0000.json"))
        {
            failureReason = "NSS capture did not produce metadata at the configured output directory";
            break;
        }

        if (!std::filesystem::is_regular_file(dataset / "0000.json") || !std::filesystem::is_regular_file(dataset / "x2/color/0000/0000.exr") ||
            !std::filesystem::is_regular_file(dataset / "x2/depth/0000/0000.exr") || !std::filesystem::is_regular_file(dataset / "x2/motion/0000/0000.exr") ||
            !std::filesystem::is_regular_file(dataset / "results/NGSDK/0000.exr") || !std::filesystem::is_regular_file(dataset / "exposure/0000/0000.exr"))
        {
            failureReason = "NSS capture dataset is missing one or more expected files under " + dataset.string() + ":";
            if (std::filesystem::exists(dataset))
            {
                for (const auto& entry : std::filesystem::recursive_directory_iterator(dataset))
                    failureReason += " " + entry.path().lexically_relative(dataset).string();
            }
            break;
        }
        {
            std::ifstream     metadata(dataset / "0000.json");
            const std::string contents((std::istreambuf_iterator<char>(metadata)), std::istreambuf_iterator<char>());
            if (contents.find("\"MotionVectorUnormBias\": 0.25") == std::string::npos || contents.find("\"MotionVectorUnormScale\": 4") == std::string::npos ||
                contents.find("\"DepthInfinite\": true") == std::string::npos || contents.find("\"directory\": \"exposure/0000\"") == std::string::npos ||
                contents.find("\"kind\": \"exposure\"") == std::string::npos || contents.find("\"width\": 2, \"height\": 3") == std::string::npos ||
                contents.find("\"FrameTimeDelta\": nan") == std::string::npos || contents.find("\"FarPlane\": -3") == std::string::npos ||
                contents.find("\"CameraFarPlane\": -3") == std::string::npos || contents.find("\"MotionVectorsJitterCancellation\": true") == std::string::npos)
            {
                failureReason = "NSS metadata did not preserve capture stream metadata";
                break;
            }
        }
        success = true;
    } while (false);

    unit_test::vk::DestroyImage(vkContext, exposureImage);
    unit_test::vk::DestroyImage(vkContext, outputImage);
    unit_test::vk::DestroyImage(vkContext, motionImage);
    unit_test::vk::DestroyImage(vkContext, depthImage);
    unit_test::vk::DestroyImage(vkContext, colorImage);
    unit_test::vk::DestroyVulkanContext(vkContext);
    if (!success)
        UNIT_TEST_FAIL(failureReason);
    return true;
}

UNIT_TEST_CASE(capture, nss_capture_non_two_x_ratio)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!InitializeCaptureContext(vkContext))
        UNIT_TEST_FAIL("Vulkan unavailable for NSS ratio capture test.");

    constexpr uint32_t      kRenderWidth   = 32;
    constexpr uint32_t      kRenderHeight  = 32;
    constexpr uint32_t      kUpscaleWidth  = 48;
    constexpr uint32_t      kUpscaleHeight = 48;
    constexpr uint32_t      kCaptureWindow = 500;
    const VkImageUsageFlags inputUsage =
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const VkImageUsageFlags outputUsage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

    unit_test::vk::Image colorImage{};
    unit_test::vk::Image depthImage{};
    unit_test::vk::Image motionImage{};
    unit_test::vk::Image outputImage{};
    bool                 success = false;
    do
    {
        if (!unit_test::vk::CreateImage2D(vkContext, kRenderWidth, kRenderHeight, VK_FORMAT_B10G11R11_UFLOAT_PACK32, inputUsage, colorImage, &errorMessage) ||
            !unit_test::vk::CreateImage2D(vkContext, kRenderWidth, kRenderHeight, VK_FORMAT_R32_SFLOAT, inputUsage, depthImage, &errorMessage) ||
            !unit_test::vk::CreateImage2D(vkContext, kRenderWidth, kRenderHeight, VK_FORMAT_R16G16_SFLOAT, inputUsage, motionImage, &errorMessage) ||
            !unit_test::vk::CreateImage2D(vkContext, kUpscaleWidth, kUpscaleHeight, VK_FORMAT_B10G11R11_UFLOAT_PACK32, outputUsage, outputImage, &errorMessage))
            break;

        VkDeviceContext                                  deviceContext = MakeDeviceContext(vkContext);
        arm::capture::neuralssuperresolution::Controller controller(&deviceContext, FFX_API_NSS_CONTEXT_FLAG_QUANTIZED);
        OutputDirectory                                  output("nss_controller_capture_non_two_x");
        const std::string                                outputPath = output.File("dataset").string();

        ffxApiConfigureDescCapture configure{};
        configure.header.type            = FFX_API_CONFIGURE_DESC_TYPE_CAPTURE;
        configure.operation              = FFX_API_CAPTURE_BEGIN;
        configure.outputDirectory        = outputPath.c_str();
        configure.startDispatch          = 0;
        configure.frameCount             = kCaptureWindow;
        configure.motionVectorUnormBias  = 0.25f;
        configure.motionVectorUnormScale = 4.0f;
        if (controller.Configure(configure) != FFX_API_RETURN_OK)
            break;

        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        if (!unit_test::vk::BeginCommandBuffer(vkContext, commandBuffer, &errorMessage))
            break;

        ffxApiDispatchDescNss dispatch{};
        dispatch.header.type   = FFX_API_DISPATCH_DESC_TYPE_NSS;
        dispatch.commandList   = commandBuffer;
        dispatch.color         = unit_test::vk::MakeApiImageResource(colorImage, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        dispatch.depth         = unit_test::vk::MakeApiImageResource(depthImage, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        dispatch.motionVectors = unit_test::vk::MakeApiImageResource(motionImage, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        dispatch.output        = unit_test::vk::MakeApiImageResource(outputImage, FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch.outputTm1     = unit_test::vk::MakeApiImageResource(outputImage, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        dispatch.renderSize    = {kRenderWidth, kRenderHeight};
        dispatch.upscaleSize   = {kUpscaleWidth, kUpscaleHeight};
        dispatch.cameraNear    = 0.1f;
        dispatch.cameraFar     = 1000.0f;
        dispatch.cameraFovAngleVertical = 1.2f;
        dispatch.exposure               = 1.0f;
        dispatch.motionVectorScale      = {static_cast<float>(kRenderWidth), static_cast<float>(kRenderHeight)};
        dispatch.frameTimeDelta         = 16.6f;
        dispatch.reset                  = true;

        FfxNssContext nssContext{};
        if (!controller.BeginDispatchCapture(dispatch, nssContext) || !controller.CompleteDispatchCapture(dispatch, nssContext) ||
            !unit_test::vk::SubmitCommandBuffer(vkContext, commandBuffer, &errorMessage))
            break;
        controller.Shutdown();

        const std::filesystem::path dataset = output.File("dataset");
        std::ifstream               metadata(dataset / "0000.json");
        const std::string           contents((std::istreambuf_iterator<char>(metadata)), std::istreambuf_iterator<char>());
        success = std::filesystem::is_regular_file(dataset / "x1_5/color/0000/0000.exr") &&
                  std::filesystem::is_regular_file(dataset / "x1_5/depth/0000/0000.exr") &&
                  std::filesystem::is_regular_file(dataset / "x1_5/motion/0000/0000.exr") && contents.find("\"x1_5_index\": 0") != std::string::npos &&
                  contents.find("\"directory\": \"x1_5/color/0000\"") != std::string::npos;
    } while (false);

    unit_test::vk::DestroyImage(vkContext, outputImage);
    unit_test::vk::DestroyImage(vkContext, motionImage);
    unit_test::vk::DestroyImage(vkContext, depthImage);
    unit_test::vk::DestroyImage(vkContext, colorImage);
    unit_test::vk::DestroyVulkanContext(vkContext);
    if (!success)
        UNIT_TEST_FAIL(errorMessage.empty() ? "NSS non-2x capture did not produce the expected ratio layout." : errorMessage);
    return true;
}

// Provider integration: the first NFRU capture frame must write its input dataset.
UNIT_TEST_CASE(capture, nfru_capture_frame)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!InitializeCaptureContext(vkContext))
        UNIT_TEST_FAIL("Vulkan unavailable for NFRU capture smoke test.");

    constexpr uint32_t kDisplayWidth        = 48;
    constexpr uint32_t kDisplayHeight       = 48;
    constexpr uint32_t kRenderWidth         = 32;
    constexpr uint32_t kRenderHeight        = 32;
    constexpr uint32_t kResizedRenderWidth  = 24;
    constexpr uint32_t kResizedRenderHeight = 24;
    constexpr uint32_t kOpticalFlowId       = FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_VECTOR;

    unit_test::vk::Image presentColor{};
    unit_test::vk::Image output{};
    unit_test::vk::Image depth{};
    unit_test::vk::Image motionVectors{};
    unit_test::vk::Image displayMotionVectors{};
    unit_test::vk::Image resizedDepth{};
    unit_test::vk::Image opticalFlow{};
    bool                 success       = false;
    std::string          failureReason = "NFRU capture smoke test failed.";

    do
    {
        const VkImageUsageFlags displayUsage =
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        const VkImageUsageFlags renderUsage =
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        if (!unit_test::vk::CreateImage2D(vkContext, kDisplayWidth, kDisplayHeight, VK_FORMAT_R8G8B8A8_UNORM, displayUsage, presentColor, &errorMessage) ||
            !unit_test::vk::CreateImage2D(vkContext, kDisplayWidth, kDisplayHeight, VK_FORMAT_R8G8B8A8_UNORM, displayUsage, output, &errorMessage) ||
            !unit_test::vk::CreateImage2D(vkContext, kRenderWidth, kRenderHeight, VK_FORMAT_R32_SFLOAT, renderUsage, depth, &errorMessage) ||
            !unit_test::vk::CreateImage2D(vkContext, kRenderWidth, kRenderHeight, VK_FORMAT_R16G16_SFLOAT, renderUsage, motionVectors, &errorMessage) ||
            !unit_test::vk::CreateImage2D(
                vkContext, kDisplayWidth, kDisplayHeight, VK_FORMAT_R16G16_SFLOAT, displayUsage, displayMotionVectors, &errorMessage) ||
            !unit_test::vk::CreateImage2D(
                vkContext, kResizedRenderWidth, kResizedRenderHeight, VK_FORMAT_R32_SFLOAT, renderUsage, resizedDepth, &errorMessage) ||
            !unit_test::vk::CreateImage2D(vkContext, kRenderWidth, kRenderHeight, VK_FORMAT_R16G16_SFLOAT, renderUsage, opticalFlow, &errorMessage))
        {
            failureReason = errorMessage.empty() ? "failed to create NFRU capture images" : errorMessage;
            break;
        }

        VkDeviceContext deviceContext = MakeDeviceContext(vkContext);
        FfxInterface    backendInterface{};
        backendInterface.fpGetResource = GetNfruCaptureResource;
        FfxResourceInternal sharedResources[kOpticalFlowId + 1]{};
        sharedResources[kOpticalFlowId].internalIndex = 1;
        gNfruCaptureOpticalFlow = unit_test::vk::MakeFfxImageResource(opticalFlow, FFX_RESOURCE_USAGE_READ_ONLY, FFX_RESOURCE_STATE_PIXEL_READ, "opticalFlow");

        arm::capture::framegenerations::CreateDesc createDesc{};
        createDesc.backendDevice = &deviceContext;
        createDesc.contextFlags =
            FFX_API_FG_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION | FFX_API_FG_CONTEXT_FLAG_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS;
        arm::capture::framegenerations::Context controllerContext{};
        controllerContext.backendInterface = &backendInterface;
        controllerContext.sharedResources  = sharedResources;
        arm::capture::framegenerations::Controller controller(createDesc, controllerContext);

        OutputDirectory             outputDirectory("nfru_controller_capture");
        const std::filesystem::path outputBase = outputDirectory.File("dataset");
        const std::string           outputPath = outputBase.string();
        ffxApiConfigureDescCapture  configure{};
        configure.header.type            = FFX_API_CONFIGURE_DESC_TYPE_CAPTURE;
        configure.operation              = FFX_API_CAPTURE_BEGIN;
        configure.outputDirectory        = outputPath.c_str();
        configure.startDispatch          = 0;
        configure.frameCount             = 2;
        configure.motionVectorUnormBias  = 0.5f;
        configure.motionVectorUnormScale = 2.0f;
        if (controller.Configure(configure) != FFX_API_RETURN_OK)
        {
            failureReason = "NFRU capture controller rejected begin configuration";
            break;
        }

        controller.OnPrepare();
        if (!controller.IsPreparePending() || !controller.ShouldResetHistoryForDispatch())
        {
            failureReason = "NFRU controller did not arm the first prepare dispatch";
            break;
        }

        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        if (!unit_test::vk::BeginCommandBuffer(vkContext, commandBuffer, &errorMessage))
        {
            failureReason = errorMessage.empty() ? "failed to begin NFRU capture command buffer" : errorMessage;
            break;
        }

        ffxApiDispatchDescFrameGeneration dispatch{};
        dispatch.header.type        = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
        dispatch.commandList        = commandBuffer;
        dispatch.presentColor       = unit_test::vk::MakeApiImageResource(presentColor, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_READ);
        dispatch.outputs[0]         = unit_test::vk::MakeApiImageResource(output, FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch.numGeneratedFrames = 1;
        dispatch.reset              = true;
        dispatch.frameID            = 1;

        ffxApiDispatchDescFrameGenerationPrepare prepare{};
        prepare.header.type            = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
        prepare.commandList            = commandBuffer;
        prepare.jitterOffset           = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()};
        prepare.motionVectorScale      = {-4.0f, -5.0f};
        prepare.frameTimeDelta         = std::numeric_limits<float>::quiet_NaN();
        prepare.cameraNear             = -2.0f;
        prepare.cameraFar              = -3.0f;
        prepare.cameraFovAngleVertical = -1.0f;
        prepare.depth                  = unit_test::vk::MakeApiImageResource(depth, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_READ);
        prepare.motionVectors = unit_test::vk::MakeApiImageResource(displayMotionVectors, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_READ);
        unit_test::vk::SetIdentityMatrix(prepare.viewProjection);
        prepare.viewProjection[0] = std::numeric_limits<float>::quiet_NaN();

        if (!controller.BeginDispatchCapture(dispatch, prepare) ||
            !controller.CompleteDispatchCapture(unit_test::vk::MakeFfxImageResource(output, FFX_RESOURCE_USAGE_UAV, FFX_RESOURCE_STATE_GENERIC_UAV, "output")))
        {
            failureReason = "NFRU controller failed BeginDispatchCapture or CompleteDispatchCapture";
            break;
        }

        if (!unit_test::vk::SubmitCommandBuffer(vkContext, commandBuffer, &errorMessage))
        {
            failureReason = errorMessage.empty() ? "failed to submit NFRU capture command buffer" : errorMessage;
            break;
        }

        controller.OnPrepare();
        if (!controller.IsPreparePending())
        {
            failureReason = "NFRU controller did not arm the second prepare dispatch";
            break;
        }

        VkCommandBuffer secondCommandBuffer = VK_NULL_HANDLE;
        if (!unit_test::vk::BeginCommandBuffer(vkContext, secondCommandBuffer, &errorMessage))
        {
            failureReason = errorMessage.empty() ? "failed to begin second NFRU capture command buffer" : errorMessage;
            break;
        }
        dispatch.commandList = secondCommandBuffer;
        dispatch.reset       = false;
        dispatch.frameID     = 2;
        prepare.commandList  = secondCommandBuffer;
        prepare.depth        = unit_test::vk::MakeApiImageResource(resizedDepth, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_READ);
        if (!controller.BeginDispatchCapture(dispatch, prepare) ||
            !controller.CompleteDispatchCapture(unit_test::vk::MakeFfxImageResource(output, FFX_RESOURCE_USAGE_UAV, FFX_RESOURCE_STATE_GENERIC_UAV, "output")))
        {
            failureReason = "NFRU controller failed the generated-frame capture";
            break;
        }
        if (!unit_test::vk::SubmitCommandBuffer(vkContext, secondCommandBuffer, &errorMessage))
        {
            failureReason = errorMessage.empty() ? "failed to submit second NFRU capture command buffer" : errorMessage;
            break;
        }
        controller.Shutdown();

        const std::filesystem::path dataset = outputBase;
        if (!std::filesystem::is_regular_file(dataset / "0000.json"))
        {
            failureReason = "NFRU capture did not produce metadata at the configured output directory";
            break;
        }
        {
            std::ifstream     metadata(dataset / "0000.json");
            const std::string contents((std::istreambuf_iterator<char>(metadata)), std::istreambuf_iterator<char>());
            const std::string opticalFlowDescriptor = "\"opticalFlow\": { \"directory\": \"optical_flow_p2/0000\", \"kind\": \"opticalFlow\", \"format\": " +
                                                      std::to_string(static_cast<uint32_t>(gNfruCaptureOpticalFlow.description.format)) +
                                                      ", \"width\": " + std::to_string(kRenderWidth) + ", \"height\": " + std::to_string(kRenderHeight) + " }";
            if (contents.find(opticalFlowDescriptor) == std::string::npos)
            {
                failureReason = "NFRU metadata is missing the optical-flow stream descriptor";
                break;
            }
            if (contents.find("\"directory\": \"x1_5/depth/0000\"") == std::string::npos ||
                contents.find("\"directory\": \"x1_5/motion_m2/0000\"") == std::string::npos || contents.find("\"x1_5_index\": 0") == std::string::npos)
            {
                failureReason = "NFRU metadata is missing the non-2x ratio layout";
                break;
            }
            if (contents.find("\"motion\": { \"directory\": \"x1_5/motion_m2/0000\", \"kind\": \"motion\", \"format\": " +
                              std::to_string(static_cast<uint32_t>(FFX_SURFACE_FORMAT_R16G16_FLOAT)) + ", \"width\": 48, \"height\": 48 }") ==
                std::string::npos)
            {
                failureReason = "NFRU metadata did not preserve the display-resolution motion extent";
                break;
            }
            if (contents.find("\"Frame\": 0") == std::string::npos || contents.find("\"Frame\": 1") == std::string::npos ||
                contents.find("\"Frame\": 2") == std::string::npos || contents.find("\"Frame\": 3") != std::string::npos)
            {
                failureReason = "NFRU metadata frame numbering does not match captured input and generated files";
                break;
            }
            if (contents.find("\"FrameTimeDelta\": nan") == std::string::npos || contents.find("\"CameraNearPlane\": -2") == std::string::npos ||
                contents.find("\"CameraFarPlane\": -3") == std::string::npos || contents.find("\"MotionVectorScale\": [ -4, -5 ]") == std::string::npos ||
                contents.find("\"ViewSpaceToMetersFactor\": 0") == std::string::npos || contents.find("\"ViewProjection\": [ nan") == std::string::npos ||
                contents.find("\"MotionVectorsJitterCancellation\": true") == std::string::npos ||
                contents.find("\"DisplayResolutionMotionVectors\": true") == std::string::npos)
            {
                failureReason = "NFRU metadata did not preserve atypical runtime values";
                break;
            }
        }

        if (!std::filesystem::is_regular_file(dataset / "0000.json") || !std::filesystem::is_regular_file(dataset / "ground_truth/0000/0000.exr") ||
            !std::filesystem::is_regular_file(dataset / "x1_5/depth/0000/0000.exr") ||
            !std::filesystem::is_regular_file(dataset / "x1_5/motion_m2/0000/0000.exr") ||
            !std::filesystem::is_regular_file(dataset / "x2/depth/0000/0002.exr") ||
            !std::filesystem::is_regular_file(dataset / "x2/motion_m2/0000/0002.exr") ||
            !std::filesystem::is_regular_file(dataset / "results/NGSDK/0001.exr") ||
            !std::filesystem::is_regular_file(dataset / "optical_flow_p2/0000/0001.exr"))
        {
            failureReason = "NFRU capture dataset is missing one or more expected non-2x input or generated-frame files";
            break;
        }
        success = true;
    } while (false);

    gNfruCaptureOpticalFlow = {};
    unit_test::vk::DestroyImage(vkContext, opticalFlow);
    unit_test::vk::DestroyImage(vkContext, resizedDepth);
    unit_test::vk::DestroyImage(vkContext, motionVectors);
    unit_test::vk::DestroyImage(vkContext, displayMotionVectors);
    unit_test::vk::DestroyImage(vkContext, depth);
    unit_test::vk::DestroyImage(vkContext, output);
    unit_test::vk::DestroyImage(vkContext, presentColor);
    unit_test::vk::DestroyVulkanContext(vkContext);
    if (!success)
        UNIT_TEST_FAIL(failureReason);
    return true;
}
