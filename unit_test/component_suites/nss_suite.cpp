/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "../framework/test_framework.h"
#include "../framework/vk_test_utils.h"

#include <FidelityFX/host/backends/vk/vk_wrapper.h>
#include <FidelityFX/host/ffx_nss.h>
#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_framegeneration.h>
#include <ffx_api/ffx_nss.h>
#include "ffx_nss_jitter_analysis.h"
#include "ffx_nss_private.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace
{
    static_assert(sizeof(void*) == 8);
    static_assert(FFX_API_NSS_CONTEXT_FLAG_LOW_DYNAMIC_RANGE == (1u << 1));
    static_assert(FFX_API_NSS_CONTEXT_FLAG_DISABLE_16BIT == (1u << 5));
    static_assert(FFX_API_NSS_CONTEXT_FLAG_PRE_PROCESS_COMPUTE == (1u << 7));
    static_assert(FFX_API_NSS_CONTEXT_FLAG_POST_PROCESS_COMPUTE == (1u << 8));

    static_assert(FFX_API_FG_CONTEXT_FLAG_WARP_COMPUTE == (1u << 5));
    static_assert(FFX_API_FG_CONTEXT_FLAG_FILL_HOLES_COMPUTE == (1u << 6));
    static_assert(FFX_API_FG_CONTEXT_FLAG_PRE_PROCESS_COMPUTE == (1u << 7));
    static_assert(FFX_API_FG_CONTEXT_FLAG_POSTPROCESS_COMPUTE == (1u << 8));
    static_assert(FFX_API_FG_CONTEXT_FLAG_DEBUG_VIEW_COMPUTE == (1u << 9));
    static_assert(FFX_API_FG_CONTEXT_FLAG_MV_HINTS_COMPUTE == (1u << 11));
    static_assert(FFX_API_FG_CONTEXT_FLAG_ALL_STAGES_COMPUTE == 0x3e0u);
    static_assert(sizeof(ffxApiCreateContextDescFrameGeneration) == 120);
    static_assert(alignof(ffxApiCreateContextDescFrameGeneration) == 8);
    static_assert(offsetof(ffxApiCreateContextDescFrameGeneration, mvDepthSize) == 112);
    static_assert(sizeof(ffxApiDispatchDescFrameGeneration) == 280);
    static_assert(alignof(ffxApiDispatchDescFrameGeneration) == 8);
    static_assert(offsetof(ffxApiDispatchDescFrameGeneration, frameID) == 272);
    static_assert(sizeof(ffxApiCreateContextDescNss) == 56);
    static_assert(alignof(ffxApiCreateContextDescNss) == 8);
    static_assert(offsetof(ffxApiCreateContextDescNss, qualityMode) == 48);
    static_assert(sizeof(ffxApiDispatchDescNss) == 432);
    static_assert(alignof(ffxApiDispatchDescNss) == 8);
    static_assert(offsetof(ffxApiDispatchDescNss, exposureTexture) == 304);
    static_assert(offsetof(ffxApiDispatchDescNss, motionVectorScale) == 352);
    static_assert(offsetof(ffxApiDispatchDescNss, debugViewMode) == 424);
    static_assert(sizeof(VkFunctionTable) == 904);
    static_assert(alignof(VkFunctionTable) == 8);
    static_assert(offsetof(VkFunctionTable, vkGetPhysicalDeviceFormatProperties) == 896);
    static_assert(alignof(FfxDeviceCapabilities) == 4);
    static_assert(offsetof(FfxDeviceCapabilities, fragmentStoresAndAtomicsSupported) == 26);
    static_assert(offsetof(FfxDeviceCapabilities, r11g11b10StorageImageSupported) == 27);
    static_assert(offsetof(FfxDeviceCapabilities, supportedOutputGridSizes) == 28);
    static_assert(sizeof(FfxDeviceCapabilities) == 52);
    static_assert(sizeof(FfxNssDispatchDescription) == 920);
    static_assert(alignof(FfxNssDispatchDescription) == 8);
    static_assert(offsetof(FfxNssDispatchDescription, exposure) == 644);
    static_assert(offsetof(FfxNssDispatchDescription, exposureTexture) == 648);
    static_assert(offsetof(FfxNssDispatchDescription, motionVectorScale) == 768);
    static_assert(offsetof(FfxNssDispatchDescription, frameTimeDelta) == 776);
    static_assert(offsetof(FfxNssDispatchDescription, flags) == 784);
    static_assert(offsetof(FfxNssDispatchDescription, debugViews) == 792);
    static_assert(offsetof(FfxNssDispatchDescription, debugViewMode) == 912);

    ffxApiHeader MakeApiHeader(const ffxStructType_t type)
    {
        ffxApiHeader header{};
        header.type = type;
        return header;
    }

    std::vector<std::string> diagnosticMessages;

    void ApiMessageSink(uint32_t, const char*)
    {
    }

    void CaptureApiMessage(uint32_t, const char* message)
    {
        diagnosticMessages.emplace_back(message);
    }

    bool SupportsNssComputePostProcess(const unit_test::vk::VulkanContext& vkContext)
    {
        VkPhysicalDeviceFeatures2 physicalDeviceFeatures{};
        physicalDeviceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        vkGetPhysicalDeviceFeatures2(vkContext.physicalDevice, &physicalDeviceFeatures);

        VkFormatProperties r11g11b10Properties{};
        vkGetPhysicalDeviceFormatProperties(vkContext.physicalDevice, VK_FORMAT_B10G11R11_UFLOAT_PACK32, &r11g11b10Properties);
        return physicalDeviceFeatures.features.shaderStorageImageExtendedFormats &&
               (r11g11b10Properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
    }

    ffxReturnCode_t CreateNssContext(unit_test::vk::VulkanContext&    vkContext,
                                     const uint32_t                   flags,
                                     const FfxApiNssShaderQualityMode qualityMode,
                                     const uint32_t                   renderWidth,
                                     const uint32_t                   renderHeight,
                                     const uint32_t                   upscaleWidth,
                                     const uint32_t                   upscaleHeight,
                                     ffxContext&                      outContext,
                                     ffxApiMessage                    message = ApiMessageSink)
    {
        VkDeviceContext deviceContext{};
        deviceContext.vkDevice              = vkContext.device;
        deviceContext.vkPhysicalDevice      = vkContext.physicalDevice;
        deviceContext.vkInstance            = vkContext.instance;
        deviceContext.vkDeviceProcAddr      = vkContext.getDeviceProcAddr;
        deviceContext.vkGetInstanceProcAddr = vkContext.getInstanceProcAddr;
        if (!InitVulkanWrapper(deviceContext))
        {
            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        }

        auto backendDesc = unit_test::vk::BuildBackendDesc(vkContext);

        ffxApiCreateContextDescNss createDesc{};
        createDesc.header.type           = FFX_API_CREATE_CONTEXT_DESC_TYPE_NSS;
        createDesc.header.pNext          = reinterpret_cast<ffxApiHeader*>(&backendDesc);
        createDesc.flags                 = flags;
        createDesc.maxRenderSize.width   = renderWidth;
        createDesc.maxRenderSize.height  = renderHeight;
        createDesc.maxUpscaleSize.width  = upscaleWidth;
        createDesc.maxUpscaleSize.height = upscaleHeight;
        createDesc.fpMessage             = message;
        createDesc.qualityMode           = qualityMode;

        return ffxCreateContext(&outContext, &createDesc.header, nullptr);
    }

    void DestroyContextIfNeeded(ffxContext& context)
    {
        if (context)
        {
            ffxDestroyContext(&context, nullptr);
        }
    }

    std::array<float, 2> ProjectJitteredPoint(const std::array<float, 3>& point, FfxFloatCoords2D jitter, FfxDimensions2D extent)
    {
        // Independently apply perspective projection jitter, then viewport conversion.
        const float clipW = -point[2];
        const float clipX = point[0] + (2.0f * jitter.x / extent.width) * point[2];
        const float clipY = point[1] + (2.0f * jitter.y / extent.height) * point[2];
        return {(clipX / clipW + 1.0f) * extent.width * 0.5f, (clipY / clipW + 1.0f) * extent.height * 0.5f};
    }

    struct CancellationScene
    {
        unit_test::vk::VulkanContext&       vk;
        ffxContext                          context = nullptr;
        unit_test::vk::Image                color{}, depth{}, motion{};
        std::array<unit_test::vk::Image, 2> output{};
        unit_test::vk::Image                debugView{};
        std::string                         error;
        bool                                useFragmentPath = false;

        explicit CancellationScene(unit_test::vk::VulkanContext& device)
            : vk(device)
        {
        }

        ~CancellationScene()
        {
            DestroyContextIfNeeded(context);
            for (auto& image : output)
                unit_test::vk::DestroyImage(vk, image);
            unit_test::vk::DestroyImage(vk, debugView);
            unit_test::vk::DestroyImage(vk, motion);
            unit_test::vk::DestroyImage(vk, depth);
            unit_test::vk::DestroyImage(vk, color);
        }

        bool Initialize(uint32_t flags, FfxApiNssShaderQualityMode quality, FfxDimensions2D render, FfxDimensions2D upscale)
        {
            constexpr uint32_t computeFlags = FFX_API_NSS_CONTEXT_FLAG_PRE_PROCESS_COMPUTE | FFX_API_NSS_CONTEXT_FLAG_POST_PROCESS_COMPUTE;
            useFragmentPath                 = (flags & computeFlags) == 0;
            const auto result               = CreateNssContext(vk, flags, quality, render.width, render.height, upscale.width, upscale.height, context);
            if (result != FFX_API_RETURN_OK)
            {
                error = "NSS cancellation context creation failed: " + std::to_string(result);
                return false;
            }

            constexpr VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            if (!unit_test::vk::CreateImage2D(vk, render.width, render.height, VK_FORMAT_B10G11R11_UFLOAT_PACK32, usage, color, &error) ||
                !unit_test::vk::CreateImage2D(vk, render.width, render.height, VK_FORMAT_R32_SFLOAT, usage, depth, &error) ||
                !unit_test::vk::CreateImage2D(vk, render.width, render.height, VK_FORMAT_R16G16_SFLOAT, usage, motion, &error))
                return false;
            for (auto& image : output)
            {
                if (!unit_test::vk::CreateImage2D(vk, upscale.width, upscale.height, VK_FORMAT_B10G11R11_UFLOAT_PACK32, usage, image, &error) ||
                    !unit_test::vk::ClearImage(vk, image, {0, 0, 0, 0}, &error))
                    return false;
            }
            if (!unit_test::vk::CreateImage2D(vk, upscale.width, upscale.height, VK_FORMAT_B10G11R11_UFLOAT_PACK32, usage, debugView, &error) ||
                !unit_test::vk::ClearImage(vk, debugView, {0, 0, 0, 0}, &error))
                return false;
            // The color edge and matching depth discontinuity exercise temporal/disocclusion paths.
            return unit_test::vk::RenderTriangle(vk, color, &error) && unit_test::vk::RenderTriangle(vk, depth, &error);
        }

        bool Dispatch(
            uint32_t frame, FfxFloatCoords2D jitter, FfxFloatCoords2D scale, const std::array<float, 2>& rawMotion, bool reset, std::vector<uint8_t>& pixels)
        {
            if (!unit_test::vk::ClearImage(vk, motion, {rawMotion[0], rawMotion[1], 0, 0}, &error))
                return false;
            VkCommandBuffer commands = VK_NULL_HANDLE;
            if (!unit_test::vk::BeginCommandBuffer(vk, commands, &error))
                return false;

            auto&                 current     = output[frame % 2];
            auto&                 previous    = output[(frame + 1) % 2];
            const uint32_t        inputState  = useFragmentPath ? FFX_API_RESOURCE_STATE_PIXEL_READ : FFX_API_RESOURCE_STATE_COMPUTE_READ;
            const uint32_t        outputUsage = useFragmentPath ? FFX_API_RESOURCE_USAGE_RENDERTARGET : FFX_API_RESOURCE_USAGE_UAV;
            const uint32_t        outputState = useFragmentPath ? FFX_API_RESOURCE_STATE_RENDER_TARGET : FFX_API_RESOURCE_STATE_UNORDERED_ACCESS;
            ffxApiDispatchDescNss desc{};
            desc.header.type            = FFX_API_DISPATCH_DESC_TYPE_NSS;
            desc.commandList            = commands;
            desc.color                  = unit_test::vk::MakeApiImageResource(color, FFX_API_RESOURCE_USAGE_READ_ONLY, inputState);
            desc.depth                  = unit_test::vk::MakeApiImageResource(depth, FFX_API_RESOURCE_USAGE_READ_ONLY, inputState);
            desc.motionVectors          = unit_test::vk::MakeApiImageResource(motion, FFX_API_RESOURCE_USAGE_READ_ONLY, inputState);
            desc.output                 = unit_test::vk::MakeApiImageResource(current, outputUsage, outputState);
            desc.outputTm1              = unit_test::vk::MakeApiImageResource(previous, FFX_API_RESOURCE_USAGE_READ_ONLY, inputState);
            desc.renderSize             = {color.createInfo.extent.width, color.createInfo.extent.height};
            desc.upscaleSize            = {current.createInfo.extent.width, current.createInfo.extent.height};
            desc.jitterOffset           = {jitter.x, jitter.y};
            desc.motionVectorScale      = {scale.x, scale.y};
            desc.cameraNear             = 0.1f;
            desc.cameraFar              = 1000.0f;
            desc.cameraFovAngleVertical = 1.2f;
            desc.exposure               = 1.0f;
            desc.frameTimeDelta         = 16.6f;
            desc.reset                  = reset;
            // Compare the quantized motion feature, not implementation-dependent neural output pixels.
            desc.flags           = FFX_API_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW;
            desc.debugViews      = unit_test::vk::MakeApiImageResource(debugView, outputUsage, outputState);
            desc.debugViewMode   = NSS_DEBUG_VIEW_MODE_TENSOR_MOTION_DETECTOR;
            const auto result    = ffxDispatch(&context, &desc.header);
            const bool submitted = unit_test::vk::SubmitCommandBuffer(vk, commands, &error);
            if (result != FFX_API_RETURN_OK)
            {
                error = "NSS cancellation dispatch failed: " + std::to_string(result);
                return false;
            }
            return submitted && unit_test::vk::ReadbackRgba8Image(vk, debugView, pixels, &error);
        }
    };

    float DecodeUnsignedFloat(uint32_t bits, uint32_t mantissaBits)
    {
        const uint32_t exponent = bits >> mantissaBits;
        const uint32_t mantissa = bits & ((1u << mantissaBits) - 1u);
        if (exponent == 31)
            return std::numeric_limits<float>::quiet_NaN();
        return std::ldexp(static_cast<float>(mantissa + (exponent == 0 ? 0u : 1u << mantissaBits)),
                          (exponent == 0 ? -14 : static_cast<int>(exponent) - 15) - static_cast<int>(mantissaBits));
    }

    bool CompareMotionDetectorPixels(const std::vector<uint8_t>& reference, const std::vector<uint8_t>& actual, bool& equivalent)
    {
        UNIT_TEST_CHECK(!reference.empty() && reference.size() == actual.size() && reference.size() % 4 == 0);
        equivalent = true;
        for (size_t i = 0; i < reference.size(); i += 4)
        {
            uint32_t expectedBits = 0, actualBits = 0;
            std::memcpy(&expectedBits, reference.data() + i, 4);
            std::memcpy(&actualBits, actual.data() + i, 4);
            for (uint32_t channel = 0; channel < 3; ++channel)
            {
                const uint32_t mantissa        = channel == 2 ? 5 : 6;
                const uint32_t shift           = channel * 11;
                const uint32_t mask            = (1u << (mantissa + 5)) - 1u;
                const uint32_t expectedChannel = (expectedBits >> shift) & mask;
                const uint32_t actualChannel   = (actualBits >> shift) & mask;
                const float    expected        = DecodeUnsignedFloat(expectedChannel, mantissa);
                const float    value           = DecodeUnsignedFloat(actualChannel, mantissa);
                UNIT_TEST_CHECK(std::isfinite(expected) && std::isfinite(value));
                const float absoluteDifference = std::fabs(expected - value);
                const int   exponent           = std::max(1, static_cast<int>(expectedChannel >> mantissa));
                const float tolerance          = std::ldexp(1.0f, exponent - 15 - static_cast<int>(mantissa));
                equivalent &= absoluteDifference <= tolerance;
            }
        }
        return true;
    }

    float Halton(uint32_t index, uint32_t base)
    {
        float result   = 0.0f;
        float fraction = 1.0f / static_cast<float>(base);
        while (index != 0)
        {
            result += static_cast<float>(index % base) * fraction;
            index /= base;
            fraction /= static_cast<float>(base);
        }
        return result;
    }

    std::vector<NssJitterSample> ValidSequence(uint32_t count)
    {
        std::vector<NssJitterSample> samples;
        samples.reserve(count);
        for (uint32_t index = 1; index <= count; ++index)
        {
            samples.push_back({Halton(index, 2) - 0.5f, Halton(index, 3) - 0.5f});
        }
        return samples;
    }

    constexpr std::array<NssJitterSample, 32> kNearZeroJitterCycle = {
        {{-0.0000034722f, -0.0000007716f}, {0.0000040509f, -0.0000054012f},  {-0.0000005787f, 0.0000064300f},
         {0.0000017361f, 0.0000018004f},   {-0.0000028935f, -0.0000028292f}, {0.0000028935f, 0.0000048868f},
         {-0.0000017361f, 0.0000002572f},  {0.0000005787f, -0.0000043724f},  {-0.0000040509f, 0.0000033436f},
         {0.0000043403f, -0.0000012860f},  {-0.0000002894f, -0.0000059156f}, {0.0000020255f, 0.0000059156f},
         {-0.0000026042f, 0.0000012860f},  {0.0000031829f, -0.0000033436f},  {-0.0000014468f, 0.0000043724f},
         {0.0000008681f, -0.0000002572f},  {-0.0000037616f, -0.0000048868f}, {0.0000037616f, 0.0000028292f},
         {-0.0000008681f, -0.0000018004f}, {0.0000014468f, -0.0000064300f},  {-0.0000031829f, 0.0000067730f},
         {0.0000026042f, 0.0000021433f},   {-0.0000020255f, -0.0000024863f}, {0.0000002894f, 0.0000052298f},
         {-0.0000043403f, 0.0000006001f},  {0.0000044850f, -0.0000040295f},  {0.0f, 0.0000023148f},
         {0.0000023148f, -0.0000023148f},  {-0.0000023148f, 0.0000054012f},  {0.0000034722f, 0.0000007716f},
         {-0.0000011574f, -0.0000038580f}, {0.0000011574f, 0.0000038580f}}};
}  // namespace

UNIT_TEST_CASE(nss_api, entry_point_contracts)
{
    ffxContext context = reinterpret_cast<ffxContext>(static_cast<uintptr_t>(0x1));
    // Creating a context requires a descriptor.
    UNIT_TEST_CHECK(ffxCreateContext(&context, nullptr, nullptr) == FFX_API_RETURN_ERROR_PARAMETER);

    ffxCreateContextDescHeader createHeader = MakeApiHeader(0xE0000001u);
    // Creating a context requires a valid output pointer.
    UNIT_TEST_CHECK(ffxCreateContext(nullptr, &createHeader, nullptr) == FFX_API_RETURN_ERROR_PARAMETER);

    context      = reinterpret_cast<ffxContext>(static_cast<uintptr_t>(0x1));
    createHeader = MakeApiHeader(0xE0000002u);
    // An unsupported create descriptor must not resolve to a provider.
    UNIT_TEST_CHECK(ffxCreateContext(&context, &createHeader, nullptr) == FFX_API_RETURN_NO_PROVIDER);
    // Failed provider lookup must clear a previously non-null output handle.
    UNIT_TEST_CHECK(context == nullptr);

    context = nullptr;
    // Destroy rejects a pointer to a null context handle.
    UNIT_TEST_CHECK(ffxDestroyContext(&context, nullptr) == FFX_API_RETURN_ERROR_PARAMETER);

    ffxConfigureDescHeader configureHeader = MakeApiHeader(0xE0000003u);
    // Configure rejects a null context handle.
    UNIT_TEST_CHECK(ffxConfigure(&context, &configureHeader) == FFX_API_RETURN_ERROR_PARAMETER);

    // Query rejects a missing descriptor.
    UNIT_TEST_CHECK(ffxQuery(nullptr, nullptr) == FFX_API_RETURN_ERROR_PARAMETER);

    ffxQueryDescHeader queryHeader = MakeApiHeader(FFX_API_QUERY_DESC_TYPE_GET_VERSIONS);
    // A context-bound query rejects a pointer to a null context handle.
    UNIT_TEST_CHECK(ffxQuery(&context, &queryHeader) == FFX_API_RETURN_ERROR_PARAMETER);

    ffxDispatchDescHeader dispatchHeader = MakeApiHeader(0xE0000004u);
    // Dispatch rejects a null context handle.
    UNIT_TEST_CHECK(ffxDispatch(&context, &dispatchHeader) == FFX_API_RETURN_ERROR_PARAMETER);

    queryHeader = MakeApiHeader(0xE0000005u);
    // A context-free query with an unsupported descriptor has no provider.
    UNIT_TEST_CHECK(ffxQuery(nullptr, &queryHeader) == FFX_API_RETURN_NO_PROVIDER);

    // Track outstanding allocations made through the provider callbacks.
    size_t                 liveAllocations = 0;
    ffxAllocationCallbacks allocator{};
    allocator.pUserData = &liveAllocations;
    allocator.alloc     = [](void* userData, uint64_t size) -> void* {
        if (size > std::numeric_limits<size_t>::max())
        {
            return nullptr;
        }
        void* memory = std::malloc(static_cast<size_t>(size));
        if (memory)
        {
            ++*static_cast<size_t*>(userData);
        }
        return memory;
    };
    allocator.dealloc = [](void* userData, void* memory) {
        if (memory)
        {
            --*static_cast<size_t*>(userData);
        }
        std::free(memory);
    };

    ffxApiCreateContextDescNss description{};
    description.header.type    = FFX_API_CREATE_CONTEXT_DESC_TYPE_NSS;
    description.flags          = FFX_API_NSS_CONTEXT_FLAG_QUANTIZED;
    description.maxRenderSize  = {96, 96};
    description.maxUpscaleSize = {192, 192};
    context                    = nullptr;

    // NSS creation without a backend must fail.
    UNIT_TEST_CHECK(ffxCreateContext(&context, &description.header, &allocator) != FFX_API_RETURN_OK);
    // Failed creation must not publish a context or leak callback allocations.
    UNIT_TEST_CHECK(context == nullptr && liveAllocations == 0);
    return true;
}

UNIT_TEST_CASE(nss_api, jitter_cancellation_projection_and_history)
{
    NssConstants              constants{};
    FfxNssDispatchDescription params{};
    params.renderSize                                        = {128, 64};
    const std::array<float, 3>                currentPoint   = {0.125f, -0.25f, -2.0f};
    const std::array<std::array<float, 3>, 2> previousPoints = {currentPoint, {0.25f, -0.125f, -2.0f}};
    const FfxFloatCoords2D                    previousJitter = {-0.375f, 0.25f};
    const FfxFloatCoords2D                    currentJitter  = {0.375f, -0.25f};
    params.jitterOffset                                      = previousJitter;
    UpdateNssJitterConstants(constants, params, true, true);
    UNIT_TEST_CHECK(constants._JitterCancellation[0] == 0 && constants._JitterCancellation[1] == 0);
    params.jitterOffset = currentJitter;
    UpdateNssJitterConstants(constants, params, false, true);

    const std::array<FfxFloatCoords2D, 5> scales = {{{1, 1}, {128, 64}, {-128, -64}, {128, -64}, {64, -32}}};
    for (const auto& previousPoint : previousPoints)
    {
        const auto previous           = ProjectJitteredPoint(previousPoint, previousJitter, params.renderSize);
        const auto current            = ProjectJitteredPoint(currentPoint, currentJitter, params.renderSize);
        const auto unjitteredPrevious = ProjectJitteredPoint(previousPoint, {0, 0}, params.renderSize);
        const auto unjitteredCurrent  = ProjectJitteredPoint(currentPoint, {0, 0}, params.renderSize);
        for (const auto& scale : scales)
        {
            const float rawMotion[] = {(previous[0] - current[0]) / scale.x, (previous[1] - current[1]) / scale.y};
            UNIT_TEST_CHECK(std::fabs(rawMotion[0] * scale.x - constants._JitterCancellation[0] - (unjitteredPrevious[0] - unjitteredCurrent[0])) < 0.00001f);
            UNIT_TEST_CHECK(std::fabs(rawMotion[1] * scale.y - constants._JitterCancellation[1] - (unjitteredPrevious[1] - unjitteredCurrent[1])) < 0.00001f);
        }
    }

    UNIT_TEST_CHECK(constants._JitterOffsetTm1[2] == previousJitter.x / 128);
    UNIT_TEST_CHECK(constants._JitterOffset[3] == currentJitter.y / 64);
    UpdateNssJitterConstants(constants, params, false, true);
    UNIT_TEST_CHECK(constants._JitterCancellation[0] == 0 && constants._JitterCancellation[1] == 0);
    params.jitterOffset = previousJitter;
    UpdateNssJitterConstants(constants, params, false, false);
    UNIT_TEST_CHECK(constants._JitterCancellation[0] == 0 && constants._JitterCancellation[1] == 0);
    params.jitterOffset = currentJitter;
    UpdateNssJitterConstants(constants, params, true, true);
    UNIT_TEST_CHECK(constants._JitterCancellation[0] == 0 && constants._JitterCancellation[1] == 0);
    UNIT_TEST_CHECK(constants._JitterOffsetTm1[0] == currentJitter.x && constants._JitterOffsetTm1[1] == currentJitter.y);
    params.jitterOffset = previousJitter;
    UpdateNssJitterConstants(constants, params, false, true);
    UNIT_TEST_CHECK(constants._JitterCancellation[0] == -0.75f && constants._JitterCancellation[1] == 0.5f);
    return true;
}

UNIT_TEST_CASE(nss_api, jitter_cancellation_matches_pre_cancelled_motion_feature)
{
    struct Device
    {
        unit_test::vk::VulkanContext context{};
        ~Device()
        {
            unit_test::vk::DestroyVulkanContext(context);
        }
    } device;
    std::string error;
    if (!unit_test::vk::InitializeVulkanContext(device.context, &error))
        UNIT_TEST_FAIL(error);
    const bool computePostProcessSupported = SupportsNssComputePostProcess(device.context);

    struct Variant
    {
        FfxApiNssShaderQualityMode quality;
        uint32_t                   flags;
        FfxDimensions2D            upscale;
        FfxFloatCoords2D           motionScale;
    };
    const std::array<Variant, 3>          variants     = {{
        {FFX_API_NSS_SHADER_QUALITY_MODE_QUALITY, 0, {256, 128}, {1, 1}},
        {FFX_API_NSS_SHADER_QUALITY_MODE_BALANCED,
         FFX_API_NSS_CONTEXT_FLAG_LOW_DYNAMIC_RANGE | FFX_API_NSS_CONTEXT_FLAG_PRE_PROCESS_COMPUTE | FFX_API_NSS_CONTEXT_FLAG_POST_PROCESS_COMPUTE |
             FFX_API_NSS_CONTEXT_FLAG_MANAGE_HISTORY,
         {192, 96},
         {128, -64}},
        {FFX_API_NSS_SHADER_QUALITY_MODE_PERFORMANCE, 0, {256, 128}, {-128, -64}},
    }};
    constexpr FfxDimensions2D             render       = {128, 64};
    const std::array<FfxFloatCoords2D, 5> jitters      = {{{-0.375f, 0.25f}, {0.375f, -0.25f}, {-0.125f, 0.375f}, {0.375f, -0.125f}, {-0.25f, 0.25f}}};
    const std::array<float, 3>            currentPoint = {0, 0, -1};
    // Use a (2, -1) render-pixel base motion so the motion feature exceeds its subpixel cutoff.
    const std::array<float, 3> previousPoint = {0.03125f, -0.03125f, -1};
    for (const auto& variant : variants)
    {
        if ((variant.flags & FFX_API_NSS_CONTEXT_FLAG_POST_PROCESS_COMPUTE) && !computePostProcessSupported)
        {
            unit_test::MarkTestSkipped("R11G11B10 storage-image support is unavailable; the NSS compute jitter-cancellation variant was not executed");
            continue;
        }

        std::array<std::vector<uint8_t>, 5> reference;
        bool                                negativeControlDiffers = false;
        // The contexts use de-jittered motion, jitter cancellation, and a disabled-cancellation negative control.
        for (uint32_t run = 0; run < 3; ++run)
        {
            CancellationScene scene(device.context);
            const uint32_t    flags =
                FFX_API_NSS_CONTEXT_FLAG_QUANTIZED | variant.flags | (run == 1 ? FFX_API_NSS_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION : 0);
            if (!scene.Initialize(flags, variant.quality, render, variant.upscale))
                UNIT_TEST_FAIL(scene.error);
            for (uint32_t frame = 0; frame < jitters.size(); ++frame)
            {
                const bool           reset = frame == 0 || frame == 3;
                std::array<float, 2> rawMotion{};
                if (!reset)
                {
                    const bool useUnjitteredMotion = run == 0;
                    const auto current             = ProjectJitteredPoint(currentPoint, useUnjitteredMotion ? FfxFloatCoords2D{0, 0} : jitters[frame], render);
                    const auto previous = ProjectJitteredPoint(previousPoint, useUnjitteredMotion ? FfxFloatCoords2D{0, 0} : jitters[frame - 1], render);
                    rawMotion           = {(previous[0] - current[0]) / variant.motionScale.x, (previous[1] - current[1]) / variant.motionScale.y};
                }
                std::vector<uint8_t> pixels;
                if (!scene.Dispatch(frame, jitters[frame], variant.motionScale, rawMotion, reset, pixels))
                    UNIT_TEST_FAIL(scene.error);
                if (run == 0)
                    reference[frame] = std::move(pixels);
                else
                {
                    bool equivalent = false;
                    UNIT_TEST_CHECK(CompareMotionDetectorPixels(reference[frame], pixels, equivalent));
                    if (run == 1 && !equivalent)
                        UNIT_TEST_FAIL("Jitter cancellation motion feature differs from pre-cancelled motion: quality=" + std::to_string(variant.quality) +
                                       ", frame=" + std::to_string(frame));
                    if (run == 2 && !reset)
                        negativeControlDiffers |= !equivalent;
                }
            }
        }
        UNIT_TEST_CHECK(negativeControlDiffers);
    }
    return true;
}

UNIT_TEST_CASE(nss_api, query_jitter_without_context)
{
    int32_t phaseCount = 0;

    ffxApiQueryDescNssGetJitterPhaseCount phaseQuery{};
    phaseQuery.header.type    = FFX_API_QUERY_DESC_TYPE_NSS_GETJITTERPHASECOUNT;
    phaseQuery.header.pNext   = nullptr;
    phaseQuery.renderWidth    = 960;
    phaseQuery.displayWidth   = 1920;
    phaseQuery.pOutPhaseCount = &phaseCount;
    UNIT_TEST_CHECK(ffxQuery(nullptr, &phaseQuery.header) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(phaseCount > 0);

    float                             jitterX = 0.0f;
    float                             jitterY = 0.0f;
    ffxApiQueryDescNssGetJitterOffset jitterQuery{};
    jitterQuery.header.type  = FFX_API_QUERY_DESC_TYPE_NSS_GETJITTEROFFSET;
    jitterQuery.header.pNext = nullptr;
    jitterQuery.index        = 1;
    jitterQuery.phaseCount   = phaseCount;
    jitterQuery.pOutX        = &jitterX;
    jitterQuery.pOutY        = &jitterY;
    UNIT_TEST_CHECK(ffxQuery(nullptr, &jitterQuery.header) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(jitterX != 0.0f || jitterY != 0.0f);
    return true;
}

UNIT_TEST_CASE(nss_api, create_destroy_with_flag_variants)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!unit_test::vk::InitializeVulkanContext(vkContext, &errorMessage))
    {
        UNIT_TEST_FAIL(std::string("Vulkan unavailable: ") + errorMessage);
    }

    struct Variant
    {
        uint32_t                   flags;
        FfxApiNssShaderQualityMode quality;
    };

    const std::vector<Variant> variants = {
        {FFX_API_NSS_CONTEXT_FLAG_QUANTIZED, FFX_API_NSS_SHADER_QUALITY_MODE_QUALITY},
        {FFX_API_NSS_CONTEXT_FLAG_QUANTIZED | FFX_API_NSS_CONTEXT_FLAG_MANAGE_HISTORY | FFX_API_NSS_CONTEXT_FLAG_DISABLE_16BIT |
             FFX_API_NSS_CONTEXT_FLAG_LOW_DYNAMIC_RANGE | FFX_API_NSS_CONTEXT_FLAG_PRE_PROCESS_COMPUTE | FFX_API_NSS_CONTEXT_FLAG_POST_PROCESS_COMPUTE,
         FFX_API_NSS_SHADER_QUALITY_MODE_PERFORMANCE}};

    const bool r11g11b10StorageImageSupported = SupportsNssComputePostProcess(vkContext);

    for (const Variant& variant : variants)
    {
        ffxContext context            = nullptr;
        const auto createResult       = CreateNssContext(vkContext, variant.flags, variant.quality, 96, 96, 192, 192, context);
        const bool computePostProcess = (variant.flags & FFX_API_NSS_CONTEXT_FLAG_POST_PROCESS_COMPUTE) != 0;
        if (computePostProcess && !r11g11b10StorageImageSupported)
        {
            UNIT_TEST_CHECK(createResult == FFX_API_RETURN_ERROR_PARAMETER);
            UNIT_TEST_CHECK(context == nullptr);
            continue;
        }
        if (createResult == FFX_API_RETURN_ERROR_RUNTIME_ERROR)
        {
            unit_test::vk::DestroyVulkanContext(vkContext);
            UNIT_TEST_FAIL("NSS context creation is unsupported on this Vulkan device/runtime.");
        }
        UNIT_TEST_CHECK(createResult == FFX_API_RETURN_OK);
        UNIT_TEST_CHECK(ffxDestroyContext(&context, nullptr) == FFX_API_RETURN_OK);
    }

    unit_test::vk::DestroyVulkanContext(vkContext);
    return true;
}

#if !FFX_CAPTURE_ENABLED
UNIT_TEST_CASE(nss_api, capture_descriptors_rejected_when_capture_disabled)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!unit_test::vk::InitializeVulkanContext(vkContext, &errorMessage))
    {
        UNIT_TEST_FAIL(std::string("Vulkan unavailable: ") + errorMessage);
    }

    ffxContext context = nullptr;
    const auto createResult =
        CreateNssContext(vkContext, FFX_API_NSS_CONTEXT_FLAG_QUANTIZED, FFX_API_NSS_SHADER_QUALITY_MODE_QUALITY, 96, 96, 192, 192, context);
    if (createResult == FFX_API_RETURN_ERROR_RUNTIME_ERROR)
    {
        unit_test::vk::DestroyVulkanContext(vkContext);
        UNIT_TEST_FAIL("NSS context creation is unsupported on this Vulkan device/runtime.");
    }
    UNIT_TEST_CHECK(createResult == FFX_API_RETURN_OK);

    ffxApiConfigureDescCapture configure{};
    configure.header.type            = FFX_API_CONFIGURE_DESC_TYPE_CAPTURE;
    configure.operation              = FFX_API_CAPTURE_BEGIN;
    configure.outputDirectory        = "capture-disabled";
    configure.frameCount             = 1;
    configure.motionVectorUnormScale = 1.0f;
    UNIT_TEST_CHECK(ffxConfigure(&context, &configure.header) == FFX_API_RETURN_ERROR_PARAMETER);

    FfxApiCaptureStatus          status{};
    ffxApiQueryDescCaptureStatus query{};
    query.header.type = FFX_API_QUERY_DESC_TYPE_CAPTURESTATUS;
    query.pOutStatus  = &status;
    UNIT_TEST_CHECK(ffxQuery(&context, &query.header) == FFX_API_RETURN_ERROR_PARAMETER);

    UNIT_TEST_CHECK(ffxDestroyContext(&context, nullptr) == FFX_API_RETURN_OK);
    unit_test::vk::DestroyVulkanContext(vkContext);
    return true;
}
#endif

UNIT_TEST_CASE(nss_api, logging_flags_and_failures_survive_the_provider)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!unit_test::vk::InitializeVulkanContext(vkContext, &errorMessage))
    {
        UNIT_TEST_FAIL(std::string("Vulkan unavailable: ") + errorMessage);
    }

    diagnosticMessages.clear();
    ffxContext context      = nullptr;
    const auto createResult = CreateNssContext(vkContext,
                                               FFX_API_NSS_CONTEXT_FLAG_QUANTIZED | FFX_API_NSS_CONTEXT_FLAG_MANAGE_HISTORY,
                                               FFX_API_NSS_SHADER_QUALITY_MODE_BALANCED,
                                               96,
                                               96,
                                               192,
                                               192,
                                               context,
                                               CaptureApiMessage);
    if (createResult != FFX_API_RETURN_OK)
    {
        unit_test::vk::DestroyVulkanContext(vkContext);
        UNIT_TEST_FAIL("NSS diagnostic integration requires a supported device or ML emulation layer.");
    }

    const auto countMessages = [](const char* text) {
        return std::count_if(
            diagnosticMessages.begin(), diagnosticMessages.end(), [text](const std::string& message) { return message.find(text) != std::string::npos; });
    };
    ffxApiDispatchDescNss dispatch{};
    dispatch.header.type       = FFX_API_DISPATCH_DESC_TYPE_NSS;
    const auto firstResult     = ffxDispatch(&context, &dispatch.header);
    const auto repeatedResult  = ffxDispatch(&context, &dispatch.header);
    const auto quietErrorCount = countMessages("event=dispatch_error");
    const auto quietFrameCount = countMessages("event=dispatch_begin");
    dispatch.flags             = FFX_API_NSS_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING;
    const auto verboseResult   = ffxDispatch(&context, &dispatch.header);
    const auto destroyResult   = ffxDestroyContext(&context, nullptr);
    unit_test::vk::DestroyVulkanContext(vkContext);

    UNIT_TEST_CHECK(firstResult == FFX_API_RETURN_ERROR_RUNTIME_ERROR);
    UNIT_TEST_CHECK(repeatedResult == firstResult && verboseResult == firstResult);
    UNIT_TEST_CHECK(quietErrorCount == 1 && quietFrameCount == 0);
    UNIT_TEST_CHECK(countMessages("event=dispatch_error") == 2);
    UNIT_TEST_CHECK(countMessages("event=dispatch_begin") == 1);
    UNIT_TEST_CHECK(countMessages("event=context_create") == 1);
    UNIT_TEST_CHECK(countMessages("event=context_destroy") == 1);
    UNIT_TEST_CHECK(destroyResult == FFX_API_RETURN_OK && context == nullptr);
    return true;
}

UNIT_TEST_CASE(nss_api, dispatch_triangle_upscale_changes_output_pixels)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!unit_test::vk::InitializeVulkanContext(vkContext, &errorMessage))
    {
        UNIT_TEST_FAIL(std::string("Vulkan unavailable: ") + errorMessage);
    }

    constexpr uint32_t kRenderWidth   = 128;
    constexpr uint32_t kRenderHeight  = 128;
    constexpr uint32_t kUpscaleWidth  = 256;
    constexpr uint32_t kUpscaleHeight = 256;

    ffxContext      context = nullptr;
    std::string     failureReason;
    ffxReturnCode_t createResult = CreateNssContext(vkContext,
                                                    FFX_API_NSS_CONTEXT_FLAG_QUANTIZED | FFX_API_NSS_CONTEXT_FLAG_MANAGE_HISTORY,
                                                    FFX_API_NSS_SHADER_QUALITY_MODE_QUALITY,
                                                    kRenderWidth,
                                                    kRenderHeight,
                                                    kUpscaleWidth,
                                                    kUpscaleHeight,
                                                    context);
    if (createResult != FFX_API_RETURN_OK)
    {
        failureReason = "CreateNssContext failed: code=" + std::to_string(static_cast<int>(createResult));
    }

    unit_test::vk::Image colorImage{};
    unit_test::vk::Image depthImage{};
    unit_test::vk::Image motionImage{};
    unit_test::vk::Image outputImage{};
    unit_test::vk::Image outputTm1Image{};

    bool success         = false;
    bool hasVisibleInput = false;
    if (failureReason.empty())
    {
        do
        {
            if (!unit_test::vk::CreateImage2D(vkContext,
                                              kRenderWidth,
                                              kRenderHeight,
                                              VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                                  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                              colorImage,
                                              &errorMessage))
            {
                failureReason = "CreateImage2D(color) failed: " + errorMessage;
                break;
            }
            if (!unit_test::vk::CreateImage2D(
                    vkContext,
                    kRenderWidth,
                    kRenderHeight,
                    VK_FORMAT_R32_SFLOAT,
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    depthImage,
                    &errorMessage))
            {
                failureReason = "CreateImage2D(depth) failed: " + errorMessage;
                break;
            }
            if (!unit_test::vk::CreateImage2D(
                    vkContext,
                    kRenderWidth,
                    kRenderHeight,
                    VK_FORMAT_R16G16_SFLOAT,
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    motionImage,
                    &errorMessage))
            {
                failureReason = "CreateImage2D(motion) failed: " + errorMessage;
                break;
            }
            if (!unit_test::vk::CreateImage2D(vkContext,
                                              kUpscaleWidth,
                                              kUpscaleHeight,
                                              VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                              outputImage,
                                              &errorMessage))
            {
                failureReason = "CreateImage2D(output) failed: " + errorMessage;
                break;
            }
            if (!unit_test::vk::CreateImage2D(
                    vkContext,
                    kUpscaleWidth,
                    kUpscaleHeight,
                    VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    outputTm1Image,
                    &errorMessage))
            {
                failureReason = "CreateImage2D(outputTm1) failed: " + errorMessage;
                break;
            }

            if (!unit_test::vk::RenderTriangle(vkContext, colorImage, &errorMessage))
            {
                failureReason = "RenderTriangle failed: " + errorMessage;
                break;
            }
            std::vector<uint8_t> inputColorRgba;
            if (!unit_test::vk::ReadbackRgba8Image(vkContext, colorImage, inputColorRgba, &errorMessage))
            {
                failureReason = "ReadbackRgba8Image(input color) failed: " + errorMessage;
                break;
            }
            for (size_t i = 0; i + 3 < inputColorRgba.size(); i += 4)
            {
                uint32_t packedPixel = 0;
                std::memcpy(&packedPixel, inputColorRgba.data() + i, sizeof(uint32_t));
                if (packedPixel != 0)
                {
                    hasVisibleInput = true;
                    break;
                }
            }
            if (!hasVisibleInput)
            {
                failureReason = "RenderTriangle produced no visible pixels in NSS input image";
                break;
            }
            if (!unit_test::vk::ClearImage(vkContext, depthImage, {1.0f, 0.0f, 0.0f, 0.0f}, &errorMessage))
            {
                failureReason = "ClearImage(depth) failed: " + errorMessage;
                break;
            }
            if (!unit_test::vk::ClearImage(vkContext, motionImage, {0.0f, 0.0f, 0.0f, 0.0f}, &errorMessage))
            {
                failureReason = "ClearImage(motion) failed: " + errorMessage;
                break;
            }
            if (!unit_test::vk::ClearImage(vkContext, outputImage, {0.0f, 0.0f, 0.0f, 1.0f}, &errorMessage))
            {
                failureReason = "ClearImage(output) failed: " + errorMessage;
                break;
            }
            if (!unit_test::vk::ClearImage(vkContext, outputTm1Image, {0.0f, 0.0f, 0.0f, 1.0f}, &errorMessage))
            {
                failureReason = "ClearImage(outputTm1) failed: " + errorMessage;
                break;
            }

            int32_t                               phaseCount = 1;
            ffxApiQueryDescNssGetJitterPhaseCount phaseQuery{};
            phaseQuery.header.type    = FFX_API_QUERY_DESC_TYPE_NSS_GETJITTERPHASECOUNT;
            phaseQuery.header.pNext   = nullptr;
            phaseQuery.renderWidth    = kRenderWidth;
            phaseQuery.displayWidth   = kUpscaleWidth;
            phaseQuery.pOutPhaseCount = &phaseCount;
            if (ffxQuery(&context, &phaseQuery.header) != FFX_API_RETURN_OK || phaseCount <= 0)
            {
                failureReason = "ffxQuery(NSS_GETJITTERPHASECOUNT) failed";
                break;
            }

            auto dispatchOneFrame = [&](const int32_t jitterIndex, const bool reset) -> bool {
                float                             jitterX = 0.0f;
                float                             jitterY = 0.0f;
                ffxApiQueryDescNssGetJitterOffset jitterQuery{};
                jitterQuery.header.type  = FFX_API_QUERY_DESC_TYPE_NSS_GETJITTEROFFSET;
                jitterQuery.header.pNext = nullptr;
                jitterQuery.index        = jitterIndex;
                jitterQuery.phaseCount   = phaseCount;
                jitterQuery.pOutX        = &jitterX;
                jitterQuery.pOutY        = &jitterY;
                if (ffxQuery(&context, &jitterQuery.header) != FFX_API_RETURN_OK)
                {
                    failureReason = "ffxQuery(NSS_GETJITTEROFFSET) failed";
                    return false;
                }

                VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
                if (!unit_test::vk::BeginCommandBuffer(vkContext, commandBuffer, &errorMessage))
                {
                    failureReason = "BeginCommandBuffer failed: " + errorMessage;
                    return false;
                }

                ffxApiDispatchDescNss dispatchDesc{};
                dispatchDesc.header.type  = FFX_API_DISPATCH_DESC_TYPE_NSS;
                dispatchDesc.header.pNext = nullptr;
                dispatchDesc.commandList  = commandBuffer;
                dispatchDesc.color =
                    unit_test::vk::MakeApiImageResource(colorImage, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
                dispatchDesc.depth =
                    unit_test::vk::MakeApiImageResource(depthImage, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
                dispatchDesc.motionVectors =
                    unit_test::vk::MakeApiImageResource(motionImage, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
                dispatchDesc.output = unit_test::vk::MakeApiImageResource(outputImage, FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
                dispatchDesc.outputTm1 =
                    unit_test::vk::MakeApiImageResource(outputTm1Image, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
                dispatchDesc.jitterOffset           = {jitterX, jitterY};
                dispatchDesc.upscaleSize            = {kUpscaleWidth, kUpscaleHeight};
                dispatchDesc.renderSize             = {kRenderWidth, kRenderHeight};
                dispatchDesc.cameraNear             = 0.1f;
                dispatchDesc.cameraFar              = 1000.0f;
                dispatchDesc.cameraFovAngleVertical = 1.2f;
                dispatchDesc.exposure               = 1.0f;
                dispatchDesc.motionVectorScale      = {static_cast<float>(kRenderWidth), static_cast<float>(kRenderHeight)};
                dispatchDesc.frameTimeDelta         = 16.6f;
                dispatchDesc.reset                  = reset;
                dispatchDesc.flags                  = 0;

                const auto dispatchResult = ffxDispatch(&context, &dispatchDesc.header);
                const bool submitOk       = unit_test::vk::SubmitCommandBuffer(vkContext, commandBuffer, &errorMessage);
                if (dispatchResult != FFX_API_RETURN_OK || !submitOk)
                {
                    failureReason = "ffxDispatch(NSS) or command submission failed";
                    return false;
                }
                return true;
            };

            const int32_t secondJitterIndex = (phaseCount > 1) ? 1 : 0;
            if (!dispatchOneFrame(0, true) || !dispatchOneFrame(secondJitterIndex, false))
            {
                break;
            }

            std::vector<uint8_t> outputRgba;
            if (!unit_test::vk::ReadbackRgba8Image(vkContext, outputImage, outputRgba, &errorMessage))
            {
                failureReason = "ReadbackRgba8Image failed: " + errorMessage;
                break;
            }

            bool hasVisibleOutput = false;
            if (outputImage.createInfo.format == VK_FORMAT_B10G11R11_UFLOAT_PACK32)
            {
                for (size_t i = 0; i + 3 < outputRgba.size(); i += 4)
                {
                    uint32_t packedPixel = 0;
                    std::memcpy(&packedPixel, outputRgba.data() + i, sizeof(uint32_t));
                    if (packedPixel != 0)
                    {
                        hasVisibleOutput = true;
                        break;
                    }
                }
            }
            else
            {
                for (size_t i = 0; i + 3 < outputRgba.size(); i += 4)
                {
                    const uint8_t red   = outputRgba[i + 0];
                    const uint8_t green = outputRgba[i + 1];
                    const uint8_t blue  = outputRgba[i + 2];
                    if (red > 8 || green > 8 || blue > 8)
                    {
                        hasVisibleOutput = true;
                        break;
                    }
                }
            }

            if (!hasVisibleOutput)
            {
                failureReason = "dispatch finished but output image remained at clear color";
            }

            success = hasVisibleOutput;
        } while (false);
    }

    unit_test::vk::DestroyImage(vkContext, outputTm1Image);
    unit_test::vk::DestroyImage(vkContext, outputImage);
    unit_test::vk::DestroyImage(vkContext, motionImage);
    unit_test::vk::DestroyImage(vkContext, depthImage);
    unit_test::vk::DestroyImage(vkContext, colorImage);
    DestroyContextIfNeeded(context);
    unit_test::vk::DestroyVulkanContext(vkContext);
    if (!success)
    {
        UNIT_TEST_FAIL(failureReason.empty() ? "NSS dispatch test failed" : failureReason);
    }
    return true;
}

UNIT_TEST_CASE(nss_api, valid_halton_jitter_sequence)
{
    const std::vector<NssJitterSample> samples  = ValidSequence(32);
    const NssJitterAnalysis            analysis = AnalyzeNssJitter(samples.data(), static_cast<uint32_t>(samples.size()), 32);

    UNIT_TEST_CHECK(analysis.sampleCount == 32);
    UNIT_TEST_CHECK(analysis.uniqueCount == 32);
    UNIT_TEST_CHECK(analysis.expectedUniqueCount == 32);
    UNIT_TEST_CHECK(analysis.finite);
    UNIT_TEST_CHECK(analysis.inBounds);
    UNIT_TEST_CHECK(analysis.centered);
    UNIT_TEST_CHECK(analysis.wideSpan);
    UNIT_TEST_CHECK(analysis.quadrants);
    UNIT_TEST_CHECK(analysis.grid);
    UNIT_TEST_CHECK(analysis.discrepancy);
    UNIT_TEST_CHECK(analysis.spacing);
    UNIT_TEST_CHECK(analysis.correlation);
    UNIT_TEST_CHECK(analysis.enoughUnique);
    UNIT_TEST_CHECK(analysis.score == 10);
    UNIT_TEST_CHECK(analysis.failedChecks == 0);
    UNIT_TEST_CHECK(std::isfinite(analysis.xyCorrelation));
    return true;
}

UNIT_TEST_CASE(nss_api, invalid_jitter_sequences_report_failed_checks)
{
    constexpr uint32_t                       sampleCount = 32;
    std::array<NssJitterSample, sampleCount> malformed{}, correlated{}, concentrated{};
    malformed.fill({0.25f, 0.25f});
    malformed[0] = {NAN, 0.0f};
    malformed[1] = {0.75f, 0.0f};
    for (uint32_t index = 0; index < sampleCount; ++index)
    {
        const float coordinate = -0.48f + static_cast<float>(index) * (0.96f / 31.0f);
        correlated[index]      = {coordinate, coordinate};
        concentrated[index]    = {-0.05f + static_cast<float>(index % 8) * (0.10f / 7.0f), -0.05f + static_cast<float>(index / 8) * (0.10f / 3.0f)};
    }

    const struct
    {
        const char*            name;
        const NssJitterSample* samples;
        uint32_t               count;
        bool (*matchesExpectedAnalysis)(const NssJitterAnalysis&);
    } cases[] = {
        {"malformed",
         malformed.data(),
         sampleCount,
         [](const NssJitterAnalysis& result) {
             return !result.finite && !result.inBounds && !result.centered && !result.wideSpan && !result.quadrants && !result.grid && !result.discrepancy &&
                    !result.spacing && !result.enoughUnique;
         }},
        {"empty",
         nullptr,
         0,
         [](const NssJitterAnalysis& result) {
             return result.uniqueCount == 0 && result.failedChecks == 9 && result.score == 1 && !result.inBounds && !result.centered && !result.wideSpan &&
                    !result.quadrants && !result.grid && !result.discrepancy && !result.spacing && !result.correlation && !result.enoughUnique;
         }},
        {"correlated",
         correlated.data(),
         sampleCount,
         [](const NssJitterAnalysis& result) {
             return result.uniqueCount == 32 && result.finite && result.inBounds && result.wideSpan && result.xyCorrelation > 0.75f && !result.correlation;
         }},
        {"concentrated",
         concentrated.data(),
         sampleCount,
         [](const NssJitterAnalysis& result) {
             return result.uniqueCount == 32 && result.finite && result.inBounds && result.spanX < 0.70f && result.spanY < 0.70f && !result.wideSpan;
         }},
        {"near-zero capture",
         kNearZeroJitterCycle.data(),
         static_cast<uint32_t>(kNearZeroJitterCycle.size()),
         [](const NssJitterAnalysis& result) {
             return result.uniqueCount == 32 && result.finite && result.inBounds && result.spanX < 0.00002f && result.spanY < 0.00002f &&
                    std::fabs(result.centroidX) < 0.00001f && std::fabs(result.centroidY) < 0.00001f && !result.wideSpan;
         }},
    };

    std::string failures;
    for (const auto& testCase : cases)
    {
        const auto result = AnalyzeNssJitter(testCase.samples, testCase.count, sampleCount);
        if (result.sampleCount != testCase.count || result.expectedUniqueCount != sampleCount || result.failedChecks == 0 || result.score >= 10 ||
            !testCase.matchesExpectedAnalysis(result))
        {
            failures += std::string(testCase.name) + ": unexpected jitter analysis; ";
        }
    }
    if (!failures.empty())
    {
        UNIT_TEST_FAIL(failures);
    }
    return true;
}
