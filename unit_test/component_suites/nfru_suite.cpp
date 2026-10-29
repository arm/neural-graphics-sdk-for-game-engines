/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "../framework/test_framework.h"
#include "../framework/vk_test_utils.h"

#include <FidelityFX/host/backends/vk/vk_wrapper.h>
#include <FidelityFX/host/ffx_frameinterpolation.h>
#include <FidelityFX/host/ffx_opticalflow.h>
#include <ffx_api/ffx_framegeneration.h>
#include "../../sdk/src/components/opticalflow/ffx_opticalflow_trace.h"
#include "../../sdk/src/components/frameinterpolation/ffx_frameinterpolation_private.h"

#include "../../ffx-api/src/backends.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

static constexpr const char* ffxLogComponentName()
{
    return "NFRUUnitTest";
}

namespace
{
    // All supported SDK target ABIs are 64-bit; pin changed public descriptors and pass values.
    static_assert(sizeof(void*) == 8);
    static_assert(FFX_FG_CONTEXT_FLAG_WARP_COMPUTE == (1u << 6));
    static_assert(FFX_FG_CONTEXT_FLAG_FILL_HOLES_COMPUTE == (1u << 7));
    static_assert(FFX_FG_CONTEXT_FLAG_PRE_PROCESS_COMPUTE == (1u << 8));
    static_assert(FFX_FG_CONTEXT_FLAG_POSTPROCESS_COMPUTE == (1u << 9));
    static_assert(FFX_FG_CONTEXT_FLAG_DEBUG_VIEW_COMPUTE == (1u << 11));
    static_assert(FFX_FG_CONTEXT_FLAG_ALL_STAGES_COMPUTE == 0xBC0u);
    static_assert(sizeof(ffxApiDispatchDescFrameGenerationPrepare) == 336);
    static_assert(alignof(ffxApiDispatchDescFrameGenerationPrepare) == 8);
    static_assert(offsetof(ffxApiDispatchDescFrameGenerationPrepare, mvSimilarityRelativeThreshold) == 56);
    static_assert(offsetof(ffxApiDispatchDescFrameGenerationPrepare, frameTimeDelta) == 60);
    static_assert(offsetof(ffxApiDispatchDescFrameGenerationPrepare, depth) == 80);
    static_assert(offsetof(ffxApiDispatchDescFrameGenerationPrepare, viewProjection) == 272);
    static_assert(sizeof(FfxFrameInterpolationPrepareDescription) == 848);
    static_assert(alignof(FfxFrameInterpolationPrepareDescription) == 8);
    static_assert(offsetof(FfxFrameInterpolationPrepareDescription, mvSimilarityRelativeThreshold) == 32);
    static_assert(offsetof(FfxFrameInterpolationPrepareDescription, frameTimeDelta) == 36);
    static_assert(offsetof(FfxFrameInterpolationPrepareDescription, depth) == 56);
    static_assert(offsetof(FfxFrameInterpolationPrepareDescription, frameID) == 776);
    static_assert(offsetof(FfxFrameInterpolationPrepareDescription, viewProjection) == 784);
    static_assert(sizeof(FfxFrameInterpolationDispatchDescription) == 424);
    static_assert(alignof(FfxFrameInterpolationDispatchDescription) == 8);
    static_assert(offsetof(FfxFrameInterpolationDispatchDescription, currentBackBuffer) == 16);
    static_assert(offsetof(FfxFrameInterpolationDispatchDescription, opticalFlowVector) == 256);
    static_assert(offsetof(FfxFrameInterpolationDispatchDescription, frameTimeDelta) == 408);
    static_assert(offsetof(FfxFrameInterpolationDispatchDescription, frameID) == 416);
    static_assert(FFX_FRAMEINTERPOLATION_PASS_INIT_WARP == 0);
    static_assert(FFX_FRAMEINTERPOLATION_PASS_DOWNSAMPLE_OF_COLOUR == 1);
    static_assert(FFX_FRAMEINTERPOLATION_PASS_DOWNSAMPLE_ALL_INPUT == 2);
    static_assert(FFX_FRAMEINTERPOLATION_PASS_DOWNSAMPLE_COLOUR == 3);
    static_assert(FFX_FRAMEINTERPOLATION_PASS_WARP_MV_TP1 == 4);
    static_assert(FFX_FRAMEINTERPOLATION_PASS_WARP_FLOW_TM1 == 5);
    static_assert(FFX_FRAMEINTERPOLATION_PASS_FILL_HOLES_TM1 == 6);
    static_assert(FFX_FRAMEINTERPOLATION_PASS_PREPROCESS == 7);
    static_assert(FFX_FRAMEINTERPOLATION_PASS_NFRU_INTERPOLATION == 8);
    static_assert(FFX_FRAMEINTERPOLATION_PASS_DOWNSAMPLE_INPAINT_SEED == 9);
    static_assert(FFX_FRAMEINTERPOLATION_PASS_DOWNSAMPLE_INPAINT_MIP == 10);
    static_assert(FFX_FRAMEINTERPOLATION_PASS_INPAINT == 11);
    static_assert(FFX_FRAMEINTERPOLATION_PASS_POSTPROCESS == 12);
    static_assert(FFX_FRAMEINTERPOLATION_PASS_DEBUG_VIEW == 13);
    static_assert(FFX_FRAMEINTERPOLATION_PASS_COUNT == 14);

    static_assert(sizeof(FfxFrameInterpolationContextDescription) == 424);
    static_assert(alignof(FfxFrameInterpolationContextDescription) == 8);
    static_assert(offsetof(FfxFrameInterpolationContextDescription, mvDepthSize) == 416);
    static_assert(sizeof(FfxOpticalFlowContextDescription) == 408);
    static_assert(alignof(FfxOpticalFlowContextDescription) == 8);
    static_assert(offsetof(FfxOpticalFlowContextDescription, mvDepthSize) == 400);

    // Ignore SDK callback severity/message inputs; these tests assert return codes and image results instead.
    void ApiMessageSink(uint32_t, const char*)
    {
    }

    // Borrow an initialized Vulkan device and create an NFRU context with the supplied context flags and pixel dimensions.
    // mvDepthSize == {0, 0} selects render size; backBufferFormat selects the colour format.
    // On success, outContext owns the API context and must be destroyed before vkContext. Return the API creation status.
    ffxReturnCode_t CreateNfruContext(unit_test::vk::VulkanContext& vkContext,
                                      const uint32_t                flags,
                                      const uint32_t                displayWidth,
                                      const uint32_t                displayHeight,
                                      const uint32_t                renderWidth,
                                      const uint32_t                renderHeight,
                                      ffxContext&                   outContext,
                                      FfxApiDimensions2D            mvDepthSize      = {},
                                      uint32_t                      backBufferFormat = FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM)
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

        ffxApiCreateContextDescFrameGeneration createDesc{};
        createDesc.header.type                        = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
        createDesc.header.pNext                       = reinterpret_cast<ffxApiHeader*>(&backendDesc);
        uint32_t           effectiveFlags             = flags;
        constexpr uint32_t fragmentAtomicComputeFlags = FFX_API_FG_CONTEXT_FLAG_WARP_COMPUTE | FFX_API_FG_CONTEXT_FLAG_PRE_PROCESS_COMPUTE;
        if ((effectiveFlags & fragmentAtomicComputeFlags) != fragmentAtomicComputeFlags)
        {
            VkPhysicalDeviceFeatures physicalFeatures{};
            vkGetPhysicalDeviceFeatures(vkContext.physicalDevice, &physicalFeatures);
            if (!physicalFeatures.fragmentStoresAndAtomics)
            {
                effectiveFlags |= fragmentAtomicComputeFlags;
                std::printf("NFRU GPU test selects compute warp/preprocess because fragmentStoresAndAtomics is unavailable.\n");
            }
        }
        createDesc.flags            = effectiveFlags;
        createDesc.displaySize      = {displayWidth, displayHeight};
        createDesc.renderSize       = {renderWidth, renderHeight};
        createDesc.mvDepthSize      = mvDepthSize;
        createDesc.backBufferFormat = backBufferFormat;
        createDesc.fpMessage        = ApiMessageSink;
        unit_test::vk::SetIdentityMatrix(createDesc.initialViewProjection);

        return ffxCreateContext(&outContext, &createDesc.header, nullptr);
    }

    // Configure an existing context for frameId using the supplied enable state and dispatch flags.
    // No swapchain or presentation callbacks are installed; return the API configuration status.
    ffxReturnCode_t ConfigureNfruContext(ffxContext& context, const bool frameGenerationEnabled, const uint32_t flags, const uint64_t frameId)
    {
        ffxApiConfigureDescFrameGeneration configureDesc{};
        configureDesc.header.type                        = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        configureDesc.header.pNext                       = nullptr;
        configureDesc.swapChain                          = nullptr;
        configureDesc.presentCallback                    = nullptr;
        configureDesc.presentCallbackUserContext         = nullptr;
        configureDesc.frameGenerationCallback            = nullptr;
        configureDesc.frameGenerationCallbackUserContext = nullptr;
        configureDesc.frameGenerationEnabled             = frameGenerationEnabled;
        configureDesc.allowAsyncWorkloads                = false;
        configureDesc.flags                              = flags;
        configureDesc.onlyPresentGenerated               = false;
        configureDesc.frameID                            = frameId;
        configureDesc.dumpGeneratedFramePath             = nullptr;

        return ffxConfigure(&context, &configureDesc.header);
    }

    // Release a non-null API context through its owning handle; null is allowed. No destruction status is returned.
    void DestroyContextIfNeeded(ffxContext& context)
    {
        if (context)
        {
            ffxDestroyContext(&context, nullptr);
        }
    }

    struct TrackingAllocator
    {
        size_t liveAllocations = 0;

        ffxAllocationCallbacks Callbacks()
        {
            ffxAllocationCallbacks callbacks{};
            callbacks.pUserData = this;
            callbacks.alloc     = &Alloc;
            callbacks.dealloc   = &Dealloc;
            return callbacks;
        }

        static void* Alloc(void* pUserData, uint64_t size)
        {
            auto* self   = static_cast<TrackingAllocator*>(pUserData);
            void* memory = std::malloc(static_cast<size_t>(size));
            if (memory)
            {
                ++self->liveAllocations;
            }
            return memory;
        }

        static void Dealloc(void* pUserData, void* pMem)
        {
            if (!pMem)
            {
                return;
            }
            auto* self = static_cast<TrackingAllocator*>(pUserData);
            if (self->liveAllocations > 0)
            {
                --self->liveAllocations;
            }
            std::free(pMem);
        }
    };

    ffxApiCreateContextDescFrameGeneration MakeFgCreateDesc(
        const uint32_t displayWidth, const uint32_t displayHeight, const uint32_t renderWidth, const uint32_t renderHeight, const uint32_t backBufferFormat)
    {
        ffxApiCreateContextDescFrameGeneration createDesc{};
        createDesc.header.type      = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
        createDesc.flags            = 0;
        createDesc.displaySize      = {displayWidth, displayHeight};
        createDesc.renderSize       = {renderWidth, renderHeight};
        createDesc.backBufferFormat = backBufferFormat;
        createDesc.fpMessage        = ApiMessageSink;
        unit_test::vk::SetIdentityMatrix(createDesc.initialViewProjection);
        return createDesc;
    }

    std::vector<std::string>* g_fakeNfruMessages = nullptr;

    void FakeNfruApiMessageSink(uint32_t, const char* message)
    {
        if (g_fakeNfruMessages && message)
        {
            g_fakeNfruMessages->emplace_back(message);
        }
    }

    bool HasFakeNfruMessage(const std::vector<std::string>& messages, const char* text)
    {
        return std::any_of(messages.begin(), messages.end(), [text](const std::string& message) { return message.find(text) != std::string::npos; });
    }

    struct FakeNfruBackend
    {
        FfxErrorCode             scheduleResult                 = FFX_OK;
        FfxErrorCode             executeResult                  = FFX_OK;
        FfxErrorCode             createBackendContextResult     = FFX_OK;
        FfxErrorCode             createResourceResult           = FFX_OK;
        FfxErrorCode             createPipelineResult           = FFX_OK;
        FfxErrorCode             destroyPipelineResult          = FFX_OK;
        FfxErrorCode             destroyResourceResult          = FFX_OK;
        FfxErrorCode             destroyBackendContextResult    = FFX_OK;
        FfxErrorCode             getDeviceCapabilitiesResult    = FFX_OK;
        FfxErrorCode             setMessageCallbackResult       = FFX_OK;
        FfxErrorCode             mapResourceResult              = FFX_OK;
        uint32_t                 scheduleCalls                  = 0;
        uint32_t                 executeCalls                   = 0;
        uint32_t                 createPipelineCalls            = 0;
        uint32_t                 createComputePipelineCalls     = 0;
        uint32_t                 createGraphicsPipelineCalls    = 0;
        uint32_t                 createOpticalFlowPipelineCalls = 0;
        uint32_t                 nextEffectContextId            = 1;
        FfxLogMessageCallback    backendMessage                 = nullptr;
        uint32_t                 constantData[512]              = {};
        FfxDeviceCapabilities    capabilities                   = {};
        std::vector<FfxResource> resources                      = {FfxResource{}};

        FakeNfruBackend()
        {
            capabilities.maximumSupportedShaderModel       = FFX_SHADER_MODEL_6_6;
            capabilities.fp16Supported                     = true;
            capabilities.tensorSupported                   = true;
            capabilities.computeSupportTensor              = true;
            capabilities.fragmentSupportTensor             = true;
            capabilities.dataGraphSupported                = true;
            capabilities.dataGraphOFSupported              = true;
            capabilities.fragmentStoresAndAtomicsSupported = true;
            capabilities.supportedOutputGridSizes          = FFX_OPTICAL_FLOW_GRID_SIZE_4X4;
            capabilities.dataGraphOFMinWidth               = 1;
            capabilities.dataGraphOFMinHeight              = 1;
            capabilities.dataGraphOFMaxWidth               = 4096;
            capabilities.dataGraphOFMaxHeight              = 4096;
        }

        static FakeNfruBackend& From(FfxInterface* backendInterface)
        {
            return *static_cast<FakeNfruBackend*>(backendInterface->device);
        }

        static bool IsValidResource(const FakeNfruBackend& backend, FfxResourceInternal resource)
        {
            return resource.internalIndex > 0 && static_cast<size_t>(resource.internalIndex) < backend.resources.size();
        }

        static FfxVersionNumber GetSDKVersion(FfxInterface*)
        {
            return FFX_SDK_MAKE_VERSION(1, 1, 1);
        }

        static FfxErrorCode GetEffectGpuMemoryUsage(FfxInterface*, FfxUInt32, FfxEffectMemoryUsage* usage)
        {
            if (usage)
            {
                *usage = {};
            }
            return FFX_OK;
        }

        static FfxErrorCode GetBackendGpuMemoryUsage(FfxInterface*, FfxEffectMemoryUsage* usage)
        {
            if (usage)
            {
                *usage = {};
            }
            return FFX_OK;
        }

        static FfxErrorCode CreateBackendContext(FfxInterface* backendInterface, FfxEffect, FfxEffectBindlessConfig*, FfxUInt32* effectContextId)
        {
            auto& backend = From(backendInterface);
            if (backend.createBackendContextResult != FFX_OK)
            {
                return backend.createBackendContextResult;
            }
            if (!effectContextId)
            {
                return FFX_ERROR_INVALID_POINTER;
            }
            *effectContextId = backend.nextEffectContextId++;
            return FFX_OK;
        }

        static FfxErrorCode GetDeviceCapabilities(FfxInterface* backendInterface, FfxDeviceCapabilities* capabilities)
        {
            auto& backend = From(backendInterface);
            if (backend.getDeviceCapabilitiesResult != FFX_OK)
            {
                return backend.getDeviceCapabilitiesResult;
            }
            if (!capabilities)
            {
                return FFX_ERROR_INVALID_POINTER;
            }
            *capabilities                        = backend.capabilities;
            backendInterface->deviceCapabilities = backend.capabilities;
            backendInterface->devCapInitialized  = true;
            return FFX_OK;
        }

        static FfxErrorCode DestroyBackendContext(FfxInterface* backendInterface, FfxUInt32)
        {
            return From(backendInterface).destroyBackendContextResult;
        }

        static FfxErrorCode CreateResource(FfxInterface*                       backendInterface,
                                           const FfxCreateResourceDescription* description,
                                           FfxUInt32,
                                           FfxResourceInternal* outResource)
        {
            auto& backend = From(backendInterface);
            if (backend.createResourceResult != FFX_OK)
            {
                return backend.createResourceResult;
            }
            if (!description || !outResource)
            {
                return FFX_ERROR_INVALID_POINTER;
            }

            FfxResource resource{};
            resource.resource    = reinterpret_cast<void*>(static_cast<uintptr_t>(backend.resources.size()));
            resource.description = description->resourceDescription;
            resource.state       = description->initialState;
            if (description->name)
            {
                const size_t nameLength = std::min(std::strlen(description->name), static_cast<size_t>(FFX_RESOURCE_NAME_SIZE - 1));
                std::memcpy(resource.name, description->name, nameLength);
                resource.name[nameLength] = '\0';
            }

            outResource->internalIndex = static_cast<int32_t>(backend.resources.size());
            outResource->alignedWidth  = 0;
            backend.resources.push_back(resource);
            return FFX_OK;
        }

        static FfxErrorCode RegisterResource(FfxInterface* backendInterface, const FfxResource* resource, FfxUInt32, FfxResourceInternal* outResource)
        {
            auto& backend = From(backendInterface);
            if (!resource || !outResource)
            {
                return FFX_ERROR_INVALID_POINTER;
            }

            outResource->internalIndex = static_cast<int32_t>(backend.resources.size());
            outResource->alignedWidth  = 0;
            backend.resources.push_back(*resource);
            return FFX_OK;
        }

        static FfxResource GetResource(FfxInterface* backendInterface, FfxResourceInternal resource)
        {
            const auto& backend = From(backendInterface);
            return IsValidResource(backend, resource) ? backend.resources[resource.internalIndex] : FfxResource{};
        }

        static FfxErrorCode UnregisterResources(FfxInterface*, FfxCommandList, FfxUInt32)
        {
            return FFX_OK;
        }

        static FfxErrorCode RegisterStaticResource(FfxInterface*, const FfxStaticResourceDescription*, FfxUInt32)
        {
            return FFX_OK;
        }

        static FfxResourceDescription GetResourceDescription(FfxInterface* backendInterface, FfxResourceInternal resource)
        {
            const auto& backend = From(backendInterface);
            return IsValidResource(backend, resource) ? backend.resources[resource.internalIndex].description : FfxResourceDescription{};
        }

        static FfxErrorCode DestroyResource(FfxInterface* backendInterface, FfxResourceInternal, FfxUInt32)
        {
            return From(backendInterface).destroyResourceResult;
        }

        static FfxErrorCode MapResource(FfxInterface* backendInterface, FfxResourceInternal, void** data)
        {
            if (data)
            {
                *data = nullptr;
            }
            return From(backendInterface).mapResourceResult;
        }

        static FfxErrorCode UnmapResource(FfxInterface*, FfxResourceInternal)
        {
            return FFX_OK;
        }

        static FfxErrorCode StageConstantBufferData(FfxInterface* backendInterface, void* data, FfxUInt32 size, FfxConstantBuffer* constantBuffer)
        {
            auto& backend = From(backendInterface);
            if (!constantBuffer || (size > 0 && !data))
            {
                return FFX_ERROR_INVALID_POINTER;
            }

            const size_t bytesToCopy = std::min<size_t>(size, sizeof(backend.constantData));
            std::memset(backend.constantData, 0, sizeof(backend.constantData));
            if (bytesToCopy > 0)
            {
                std::memcpy(backend.constantData, data, bytesToCopy);
            }
            constantBuffer->data            = backend.constantData;
            constantBuffer->num32BitEntries = static_cast<uint32_t>((bytesToCopy + sizeof(uint32_t) - 1) / sizeof(uint32_t));
            return FFX_OK;
        }

        static FfxErrorCode CreatePipeline(
            FfxInterface* backendInterface, FfxEffect, FfxPass, uint32_t, const FfxPipelineDescription*, FfxUInt32, FfxPipelineState* outPipeline)
        {
            auto& backend = From(backendInterface);
            ++backend.createPipelineCalls;
            if (backend.createPipelineResult != FFX_OK)
            {
                return backend.createPipelineResult;
            }
            if (!outPipeline)
            {
                return FFX_ERROR_INVALID_POINTER;
            }
            *outPipeline          = {};
            outPipeline->pipeline = reinterpret_cast<FfxPipeline>(static_cast<uintptr_t>(1));
            return FFX_OK;
        }

        static FfxErrorCode CreateComputePipeline(FfxInterface*                 backendInterface,
                                                  FfxEffect                     effect,
                                                  FfxPass                       pass,
                                                  uint32_t                      permutationOptions,
                                                  const FfxPipelineDescription* pipelineDescription,
                                                  FfxUInt32                     effectContextId,
                                                  FfxPipelineState*             outPipeline)
        {
            ++From(backendInterface).createComputePipelineCalls;
            return CreatePipeline(backendInterface, effect, pass, permutationOptions, pipelineDescription, effectContextId, outPipeline);
        }

        static FfxErrorCode CreateGraphicsPipeline(FfxInterface*                 backendInterface,
                                                   FfxEffect                     effect,
                                                   FfxPass                       pass,
                                                   uint32_t                      permutationOptions,
                                                   const FfxPipelineDescription* pipelineDescription,
                                                   FfxUInt32                     effectContextId,
                                                   FfxPipelineState*             outPipeline)
        {
            ++From(backendInterface).createGraphicsPipelineCalls;
            return CreatePipeline(backendInterface, effect, pass, permutationOptions, pipelineDescription, effectContextId, outPipeline);
        }

        static FfxErrorCode CreateDataGraphPipeline(FfxInterface*                 backendInterface,
                                                    FfxEffect                     effect,
                                                    FfxPass                       pass,
                                                    uint32_t                      permutationOptions,
                                                    const FfxPipelineDescription* pipelineDescription,
                                                    FfxUInt32                     effectContextId,
                                                    FfxUInt32,
                                                    FfxUInt32,
                                                    FfxPipelineState* outPipeline)
        {
            return CreatePipeline(backendInterface, effect, pass, permutationOptions, pipelineDescription, effectContextId, outPipeline);
        }

        static FfxErrorCode CreateOpticalFlowPipeline(
            FfxInterface* backendInterface, const char*, const FfxOpticalFlowDescription&, FfxUInt32, FfxPipelineState* outPipeline)
        {
            ++From(backendInterface).createOpticalFlowPipelineCalls;
            return CreatePipeline(backendInterface, static_cast<FfxEffect>(0), static_cast<FfxPass>(0), 0, nullptr, 0, outPipeline);
        }

        static FfxErrorCode DestroyPipeline(FfxInterface* backendInterface, FfxPipelineState* pipeline, FfxUInt32)
        {
            if (pipeline)
            {
                *pipeline = {};
            }
            return From(backendInterface).destroyPipelineResult;
        }

        static FfxErrorCode ScheduleGpuJob(FfxInterface* backendInterface, const FfxGpuJobDescription*)
        {
            auto& backend = From(backendInterface);
            ++backend.scheduleCalls;
            return backend.scheduleResult;
        }

        static FfxErrorCode ExecuteGpuJobs(FfxInterface* backendInterface, FfxCommandList, FfxUInt32)
        {
            auto& backend = From(backendInterface);
            ++backend.executeCalls;
            return backend.executeResult;
        }

        static FfxErrorCode SetMessageCallback(FfxInterface* backendInterface, FfxLogMessageCallback callback)
        {
            auto& backend          = From(backendInterface);
            backend.backendMessage = callback;
            return backend.setMessageCallbackResult;
        }

        static void RegisterConstantBufferAllocator(FfxInterface*, FfxConstantBufferAllocator)
        {
        }

        static void BackendMessage(FfxMsgType, const char*)
        {
        }

        FfxInterface MakeInterface()
        {
            FfxInterface backendInterface{};
            backendInterface.fpSetMessageCallback              = &SetMessageCallback;
            backendInterface.fpGetSDKVersion                   = &GetSDKVersion;
            backendInterface.fpGetEffectGpuMemoryUsage         = &GetEffectGpuMemoryUsage;
            backendInterface.fpGetBackendGpuMemoryUsage        = &GetBackendGpuMemoryUsage;
            backendInterface.fpCreateBackendContext            = &CreateBackendContext;
            backendInterface.fpGetDeviceCapabilities           = &GetDeviceCapabilities;
            backendInterface.fpDestroyBackendContext           = &DestroyBackendContext;
            backendInterface.fpCreateResource                  = &CreateResource;
            backendInterface.fpRegisterResource                = &RegisterResource;
            backendInterface.fpGetResource                     = &GetResource;
            backendInterface.fpUnregisterResources             = &UnregisterResources;
            backendInterface.fpRegisterStaticResource          = &RegisterStaticResource;
            backendInterface.fpGetResourceDescription          = &GetResourceDescription;
            backendInterface.fpDestroyResource                 = &DestroyResource;
            backendInterface.fpMapResource                     = &MapResource;
            backendInterface.fpUnmapResource                   = &UnmapResource;
            backendInterface.fpStageConstantBufferDataFunc     = &StageConstantBufferData;
            backendInterface.fpCreateComputePipeline           = &CreateComputePipeline;
            backendInterface.fpCreateGraphicsPipeline          = &CreateGraphicsPipeline;
            backendInterface.fpCreateDataGraphPipeline         = &CreateDataGraphPipeline;
            backendInterface.fpCreateOpticalFlowPipeline       = &CreateOpticalFlowPipeline;
            backendInterface.fpDestroyPipeline                 = &DestroyPipeline;
            backendInterface.fpScheduleGpuJob                  = &ScheduleGpuJob;
            backendInterface.fpExecuteGpuJobs                  = &ExecuteGpuJobs;
            backendInterface.fpRegisterConstantBufferAllocator = &RegisterConstantBufferAllocator;
            backendInterface.fpMessage                         = &BackendMessage;
            backendInterface.device                            = this;
            return backendInterface;
        }

        ffxCreateBackendTestDesc MakeBackendDesc()
        {
            ffxCreateBackendTestDesc backendDesc{};
            backendDesc.header.type      = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_TEST;
            backendDesc.header.pNext     = nullptr;
            backendDesc.backendInterface = MakeInterface();
            return backendDesc;
        }
    };

    ffxReturnCode_t CreateFakeNfruContext(FakeNfruBackend& backend,
                                          const uint32_t   flags,
                                          ffxContext&      outContext,
                                          const uint32_t   backBufferFormat = FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM)
    {
        auto backendDesc        = backend.MakeBackendDesc();
        auto createDesc         = MakeFgCreateDesc(192, 192, 96, 96, backBufferFormat);
        createDesc.header.pNext = reinterpret_cast<ffxApiHeader*>(&backendDesc.header);
        createDesc.flags        = flags;
        createDesc.fpMessage    = FakeNfruApiMessageSink;
        outContext              = nullptr;
        return ffxCreateContext(&outContext, &createDesc.header, nullptr);
    }

    FfxApiResource MakeFakeApiResource(const uint32_t width, const uint32_t height, const uint32_t format, const uint32_t usage, const uint32_t state)
    {
        FfxApiResource resource{};
        resource.resource             = reinterpret_cast<void*>(static_cast<uintptr_t>(1));
        resource.description.type     = FFX_API_RESOURCE_TYPE_TEXTURE2D;
        resource.description.format   = format;
        resource.description.width    = width;
        resource.description.height   = height;
        resource.description.depth    = 1;
        resource.description.mipCount = 1;
        resource.description.flags    = FFX_API_RESOURCE_FLAGS_NONE;
        resource.description.usage    = usage;
        resource.state                = state;
        return resource;
    }

    FfxResource MakeFakeOpticalFlowResource(uintptr_t& nextResourceHandle, FfxDimensions2D size, FfxSurfaceFormat format, FfxResourceUsage usage)
    {
        FfxResource resource{};
        resource.resource             = reinterpret_cast<void*>(nextResourceHandle++);
        resource.description.type     = FFX_RESOURCE_TYPE_TEXTURE2D;
        resource.description.format   = format;
        resource.description.width    = size.width;
        resource.description.height   = size.height;
        resource.description.depth    = 1;
        resource.description.mipCount = 1;
        resource.description.flags    = FFX_RESOURCE_FLAGS_NONE;
        resource.description.usage    = usage;
        resource.state                = FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ;
        return resource;
    }

    ffxApiDispatchDescFrameGenerationPrepare MakeFakePrepareDesc(const uint64_t frameId, const bool includeDepthHistory)
    {
        ffxApiDispatchDescFrameGenerationPrepare prepareDesc{};
        prepareDesc.header.type             = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
        prepareDesc.frameID                 = frameId;
        prepareDesc.commandList             = reinterpret_cast<void*>(static_cast<uintptr_t>(1));
        prepareDesc.motionVectorScale       = {96.0f, 96.0f};
        prepareDesc.frameTimeDelta          = 16.6f;
        prepareDesc.cameraNear              = 0.1f;
        prepareDesc.cameraFar               = 1000.0f;
        prepareDesc.cameraFovAngleVertical  = 1.2f;
        prepareDesc.viewSpaceToMetersFactor = 1.0f;
        prepareDesc.depth =
            MakeFakeApiResource(96, 96, FFX_API_SURFACE_FORMAT_R32_FLOAT, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        prepareDesc.motionVectors =
            MakeFakeApiResource(96, 96, FFX_API_SURFACE_FORMAT_R16G16_FLOAT, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        if (includeDepthHistory)
        {
            prepareDesc.depthTm1 =
                MakeFakeApiResource(96, 96, FFX_API_SURFACE_FORMAT_R32_FLOAT, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        }
        unit_test::vk::SetIdentityMatrix(prepareDesc.viewProjection);
        return prepareDesc;
    }

    ffxApiDispatchDescFrameGeneration MakeFakeDispatchDesc(const uint64_t frameId)
    {
        ffxApiDispatchDescFrameGeneration dispatchDesc{};
        dispatchDesc.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
        dispatchDesc.commandList = reinterpret_cast<void*>(static_cast<uintptr_t>(1));
        dispatchDesc.presentColor =
            MakeFakeApiResource(192, 192, FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        dispatchDesc.outputs[0] =
            MakeFakeApiResource(192, 192, FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM, FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatchDesc.numGeneratedFrames = 1;
        dispatchDesc.reset              = true;
        dispatchDesc.frameID            = frameId;
        return dispatchDesc;
    }

    ffxReturnCode_t FakePrepare(ffxContext& context, const uint64_t frameId, const bool includeDepthHistory)
    {
        auto prepareDesc = MakeFakePrepareDesc(frameId, includeDepthHistory);
        return ffxDispatch(&context, &prepareDesc.header);
    }

    ffxReturnCode_t FakeDispatch(ffxContext& context, const uint64_t frameId)
    {
        auto dispatchDesc = MakeFakeDispatchDesc(frameId);
        return ffxDispatch(&context, &dispatchDesc.header);
    }

}  // namespace

UNIT_TEST_CASE(nfru_provider_fake_backend, rejects_unmatched_and_duplicate_frames)
{
    std::vector<std::string> messages;
    struct MessageCaptureGuard
    {
        explicit MessageCaptureGuard(std::vector<std::string>* captured)
        {
            g_fakeNfruMessages = captured;
        }
        ~MessageCaptureGuard()
        {
            g_fakeNfruMessages = nullptr;
        }
    } messageCapture(&messages);

    FakeNfruBackend backend;
    struct ContextGuard
    {
        ffxContext context = nullptr;
        ~ContextGuard()
        {
            DestroyContextIfNeeded(context);
        }
    } contextGuard;

    const uint32_t flags = FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH | FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR;
    UNIT_TEST_CHECK(CreateFakeNfruContext(backend, flags, contextGuard.context) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(ConfigureNfruContext(contextGuard.context, true, FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY, 1) == FFX_API_RETURN_OK);

    UNIT_TEST_CHECK(FakeDispatch(contextGuard.context, 1) == FFX_API_RETURN_ERROR_PARAMETER);
    UNIT_TEST_CHECK(FakePrepare(contextGuard.context, 1, true) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(FakePrepare(contextGuard.context, 2, true) == FFX_API_RETURN_ERROR_PARAMETER);
    UNIT_TEST_CHECK(FakeDispatch(contextGuard.context, 2) == FFX_API_RETURN_ERROR_PARAMETER);
    UNIT_TEST_CHECK(backend.scheduleCalls == 0);
    return true;
}

UNIT_TEST_CASE(nfru_provider_fake_backend, retries_after_prepare_validation)
{
    FakeNfruBackend backend;
    struct ContextGuard
    {
        ffxContext context = nullptr;
        ~ContextGuard()
        {
            DestroyContextIfNeeded(context);
        }
    } contextGuard;

    UNIT_TEST_CHECK(CreateFakeNfruContext(backend, FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR, contextGuard.context) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(ConfigureNfruContext(contextGuard.context, true, FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY, 1) == FFX_API_RETURN_OK);

    UNIT_TEST_CHECK(FakePrepare(contextGuard.context, 1, false) == FFX_API_RETURN_ERROR_PARAMETER);
    UNIT_TEST_CHECK(FakePrepare(contextGuard.context, 1, true) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(FakeDispatch(contextGuard.context, 1) == FFX_API_RETURN_OK);
    return true;
}

UNIT_TEST_CASE(nfru_provider_fake_backend, maps_invalid_prepare_arguments_to_parameter_error)
{
    FakeNfruBackend backend;
    struct ContextGuard
    {
        ffxContext context = nullptr;
        ~ContextGuard()
        {
            DestroyContextIfNeeded(context);
        }
    } contextGuard;

    const uint32_t flags = FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH | FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR;
    UNIT_TEST_CHECK(CreateFakeNfruContext(backend, flags, contextGuard.context) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(ConfigureNfruContext(contextGuard.context, true, FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY, 1) == FFX_API_RETURN_OK);

    auto prepareDesc                          = MakeFakePrepareDesc(1, true);
    prepareDesc.mvSimilarityRelativeThreshold = std::numeric_limits<float>::infinity();
    const size_t   resourceCount              = backend.resources.size();
    const uint32_t scheduleCalls              = backend.scheduleCalls;

    UNIT_TEST_CHECK(ffxDispatch(&contextGuard.context, &prepareDesc.header) == FFX_API_RETURN_ERROR_PARAMETER);
    UNIT_TEST_CHECK(backend.resources.size() == resourceCount);
    UNIT_TEST_CHECK(backend.scheduleCalls == scheduleCalls);
    return true;
}

UNIT_TEST_CASE(nfru_provider_fake_backend, reports_partial_cleanup_failure)
{
    std::vector<std::string> messages;
    struct MessageCaptureGuard
    {
        explicit MessageCaptureGuard(std::vector<std::string>* captured)
        {
            g_fakeNfruMessages = captured;
        }
        ~MessageCaptureGuard()
        {
            g_fakeNfruMessages = nullptr;
        }
    } messageCapture(&messages);

    FakeNfruBackend backend;
    backend.createResourceResult        = FFX_ERROR_BACKEND_API_ERROR;
    backend.destroyBackendContextResult = FFX_ERROR_BACKEND_API_ERROR;
    ffxContext context                  = nullptr;

    UNIT_TEST_CHECK(CreateFakeNfruContext(backend, FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH | FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR, context) ==
                    FFX_API_RETURN_ERROR_RUNTIME_ERROR);
    UNIT_TEST_CHECK(context == nullptr);
    UNIT_TEST_CHECK(HasFakeNfruMessage(messages, "event=context_destroy result=failed"));
    UNIT_TEST_CHECK(!HasFakeNfruMessage(messages, "event=context_create"));
    UNIT_TEST_CHECK(!HasFakeNfruMessage(messages, "event=context_pipeline"));
    return true;
}

UNIT_TEST_CASE(nfru_provider_fake_backend, rejects_after_schedule_failure)
{
    FakeNfruBackend backend;
    struct ContextGuard
    {
        ffxContext context = nullptr;
        ~ContextGuard()
        {
            DestroyContextIfNeeded(context);
        }
    } contextGuard;

    const uint32_t flags = FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH | FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR;
    UNIT_TEST_CHECK(CreateFakeNfruContext(backend, flags, contextGuard.context) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(ConfigureNfruContext(contextGuard.context, true, FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY, 1) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(FakePrepare(contextGuard.context, 1, true) == FFX_API_RETURN_OK);

    backend.scheduleResult = FFX_ERROR_BACKEND_API_ERROR;
    UNIT_TEST_CHECK(FakeDispatch(contextGuard.context, 1) == FFX_API_RETURN_ERROR_RUNTIME_ERROR);
    backend.scheduleResult = FFX_OK;
    UNIT_TEST_CHECK(FakePrepare(contextGuard.context, 2, true) == FFX_API_RETURN_ERROR_RUNTIME_ERROR);
    return true;
}

UNIT_TEST_CASE(nfru_provider_fake_backend, rejects_after_execute_failure)
{
    FakeNfruBackend backend;
    struct ContextGuard
    {
        ffxContext context = nullptr;
        ~ContextGuard()
        {
            DestroyContextIfNeeded(context);
        }
    } contextGuard;

    const uint32_t flags = FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH | FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR;
    UNIT_TEST_CHECK(CreateFakeNfruContext(backend, flags, contextGuard.context) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(ConfigureNfruContext(contextGuard.context, true, FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY, 1) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(FakePrepare(contextGuard.context, 1, true) == FFX_API_RETURN_OK);

    backend.executeResult = FFX_ERROR_BACKEND_API_ERROR;
    UNIT_TEST_CHECK(FakeDispatch(contextGuard.context, 1) == FFX_API_RETURN_ERROR_RUNTIME_ERROR);
    backend.executeResult = FFX_OK;
    UNIT_TEST_CHECK(FakePrepare(contextGuard.context, 2, true) == FFX_API_RETURN_ERROR_RUNTIME_ERROR);
    return true;
}

UNIT_TEST_CASE(nfru_provider_fake_backend, emits_verbose_callback_records)
{
    std::vector<std::string> messages;
    struct MessageCaptureGuard
    {
        explicit MessageCaptureGuard(std::vector<std::string>* captured)
        {
            g_fakeNfruMessages = captured;
        }
        ~MessageCaptureGuard()
        {
            g_fakeNfruMessages = nullptr;
        }
    } messageCapture(&messages);

    const struct
    {
        uint32_t               backBufferFormat;
        FfxOpticalFlowGridSize opticalFlowGrid;
        bool                   managedColor;
        const char*            historyFormatName;
    } cases[] = {
        {FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM, FFX_OPTICAL_FLOW_GRID_SIZE_4X4, true, "R8G8B8A8_UNORM"},
        {FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM, FFX_OPTICAL_FLOW_GRID_SIZE_2X2, true, "B8G8R8A8_UNORM"},
        {FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM, FFX_OPTICAL_FLOW_GRID_SIZE_4X4, false, "R11G11B10_FLOAT"},
    };

    for (const auto& testCase : cases)
    {
        FakeNfruBackend backend;
        backend.capabilities.supportedOutputGridSizes = testCase.opticalFlowGrid;
        struct ContextGuard
        {
            ffxContext context = nullptr;
            ~ContextGuard()
            {
                DestroyContextIfNeeded(context);
            }
        } contextGuard;

        const uint32_t flags      = FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH | (testCase.managedColor ? FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR : 0);
        const uint32_t quietFlags = FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY;
        const uint32_t debugFlags = quietFlags | FFX_API_FG_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING;
        messages.clear();
        UNIT_TEST_CHECK(CreateFakeNfruContext(backend, flags, contextGuard.context, testCase.backBufferFormat) == FFX_API_RETURN_OK);
        UNIT_TEST_CHECK(HasFakeNfruMessage(messages, "event=context_create"));
        UNIT_TEST_CHECK(HasFakeNfruMessage(messages, (" format_id=" + std::to_string(testCase.backBufferFormat)).c_str()));
        const std::string grid = std::to_string(static_cast<unsigned>(testCase.opticalFlowGrid));
        UNIT_TEST_CHECK(HasFakeNfruMessage(messages, ("of_grid=" + grid + "x" + grid).c_str()));
        UNIT_TEST_CHECK(HasFakeNfruMessage(messages, "event=versions nfru_component_version="));
        UNIT_TEST_CHECK(!HasFakeNfruMessage(messages, "nfru_model_version"));
        UNIT_TEST_CHECK(!HasFakeNfruMessage(messages, "nfru_pipeline_version"));

        auto prepareDesc         = MakeFakePrepareDesc(1, true);
        prepareDesc.jitterOffset = {0.75f, -0.75f};
        prepareDesc.colorTm1 =
            MakeFakeApiResource(192, 192, FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        auto dispatchDesc                            = MakeFakeDispatchDesc(1);
        dispatchDesc.presentColor.description.format = testCase.backBufferFormat;
        dispatchDesc.outputs[0].description.format   = testCase.backBufferFormat;
        UNIT_TEST_CHECK(ConfigureNfruContext(contextGuard.context, true, quietFlags, 1) == FFX_API_RETURN_OK);
        messages.clear();
        UNIT_TEST_CHECK(ffxDispatch(&contextGuard.context, &prepareDesc.header) == FFX_API_RETURN_OK);
        UNIT_TEST_CHECK(ffxDispatch(&contextGuard.context, &dispatchDesc.header) == FFX_API_RETURN_OK);
        UNIT_TEST_CHECK(messages.empty());

        UNIT_TEST_CHECK(ConfigureNfruContext(contextGuard.context, true, debugFlags, 2) == FFX_API_RETURN_OK);
        messages.clear();
        prepareDesc.frameID  = 2;
        dispatchDesc.frameID = 2;
        UNIT_TEST_CHECK(ffxDispatch(&contextGuard.context, &prepareDesc.header) == FFX_API_RETURN_OK);
        UNIT_TEST_CHECK(ffxDispatch(&contextGuard.context, &dispatchDesc.header) == FFX_API_RETURN_OK);
        UNIT_TEST_CHECK(backend.executeCalls > 0);
        UNIT_TEST_CHECK(HasFakeNfruMessage(messages, "event=prepare_begin"));
        UNIT_TEST_CHECK(HasFakeNfruMessage(messages, "jitterOffset"));
        UNIT_TEST_CHECK(HasFakeNfruMessage(messages, "event=prepare_end"));
        UNIT_TEST_CHECK(HasFakeNfruMessage(messages, "event=dispatch_begin"));
        UNIT_TEST_CHECK(HasFakeNfruMessage(messages, "event=dispatch_end"));
        UNIT_TEST_CHECK(HasFakeNfruMessage(messages, (std::string("color=192x192/") + testCase.historyFormatName).c_str()));
        UNIT_TEST_CHECK(HasFakeNfruMessage(messages, testCase.managedColor ? "color_source=managed" : "color_source=application"));
        UNIT_TEST_CHECK(ffxDestroyContext(&contextGuard.context, nullptr) == FFX_API_RETURN_OK);
    }
    return true;
}

UNIT_TEST_CASE(nfru_provider_fake_backend, reports_fp32_fallback_in_context_diagnostics)
{
    std::vector<std::string> messages;
    struct MessageCaptureGuard
    {
        explicit MessageCaptureGuard(std::vector<std::string>* captured)
        {
            g_fakeNfruMessages = captured;
        }
        ~MessageCaptureGuard()
        {
            g_fakeNfruMessages = nullptr;
        }
    } messageCapture(&messages);

    FakeNfruBackend backend;
    backend.capabilities.fp16Supported = false;
    struct ContextGuard
    {
        ffxContext context = nullptr;
        ~ContextGuard()
        {
            DestroyContextIfNeeded(context);
        }
    } contextGuard;

    const uint32_t flags = FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH | FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR;
    UNIT_TEST_CHECK(CreateFakeNfruContext(backend, flags, contextGuard.context) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(HasFakeNfruMessage(messages, "precision=fp32"));
    UNIT_TEST_CHECK(HasFakeNfruMessage(messages, "fp16_fallback=required_features_unsupported"));
    UNIT_TEST_CHECK(ffxDestroyContext(&contextGuard.context, nullptr) == FFX_API_RETURN_OK);
    return true;
}

namespace
{
    // Project a view-space point (negative Z) to UV with a simplified perspective camera.
    // Inputs: point in view units, sample-position jitter in render pixels, and nonzero render dimensions.
    // Return UV = unjitteredUV - jitter / render; this geometric reference is independent of SDK cancellation constants.
    std::array<float, 2> ProjectJitteredPoint(const std::array<float, 3>& point, FfxFloatCoords2D jitter, FfxDimensions2D render)
    {
        // Apply sample-position jitter in perspective clip space before viewport conversion.
        const float clipW = -point[2];
        const float clipX = point[0] + (2.0f * jitter.x / render.width) * point[2];
        const float clipY = point[1] + (2.0f * jitter.y / render.height) * point[2];
        return {(clipX / clipW + 1.0f) * 0.5f, (clipY / clipW + 1.0f) * 0.5f};
    }

    // Host-only fixture: execute real component prepare/dispatch logic but replace GPU work with backend stubs.
    // staged captures the constants that would reach shaders; this fixture cannot validate shader execution.
    struct JitterBackend
    {
        std::unique_ptr<FfxFrameInterpolationContext> context = std::make_unique<FfxFrameInterpolationContext>();
        FfxFrameInterpolationPrepareDescription       prepare{};
        FrameInterpolationConstants                   staged{};
        uint32_t                                      registrations = FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNT;

        // Build private component state from pixel dimensions and component flags, without allocating GPU resources.
        // render defines jitter units; mvDepth/display determine MV normalization and internal resolution lanes.
        JitterBackend(FfxDimensions2D render, FfxDimensions2D mvDepth, FfxDimensions2D display, uint32_t flags)
        {
            // The opaque component context owns the private storage, as in the component API itself.
            auto& state                            = *reinterpret_cast<FfxArmFrameInterpolationContext_Private*>(context.get());
            state.contextDescription.maxRenderSize = render;
            state.contextDescription.mvDepthSize   = mvDepth;
            state.contextDescription.displaySize   = display;
            state.contextDescription.flags         = flags;
            state.resolutionPlan                   = ffxNfruDeriveResolutionPlan(display, mvDepth, FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT);
            state.mvDepthLane                      = state.resolutionPlan.mvDepthLane;
            state.flowLane                         = state.resolutionPlan.flowLane;
            state.enableDataGraphFI                = false;
            state.constants._RenderSize[0]         = static_cast<int32_t>(render.width);
            state.constants._RenderSize[1]         = static_cast<int32_t>(render.height);
            state.constants._DisplaySize[0]        = static_cast<int32_t>(display.width);
            state.constants._DisplaySize[1]        = static_cast<int32_t>(display.height);
            state.lastFrameViewProjection[0] = state.lastFrameViewProjection[5] = state.lastFrameViewProjection[10] = state.lastFrameViewProjection[15] = 1.0f;
            // Synthetic handles and minimal binding counts let dispatch schedule jobs without a real backend.
            for (uint32_t i = 0; i < FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNT; ++i)
            {
                state.srvResources[i].internalIndex = static_cast<int32_t>(i + 1);
                state.uavResources[i]               = state.srvResources[i];
            }
            state.pipelineDownsampleInpaintSeed.uavTextureCount = 1;
            state.pipelineDownsampleInpaintMip.srvTextureCount  = 1;
            state.pipelineDownsampleInpaintMip.uavTextureCount  = 1;

            auto& backend         = state.contextDescription.backendInterface;
            backend.scratchBuffer = this;
            // Map any input resource to a fresh synthetic output handle; no image contents are accessed.
            backend.fpRegisterResource = [](FfxInterface* interface, const FfxResource*, FfxUInt32, FfxResourceInternal* out) -> FfxErrorCode {
                auto& self         = *static_cast<JitterBackend*>(interface->scratchBuffer);
                out->internalIndex = static_cast<int32_t>(++self.registrations);
                return FFX_OK;
            };
            // Copy the input constant-buffer bytes into staged; reject unexpected layouts rather than misreading them.
            backend.fpStageConstantBufferDataFunc = [](FfxInterface* interface, void* data, FfxUInt32 size, FfxConstantBuffer*) -> FfxErrorCode {
                auto& self = *static_cast<JitterBackend*>(interface->scratchBuffer);
                if (size != sizeof(self.staged))
                    return FFX_ERROR_INVALID_ARGUMENT;
                std::memcpy(&self.staged, data, size);
                return FFX_OK;
            };
            // Accept job/command inputs without executing them or producing GPU outputs; only host state is under test.
            backend.fpScheduleGpuJob      = [](FfxInterface*, const FfxGpuJobDescription*) -> FfxErrorCode { return FFX_OK; };
            backend.fpExecuteGpuJobs      = [](FfxInterface*, FfxCommandList, FfxUInt32) -> FfxErrorCode { return FFX_OK; };
            backend.fpUnregisterResources = [](FfxInterface*, FfxCommandList, FfxUInt32) -> FfxErrorCode { return FFX_OK; };

            prepare.commandList            = this;  // Non-null placeholder; stub backend callbacks never dereference it.
            prepare.cameraNear             = 0.1f;
            prepare.cameraFar              = 1000.0f;
            prepare.cameraFovAngleVertical = 1.2f;
            prepare.motionVectorScale      = {1.0f, 1.0f};
            prepare.viewProjection[0] = prepare.viewProjection[5] = prepare.viewProjection[10] = prepare.viewProjection[15] = 1.0f;
            // Non-null resource placeholders satisfy host validation; historical depth uses the resolved internal lane.
            prepare.depth.resource              = this;
            prepare.depth.description.width     = mvDepth.width;
            prepare.depth.description.height    = mvDepth.height;
            prepare.motionVectors               = prepare.depth;
            prepare.depthTm1                    = prepare.depth;
            prepare.depthTm1.description.width  = state.mvDepthLane.width;
            prepare.depthTm1.description.height = state.mvDepthLane.height;
            prepare.depthTm1Next                = prepare.depthTm1;
        }

        // Prepare frameID with sample jitter in render pixels and return the component status.
        // The real prepare path computes the delta against stored jitter, then updates that history.
        FfxErrorCode Prepare(uint64_t frameID, FfxFloatCoords2D jitter)
        {
            prepare.frameID      = frameID;
            prepare.jitterOffset = jitter;
            return ffxFrameInterpolationPrepare(context.get(), &prepare);
        }

        // Dispatch the prepared frameID, optionally requesting reset, and return the component status.
        // The staging callback exposes dispatch-time constants, including any reset/flag cancellation suppression.
        FfxErrorCode Dispatch(uint64_t frameID, bool reset = false)
        {
            FfxFrameInterpolationDispatchDescription dispatch{};
            dispatch.commandList = this;  // Non-null placeholder; stub backend callbacks never dereference it.
            dispatch.frameID     = frameID;
            dispatch.reset       = reset;
            return ffxFrameInterpolationDispatch(context.get(), &dispatch);
        }
    };

}  // namespace

UNIT_TEST_CASE(nfru_api, infinite_depth_uses_near_plane)
{
    const float epsilon = std::numeric_limits<float>::epsilon();

    for (const bool inverted : {false, true})
    {
        const uint32_t flags = FFX_FG_CONTEXT_FLAG_ENABLE_DEPTH_INFINITE | (inverted ? FFX_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED : 0);
        JitterBackend  backend({128, 64}, {64, 32}, {256, 128}, flags);
        backend.prepare.cameraFar = -1.0f;

        UNIT_TEST_CHECK(backend.Prepare(1, {0, 0}) == FFX_OK);
        UNIT_TEST_CHECK(backend.Dispatch(1) == FFX_OK);

        const float expectedDepthC = inverted ? -epsilon : 1.0f + epsilon;
        const float expectedDepthE = inverted ? backend.prepare.cameraNear : -backend.prepare.cameraNear - epsilon;
        UNIT_TEST_CHECK(std::abs(backend.staged._DeviceToViewDepth[0] - expectedDepthC) < 1e-6f);
        UNIT_TEST_CHECK(std::abs(backend.staged._DeviceToViewDepth[1] - expectedDepthE) < 1e-6f);
    }

    return true;
}

// Parameterless host regression: recover unjittered backward UV motion using captured SDK constants.
// Return true only if all resolution, polarity, static/moving-point, and first-frame assertions pass.
UNIT_TEST_CASE(nfru_api, jitter_cancellation_projection_and_history)
{
    constexpr FfxDimensions2D      render{128, 64};
    constexpr FfxFloatCoords2D     previousJitter{-0.375f, 0.25f}, currentJitter{0.375f, -0.25f};
    constexpr std::array<float, 3> currentPoint{0.125f, -0.25f, -2.0f};
    // The first pair has jitter-only motion; the second also has genuine scene motion that must survive cancellation.
    const std::array<std::array<float, 3>, 2> previousPoints = {{currentPoint, {0.25f, -0.125f, -2.0f}}};
    // Change MV/depth resolution independently of render size to catch use of the wrong jitter denominator.
    for (const FfxDimensions2D mvDepth : {FfxDimensions2D{64, 32}, render, FfxDimensions2D{256, 128}})
    {
        for (const bool displayMotion : {false, true})
        {
            // target is the pixel space selected for MV decoding, not necessarily the render-pixel jitter space.
            const FfxDimensions2D display{512, 256};
            const FfxDimensions2D target = displayMotion ? display : mvDepth;
            const float           width = static_cast<float>(target.width), height = static_cast<float>(target.height);
            // Cover pixel/UV storage and axis sign conventions; cancellation itself must not acquire another sign.
            const std::array<FfxFloatCoords2D, 5> scales = {{{1, 1}, {width, height}, {-width, -height}, {width, -height}, {-1, 1}}};
            for (const auto scale : scales)
            {
                const uint32_t flags = FFX_FG_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION |
                                       (displayMotion ? FFX_FG_CONTEXT_FLAG_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS : 0);
                JitterBackend backend(render, mvDepth, display, flags);
                backend.prepare.motionVectorScale = scale;
                // Seed history on the reset frame; its cancellation is zero, but its jitter is needed by frame 2.
                UNIT_TEST_CHECK(backend.Prepare(1, previousJitter) == FFX_OK);
                UNIT_TEST_CHECK(backend.Dispatch(1) == FFX_OK);
                UNIT_TEST_CHECK(backend.staged._Reset && backend.staged._JitterCancellation[0] == 0 && backend.staged._JitterCancellation[1] == 0);
                UNIT_TEST_CHECK(backend.Prepare(2, currentJitter) == FFX_OK);
                UNIT_TEST_CHECK(backend.Dispatch(2) == FFX_OK);
                UNIT_TEST_CHECK(!backend.staged._Reset);
                for (const auto& previousPoint : previousPoints)
                {
                    // Jittered projections generate the input; zero-jitter projections supply the expected true motion.
                    const auto previous          = ProjectJitteredPoint(previousPoint, previousJitter, render);
                    const auto current           = ProjectJitteredPoint(currentPoint, currentJitter, render);
                    const auto referencePrevious = ProjectJitteredPoint(previousPoint, {0, 0}, render);
                    const auto referenceCurrent  = ProjectJitteredPoint(currentPoint, {0, 0}, render);
                    // Encode backward UV motion into stored units by inverting rawMotion * scale / target.
                    const float rawMotion[] = {(previous[0] - current[0]) * width / scale.x, (previous[1] - current[1]) * height / scale.y};
                    for (uint32_t axis = 0; axis < 2; ++axis)
                    {
                        // Use the SDK's captured scale and jitter delta, not independently reconstructed constants.
                        const float corrected = rawMotion[axis] * backend.staged._MotionVectorScale[axis] - backend.staged._JitterCancellation[axis];
                        UNIT_TEST_CHECK(std::abs(corrected - (referencePrevious[axis] - referenceCurrent[axis])) < 1e-7f);
                    }
                }
            }
        }
    }
    return true;
}

// Parameterless host regression: verify feature gating, reset suppression, and jitter-history recovery.
// Return true only if every transition preserves the existing component frame-ID contract.
UNIT_TEST_CASE(nfru_api, jitter_cancellation_reset_and_flag_contract)
{
    for (const bool enabled : {false, true})
    {
        JitterBackend backend({128, 64}, {64, 32}, {256, 128}, enabled ? FFX_FG_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION : 0);
        UNIT_TEST_CHECK(backend.Prepare(1, {-0.375f, 0.25f}) == FFX_OK);
        UNIT_TEST_CHECK(backend.Dispatch(1) == FFX_OK);
        UNIT_TEST_CHECK(backend.staged._Reset && backend.staged._JitterCancellation[0] == 0 && backend.staged._JitterCancellation[1] == 0);
        // The jitter delta is (+0.75, -0.5) render pixels, normalized by (128, 64), not MV/depth size (64, 32).
        UNIT_TEST_CHECK(backend.Prepare(2, {0.375f, -0.25f}) == FFX_OK);
        UNIT_TEST_CHECK(backend.Dispatch(2) == FFX_OK);
        UNIT_TEST_CHECK(!backend.staged._Reset);
        UNIT_TEST_CHECK(backend.staged._JitterCancellation[0] == (enabled ? 0.75f / 128 : 0));
        UNIT_TEST_CHECK(backend.staged._JitterCancellation[1] == (enabled ? -0.5f / 64 : 0));

        // Exercise explicit reset at 3, a gap from 4 to 7, and a decrease from 8 to 4.
        for (const uint64_t frame : {3u, 7u, 4u})
        {
            UNIT_TEST_CHECK(backend.Prepare(frame, {-0.375f, 0.25f}) == FFX_OK);
            UNIT_TEST_CHECK(backend.Dispatch(frame, frame == 3) == FFX_OK);
            UNIT_TEST_CHECK(backend.staged._Reset && backend.staged._JitterCancellation[0] == 0 && backend.staged._JitterCancellation[1] == 0);
            // A reset must suppress the current delta without losing the jitter needed by the next contiguous frame.
            UNIT_TEST_CHECK(backend.Prepare(frame + 1, {0.375f, -0.25f}) == FFX_OK);
            UNIT_TEST_CHECK(backend.Dispatch(frame + 1) == FFX_OK);
            UNIT_TEST_CHECK(!backend.staged._Reset);
            UNIT_TEST_CHECK(backend.staged._JitterCancellation[0] == (enabled ? 0.75f / 128 : 0));
            UNIT_TEST_CHECK(backend.staged._JitterCancellation[1] == (enabled ? -0.5f / 64 : 0));
        }
        // Repeat ID 5 with unchanged jitter: the existing component neither resets nor produces a jitter delta.
        // This preserves behavior, rather than recommending duplicate frame IDs to API callers.
        UNIT_TEST_CHECK(backend.Prepare(5, {0.375f, -0.25f}) == FFX_OK);
        UNIT_TEST_CHECK(backend.Dispatch(5) == FFX_OK);
        UNIT_TEST_CHECK(!backend.staged._Reset && backend.staged._JitterCancellation[0] == 0 && backend.staged._JitterCancellation[1] == 0);
    }
    return true;
}

UNIT_TEST_CASE(nfru_api, relative_similarity_threshold_validates_finite_values)
{
    struct Variant
    {
        const char* name;
        float       value;
        bool        shouldReject;
    };
    const std::array<Variant, 4> variants = {{
        {"NaN", std::numeric_limits<float>::quiet_NaN(), true},
        {"+infinity", std::numeric_limits<float>::infinity(), true},
        {"-infinity", -std::numeric_limits<float>::infinity(), true},
        {"finite-positive", 0.002f, false},
    }};

    std::string failures;
    for (const Variant& variant : variants)
    {
        JitterBackend backend({128, 64}, {64, 32}, {256, 128}, 0);
        backend.prepare.mvSimilarityRelativeThreshold = variant.value;
        const uint32_t     registrationsBefore        = backend.registrations;
        const FfxErrorCode result                     = backend.Prepare(1, {0, 0});
        if ((variant.shouldReject && result != FFX_ERROR_INVALID_ARGUMENT) || (!variant.shouldReject && result != FFX_OK))
        {
            failures += std::string(variant.name) + " returned the wrong status; ";
        }
        if (variant.shouldReject && result == FFX_ERROR_INVALID_ARGUMENT && backend.registrations != registrationsBefore)
        {
            failures += std::string(variant.name) + " registered resources before rejection; ";
        }
        if (!variant.shouldReject && result == FFX_OK && backend.registrations == registrationsBefore)
        {
            failures += std::string(variant.name) + " did not execute the valid prepare path; ";
        }
    }
    if (!failures.empty())
    {
        UNIT_TEST_FAIL(failures);
    }
    return true;
}

UNIT_TEST_CASE(nfru_api, create_destroy_with_flag_variants)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!unit_test::vk::InitializeVulkanContext(vkContext, &errorMessage))
    {
        UNIT_TEST_FAIL(std::string("Vulkan unavailable: ") + errorMessage);
    }

    const uint32_t              baseFlags = 0;
    const std::vector<uint32_t> variants  = {baseFlags,
                                            baseFlags | FFX_API_FG_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION,
                                            baseFlags | FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED | FFX_API_FG_CONTEXT_FLAG_ENABLE_HIGH_DYNAMIC_RANGE,
                                            baseFlags | FFX_API_FG_CONTEXT_FLAG_ALL_STAGES_COMPUTE};

    for (const uint32_t flags : variants)
    {
        ffxContext context      = nullptr;
        const auto createResult = CreateNfruContext(vkContext, flags, 256, 256, 128, 128, context);
        if (createResult == FFX_API_RETURN_ERROR_RUNTIME_ERROR)
        {
            unit_test::vk::DestroyVulkanContext(vkContext);
            UNIT_TEST_FAIL("NFRU context creation is unsupported on this Vulkan device/runtime.");
        }
        UNIT_TEST_CHECK(createResult == FFX_API_RETURN_OK);
        UNIT_TEST_CHECK(ffxDestroyContext(&context, nullptr) == FFX_API_RETURN_OK);
    }

    unit_test::vk::DestroyVulkanContext(vkContext);
    return true;
}

UNIT_TEST_CASE(nfru_api, default_frame_interpolation_path_requires_graphics_callback)
{
    FakeNfruBackend backend;
    backend.createPipelineResult = FFX_ERROR_BACKEND_API_ERROR;

    auto description                                      = FfxFrameInterpolationContextDescription{};
    description.backendInterface                          = backend.MakeInterface();
    description.backendInterface.fpCreateGraphicsPipeline = nullptr;
    description.maxRenderSize                             = {96, 96};
    description.displaySize                               = {96, 96};
    description.opticalFlowSize                           = {24, 24};
    description.opticalFlowGridSize                       = FFX_OPTICAL_FLOW_GRID_SIZE_4X4;
    auto context                                          = std::make_unique<FfxFrameInterpolationContext>();

    UNIT_TEST_CHECK(ffxFrameInterpolationContextCreate(context.get(), &description) == FFX_ERROR_INCOMPLETE_INTERFACE);
    UNIT_TEST_CHECK(backend.createPipelineCalls == 0);

    description.flags = FFX_FG_CONTEXT_FLAG_ALL_STAGES_COMPUTE;
    UNIT_TEST_CHECK(ffxFrameInterpolationContextCreate(context.get(), &description) == FFX_ERROR_BACKEND_API_ERROR);
    UNIT_TEST_CHECK(backend.createPipelineCalls > 0);
    return true;
}

UNIT_TEST_CASE(nfru_api, fragment_atomic_paths_require_device_support)
{
    FakeNfruBackend backend;
    backend.capabilities.fragmentStoresAndAtomicsSupported = false;

    FfxFrameInterpolationContextDescription description{};
    description.backendInterface    = backend.MakeInterface();
    description.maxRenderSize       = {96, 96};
    description.displaySize         = {96, 96};
    description.opticalFlowSize     = {24, 24};
    description.opticalFlowGridSize = FFX_OPTICAL_FLOW_GRID_SIZE_4X4;
    auto context                    = std::make_unique<FfxFrameInterpolationContext>();

    UNIT_TEST_CHECK(ffxFrameInterpolationContextCreate(context.get(), &description) == FFX_ERROR_INVALID_ARGUMENT);
    UNIT_TEST_CHECK(backend.createPipelineCalls == 0);

    description.flags = FFX_FG_CONTEXT_FLAG_WARP_COMPUTE | FFX_FG_CONTEXT_FLAG_PRE_PROCESS_COMPUTE;
    UNIT_TEST_CHECK(ffxFrameInterpolationContextCreate(context.get(), &description) == FFX_OK);
    UNIT_TEST_CHECK(backend.createPipelineCalls > 0);
    UNIT_TEST_CHECK(ffxFrameInterpolationContextDestroy(context.get()) == FFX_OK);
    return true;
}

UNIT_TEST_CASE(nfru_api, optical_flow_motion_hints_default_to_fragment_callback)
{
    FakeNfruBackend                  backend;
    FfxOpticalFlowContextDescription description{};
    description.backendInterface                          = backend.MakeInterface();
    description.backendInterface.fpCreateGraphicsPipeline = nullptr;
    description.flags                                     = FFX_OPTICALFLOW_ENABLE_MV_HINTS;
    auto context                                          = std::make_unique<FfxOpticalFlowContext>();

    UNIT_TEST_CHECK(ffxOpticalFlowContextCreate(context.get(), &description) == FFX_ERROR_INCOMPLETE_INTERFACE);
    UNIT_TEST_CHECK(backend.createPipelineCalls == 0);
    return true;
}

UNIT_TEST_CASE(nfru_api, optical_flow_compute_motion_hints_require_compute_callback)
{
    FakeNfruBackend backend;
    backend.createBackendContextResult = FFX_ERROR_BACKEND_API_ERROR;
    FfxOpticalFlowContextDescription description{};
    description.backendInterface                         = backend.MakeInterface();
    description.backendInterface.fpCreateComputePipeline = nullptr;
    description.flags                                    = FFX_OPTICALFLOW_ENABLE_MV_HINTS | FFX_OPTICALFLOW_ENABLE_MV_HINTS_COMPUTE;
    auto context                                         = std::make_unique<FfxOpticalFlowContext>();

    UNIT_TEST_CHECK(ffxOpticalFlowContextCreate(context.get(), &description) == FFX_ERROR_INCOMPLETE_INTERFACE);
    UNIT_TEST_CHECK(backend.createPipelineCalls == 0);
    return true;
}

UNIT_TEST_CASE(nfru_api, optical_flow_compute_motion_hints_use_compute_callback)
{
    FakeNfruBackend                  backend;
    FfxOpticalFlowContextDescription description{};
    description.backendInterface                          = backend.MakeInterface();
    description.backendInterface.fpCreateGraphicsPipeline = nullptr;
    description.flags                                     = FFX_OPTICALFLOW_ENABLE_MV_HINTS | FFX_OPTICALFLOW_ENABLE_MV_HINTS_COMPUTE;
    description.resolution                                = {96, 96};
    description.maxRenderSize                             = {96, 96};
    description.gridSize                                  = FFX_OPTICAL_FLOW_GRID_SIZE_4X4;
    auto context                                          = std::make_unique<FfxOpticalFlowContext>();

    const FfxErrorCode createResult = ffxOpticalFlowContextCreate(context.get(), &description);
    if (createResult != FFX_OK)
    {
        UNIT_TEST_FAIL("optical-flow context creation failed with the compute callback");
    }
    const bool         computePipelineCreated     = backend.createComputePipelineCalls > 0;
    const bool         graphicsPipelineWasSkipped = backend.createGraphicsPipelineCalls == 0;
    const bool         opticalFlowPipelineCreated = backend.createOpticalFlowPipelineCalls > 0;
    const FfxErrorCode destroyResult              = ffxOpticalFlowContextDestroy(context.get());
    UNIT_TEST_CHECK(computePipelineCreated);
    UNIT_TEST_CHECK(graphicsPipelineWasSkipped);
    UNIT_TEST_CHECK(opticalFlowPipelineCreated);
    UNIT_TEST_CHECK(destroyResult == FFX_OK);
    return true;
}

UNIT_TEST_CASE(nfru_api, optical_flow_rejects_partial_mv_depth_size)
{
    FakeNfruBackend                  backend;
    FfxOpticalFlowContextDescription description{};
    description.backendInterface = backend.MakeInterface();
    description.resolution       = {96, 96};
    description.maxRenderSize    = {96, 96};
    description.mvDepthSize      = {0, 48};
    description.gridSize         = FFX_OPTICAL_FLOW_GRID_SIZE_4X4;
    auto context                 = std::make_unique<FfxOpticalFlowContext>();

    UNIT_TEST_CHECK(ffxOpticalFlowContextCreate(context.get(), &description) == FFX_ERROR_INVALID_ARGUMENT);
    return true;
}

UNIT_TEST_CASE(nfru_api, optical_flow_rejects_unrepresentable_mv_depth_size)
{
    FakeNfruBackend                  backend;
    constexpr uint32_t               firstUnrepresentableDimension = static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) + 1u;
    FfxOpticalFlowContextDescription description{};
    description.backendInterface = backend.MakeInterface();
    description.resolution       = {96, 96};
    description.maxRenderSize    = {96, 96};
    description.mvDepthSize      = {firstUnrepresentableDimension, firstUnrepresentableDimension};
    description.gridSize         = FFX_OPTICAL_FLOW_GRID_SIZE_4X4;
    auto context                 = std::make_unique<FfxOpticalFlowContext>();

    UNIT_TEST_CHECK(ffxOpticalFlowContextCreate(context.get(), &description) == FFX_ERROR_INVALID_ARGUMENT);
    return true;
}

UNIT_TEST_CASE(nfru_api, optical_flow_rejects_unrepresentable_derived_dimensions)
{
    struct Variant
    {
        const char*     name;
        FfxDimensions2D resolution;
        FfxDimensions2D maxRenderSize;
    };
    constexpr uint32_t           firstUnrepresentableShaderDimension = static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) + 1u;
    const std::array<Variant, 5> variants                            = {{
        {"resolution width exceeds signed shader range", {firstUnrepresentableShaderDimension, 1}, {128, 540}},
        {"resolution height exceeds signed shader range", {96, firstUnrepresentableShaderDimension}, {96, 96}},
        {"maxRenderSize width exceeds signed shader range", {96, 96}, {firstUnrepresentableShaderDimension, 540}},
        {"maxRenderSize height exceeds signed shader range", {96, 96}, {128, firstUnrepresentableShaderDimension}},
        {"derived lane width exceeds uint32 range", {static_cast<uint32_t>(std::numeric_limits<int32_t>::max()), 1}, {128, 540}},
    }};

    for (const Variant& variant : variants)
    {
        FakeNfruBackend                  backend;
        FfxOpticalFlowContextDescription description{};
        description.backendInterface = backend.MakeInterface();
        description.resolution       = variant.resolution;
        description.maxRenderSize    = variant.maxRenderSize;
        description.gridSize         = FFX_OPTICAL_FLOW_GRID_SIZE_4X4;
        auto context                 = std::make_unique<FfxOpticalFlowContext>();

        const FfxErrorCode result = ffxOpticalFlowContextCreate(context.get(), &description);
        if (result != FFX_ERROR_INVALID_ARGUMENT)
        {
            UNIT_TEST_FAIL(std::string(variant.name) + " was not rejected");
        }
    }
    return true;
}

UNIT_TEST_CASE(nfru_api, optical_flow_shared_depth_history_uses_resolved_mv_depth_size)
{
    struct Variant
    {
        const char*     name;
        FfxDimensions2D mvDepthSize;
        FfxDimensions2D expectedLane;
    };
    const FfxDimensions2D        renderSize{1920, 1080};
    const FfxDimensions2D        displaySize{3840, 2160};
    const FfxDimensions2D        derivedLane = ffxNfruDeriveResolutionPlan(displaySize, renderSize, FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT).mvDepthLane;
    const std::array<Variant, 2> variants    = {{
        {"explicit reduced lane", {960, 540}, {960, 540}},
        {"derived lane", {}, derivedLane},
    }};

    for (const Variant& variant : variants)
    {
        FakeNfruBackend                  backend;
        FfxOpticalFlowContextDescription description{};
        description.backendInterface = backend.MakeInterface();
        description.resolution       = displaySize;
        description.maxRenderSize    = renderSize;
        description.mvDepthSize      = variant.mvDepthSize;
        description.gridSize         = FFX_OPTICAL_FLOW_GRID_SIZE_4X4;
        auto context                 = std::make_unique<FfxOpticalFlowContext>();

        const FfxErrorCode createResult = ffxOpticalFlowContextCreate(context.get(), &description);
        if (createResult != FFX_OK)
        {
            UNIT_TEST_FAIL(std::string("optical-flow context creation failed for ") + variant.name);
        }

        FfxOpticalFlowSharedResourceDescriptions sharedResources{};
        const FfxErrorCode                       descriptionResult = ffxOpticalFlowGetSharedResourceDescriptions(context.get(), &sharedResources);
        const FfxErrorCode                       destroyResult     = ffxOpticalFlowContextDestroy(context.get());
        UNIT_TEST_CHECK(descriptionResult == FFX_OK);
        UNIT_TEST_CHECK(destroyResult == FFX_OK);
        UNIT_TEST_CHECK(sharedResources.depthTm1.resourceDescription.width == variant.expectedLane.width);
        UNIT_TEST_CHECK(sharedResources.depthTm1.resourceDescription.height == variant.expectedLane.height);
        UNIT_TEST_CHECK(sharedResources.depthTm1Next.resourceDescription.width == variant.expectedLane.width);
        UNIT_TEST_CHECK(sharedResources.depthTm1Next.resourceDescription.height == variant.expectedLane.height);
    }
    return true;
}

UNIT_TEST_CASE(nfru_api, optical_flow_hint_depth_resources_validate_input_and_history_sizes)
{
    struct Variant
    {
        const char*     name;
        FfxDimensions2D currentDepthSize;
        FfxDimensions2D previousDepthSize;
    };
    const std::array<Variant, 2> variants = {{
        {"current depth", {960, 544}, {960, 544}},
        {"previous depth", {1920, 1080}, {960, 540}},
    }};
    const FfxDimensions2D        inputSize{1920, 1080};
    const FfxDimensions2D        laneSize{960, 544};

    std::string failures;
    for (const Variant& variant : variants)
    {
        FakeNfruBackend                  backend;
        FfxOpticalFlowContextDescription description{};
        description.backendInterface = backend.MakeInterface();
        description.flags            = FFX_OPTICALFLOW_ENABLE_MV_HINTS | FFX_OPTICALFLOW_ENABLE_MV_HINTS_COMPUTE;
        description.resolution       = inputSize;
        description.maxRenderSize    = inputSize;
        description.mvDepthSize      = laneSize;
        description.backBufferFormat = FFX_SURFACE_FORMAT_R8G8B8A8_UNORM;
        description.gridSize         = FFX_OPTICAL_FLOW_GRID_SIZE_4X4;
        unit_test::vk::SetIdentityMatrix(description.initialViewProjection);
        auto               context      = std::make_unique<FfxOpticalFlowContext>();
        const FfxErrorCode createResult = ffxOpticalFlowContextCreateForFrameGenerationWithDepthInput(context.get(), &description, inputSize);
        if (createResult != FFX_OK)
        {
            UNIT_TEST_FAIL(std::string("optical-flow context creation failed: ") + std::to_string(createResult));
        }

        uintptr_t nextResourceHandle = 1;

        FfxOpticalFlowDispatchDescription dispatch{};
        dispatch.commandList = reinterpret_cast<FfxCommandList>(static_cast<uintptr_t>(1));
        dispatch.color       = MakeFakeOpticalFlowResource(nextResourceHandle, inputSize, FFX_SURFACE_FORMAT_R8G8B8A8_UNORM, FFX_RESOURCE_USAGE_READ_ONLY);
        dispatch.depth = MakeFakeOpticalFlowResource(nextResourceHandle, variant.currentDepthSize, FFX_SURFACE_FORMAT_R32_FLOAT, FFX_RESOURCE_USAGE_READ_ONLY);
        dispatch.depthTm1 =
            MakeFakeOpticalFlowResource(nextResourceHandle, variant.previousDepthSize, FFX_SURFACE_FORMAT_R32_FLOAT, FFX_RESOURCE_USAGE_READ_ONLY);
        dispatch.colorTm1 = MakeFakeOpticalFlowResource(nextResourceHandle, inputSize, FFX_SURFACE_FORMAT_R8G8B8A8_UNORM, FFX_RESOURCE_USAGE_READ_ONLY);
        dispatch.opticalFlowVector =
            MakeFakeOpticalFlowResource(nextResourceHandle, ffxOpticalFlowGetSize(context.get()), FFX_SURFACE_FORMAT_R16G16_FLOAT, FFX_RESOURCE_USAGE_UAV);
        unit_test::vk::SetIdentityMatrix(dispatch.viewProjection);
        const size_t       resourceCount  = backend.resources.size();
        const uint32_t     scheduleCalls  = backend.scheduleCalls;
        const FfxErrorCode dispatchResult = ffxOpticalFlowContextDispatch(context.get(), &dispatch);
        if (dispatchResult != FFX_ERROR_INVALID_ARGUMENT || backend.resources.size() != resourceCount || backend.scheduleCalls != scheduleCalls)
        {
            failures += std::string(variant.name) + " mismatch was not rejected before resource registration and GPU scheduling; ";
        }

        const FfxErrorCode destroyResult = ffxOpticalFlowContextDestroy(context.get());
        if (destroyResult != FFX_OK)
        {
            failures += std::string(variant.name) + " test context failed to destroy; ";
        }
    }
    if (!failures.empty())
    {
        UNIT_TEST_FAIL(failures);
    }
    return true;
}

UNIT_TEST_CASE(nfru_api, optical_flow_framegeneration_depth_input_and_history_lane_may_differ)
{
    struct Variant
    {
        const char*     name;
        FfxDimensions2D historySize;
    };
    const std::array<Variant, 2> variants = {{{"lane-sized history", {960, 544}}, {"input-sized history", {1920, 1080}}}};

    std::string failures;
    for (const Variant& variant : variants)
    {
        FakeNfruBackend                  backend;
        FfxOpticalFlowContextDescription description{};
        description.backendInterface = backend.MakeInterface();
        description.flags            = FFX_OPTICALFLOW_ENABLE_MV_HINTS | FFX_OPTICALFLOW_ENABLE_MV_HINTS_COMPUTE;
        description.resolution       = {1920, 1080};
        description.maxRenderSize    = {1920, 1080};
        description.mvDepthSize      = {960, 544};
        description.backBufferFormat = FFX_SURFACE_FORMAT_R8G8B8A8_UNORM;
        description.gridSize         = FFX_OPTICAL_FLOW_GRID_SIZE_4X4;
        unit_test::vk::SetIdentityMatrix(description.initialViewProjection);
        auto context = std::make_unique<FfxOpticalFlowContext>();

        const FfxErrorCode createResult = ffxOpticalFlowContextCreateForFrameGenerationWithDepthInput(context.get(), &description, {1920, 1080});
        if (createResult != FFX_OK)
        {
            UNIT_TEST_FAIL(std::string("frame-generation optical-flow context creation failed: ") + std::to_string(createResult));
        }

        uintptr_t                         nextResourceHandle = 1;
        FfxOpticalFlowDispatchDescription dispatch{};
        dispatch.commandList = reinterpret_cast<FfxCommandList>(static_cast<uintptr_t>(1));
        dispatch.color       = MakeFakeOpticalFlowResource(nextResourceHandle, {1920, 1080}, FFX_SURFACE_FORMAT_R8G8B8A8_UNORM, FFX_RESOURCE_USAGE_READ_ONLY);
        dispatch.depth       = MakeFakeOpticalFlowResource(nextResourceHandle, {1920, 1080}, FFX_SURFACE_FORMAT_R32_FLOAT, FFX_RESOURCE_USAGE_READ_ONLY);
        dispatch.depthTm1    = MakeFakeOpticalFlowResource(nextResourceHandle, variant.historySize, FFX_SURFACE_FORMAT_R32_FLOAT, FFX_RESOURCE_USAGE_READ_ONLY);
        dispatch.colorTm1    = MakeFakeOpticalFlowResource(nextResourceHandle, {1920, 1080}, FFX_SURFACE_FORMAT_R8G8B8A8_UNORM, FFX_RESOURCE_USAGE_READ_ONLY);
        dispatch.opticalFlowVector =
            MakeFakeOpticalFlowResource(nextResourceHandle, ffxOpticalFlowGetSize(context.get()), FFX_SURFACE_FORMAT_R16G16_FLOAT, FFX_RESOURCE_USAGE_UAV);
        dispatch.reset = true;
        unit_test::vk::SetIdentityMatrix(dispatch.viewProjection);

        const size_t       resourceCount  = backend.resources.size();
        const uint32_t     scheduleCalls  = backend.scheduleCalls;
        const FfxErrorCode dispatchResult = ffxOpticalFlowContextDispatch(context.get(), &dispatch);
        if (dispatchResult != FFX_OK || backend.resources.size() <= resourceCount || backend.scheduleCalls <= scheduleCalls)
        {
            failures += std::string(variant.name) + " was not accepted and scheduled; ";
        }

        const FfxErrorCode destroyResult = ffxOpticalFlowContextDestroy(context.get());
        if (destroyResult != FFX_OK)
        {
            failures += std::string(variant.name) + " context failed to destroy; ";
        }
    }
    if (!failures.empty())
    {
        UNIT_TEST_FAIL(failures);
    }
    return true;
}

UNIT_TEST_CASE(nfru_api, rejected_context_creation_does_not_leak)
{
    struct Case
    {
        const char*                            name;
        ffxApiCreateContextDescFrameGeneration description;
        ffxReturnCode_t                        expectedResult;
    };
    const std::array<Case, 3> cases = {{
        {"invalid dimensions", MakeFgCreateDesc(0, 256, 128, 128, FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM), FFX_API_RETURN_ERROR_PARAMETER},
        {"unsupported format", MakeFgCreateDesc(256, 256, 128, 128, FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT), FFX_API_RETURN_ERROR_PARAMETER},
        {"missing backend", MakeFgCreateDesc(256, 256, 128, 128, FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM), FFX_API_RETURN_ERROR},
    }};

    std::string failures;
    for (auto testCase : cases)
    {
        TrackingAllocator tracker;
        auto              callbacks = tracker.Callbacks();
        ffxContext        context   = nullptr;
        const auto        result    = ffxCreateContext(&context, &testCase.description.header, &callbacks);
        if (result != testCase.expectedResult || context != nullptr || tracker.liveAllocations != 0)
        {
            if (!failures.empty())
            {
                failures += '\n';
            }
            failures += "- ";
            failures += testCase.name;
            failures += " did not preserve the failed-creation cleanup contract";
        }
    }
    if (!failures.empty())
    {
        std::string message = "Failed-creation cleanup cases failed:";
        message += '\n';
        message += failures;
        UNIT_TEST_FAIL(message);
    }
    return true;
}

#if !FFX_CAPTURE_ENABLED
UNIT_TEST_CASE(nfru_api, capture_descriptors_rejected_when_capture_disabled)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!unit_test::vk::InitializeVulkanContext(vkContext, &errorMessage))
    {
        UNIT_TEST_FAIL(std::string("Vulkan unavailable: ") + errorMessage);
    }

    ffxContext context      = nullptr;
    const auto createResult = CreateNfruContext(vkContext, 0, 256, 256, 128, 128, context);
    if (createResult == FFX_API_RETURN_ERROR_RUNTIME_ERROR)
    {
        unit_test::vk::DestroyVulkanContext(vkContext);
        UNIT_TEST_FAIL("NFRU context creation is unsupported on this Vulkan device/runtime.");
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

UNIT_TEST_CASE(nfru_api, create_duplicate_backend_does_not_leak)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!unit_test::vk::InitializeVulkanContext(vkContext, &errorMessage))
    {
        UNIT_TEST_FAIL(std::string("Vulkan unavailable: ") + errorMessage);
    }

    auto duplicateBackendDesc = unit_test::vk::BuildBackendDesc(vkContext);
    auto backendDesc          = unit_test::vk::BuildBackendDesc(vkContext);
    backendDesc.header.pNext  = &duplicateBackendDesc.header;
    auto createDesc           = MakeFgCreateDesc(256, 256, 128, 128, FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM);
    createDesc.header.pNext   = &backendDesc.header;

    TrackingAllocator tracker;
    auto              callbacks    = tracker.Callbacks();
    ffxContext        context      = nullptr;
    const auto        createResult = ffxCreateContext(&context, &createDesc.header, &callbacks);
    if (context)
    {
        ffxDestroyContext(&context, &callbacks);
    }
    unit_test::vk::DestroyVulkanContext(vkContext);

    UNIT_TEST_CHECK(createResult == FFX_API_RETURN_ERROR);
    UNIT_TEST_CHECK(context == nullptr);
    UNIT_TEST_CHECK(tracker.liveAllocations == 0);
    return true;
}

UNIT_TEST_CASE(nfru_api, create_destroy_releases_tracked_allocations)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!unit_test::vk::InitializeVulkanContext(vkContext, &errorMessage))
    {
        UNIT_TEST_FAIL(std::string("Vulkan unavailable: ") + errorMessage);
    }

    VkDeviceContext deviceContext{};
    deviceContext.vkDevice              = vkContext.device;
    deviceContext.vkPhysicalDevice      = vkContext.physicalDevice;
    deviceContext.vkInstance            = vkContext.instance;
    deviceContext.vkDeviceProcAddr      = vkContext.getDeviceProcAddr;
    deviceContext.vkGetInstanceProcAddr = vkContext.getInstanceProcAddr;
    if (!InitVulkanWrapper(deviceContext))
    {
        unit_test::vk::DestroyVulkanContext(vkContext);
        UNIT_TEST_FAIL("InitVulkanWrapper failed");
    }

    auto backendDesc        = unit_test::vk::BuildBackendDesc(vkContext);
    auto createDesc         = MakeFgCreateDesc(256, 256, 128, 128, FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM);
    createDesc.header.pNext = reinterpret_cast<ffxApiHeader*>(&backendDesc);
    createDesc.flags        = FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH | FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR;

    TrackingAllocator tracker;
    auto              callbacks    = tracker.Callbacks();
    ffxContext        context      = nullptr;
    const auto        createResult = ffxCreateContext(&context, &createDesc.header, &callbacks);
    if (createResult == FFX_API_RETURN_ERROR_RUNTIME_ERROR)
    {
        unit_test::vk::DestroyVulkanContext(vkContext);
        UNIT_TEST_FAIL("NFRU context creation is unsupported on this Vulkan device/runtime.");
    }
    UNIT_TEST_CHECK(createResult == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(context != nullptr);
    UNIT_TEST_CHECK(tracker.liveAllocations > 0);
    UNIT_TEST_CHECK(ffxDestroyContext(&context, &callbacks) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(tracker.liveAllocations == 0);

    unit_test::vk::DestroyVulkanContext(vkContext);
    return true;
}

UNIT_TEST_CASE(nfru_api, configure_and_query_gpu_memory_usage)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!unit_test::vk::InitializeVulkanContext(vkContext, &errorMessage))
    {
        UNIT_TEST_FAIL(std::string("Vulkan unavailable: ") + errorMessage);
    }

    const uint32_t createFlags = FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH | FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR;

    ffxContext context      = nullptr;
    const auto createResult = CreateNfruContext(vkContext, createFlags, 256, 256, 128, 128, context);
    if (createResult == FFX_API_RETURN_ERROR_RUNTIME_ERROR)
    {
        unit_test::vk::DestroyVulkanContext(vkContext);
        UNIT_TEST_FAIL("NFRU context creation is unsupported on this Vulkan device/runtime.");
    }
    UNIT_TEST_CHECK(createResult == FFX_API_RETURN_OK);

    UNIT_TEST_CHECK(ConfigureNfruContext(context, true, FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY, 1) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(ConfigureNfruContext(context, false, FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY, 2) == FFX_API_RETURN_OK);

    FfxApiEffectMemoryUsage                         gpuMemoryUsage{};
    ffxApiQueryDescFrameGenerationGetGPUMemoryUsage query{};
    query.header.type                   = FFX_API_QUERY_DESC_TYPE_FG_GPU_MEMORY_USAGE;
    query.header.pNext                  = nullptr;
    query.gpuMemoryUsageFrameGeneration = &gpuMemoryUsage;
    UNIT_TEST_CHECK(ffxQuery(&context, &query.header) == FFX_API_RETURN_OK);

    UNIT_TEST_CHECK(ffxDestroyContext(&context, nullptr) == FFX_API_RETURN_OK);
    unit_test::vk::DestroyVulkanContext(vkContext);
    return true;
}

#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
UNIT_TEST_CASE(nfru_api, query_cpu_heap_memory_usage)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!unit_test::vk::InitializeVulkanContext(vkContext, &errorMessage))
    {
        UNIT_TEST_FAIL(std::string("Vulkan unavailable: ") + errorMessage);
    }

    const uint32_t createFlags = FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH | FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR;

    ffxContext context      = nullptr;
    const auto createResult = CreateNfruContext(vkContext, createFlags, 256, 256, 128, 128, context);
    if (createResult == FFX_API_RETURN_ERROR_RUNTIME_ERROR)
    {
        unit_test::vk::DestroyVulkanContext(vkContext);
        UNIT_TEST_FAIL("NFRU context creation is unsupported on this Vulkan device/runtime.");
    }
    UNIT_TEST_CHECK(createResult == FFX_API_RETURN_OK);

    FfxApiEffectCpuMemoryUsage                      cpuMemoryUsage{};
    ffxApiQueryDescFrameGenerationGetCPUMemoryUsage query{};
    query.header.type                   = FFX_API_QUERY_DESC_TYPE_FG_CPU_MEMORY_USAGE;
    query.header.pNext                  = nullptr;
    query.cpuMemoryUsageFrameGeneration = &cpuMemoryUsage;
    UNIT_TEST_CHECK(ffxQuery(&context, &query.header) == FFX_API_RETURN_OK);
    UNIT_TEST_CHECK(cpuMemoryUsage.currentUsageInBytes <= cpuMemoryUsage.peakUsageInBytes);

    UNIT_TEST_CHECK(ffxDestroyContext(&context, nullptr) == FFX_API_RETURN_OK);
    unit_test::vk::DestroyVulkanContext(vkContext);
    return true;
}
#endif

static bool DispatchTriangleGeneration(FfxApiDimensions2D mvDepthSize               = {},
                                       uint32_t           extraFlags                = 0,
                                       bool               managePreviousDepth       = true,
                                       FfxApiDimensions2D displaySize               = {192, 192},
                                       FfxApiDimensions2D renderSize                = {96, 96},
                                       bool               depthHistoryUsesD32Format = false)
{
    unit_test::vk::VulkanContext vkContext{};
    std::string                  errorMessage;
    if (!unit_test::vk::InitializeVulkanContext(vkContext, &errorMessage))
    {
        UNIT_TEST_FAIL(std::string("Vulkan unavailable: ") + errorMessage);
    }

    const uint32_t kDisplayWidth          = displaySize.width;
    const uint32_t kDisplayHeight         = displaySize.height;
    const uint32_t kRenderWidth           = renderSize.width;
    const uint32_t kRenderHeight          = renderSize.height;
    const uint32_t kMvDepthWidth          = mvDepthSize.width ? mvDepthSize.width : kRenderWidth;
    const uint32_t kMvDepthHeight         = mvDepthSize.height ? mvDepthSize.height : kRenderHeight;
    const bool     postprocessUsesCompute = (extraFlags & FFX_API_FG_CONTEXT_FLAG_POSTPROCESS_COMPUTE) != 0;

    const uint32_t createFlags =
        (managePreviousDepth ? FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH : 0) | FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR | extraFlags;

    ffxContext  context = nullptr;
    std::string failureReason;
    if (CreateNfruContext(vkContext, createFlags, kDisplayWidth, kDisplayHeight, kRenderWidth, kRenderHeight, context, mvDepthSize) != FFX_API_RETURN_OK)
    {
        failureReason = "CreateNfruContext failed";
    }

    bool                 success = false;
    unit_test::vk::Image presentColor{};
    unit_test::vk::Image output{};
    unit_test::vk::Image depth{};
    unit_test::vk::Image motionVectors{};
    unit_test::vk::Image depthTm1{};
    unit_test::vk::Image colorTm1{};

    if (failureReason.empty())
    {
        do
        {
            if (ConfigureNfruContext(context, true, FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY, 1) != FFX_API_RETURN_OK)
            {
                failureReason = "ConfigureNfruContext failed";
                break;
            }

            if (!unit_test::vk::CreateImage2D(vkContext,
                                              kDisplayWidth,
                                              kDisplayHeight,
                                              VK_FORMAT_R8G8B8A8_UNORM,
                                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                              presentColor,
                                              &errorMessage))
            {
                failureReason = "CreateImage2D(presentColor) failed: " + errorMessage;
                break;
            }
            if (!unit_test::vk::CreateImage2D(
                    vkContext,
                    kDisplayWidth,
                    kDisplayHeight,
                    VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    output,
                    &errorMessage))
            {
                failureReason = "CreateImage2D(output) failed: " + errorMessage;
                break;
            }
            if (!unit_test::vk::CreateImage2D(
                    vkContext,
                    kMvDepthWidth,
                    kMvDepthHeight,
                    VK_FORMAT_R32_SFLOAT,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    depth,
                    &errorMessage))
            {
                failureReason = "CreateImage2D(depth) failed: " + errorMessage;
                break;
            }
            if (!unit_test::vk::CreateImage2D(
                    vkContext,
                    kMvDepthWidth,
                    kMvDepthHeight,
                    VK_FORMAT_R16G16_SFLOAT,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    motionVectors,
                    &errorMessage))
            {
                failureReason = "CreateImage2D(motionVectors) failed: " + errorMessage;
                break;
            }
            if (!unit_test::vk::CreateImage2D(
                    vkContext,
                    kMvDepthWidth,
                    kMvDepthHeight,
                    depthHistoryUsesD32Format ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_R32_SFLOAT,
                    depthHistoryUsesD32Format
                        ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                        : VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    depthTm1,
                    &errorMessage))
            {
                failureReason = "CreateImage2D(depthTm1) failed: " + errorMessage;
                break;
            }
            if (!unit_test::vk::CreateImage2D(vkContext,
                                              kRenderWidth,
                                              kRenderHeight,
                                              VK_FORMAT_R8G8B8A8_UNORM,
                                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                              colorTm1,
                                              &errorMessage))
            {
                failureReason = "CreateImage2D(colorTm1) failed: " + errorMessage;
                break;
            }

            if (!unit_test::vk::RenderTriangle(vkContext, presentColor, &errorMessage))
            {
                failureReason = "RenderTriangle(presentColor) failed: " + errorMessage;
                break;
            }
            if (!unit_test::vk::RenderTriangle(vkContext, colorTm1, &errorMessage))
            {
                failureReason = "RenderTriangle(colorTm1) failed: " + errorMessage;
                break;
            }
            if (!unit_test::vk::ClearImage(vkContext, depth, {1.0f, 0.0f, 0.0f, 0.0f}, &errorMessage))
            {
                failureReason = "ClearImage(depth) failed: " + errorMessage;
                break;
            }
            if (!unit_test::vk::ClearImage(vkContext, depthTm1, {1.0f, 0.0f, 0.0f, 0.0f}, &errorMessage))
            {
                failureReason = "ClearImage(depthTm1) failed: " + errorMessage;
                break;
            }
            if (!unit_test::vk::ClearImage(vkContext, motionVectors, {0.0f, 0.0f, 0.0f, 0.0f}, &errorMessage))
            {
                failureReason = "ClearImage(motionVectors) failed: " + errorMessage;
                break;
            }
            if (!unit_test::vk::ClearImage(vkContext, output, {0.0f, 0.0f, 0.0f, 1.0f}, &errorMessage))
            {
                failureReason = "ClearImage(output) failed: " + errorMessage;
                break;
            }

            auto dispatchOneFrame = [&](const uint64_t frameId, const bool reset) -> bool {
                if (ConfigureNfruContext(context, true, FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY, frameId) != FFX_API_RETURN_OK)
                {
                    failureReason = "ConfigureNfruContext failed";
                    return false;
                }

                VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
                if (!unit_test::vk::BeginCommandBuffer(vkContext, commandBuffer, &errorMessage))
                {
                    failureReason = "BeginCommandBuffer failed: " + errorMessage;
                    return false;
                }

                ffxApiDispatchDescFrameGenerationPrepare prepareDesc{};
                prepareDesc.header.type             = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
                prepareDesc.header.pNext            = nullptr;
                prepareDesc.frameID                 = frameId;
                prepareDesc.commandList             = commandBuffer;
                prepareDesc.jitterOffset            = {0.0f, 0.0f};
                prepareDesc.motionVectorScale       = {static_cast<float>(kMvDepthWidth), static_cast<float>(kMvDepthHeight)};
                prepareDesc.frameTimeDelta          = 16.6f;
                prepareDesc.cameraNear              = 0.1f;
                prepareDesc.cameraFar               = 1000.0f;
                prepareDesc.cameraFovAngleVertical  = 1.2f;
                prepareDesc.viewSpaceToMetersFactor = 1.0f;
                prepareDesc.depth = unit_test::vk::MakeApiImageResource(depth, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
                prepareDesc.motionVectors =
                    unit_test::vk::MakeApiImageResource(motionVectors, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
                prepareDesc.depthTm1 =
                    unit_test::vk::MakeApiImageResource(depthTm1, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
                prepareDesc.colorTm1 =
                    unit_test::vk::MakeApiImageResource(colorTm1, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
                unit_test::vk::SetIdentityMatrix(prepareDesc.viewProjection);

                ffxApiDispatchDescFrameGeneration frameGenerationDesc{};
                frameGenerationDesc.header.type  = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
                frameGenerationDesc.header.pNext = nullptr;
                frameGenerationDesc.commandList  = commandBuffer;
                frameGenerationDesc.presentColor =
                    unit_test::vk::MakeApiImageResource(presentColor, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
                frameGenerationDesc.outputs[0] = unit_test::vk::MakeApiImageResource(
                    output,
                    postprocessUsesCompute ? FFX_API_RESOURCE_USAGE_UAV : FFX_API_RESOURCE_USAGE_RENDERTARGET,
                    postprocessUsesCompute ? FFX_API_RESOURCE_STATE_UNORDERED_ACCESS : FFX_API_RESOURCE_STATE_RENDER_TARGET);
                frameGenerationDesc.numGeneratedFrames = 1;
                frameGenerationDesc.reset              = reset;
                frameGenerationDesc.frameID            = frameId;

                const auto prepareResult  = ffxDispatch(&context, &prepareDesc.header);
                const auto dispatchResult = ffxDispatch(&context, &frameGenerationDesc.header);
                const bool submitOk       = unit_test::vk::SubmitCommandBuffer(vkContext, commandBuffer, &errorMessage);
                if (prepareResult != FFX_API_RETURN_OK || dispatchResult != FFX_API_RETURN_OK || !submitOk)
                {
                    failureReason = "ffxDispatch(FrameGeneration) or command submission failed";
                    return false;
                }
                return true;
            };

            if (!dispatchOneFrame(1, true))
            {
                break;
            }
            if (!dispatchOneFrame(2, false) || !dispatchOneFrame(3, false))
            {
                break;
            }

            std::vector<uint8_t> outputRgba;
            if (!unit_test::vk::ReadbackRgba8Image(vkContext, output, outputRgba, &errorMessage))
            {
                failureReason = "ReadbackRgba8Image failed: " + errorMessage;
                break;
            }
            for (size_t i = 0; i + 3 < outputRgba.size(); i += 4)
            {
                const uint8_t red   = outputRgba[i + 0];
                const uint8_t green = outputRgba[i + 1];
                const uint8_t blue  = outputRgba[i + 2];
                if (red > 8 || green > 8 || blue > 8)
                {
                    success = true;
                    break;
                }
            }

            if (!success)
            {
                failureReason = "dispatch finished but output image remained at clear color";
            }
        } while (false);
    }

    unit_test::vk::DestroyImage(vkContext, colorTm1);
    unit_test::vk::DestroyImage(vkContext, depthTm1);
    unit_test::vk::DestroyImage(vkContext, motionVectors);
    unit_test::vk::DestroyImage(vkContext, depth);
    unit_test::vk::DestroyImage(vkContext, output);
    unit_test::vk::DestroyImage(vkContext, presentColor);
    DestroyContextIfNeeded(context);
    unit_test::vk::DestroyVulkanContext(vkContext);
    if (!success)
    {
        UNIT_TEST_FAIL(failureReason.empty() ? "NFRU dispatch test failed" : failureReason);
    }
    return true;
}

UNIT_TEST_CASE(nfru_api, dispatch_triangle_generation_changes_output_pixels)
{
    return DispatchTriangleGeneration();
}

UNIT_TEST_CASE(nfru_api, dispatch_equal_mv_depth_size)
{
    return DispatchTriangleGeneration({96, 96});
}

UNIT_TEST_CASE(nfru_api, dispatch_smaller_mv_depth_with_hints)
{
    return DispatchTriangleGeneration({48, 48}, FFX_API_FG_CONTEXT_FLAG_ENABLE_MV_HINTS);
}

UNIT_TEST_CASE(nfru_api, dispatch_smaller_mv_depth_with_external_history_fragment)
{
    return DispatchTriangleGeneration({48, 48}, FFX_API_FG_CONTEXT_FLAG_ENABLE_MV_HINTS, false);
}

UNIT_TEST_CASE(nfru_api, dispatch_smaller_mv_depth_with_external_history_compute)
{
    return DispatchTriangleGeneration(
        {48, 48}, FFX_API_FG_CONTEXT_FLAG_ENABLE_MV_HINTS | FFX_API_FG_CONTEXT_FLAG_MV_HINTS_COMPUTE | FFX_API_FG_CONTEXT_FLAG_ALL_STAGES_COMPUTE, false);
}

UNIT_TEST_CASE(nfru_api, dispatch_external_d32_depth_history)
{
    return DispatchTriangleGeneration({}, 0, false, {192, 192}, {96, 96}, true);
}

UNIT_TEST_CASE(nfru_api, dispatch_capped_colour_with_pre_downsampled_mv_depth)
{
    // Keep the image narrow to exercise the 2160/1080/540 height thresholds cheaply under emulation.
    return DispatchTriangleGeneration({32, 540}, FFX_API_FG_CONTEXT_FLAG_ENABLE_MV_HINTS, true, {128, 2160}, {64, 1080});
}

UNIT_TEST_CASE(nfru_api, dispatch_capped_colour_with_render_sized_mv_depth)
{
    return DispatchTriangleGeneration({}, FFX_API_FG_CONTEXT_FLAG_ENABLE_MV_HINTS, true, {128, 2160}, {64, 1080});
}

namespace
{
    // Decode one readable four-byte VK_FORMAT_B10G11R11_UFLOAT_PACK32 texel into floating-point RGB (no alpha).
    // Exponent-31 encodings return NaN so comparison assertions reject both infinity and NaN inputs.
    std::array<float, 3> DecodeRgb11(const uint8_t* pixel)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, pixel, sizeof(bits));
        std::array<float, 3> rgb{};
        for (uint32_t channel = 0; channel < 3; ++channel)
        {
            // R/G have 6 mantissa bits, B has 5; all have 5 exponent bits with bias 15.
            const uint32_t mantissaBits = channel == 2 ? 5 : 6;
            const uint32_t value        = (bits >> (channel * 11)) & ((1u << (mantissaBits + 5)) - 1);
            const uint32_t exponent     = value >> mantissaBits;
            const uint32_t mantissa     = value & ((1u << mantissaBits) - 1);
            // Subnormals have no implicit leading one and use exponent -14; normal values restore that leading bit.
            rgb[channel] = exponent == 31 ? std::numeric_limits<float>::quiet_NaN()
                                          : std::ldexp(static_cast<float>(mantissa + (exponent == 0 ? 0u : 1u << mantissaBits)),
                                                       (exponent == 0 ? -14 : static_cast<int>(exponent) - 15) - static_cast<int>(mantissaBits));
        }
        return rgb;
    }

    // Own one real NFRU context and synthetic scene images while borrowing the Vulkan device.
    // Fixed colour/depth and uniform encoded MV isolate cancellation behavior, not full scene-rendering correctness.
    struct JitterGpuScene
    {
        unit_test::vk::VulkanContext& vk;
        ffxContext                    context = nullptr;
        unit_test::vk::Image          color{}, depth{}, motion{}, output{}, previousColor{}, previousDepth{};
        float                         deviceDepth = 0.5f;  // Device-depth value, not linear distance; sky uses the far-plane endpoint.
        std::string                   error;

        // Borrow device, which must outlive the scene; Initialize creates the owned GPU resources.
        explicit JitterGpuScene(unit_test::vk::VulkanContext& device)
            : vk(device)
        {
        }
        // Release context before its input/output images, including resources left by a partially failed Initialize.
        ~JitterGpuScene()
        {
            DestroyContextIfNeeded(context);
            for (auto* image : {&output, &motion, &depth, &color, &previousColor, &previousDepth})
                unit_test::vk::DestroyImage(vk, *image);
        }

        // Initialize a fresh scene from context flags and render/MV-depth/display pixel dimensions.
        // sky chooses far-plane depth instead of ordinary geometry; flags determine depth polarity and MV dimensions.
        // Return true with resources ready, or false with error populated; the destructor owns partial cleanup.
        bool Initialize(uint32_t flags, FfxDimensions2D render, FfxDimensions2D mvDepth, FfxDimensions2D display, bool sky)
        {
            const auto result = CreateNfruContext(vk,
                                                  flags,
                                                  display.width,
                                                  display.height,
                                                  render.width,
                                                  render.height,
                                                  context,
                                                  {mvDepth.width, mvDepth.height},
                                                  FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT);
            if (result != FFX_API_RETURN_OK)
            {
                error = "NFRU jitter context creation failed: " + std::to_string(result);
                return false;
            }
            constexpr VkImageUsageFlags sampledInputUsage  = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            constexpr VkImageUsageFlags renderedColorUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            constexpr VkImageUsageFlags colorReadbackUsage = renderedColorUsage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            constexpr VkImageUsageFlags outputUsage        = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                                      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            // Display-resolution MV changes only motionSize; depth still uses mvDepth, and jitter still uses render.
            const FfxDimensions2D motionSize = (flags & FFX_API_FG_CONTEXT_FLAG_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) ? display : mvDepth;
            if (!unit_test::vk::CreateImage2D(vk, display.width, display.height, VK_FORMAT_B10G11R11_UFLOAT_PACK32, colorReadbackUsage, color, &error) ||
                !unit_test::vk::CreateImage2D(vk, display.width, display.height, VK_FORMAT_B10G11R11_UFLOAT_PACK32, outputUsage, output, &error) ||
                !unit_test::vk::CreateImage2D(vk, mvDepth.width, mvDepth.height, VK_FORMAT_R32_SFLOAT, sampledInputUsage, depth, &error) ||
                !unit_test::vk::CreateImage2D(
                    vk, display.width, display.height, VK_FORMAT_B10G11R11_UFLOAT_PACK32, renderedColorUsage, previousColor, &error) ||
                !unit_test::vk::CreateImage2D(vk, mvDepth.width, mvDepth.height, VK_FORMAT_R32_SFLOAT, sampledInputUsage, previousDepth, &error) ||
                !unit_test::vk::CreateImage2D(vk, motionSize.width, motionSize.height, VK_FORMAT_R16G16_SFLOAT, sampledInputUsage, motion, &error))
                return false;
            // Far-plane depth exercises the shader's camera-motion fallback rather than game-MV cancellation.
            deviceDepth = sky ? ((flags & FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED) ? 0.0f : 1.0f) : 0.5f;
            return unit_test::vk::RenderTriangle(vk, color, &error) && unit_test::vk::ClearImage(vk, depth, {deviceDepth, 0, 0, 0}, &error) &&
                   unit_test::vk::ClearImage(vk, output, {0, 0, 0, 0}, &error);
        }

        // Prepare and generate one frame using frameID, render-pixel jitter, and a uniform stored rawMotion vector.
        // scale converts stored MV to target pixels (display or MV/depth); reset requests history reset.
        // debug selects the 3x3 diagnostic view instead of normal output.
        // On success, return true and fill pixels with display-sized, four-byte packed-float texels, not RGBA8 channels.
        // On failure, return false with error populated; callers must not compare pixels.
        bool Dispatch(uint64_t              frameID,
                      FfxFloatCoords2D      jitter,
                      FfxFloatCoords2D      scale,
                      std::array<float, 2>  rawMotion,
                      bool                  reset,
                      bool                  debug,
                      std::vector<uint8_t>& pixels)
        {
            // OF clears external history on reset; it must not alias current-frame input.
            if (!unit_test::vk::RenderTriangle(vk, previousColor, &error) || !unit_test::vk::ClearImage(vk, previousDepth, {deviceDepth, 0, 0, 0}, &error) ||
                !unit_test::vk::ClearImage(vk, motion, {rawMotion[0], rawMotion[1], 0, 0}, &error))
                return false;
            // This offscreen fixture dispatches directly and has no swapchain context to notify.
            const uint32_t flags      = FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY | (debug ? FFX_API_FG_DISPATCH_FLAG_DRAW_DEBUG_VIEW : 0);
            const auto     configured = ConfigureNfruContext(context, true, flags, frameID);
            if (configured != FFX_API_RETURN_OK)
            {
                error = "NFRU jitter configure failed: " + std::to_string(configured);
                return false;
            }
            VkCommandBuffer commands = VK_NULL_HANDLE;
            if (!unit_test::vk::BeginCommandBuffer(vk, commands, &error))
                return false;
            // Declare the real read layout, including when the provider registers colour again after OF downsampling.
            for (auto* image : {&color, &depth, &motion, &previousColor, &previousDepth})
            {
                if (image->layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                    continue;
                VkImageMemoryBarrier barrier{};
                barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barrier.srcAccessMask       = VK_ACCESS_MEMORY_WRITE_BIT;
                barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
                barrier.oldLayout           = image->layout;
                barrier.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image               = image->image;
                barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
                image->layout = barrier.newLayout;
            }
            ffxApiDispatchDescFrameGenerationPrepare prepare{};
            prepare.header.type             = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
            prepare.commandList             = commands;
            prepare.frameID                 = frameID;
            prepare.jitterOffset            = {jitter.x, jitter.y};
            prepare.motionVectorScale       = {scale.x, scale.y};
            prepare.frameTimeDelta          = 16.6f;
            prepare.cameraNear              = 0.1f;
            prepare.cameraFar               = 1000.0f;
            prepare.cameraFovAngleVertical  = 1.2f;
            prepare.viewSpaceToMetersFactor = 1.0f;
            prepare.depth         = unit_test::vk::MakeApiImageResource(depth, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
            prepare.motionVectors = unit_test::vk::MakeApiImageResource(motion, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
            prepare.depthTm1 = unit_test::vk::MakeApiImageResource(previousDepth, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
            prepare.colorTm1 = unit_test::vk::MakeApiImageResource(previousColor, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
            // Keep camera motion zero and unjittered; synthetic object motion comes only from the MV texture.
            unit_test::vk::SetIdentityMatrix(prepare.viewProjection);

            ffxApiDispatchDescFrameGeneration dispatch{};
            dispatch.header.type  = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
            dispatch.commandList  = commands;
            dispatch.frameID      = frameID;
            dispatch.reset        = reset;
            dispatch.presentColor = unit_test::vk::MakeApiImageResource(color, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
            dispatch.outputs[0]   = unit_test::vk::MakeApiImageResource(output, FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
            dispatch.numGeneratedFrames = 1;
            // Prepare and FI share a frame ID and command buffer; do not dispatch FI after a failed prepare.
            const auto prepared   = ffxDispatch(&context, &prepare.header);
            const auto dispatched = prepared == FFX_API_RETURN_OK ? ffxDispatch(&context, &dispatch.header) : prepared;
            if (dispatched != FFX_API_RETURN_OK)
            {
                vkFreeCommandBuffers(vk.device, vk.commandPool, 1, &commands);
                error = "NFRU jitter prepare/dispatch failed: " + std::to_string(dispatched);
                return false;
            }
            return unit_test::vk::SubmitCommandBuffer(vk, commands, &error) && unit_test::vk::ReadbackRgba8Image(vk, output, pixels, &error);
        }
    };

    // Compare unjittered-MV reference, corrected jittered MV, and uncorrected jittered MV on a real Vulkan device.
    // fragment selects the fragment pipeline; sky selects camera-motion fallback and requires all three runs to agree.
    // Return true only when all image/reset/negative-control assertions pass; failures are reported through the test framework.
    // Print maximum RGB errors in 1/255 units for each topology and normal/debug output mode.
    bool RunJitterGpuEquivalence(bool fragment, bool sky)
    {
        struct Device
        {
            unit_test::vk::VulkanContext context{};
            // Keep the device alive across scene runs and release it even when a test assertion returns early.
            ~Device()
            {
                unit_test::vk::DestroyVulkanContext(context);
            }
        } device;
        std::string error;
        if (!unit_test::vk::InitializeVulkanContext(device.context, &error))
            UNIT_TEST_FAIL(error);
        // Fragment warp writes a storage image with imageAtomicMax; ordinary framebuffer support alone is insufficient.
        VkPhysicalDeviceFeatures features{};
        vkGetPhysicalDeviceFeatures(device.context.physicalDevice, &features);
        if (fragment && !features.fragmentStoresAndAtomics)
        {
            UNIT_TEST_SKIP("fragmentStoresAndAtomics is unavailable; the NFRU fragment jitter path was not executed");
        }
        struct Variant
        {
            // render defines jitter units; mvDepth/display drive input sizes and the internal downsampling topology.
            FfxDimensions2D render, mvDepth, display;
            // Per-axis conversion from stored MV to target pixels, including sign; flags are context options.
            FfxFloatCoords2D scale;
            uint32_t         flags;
        };
        const uint32_t               managed  = FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH | FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR;
        const std::array<Variant, 3> variants = {{
            // No downsampling: pixel-unit MV with reversed Y, and caller-provided history.
            {{128, 64}, {128, 64}, {128, 64}, {1, -1}, 0},
            // Colour-only downsampling: display-sized UV MV with both axes reversed, inverted depth, managed history.
            {{128, 64},
             {64, 32},
             {256, 128},
             {-256, -128},
             managed | FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED | FFX_API_FG_CONTEXT_FLAG_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS},
            // All-input downsampling: tall images cross resolution thresholds; UV MV reverses Y.
            {{64, 1024}, {64, 1024}, {128, 2048}, {64, -1024}, managed},
        }};
        // Render-pixel sample offsets for five consecutive frame IDs; both axes vary before and after explicit reset.
        const std::array<FfxFloatCoords2D, 5> jitters = {{{-0.375f, 0.25f}, {0.375f, -0.25f}, {-0.125f, 0.375f}, {0.375f, -0.125f}, {-0.25f, 0.25f}}};
        // Sky covers ordinary and inverted far-plane depth using the first two configurations.
        for (size_t variantIndex = 0; variantIndex < (sky ? 2u : variants.size()); ++variantIndex)
        {
            const auto& variant    = variants[variantIndex];
            const auto  resolution = ffxNfruDeriveResolutionPlan(variant.display, variant.mvDepth, FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT);
            // Assert actual topology so future planner changes cannot silently remove the intended path coverage.
            UNIT_TEST_CHECK((variantIndex == 0 && resolution.noDownsample) || (variantIndex == 1 && resolution.needsColourDownsample) ||
                            (variantIndex == 2 && resolution.needsAllInputDownsample));
            for (const bool debug : {false, true})
            {
                // Save each baseline frame separately; history and reset make frames non-interchangeable.
                std::array<std::vector<uint8_t>, 5> reference;
                // At least one non-reset negative-control frame must differ; otherwise this scene may hide MV errors.
                bool  negativeControlDiffers = false;
                float enabledMaxError = 0, negativeMaxError = 0;
                // 0: unjittered MV, cancellation off. 1: jittered MV, cancellation on. 2: jittered MV, cancellation off.
                for (uint32_t run = 0; run < 3; ++run)
                {
                    // Isolate temporal history between runs while keeping the same synthetic colour/depth inputs.
                    JitterGpuScene scene(device.context);
                    const uint32_t flags = variant.flags | (fragment ? 0 : FFX_API_FG_CONTEXT_FLAG_ALL_STAGES_COMPUTE) |
                                           (run == 1 ? FFX_API_FG_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION : 0);
                    if (!scene.Initialize(flags, variant.render, variant.mvDepth, variant.display, sky))
                        UNIT_TEST_FAIL(scene.error + ", topology=" + std::to_string(variantIndex));
                    std::vector<uint8_t> source;
                    if (!unit_test::vk::ReadbackRgba8Image(device.context, scene.color, source, &error))
                        UNIT_TEST_FAIL(error);
                    // Reject non-finite or flat source images: a blank/constant image could make incorrect warps look equivalent.
                    float minimum = 1.0f, maximum = 0;
                    for (size_t byte = 0; byte < source.size(); byte += 4)
                    {
                        for (const float channel : DecodeRgb11(source.data() + byte))
                        {
                            UNIT_TEST_CHECK(std::isfinite(channel));
                            minimum = std::min(minimum, channel);
                            maximum = std::max(maximum, channel);
                        }
                    }
                    UNIT_TEST_CHECK(!source.empty() && maximum - minimum > 0.05f);
                    for (uint32_t frame = 0; frame < jitters.size(); ++frame)
                    {
                        // IDs 1 and 4 reset and copy the source; ID 5 checks recovery using the reset frame's jitter.
                        const bool           reset = frame == 0 || frame == 3;
                        std::array<float, 2> rawMotion{};
                        if (!reset)
                        {
                            // Ordinary geometry has true backward motion; sky is static and must use zero camera motion.
                            const std::array<float, 3> currentPoint{0, 0, -1};
                            const std::array<float, 3> previousPoint = sky ? currentPoint : std::array<float, 3>{0.0078125f, -0.00390625f, -1};
                            // current/previous are projected UV positions, not pixel coordinates or motion vectors.
                            // Only the reference removes jitter from MV generation; all runs pass actual jitter to prepare.
                            const auto current  = ProjectJitteredPoint(currentPoint, run == 0 ? FfxFloatCoords2D{0, 0} : jitters[frame], variant.render);
                            const auto previous = ProjectJitteredPoint(previousPoint, run == 0 ? FfxFloatCoords2D{0, 0} : jitters[frame - 1], variant.render);
                            const auto target = (flags & FFX_API_FG_CONTEXT_FLAG_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) ? variant.display : variant.mvDepth;
                            // Encode current-to-previous UV displacement: SDK decoding multiplies by scale / target.
                            rawMotion = {(previous[0] - current[0]) * target.width / variant.scale.x,
                                         (previous[1] - current[1]) * target.height / variant.scale.y};
                        }
                        std::vector<uint8_t> pixels;
                        if (!scene.Dispatch(frame + 1, jitters[frame], variant.scale, rawMotion, reset, debug, pixels))
                            UNIT_TEST_FAIL(scene.error);
                        UNIT_TEST_CHECK(pixels.size() == source.size());
                        // Reset bypasses interpolation, so require an exact packed-texel copy even in debug mode.
                        if (reset && pixels != source)
                        {
                            const size_t byte = static_cast<size_t>(std::mismatch(pixels.begin(), pixels.end(), source.begin()).first - pixels.begin());
                            UNIT_TEST_FAIL("NFRU reset copy mismatch: topology=" + std::to_string(variantIndex) + ", run=" + std::to_string(run) +
                                           ", debug=" + std::to_string(debug) + ", frame=" + std::to_string(frame) + ", byte=" + std::to_string(byte) +
                                           ", actual=" + std::to_string(pixels[byte]) + ", expected=" + std::to_string(source[byte]));
                        }
                        if (run == 0)
                        {
                            reference[frame] = std::move(pixels);
                            continue;
                        }
                        // Worst absolute RGB-channel error across selected pixels, expressed in 1/255 units, not an average.
                        float maxError = 0;
                        for (uint32_t y = 0; y < variant.display.height; ++y)
                        {
                            for (uint32_t x = 0; x < variant.display.width; ++x)
                            {
                                // Debug cells 4/5 expose MV warps; cell 8 exposes the dynamic mask independently of neural OF fallback.
                                if (debug && !reset)
                                {
                                    const uint32_t cell = (y / (variant.display.height / 3)) * 3 + x / (variant.display.width / 3);
                                    if (cell != 4 && cell != 5 && cell != 8)
                                        continue;
                                }
                                const size_t offset   = (static_cast<size_t>(y) * variant.display.width + x) * 4;
                                const auto   actual   = DecodeRgb11(pixels.data() + offset);
                                const auto   expected = DecodeRgb11(reference[frame].data() + offset);
                                for (uint32_t channel = 0; channel < 3; ++channel)
                                {
                                    UNIT_TEST_CHECK(std::isfinite(actual[channel]) && std::isfinite(expected[channel]));
                                    // Preserve the 1/255 absolute RGB tolerance with the shader's actual packed-float output format.
                                    maxError = std::max(maxError, 255.0f * std::abs(actual[channel] - expected[channel]));
                                }
                            }
                        }
                        // Corrected MV must match the baseline. Sky must also match with cancellation disabled:
                        // the shader replaces game MV with camera motion, which must not receive jitter correction.
                        if ((run == 1 || sky) && maxError > 1)
                            UNIT_TEST_FAIL("NFRU jitter RGB mismatch: topology=" + std::to_string(variantIndex) + ", fragment=" + std::to_string(fragment) +
                                           ", debug=" + std::to_string(debug) + ", frame=" + std::to_string(frame) + ", maxError=" + std::to_string(maxError));
                        if (run == 2 && !reset)
                        {
                            negativeControlDiffers |= maxError > 1;
                            negativeMaxError = std::max(negativeMaxError, maxError);
                        }
                        if (run == 1)
                            enabledMaxError = std::max(enabledMaxError, maxError);
                    }
                }
                // Require a detectable error without cancellation, except where sky intentionally ignores game MV.
                if (!sky)
                    UNIT_TEST_CHECK(negativeControlDiffers);
                std::printf("NFRU jitter: topology=%zu fragment=%d debug=%d sky=%d max RGB error (1/255 units) enabled=%g disabled=%g\n",
                            variantIndex,
                            fragment,
                            debug,
                            sky,
                            enabledMaxError,
                            negativeMaxError);
                std::fflush(stdout);
            }
        }
        return true;
    }
}  // namespace

// Parameterless GPU regression; return the shared equivalence result for ordinary geometry on the compute path.
UNIT_TEST_CASE(nfru_api, jitter_cancellation_gpu_compute)
{
    return RunJitterGpuEquivalence(false, false);
}

// Parameterless GPU regression; return the same equivalence result on the fragment path, including storage atomics.
UNIT_TEST_CASE(nfru_api, jitter_cancellation_gpu_fragment)
{
    return RunJitterGpuEquivalence(true, false);
}

// Parameterless GPU regression; return whether compute sky fallback remains unchanged with cancellation on or off.
UNIT_TEST_CASE(nfru_api, jitter_cancellation_preserves_sky_motion_compute)
{
    return RunJitterGpuEquivalence(false, true);
}

// Parameterless GPU regression; return whether fragment sky fallback remains unchanged with cancellation on or off.
UNIT_TEST_CASE(nfru_api, jitter_cancellation_preserves_sky_motion_fragment)
{
    return RunJitterGpuEquivalence(true, true);
}
