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
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "FidelityFX/host/backends/vk/ffx_hash.h"
#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <FidelityFX/host/backends/vk/vk_wrapper.h>
#include <FidelityFX/host/ffx_assert.h>
#include <FidelityFX/host/ffx_interface.h>
#include <FidelityFX/host/ffx_util.h>
#include <ffx_shader_blobs.h>

#ifdef _WIN32
#if !defined(__UNREAL__)  // Unreal wants to include its own minimal windows .h
#include <windows.h>
#endif  // if !__UNREAL__
#else
#include <codecvt>  // this is deprecated so it's just a fallback solution
#endif
#include <vector>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <vector>
// For graph shape inference
#include "spirv-tools/libspirv.hpp"
#include "spirv/unified1/spirv.hpp11"

// Solution from mozilla:
//     https://hg-edge.mozilla.org/try/diff/73f0522fdf9cb081fe9f859df3727165848c46cd/gfx/src/X11UndefineNone.h
// The header <X11/X.h> defines "None" as a macro that expands to "0L".
// This is terrible because many enumerations have an enumerator named "None".
// To work around this, we undefine the macro "None", and define a replacement
// macro named "X11None".
#ifdef None
#undef None
#define X11None 0L
#ifdef RevertToNone
#undef RevertToNone
#define RevertToNone (int)X11None
#endif
#endif

#ifdef Success
#undef Success
#endif

namespace
{
    // Type aliases
    using TensorBinding = std::pair<uint32_t, uint32_t>;  // set, binding id

    using DescriptorSetBindingToShapeMap = std::map<TensorBinding, std::vector<int64_t>>;

    // Result structure
    struct ShapeInferenceResults
    {
        bool                           Success;
        DescriptorSetBindingToShapeMap Shapes;
        std::vector<uint32_t>          NewCode;
    };

    // Internal: describe a session-memory object requirement for data-graph pipelines
    struct SessionMemoryObjectRequirement
    {
        VkDataGraphPipelineSessionBindPointARM bindPoint;
        uint32_t                               objectIndex;
        VkMemoryRequirements                   memoryRequirements;
        VkDeviceSize                           offset;
    };

    // Custom deleter for SPIR-V resources
    template <typename T, void (*DeleterFunc)(T*)>
    struct CustomDeleter
    {
        void operator()(T* ptr) const
        {
            if (ptr)
            {
                DeleterFunc(ptr);
            }
        }
    };

    template <typename T, void (*DeleterFunc)(T*)>
    using UniqueSpirvPtr = std::unique_ptr<T, CustomDeleter<T, DeleterFunc>>;

    // Helper function
    uint32_t GetSingleWordOperand(const spv_parsed_instruction_t& instr, int operandIdx)
    {
        //assert(instr.operands[operandIdx].num_words == 1);
        return instr.words[instr.operands[operandIdx].offset];
    }

    DescriptorSetBindingToShapeMap GetInputShapes(const FfxDataGraphBlob& dataGraphBlob, const uint32_t width, const uint32_t height)
    {
        DescriptorSetBindingToShapeMap inputShapes{};
        for (FfxUInt32 i = 0; i < dataGraphBlob.inputTensorNums; ++i)
        {
            const auto           dimSize = dataGraphBlob.inputTensorDimSize[i];
            std::vector<int64_t> currentTensorDimension{};
            currentTensorDimension.push_back(1);  // Batch size is always 1
            bool is_width = false;
            for (FfxUInt32 j = 1; j < dimSize; ++j)
            {
                // If a invalid size is found, it's not static graph.
                auto curSize = dataGraphBlob.inputTensorDims[i][j];

                if (curSize <= 0)
                {
                    // Replace the unknown size with resolution.
                    curSize  = is_width ? width : height;
                    is_width = true;
                }
                currentTensorDimension.push_back(curSize);
            }
            const std::pair<uint32_t, uint32_t> binding{0, dataGraphBlob.inputTensorBindings[i]};  // [set, binding]
            inputShapes.emplace(binding, std::move(currentTensorDimension));
        }
        return inputShapes;
    }

    ShapeInferenceResults RunShapeInference(const uint32_t* code, const uint32_t codeSize, const DescriptorSetBindingToShapeMap& inputShapes)
    {
        UniqueSpirvPtr<spv_optimizer_t, spvOptimizerDestroy> optimizer(spvOptimizerCreate(SPV_ENV_VULKAN_1_3));
        if (!optimizer)
            return {false};

        spvOptimizerSetMessageConsumer(optimizer.get(), [](spv_message_level_t, const char*, const spv_position_t*, const char* message) {});

        UniqueSpirvPtr<spv_optimizer_options_t, spvOptimizerOptionsDestroy> options(spvOptimizerOptionsCreate());
        if (!options)
        {
            return {false};
        }

        std::vector<spv_graph_shape_input> shapeInputs;
        for (const auto& [key, shape] : inputShapes)
        {
            shapeInputs.push_back({key.first, key.second, static_cast<uint32_t>(shape.size()), shape.data()});
        }

        spvOptimizerRegisterGraphShapePass(optimizer.get(), shapeInputs.size(), shapeInputs.data());

        spv_binary_t*                                  rawBinary = nullptr;
        spv_result_t                                   result    = spvOptimizerRun(optimizer.get(), code, codeSize, &rawBinary, options.get());
        UniqueSpirvPtr<spv_binary_t, spvBinaryDestroy> optimizedCode(rawBinary);
        if (result != SPV_SUCCESS)
        {
            return {false};
        }

        UniqueSpirvPtr<spv_context_t, spvContextDestroy> context(spvContextCreate(SPV_ENV_VULKAN_1_3));
        if (!context)
        {
            return {false};
        }

        struct ParsingResultsStruct
        {
            std::vector<spv_parsed_instruction_t>                    ParsedInstructions;
            std::vector<std::vector<spv_parsed_operand_t>>           ParsedOperands;
            std::map<uint32_t, uint32_t>                             IdToInstructionIdx;
            std::map<std::pair<uint32_t, spv::Decoration>, uint32_t> Decorations;
        } parsingResults;

        auto parsingCallback = [](void* userData, const spv_parsed_instruction_t* instr) -> spv_result_t {
            auto& results = *static_cast<ParsingResultsStruct*>(userData);
            results.ParsedInstructions.push_back(*instr);
            results.ParsedOperands.emplace_back(instr->operands, instr->operands + instr->num_operands);
            results.ParsedInstructions.back().operands = results.ParsedOperands.back().data();

            if (instr->result_id)
            {
                results.IdToInstructionIdx[instr->result_id] = static_cast<uint32_t>(results.ParsedInstructions.size()) - 1;
            }

            if (instr->opcode == static_cast<uint16_t>(spv::Op::OpDecorate) && instr->num_operands >= 3)
            {
                auto decoratedId = GetSingleWordOperand(*instr, 0);
                auto kind        = static_cast<spv::Decoration>(GetSingleWordOperand(*instr, 1));
                auto value       = GetSingleWordOperand(*instr, 2);
                if (kind == spv::Decoration::Binding || kind == spv::Decoration::DescriptorSet)
                {
                    results.Decorations[{decoratedId, kind}] = value;
                }
            }
            return SPV_SUCCESS;
        };

        spv_diagnostic_t* rawDiagnostic = nullptr;
        result = spvBinaryParse(context.get(), &parsingResults, optimizedCode->code, optimizedCode->wordCount, nullptr, parsingCallback, &rawDiagnostic);
        UniqueSpirvPtr<spv_diagnostic_t, spvDiagnosticDestroy> diagnostic(rawDiagnostic);
        if (result != SPV_SUCCESS)
        {
            //std::cerr << "spvBinaryParse failed: " << diagnostic->error << std::endl;
            return {false};
        }

        ShapeInferenceResults inferenceResults;
        for (const auto& instr : parsingResults.ParsedInstructions)
        {
            if (instr.opcode != static_cast<uint32_t>(spv::Op::OpVariable))
            {
                continue;
            }

            uint32_t varId      = instr.result_id;
            auto     varDeclIdx = parsingResults.IdToInstructionIdx[varId];
            auto&    varDecl    = parsingResults.ParsedInstructions[varDeclIdx];

            auto ptrTypeDecl  = parsingResults.ParsedInstructions[parsingResults.IdToInstructionIdx[varDecl.type_id]];
            auto tensorTypeId = GetSingleWordOperand(ptrTypeDecl, 2);
            auto tensorDecl   = parsingResults.ParsedInstructions[parsingResults.IdToInstructionIdx[tensorTypeId]];
            if (tensorDecl.opcode != static_cast<uint32_t>(spv::Op::OpTypeTensorARM))
            {
                continue;
            }

            auto shapeId   = GetSingleWordOperand(tensorDecl, 3);
            auto shapeDecl = parsingResults.ParsedInstructions[parsingResults.IdToInstructionIdx[shapeId]];
            if (shapeDecl.opcode != static_cast<uint32_t>(spv::Op::OpConstantComposite))
            {
                continue;
            }

            std::vector<int64_t> shape;
            for (int i = 0; i < shapeDecl.num_operands - 2; ++i)
            {
                auto dimId   = GetSingleWordOperand(shapeDecl, i + 2);
                auto dimDecl = parsingResults.ParsedInstructions[parsingResults.IdToInstructionIdx[dimId]];
                if (dimDecl.opcode != static_cast<uint32_t>(spv::Op::OpConstant))
                {
                    continue;
                }

                shape.push_back(GetSingleWordOperand(dimDecl, 2));
            }

            auto set                                = parsingResults.Decorations[{varId, spv::Decoration::DescriptorSet}];
            auto binding                            = parsingResults.Decorations[{varId, spv::Decoration::Binding}];
            inferenceResults.Shapes[{set, binding}] = std::move(shape);
        }

        inferenceResults.NewCode.assign(optimizedCode->code, optimizedCode->code + optimizedCode->wordCount);
        inferenceResults.Success = true;
        return inferenceResults;
    }
}  // namespace

static void PrintMessage(FfxInterface* backendInterface, uint32_t type, const char* message)
{
    if (backendInterface && backendInterface->fpMessage && message != nullptr)
    {
        backendInterface->fpMessage(type, message);
    }
}

// prototypes for functions in the interface
FfxErrorCode     SetupMessageCallback(FfxInterface* backendInterface, FfxBackendMessage ffxMessageCallback);
FfxVersionNumber GetSDKVersionVK(FfxInterface* backendInterface);
FfxErrorCode     GetEffectGpuMemoryUsageVK(FfxInterface* backendInterface, FfxUInt32 effectContextId, FfxEffectMemoryUsage* outVramUsage);
FfxErrorCode     CreateBackendContextVK(FfxInterface* backendInterface, FfxEffect effect, FfxEffectBindlessConfig* bindlessConfig, FfxUInt32* effectContextId);
FfxErrorCode     GetDeviceCapabilitiesVK(FfxInterface* backendInterface, FfxDeviceCapabilities* deviceCapabilities);
FfxErrorCode     DestroyBackendContextVK(FfxInterface* backendInterface, FfxUInt32 effectContextId);
FfxErrorCode     CreateResourceVK(FfxInterface*                       backendInterface,
                                  const FfxCreateResourceDescription* desc,
                                  FfxUInt32                           effectContextId,
                                  FfxResourceInternal*                outTexture);
FfxErrorCode     DestroyResourceVK(FfxInterface* backendInterface, FfxResourceInternal resource, FfxUInt32 effectContextId);
FfxErrorCode     MapResourceVK(FfxInterface* backendInterface, FfxResourceInternal resource, void** ptr);
FfxErrorCode     UnmapResourceVK(FfxInterface* backendInterface, FfxResourceInternal resource);
FfxErrorCode     RegisterResourceVK(FfxInterface*        backendInterface,
                                    const FfxResource*   inResource,
                                    FfxUInt32            effectContextId,
                                    FfxResourceInternal* outResourceInternal);
FfxResource      GetResourceVK(FfxInterface* backendInterface, FfxResourceInternal resource);
FfxErrorCode     UnregisterResourcesVK(FfxInterface* backendInterface, FfxCommandList commandList, FfxUInt32 effectContextId);
FfxErrorCode     RegisterStaticResourceVK(FfxInterface* backendInterface, const FfxStaticResourceDescription* desc, FfxUInt32 effectContextId);
FfxResourceDescription GetResourceDescriptionVK(FfxInterface* backendInterface, FfxResourceInternal resource);
FfxErrorCode           StageConstantBufferDataVK(FfxInterface* backendInterface, void* data, FfxUInt32 size, FfxConstantBuffer* constantBuffer);
FfxErrorCode           CreateComputePipelineVK(FfxInterface*                 backendInterface,
                                               FfxEffect                     effect,
                                               FfxPass                       passId,
                                               uint32_t                      permutationOptions,
                                               const FfxPipelineDescription* desc,
                                               FfxUInt32                     effectContextId,
                                               FfxPipelineState*             outPass);
FfxErrorCode           CreateGraphicsPipelineVK(FfxInterface*                 backendInterface,
                                                FfxEffect                     effect,
                                                FfxPass                       passId,
                                                uint32_t                      permutationOptions,
                                                const FfxPipelineDescription* desc,
                                                FfxUInt32                     effectContextId,
                                                FfxPipelineState*             outPass);
FfxErrorCode           CreateDataGraphPipelineVK(FfxInterface*                 backendInterface,
                                                 FfxEffect                     effect,
                                                 FfxPass                       passId,
                                                 uint32_t                      permutationOptions,
                                                 const FfxPipelineDescription* desc,
                                                 FfxUInt32                     effectContextId,
                                                 FfxUInt32                     dataGraphWidth,
                                                 FfxUInt32                     dataGraphHeight,
                                                 FfxPipelineState*             outPass);
FfxErrorCode           CreateOpticalFlowPipelineVK(
              FfxInterface* backendInterface, const char* name, const FfxOpticalFlowDescription& ofDesc, FfxUInt32 effectContextId, FfxPipelineState* outPipeline);

FfxErrorCode DestroyPipelineVK(FfxInterface* backendInterface, FfxPipelineState* pipeline, FfxUInt32 effectContextId);
FfxErrorCode ScheduleGpuJobVK(FfxInterface* backendInterface, const FfxGpuJobDescription* job);
FfxErrorCode ExecuteGpuJobsVK(FfxInterface* backendInterface, FfxCommandList commandList, FfxUInt32 effectContextId);

static VkDeviceContext sVkDeviceContext = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};

#define MAX_PIPELINE_USAGE_PER_FRAME (10)  // Required to make sure passes that are called more than once per-frame don't have their descriptors overwritten.

#define MAX_DESCRIPTOR_SET_LAYOUTS  (32)
#define MAX_DESCRIPTOR_SETS         (2)
#define MAX_RENDER_PASS_COUNT       (FFX_MAX_QUEUED_FRAMES)
#define MAX_IMAGE_VIEW_COUNT        (FFX_MAX_QUEUED_FRAMES * 4)
#define MAX_FRAME_BUFFER_COUNT      (FFX_MAX_QUEUED_FRAMES)
#define MAX_GRAPHICS_PIPELINE_COUNT (FFX_MAX_QUEUED_FRAMES)

#define FFX_MAX_BINDLESS_DESCRIPTOR_COUNT (65536)

// Constant buffer allocation callback
static FfxConstantBufferAllocator s_fpConstantAllocator = nullptr;

// Offset the binding of samplers to avoid collisions
constexpr uint32_t SAMPLER_BINDING_SHIFT = 1000;

typedef struct ObjectBase_VK
{
    uint64_t  hash;
    FfxUInt32 visitedFlag;
} ObjectBase_VK;

typedef struct GraphicPipeline_VK : ObjectBase_VK
{
    VkPipeline handle;
} GraphicPipeline_VK;

typedef struct FrameBuffer_VK : ObjectBase_VK
{
    VkFramebuffer handle;
} FrameBuffer_VK;

typedef struct ImageView_VK : ObjectBase_VK
{
    VkImageView handle;
} ImageView_VK;

typedef struct RenderPass_VK : ObjectBase_VK
{
    VkRenderPass handle;
} RenderPass_VK;

typedef struct BackendContext_VK
{
    // store for resources and resourceViews
    typedef struct Resource
    {
#ifdef _DEBUG
        char resourceName[64] = {};
#endif
        union
        {
            struct
            {
                VkImage  imageResource;
                VkBuffer aliasedBufferResource;
            };
            VkBuffer bufferResource;
            struct
            {
                VkTensorARM tensorResource;
                VkImage     aliasedTensorImageResource;   // for tensor-image aliasing
                VkBuffer    aliasedTensorBufferResource;  // for tensor-buffer aliasing
            };
        };

        FfxResourceDescription resourceDescription;
        FfxResourceStates      initialState;
        FfxResourceStates      currentState;
        int32_t                srvViewIndex;
        int32_t                uavViewIndex;
        uint32_t               uavViewCount;
        int32_t                tensorViewIndex;

        VkDeviceMemory        deviceMemory;
        VkDeviceSize          allocationSize;
        VkMemoryPropertyFlags memoryProperties;

        bool undefined;
        bool dynamic;

    } Resource;

    typedef struct PipelineLayout
    {
        VkSampler             samplers[FFX_MAX_SAMPLERS];
        VkDescriptorSetLayout descriptorSetLayout;
        VkDescriptorSet       descriptorSets[FFX_MAX_QUEUED_FRAMES * MAX_PIPELINE_USAGE_PER_FRAME];
        uint32_t              descriptorSetIndex;
        VkPipelineLayout      pipelineLayout;
        int32_t               staticTextureSrvSet;
        int32_t               staticBufferSrvSet;
        int32_t               staticTextureUavSet;
        int32_t               staticBufferUavSet;

        // Only used by graphics pipeline
        VkShaderModule fragShaderModule;
        VkShaderModule vertShaderModule;

        RenderPass_VK renderPass[MAX_RENDER_PASS_COUNT];
        FfxUInt32     renderPassIndex;

        ImageView_VK imageView[MAX_IMAGE_VIEW_COUNT];
        FfxUInt32    imageViewIndex;

        FrameBuffer_VK frameBuffer[MAX_FRAME_BUFFER_COUNT];
        FfxUInt32      frameBufferIndex;

        GraphicPipeline_VK graphicsPipeline[MAX_GRAPHICS_PIPELINE_COUNT];
        FfxUInt32          graphicsPipelineIndex;

        // Data graph pipeline session memory owned by this pipeline layout.
        VkDeviceMemory dataGraphSessionMemory;
        VkDeviceSize   dataGraphSessionMemorySize;

        char      name[64];
        FfxUInt32 effectContextId;
    } PipelineLayout;

    uint32_t refCount;
    uint32_t maxEffectContexts;

    FfxGpuVendorId   gpuVendorId    = FfxGpuVendorId::NotQueried;
    VkInstance       instance       = VK_NULL_HANDLE;
    VkDevice         device         = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

    FfxGpuJobDescription* pGpuJobs;
    uint32_t              gpuJobCount = 0;

    typedef struct VkResourceView
    {
        VkImageView     imageView;
        VkTensorViewARM tensorView;
    } VkResourceView;
    VkResourceView* pResourceViews;

    uint8_t* pStagingRingBuffer;
    uint32_t stagingRingBufferBase = 0;

    PipelineLayout* pPipelineLayouts;

    VkDescriptorPool descriptorPool;
    uint32_t         bindlessBase;

    VkImageMemoryBarrier2    imageMemoryBarriers[FFX_MAX_BARRIERS]  = {};
    VkBufferMemoryBarrier2   bufferMemoryBarriers[FFX_MAX_BARRIERS] = {};
    VkTensorMemoryBarrierARM tensorMemoryBarriers[FFX_MAX_BARRIERS] = {};
    uint32_t                 scheduledImageBarrierCount             = 0;
    uint32_t                 scheduledBufferBarrierCount            = 0;
    uint32_t                 scheduledTensorBarrierCount            = 0;

    typedef struct alignas(32) EffectContext
    {
        // Effect identifier -- used for various resource callbacks to application
        FfxEffect effectId;

        // Resource allocation
        uint32_t nextStaticResource;
        uint32_t nextDynamicResource;

        // UAV offsets
        uint32_t nextStaticResourceView;
        uint32_t nextDynamicResourceView[FFX_MAX_QUEUED_FRAMES];

        // Bindless descriptors
        uint32_t              bindlessTextureSrvHeapStart;
        uint32_t              bindlessTextureSrvHeapSize;
        uint32_t              bindlessBufferSrvHeapSize;
        uint32_t              bindlessTextureUavHeapStart;
        uint32_t              bindlessTextureUavHeapSize;
        uint32_t              bindlessBufferUavHeapSize;
        VkDescriptorPool      bindlessDescriptorPool;
        VkDescriptorSetLayout bindlessTextureSrvDescriptorSetLayout;
        VkDescriptorSetLayout bindlessBufferSrvDescriptorSetLayout;
        VkDescriptorSetLayout bindlessTextureUavDescriptorSetLayout;
        VkDescriptorSetLayout bindlessBufferUavDescriptorSetLayout;
        VkDescriptorSet       bindlessTextureSrvDescriptorSet;
        VkDescriptorSet       bindlessBufferSrvDescriptorSet;
        VkDescriptorSet       bindlessTextureUavDescriptorSet;
        VkDescriptorSet       bindlessBufferUavDescriptorSet;

        // Pipeline layout
        uint32_t nextPipelineLayout;

        // the frame index for the context
        uint32_t frameIndex;

        // Usage
        bool active;

        // VRAM usage
        FfxEffectMemoryUsage vramUsage;

    } EffectContext;

    Resource*      pResources;
    EffectContext* pEffectContexts;

    // Allocation defaults
    FfxConstantAllocation FallbackConstantAllocator(void* data, FfxUInt64 dataSize);
    VkDeviceMemory        uniformBufferMemory = VK_NULL_HANDLE;
    VkMemoryPropertyFlags uniformBufferMemoryProperties;
    VkDeviceSize          uniformBufferAlignment = 0;
    void*                 uniformBufferMem       = nullptr;
    VkBuffer              uniformBuffer          = VK_NULL_HANDLE;
    VkDeviceSize          uniformBufferSize      = 0;
    VkDeviceSize          uniformBufferOffset    = 0;
    std::mutex            uniformBufferMutex;

    uint32_t               numDeviceExtensions = 0;
    VkExtensionProperties* extensionProperties = nullptr;

} BackendContext_VK;

FFX_API size_t ffxGetScratchMemorySizeVK(VkDeviceContext& deviceContext, size_t maxContexts)
{
    uint32_t numExtensions = 0;

    if (deviceContext.vkInstance)
    {
        VulkanWrapper().vkEnumerateDeviceExtensionProperties(deviceContext.vkPhysicalDevice, nullptr, &numExtensions, nullptr);
    }

    uint32_t extensionPropArraySize = sizeof(VkExtensionProperties) * numExtensions;
    uint32_t gpuJobDescArraySize    = FFX_ALIGN_UP(maxContexts * FFX_MAX_GPU_JOBS * sizeof(FfxGpuJobDescription), sizeof(uint32_t));
    uint32_t resourceViewArraySize  = FFX_ALIGN_UP(
        ((maxContexts * FFX_MAX_QUEUED_FRAMES * FFX_MAX_RESOURCE_COUNT * 2) + FFX_MAX_BINDLESS_DESCRIPTOR_COUNT) * sizeof(BackendContext_VK::VkResourceView),
        sizeof(uint32_t));
    uint32_t stagingRingBufferArraySize = FFX_ALIGN_UP(maxContexts * FFX_CONSTANT_BUFFER_RING_BUFFER_SIZE, sizeof(uint32_t));
    uint32_t pipelineArraySize          = FFX_ALIGN_UP(maxContexts * FFX_MAX_PASS_COUNT * sizeof(BackendContext_VK::PipelineLayout), sizeof(uint32_t));
    uint32_t resourceArraySize          = FFX_ALIGN_UP(maxContexts * FFX_MAX_RESOURCE_COUNT * sizeof(BackendContext_VK::Resource), sizeof(uint32_t));
    uint32_t contextArraySize           = FFX_ALIGN_UP(maxContexts * sizeof(BackendContext_VK::EffectContext), sizeof(uint32_t));
    // Extra padding to allow aligning pEffectContexts to its required alignment
    uint32_t effectContextAlignPadding = alignof(BackendContext_VK::EffectContext) - 1;

    return FFX_ALIGN_UP(sizeof(BackendContext_VK) + extensionPropArraySize + gpuJobDescArraySize + resourceViewArraySize + stagingRingBufferArraySize +
                            pipelineArraySize + resourceArraySize + effectContextAlignPadding + contextArraySize,
                        sizeof(uint64_t));
}

// Create a FfxDevice from a VkDevice
FfxDevice ffxGetDeviceVK(VkDeviceContext* vkDeviceContext)
{
    sVkDeviceContext = *vkDeviceContext;
    return reinterpret_cast<FfxDevice>(&sVkDeviceContext);
}

FfxErrorCode ffxGetInterfaceVK(FfxInterface* backendInterface, FfxDevice device, void* scratchBuffer, size_t scratchBufferSize, size_t maxContexts)
{
    FFX_RETURN_ON_ERROR(backendInterface, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(scratchBuffer, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(scratchBufferSize >= ffxGetScratchMemorySizeVK(*(VkDeviceContext*)device, maxContexts), FFX_ERROR_INSUFFICIENT_MEMORY);

    // The backend print message callback should set by user by calling fpSetMessageCallback.
    backendInterface->fpMessage                     = nullptr;
    backendInterface->fpSetMessageCallback          = SetupMessageCallback;
    backendInterface->fpGetSDKVersion               = GetSDKVersionVK;
    backendInterface->fpGetEffectGpuMemoryUsage     = GetEffectGpuMemoryUsageVK;
    backendInterface->fpCreateBackendContext        = CreateBackendContextVK;
    backendInterface->fpGetDeviceCapabilities       = GetDeviceCapabilitiesVK;
    backendInterface->fpDestroyBackendContext       = DestroyBackendContextVK;
    backendInterface->fpCreateResource              = CreateResourceVK;
    backendInterface->fpDestroyResource             = DestroyResourceVK;
    backendInterface->fpMapResource                 = MapResourceVK;
    backendInterface->fpUnmapResource               = UnmapResourceVK;
    backendInterface->fpRegisterResource            = RegisterResourceVK;
    backendInterface->fpGetResource                 = GetResourceVK;
    backendInterface->fpUnregisterResources         = UnregisterResourcesVK;
    backendInterface->fpRegisterStaticResource      = RegisterStaticResourceVK;
    backendInterface->fpGetResourceDescription      = GetResourceDescriptionVK;
    backendInterface->fpStageConstantBufferDataFunc = StageConstantBufferDataVK;
    backendInterface->fpCreateComputePipeline       = CreateComputePipelineVK;

    // ARM
    backendInterface->fpCreateGraphicsPipeline    = CreateGraphicsPipelineVK;
    backendInterface->fpCreateDataGraphPipeline   = CreateDataGraphPipelineVK;
    backendInterface->fpCreateOpticalFlowPipeline = CreateOpticalFlowPipelineVK;
    // ~ARM
    backendInterface->fpDestroyPipeline           = DestroyPipelineVK;
    backendInterface->fpGetPermutationBlobByIndex = ffxGetPermutationBlobByIndex;
    backendInterface->fpScheduleGpuJob            = ScheduleGpuJobVK;
    backendInterface->fpExecuteGpuJobs            = ExecuteGpuJobsVK;
    //backendInterface->fpRegisterConstantBufferAllocator   = RegisterConstantBufferAllocatorVK;
    backendInterface->fpSwapChainConfigureFrameGeneration = ffxSetFrameGenerationConfigToSwapchainVK;

    // Memory assignments
    backendInterface->scratchBuffer     = scratchBuffer;
    backendInterface->scratchBufferSize = scratchBufferSize;

    BackendContext_VK* backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;

    FFX_RETURN_ON_ERROR(!backendContext->refCount, FFX_ERROR_BACKEND_API_ERROR);

    // Clear everything out
    memset(backendContext, 0, sizeof(*backendContext));

    // Map the device
    backendInterface->device = device;

    // Assign the max number of contexts we'll be using
    backendContext->maxEffectContexts = (uint32_t)maxContexts;

    // Mark the device capabilities as uninitialized
    backendInterface->devCapInitialized = false;

    return FFX_OK;
}

FfxCommandList ffxGetCommandListVK(VkCommandBuffer cmdBuf)
{
    FFX_ASSERT(NULL != cmdBuf);
    return reinterpret_cast<FfxCommandList>(cmdBuf);
}

FfxPipeline ffxGetPipelineVK(VkPipeline pipeline)
{
    FFX_ASSERT(NULL != pipeline);
    return reinterpret_cast<FfxPipeline>(pipeline);
}

FfxResource ffxGetResourceVK(void*                  vkResource,
                             FfxResourceDescription ffxResDescription,
                             const char*            ffxResName,
                             FfxResourceStates      state /*=FFX_RESOURCE_STATE_COMPUTE_READ*/)
{
    FfxResource resource = {};
    resource.resource    = vkResource;
    resource.state       = state;
    resource.description = ffxResDescription;

#ifdef _DEBUG
    if (ffxResName)
    {
        strncpy(resource.name, ffxResName, FFX_RESOURCE_NAME_SIZE - 1);
        resource.name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
    }
#endif

    return resource;
}

uint32_t findMemoryTypeIndex(BackendContext_VK*     backendContext,
                             VkMemoryRequirements   memRequirements,
                             VkMemoryPropertyFlags  requestedProperties,
                             VkMemoryPropertyFlags& outProperties)
{
    FFX_ASSERT(NULL != backendContext);
    FFX_ASSERT(NULL != backendContext->physicalDevice);

    VkPhysicalDeviceMemoryProperties memProperties;
    VulkanWrapper().vkGetPhysicalDeviceMemoryProperties(backendContext->physicalDevice, &memProperties);

    uint32_t bestCandidate = UINT32_MAX;

    // If just device-local memory is requested, make sure this is the invisible heap to prevent over-subscribing the local heap
    if (requestedProperties == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    {
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        {
            const auto propertyFlags    = memProperties.memoryTypes[i].propertyFlags;
            const bool onlyDeviceMemory = (propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && !(propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            if ((memRequirements.memoryTypeBits & (1 << i)) && onlyDeviceMemory)
            {
                bestCandidate = i;
                outProperties = propertyFlags;
                break;
            }
        }
    }

    // If there is no candidate found yet, let's use whatever that suites our needs.
    if (UINT32_MAX == bestCandidate)
    {
        const bool requestHostVisible = requestedProperties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        {
            const auto propertyFlags = memProperties.memoryTypes[i].propertyFlags;
            if ((memRequirements.memoryTypeBits & (1 << i)) && (propertyFlags & requestedProperties))
            {
                bestCandidate = i;
                outProperties = propertyFlags;

                // If host-visible memory is requested, we keep iterating unless host coherent is found.
                // Otherwise, we just choose the first one that suites our needs.
                if (!requestHostVisible || (propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
                {
                    break;
                }
            }
        }
    }

    return bestCandidate;
}

VkBufferUsageFlags ffxGetVKBufferUsageFlagsFromResourceUsage(FfxResourceUsage flags)
{
    VkBufferUsageFlags indirectBit = 0;

    if (FFX_CONTAINS_FLAG(flags, FFX_RESOURCE_USAGE_INDIRECT))
        indirectBit = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

    if (FFX_CONTAINS_FLAG(flags, FFX_RESOURCE_USAGE_UAV))
        return indirectBit | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    else
        return indirectBit | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
}

VkImageType ffxGetVKImageTypeFromResourceType(FfxResourceType type)
{
    switch (type)
    {
    case (FFX_RESOURCE_TYPE_TEXTURE1D):
        return VK_IMAGE_TYPE_1D;
    case (FFX_RESOURCE_TYPE_TEXTURE2D):
        return VK_IMAGE_TYPE_2D;
    case (FFX_RESOURCE_TYPE_TEXTURE_CUBE):
    case (FFX_RESOURCE_TYPE_TEXTURE3D):
        return VK_IMAGE_TYPE_3D;
    default:
        return VK_IMAGE_TYPE_MAX_ENUM;
    }
}

bool ffxIsSurfaceFormatSRGB(FfxSurfaceFormat fmt)
{
    switch (fmt)
    {
    case (FFX_SURFACE_FORMAT_R8G8B8A8_SRGB):
    case (FFX_SURFACE_FORMAT_B8G8R8A8_SRGB):
        return true;
    case (FFX_SURFACE_FORMAT_R32G32B32A32_TYPELESS):
    case (FFX_SURFACE_FORMAT_R32G32B32A32_UINT):
    case (FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT):
    case (FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT):
    case (FFX_SURFACE_FORMAT_R32G32B32_FLOAT):
    case (FFX_SURFACE_FORMAT_R32G32_FLOAT):
    case (FFX_SURFACE_FORMAT_R8_UINT):
    case (FFX_SURFACE_FORMAT_R32_UINT):
    case (FFX_SURFACE_FORMAT_R8G8B8A8_TYPELESS):
    case (FFX_SURFACE_FORMAT_R8G8B8A8_UNORM):
    case (FFX_SURFACE_FORMAT_R8G8B8A8_SNORM):
    case (FFX_SURFACE_FORMAT_B8G8R8A8_TYPELESS):
    case (FFX_SURFACE_FORMAT_B8G8R8A8_UNORM):
    case (FFX_SURFACE_FORMAT_R11G11B10_FLOAT):
    case (FFX_SURFACE_FORMAT_R10G10B10A2_UNORM):
    case (FFX_SURFACE_FORMAT_R16G16_FLOAT):
    case (FFX_SURFACE_FORMAT_R16G16_UINT):
    case (FFX_SURFACE_FORMAT_R16G16_SINT):
    case (FFX_SURFACE_FORMAT_R16_FLOAT):
    case (FFX_SURFACE_FORMAT_R16_UINT):
    case (FFX_SURFACE_FORMAT_R16_UNORM):
    case (FFX_SURFACE_FORMAT_R16_SNORM):
    case (FFX_SURFACE_FORMAT_R8_UNORM):
    case (FFX_SURFACE_FORMAT_R8_SNORM):
    case (FFX_SURFACE_FORMAT_R8_SINT):
    case (FFX_SURFACE_FORMAT_R8G8_UNORM):
    case (FFX_SURFACE_FORMAT_R8G8_UINT):
    case (FFX_SURFACE_FORMAT_R8G8_SINT):
    case (FFX_SURFACE_FORMAT_R32_FLOAT):
    case (FFX_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP):
    case (FFX_SURFACE_FORMAT_UNKNOWN):
        return false;
    default:
        FFX_ASSERT_MESSAGE(false, "Format not yet supported");
        return false;
    }
}

FfxSurfaceFormat ffxGetSurfaceFormatFromGamma(FfxSurfaceFormat fmt)
{
    switch (fmt)
    {
    case (FFX_SURFACE_FORMAT_R8G8B8A8_SRGB):
        return FFX_SURFACE_FORMAT_R8G8B8A8_UNORM;
    case (FFX_SURFACE_FORMAT_B8G8R8A8_SRGB):
        return FFX_SURFACE_FORMAT_B8G8R8A8_UNORM;
    default:
        return fmt;
    }
}

FfxSurfaceFormat ffxGetSurfaceFormatToGamma(FfxSurfaceFormat fmt)
{
    switch (fmt)
    {
    case (FFX_SURFACE_FORMAT_R8G8B8A8_UNORM):
        return FFX_SURFACE_FORMAT_R8G8B8A8_SRGB;
    case (FFX_SURFACE_FORMAT_B8G8R8A8_UNORM):
        return FFX_SURFACE_FORMAT_B8G8R8A8_SRGB;
    default:
        return fmt;
    }
}

uint8_t ffxGetBytesPerElementFromSurfaceFormat(FfxSurfaceFormat fmt)
{
    switch (fmt)
    {
    case (FFX_SURFACE_FORMAT_R16G16_FLOAT):
        return 4;
    case (FFX_SURFACE_FORMAT_R16_UINT):
        return 2;
    case (FFX_SURFACE_FORMAT_R8_UNORM):
        return 1;
        // do not care about this number for other formats
    default:
        return 1;
    }
}

VkFormat ffxGetVkFormatFromSurfaceFormat(FfxSurfaceFormat fmt)
{
    switch (fmt)
    {
    case (FFX_SURFACE_FORMAT_UNKNOWN):
        return VK_FORMAT_UNDEFINED;
    case (FFX_SURFACE_FORMAT_R32G32B32A32_TYPELESS):
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    case (FFX_SURFACE_FORMAT_R32G32B32A32_UINT):
        return VK_FORMAT_R32G32B32A32_UINT;
    case (FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT):
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    case (FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT):
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    case (FFX_SURFACE_FORMAT_R32G32B32_FLOAT):
        return VK_FORMAT_R32G32B32_SFLOAT;
    case (FFX_SURFACE_FORMAT_R32G32_FLOAT):
        return VK_FORMAT_R32G32_SFLOAT;
    case (FFX_SURFACE_FORMAT_R8_UINT):
        return VK_FORMAT_R8_UINT;
    case (FFX_SURFACE_FORMAT_R8_SINT):
        return VK_FORMAT_R8_SINT;
    case (FFX_SURFACE_FORMAT_R32_UINT):
        return VK_FORMAT_R32_UINT;
    case (FFX_SURFACE_FORMAT_R8G8B8A8_TYPELESS):
        return VK_FORMAT_R8G8B8A8_UNORM;
    case (FFX_SURFACE_FORMAT_R8G8B8A8_UNORM):
        return VK_FORMAT_R8G8B8A8_UNORM;
    case (FFX_SURFACE_FORMAT_R8G8B8A8_SNORM):
        return VK_FORMAT_R8G8B8A8_SNORM;
    case (FFX_SURFACE_FORMAT_R8G8B8A8_SRGB):
        return VK_FORMAT_R8G8B8A8_SRGB;
    case (FFX_SURFACE_FORMAT_B8G8R8A8_TYPELESS):
        return VK_FORMAT_B8G8R8A8_UNORM;
    case (FFX_SURFACE_FORMAT_B8G8R8A8_UNORM):
        return VK_FORMAT_B8G8R8A8_UNORM;
    case (FFX_SURFACE_FORMAT_B8G8R8A8_SRGB):
        return VK_FORMAT_B8G8R8A8_SRGB;
    case (FFX_SURFACE_FORMAT_R11G11B10_FLOAT):
        return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case (FFX_SURFACE_FORMAT_R10G10B10A2_UNORM):
        return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case (FFX_SURFACE_FORMAT_R16G16_FLOAT):
        return VK_FORMAT_R16G16_SFLOAT;
    case (FFX_SURFACE_FORMAT_R16G16_UINT):
        return VK_FORMAT_R16G16_UINT;
    case (FFX_SURFACE_FORMAT_R16G16_SINT):
        return VK_FORMAT_R16G16_SINT;
    case (FFX_SURFACE_FORMAT_R16_FLOAT):
        return VK_FORMAT_R16_SFLOAT;
    case (FFX_SURFACE_FORMAT_R16_UINT):
        return VK_FORMAT_R16_UINT;
    case (FFX_SURFACE_FORMAT_R16_UNORM):
        return VK_FORMAT_R16_UNORM;
    case (FFX_SURFACE_FORMAT_R16_SNORM):
        return VK_FORMAT_R16_SNORM;
    case (FFX_SURFACE_FORMAT_R8_UNORM):
        return VK_FORMAT_R8_UNORM;
    case (FFX_SURFACE_FORMAT_R8_SNORM):
        return VK_FORMAT_R8_SNORM;
    case (FFX_SURFACE_FORMAT_R8G8_UNORM):
        return VK_FORMAT_R8G8_UNORM;
    case (FFX_SURFACE_FORMAT_R8G8_UINT):
        return VK_FORMAT_R8G8_UINT;
    case (FFX_SURFACE_FORMAT_R8G8_SINT):
        return VK_FORMAT_R8G8_SINT;
    case (FFX_SURFACE_FORMAT_R32_FLOAT):
        return VK_FORMAT_R32_SFLOAT;
    case (FFX_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP):
        return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;

    default:
        FFX_ASSERT_MESSAGE(false, "Format not yet supported");
        return VK_FORMAT_UNDEFINED;
    }
}

VkFormat ffxGetVKUAVFormatFromSurfaceFormat(FfxSurfaceFormat fmt)
{
    switch (fmt)
    {
    case (FFX_SURFACE_FORMAT_UNKNOWN):
        return VK_FORMAT_UNDEFINED;
    case (FFX_SURFACE_FORMAT_R32G32B32A32_TYPELESS):
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    case (FFX_SURFACE_FORMAT_R32G32B32A32_UINT):
        return VK_FORMAT_R32G32B32A32_UINT;
    case (FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT):
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    case (FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT):
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    case (FFX_SURFACE_FORMAT_R32G32B32_FLOAT):
        return VK_FORMAT_R32G32B32_SFLOAT;
    case (FFX_SURFACE_FORMAT_R32G32_FLOAT):
        return VK_FORMAT_R32G32_SFLOAT;
    case (FFX_SURFACE_FORMAT_R8_UINT):
        return VK_FORMAT_R8_UINT;
    case (FFX_SURFACE_FORMAT_R8_SINT):
        return VK_FORMAT_R8_SINT;
    case (FFX_SURFACE_FORMAT_R32_UINT):
        return VK_FORMAT_R32_UINT;
    case (FFX_SURFACE_FORMAT_R8G8B8A8_TYPELESS):
    case (FFX_SURFACE_FORMAT_R8G8B8A8_UNORM):
    case (FFX_SURFACE_FORMAT_R8G8B8A8_SRGB):
        return VK_FORMAT_R8G8B8A8_UNORM;
    case (FFX_SURFACE_FORMAT_R8G8B8A8_SNORM):
        return VK_FORMAT_R8G8B8A8_SNORM;
    case (FFX_SURFACE_FORMAT_B8G8R8A8_TYPELESS):
    case (FFX_SURFACE_FORMAT_B8G8R8A8_UNORM):
    case (FFX_SURFACE_FORMAT_B8G8R8A8_SRGB):
        return VK_FORMAT_B8G8R8A8_UNORM;
    case (FFX_SURFACE_FORMAT_R11G11B10_FLOAT):
        return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case (FFX_SURFACE_FORMAT_R10G10B10A2_UNORM):
        return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case (FFX_SURFACE_FORMAT_R16G16_FLOAT):
        return VK_FORMAT_R16G16_SFLOAT;
    case (FFX_SURFACE_FORMAT_R16G16_UINT):
        return VK_FORMAT_R16G16_UINT;
    case (FFX_SURFACE_FORMAT_R16G16_SINT):
        return VK_FORMAT_R16G16_SINT;
    case (FFX_SURFACE_FORMAT_R16_FLOAT):
        return VK_FORMAT_R16_SFLOAT;
    case (FFX_SURFACE_FORMAT_R16_UINT):
        return VK_FORMAT_R16_UINT;
    case (FFX_SURFACE_FORMAT_R16_UNORM):
        return VK_FORMAT_R16_UNORM;
    case (FFX_SURFACE_FORMAT_R16_SNORM):
        return VK_FORMAT_R16_SNORM;
    case (FFX_SURFACE_FORMAT_R8_UNORM):
        return VK_FORMAT_R8_UNORM;
    case (FFX_SURFACE_FORMAT_R8_SNORM):
        return VK_FORMAT_R8_SNORM;
    case (FFX_SURFACE_FORMAT_R8G8_UNORM):
        return VK_FORMAT_R8G8_UNORM;
    case (FFX_SURFACE_FORMAT_R8G8_UINT):
        return VK_FORMAT_R8G8_UINT;
    case (FFX_SURFACE_FORMAT_R8G8_SINT):
        return VK_FORMAT_R8G8_SINT;
    case (FFX_SURFACE_FORMAT_R32_FLOAT):
        return VK_FORMAT_R32_SFLOAT;
    case (FFX_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP):
        return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;

    default:
        FFX_ASSERT_MESSAGE(false, "Format not yet supported");
        return VK_FORMAT_UNDEFINED;
    }
}

FfxSurfaceFormat ffxGetSurfaceFormatVK(VkFormat fmt)
{
    switch (fmt)
    {
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT;
    case VK_FORMAT_R32G32B32A32_UINT:
        return FFX_SURFACE_FORMAT_R32G32B32A32_UINT;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    case VK_FORMAT_R32G32_SFLOAT:
        return FFX_SURFACE_FORMAT_R32G32_FLOAT;
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
        return FFX_SURFACE_FORMAT_R32_UINT;
    case VK_FORMAT_R8G8B8A8_UNORM:
        return FFX_SURFACE_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_R8G8B8A8_SNORM:
        return FFX_SURFACE_FORMAT_R8G8B8A8_SNORM;
    case VK_FORMAT_R8G8B8A8_SRGB:
        return FFX_SURFACE_FORMAT_R8G8B8A8_SRGB;
    case VK_FORMAT_B8G8R8A8_UNORM:
        return FFX_SURFACE_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB:
        return FFX_SURFACE_FORMAT_B8G8R8A8_SRGB;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        return FFX_SURFACE_FORMAT_R11G11B10_FLOAT;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        return FFX_SURFACE_FORMAT_R10G10B10A2_UNORM;
    case VK_FORMAT_R16G16_SFLOAT:
        return FFX_SURFACE_FORMAT_R16G16_FLOAT;
    case VK_FORMAT_R16G16_UINT:
        return FFX_SURFACE_FORMAT_R16G16_UINT;
    case VK_FORMAT_R16G16_SINT:
        return FFX_SURFACE_FORMAT_R16G16_SINT;
    case VK_FORMAT_R16_SFLOAT:
        return FFX_SURFACE_FORMAT_R16_FLOAT;
    case VK_FORMAT_R16_UINT:
        return FFX_SURFACE_FORMAT_R16_UINT;
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D16_UNORM_S8_UINT:
        return FFX_SURFACE_FORMAT_R16_UNORM;
    case VK_FORMAT_R16_SNORM:
        return FFX_SURFACE_FORMAT_R16_SNORM;
    case VK_FORMAT_R8_UNORM:
        return FFX_SURFACE_FORMAT_R8_UNORM;
    case VK_FORMAT_R8_SNORM:
        return FFX_SURFACE_FORMAT_R8_SNORM;
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_S8_UINT:
        return FFX_SURFACE_FORMAT_R8_UINT;
    case VK_FORMAT_R8G8_UNORM:
        return FFX_SURFACE_FORMAT_R8G8_UNORM;
    case VK_FORMAT_R8G8_UINT:
        return FFX_SURFACE_FORMAT_R8G8_UINT;
    case VK_FORMAT_R8G8_SINT:
        return FFX_SURFACE_FORMAT_R8G8_SINT;
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return FFX_SURFACE_FORMAT_R32_FLOAT;
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
        return FFX_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP;
    case VK_FORMAT_UNDEFINED:
        return FFX_SURFACE_FORMAT_UNKNOWN;

    default:
        // NOTE: we do not support typeless formats here
        FFX_ASSERT_MESSAGE(false, "Format not yet supported");
        return FFX_SURFACE_FORMAT_UNKNOWN;
    }
}

bool isDepthFormat(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return true;

    default:
        return false;
    }
}

bool isStencilFormat(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_S8_UINT:
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return true;

    default:
        return false;
    }
}

VkDataGraphPipelineNodeConnectionTypeARM ffxGetVkOpticalFlowConnectionType(const FfxOpticalFlowConnectionType& connectionType)
{
    switch (connectionType)
    {
    case FFX_OPTICAL_FLOW_CONNECTION_INPUT:
        return VK_DATA_GRAPH_PIPELINE_NODE_CONNECTION_TYPE_OPTICAL_FLOW_INPUT_ARM;
    case FFX_OPTICAL_FLOW_CONNECTION_REFERENCE:
        return VK_DATA_GRAPH_PIPELINE_NODE_CONNECTION_TYPE_OPTICAL_FLOW_REFERENCE_ARM;
    case FFX_OPTICAL_FLOW_CONNECTION_HINT:
        return VK_DATA_GRAPH_PIPELINE_NODE_CONNECTION_TYPE_OPTICAL_FLOW_HINT_ARM;
    case FFX_OPTICAL_FLOW_CONNECTION_FLOW_VECTOR:
        return VK_DATA_GRAPH_PIPELINE_NODE_CONNECTION_TYPE_OPTICAL_FLOW_FLOW_VECTOR_ARM;
    case FFX_OPTICAL_FLOW_CONNECTION_COST:
        return VK_DATA_GRAPH_PIPELINE_NODE_CONNECTION_TYPE_OPTICAL_FLOW_COST_ARM;
    case FFX_OPTICAL_FLOW_CONNECTION_BACKWARD_FLOW_VECTOR:
    case FFX_OPTICAL_FLOW_CONNECTION_BACKWARD_COST:
    case FFX_OPTICAL_FLOW_CONNECTION_UNKNOWN:
    default:
        FFX_ASSERT_MESSAGE(false, "Unknown/unsupported optical flow connection type");
        return VK_DATA_GRAPH_PIPELINE_NODE_CONNECTION_TYPE_MAX_ENUM_ARM;
    }
}

VkDataGraphOpticalFlowPerformanceLevelARM ffxGetVkOpticalFlowPerformanceLevel(const FfxOpticalFlowPerformanceLevel& performanceLevel)
{
    switch (performanceLevel)
    {
    case FFX_OPTICAL_FLOW_PERFORMANCE_LEVEL_SLOW:
        return VK_DATA_GRAPH_OPTICAL_FLOW_PERFORMANCE_LEVEL_SLOW_ARM;
    case FFX_OPTICAL_FLOW_PERFORMANCE_LEVEL_MEDIUM:
        return VK_DATA_GRAPH_OPTICAL_FLOW_PERFORMANCE_LEVEL_MEDIUM_ARM;
    case FFX_OPTICAL_FLOW_PERFORMANCE_LEVEL_FAST:
        return VK_DATA_GRAPH_OPTICAL_FLOW_PERFORMANCE_LEVEL_FAST_ARM;
    case FFX_OPTICAL_FLOW_PERFORMANCE_LEVEL_UNKNOWN:
    default:
        return VK_DATA_GRAPH_OPTICAL_FLOW_PERFORMANCE_LEVEL_UNKNOWN_ARM;
    }
}

VkDataGraphOpticalFlowGridSizeFlagsARM ffxGetVkOpticalFlowGridSize(const FfxOpticalFlowGridSize& gridSize)
{
    static_assert(static_cast<uint32_t>(FFX_OPTICAL_FLOW_GRID_SIZE_UNKNOWN) == static_cast<uint32_t>(VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_UNKNOWN_ARM));
    static_assert(static_cast<uint32_t>(FFX_OPTICAL_FLOW_GRID_SIZE_1X1) == static_cast<uint32_t>(VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_1X1_BIT_ARM));
    static_assert(static_cast<uint32_t>(FFX_OPTICAL_FLOW_GRID_SIZE_2X2) == static_cast<uint32_t>(VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_2X2_BIT_ARM));
    static_assert(static_cast<uint32_t>(FFX_OPTICAL_FLOW_GRID_SIZE_4X4) == static_cast<uint32_t>(VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_ARM));
    static_assert(static_cast<uint32_t>(FFX_OPTICAL_FLOW_GRID_SIZE_8X8) == static_cast<uint32_t>(VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_8X8_BIT_ARM));

    return static_cast<VkDataGraphOpticalFlowGridSizeFlagsARM>(gridSize);
}

uint32_t ffxGetVkOpticalFlowExecuteFlags(const uint32_t executeFlags)
{
    uint32_t flags = 0;
    if (executeFlags & FFX_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_DISABLE_TEMPORAL_HINTS)
    {
        flags |= VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_DISABLE_TEMPORAL_HINTS_BIT_ARM;
    }
    if (executeFlags & FFX_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_INPUT_UNCHANGED)
    {
        flags |= VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_INPUT_UNCHANGED_BIT_ARM;
    }
    if (executeFlags & FFX_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_REFERENCE_UNCHANGED)
    {
        flags |= VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_REFERENCE_UNCHANGED_BIT_ARM;
    }
    if (executeFlags & FFX_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_INPUT_IS_PREVIOUS_REFERENCE)
    {
        flags |= VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_INPUT_IS_PREVIOUS_REFERENCE_BIT_ARM;
    }
    if (executeFlags & FFX_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_REFERENCE_IS_PREVIOUS_INPUT)
    {
        flags |= VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_REFERENCE_IS_PREVIOUS_INPUT_BIT_ARM;
    }
    return flags;
}

// For tensor aliased images, we need to ensure that the format is compatible with the number of channels and bits per pixel.
// If the format is not supported, we return VK_FORMAT_UNDEFINED.
VkFormat ffxGetVkFormatForTensorAliasedImage(FfxSurfaceFormat tensorFormat, const uint64_t channel)
{
    VkFormat vkTensorFormat = ffxGetVkFormatFromSurfaceFormat(tensorFormat);

    // NOTE:
    // For 1 byte per pixel formats, we use the SNORM version of the format, even though the underlying tensor data is SINT.
    // This is to allow hardware sampling using a linear sampler (linear sampling not allowed with R8_SINT format).
    // The sampled SNORM value will be dequantized to a float value in the shader taking into consideration the implicit conversion from SINT.

    if (channel == 4)
    {
        switch (vkTensorFormat)
        {
        case VK_FORMAT_R32_SFLOAT:
            return VK_FORMAT_R32G32B32A32_SFLOAT;
        case VK_FORMAT_R8_SINT:
            return VK_FORMAT_R8G8B8A8_SNORM;
        default:
            FFX_ASSERT_MESSAGE(false, "Tensor format not yet supported");
            return VK_FORMAT_UNDEFINED;
        }
    }
    else if (channel == 1)
    {
        switch (vkTensorFormat)
        {
        case VK_FORMAT_R32_SFLOAT:
            return VK_FORMAT_R32_SFLOAT;
        case VK_FORMAT_R8_SINT:
            return VK_FORMAT_R8_SNORM;
        default:
            FFX_ASSERT_MESSAGE(false, "Tensor format not yet supported");
            return VK_FORMAT_UNDEFINED;
        }
    }
    else
    {
        FFX_ASSERT_MESSAGE(false, "Dimensions of tensor not yet supported in image aliasing.");
        return VK_FORMAT_UNDEFINED;
    }
}

VkImageAspectFlags getImageAspect(FfxResourceUsage usage)
{
    // No VK_IMAGE_ASPECT_STENCIL_BIT support for now
    if (FFX_CONTAINS_FLAG(usage, FFX_RESOURCE_USAGE_DEPTHTARGET))
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    else
        return VK_IMAGE_ASPECT_COLOR_BIT;
}

VkFormat getVkFormatFromSurfaceFormatAndUsage(FfxSurfaceFormat fmt, FfxResourceUsage usage)
{
    VkFormat format     = ffxGetVkFormatFromSurfaceFormat(fmt);
    bool     hasDepth   = FFX_CONTAINS_FLAG(usage, FFX_RESOURCE_USAGE_DEPTHTARGET);
    bool     hasStencil = FFX_CONTAINS_FLAG(usage, FFX_RESOURCE_USAGE_STENCILTARGET);
    if (hasDepth || hasStencil)
    {
        switch (fmt)
        {
        case FFX_SURFACE_FORMAT_R16_UNORM:
        {
            if (hasDepth && hasStencil)
                return VK_FORMAT_D16_UNORM_S8_UINT;
            else if (hasDepth)
                return VK_FORMAT_D16_UNORM;
            // stencil only doesn't exist
            break;
        }
        case FFX_SURFACE_FORMAT_R32_FLOAT:
        {
            if (hasDepth && hasStencil)
                return VK_FORMAT_D32_SFLOAT_S8_UINT;
            else if (hasDepth)
                return VK_FORMAT_D32_SFLOAT;
            // stencil only doesn't exist
            break;
        }
        case FFX_SURFACE_FORMAT_R32_UINT:
        {
            if (hasDepth && hasStencil)
                return VK_FORMAT_D24_UNORM_S8_UINT;
            else if (hasDepth)
                return VK_FORMAT_X8_D24_UNORM_PACK32;
            // stencil only doesn't exist
            break;
        }
        case FFX_SURFACE_FORMAT_R8_UINT:
        {
            if (hasStencil)
                return VK_FORMAT_S8_UINT;
            break;
        }
        default:
            break;
        }
    }

    return format;
}

VkImageUsageFlags getVKImageUsageFlagsFromResourceUsage(FfxResourceUsage flags)
{
    VkImageUsageFlags ret = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (FFX_CONTAINS_FLAG(flags, FFX_RESOURCE_USAGE_RENDERTARGET))
        ret |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (FFX_CONTAINS_FLAG(flags, FFX_RESOURCE_USAGE_UAV))
        ret |= (VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    return ret;
}

FfxResourceDescription ffxGetBufferResourceDescriptionVK(const VkBuffer           buffer,
                                                         const VkBufferCreateInfo createInfo,
                                                         FfxResourceUsage         additionalUsages /*=FFX_RESOURCE_USAGE_READ_ONLY*/)
{
    FfxResourceDescription resourceDescription = {};

    // This is valid
    if (buffer == VK_NULL_HANDLE)
        return resourceDescription;

    resourceDescription.flags  = FFX_RESOURCE_FLAGS_NONE;
    resourceDescription.usage  = additionalUsages;
    resourceDescription.size   = (uint32_t)createInfo.size;
    resourceDescription.stride = 0;
    resourceDescription.format = FFX_SURFACE_FORMAT_UNKNOWN;

    if (FFX_CONTAINS_FLAG(createInfo.usage, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT))
        resourceDescription.usage = (FfxResourceUsage)(resourceDescription.usage | FFX_RESOURCE_USAGE_UAV);
    if (FFX_CONTAINS_FLAG(createInfo.usage, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT))
        resourceDescription.usage = (FfxResourceUsage)(resourceDescription.usage | FFX_RESOURCE_USAGE_INDIRECT);

    // What should we initialize this to?? No case for this yet
    resourceDescription.depth    = 0;
    resourceDescription.mipCount = 0;

    // Set the type
    resourceDescription.type = FFX_RESOURCE_TYPE_BUFFER;

    return resourceDescription;
}

FfxResourceDescription ffxGetImageResourceDescriptionVK(const VkImage           image,
                                                        const VkImageCreateInfo createInfo,
                                                        FfxResourceUsage        additionalUsages /*=FFX_RESOURCE_USAGE_READ_ONLY*/)
{
    FfxResourceDescription resourceDescription = {};

    // This is valid
    if (image == VK_NULL_HANDLE)
        return resourceDescription;

    // Set flags properly for resource registration
    resourceDescription.flags = FFX_RESOURCE_FLAGS_NONE;
    resourceDescription.usage = FFX_RESOURCE_USAGE_READ_ONLY;

    // Check for depth and stencil use
    if (isDepthFormat(createInfo.format))
        resourceDescription.usage = (FfxResourceUsage)(resourceDescription.usage | FFX_RESOURCE_USAGE_DEPTHTARGET);
    if (isStencilFormat(createInfo.format))
        resourceDescription.usage = (FfxResourceUsage)(resourceDescription.usage | FFX_RESOURCE_USAGE_STENCILTARGET);

    // Unordered access use
    if (FFX_CONTAINS_FLAG(createInfo.usage, VK_IMAGE_USAGE_STORAGE_BIT))
        resourceDescription.usage = (FfxResourceUsage)(resourceDescription.usage | FFX_RESOURCE_USAGE_UAV);

    // Resource-specific supplemental use flags
    resourceDescription.usage = (FfxResourceUsage)(resourceDescription.usage | additionalUsages);

    resourceDescription.width    = createInfo.extent.width;
    resourceDescription.height   = createInfo.extent.height;
    resourceDescription.mipCount = createInfo.mipLevels;
    resourceDescription.format   = ffxGetSurfaceFormatVK(createInfo.format);

    // if the mutable flag is present, assume that the real format is sRGB
    if (FFX_CONTAINS_FLAG(createInfo.flags, VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT))
        resourceDescription.format = ffxGetSurfaceFormatToGamma(resourceDescription.format);

    switch (createInfo.imageType)
    {
    case VK_IMAGE_TYPE_1D:
        resourceDescription.type = FFX_RESOURCE_TYPE_TEXTURE1D;
        break;
    case VK_IMAGE_TYPE_2D:
        resourceDescription.depth = createInfo.arrayLayers;
        if (FFX_CONTAINS_FLAG(additionalUsages, FFX_RESOURCE_USAGE_ARRAYVIEW))
            resourceDescription.type = FFX_RESOURCE_TYPE_TEXTURE2D;
        else if (FFX_CONTAINS_FLAG(createInfo.flags, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT))
            resourceDescription.type = FFX_RESOURCE_TYPE_TEXTURE_CUBE;
        else
            resourceDescription.type = FFX_RESOURCE_TYPE_TEXTURE2D;
        break;
    case VK_IMAGE_TYPE_3D:
        resourceDescription.depth = createInfo.extent.depth;
        resourceDescription.type  = FFX_RESOURCE_TYPE_TEXTURE3D;
        break;
    default:
        FFX_ASSERT_MESSAGE(false, "FFXInterface: VK: Unsupported texture dimension requested. Please implement.");
        break;
    }

    return resourceDescription;
}

FfxErrorCode allocateDeviceMemory(BackendContext_VK*           backendContext,
                                  VkMemoryRequirements         memRequirements,
                                  VkMemoryPropertyFlags        requiredMemoryProperties,
                                  BackendContext_VK::Resource* backendResource)
{
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize  = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryTypeIndex(backendContext, memRequirements, requiredMemoryProperties, backendResource->memoryProperties);

    if (allocInfo.memoryTypeIndex == UINT32_MAX)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    VkResult result = VulkanWrapper().vkAllocateMemory(backendContext->device, &allocInfo, nullptr, &backendResource->deviceMemory);

    if (result != VK_SUCCESS)
    {
        switch (result)
        {
        case (VK_ERROR_OUT_OF_HOST_MEMORY):
        case (VK_ERROR_OUT_OF_DEVICE_MEMORY):
            return FFX_ERROR_OUT_OF_MEMORY;
        default:
            return FFX_ERROR_BACKEND_API_ERROR;
        }
    }

    return FFX_OK;
}

void setVKObjectName(VkDevice device, VkObjectType objectType, uint64_t object, const char* name)
{
    if (!VulkanWrapper().vkSetDebugUtilsObjectNameEXT)
    {
        return;
    }

    VkDebugUtilsObjectNameInfoEXT s{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr, objectType, object, name};
    VulkanWrapper().vkSetDebugUtilsObjectNameEXT(device, &s);
}

uint32_t getDynamicResourcesStartIndex(uint32_t effectContextId)
{
    // dynamic resources are tracked from the max index
    return (effectContextId * FFX_MAX_RESOURCE_COUNT) + FFX_MAX_RESOURCE_COUNT - 1;
}
uint32_t getDynamicResourceViewsStartIndex(uint32_t effectContextId, uint32_t frameIndex)
{
    // dynamic resource views are tracked from the max index
    return (effectContextId * FFX_MAX_QUEUED_FRAMES * FFX_MAX_RESOURCE_COUNT * 2) + (frameIndex * FFX_MAX_RESOURCE_COUNT * 2) + (FFX_MAX_RESOURCE_COUNT * 2) -
           1;
}

void destroyDynamicViews(BackendContext_VK* backendContext, uint32_t effectContextId, uint32_t frameIndex)
{
    BackendContext_VK::EffectContext& effectContext = backendContext->pEffectContexts[effectContextId];

    // Release image views for dynamic resources
    const uint32_t dynamicResourceViewIndexStart = getDynamicResourceViewsStartIndex(effectContextId, frameIndex);
    for (uint32_t dynamicViewIndex = effectContext.nextDynamicResourceView[frameIndex] + 1; dynamicViewIndex <= dynamicResourceViewIndexStart;
         ++dynamicViewIndex)
    {
        VulkanWrapper().vkDestroyImageView(backendContext->device, backendContext->pResourceViews[dynamicViewIndex].imageView, VK_NULL_HANDLE);
        backendContext->pResourceViews[dynamicViewIndex].imageView = VK_NULL_HANDLE;
    }
    effectContext.nextDynamicResourceView[frameIndex] = dynamicResourceViewIndexStart;
}

VkAccessFlags2 getVKAccessFlagsFromResourceState(FfxResourceStates state)
{
    switch (state)
    {
    case (FFX_RESOURCE_STATE_COMMON):
        return VK_ACCESS_2_NONE;
    case (FFX_RESOURCE_STATE_GENERIC_READ):
        return VK_ACCESS_2_SHADER_READ_BIT;
    case (FFX_RESOURCE_STATE_GENERIC_UAV):
    case (FFX_RESOURCE_STATE_COMPUTE_UAV):
    case (FFX_RESOURCE_STATE_PIXEL_UAV):
        return VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    case (FFX_RESOURCE_STATE_COMPUTE_READ):
    case (FFX_RESOURCE_STATE_PIXEL_READ):
    case (FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ):
        return VK_ACCESS_2_SHADER_READ_BIT;
    case FFX_RESOURCE_STATE_COPY_SRC:
        return VK_ACCESS_2_TRANSFER_READ_BIT;
    case FFX_RESOURCE_STATE_COPY_DEST:
        return VK_ACCESS_2_TRANSFER_WRITE_BIT;
    case FFX_RESOURCE_STATE_INDIRECT_ARGUMENT:
        return VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    case FFX_RESOURCE_STATE_PRESENT:
        return VK_ACCESS_2_NONE;
    case FFX_RESOURCE_STATE_RENDER_TARGET:
        return VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
    case FFX_RESOURCE_STATE_DATA_GRAPH_READ:
        return VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM;
    case FFX_RESOURCE_STATE_DATA_GRAPH_WRITE:
        return VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM;
    case FFX_RESOURCE_STATE_DEPTH_ATTACHEMENT:
        return VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    default:
        FFX_ASSERT_MESSAGE(false, "State flag not yet supported");
        return VK_ACCESS_2_SHADER_READ_BIT;
    }
}

VkPipelineStageFlags2 getVKPipelineStageFlagsFromResourceState(FfxResourceStates state)
{
    switch (state)
    {
    case (FFX_RESOURCE_STATE_COMMON):
    case (FFX_RESOURCE_STATE_GENERIC_READ):
    case (FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ):
    case (FFX_RESOURCE_STATE_GENERIC_UAV):
        return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    case (FFX_RESOURCE_STATE_PIXEL_UAV):
        return VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    case (FFX_RESOURCE_STATE_COMPUTE_UAV):
    case (FFX_RESOURCE_STATE_COMPUTE_READ):
        return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    case (FFX_RESOURCE_STATE_PIXEL_READ):
        return VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    case (FFX_RESOURCE_STATE_INDIRECT_ARGUMENT):
        return VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    case FFX_RESOURCE_STATE_COPY_SRC:
    case FFX_RESOURCE_STATE_COPY_DEST:
        return VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    case FFX_RESOURCE_STATE_PRESENT:
        return VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    case FFX_RESOURCE_STATE_RENDER_TARGET:
        return VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    case FFX_RESOURCE_STATE_DATA_GRAPH_READ:
    case FFX_RESOURCE_STATE_DATA_GRAPH_WRITE:
        return VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM;
    case (FFX_RESOURCE_STATE_DEPTH_ATTACHEMENT):
        return VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    default:
        FFX_ASSERT_MESSAGE(false, "Pipeline stage flag not yet supported");
        return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    }
}

VkImageLayout getVKImageLayoutFromResourceState(FfxResourceStates state)
{
    switch (state)
    {
    case (FFX_RESOURCE_STATE_COMMON):
        return VK_IMAGE_LAYOUT_GENERAL;
    case (FFX_RESOURCE_STATE_GENERIC_READ):
        return VK_IMAGE_LAYOUT_GENERAL;
    case (FFX_RESOURCE_STATE_GENERIC_UAV):
    case (FFX_RESOURCE_STATE_COMPUTE_UAV):
    case (FFX_RESOURCE_STATE_PIXEL_UAV):
        return VK_IMAGE_LAYOUT_GENERAL;
    case (FFX_RESOURCE_STATE_COMPUTE_READ):
    case (FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ):
    case (FFX_RESOURCE_STATE_PIXEL_READ):
        return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    case FFX_RESOURCE_STATE_COPY_SRC:
        return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    case FFX_RESOURCE_STATE_COPY_DEST:
        return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    case FFX_RESOURCE_STATE_PRESENT:
        return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    case FFX_RESOURCE_STATE_RENDER_TARGET:
        return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    case FFX_RESOURCE_STATE_DATA_GRAPH_READ:
    case FFX_RESOURCE_STATE_DATA_GRAPH_WRITE:
        return VK_IMAGE_LAYOUT_TENSOR_ALIASING_ARM;
    case FFX_RESOURCE_STATE_DEPTH_ATTACHEMENT:
        return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    case FFX_RESOURCE_STATE_INDIRECT_ARGUMENT:
        // this case is for buffers
    default:
        FFX_ASSERT_MESSAGE(false, "Image layout flag not yet supported");
        return VK_IMAGE_LAYOUT_GENERAL;
    }
}

void setupImageViewForSRV(VkImageViewCreateInfo&      imageViewCreateInfo,
                          VkImageViewUsageCreateInfo& imageViewUsageCreateInfo,
                          FfxResourceDescription      resourceDescription)
{
    if (ffxIsSurfaceFormatSRGB(resourceDescription.format) && FFX_CONTAINS_FLAG(resourceDescription.usage, FFX_RESOURCE_USAGE_UAV))
    {
        // mutable is only for sRGB textures that will need a storage
        imageViewUsageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
        imageViewUsageCreateInfo.pNext = nullptr;
        imageViewUsageCreateInfo.usage = getVKImageUsageFlagsFromResourceUsage(FFX_RESOURCE_USAGE_READ_ONLY);  // we can assume that SRV is enough
        imageViewCreateInfo.pNext      = &imageViewUsageCreateInfo;
    }
}

void copyResourceState(BackendContext_VK::Resource* backendResource, const FfxResource* inFfxResource)
{
    FfxResourceStates state = inFfxResource->state;

    // copy the new states
    backendResource->initialState = state;
    backendResource->currentState = state;
    backendResource->undefined    = false;
    backendResource->dynamic      = true;

    // If the internal resource state is undefined, that means we are importing a resource that
    // has not yet been initialized, so tag the resource as undefined so we can transition it accordingly.
    if (backendResource->resourceDescription.flags & FFX_RESOURCE_FLAGS_UNDEFINED)
    {
        backendResource->undefined                 = true;
        backendResource->resourceDescription.flags = (FfxResourceFlags)((int)backendResource->resourceDescription.flags & ~FFX_RESOURCE_FLAGS_UNDEFINED);
    }
}

void beginMarkerVK(BackendContext_VK* backendContext, VkCommandBuffer commandBuffer, const char* label)
{
    if (VulkanWrapper().vkCmdBeginDebugUtilsLabelEXT == nullptr)
    {
        // Debug markers are not supported, so we return early.
        return;
    }

    VkDebugUtilsLabelEXT debugLabel = {};
    debugLabel.sType                = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    debugLabel.pNext                = nullptr;
    debugLabel.pLabelName           = label;

    // not to saturated red
    debugLabel.color[0] = 1.0f;
    debugLabel.color[1] = 0.14f;
    debugLabel.color[2] = 0.14f;
    debugLabel.color[3] = 1.0f;
    VulkanWrapper().vkCmdBeginDebugUtilsLabelEXT(commandBuffer, &debugLabel);
}

void endMarkerVK(BackendContext_VK* backendContext, VkCommandBuffer commandBuffer)
{
    if (VulkanWrapper().vkCmdEndDebugUtilsLabelEXT == nullptr)
    {
        // Debug markers are not supported, so we return early.
        return;
    }

    VulkanWrapper().vkCmdEndDebugUtilsLabelEXT(commandBuffer);
}

void addBarrier(BackendContext_VK* backendContext, FfxResourceInternal* resource, FfxResourceStates newState)
{
    FFX_ASSERT(NULL != backendContext);
    FFX_ASSERT(NULL != resource);

    BackendContext_VK::Resource& ffxResource = backendContext->pResources[resource->internalIndex];
    FfxResourceStates&           curState    = backendContext->pResources[resource->internalIndex].currentState;

    auto addBufferBarrier = [&](VkBuffer buffer) {
        VkBufferMemoryBarrier2* barrier = &backendContext->bufferMemoryBarriers[backendContext->scheduledBufferBarrierCount];

        barrier->sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barrier->pNext               = nullptr;
        barrier->srcStageMask        = getVKPipelineStageFlagsFromResourceState(curState);
        barrier->srcAccessMask       = getVKAccessFlagsFromResourceState(curState);
        barrier->dstStageMask        = getVKPipelineStageFlagsFromResourceState(newState);
        barrier->dstAccessMask       = getVKAccessFlagsFromResourceState(newState);
        barrier->srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier->dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier->buffer              = buffer;
        barrier->offset              = 0;
        barrier->size                = VK_WHOLE_SIZE;

        ++backendContext->scheduledBufferBarrierCount;
    };

    auto addImageBarrier = [&](VkImage image) {
        VkImageMemoryBarrier2* barrier = &backendContext->imageMemoryBarriers[backendContext->scheduledImageBarrierCount];

        VkImageSubresourceRange range;
        range.aspectMask     = getImageAspect(ffxResource.resourceDescription.usage);
        range.baseMipLevel   = 0;
        range.levelCount     = VK_REMAINING_MIP_LEVELS;
        range.baseArrayLayer = 0;
        range.layerCount     = VK_REMAINING_ARRAY_LAYERS;

        barrier->sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier->pNext               = nullptr;
        barrier->srcStageMask        = getVKPipelineStageFlagsFromResourceState(curState);
        barrier->srcAccessMask       = getVKAccessFlagsFromResourceState(curState);
        barrier->dstStageMask        = getVKPipelineStageFlagsFromResourceState(newState);
        barrier->dstAccessMask       = getVKAccessFlagsFromResourceState(newState);
        barrier->oldLayout           = ffxResource.undefined ? VK_IMAGE_LAYOUT_UNDEFINED : getVKImageLayoutFromResourceState(curState);
        barrier->newLayout           = getVKImageLayoutFromResourceState(newState);
        barrier->srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier->dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier->image               = image;
        barrier->subresourceRange    = range;

        ++backendContext->scheduledImageBarrierCount;
    };

    if (ffxResource.resourceDescription.type == FFX_RESOURCE_TYPE_BUFFER)
    {
        addBufferBarrier(ffxResource.bufferResource);
    }
    else if (ffxResource.resourceDescription.type == FFX_RESOURCE_TYPE_TENSOR)
    {
        // Tensor barrier
        VkTensorARM               vkResource = ffxResource.tensorResource;
        VkTensorMemoryBarrierARM* barrier    = &backendContext->tensorMemoryBarriers[backendContext->scheduledTensorBarrierCount];

        barrier->sType               = VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_ARM;
        barrier->pNext               = nullptr;
        barrier->srcStageMask        = getVKPipelineStageFlagsFromResourceState(curState);
        barrier->srcAccessMask       = getVKAccessFlagsFromResourceState(curState);
        barrier->dstStageMask        = getVKPipelineStageFlagsFromResourceState(newState);
        barrier->dstAccessMask       = getVKAccessFlagsFromResourceState(newState);
        barrier->srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier->dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier->tensor              = vkResource;

        ++backendContext->scheduledTensorBarrierCount;

        const auto flags         = ffxResource.resourceDescription.flags;
        const bool imageAliased  = (flags & FFX_RESOURCE_FLAGS_IMAGE_ALIASED) == FFX_RESOURCE_FLAGS_IMAGE_ALIASED;
        const bool bufferAliased = (flags & FFX_RESOURCE_FLAGS_BUFFER_ALIASED) == FFX_RESOURCE_FLAGS_BUFFER_ALIASED;

        if (imageAliased && ffxResource.aliasedTensorImageResource != VK_NULL_HANDLE)
        {
            addImageBarrier(ffxResource.aliasedTensorImageResource);
        }

        if (bufferAliased && ffxResource.aliasedTensorBufferResource != VK_NULL_HANDLE)
        {
            addBufferBarrier(ffxResource.aliasedTensorBufferResource);
        }
    }
    else
    {
        addImageBarrier(ffxResource.imageResource);

        if (ffxResource.aliasedBufferResource != VK_NULL_HANDLE)
        {
            addBufferBarrier(ffxResource.aliasedBufferResource);
        }
    }

    curState = newState;

    if (ffxResource.undefined)
        ffxResource.undefined = false;
}

void flushBarriers(BackendContext_VK* backendContext, VkCommandBuffer vkCommandBuffer)
{
    FFX_ASSERT(NULL != backendContext);
    FFX_ASSERT(NULL != vkCommandBuffer);

    uint32_t totalCount =
        backendContext->scheduledImageBarrierCount + backendContext->scheduledBufferBarrierCount + backendContext->scheduledTensorBarrierCount;
    if (totalCount)
    {
        const VkTensorDependencyInfoARM tensorDependencyInfo = {VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_ARM,
                                                                nullptr,
                                                                static_cast<uint32_t>(backendContext->scheduledTensorBarrierCount),
                                                                backendContext->tensorMemoryBarriers};

        const VkDependencyInfo dependencyInfo = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                                 backendContext->scheduledTensorBarrierCount != 0 ? &tensorDependencyInfo : NULL,
                                                 VK_DEPENDENCY_BY_REGION_BIT,  // dependencyFlags
                                                 0,                            // memoryBarrierCount
                                                 nullptr,                      // pMemoryBarriers
                                                 backendContext->scheduledBufferBarrierCount,
                                                 backendContext->bufferMemoryBarriers,
                                                 backendContext->scheduledImageBarrierCount,
                                                 backendContext->imageMemoryBarriers};

        VulkanWrapper().vkCmdPipelineBarrier2(vkCommandBuffer, &dependencyInfo);

        backendContext->scheduledTensorBarrierCount = 0;
        backendContext->scheduledImageBarrierCount  = 0;
        backendContext->scheduledBufferBarrierCount = 0;
    }
}

FfxConstantAllocation BackendContext_VK::FallbackConstantAllocator(void* data, FfxUInt64 dataSize)
{
    FfxConstantAllocation       allocation;
    std::lock_guard<std::mutex> cbLock{uniformBufferMutex};
    memset(&allocation, 0, sizeof(FfxConstantAllocation));

    if (!uniformBufferMem)
    {
        // allocate dynamic uniform buffer

        // get alignment
        VkPhysicalDeviceProperties physicalDeviceProperties = {};
        VulkanWrapper().vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
        uniformBufferAlignment = physicalDeviceProperties.limits.minUniformBufferOffsetAlignment;

        uniformBufferSize = FFX_ALIGN_UP(FFX_BUFFER_SIZE, uniformBufferAlignment) * maxEffectContexts * FFX_MAX_PASS_COUNT * FFX_MAX_QUEUED_FRAMES;

        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size               = static_cast<VkDeviceSize>(uniformBufferSize);
        bufferInfo.usage              = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;

        VkResult res = VulkanWrapper().vkCreateBuffer(device, &bufferInfo, NULL, &uniformBuffer);
        FFX_ASSERT(res == VK_SUCCESS);

        VkMemoryAllocateInfo allocInfo = {};
        if (res == VK_SUCCESS)
        {
            // allocate memory block for all uniform buffers
            VkMemoryRequirements memRequirements = {};
            VulkanWrapper().vkGetBufferMemoryRequirements(device, uniformBuffer, &memRequirements);

            // this is the real alignment
            uniformBufferAlignment = memRequirements.alignment;

            VkMemoryPropertyFlags requiredMemoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize  = uniformBufferSize;
            allocInfo.memoryTypeIndex = findMemoryTypeIndex(this, memRequirements, requiredMemoryProperties, uniformBufferMemoryProperties);

            if (allocInfo.memoryTypeIndex == UINT32_MAX)
            {
                requiredMemoryProperties  = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
                allocInfo.memoryTypeIndex = findMemoryTypeIndex(this, memRequirements, requiredMemoryProperties, uniformBufferMemoryProperties);

                if (allocInfo.memoryTypeIndex == UINT32_MAX)
                    res = VK_ERROR_INITIALIZATION_FAILED;
            }
        }

        if (res == VK_SUCCESS)
            res = VulkanWrapper().vkAllocateMemory(device, &allocInfo, nullptr, &uniformBufferMemory);
        FFX_ASSERT(res == VK_SUCCESS);

        if (res == VK_SUCCESS)
            res = VulkanWrapper().vkBindBufferMemory(device, uniformBuffer, uniformBufferMemory, 0);
        FFX_ASSERT(res == VK_SUCCESS);

        // map the memory block
        if (res == VK_SUCCESS)
            res = VulkanWrapper().vkMapMemory(device, uniformBufferMemory, 0, uniformBufferSize, 0, &uniformBufferMem);
        FFX_ASSERT(res == VK_SUCCESS);
    }

    FFX_ASSERT(uniformBufferMem);

    allocation.resource.resource = uniformBuffer;
    allocation.handle            = 0;

    if (data && uniformBufferMem)
    {
        if (uniformBufferOffset + dataSize >= uniformBufferSize)
            uniformBufferOffset = 0;

        allocation.handle = static_cast<FfxUInt64>(uniformBufferOffset);

        void* pBuffer = static_cast<void*>(static_cast<uint8_t*>(uniformBufferMem) + uniformBufferOffset);
        memcpy(pBuffer, data, dataSize);

        // flush mapped range if memory type is not coherent
        if ((uniformBufferMemoryProperties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)
        {
            VkMappedMemoryRange memoryRange;
            memset(&memoryRange, 0, sizeof(memoryRange));

            memoryRange.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            memoryRange.memory = uniformBufferMemory;
            memoryRange.offset = uniformBufferOffset;
            memoryRange.size   = dataSize;

            VulkanWrapper().vkFlushMappedMemoryRanges(device, 1, &memoryRange);
        }

        uniformBufferOffset += FFX_ALIGN_UP(dataSize, uniformBufferAlignment);
        if (uniformBufferOffset > uniformBufferSize)
            uniformBufferOffset = 0;
    }

    return allocation;
}

void resetBackendContext(BackendContext_VK* backendContext)
{
    // reset the context except the maxEffectContexts in case the memory is reused for a new context
    uint32_t maxEffectContexts = backendContext->maxEffectContexts;

    memset(backendContext, 0, sizeof(BackendContext_VK));

    // restore the maxEffectContexts
    backendContext->maxEffectContexts = maxEffectContexts;
}

//////////////////////////////////////////////////////////////////////////
// VK back end implementation

FfxErrorCode SetupMessageCallback(FfxInterface* backendInterface, FfxBackendMessage ffxMessageCallback)
{
    FFX_ASSERT(NULL != backendInterface);

    backendInterface->fpMessage = ffxMessageCallback;

    return FFX_OK;
}

FfxVersionNumber GetSDKVersionVK(FfxInterface* backendInterface)
{
    return FFX_SDK_MAKE_VERSION(FFX_SDK_VERSION_MAJOR, FFX_SDK_VERSION_MINOR, FFX_SDK_VERSION_PATCH);
}

FfxErrorCode GetEffectGpuMemoryUsageVK(FfxInterface* backendInterface, FfxUInt32 effectContextId, FfxEffectMemoryUsage* outVramUsage)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != outVramUsage);

    BackendContext_VK*                backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;
    BackendContext_VK::EffectContext& effectContext  = backendContext->pEffectContexts[effectContextId];

    *outVramUsage = effectContext.vramUsage;

    return FFX_OK;
}

class VkFuncLoader
{
public:
    explicit VkFuncLoader(VkDeviceContext* vkDeviceContext)
        : getInstanceProcAddrPtr(vkDeviceContext->vkGetInstanceProcAddr)
        , getDeviceProcAddrPtr(vkDeviceContext->vkDeviceProcAddr)
        , device(vkDeviceContext->vkDevice)
        , instance(vkDeviceContext->vkInstance)
    {
        FFX_ASSERT(NULL != getInstanceProcAddrPtr);
        FFX_ASSERT(NULL != getDeviceProcAddrPtr);
        FFX_ASSERT(VK_NULL_HANDLE != device);
        FFX_ASSERT(VK_NULL_HANDLE != instance);
    }

    template <typename T>
    bool getDeviceProc(T& funcPtr, const char* name)
    {
        return getProcInternal(funcPtr, name, getDeviceProcAddrPtr, device);
    }

    template <typename T>
    bool getInstanceProc(T& funcPtr, const char* name)
    {
        return getProcInternal(funcPtr, name, getInstanceProcAddrPtr, instance);
    }

private:
    template <typename T, typename FuncType, typename HandleType>
    bool getProcInternal(T& funcPtr, const char* name, const FuncType& getProcAddrPtr, HandleType handle)
    {
        FFX_ASSERT(NULL != getProcAddrPtr);

        funcPtr = reinterpret_cast<T>(getProcAddrPtr(handle, name));

        if (!funcPtr)
        {
            // Try to load the khr version function if the function is not available
            char aliasNameKHR[128];
            snprintf(aliasNameKHR, sizeof(aliasNameKHR), "%sKHR", name);
            funcPtr = reinterpret_cast<T>(getProcAddrPtr(handle, aliasNameKHR));
        }

        if (!funcPtr)
        {
            // Try to load the ext version function if the function is not available
            char aliasNameEXT[128];
            snprintf(aliasNameEXT, sizeof(aliasNameEXT), "%sEXT", name);
            funcPtr = reinterpret_cast<T>(getProcAddrPtr(handle, aliasNameEXT));
        }

        return funcPtr != nullptr;
    }

    PFN_vkGetInstanceProcAddr getInstanceProcAddrPtr;
    PFN_vkGetDeviceProcAddr   getDeviceProcAddrPtr;
    VkInstance                instance = VK_NULL_HANDLE;
    VkDevice                  device   = VK_NULL_HANDLE;
};

FfxErrorCode CreateBackendContextVK(FfxInterface* backendInterface, FfxEffect effect, FfxEffectBindlessConfig* bindlessConfig, FfxUInt32* effectContextId)
{
    VkDeviceContext* vkDeviceContext = reinterpret_cast<VkDeviceContext*>(backendInterface->device);

    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != vkDeviceContext);
    FFX_ASSERT(VK_NULL_HANDLE != vkDeviceContext->vkDevice);
    FFX_ASSERT(VK_NULL_HANDLE != vkDeviceContext->vkPhysicalDevice);

    // set up some internal resources we need (space for resource views and constant buffers)
    BackendContext_VK* backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;

    // Set things up if this is the first invocation
    if (!backendContext->refCount)
    {
        resetBackendContext(backendContext);

        new (&backendContext->uniformBufferMutex) std::mutex();

        // Map all of our pointers
        uint32_t gpuJobDescArraySize = FFX_ALIGN_UP(backendContext->maxEffectContexts * FFX_MAX_GPU_JOBS * sizeof(FfxGpuJobDescription), sizeof(uint32_t));
        uint32_t resourceViewArraySize =
            FFX_ALIGN_UP(((backendContext->maxEffectContexts * FFX_MAX_QUEUED_FRAMES * FFX_MAX_RESOURCE_COUNT * 2) + FFX_MAX_BINDLESS_DESCRIPTOR_COUNT) *
                             sizeof(BackendContext_VK::VkResourceView),
                         sizeof(uint32_t));
        uint32_t stagingRingBufferArraySize = FFX_ALIGN_UP(backendContext->maxEffectContexts * FFX_CONSTANT_BUFFER_RING_BUFFER_SIZE, sizeof(uint32_t));
        uint32_t pipelineArraySize =
            FFX_ALIGN_UP(backendContext->maxEffectContexts * FFX_MAX_PASS_COUNT * sizeof(BackendContext_VK::PipelineLayout), sizeof(uint32_t));
        uint32_t resourceArraySize =
            FFX_ALIGN_UP(backendContext->maxEffectContexts * FFX_MAX_RESOURCE_COUNT * sizeof(BackendContext_VK::Resource), sizeof(uint32_t));
        uint32_t contextArraySize = FFX_ALIGN_UP(backendContext->maxEffectContexts * sizeof(BackendContext_VK::EffectContext), sizeof(uint32_t));
        uint8_t* pMem             = (uint8_t*)((BackendContext_VK*)(backendContext + 1));

        // Map gpu job array
        backendContext->pGpuJobs = (FfxGpuJobDescription*)pMem;
        memset(backendContext->pGpuJobs, 0, gpuJobDescArraySize);
        pMem += gpuJobDescArraySize;

        // Map the resource view array
        backendContext->pResourceViews = (BackendContext_VK::VkResourceView*)(pMem);
        memset(backendContext->pResourceViews, 0, resourceViewArraySize);
        pMem += resourceViewArraySize;

        // Map the staging ring buffer array
        backendContext->pStagingRingBuffer = (uint8_t*)pMem;
        memset(backendContext->pStagingRingBuffer, 0, stagingRingBufferArraySize);
        pMem += stagingRingBufferArraySize;

        // Map pipeline array
        backendContext->pPipelineLayouts = (BackendContext_VK::PipelineLayout*)pMem;
        memset(backendContext->pPipelineLayouts, 0, pipelineArraySize);
        pMem += pipelineArraySize;

        // Map resource array
        backendContext->pResources = (BackendContext_VK::Resource*)pMem;
        memset(backendContext->pResources, 0, resourceArraySize);
        pMem += resourceArraySize;

        // Clear out all resource mappings
        for (uint32_t i = 0; i < backendContext->maxEffectContexts * FFX_MAX_RESOURCE_COUNT; ++i)
        {
            backendContext->pResources[i].uavViewIndex = backendContext->pResources[i].srvViewIndex = -1;
        }

        // Map context array — align pMem to EffectContext's required alignment before assigning
        pMem                            = (uint8_t*)FFX_ALIGN_UP((uintptr_t)pMem, alignof(BackendContext_VK::EffectContext));
        backendContext->pEffectContexts = (BackendContext_VK::EffectContext*)pMem;
        memset(backendContext->pEffectContexts, 0, contextArraySize);
        pMem += contextArraySize;

        // Map extension array
        backendContext->extensionProperties = (VkExtensionProperties*)pMem;

        FFX_ASSERT(vkDeviceContext->vkDeviceProcAddr != NULL);

        if (vkDeviceContext->vkInstance != VK_NULL_HANDLE)
        {
            backendContext->instance = vkDeviceContext->vkInstance;
        }

        if (vkDeviceContext->vkDevice != VK_NULL_HANDLE)
        {
            backendContext->device = vkDeviceContext->vkDevice;
        }

        if (vkDeviceContext->vkPhysicalDevice != VK_NULL_HANDLE)
        {
            backendContext->physicalDevice = vkDeviceContext->vkPhysicalDevice;
        }

        // enumerate all the device extensions
        backendContext->numDeviceExtensions = 0;
        VulkanWrapper().vkEnumerateDeviceExtensionProperties(backendContext->physicalDevice, nullptr, &backendContext->numDeviceExtensions, nullptr);
        VulkanWrapper().vkEnumerateDeviceExtensionProperties(
            backendContext->physicalDevice, nullptr, &backendContext->numDeviceExtensions, backendContext->extensionProperties);

        // Initialize the device capabilities
        GetDeviceCapabilitiesVK(backendInterface, &backendInterface->deviceCapabilities);

        // create a global descriptor pool to hold all descriptors we'll need
        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
        VkDescriptorPoolSize       poolSizes[]              = {
            {VK_DESCRIPTOR_TYPE_SAMPLER,
             backendContext->maxEffectContexts * FFX_MAX_RESOURCE_COUNT * FFX_MAX_PASS_COUNT * FFX_MAX_QUEUED_FRAMES * MAX_PIPELINE_USAGE_PER_FRAME},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
             backendContext->maxEffectContexts * FFX_MAX_RESOURCE_COUNT * FFX_MAX_PASS_COUNT * FFX_MAX_QUEUED_FRAMES * MAX_PIPELINE_USAGE_PER_FRAME},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
             backendContext->maxEffectContexts * FFX_MAX_RESOURCE_COUNT * FFX_MAX_PASS_COUNT * FFX_MAX_QUEUED_FRAMES * MAX_PIPELINE_USAGE_PER_FRAME},
            {VK_DESCRIPTOR_TYPE_SAMPLER,
             backendContext->maxEffectContexts * FFX_MAX_RESOURCE_COUNT * FFX_MAX_PASS_COUNT * FFX_MAX_QUEUED_FRAMES * MAX_PIPELINE_USAGE_PER_FRAME},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
             backendContext->maxEffectContexts * FFX_MAX_RESOURCE_COUNT * FFX_MAX_PASS_COUNT * FFX_MAX_QUEUED_FRAMES * MAX_PIPELINE_USAGE_PER_FRAME},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             backendContext->maxEffectContexts * FFX_MAX_RESOURCE_COUNT * FFX_MAX_PASS_COUNT * FFX_MAX_QUEUED_FRAMES * MAX_PIPELINE_USAGE_PER_FRAME},
            {VK_DESCRIPTOR_TYPE_TENSOR_ARM,
             backendContext->maxEffectContexts * FFX_MAX_RESOURCE_COUNT * FFX_MAX_PASS_COUNT * FFX_MAX_QUEUED_FRAMES * MAX_PIPELINE_USAGE_PER_FRAME},
        };

        descriptorPoolCreateInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.pNext         = nullptr;
        descriptorPoolCreateInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        descriptorPoolCreateInfo.poolSizeCount = 7;
        descriptorPoolCreateInfo.pPoolSizes    = poolSizes;
        descriptorPoolCreateInfo.maxSets       = backendContext->maxEffectContexts * FFX_MAX_PASS_COUNT * MAX_PIPELINE_USAGE_PER_FRAME * FFX_MAX_QUEUED_FRAMES;

        if (VulkanWrapper().vkCreateDescriptorPool(backendContext->device, &descriptorPoolCreateInfo, nullptr, &backendContext->descriptorPool) != VK_SUCCESS)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }

        // set bindless resource view to base
        backendContext->bindlessBase = (backendContext->maxEffectContexts * FFX_MAX_QUEUED_FRAMES * FFX_MAX_RESOURCE_COUNT * 2);

        // allocate dynamic uniform buffer
        {
            // get alignment
            VkPhysicalDeviceProperties physicalDeviceProperties = {};
            VulkanWrapper().vkGetPhysicalDeviceProperties(backendContext->physicalDevice, &physicalDeviceProperties);
            backendContext->uniformBufferAlignment = physicalDeviceProperties.limits.minUniformBufferOffsetAlignment;

            backendContext->uniformBufferSize = FFX_ALIGN_UP(FFX_BUFFER_SIZE, backendContext->uniformBufferAlignment) * backendContext->maxEffectContexts *
                                                FFX_MAX_PASS_COUNT * FFX_MAX_QUEUED_FRAMES;

            backendContext->gpuVendorId = static_cast<FfxGpuVendorId>(physicalDeviceProperties.vendorID);

            VkBufferCreateInfo bufferInfo = {};
            bufferInfo.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size               = static_cast<VkDeviceSize>(backendContext->uniformBufferSize);
            bufferInfo.usage              = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bufferInfo.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;

            if (VulkanWrapper().vkCreateBuffer(backendContext->device, &bufferInfo, NULL, &backendContext->uniformBuffer) != VK_SUCCESS)
            {
                return FFX_ERROR_BACKEND_API_ERROR;
            }

            // allocate memory block for all uniform buffers
            VkMemoryRequirements memRequirements = {};
            VulkanWrapper().vkGetBufferMemoryRequirements(backendContext->device, backendContext->uniformBuffer, &memRequirements);

            // this is the real alignment
            backendContext->uniformBufferAlignment = memRequirements.alignment;

            VkMemoryPropertyFlags requiredMemoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = backendContext->uniformBufferSize;
            allocInfo.memoryTypeIndex =
                findMemoryTypeIndex(backendContext, memRequirements, requiredMemoryProperties, backendContext->uniformBufferMemoryProperties);

            if (allocInfo.memoryTypeIndex == UINT32_MAX)
            {
                requiredMemoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
                allocInfo.memoryTypeIndex =
                    findMemoryTypeIndex(backendContext, memRequirements, requiredMemoryProperties, backendContext->uniformBufferMemoryProperties);

                if (allocInfo.memoryTypeIndex == UINT32_MAX)
                {
                    return FFX_ERROR_BACKEND_API_ERROR;
                }
            }

            VkResult result = VulkanWrapper().vkAllocateMemory(backendContext->device, &allocInfo, nullptr, &backendContext->uniformBufferMemory);

            if (result != VK_SUCCESS)
            {
                switch (result)
                {
                case (VK_ERROR_OUT_OF_HOST_MEMORY):
                case (VK_ERROR_OUT_OF_DEVICE_MEMORY):
                    return FFX_ERROR_OUT_OF_MEMORY;
                default:
                    return FFX_ERROR_BACKEND_API_ERROR;
                }
            }

            // map the memory block
            if (VulkanWrapper().vkMapMemory(
                    backendContext->device, backendContext->uniformBufferMemory, 0, backendContext->uniformBufferSize, 0, &backendContext->uniformBufferMem) !=
                VK_SUCCESS)
            {
                return FFX_ERROR_BACKEND_API_ERROR;
            }

            if (VulkanWrapper().vkBindBufferMemory(backendContext->device, backendContext->uniformBuffer, backendContext->uniformBufferMemory, 0) != VK_SUCCESS)
            {
                return FFX_ERROR_BACKEND_API_ERROR;
            }
        }
    }

    // Increment the ref count
    ++backendContext->refCount;

    // Get an available context id
    for (uint32_t i = 0; i < backendContext->maxEffectContexts; ++i)
    {
        if (!backendContext->pEffectContexts[i].active)
        {
            *effectContextId = i;

            // Reset everything accordingly
            BackendContext_VK::EffectContext& effectContext = backendContext->pEffectContexts[i];
            effectContext.active                            = true;
            effectContext.effectId                          = effect;

            effectContext.nextStaticResource     = (i * FFX_MAX_RESOURCE_COUNT) + 1;
            effectContext.nextDynamicResource    = getDynamicResourcesStartIndex(i);
            effectContext.nextStaticResourceView = (i * FFX_MAX_QUEUED_FRAMES * FFX_MAX_RESOURCE_COUNT * 2);
            for (uint32_t frameIndex = 0; frameIndex < FFX_MAX_QUEUED_FRAMES; ++frameIndex)
            {
                effectContext.nextDynamicResourceView[frameIndex] = getDynamicResourceViewsStartIndex(i, frameIndex);
            }
            effectContext.nextPipelineLayout = (i * FFX_MAX_PASS_COUNT);
            effectContext.frameIndex         = 0;

            if (bindlessConfig)
            {
                effectContext.bindlessTextureSrvHeapStart = backendContext->bindlessBase;
                effectContext.bindlessTextureSrvHeapSize  = bindlessConfig->maxTextureSrvs;

                backendContext->bindlessBase += bindlessConfig->maxTextureSrvs;

                effectContext.bindlessBufferSrvHeapSize = bindlessConfig->maxBufferSrvs;

                effectContext.bindlessTextureUavHeapStart = backendContext->bindlessBase;
                effectContext.bindlessTextureUavHeapSize  = bindlessConfig->maxTextureUavs;

                backendContext->bindlessBase += bindlessConfig->maxTextureUavs;

                effectContext.bindlessBufferUavHeapSize = bindlessConfig->maxBufferUavs;

                // create a bindless descriptor pool local to the current effect
                VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
                VkDescriptorPoolSize       poolSizes[3];

                uint32_t poolSizeCount = 0;

                if (bindlessConfig->maxTextureSrvs > 0)
                {
                    VkDescriptorPoolSize& poolSize = poolSizes[poolSizeCount++];

                    poolSize.type            = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                    poolSize.descriptorCount = bindlessConfig->maxTextureSrvs;
                }

                if (bindlessConfig->maxTextureUavs > 0)
                {
                    VkDescriptorPoolSize& poolSize = poolSizes[poolSizeCount++];

                    poolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    poolSize.descriptorCount = bindlessConfig->maxTextureUavs;
                }

                if (bindlessConfig->maxBufferSrvs > 0 || bindlessConfig->maxBufferUavs > 0)
                {
                    VkDescriptorPoolSize& poolSize = poolSizes[poolSizeCount++];

                    poolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    poolSize.descriptorCount = bindlessConfig->maxBufferSrvs + bindlessConfig->maxBufferUavs;
                }

                descriptorPoolCreateInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                descriptorPoolCreateInfo.pNext         = nullptr;
                descriptorPoolCreateInfo.flags         = 0;
                descriptorPoolCreateInfo.poolSizeCount = poolSizeCount;
                descriptorPoolCreateInfo.pPoolSizes    = poolSizes;
                descriptorPoolCreateInfo.maxSets       = poolSizeCount;

                if (VulkanWrapper().vkCreateDescriptorPool(backendContext->device, &descriptorPoolCreateInfo, nullptr, &effectContext.bindlessDescriptorPool) !=
                    VK_SUCCESS)
                {
                    return FFX_ERROR_BACKEND_API_ERROR;
                }

                // create the descriptor layout for bindless texture srv buffers
                if (bindlessConfig->maxTextureSrvs > 0)
                {
                    VkDescriptorSetLayoutBinding binding = {
                        0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, bindlessConfig->maxTextureSrvs, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

                    VkDescriptorBindingFlags bindingFlags[] = {VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT};

                    VkDescriptorSetLayoutBindingFlagsCreateInfo setLayoutBindingFlags = {};

                    setLayoutBindingFlags.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
                    setLayoutBindingFlags.bindingCount  = 1;
                    setLayoutBindingFlags.pBindingFlags = bindingFlags;

                    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
                    layoutInfo.sType                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                    layoutInfo.pNext                           = &setLayoutBindingFlags;
                    layoutInfo.bindingCount                    = 1;
                    layoutInfo.pBindings                       = &binding;

                    if (VulkanWrapper().vkCreateDescriptorSetLayout(
                            backendContext->device, &layoutInfo, nullptr, &effectContext.bindlessTextureSrvDescriptorSetLayout) != VK_SUCCESS)
                    {
                        return FFX_ERROR_BACKEND_API_ERROR;
                    }

                    // allocate descriptor set
                    VkDescriptorSetAllocateInfo setAllocateInfo = {};
                    setAllocateInfo.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    setAllocateInfo.descriptorPool              = effectContext.bindlessDescriptorPool;
                    setAllocateInfo.descriptorSetCount          = 1;
                    setAllocateInfo.pSetLayouts                 = &effectContext.bindlessTextureSrvDescriptorSetLayout;

                    VulkanWrapper().vkAllocateDescriptorSets(backendContext->device, &setAllocateInfo, &effectContext.bindlessTextureSrvDescriptorSet);
                }

                // create the descriptor layout for bindless buffer srv buffers
                if (bindlessConfig->maxBufferSrvs > 0)
                {
                    VkDescriptorSetLayoutBinding binding = {
                        0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bindlessConfig->maxBufferSrvs, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

                    VkDescriptorBindingFlags bindingFlags[] = {VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT};

                    VkDescriptorSetLayoutBindingFlagsCreateInfo setLayoutBindingFlags = {};

                    setLayoutBindingFlags.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
                    setLayoutBindingFlags.bindingCount  = 1;
                    setLayoutBindingFlags.pBindingFlags = bindingFlags;

                    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
                    layoutInfo.sType                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                    layoutInfo.pNext                           = &setLayoutBindingFlags;
                    layoutInfo.bindingCount                    = 1;
                    layoutInfo.pBindings                       = &binding;

                    if (VulkanWrapper().vkCreateDescriptorSetLayout(
                            backendContext->device, &layoutInfo, nullptr, &effectContext.bindlessBufferSrvDescriptorSetLayout) != VK_SUCCESS)
                    {
                        return FFX_ERROR_BACKEND_API_ERROR;
                    }

                    // allocate descriptor set
                    VkDescriptorSetAllocateInfo setAllocateInfo = {};
                    setAllocateInfo.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    setAllocateInfo.descriptorPool              = effectContext.bindlessDescriptorPool;
                    setAllocateInfo.descriptorSetCount          = 1;
                    setAllocateInfo.pSetLayouts                 = &effectContext.bindlessBufferSrvDescriptorSetLayout;

                    VulkanWrapper().vkAllocateDescriptorSets(backendContext->device, &setAllocateInfo, &effectContext.bindlessBufferSrvDescriptorSet);
                }

                // create the descriptor layout for bindless texture uav buffers
                if (bindlessConfig->maxTextureUavs > 0)
                {
                    VkDescriptorSetLayoutBinding binding = {
                        0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, bindlessConfig->maxTextureUavs, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

                    VkDescriptorBindingFlags bindingFlags[] = {VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT};

                    VkDescriptorSetLayoutBindingFlagsCreateInfo setLayoutBindingFlags = {};

                    setLayoutBindingFlags.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
                    setLayoutBindingFlags.bindingCount  = 1;
                    setLayoutBindingFlags.pBindingFlags = bindingFlags;

                    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
                    layoutInfo.sType                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                    layoutInfo.pNext                           = &setLayoutBindingFlags;
                    layoutInfo.bindingCount                    = 1;
                    layoutInfo.pBindings                       = &binding;

                    if (VulkanWrapper().vkCreateDescriptorSetLayout(
                            backendContext->device, &layoutInfo, nullptr, &effectContext.bindlessTextureUavDescriptorSetLayout) != VK_SUCCESS)
                    {
                        return FFX_ERROR_BACKEND_API_ERROR;
                    }

                    // allocate descriptor set
                    VkDescriptorSetAllocateInfo setAllocateInfo = {};
                    setAllocateInfo.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    setAllocateInfo.descriptorPool              = effectContext.bindlessDescriptorPool;
                    setAllocateInfo.descriptorSetCount          = 1;
                    setAllocateInfo.pSetLayouts                 = &effectContext.bindlessTextureUavDescriptorSetLayout;

                    VulkanWrapper().vkAllocateDescriptorSets(backendContext->device, &setAllocateInfo, &effectContext.bindlessTextureUavDescriptorSet);
                }

                // create the descriptor layout for bindless buffer uav buffers
                if (bindlessConfig->maxBufferUavs > 0)
                {
                    VkDescriptorSetLayoutBinding binding = {
                        0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bindlessConfig->maxBufferUavs, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

                    VkDescriptorBindingFlags bindingFlags[] = {VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT};

                    VkDescriptorSetLayoutBindingFlagsCreateInfo setLayoutBindingFlags = {};

                    setLayoutBindingFlags.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
                    setLayoutBindingFlags.bindingCount  = 1;
                    setLayoutBindingFlags.pBindingFlags = bindingFlags;

                    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
                    layoutInfo.sType                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                    layoutInfo.pNext                           = &setLayoutBindingFlags;
                    layoutInfo.bindingCount                    = 1;
                    layoutInfo.pBindings                       = &binding;

                    if (VulkanWrapper().vkCreateDescriptorSetLayout(
                            backendContext->device, &layoutInfo, nullptr, &effectContext.bindlessBufferUavDescriptorSetLayout) != VK_SUCCESS)
                    {
                        return FFX_ERROR_BACKEND_API_ERROR;
                    }

                    // allocate descriptor set
                    VkDescriptorSetAllocateInfo setAllocateInfo = {};
                    setAllocateInfo.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    setAllocateInfo.descriptorPool              = effectContext.bindlessDescriptorPool;
                    setAllocateInfo.descriptorSetCount          = 1;
                    setAllocateInfo.pSetLayouts                 = &effectContext.bindlessBufferUavDescriptorSetLayout;

                    VulkanWrapper().vkAllocateDescriptorSets(backendContext->device, &setAllocateInfo, &effectContext.bindlessBufferUavDescriptorSet);
                }
            }
            else
            {
                effectContext.bindlessTextureSrvHeapStart = 0;
                effectContext.bindlessTextureSrvHeapSize  = 0;
                effectContext.bindlessBufferSrvHeapSize   = 0;
                effectContext.bindlessTextureUavHeapStart = 0;
                effectContext.bindlessTextureUavHeapSize  = 0;
                effectContext.bindlessBufferUavHeapSize   = 0;
            }

            break;
        }
    }

    return FFX_OK;
}

FfxErrorCode GetDeviceCapabilitiesVK(FfxInterface* backendInterface, FfxDeviceCapabilities* deviceCapabilities)
{
    FFX_ASSERT(backendInterface);
    FFX_ASSERT(deviceCapabilities);

    // Check if already initialized
    if (backendInterface->devCapInitialized)
    {
        if (deviceCapabilities != &backendInterface->deviceCapabilities)
        {
            memcpy(deviceCapabilities, &backendInterface->deviceCapabilities, sizeof(FfxDeviceCapabilities));
        }
        return FFX_OK;
    }

    BackendContext_VK* backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;
    FFX_ASSERT(backendContext);
    if (backendContext->physicalDevice == VK_NULL_HANDLE)
    {
        // In case the backend context is not created, early out
        return FFX_OK;
    }

    memset(deviceCapabilities, 0, sizeof(FfxDeviceCapabilities));
    // no shader model in vulkan so assume the minimum
    deviceCapabilities->maximumSupportedShaderModel = FFX_SHADER_MODEL_5_1;
    // We will get this from the query below, but we should be clear here we are expecting 16 lanes
    deviceCapabilities->waveLaneCountMin = 16;
    deviceCapabilities->waveLaneCountMax = 16;

    // check if extensions are enabled

    for (uint32_t i = 0; i < backendContext->numDeviceExtensions; i++)
    {
        const char* extensionName = backendContext->extensionProperties[i].extensionName;

        if (strcmp(extensionName, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) == 0)
        {
            // check if we the max subgroup size allows us to use wave64
            VkPhysicalDeviceSubgroupSizeControlProperties subgroupSizeControlProperties = {};
            subgroupSizeControlProperties.sType                                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;

            VkPhysicalDeviceProperties2 deviceProperties2 = {};
            deviceProperties2.sType                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            deviceProperties2.pNext                       = &subgroupSizeControlProperties;
            VulkanWrapper().vkGetPhysicalDeviceProperties2(backendContext->physicalDevice, &deviceProperties2);

            deviceCapabilities->waveLaneCountMin = subgroupSizeControlProperties.minSubgroupSize;
            deviceCapabilities->waveLaneCountMax = subgroupSizeControlProperties.maxSubgroupSize;
        }
        else if (strcmp(extensionName, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) == 0)
        {
            // check for fp16 support
            VkPhysicalDeviceShaderFloat16Int8Features shaderFloat16Int8Features = {};
            shaderFloat16Int8Features.sType                                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;

            VkPhysicalDeviceFeatures2 physicalDeviceFeatures2 = {};
            physicalDeviceFeatures2.sType                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            physicalDeviceFeatures2.pNext                     = &shaderFloat16Int8Features;

            VulkanWrapper().vkGetPhysicalDeviceFeatures2(backendContext->physicalDevice, &physicalDeviceFeatures2);

            deviceCapabilities->fp16Supported = (bool)shaderFloat16Int8Features.shaderFloat16;
        }
        else if (strcmp(extensionName, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0)
        {
            // check for ray tracing support
            VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures = {};
            accelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;

            VkPhysicalDeviceFeatures2 physicalDeviceFeatures2 = {};
            physicalDeviceFeatures2.sType                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            physicalDeviceFeatures2.pNext                     = &accelerationStructureFeatures;

            VulkanWrapper().vkGetPhysicalDeviceFeatures2(backendContext->physicalDevice, &physicalDeviceFeatures2);

            deviceCapabilities->raytracingSupported = (bool)accelerationStructureFeatures.accelerationStructure;
        }
        else if (strcmp(extensionName, VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME) == 0)
        {
            // check for coherent memory support
            VkPhysicalDeviceCoherentMemoryFeaturesAMD coherentMemoryFeatures = {};
            coherentMemoryFeatures.sType                                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD;

            VkPhysicalDeviceFeatures2 physicalDeviceFeatures2 = {};
            physicalDeviceFeatures2.sType                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            physicalDeviceFeatures2.pNext                     = &coherentMemoryFeatures;

            VulkanWrapper().vkGetPhysicalDeviceFeatures2(backendContext->physicalDevice, &physicalDeviceFeatures2);

            deviceCapabilities->deviceCoherentMemorySupported = (bool)coherentMemoryFeatures.deviceCoherentMemory;
        }
        else if (strcmp(extensionName, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME) == 0)
        {
            // no features structure so extension name is enough
            deviceCapabilities->dedicatedAllocationSupported = true;
        }
        else if (strcmp(extensionName, VK_AMD_BUFFER_MARKER_EXTENSION_NAME) == 0)
        {
            // no features structure so extension name is enough
            deviceCapabilities->bufferMarkerSupported = true;
        }
        else if (strcmp(extensionName, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0)
        {
            // check for extended synchronization support
            VkPhysicalDeviceSynchronization2FeaturesKHR synchronizationFeatures = {};
            synchronizationFeatures.sType                                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR;

            VkPhysicalDeviceFeatures2 physicalDeviceFeatures2 = {};
            physicalDeviceFeatures2.sType                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            physicalDeviceFeatures2.pNext                     = &synchronizationFeatures;

            VulkanWrapper().vkGetPhysicalDeviceFeatures2(backendContext->physicalDevice, &physicalDeviceFeatures2);

            deviceCapabilities->extendedSynchronizationSupported = (bool)synchronizationFeatures.synchronization2;
        }
        else if (strcmp(extensionName, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME) == 0)
        {
            // check for coherent memory support
            VkPhysicalDeviceDescriptorIndexingFeatures descriptorIndexingFeatures = {};
            descriptorIndexingFeatures.sType                                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;

            VkPhysicalDeviceFeatures2 physicalDeviceFeatures2 = {};
            physicalDeviceFeatures2.sType                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            physicalDeviceFeatures2.pNext                     = &descriptorIndexingFeatures;

            VulkanWrapper().vkGetPhysicalDeviceFeatures2(backendContext->physicalDevice, &physicalDeviceFeatures2);

            deviceCapabilities->shaderStorageBufferArrayNonUniformIndexing = (bool)descriptorIndexingFeatures.shaderStorageBufferArrayNonUniformIndexing;
        }
        else if (strcmp(extensionName, VK_ARM_TENSORS_EXTENSION_NAME) == 0)
        {
            // no features structure so extension name is enough
            deviceCapabilities->tensorSupported = true;

            VkPhysicalDeviceTensorPropertiesARM tensorProps = {};
            tensorProps.sType                               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_PROPERTIES_ARM;
            tensorProps.pNext                               = nullptr;

            VkPhysicalDeviceProperties2 deviceProperties2 = {};
            deviceProperties2.sType                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            deviceProperties2.pNext                       = &tensorProps;

            VulkanWrapper().vkGetPhysicalDeviceProperties2(backendContext->physicalDevice, &deviceProperties2);

            deviceCapabilities->computeSupportTensor  = static_cast<bool>(tensorProps.shaderTensorSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT);
            deviceCapabilities->fragmentSupportTensor = static_cast<bool>(tensorProps.shaderTensorSupportedStages & VK_SHADER_STAGE_FRAGMENT_BIT);
        }
        else if (strcmp(extensionName, VK_ARM_DATA_GRAPH_EXTENSION_NAME) == 0)
        {
            // no features structure so extension name is enough
            deviceCapabilities->dataGraphSupported = true;
        }
        else if (strcmp(extensionName, VK_ARM_DATA_GRAPH_OPTICAL_FLOW_EXTENSION_NAME) == 0)
        {
            // Check whether the data graph optical flow feature is supported.
            VkPhysicalDeviceDataGraphOpticalFlowFeaturesARM dataGraphOpticalFlowFeatures = {};
            dataGraphOpticalFlowFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_OPTICAL_FLOW_FEATURES_ARM;

            VkPhysicalDeviceFeatures2 physicalDeviceFeatures2 = {};
            physicalDeviceFeatures2.sType                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            physicalDeviceFeatures2.pNext                     = &dataGraphOpticalFlowFeatures;

            VulkanWrapper().vkGetPhysicalDeviceFeatures2(backendContext->physicalDevice, &physicalDeviceFeatures2);
            deviceCapabilities->dataGraphOFSupported = static_cast<bool>(dataGraphOpticalFlowFeatures.dataGraphOpticalFlow);

            // Check the detailed properties of the data graph optical flow.
            if (deviceCapabilities->dataGraphOFSupported)
            {
                VkQueueFamilyDataGraphOpticalFlowPropertiesARM dataGraphOFProperties = {};
                dataGraphOFProperties.sType                                          = VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_OPTICAL_FLOW_PROPERTIES_ARM;

                const auto queryQueueFamilyDataGraphOpProperties = VulkanWrapper().vkGetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM;
                if (!queryQueueFamilyDataGraphOpProperties)
                {
                    // Extension is exposed but the query entrypoint is not available.
                    deviceCapabilities->dataGraphOFSupported = false;
                    continue;
                }

                uint32_t queueFamilyCount = 0;
                VulkanWrapper().vkGetPhysicalDeviceQueueFamilyProperties(backendContext->physicalDevice, &queueFamilyCount, nullptr);

                for (uint32_t queueFamilyIndex = 0; queueFamilyIndex < queueFamilyCount; ++queueFamilyIndex)
                {
                    VkQueueFamilyDataGraphPropertiesARM queueFamilyDataGraphProperties = {};
                    queueFamilyDataGraphProperties.sType                               = VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_PROPERTIES_ARM;
                    queueFamilyDataGraphProperties.engine.type                         = VK_PHYSICAL_DEVICE_DATA_GRAPH_PROCESSING_ENGINE_TYPE_DEFAULT_ARM;
                    queueFamilyDataGraphProperties.engine.isForeign                    = VK_FALSE;
                    queueFamilyDataGraphProperties.operation.operationType             = VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_OPTICAL_FLOW_ARM;
                    std::strncpy(
                        queueFamilyDataGraphProperties.operation.name, "OpticalFlow", VK_MAX_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_SET_NAME_SIZE_ARM - 1);
                    queueFamilyDataGraphProperties.operation.version = 1;

                    const VkResult queryRes = queryQueueFamilyDataGraphOpProperties(backendContext->physicalDevice,
                                                                                    queueFamilyIndex,
                                                                                    &queueFamilyDataGraphProperties,
                                                                                    reinterpret_cast<VkBaseOutStructure*>(&dataGraphOFProperties));

                    if (queryRes == VK_SUCCESS)
                    {
                        deviceCapabilities->costSupported            = static_cast<bool>(dataGraphOFProperties.costSupported);
                        deviceCapabilities->hintSupported            = static_cast<bool>(dataGraphOFProperties.hintSupported);
                        deviceCapabilities->supportedHintGridSizes   = static_cast<uint32_t>(dataGraphOFProperties.supportedHintGridSizes);
                        deviceCapabilities->supportedOutputGridSizes = static_cast<uint32_t>(dataGraphOFProperties.supportedOutputGridSizes);
                        deviceCapabilities->dataGraphOFMinWidth      = dataGraphOFProperties.minWidth;
                        deviceCapabilities->dataGraphOFMinHeight     = dataGraphOFProperties.minHeight;
                        deviceCapabilities->dataGraphOFMaxWidth      = dataGraphOFProperties.maxWidth;
                        deviceCapabilities->dataGraphOFMaxHeight     = dataGraphOFProperties.maxHeight;
                        break;
                    }
                }
            }
        }
    }

    backendInterface->devCapInitialized = true;
    if (deviceCapabilities != &backendInterface->deviceCapabilities)
    {
        // cache the result
        memcpy(&backendInterface->deviceCapabilities, deviceCapabilities, sizeof(FfxDeviceCapabilities));
        return FFX_OK;
    }

    return FFX_OK;
}

FfxErrorCode DestroyBackendContextVK(FfxInterface* backendInterface, FfxUInt32 effectContextId)
{
    FFX_ASSERT(NULL != backendInterface);
    BackendContext_VK* backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;
    FFX_ASSERT(backendContext->refCount > 0);

    // Delete any resources allocated by this context
    BackendContext_VK::EffectContext& effectContext = backendContext->pEffectContexts[effectContextId];
    for (uint32_t currentStaticResourceIndex = effectContextId * FFX_MAX_RESOURCE_COUNT; currentStaticResourceIndex < effectContext.nextStaticResource;
         ++currentStaticResourceIndex)
    {
        if (backendContext->pResources[currentStaticResourceIndex].imageResource != VK_NULL_HANDLE)
        {
            FFX_ASSERT_MESSAGE(false,
                               "FFXInterface: Vulkan: SDK Resource was not destroyed prior to destroying the backend context. There is a resource leak.");
            FfxResourceInternal internalResource = {static_cast<int32_t>(currentStaticResourceIndex)};
            DestroyResourceVK(backendInterface, internalResource, effectContextId);
        }
    }

    for (uint32_t frameIndex = 0; frameIndex < FFX_MAX_QUEUED_FRAMES; ++frameIndex)
        destroyDynamicViews(backendContext, effectContextId, frameIndex);

    // clean up descriptor set layouts
    if (effectContext.bindlessTextureSrvDescriptorSetLayout)
    {
        VulkanWrapper().vkFreeDescriptorSets(backendContext->device, effectContext.bindlessDescriptorPool, 1, &effectContext.bindlessTextureSrvDescriptorSet);
        VulkanWrapper().vkDestroyDescriptorSetLayout(backendContext->device, effectContext.bindlessTextureSrvDescriptorSetLayout, VK_NULL_HANDLE);
        effectContext.bindlessTextureSrvDescriptorSetLayout = VK_NULL_HANDLE;
    }

    if (effectContext.bindlessBufferSrvDescriptorSetLayout)
    {
        VulkanWrapper().vkFreeDescriptorSets(backendContext->device, effectContext.bindlessDescriptorPool, 1, &effectContext.bindlessBufferSrvDescriptorSet);
        VulkanWrapper().vkDestroyDescriptorSetLayout(backendContext->device, effectContext.bindlessBufferSrvDescriptorSetLayout, VK_NULL_HANDLE);
        effectContext.bindlessBufferSrvDescriptorSetLayout = VK_NULL_HANDLE;
    }

    if (effectContext.bindlessTextureUavDescriptorSetLayout)
    {
        VulkanWrapper().vkFreeDescriptorSets(backendContext->device, effectContext.bindlessDescriptorPool, 1, &effectContext.bindlessTextureUavDescriptorSet);
        VulkanWrapper().vkDestroyDescriptorSetLayout(backendContext->device, effectContext.bindlessTextureUavDescriptorSetLayout, VK_NULL_HANDLE);
        effectContext.bindlessTextureUavDescriptorSetLayout = VK_NULL_HANDLE;
    }

    if (effectContext.bindlessBufferUavDescriptorSetLayout)
    {
        VulkanWrapper().vkFreeDescriptorSets(backendContext->device, effectContext.bindlessDescriptorPool, 1, &effectContext.bindlessBufferUavDescriptorSet);
        VulkanWrapper().vkDestroyDescriptorSetLayout(backendContext->device, effectContext.bindlessBufferUavDescriptorSetLayout, VK_NULL_HANDLE);
        effectContext.bindlessBufferUavDescriptorSetLayout = VK_NULL_HANDLE;
    }

    if (effectContext.bindlessDescriptorPool)
    {
        VulkanWrapper().vkDestroyDescriptorPool(backendContext->device, effectContext.bindlessDescriptorPool, VK_NULL_HANDLE);
        effectContext.bindlessTextureSrvDescriptorSetLayout = VK_NULL_HANDLE;
    }

    // Free up for use by another context
    effectContext.nextStaticResource = 0;
    effectContext.active             = false;

    // Decrement ref count
    --backendContext->refCount;

    if (!backendContext->refCount)
    {
        // clean up descriptor pool
        VulkanWrapper().vkDestroyDescriptorPool(backendContext->device, backendContext->descriptorPool, VK_NULL_HANDLE);
        backendContext->descriptorPool = VK_NULL_HANDLE;

        // clean up dynamic uniform buffer & memory
        VulkanWrapper().vkUnmapMemory(backendContext->device, backendContext->uniformBufferMemory);
        VulkanWrapper().vkFreeMemory(backendContext->device, backendContext->uniformBufferMemory, VK_NULL_HANDLE);
        VulkanWrapper().vkDestroyBuffer(backendContext->device, backendContext->uniformBuffer, VK_NULL_HANDLE);

        backendContext->device         = VK_NULL_HANDLE;
        backendContext->physicalDevice = VK_NULL_HANDLE;

        resetBackendContext(backendContext);
    }

    return FFX_OK;
}

static bool validateTensorSurfaceFormat(FfxInterface* backendInterface, FfxSurfaceFormat format)
{
    BackendContext_VK*         backendContext           = (BackendContext_VK*)backendInterface->scratchBuffer;
    VkPhysicalDeviceProperties physicalDeviceProperties = {};
    VulkanWrapper().vkGetPhysicalDeviceProperties(backendContext->physicalDevice, &physicalDeviceProperties);
    bool res = true;

    //Validate tensor format
    //Current Mali GPU only support part of FfxSurfaceFormat.
    //Will add more vendor name checking if subsequent mali GPU support different format.
    if (physicalDeviceProperties.vendorID == 0x13B5)  //Vendor: Arm (Mali)
    {
        switch (format)
        {
        case FFX_SURFACE_FORMAT_R8_UINT:
        case FFX_SURFACE_FORMAT_R8_SINT:
        case FFX_SURFACE_FORMAT_R16_UINT:
            res = true;
            break;
        default:
            res = false;
            break;
        }
    }

    return res;
}

static VkTensorTilingARM GetTensorTiling(const bool bufferAliased)
{
    // Buffer aliasing requires LINEAR tiling so the VkBuffer can address the tensor data in a
    // predictable row-major NHWC layout.  OPTIMAL tiling uses an opaque brick/block layout
    // that is not addressable as a flat buffer.
    return bufferAliased ? VK_TENSOR_TILING_LINEAR_ARM : VK_TENSOR_TILING_OPTIMAL_ARM;
}

FfxErrorCode CreateTensorResourceVK(FfxInterface*                       backendInterface,
                                    const FfxCreateResourceDescription* createResourceDescription,
                                    FfxUInt32                           effectContextId,
                                    FfxResourceInternal*                outResource)
{
    BackendContext_VK*                backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;
    BackendContext_VK::EffectContext& effectContext  = backendContext->pEffectContexts[effectContextId];
    VkDevice                          device         = backendContext->device;

    FFX_ASSERT(VK_NULL_HANDLE != device);
    FFX_ASSERT(createResourceDescription);
    FFX_ASSERT(FFX_RESOURCE_TYPE_TENSOR == createResourceDescription->resourceDescription.type);

    // Setup the resource description
    FfxResourceDescription resourceDesc = createResourceDescription->resourceDescription;
    FFX_ASSERT(validateTensorSurfaceFormat(backendInterface, resourceDesc.format));

    outResource->alignedWidth = createResourceDescription->resourceDescription.width;
    FFX_ASSERT(effectContext.nextStaticResource + 1 < effectContext.nextDynamicResource);
    outResource->internalIndex                   = effectContext.nextStaticResource++;
    BackendContext_VK::Resource* backendResource = &backendContext->pResources[outResource->internalIndex];
    backendResource->undefined           = true;   // A flag to make sure the first barrier for this image resource always uses an src layout of undefined
    backendResource->dynamic             = false;  // Not a dynamic resource (need to track them separately for image views)
    backendResource->resourceDescription = resourceDesc;
    backendResource->allocationSize      = 0;

    const FfxResourceStates resourceState = ((createResourceDescription->initData.type != FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED) &&
                                             (createResourceDescription->heapType != FFX_HEAP_TYPE_UPLOAD))
                                                ? FFX_RESOURCE_STATE_COPY_DEST
                                                : createResourceDescription->initialState;
    backendResource->initialState = resourceState;
    backendResource->currentState = resourceState;

#ifdef _DEBUG
    if (createResourceDescription->name)
    {
        strncpy(backendResource->resourceName, createResourceDescription->name, sizeof(backendResource->resourceName) - 1);
        backendResource->resourceName[sizeof(backendResource->resourceName) - 1] = '\0';
    }
#endif

    std::vector<int64_t> dimensions;

    FFX_ASSERT(createResourceDescription->resourceDescription.shapeSize <= 4);
    for (uint32_t i = 0; i < createResourceDescription->resourceDescription.shapeSize; i++)
    {
        switch (i)
        {
        case 0:
            dimensions.push_back(createResourceDescription->resourceDescription.batchSize);
            break;

        case 1:
            dimensions.push_back(createResourceDescription->resourceDescription.height);
            break;

        case 2:
            dimensions.push_back(createResourceDescription->resourceDescription.width);
            break;

        case 3:
            dimensions.push_back(createResourceDescription->resourceDescription.channel);
            break;

        default:
            break;
        }
    }

    const bool imageAliased  = (createResourceDescription->resourceDescription.flags & FFX_RESOURCE_FLAGS_IMAGE_ALIASED) == FFX_RESOURCE_FLAGS_IMAGE_ALIASED;
    const bool bufferAliased = (createResourceDescription->resourceDescription.flags & FFX_RESOURCE_FLAGS_BUFFER_ALIASED) == FFX_RESOURCE_FLAGS_BUFFER_ALIASED;
    // IMAGE_ALIASING_BIT is incompatible with LINEAR (VUID-VkTensorDescriptionARM), so imageAliased and bufferAliased are mutually exclusive
    FFX_ASSERT_MESSAGE(!imageAliased || !bufferAliased, "Cant alias tensor both as image and buffer!");

    VkTensorUsageFlagsARM tensorUsage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM;
    if (imageAliased)
    {
        // Image aliasing with a linearly tiled tensor doesn't need this flag.
        // This is only necessary for VK_TENSOR_TILING_OPTIMAL_ARM
        tensorUsage |= VK_TENSOR_USAGE_IMAGE_ALIASING_BIT_ARM;
    }

    const VkTensorDescriptionARM tensorDescription = {VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
                                                      nullptr,
                                                      GetTensorTiling(bufferAliased),
                                                      ffxGetVkFormatFromSurfaceFormat(createResourceDescription->resourceDescription.format),
                                                      createResourceDescription->resourceDescription.shapeSize,
                                                      dimensions.data(),
                                                      nullptr,  // pStrides, the tensor will be packed
                                                      tensorUsage};

    const VkTensorCreateInfoARM tensorCreateInfo = {
        VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM,
        nullptr,
        0,  // flags
        &tensorDescription,
        VK_SHARING_MODE_EXCLUSIVE,
        0,        // queueFamilyIndexCount
        nullptr,  // pQueueFamilyIndices
    };

    if (VulkanWrapper().vkCreateTensorARM(device, &tensorCreateInfo, nullptr, &backendResource->tensorResource))
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // Aliased image initialization
    uint32_t channels = createResourceDescription->resourceDescription.channel;
    VkFormat aliasedFormat =
        imageAliased ? ffxGetVkFormatForTensorAliasedImage(createResourceDescription->resourceDescription.format, channels) : VK_FORMAT_UNDEFINED;
    if (imageAliased)
    {
        VkExtent3D              imageOutExtent     = {static_cast<uint32_t>(dimensions[2]), static_cast<uint32_t>(dimensions[1]), 1};
        const VkImageCreateInfo imageOutCreateInfo = {
            VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            nullptr,
            0,  // flags
            VK_IMAGE_TYPE_2D,
            aliasedFormat,
            imageOutExtent,
            1,                        // mipLevels
            1,                        // arrayLayers
            VK_SAMPLE_COUNT_1_BIT,    // samples
            VK_IMAGE_TILING_OPTIMAL,  // See VUID-VkImageCreateInfo-tiling-09711
                                      // If tiling is VK_IMAGE_TILING_LINEAR then VK_IMAGE_USAGE_TENSOR_ALIASING_BIT_ARM must not be set in usage
            (VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TENSOR_ALIASING_BIT_ARM |  // Aliasing bit required when using optimal tiling.
             VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT),
            VK_SHARING_MODE_EXCLUSIVE,
            0,        // queueFamilyIndexCount
            nullptr,  // pQueueFamilyIndices
            VK_IMAGE_LAYOUT_UNDEFINED};

        if (VulkanWrapper().vkCreateImage(device, &imageOutCreateInfo, nullptr, &backendResource->aliasedTensorImageResource))
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }
    }

#ifdef _DEBUG
    // setVKObjectName(VulkanWrapper(), backendContext->device, VK_OBJECT_TYPE_TENSOR_ARM, (uint64_t)backendResource->tensorResource, backendResource->resourceName);
#endif

    const VkTensorMemoryRequirementsInfoARM tensorMemInfo = {VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_ARM, nullptr, backendResource->tensorResource};

    VkMemoryRequirements2 tensorMemreqs;
    tensorMemreqs.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    tensorMemreqs.pNext = nullptr;
    VulkanWrapper().vkGetTensorMemoryRequirementsARM(device, &tensorMemInfo, &tensorMemreqs);

    // Capture raw tensor data size before it may be enlarged to fit a co-aliased image.
    const VkDeviceSize tensorDataSize = tensorMemreqs.memoryRequirements.size;

    VkDeviceSize memoryOffset = 0u;

    // Aliased image initialization
    if (imageAliased)
    {
        const VkImageMemoryRequirementsInfo2 imageMemInfo = {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, nullptr, backendResource->aliasedTensorImageResource};
        VkMemoryRequirements2 imageMemReqs = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, nullptr, {}};
        VulkanWrapper().vkGetImageMemoryRequirements2(device, &imageMemInfo, &imageMemReqs);

        constexpr VkImageSubresource imageOutSubresource = {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,  // mipLevel
            0   // arrayLayer
        };
        VkSubresourceLayout imageOutSubresourceLayout{};
        VulkanWrapper().vkGetImageSubresourceLayout(device, backendResource->aliasedTensorImageResource, &imageOutSubresource, &imageOutSubresourceLayout);

        // Combine the memory requirements of the tensor and the aliased image into the tensor mem reqs
        memoryOffset                          = imageOutSubresourceLayout.offset;
        tensorMemreqs.memoryRequirements.size = FFX_MAXIMUM(imageMemReqs.memoryRequirements.size, tensorMemreqs.memoryRequirements.size + memoryOffset);
        tensorMemreqs.memoryRequirements.memoryTypeBits = imageMemReqs.memoryRequirements.memoryTypeBits & tensorMemreqs.memoryRequirements.memoryTypeBits;
    }

    VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if (allocateDeviceMemory(backendContext, tensorMemreqs.memoryRequirements, memoryProperties, backendResource))
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    const VkBindTensorMemoryInfoARM tensorBindInfo = {
        VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM, nullptr, backendResource->tensorResource, backendResource->deviceMemory, memoryOffset};

    if (VulkanWrapper().vkBindTensorMemoryARM(device, 1, &tensorBindInfo))
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    if (imageAliased)
    {
        const VkBindImageMemoryInfo bindImageMemoryInfo = {
            VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, nullptr, backendResource->aliasedTensorImageResource, backendResource->deviceMemory, 0};

        if (VulkanWrapper().vkBindImageMemory2(device, 1, &bindImageMemoryInfo))
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }
    }

    const VkTensorViewCreateInfoARM tensorViewCreateInfo = {VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_ARM,
                                                            nullptr,
                                                            0,  // flags
                                                            backendResource->tensorResource,
                                                            ffxGetVkFormatFromSurfaceFormat(createResourceDescription->resourceDescription.format)};

    FFX_ASSERT_MESSAGE(effectContext.nextStaticResourceView + 1 < effectContext.nextDynamicResourceView[0],
                       "ffxInterface: Vulkan: We've run out of resource views. Please increase the size.");
    backendResource->tensorViewIndex = effectContext.nextStaticResourceView++;

    if (VulkanWrapper().vkCreateTensorViewARM(
            device, &tensorViewCreateInfo, nullptr, &backendContext->pResourceViews[backendResource->tensorViewIndex].tensorView))
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    if (imageAliased)
    {
        // copied straight from the create resource vk function
        FFX_ASSERT_MESSAGE(effectContext.nextStaticResourceView + 1 < effectContext.nextDynamicResourceView[0],
                           "FFXInterface: Vulkan: We've run out of resource views. Please increase the size.");
        backendResource->srvViewIndex = effectContext.nextStaticResourceView++;

        VkImageViewCreateInfo imageViewCreateInfo = {};
        imageViewCreateInfo.sType                 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCreateInfo.pNext                 = nullptr;

        // The tensor aliased image is always type 2D. The tensor's channel count is encoded in the image FORMAT (e.g. RGBA32F for 4-channel)
        imageViewCreateInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCreateInfo.image                           = backendResource->aliasedTensorImageResource;
        imageViewCreateInfo.format                          = aliasedFormat;
        imageViewCreateInfo.components.r                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.g                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.b                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.a                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.subresourceRange.aspectMask     = getImageAspect(backendResource->resourceDescription.usage);
        imageViewCreateInfo.subresourceRange.baseMipLevel   = 0;
        imageViewCreateInfo.subresourceRange.levelCount     = backendResource->resourceDescription.mipCount;
        imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        imageViewCreateInfo.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;

        VkImageViewUsageCreateInfo imageViewUsageCreateInfo = {};
        setupImageViewForSRV(imageViewCreateInfo, imageViewUsageCreateInfo, backendResource->resourceDescription);

        // create an image view containing all mip levels for use as an srv
        if (VulkanWrapper().vkCreateImageView(
                backendContext->device, &imageViewCreateInfo, NULL, &backendContext->pResourceViews[backendResource->srvViewIndex].imageView) != VK_SUCCESS)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }
#ifdef _DEBUG
        setVKObjectName(backendContext->device,
                        VK_OBJECT_TYPE_IMAGE_VIEW,
                        (uint64_t)backendContext->pResourceViews[backendResource->srvViewIndex].imageView,
                        backendResource->resourceName);
#endif

        // create image views of individual mip levels for use as a uav
        if (FFX_CONTAINS_FLAG(backendResource->resourceDescription.usage, FFX_RESOURCE_USAGE_UAV))
        {
            const int32_t uavResourceViewCount = backendResource->resourceDescription.mipCount;
            FFX_ASSERT(effectContext.nextStaticResourceView + uavResourceViewCount < effectContext.nextDynamicResourceView[0]);

            backendResource->uavViewIndex = effectContext.nextStaticResourceView;
            backendResource->uavViewCount = uavResourceViewCount;
            imageViewCreateInfo.format    = aliasedFormat;

            for (uint32_t mip = 0; mip < backendResource->resourceDescription.mipCount; ++mip)
            {
                imageViewCreateInfo.subresourceRange.levelCount   = 1;
                imageViewCreateInfo.subresourceRange.baseMipLevel = mip;

                if (VulkanWrapper().vkCreateImageView(
                        backendContext->device, &imageViewCreateInfo, NULL, &backendContext->pResourceViews[backendResource->uavViewIndex + mip].imageView) !=
                    VK_SUCCESS)
                {
                    return FFX_ERROR_BACKEND_API_ERROR;
                }
#ifdef _DEBUG
                setVKObjectName(backendContext->device,
                                VK_OBJECT_TYPE_IMAGE_VIEW,
                                (uint64_t)backendContext->pResourceViews[backendResource->uavViewIndex + mip].imageView,
                                backendResource->resourceName);
#endif
            }

            effectContext.nextStaticResourceView += uavResourceViewCount;
        }
    }

    if (bufferAliased)
    {
        // Create a VkBuffer that shares the tensor's device memory.
        // The tensor is bound at memoryOffset; the buffer starts there and spans exactly the raw tensor data.
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size               = tensorDataSize;
        bufferInfo.usage              = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferInfo.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;

        if (VulkanWrapper().vkCreateBuffer(device, &bufferInfo, nullptr, &backendResource->aliasedTensorBufferResource) != VK_SUCCESS)
            return FFX_ERROR_BACKEND_API_ERROR;

        if (VulkanWrapper().vkBindBufferMemory(device, backendResource->aliasedTensorBufferResource, backendResource->deviceMemory, memoryOffset) != VK_SUCCESS)
            return FFX_ERROR_BACKEND_API_ERROR;

        // Expose tensor width as alignedWidth so shaders can stride correctly.
        const uint32_t shapeSize  = createResourceDescription->resourceDescription.shapeSize;
        outResource->alignedWidth = shapeSize >= 3 ? static_cast<uint32_t>(dimensions[2]) : static_cast<uint32_t>(dimensions.back());
    }

#ifdef _DEBUG
    // setVKObjectName(VulkanWrapper(), backendContext->device, VK_OBJECT_TYPE_TENSOR_VIEW_ARM, (uint64_t)backendContext->pResourceViews[backendResource->tensorViewIndex].tensorView, backendResource->resourceName);
#endif

    return FFX_OK;
}

// create a internal resource that will stay alive until effect gets shut down
FfxErrorCode CreateResourceVK(FfxInterface*                       backendInterface,
                              const FfxCreateResourceDescription* createResourceDescription,
                              FfxUInt32                           effectContextId,
                              FfxResourceInternal*                outResource)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != createResourceDescription);
    FFX_ASSERT(NULL != outResource);
    FFX_ASSERT_MESSAGE(createResourceDescription->initData.type != FFX_RESOURCE_INIT_DATA_TYPE_INVALID,
                       "InitData type cannot be FFX_RESOURCE_INIT_DATA_TYPE_INVALID. Please explicitly specify the resource initialization type.");

    if (createResourceDescription->resourceDescription.type == FFX_RESOURCE_TYPE_TENSOR)
        return CreateTensorResourceVK(backendInterface, createResourceDescription, effectContextId, outResource);

    BackendContext_VK*                backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;
    BackendContext_VK::EffectContext& effectContext  = backendContext->pEffectContexts[effectContextId];
    VkDevice                          vkDevice       = backendContext->device;

    FFX_ASSERT(VK_NULL_HANDLE != vkDevice);

    VkMemoryPropertyFlags requiredMemoryProperties;

    switch (createResourceDescription->heapType)
    {
    case FFX_HEAP_TYPE_DEFAULT:
        requiredMemoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    case FFX_HEAP_TYPE_UPLOAD:
    case FFX_HEAP_TYPE_READBACK:
        requiredMemoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        break;
    default:
        requiredMemoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    }

    // Setup the resource description
    FfxResourceDescription resourceDesc = createResourceDescription->resourceDescription;

    if (resourceDesc.mipCount == 0)
    {
        resourceDesc.mipCount = (uint32_t)(
            1 + floor(log2(FFX_MAXIMUM(FFX_MAXIMUM(createResourceDescription->resourceDescription.width, createResourceDescription->resourceDescription.height),
                                       createResourceDescription->resourceDescription.depth))));
    }

    FFX_ASSERT(effectContext.nextStaticResource + 1 < effectContext.nextDynamicResource);
    outResource->internalIndex                   = effectContext.nextStaticResource++;
    BackendContext_VK::Resource* backendResource = &backendContext->pResources[outResource->internalIndex];
    backendResource->undefined           = true;   // A flag to make sure the first barrier for this image resource always uses an src layout of undefined
    backendResource->dynamic             = false;  // Not a dynamic resource (need to track them separately for image views)
    backendResource->resourceDescription = resourceDesc;
    backendResource->allocationSize      = 0;

    const auto& initData = createResourceDescription->initData;

    const FfxResourceStates resourceState =
        ((initData.type != FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED) && (createResourceDescription->heapType != FFX_HEAP_TYPE_UPLOAD))
            ? FFX_RESOURCE_STATE_COPY_DEST
            : createResourceDescription->initialState;
    backendResource->initialState = resourceState;
    backendResource->currentState = resourceState;

#ifdef _DEBUG
    if (createResourceDescription->name)
    {
        strncpy(backendResource->resourceName, createResourceDescription->name, sizeof(backendResource->resourceName) - 1);
        backendResource->resourceName[sizeof(backendResource->resourceName) - 1] = '\0';
    }
#endif

    VkMemoryRequirements memRequirements = {};

    const bool createAliasedBuffer = (resourceDesc.flags & FFX_RESOURCE_FLAGS_BUFFER_ALIASED) == FFX_RESOURCE_FLAGS_BUFFER_ALIASED;

    switch (createResourceDescription->resourceDescription.type)
    {
    case FFX_RESOURCE_TYPE_BUFFER:
    {
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size               = createResourceDescription->resourceDescription.width;
        bufferInfo.usage              = ffxGetVKBufferUsageFlagsFromResourceUsage(resourceDesc.usage);
        bufferInfo.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;

        if (initData.type != FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED)
            bufferInfo.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        if (resourceState == FFX_RESOURCE_STATE_COPY_SRC)
            bufferInfo.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        if (resourceState == FFX_RESOURCE_STATE_COPY_DEST)
            bufferInfo.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        if (VulkanWrapper().vkCreateBuffer(backendContext->device, &bufferInfo, NULL, &backendResource->bufferResource) != VK_SUCCESS)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }

#ifdef _DEBUG
        setVKObjectName(backendContext->device, VK_OBJECT_TYPE_BUFFER, (uint64_t)backendResource->bufferResource, backendResource->resourceName);
#endif

        VulkanWrapper().vkGetBufferMemoryRequirements(backendContext->device, backendResource->bufferResource, &memRequirements);

        // allocate the memory
        FfxErrorCode errorCode = allocateDeviceMemory(backendContext, memRequirements, requiredMemoryProperties, backendResource);
        if (FFX_OK != errorCode)
            return errorCode;

        if (VulkanWrapper().vkBindBufferMemory(backendContext->device, backendResource->bufferResource, backendResource->deviceMemory, 0) != VK_SUCCESS)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }

        // if this is an upload buffer (currently only support upload buffers), copy the data and return
        if (createResourceDescription->heapType == FFX_HEAP_TYPE_UPLOAD && initData.size > 0)
        {
            // only allow copies directly into mapped memory for buffer resources since all texture resources are in optimal tiling
            void* data = NULL;

            if (VulkanWrapper().vkMapMemory(backendContext->device, backendResource->deviceMemory, 0, initData.size, 0, &data) != VK_SUCCESS)
            {
                return FFX_ERROR_BACKEND_API_ERROR;
            }

            if (initData.type == FFX_RESOURCE_INIT_DATA_TYPE_BUFFER)
                memcpy(data, initData.buffer, initData.size);
            else if (initData.type == FFX_RESOURCE_INIT_DATA_TYPE_VALUE)
                memset(data, initData.value, initData.size);

            // flush mapped range if memory type is not coherant
            if ((backendResource->memoryProperties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)
            {
                VkMappedMemoryRange memoryRange = {};
                memoryRange.sType               = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
                memoryRange.memory              = backendResource->deviceMemory;
                memoryRange.size                = initData.size;

                VulkanWrapper().vkFlushMappedMemoryRanges(backendContext->device, 1, &memoryRange);
            }

            VulkanWrapper().vkUnmapMemory(backendContext->device, backendResource->deviceMemory);
            return FFX_OK;
        }

        // TODO ARM: This is rather janky, with the aligned width rework we should also figure this out.
        // A buffer doesn't have a width - but when working in the CT OF component, buffers are aliasing images, so they actually do have a width.
        // This width, however, is _not_ the buffer stride so it needs a different field, or image-buffer aliasing for all buffers whose width is used in the shaders.
        outResource->alignedWidth = createResourceDescription->resourceDescription.stride;

        break;
    }
    case FFX_RESOURCE_TYPE_TEXTURE1D:
    case FFX_RESOURCE_TYPE_TEXTURE2D:
    case FFX_RESOURCE_TYPE_TEXTURE_CUBE:
    case FFX_RESOURCE_TYPE_TEXTURE3D:
    {
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType         = ffxGetVKImageTypeFromResourceType(createResourceDescription->resourceDescription.type);
        imageInfo.extent.width      = createResourceDescription->resourceDescription.width;
        imageInfo.extent.height =
            createResourceDescription->resourceDescription.type == FFX_RESOURCE_TYPE_TEXTURE1D ? 1 : createResourceDescription->resourceDescription.height;
        imageInfo.extent.depth = (createResourceDescription->resourceDescription.type == FFX_RESOURCE_TYPE_TEXTURE3D ||
                                  createResourceDescription->resourceDescription.type == FFX_RESOURCE_TYPE_TEXTURE_CUBE)
                                     ? createResourceDescription->resourceDescription.depth
                                     : 1;
        imageInfo.mipLevels   = backendResource->resourceDescription.mipCount;
        imageInfo.arrayLayers = (createResourceDescription->resourceDescription.type == FFX_RESOURCE_TYPE_TEXTURE1D ||
                                 createResourceDescription->resourceDescription.type == FFX_RESOURCE_TYPE_TEXTURE2D)
                                    ? createResourceDescription->resourceDescription.depth
                                    : 1;
        imageInfo.format        = getVkFormatFromSurfaceFormatAndUsage(createResourceDescription->resourceDescription.format, resourceDesc.usage);
        imageInfo.tiling        = createAliasedBuffer ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;  // GetImageSubresourceLayout requires linear tiling
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage         = getVKImageUsageFlagsFromResourceUsage(resourceDesc.usage);
        imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

        if (FFX_CONTAINS_FLAG(resourceDesc.usage, FFX_RESOURCE_USAGE_UAV) && ffxIsSurfaceFormatSRGB(createResourceDescription->resourceDescription.format))
        {
            imageInfo.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
            imageInfo.format = ffxGetVkFormatFromSurfaceFormat(ffxGetSurfaceFormatFromGamma(createResourceDescription->resourceDescription.format));
        }

        if (VulkanWrapper().vkCreateImage(backendContext->device, &imageInfo, nullptr, &backendResource->imageResource) != VK_SUCCESS)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }

#ifdef _DEBUG
        setVKObjectName(backendContext->device, VK_OBJECT_TYPE_IMAGE, (uint64_t)backendResource->imageResource, backendResource->resourceName);
#endif

        VulkanWrapper().vkGetImageMemoryRequirements(backendContext->device, backendResource->imageResource, &memRequirements);

        // allocate the memory
        FfxErrorCode errorCode = allocateDeviceMemory(backendContext, memRequirements, requiredMemoryProperties, backendResource);
        if (FFX_OK != errorCode)
            return errorCode;

        if (VulkanWrapper().vkBindImageMemory(backendContext->device, backendResource->imageResource, backendResource->deviceMemory, 0) != VK_SUCCESS)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }
        outResource->alignedWidth = createResourceDescription->resourceDescription.width;

        break;
    }
    default:
        FFX_ASSERT_MESSAGE(false, "FFXInterface: Vulkan: Unsupported resource type creation requested.");
        break;
    }

    // Create SRVs and UAVs
    switch (createResourceDescription->resourceDescription.type)
    {
    case FFX_RESOURCE_TYPE_BUFFER:
        break;
    case FFX_RESOURCE_TYPE_TEXTURE1D:
    case FFX_RESOURCE_TYPE_TEXTURE2D:
    case FFX_RESOURCE_TYPE_TEXTURE_CUBE:
    case FFX_RESOURCE_TYPE_TEXTURE3D:
    {
        FFX_ASSERT_MESSAGE(effectContext.nextStaticResourceView + 1 < effectContext.nextDynamicResourceView[0],
                           "FFXInterface: Vulkan: We've run out of resource views. Please increase the size.");
        backendResource->srvViewIndex = effectContext.nextStaticResourceView++;

        FfxResourceType       type                = createResourceDescription->resourceDescription.type;
        VkImageViewCreateInfo imageViewCreateInfo = {};
        imageViewCreateInfo.sType                 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCreateInfo.pNext                 = nullptr;

        bool requestArrayView = FFX_CONTAINS_FLAG(backendResource->resourceDescription.usage, FFX_RESOURCE_USAGE_ARRAYVIEW);

        switch (type)
        {
        case FFX_RESOURCE_TYPE_TEXTURE1D:
            imageViewCreateInfo.viewType =
                (backendResource->resourceDescription.depth > 1 || requestArrayView) ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
            break;
        default:
        case FFX_RESOURCE_TYPE_TEXTURE2D:
            imageViewCreateInfo.viewType =
                (backendResource->resourceDescription.depth > 1 || requestArrayView) ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
            break;
        case FFX_RESOURCE_TYPE_TEXTURE_CUBE:
            imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
            break;
        case FFX_RESOURCE_TYPE_TEXTURE3D:
            imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
            break;
        }

        imageViewCreateInfo.image = backendResource->imageResource;
        imageViewCreateInfo.format =
            getVkFormatFromSurfaceFormatAndUsage(createResourceDescription->resourceDescription.format, backendResource->resourceDescription.usage);
        imageViewCreateInfo.components.r                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.g                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.b                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.a                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.subresourceRange.aspectMask     = getImageAspect(backendResource->resourceDescription.usage);
        imageViewCreateInfo.subresourceRange.baseMipLevel   = 0;
        imageViewCreateInfo.subresourceRange.levelCount     = backendResource->resourceDescription.mipCount;
        imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        imageViewCreateInfo.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;

        VkImageViewUsageCreateInfo imageViewUsageCreateInfo = {};
        setupImageViewForSRV(imageViewCreateInfo, imageViewUsageCreateInfo, backendResource->resourceDescription);

        // create an image view containing all mip levels for use as an srv
        if (VulkanWrapper().vkCreateImageView(
                backendContext->device, &imageViewCreateInfo, NULL, &backendContext->pResourceViews[backendResource->srvViewIndex].imageView) != VK_SUCCESS)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }
#ifdef _DEBUG
        setVKObjectName(backendContext->device,
                        VK_OBJECT_TYPE_IMAGE_VIEW,
                        (uint64_t)backendContext->pResourceViews[backendResource->srvViewIndex].imageView,
                        backendResource->resourceName);
#endif

        // create image views of individual mip levels for use as a uav
        if (FFX_CONTAINS_FLAG(backendResource->resourceDescription.usage, FFX_RESOURCE_USAGE_UAV))
        {
            const int32_t uavResourceViewCount = backendResource->resourceDescription.mipCount;
            FFX_ASSERT(effectContext.nextStaticResourceView + uavResourceViewCount < effectContext.nextDynamicResourceView[0]);

            backendResource->uavViewIndex = effectContext.nextStaticResourceView;
            backendResource->uavViewCount = uavResourceViewCount;

            imageViewCreateInfo.format =
                getVkFormatFromSurfaceFormatAndUsage(createResourceDescription->resourceDescription.format, backendResource->resourceDescription.usage);

            for (uint32_t mip = 0; mip < backendResource->resourceDescription.mipCount; ++mip)
            {
                imageViewCreateInfo.subresourceRange.levelCount   = 1;
                imageViewCreateInfo.subresourceRange.baseMipLevel = mip;

                if (VulkanWrapper().vkCreateImageView(
                        backendContext->device, &imageViewCreateInfo, NULL, &backendContext->pResourceViews[backendResource->uavViewIndex + mip].imageView) !=
                    VK_SUCCESS)
                {
                    return FFX_ERROR_BACKEND_API_ERROR;
                }
#ifdef _DEBUG
                setVKObjectName(backendContext->device,
                                VK_OBJECT_TYPE_IMAGE_VIEW,
                                (uint64_t)backendContext->pResourceViews[backendResource->uavViewIndex + mip].imageView,
                                backendResource->resourceName);
#endif
            }

            effectContext.nextStaticResourceView += uavResourceViewCount;
        }

        if (createAliasedBuffer)
        {
            // We need to be carefull about the case where width != stride
            VkImageSubresource  subResource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
            VkSubresourceLayout subResourceLayout;
            VulkanWrapper().vkGetImageSubresourceLayout(backendContext->device, backendResource->imageResource, &subResource, &subResourceLayout);
            outResource->alignedWidth =
                subResourceLayout.rowPitch / ffxGetBytesPerElementFromSurfaceFormat(createResourceDescription->resourceDescription.format);

            // Create Aliased Buffer which shares the same device memory as Image
            VkBufferCreateInfo bufferInfo = {};
            bufferInfo.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size               = subResourceLayout.size;
            bufferInfo.usage              = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bufferInfo.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;

            if (VulkanWrapper().vkCreateBuffer(backendContext->device, &bufferInfo, NULL, &backendResource->aliasedBufferResource) != VK_SUCCESS)
            {
                return FFX_ERROR_BACKEND_API_ERROR;
            }

            if (VulkanWrapper().vkBindBufferMemory(backendContext->device, backendResource->aliasedBufferResource, backendResource->deviceMemory, 0) !=
                VK_SUCCESS)
            {
                return FFX_ERROR_BACKEND_API_ERROR;
            }
        }
        break;
    }
    default:
        FFX_ASSERT_MESSAGE(false, "FFXInterface: Vulkan: Unsupported resource view type creation requested.");
        break;
    }

    // create upload resource and upload job if needed
    if (initData.type != FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED)
    {
        FfxResourceInternal          copySrc;
        FfxCreateResourceDescription uploadDesc = {*createResourceDescription};
        uploadDesc.heapType                     = FFX_HEAP_TYPE_UPLOAD;
        uploadDesc.resourceDescription.type     = FFX_RESOURCE_TYPE_BUFFER;
        uploadDesc.resourceDescription.width    = static_cast<uint32_t>(initData.size);
        uploadDesc.resourceDescription.usage    = FFX_RESOURCE_USAGE_READ_ONLY;
        uploadDesc.initialState                 = FFX_RESOURCE_STATE_GENERIC_READ;
        uploadDesc.initData                     = createResourceDescription->initData;

        backendInterface->fpCreateResource(backendInterface, &uploadDesc, effectContextId, &copySrc);

        // setup the upload job
        FfxGpuJobDescription copyJob = {FFX_GPU_JOB_COPY};
#ifdef FFX_DEBUG
        strncpy(copyJob.jobLabel, "Resource Initialization Copy", FFX_RESOURCE_NAME_SIZE - 1);
        copyJob.jobLabel[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
        copyJob.copyJobDescriptor.src       = copySrc;
        copyJob.copyJobDescriptor.dst       = *outResource;
        copyJob.copyJobDescriptor.srcOffset = 0;
        copyJob.copyJobDescriptor.dstOffset = 0;
        copyJob.copyJobDescriptor.size      = 0;

        backendInterface->fpScheduleGpuJob(backendInterface, &copyJob);
    }

    backendResource->allocationSize = memRequirements.size;
    effectContext.vramUsage.totalUsageInBytes += static_cast<uint64_t>(backendResource->allocationSize);
    if ((createResourceDescription->resourceDescription.flags & FFX_RESOURCE_FLAGS_ALIASABLE) == FFX_RESOURCE_FLAGS_ALIASABLE)
    {
        effectContext.vramUsage.aliasableUsageInBytes += static_cast<uint64_t>(backendResource->allocationSize);
    }

    return FFX_OK;
}

FfxErrorCode DestroyResourceVK(FfxInterface* backendInterface, FfxResourceInternal resource, FfxUInt32 effectContextId)
{
    FFX_ASSERT(backendInterface != nullptr);
    BackendContext_VK*                backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;
    BackendContext_VK::EffectContext& effectContext  = backendContext->pEffectContexts[effectContextId];

    if ((resource.internalIndex >= int32_t(effectContextId * FFX_MAX_RESOURCE_COUNT)) && (resource.internalIndex < int32_t(effectContext.nextStaticResource)))
    {
        BackendContext_VK::Resource& backgroundResource = backendContext->pResources[resource.internalIndex];

        if (backgroundResource.resourceDescription.type == FFX_RESOURCE_TYPE_BUFFER)
        {
            // Destroy the resource
            if (backgroundResource.bufferResource != VK_NULL_HANDLE)
            {
                VulkanWrapper().vkDestroyBuffer(backendContext->device, backgroundResource.bufferResource, nullptr);
                backgroundResource.bufferResource = VK_NULL_HANDLE;
            }
        }
        else if (backgroundResource.resourceDescription.type == FFX_RESOURCE_TYPE_TENSOR)
        {
            // Destroy tensor view
            if (backendContext->pResourceViews[backgroundResource.tensorViewIndex].tensorView != VK_NULL_HANDLE)
            {
                VulkanWrapper().vkDestroyTensorViewARM(
                    backendContext->device, backendContext->pResourceViews[backgroundResource.tensorViewIndex].tensorView, nullptr);
                backendContext->pResourceViews[backgroundResource.tensorViewIndex].tensorView = VK_NULL_HANDLE;
                backgroundResource.tensorViewIndex                                            = -1;
            }
            // Destroy the tensor
            if (backgroundResource.tensorResource != VK_NULL_HANDLE)
            {
                VulkanWrapper().vkDestroyTensorARM(backendContext->device, backgroundResource.tensorResource, nullptr);
                backgroundResource.tensorResource = VK_NULL_HANDLE;
            }

            const auto flags         = backgroundResource.resourceDescription.flags;
            const bool imageAliased  = (flags & FFX_RESOURCE_FLAGS_IMAGE_ALIASED) == FFX_RESOURCE_FLAGS_IMAGE_ALIASED;
            const bool bufferAliased = (flags & FFX_RESOURCE_FLAGS_BUFFER_ALIASED) == FFX_RESOURCE_FLAGS_BUFFER_ALIASED;

            if (imageAliased && backgroundResource.aliasedTensorImageResource != VK_NULL_HANDLE)
            {
                // Destroy SRV
                if (backgroundResource.srvViewIndex >= 0)
                {
                    VulkanWrapper().vkDestroyImageView(
                        backendContext->device, backendContext->pResourceViews[backgroundResource.srvViewIndex].imageView, nullptr);
                    backendContext->pResourceViews[backgroundResource.srvViewIndex].imageView = VK_NULL_HANDLE;
                    backgroundResource.srvViewIndex                                           = 0;
                }

                // And UAVs
                if (FFX_CONTAINS_FLAG(backgroundResource.resourceDescription.usage, FFX_RESOURCE_USAGE_UAV))
                {
                    for (uint32_t i = 0; i < backgroundResource.uavViewCount; ++i)
                    {
                        if (backendContext->pResourceViews[backgroundResource.uavViewIndex + i].imageView != VK_NULL_HANDLE)
                        {
                            VulkanWrapper().vkDestroyImageView(
                                backendContext->device, backendContext->pResourceViews[backgroundResource.uavViewIndex + i].imageView, nullptr);
                            backendContext->pResourceViews[backgroundResource.uavViewIndex + i].imageView = VK_NULL_HANDLE;
                        }
                    }
                }

                // Reset indices to resource views
                backgroundResource.uavViewIndex = backgroundResource.srvViewIndex = -1;
                backgroundResource.uavViewCount                                   = 0;

                VulkanWrapper().vkDestroyImage(backendContext->device, backgroundResource.aliasedTensorImageResource, nullptr);
                backgroundResource.aliasedTensorImageResource = VK_NULL_HANDLE;
            }

            // Destroy aliased buffer
            if (bufferAliased && backgroundResource.aliasedTensorBufferResource != VK_NULL_HANDLE)
            {
                VulkanWrapper().vkDestroyBuffer(backendContext->device, backgroundResource.aliasedTensorBufferResource, nullptr);
                backgroundResource.aliasedTensorBufferResource = VK_NULL_HANDLE;
            }
        }
        else
        {
            // Destroy SRV
            if (backgroundResource.srvViewIndex >= 0)
            {
                VulkanWrapper().vkDestroyImageView(backendContext->device, backendContext->pResourceViews[backgroundResource.srvViewIndex].imageView, nullptr);
                backendContext->pResourceViews[backgroundResource.srvViewIndex].imageView = VK_NULL_HANDLE;
                backgroundResource.srvViewIndex                                           = 0;
            }

            // And UAVs
            if (FFX_CONTAINS_FLAG(backgroundResource.resourceDescription.usage, FFX_RESOURCE_USAGE_UAV))
            {
                for (uint32_t i = 0; i < backgroundResource.uavViewCount; ++i)
                {
                    if (backendContext->pResourceViews[backgroundResource.uavViewIndex + i].imageView != VK_NULL_HANDLE)
                    {
                        VulkanWrapper().vkDestroyImageView(
                            backendContext->device, backendContext->pResourceViews[backgroundResource.uavViewIndex + i].imageView, nullptr);
                        backendContext->pResourceViews[backgroundResource.uavViewIndex + i].imageView = VK_NULL_HANDLE;
                    }
                }
            }

            // Reset indices to resource views
            backgroundResource.uavViewIndex = backgroundResource.srvViewIndex = -1;
            backgroundResource.uavViewCount                                   = 0;

            // Destroy the resource
            if (backgroundResource.imageResource != VK_NULL_HANDLE)
            {
                VulkanWrapper().vkDestroyImage(backendContext->device, backgroundResource.imageResource, nullptr);
                backgroundResource.imageResource = VK_NULL_HANDLE;
                // Destroy aliased buffer
                if (backgroundResource.aliasedBufferResource != VK_NULL_HANDLE)
                {
                    VulkanWrapper().vkDestroyBuffer(backendContext->device, backgroundResource.aliasedBufferResource, nullptr);
                    backgroundResource.aliasedBufferResource = VK_NULL_HANDLE;
                }
            }
        }

        if (backgroundResource.deviceMemory)
        {
            VulkanWrapper().vkFreeMemory(backendContext->device, backgroundResource.deviceMemory, nullptr);
            backgroundResource.deviceMemory = VK_NULL_HANDLE;

            effectContext.vramUsage.totalUsageInBytes -= static_cast<uint64_t>(backgroundResource.allocationSize);
            if ((backendContext->pResources[resource.internalIndex].resourceDescription.flags & FFX_RESOURCE_FLAGS_ALIASABLE) == FFX_RESOURCE_FLAGS_ALIASABLE)
            {
                effectContext.vramUsage.aliasableUsageInBytes -= static_cast<uint64_t>(backgroundResource.allocationSize);
            }
            backgroundResource.allocationSize = 0;
        }
    }

    return FFX_OK;
}

FfxErrorCode MapResourceVK(FfxInterface* backendInterface, FfxResourceInternal resource, void** ptr)
{
    FFX_ASSERT(NULL != backendInterface);

    BackendContext_VK* backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;

    BackendContext_VK::Resource* res = &backendContext->pResources[resource.internalIndex];

    if ((VulkanWrapper().vkMapMemory(backendContext->device, res->deviceMemory, 0, res->resourceDescription.size, 0, ptr) != VK_SUCCESS))
        return FFX_ERROR_BACKEND_API_ERROR;

    return FFX_OK;
}

FfxErrorCode UnmapResourceVK(FfxInterface* backendInterface, FfxResourceInternal resource)
{
    FFX_ASSERT(NULL != backendInterface);

    BackendContext_VK* backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;

    BackendContext_VK::Resource* res = &backendContext->pResources[resource.internalIndex];

    VulkanWrapper().vkUnmapMemory(backendContext->device, res->deviceMemory);

    return FFX_OK;
}

FfxErrorCode RegisterResourceVK(FfxInterface*        backendInterface,
                                const FfxResource*   inFfxResource,
                                FfxUInt32            effectContextId,
                                FfxResourceInternal* outFfxResourceInternal)
{
    FFX_ASSERT(NULL != backendInterface);
    BackendContext_VK*                backendContext = (BackendContext_VK*)(backendInterface->scratchBuffer);
    BackendContext_VK::EffectContext& effectContext  = backendContext->pEffectContexts[effectContextId];

    if (inFfxResource->resource == nullptr)
    {
        outFfxResourceInternal->internalIndex = 0;  // Always maps to FFX_<feature>_RESOURCE_IDENTIFIER_NULL;
        return FFX_OK;
    }

    // In vulkan we need to treat dynamic resources a little differently due to needing views to live as long as the GPU needs them.
    // We will treat them more like static resources and use the nextDynamicResource as a "hint" for where it should be.
    // Failure to find the pre-existing resource at the expected location will force a search until the resource is found.
    // If it is not found, a new entry will be created
    FFX_ASSERT(effectContext.nextDynamicResource > effectContext.nextStaticResource);
    outFfxResourceInternal->internalIndex = effectContext.nextDynamicResource--;

    //bool setupDynamicResource = false;
    BackendContext_VK::Resource* backendResource = &backendContext->pResources[outFfxResourceInternal->internalIndex];

    // Start by seeing if this entry is empty, as that triggers an automatic setup of a new dynamic resource
    /*if (backendResource->uavViewIndex < 0 && backendResource->srvViewIndex < 0)
    {
        setupDynamicResource = true;
    }

    // If not a new resource, does it match what's current slotted for this dynamic resource
    if (!setupDynamicResource)
    {
        // If this is us, just return as everything is setup as needed
        if ((backendResource->resourceDescription.type == FFX_RESOURCE_TYPE_BUFFER && backendResource->bufferResource == (VkBuffer)inFfxResource->resource) ||
            (backendResource->resourceDescription.type != FFX_RESOURCE_TYPE_BUFFER && backendResource->imageResource == (VkImage)inFfxResource->resource))
            return FFX_OK;

        // If this isn't us, search until we either find our entry or an empty resource
        outFfxResourceInternal->internalIndex = (effectContextId * FFX_MAX_RESOURCE_COUNT) + FFX_MAX_RESOURCE_COUNT - 1;
        while (!setupDynamicResource)
        {
            FFX_ASSERT(outFfxResourceInternal->internalIndex > effectContext.nextStaticResource); // Safety check while iterating
            backendResource = &backendContext->pResources[outFfxResourceInternal->internalIndex];

            // Is this us?
            if ((backendResource->resourceDescription.type == FFX_RESOURCE_TYPE_BUFFER && backendResource->bufferResource == (VkBuffer)inFfxResource->resource) ||
                (backendResource->resourceDescription.type != FFX_RESOURCE_TYPE_BUFFER && backendResource->imageResource == (VkImage)inFfxResource->resource))
            {
                 copyResourceState(backendResource, inFfxResource);
                 return FFX_OK;
            }

            // Empty?
            if (backendResource->uavViewIndex == -1 && backendResource->srvViewIndex == -1)
            {
                setupDynamicResource = true;
                break;
            }

            --outFfxResourceInternal->internalIndex;
        }
    }*/

    // If we got here, we are setting up a new dynamic entry
    backendResource->resourceDescription = inFfxResource->description;
    if (inFfxResource->description.type == FFX_RESOURCE_TYPE_BUFFER)
        backendResource->bufferResource = reinterpret_cast<VkBuffer>(inFfxResource->resource);
    else
        backendResource->imageResource = reinterpret_cast<VkImage>(inFfxResource->resource);

    copyResourceState(backendResource, inFfxResource);

#ifdef _DEBUG
    if (inFfxResource->name[0])
    {
        strncpy(backendResource->resourceName, inFfxResource->name, sizeof(backendResource->resourceName) - 1);
        backendResource->resourceName[sizeof(backendResource->resourceName) - 1] = '\0';
    }
#endif

    // the first call of RegisterResource can be identified because

    //////////////////////////////////////////////////////////////////////////
    // Create SRVs and UAVs
    switch (backendResource->resourceDescription.type)
    {
    case FFX_RESOURCE_TYPE_BUFFER:
        break;
    case FFX_RESOURCE_TYPE_TEXTURE1D:
    case FFX_RESOURCE_TYPE_TEXTURE2D:
    case FFX_RESOURCE_TYPE_TEXTURE_CUBE:
    case FFX_RESOURCE_TYPE_TEXTURE3D:
    {
        FfxResourceType       type                = backendResource->resourceDescription.type;
        VkImageViewCreateInfo imageViewCreateInfo = {};
        imageViewCreateInfo.sType                 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCreateInfo.pNext                 = nullptr;

        bool requestArrayView = FFX_CONTAINS_FLAG(backendResource->resourceDescription.usage, FFX_RESOURCE_USAGE_ARRAYVIEW);

        switch (type)
        {
        case FFX_RESOURCE_TYPE_TEXTURE1D:
            imageViewCreateInfo.viewType =
                (backendResource->resourceDescription.depth > 1 || requestArrayView) ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
            break;
        default:
        case FFX_RESOURCE_TYPE_TEXTURE2D:
            imageViewCreateInfo.viewType =
                (backendResource->resourceDescription.depth > 1 || requestArrayView) ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
            break;
        case FFX_RESOURCE_TYPE_TEXTURE_CUBE:
            imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
            break;
        case FFX_RESOURCE_TYPE_TEXTURE3D:
            imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
            break;
        }

        imageViewCreateInfo.image = backendResource->imageResource;
        imageViewCreateInfo.format =
            getVkFormatFromSurfaceFormatAndUsage(backendResource->resourceDescription.format, backendResource->resourceDescription.usage);
        imageViewCreateInfo.components.r                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.g                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.b                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.a                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.subresourceRange.aspectMask     = getImageAspect(backendResource->resourceDescription.usage);
        imageViewCreateInfo.subresourceRange.baseMipLevel   = 0;
        imageViewCreateInfo.subresourceRange.levelCount     = backendResource->resourceDescription.mipCount;
        imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        imageViewCreateInfo.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;

        // create an image view containing all mip levels for use as an srv
        FFX_ASSERT(effectContext.nextDynamicResourceView[effectContext.frameIndex] >
                   ((effectContext.frameIndex == 0) ? effectContext.nextStaticResourceView
                                                    : getDynamicResourceViewsStartIndex(effectContextId, effectContext.frameIndex - 1)));
        backendResource->srvViewIndex = effectContext.nextDynamicResourceView[effectContext.frameIndex]--;

        VkImageViewUsageCreateInfo imageViewUsageCreateInfo = {};
        setupImageViewForSRV(imageViewCreateInfo, imageViewUsageCreateInfo, backendResource->resourceDescription);

        if (VulkanWrapper().vkCreateImageView(
                backendContext->device, &imageViewCreateInfo, NULL, &backendContext->pResourceViews[backendResource->srvViewIndex].imageView) != VK_SUCCESS)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }
#ifdef _DEBUG
        setVKObjectName(backendContext->device,
                        VK_OBJECT_TYPE_IMAGE_VIEW,
                        (uint64_t)backendContext->pResourceViews[backendResource->srvViewIndex].imageView,
                        backendResource->resourceName);
#endif

        // create image views of individual mip levels for use as a uav
        if (FFX_CONTAINS_FLAG(backendResource->resourceDescription.usage, FFX_RESOURCE_USAGE_UAV))
        {
            const int32_t uavResourceViewCount = backendResource->resourceDescription.mipCount;
            FFX_ASSERT(effectContext.nextDynamicResourceView[effectContext.frameIndex] - uavResourceViewCount + 1 >
                       ((effectContext.frameIndex == 0) ? effectContext.nextStaticResourceView
                                                        : getDynamicResourceViewsStartIndex(effectContextId, effectContext.frameIndex - 1)));
            backendResource->uavViewIndex = effectContext.nextDynamicResourceView[effectContext.frameIndex] - uavResourceViewCount + 1;
            backendResource->uavViewCount = uavResourceViewCount;

            imageViewCreateInfo.format =
                getVkFormatFromSurfaceFormatAndUsage(backendResource->resourceDescription.format, backendResource->resourceDescription.usage);
            imageViewCreateInfo.pNext = nullptr;

            for (uint32_t mip = 0; mip < backendResource->resourceDescription.mipCount; ++mip)
            {
                imageViewCreateInfo.subresourceRange.levelCount   = 1;
                imageViewCreateInfo.subresourceRange.baseMipLevel = mip;

                if (VulkanWrapper().vkCreateImageView(
                        backendContext->device, &imageViewCreateInfo, NULL, &backendContext->pResourceViews[backendResource->uavViewIndex + mip].imageView) !=
                    VK_SUCCESS)
                {
                    return FFX_ERROR_BACKEND_API_ERROR;
                }
#ifdef _DEBUG
                setVKObjectName(backendContext->device,
                                VK_OBJECT_TYPE_IMAGE_VIEW,
                                (uint64_t)backendContext->pResourceViews[backendResource->uavViewIndex + mip].imageView,
                                backendResource->resourceName);
#endif
            }
            effectContext.nextDynamicResourceView[effectContext.frameIndex] -= uavResourceViewCount;
        }
        break;
    }
    default:
        FFX_ASSERT_MESSAGE(false, "FFXInterface: Vulkan: Unsupported resource view type creation requested.");
        break;
    }

    return FFX_OK;
}

FfxResource GetResourceVK(FfxInterface* backendInterface, FfxResourceInternal inResource)
{
    FFX_ASSERT(nullptr != backendInterface);
    BackendContext_VK* backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;

    FfxResourceDescription ffxResDescription = backendInterface->fpGetResourceDescription(backendInterface, inResource);

    FfxResource resource = {};
    resource.resource    = reinterpret_cast<void*>(backendContext->pResources[inResource.internalIndex].imageResource);
    // If the internal resource state is undefined, that means we are importing a resource that
    // has not yet been initialized, so we will flag it as such to finish initializing it later
    // before it is used.
    if (backendContext->pResources[inResource.internalIndex].undefined)
    {
        ffxResDescription.flags = (FfxResourceFlags)((int)ffxResDescription.flags | FFX_RESOURCE_FLAGS_UNDEFINED);
        // Flag it as no longer being undefined as it will no longer be after workload
        // execution
        backendContext->pResources[inResource.internalIndex].undefined = false;
    }
    resource.state       = backendContext->pResources[inResource.internalIndex].currentState;
    resource.description = ffxResDescription;

#ifdef _DEBUG
    if (backendContext->pResources[inResource.internalIndex].resourceName)
    {
        strncpy(resource.name, backendContext->pResources[inResource.internalIndex].resourceName, FFX_RESOURCE_NAME_SIZE - 1);
        resource.name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
    }
#endif

    return resource;
}

// dispose dynamic resources: This should be called at the end of the frame
FfxErrorCode UnregisterResourcesVK(FfxInterface* backendInterface, FfxCommandList commandList, FfxUInt32 effectContextId)
{
    FFX_ASSERT(NULL != backendInterface);
    BackendContext_VK*                backendContext = (BackendContext_VK*)(backendInterface->scratchBuffer);
    BackendContext_VK::EffectContext& effectContext  = backendContext->pEffectContexts[effectContextId];

    // Walk back all the resources that don't belong to us and reset them to their initial state
    const uint32_t dynamicResourceIndexStart = getDynamicResourcesStartIndex(effectContextId);
    for (uint32_t resourceIndex = ++effectContext.nextDynamicResource; resourceIndex <= dynamicResourceIndexStart; ++resourceIndex)
    {
        FfxResourceInternal internalResource;
        internalResource.internalIndex = resourceIndex;

        BackendContext_VK::Resource* backendResource = &backendContext->pResources[resourceIndex];

        // Also clear out their srv/uav indices so they are regenerated each frame
        backendResource->uavViewIndex    = -1;
        backendResource->srvViewIndex    = -1;
        backendResource->tensorViewIndex = -1;

        // Add the barrier
        addBarrier(backendContext, &internalResource, backendResource->initialState);
    }

    FFX_ASSERT(nullptr != commandList);
    VkCommandBuffer pCmdList = reinterpret_cast<VkCommandBuffer>(commandList);

    flushBarriers(backendContext, pCmdList);

    // Just reset the dynamic resource index, but leave the images views.
    // They will be deleted in the first pipeline destroy call as they need to live until then
    effectContext.nextDynamicResource = dynamicResourceIndexStart;

    // destroy the views of the next frame
    effectContext.frameIndex = (effectContext.frameIndex + 1) % FFX_MAX_QUEUED_FRAMES;
    destroyDynamicViews(backendContext, effectContextId, effectContext.frameIndex);

    return FFX_OK;
}

FfxErrorCode registerStaticTextureSrv(BackendContext_VK* backendContext, const FfxResource* inResource, uint32_t index, FfxUInt32 effectContextId)
{
    BackendContext_VK::EffectContext& effectContext = backendContext->pEffectContexts[effectContextId];

    if (effectContext.bindlessTextureSrvHeapSize <= index)
    {
        FFX_ASSERT(false);
        return FFX_ERROR_INSUFFICIENT_MEMORY;
    }

    VkImage vkImage = reinterpret_cast<VkImage>(inResource->resource);

    // Create SRVs and UAVs
    switch (inResource->description.type)
    {
    case FFX_RESOURCE_TYPE_BUFFER:
    {
        FFX_ASSERT_MESSAGE(false, "FFXInterface: Vulkan: Attempting to register a Buffer as a Texture SRV.");
        break;
    }
    case FFX_RESOURCE_TYPE_TEXTURE1D:
    case FFX_RESOURCE_TYPE_TEXTURE2D:
    case FFX_RESOURCE_TYPE_TEXTURE_CUBE:
    case FFX_RESOURCE_TYPE_TEXTURE3D:
    {
        FfxResourceType       type                = inResource->description.type;
        VkImageViewCreateInfo imageViewCreateInfo = {};
        imageViewCreateInfo.sType                 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCreateInfo.pNext                 = nullptr;

        bool requestArrayView = FFX_CONTAINS_FLAG(inResource->description.usage, FFX_RESOURCE_USAGE_ARRAYVIEW);

        switch (type)
        {
        case FFX_RESOURCE_TYPE_TEXTURE1D:
            imageViewCreateInfo.viewType = (inResource->description.depth > 1 || requestArrayView) ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
            break;
        default:
        case FFX_RESOURCE_TYPE_TEXTURE2D:
            imageViewCreateInfo.viewType = (inResource->description.depth > 1 || requestArrayView) ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
            break;
        case FFX_RESOURCE_TYPE_TEXTURE_CUBE:
            imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
            break;
        case FFX_RESOURCE_TYPE_TEXTURE3D:
            imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
            break;
        }

        imageViewCreateInfo.image                         = vkImage;
        imageViewCreateInfo.format                        = getVkFormatFromSurfaceFormatAndUsage(inResource->description.format, inResource->description.usage);
        imageViewCreateInfo.components.r                  = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.g                  = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.b                  = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.a                  = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.subresourceRange.aspectMask   = getImageAspect(inResource->description.usage);
        imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
        imageViewCreateInfo.subresourceRange.levelCount   = inResource->description.mipCount;
        imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        imageViewCreateInfo.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;

        uint32_t imageViewIndex = effectContext.bindlessTextureSrvHeapStart + index;

        if (backendContext->pResourceViews[imageViewIndex].imageView)
            VulkanWrapper().vkDestroyImageView(backendContext->device, backendContext->pResourceViews[imageViewIndex].imageView, nullptr);

        if (VulkanWrapper().vkCreateImageView(backendContext->device, &imageViewCreateInfo, NULL, &backendContext->pResourceViews[imageViewIndex].imageView) !=
            VK_SUCCESS)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }

#ifdef _DEBUG
        setVKObjectName(
            backendContext->device, VK_OBJECT_TYPE_IMAGE_VIEW, (uint64_t)backendContext->pResourceViews[imageViewIndex].imageView, inResource->name);
#endif

        VkWriteDescriptorSet  writeDescriptorSet  = {};
        VkDescriptorImageInfo imageDescriptorInfo = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};

        writeDescriptorSet.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.dstSet          = effectContext.bindlessTextureSrvDescriptorSet;
        writeDescriptorSet.descriptorCount = 1;
        writeDescriptorSet.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writeDescriptorSet.pImageInfo      = &imageDescriptorInfo;
        writeDescriptorSet.dstBinding      = 0;
        writeDescriptorSet.dstArrayElement = index;

        imageDescriptorInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageDescriptorInfo.imageView   = backendContext->pResourceViews[imageViewIndex].imageView;

        VulkanWrapper().vkUpdateDescriptorSets(backendContext->device, 1, &writeDescriptorSet, 0, nullptr);

        return FFX_OK;

        break;
    }
    default:
        break;
    }

    return FFX_ERROR_INVALID_ARGUMENT;
}

FfxErrorCode registerStaticBufferSrv(BackendContext_VK* backendContext,
                                     const FfxResource* inResource,
                                     uint32_t           offset,
                                     uint32_t           size,
                                     uint32_t           stride,
                                     uint32_t           index,
                                     FfxUInt32          effectContextId)
{
    BackendContext_VK::EffectContext& effectContext = backendContext->pEffectContexts[effectContextId];

    if (effectContext.bindlessBufferSrvHeapSize <= index)
    {
        FFX_ASSERT(false);
        return FFX_ERROR_INSUFFICIENT_MEMORY;
    }

    if (inResource->description.type == FFX_RESOURCE_TYPE_BUFFER)
    {
        VkBuffer buffer = reinterpret_cast<VkBuffer>(inResource->resource);

        VkWriteDescriptorSet   writeDescriptorSet   = {};
        VkDescriptorBufferInfo bufferDescriptorInfo = {VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};

        writeDescriptorSet.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.dstSet          = effectContext.bindlessBufferSrvDescriptorSet;
        writeDescriptorSet.descriptorCount = 1;
        writeDescriptorSet.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSet.pBufferInfo     = &bufferDescriptorInfo;
        writeDescriptorSet.dstBinding      = 0;
        writeDescriptorSet.dstArrayElement = index;

        bufferDescriptorInfo.buffer = buffer;
        bufferDescriptorInfo.offset = offset;
        bufferDescriptorInfo.range  = size > 0 ? size : VK_WHOLE_SIZE;

        VulkanWrapper().vkUpdateDescriptorSets(backendContext->device, 1, &writeDescriptorSet, 0, nullptr);

        return FFX_OK;
    }
    else
        FFX_ASSERT_MESSAGE(false, "FFXInterface: Vulkan: Attempting to register a Texture as a Buffer SRV.");

    return FFX_ERROR_INVALID_ARGUMENT;
}

FfxErrorCode registerStaticTextureUav(BackendContext_VK* backendContext, const FfxResource* inResource, uint32_t mip, uint32_t index, FfxUInt32 effectContextId)
{
    BackendContext_VK::EffectContext& effectContext = backendContext->pEffectContexts[effectContextId];

    if (effectContext.bindlessTextureUavHeapSize <= index)
    {
        FFX_ASSERT(false);
        return FFX_ERROR_INSUFFICIENT_MEMORY;
    }

    VkImage vkImage = reinterpret_cast<VkImage>(inResource->resource);

    // Create SRVs and UAVs
    switch (inResource->description.type)
    {
    case FFX_RESOURCE_TYPE_BUFFER:
    {
        FFX_ASSERT_MESSAGE(false, "FFXInterface: Vulkan: Attempting to register a Buffer as a Texture UAV.");
        break;
    }
    case FFX_RESOURCE_TYPE_TEXTURE1D:
    case FFX_RESOURCE_TYPE_TEXTURE2D:
    case FFX_RESOURCE_TYPE_TEXTURE_CUBE:
    case FFX_RESOURCE_TYPE_TEXTURE3D:
    {
        FfxResourceType       type                = inResource->description.type;
        VkImageViewCreateInfo imageViewCreateInfo = {};
        imageViewCreateInfo.sType                 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCreateInfo.pNext                 = nullptr;

        bool requestArrayView = FFX_CONTAINS_FLAG(inResource->description.usage, FFX_RESOURCE_USAGE_ARRAYVIEW);

        switch (type)
        {
        case FFX_RESOURCE_TYPE_TEXTURE1D:
            imageViewCreateInfo.viewType = (inResource->description.depth > 1 || requestArrayView) ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
            break;
        default:
        case FFX_RESOURCE_TYPE_TEXTURE2D:
            imageViewCreateInfo.viewType = (inResource->description.depth > 1 || requestArrayView) ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
            break;
        case FFX_RESOURCE_TYPE_TEXTURE_CUBE:
            imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
            break;
        case FFX_RESOURCE_TYPE_TEXTURE3D:
            imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
            break;
        }

        imageViewCreateInfo.image                         = vkImage;
        imageViewCreateInfo.format                        = getVkFormatFromSurfaceFormatAndUsage(inResource->description.format, inResource->description.usage);
        imageViewCreateInfo.components.r                  = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.g                  = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.b                  = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.a                  = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.subresourceRange.aspectMask   = getImageAspect(inResource->description.usage);
        imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
        imageViewCreateInfo.subresourceRange.levelCount   = inResource->description.mipCount;
        imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        imageViewCreateInfo.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;

        uint32_t imageViewIndex = effectContext.bindlessTextureUavHeapStart + index;

        if (backendContext->pResourceViews[imageViewIndex].imageView)
            VulkanWrapper().vkDestroyImageView(backendContext->device, backendContext->pResourceViews[imageViewIndex].imageView, nullptr);

        if (VulkanWrapper().vkCreateImageView(backendContext->device, &imageViewCreateInfo, NULL, &backendContext->pResourceViews[imageViewIndex].imageView) !=
            VK_SUCCESS)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }

#ifdef _DEBUG
        setVKObjectName(
            backendContext->device, VK_OBJECT_TYPE_IMAGE_VIEW, (uint64_t)backendContext->pResourceViews[imageViewIndex].imageView, inResource->name);
#endif

        VkWriteDescriptorSet  writeDescriptorSet  = {};
        VkDescriptorImageInfo imageDescriptorInfo = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};

        writeDescriptorSet.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.dstSet          = effectContext.bindlessTextureUavDescriptorSet;
        writeDescriptorSet.descriptorCount = 1;
        writeDescriptorSet.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writeDescriptorSet.pImageInfo      = &imageDescriptorInfo;
        writeDescriptorSet.dstBinding      = 0;
        writeDescriptorSet.dstArrayElement = index;

        imageDescriptorInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageDescriptorInfo.imageView   = backendContext->pResourceViews[imageViewIndex].imageView;

        VulkanWrapper().vkUpdateDescriptorSets(backendContext->device, 1, &writeDescriptorSet, 0, nullptr);

        return FFX_OK;

        break;
    }
    default:
        break;
    }

    return FFX_ERROR_INVALID_ARGUMENT;
}

FfxErrorCode registerStaticBufferUav(BackendContext_VK* backendContext,
                                     const FfxResource* inResource,
                                     uint32_t           offset,
                                     uint32_t           size,
                                     uint32_t           stride,
                                     uint32_t           index,
                                     FfxUInt32          effectContextId)
{
    BackendContext_VK::EffectContext& effectContext = backendContext->pEffectContexts[effectContextId];

    if (effectContext.bindlessBufferUavHeapSize <= index)
    {
        FFX_ASSERT(false);
        return FFX_ERROR_INSUFFICIENT_MEMORY;
    }

    if (inResource->description.type == FFX_RESOURCE_TYPE_BUFFER)
    {
        VkBuffer buffer = reinterpret_cast<VkBuffer>(inResource->resource);

        VkWriteDescriptorSet   writeDescriptorSet   = {};
        VkDescriptorBufferInfo bufferDescriptorInfo = {VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};

        writeDescriptorSet.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.dstSet          = effectContext.bindlessBufferUavDescriptorSet;
        writeDescriptorSet.descriptorCount = 1;
        writeDescriptorSet.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSet.pBufferInfo     = &bufferDescriptorInfo;
        writeDescriptorSet.dstBinding      = 0;
        writeDescriptorSet.dstArrayElement = index;

        bufferDescriptorInfo.buffer = buffer;
        bufferDescriptorInfo.offset = offset;
        bufferDescriptorInfo.range  = size > 0 ? size : VK_WHOLE_SIZE;

        VulkanWrapper().vkUpdateDescriptorSets(backendContext->device, 1, &writeDescriptorSet, 0, nullptr);

        return FFX_OK;
    }
    else
        FFX_ASSERT_MESSAGE(false, "FFXInterface: Vulkan: Attempting to register a Texture as a Buffer UAV.");

    return FFX_ERROR_INVALID_ARGUMENT;
}

FfxErrorCode RegisterStaticResourceVK(FfxInterface* backendInterface, const FfxStaticResourceDescription* desc, FfxUInt32 effectContextId)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != desc);

    BackendContext_VK* backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;

    switch (desc->descriptorType)
    {
    case FFX_DESCRIPTOR_TEXTURE_SRV:
        return registerStaticTextureSrv(backendContext, desc->resource, desc->descriptorIndex, effectContextId);
    case FFX_DESCRIPTOR_BUFFER_SRV:
        return registerStaticBufferSrv(
            backendContext, desc->resource, desc->bufferOffset, desc->bufferSize, desc->bufferStride, desc->descriptorIndex, effectContextId);
    case FFX_DESCRIPTOR_TEXTURE_UAV:
        return registerStaticTextureUav(backendContext, desc->resource, desc->textureUavMip, desc->descriptorIndex, effectContextId);
    case FFX_DESCRIPTOR_BUFFER_UAV:
        return registerStaticBufferUav(
            backendContext, desc->resource, desc->bufferOffset, desc->bufferSize, desc->bufferStride, desc->descriptorIndex, effectContextId);
    default:
        return FFX_ERROR_INVALID_ARGUMENT;
    }
}

FfxResourceDescription GetResourceDescriptionVK(FfxInterface* backendInterface, FfxResourceInternal resource)
{
    FFX_ASSERT(NULL != backendInterface);
    BackendContext_VK* backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;

    FfxResourceDescription resourceDescription = backendContext->pResources[resource.internalIndex].resourceDescription;
    return resourceDescription;
}

FfxErrorCode StageConstantBufferDataVK(FfxInterface* backendInterface, void* data, FfxUInt32 size, FfxConstantBuffer* constantBuffer)
{
    FFX_ASSERT(NULL != backendInterface);
    BackendContext_VK* backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;

    if (data && constantBuffer)
    {
        if ((backendContext->stagingRingBufferBase + FFX_ALIGN_UP(size, 256)) >= FFX_CONSTANT_BUFFER_RING_BUFFER_SIZE)
            backendContext->stagingRingBufferBase = 0;

        uint32_t* dstPtr = (uint32_t*)(backendContext->pStagingRingBuffer + backendContext->stagingRingBufferBase);

        memcpy(dstPtr, data, size);

        constantBuffer->data            = dstPtr;
        constantBuffer->num32BitEntries = size / sizeof(uint32_t);

        backendContext->stagingRingBufferBase += FFX_ALIGN_UP(size, 256);

        return FFX_OK;
    }
    else
        return FFX_ERROR_INVALID_POINTER;
}

VkSamplerAddressMode FfxGetAddressModeVK(const FfxAddressMode& addressMode)
{
    switch (addressMode)
    {
    case FFX_ADDRESS_MODE_WRAP:
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case FFX_ADDRESS_MODE_MIRROR:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case FFX_ADDRESS_MODE_CLAMP:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case FFX_ADDRESS_MODE_BORDER:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case FFX_ADDRESS_MODE_MIRROR_ONCE:
        return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    default:
        FFX_ASSERT_MESSAGE(false, "Unsupported addressing mode requested. Please implement");
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        break;
    }
}

FfxErrorCode CreateComputePipelineVK(FfxInterface*                 backendInterface,
                                     FfxEffect                     effect,
                                     FfxPass                       pass,
                                     uint32_t                      permutationOptions,
                                     const FfxPipelineDescription* pipelineDescription,
                                     FfxUInt32                     effectContextId,
                                     FfxPipelineState*             outPipeline)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != pipelineDescription);

    BackendContext_VK*                backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;
    BackendContext_VK::EffectContext& effectContext  = backendContext->pEffectContexts[effectContextId];

    // start by fetching the shader blob
    FfxShaderBlob shaderBlob = {};
    // WON'T WORK WITH FSR3!!
    FFX_VALIDATE(backendInterface->fpGetPermutationBlobByIndex(effect, pass, permutationOptions, &shaderBlob, nullptr, nullptr));
    FFX_ASSERT(shaderBlob.data && shaderBlob.size);

    //////////////////////////////////////////////////////////////////////////
    // One root signature (or pipeline layout) per pipeline
    FFX_ASSERT_MESSAGE(effectContext.nextPipelineLayout < (effectContextId * FFX_MAX_PASS_COUNT) + FFX_MAX_PASS_COUNT,
                       "FFXInterface: Vulkan: Ran out of pipeline layouts. Please increase FFX_MAX_PASS_COUNT");
    BackendContext_VK::PipelineLayout* pPipelineLayout = &backendContext->pPipelineLayouts[effectContext.nextPipelineLayout++];

    // Start by creating samplers
    FFX_ASSERT(pipelineDescription->samplerCount <= FFX_MAX_SAMPLERS);
    const size_t samplerCount = pipelineDescription->samplerCount;
    for (uint32_t currentSamplerIndex = 0; currentSamplerIndex < samplerCount; ++currentSamplerIndex)
    {
        VkSamplerCreateInfo createInfo = {};

        createInfo.maxLod                  = pipelineDescription->samplers[currentSamplerIndex].unnormalizedCoordinates ? 0.f : VK_LOD_CLAMP_NONE;
        createInfo.unnormalizedCoordinates = pipelineDescription->samplers[currentSamplerIndex].unnormalizedCoordinates ? VK_TRUE : VK_FALSE;

        createInfo.sType  = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        createInfo.pNext  = nullptr;
        createInfo.flags  = 0;
        createInfo.minLod = 0.f;

        createInfo.anisotropyEnable = false;
        createInfo.compareEnable    = false;
        createInfo.compareOp        = VK_COMPARE_OP_NEVER;
        createInfo.borderColor      = VK_BORDER_COLOR_INT_TRANSPARENT_BLACK;

        createInfo.addressModeU = FfxGetAddressModeVK(pipelineDescription->samplers[currentSamplerIndex].addressModeU);
        createInfo.addressModeV = FfxGetAddressModeVK(pipelineDescription->samplers[currentSamplerIndex].addressModeV);
        createInfo.addressModeW = FfxGetAddressModeVK(pipelineDescription->samplers[currentSamplerIndex].addressModeW);

        // Set the right filter
        switch (pipelineDescription->samplers[currentSamplerIndex].filter)
        {
        case FFX_FILTER_TYPE_MINMAGMIP_POINT:
            createInfo.minFilter  = VK_FILTER_NEAREST;
            createInfo.magFilter  = VK_FILTER_NEAREST;
            createInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            break;
        case FFX_FILTER_TYPE_MINMAGMIP_LINEAR:
            createInfo.minFilter  = VK_FILTER_LINEAR;
            createInfo.magFilter  = VK_FILTER_LINEAR;
            createInfo.mipmapMode = createInfo.unnormalizedCoordinates ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR;
            break;
        case FFX_FILTER_TYPE_MINMAGLINEARMIP_POINT:
            createInfo.minFilter  = VK_FILTER_LINEAR;
            createInfo.magFilter  = VK_FILTER_LINEAR;
            createInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            break;

        default:
            FFX_ASSERT_MESSAGE(false, "FFXInterface: Vulkan: Unsupported filter type requested. Please implement");
            break;
        }

        if (VulkanWrapper().vkCreateSampler(backendContext->device, &createInfo, nullptr, &pPipelineLayout->samplers[currentSamplerIndex]) != VK_SUCCESS)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }
    }

    int32_t staticTextureSrvCount = 0;
    int32_t staticBufferSrvCount  = 0;
    int32_t staticTextureUavCount = 0;
    int32_t staticBufferUavCount  = 0;

    pPipelineLayout->staticTextureSrvSet = -1;
    pPipelineLayout->staticBufferSrvSet  = -1;
    pPipelineLayout->staticTextureUavSet = -1;
    pPipelineLayout->staticBufferUavSet  = -1;

    // Setup descriptor sets
    VkDescriptorSetLayoutBinding layoutBindings[MAX_DESCRIPTOR_SET_LAYOUTS];
    uint32_t                     numLayoutBindings = 0;

    // Support more when needed
    VkShaderStageFlags shaderStageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Samplers - just the static ones for now
    for (uint32_t currentSamplerIndex = 0; currentSamplerIndex < samplerCount; ++currentSamplerIndex)
        layoutBindings[numLayoutBindings++] = {
            currentSamplerIndex + SAMPLER_BINDING_SHIFT, VK_DESCRIPTOR_TYPE_SAMPLER, 1, shaderStageFlags, &pPipelineLayout->samplers[currentSamplerIndex]};

    // Texture SRVs
    for (uint32_t srvIndex = 0; srvIndex < shaderBlob.srvTextureCount; ++srvIndex)
    {
        // count static srvs separately.
        if (shaderBlob.boundSRVTextureSpaces[srvIndex] != 0)
        {
            if (staticTextureSrvCount > 0)
                FFX_ASSERT(pPipelineLayout->staticTextureSrvSet != shaderBlob.boundSRVTextureSpaces[srvIndex]);

            staticTextureSrvCount += shaderBlob.boundSRVTextureCounts[srvIndex];
            pPipelineLayout->staticTextureSrvSet = shaderBlob.boundSRVTextureSpaces[srvIndex];
            continue;
        }

        layoutBindings[numLayoutBindings++] = {
            shaderBlob.boundSRVTextures[srvIndex], VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, shaderBlob.boundSRVTextureCounts[srvIndex], shaderStageFlags, nullptr};
    }

    // Buffer SRVs
    for (uint32_t srvIndex = 0; srvIndex < shaderBlob.srvBufferCount; ++srvIndex)
    {
        // count static srvs separately.
        if (shaderBlob.boundSRVBufferSpaces[srvIndex] != 0)
        {
            if (staticBufferSrvCount > 0)
                FFX_ASSERT(pPipelineLayout->staticBufferSrvSet != shaderBlob.boundSRVBufferSpaces[srvIndex]);

            staticBufferSrvCount += shaderBlob.boundSRVBufferCounts[srvIndex];
            pPipelineLayout->staticBufferSrvSet = shaderBlob.boundSRVBufferSpaces[srvIndex];
            continue;
        }

        layoutBindings[numLayoutBindings++] = {
            shaderBlob.boundSRVBuffers[srvIndex], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, shaderBlob.boundSRVBufferCounts[srvIndex], shaderStageFlags, nullptr};
    }

    // Texture UAVs
    for (uint32_t uavIndex = 0; uavIndex < shaderBlob.uavTextureCount; ++uavIndex)
    {
        // count static uavs separately.
        if (shaderBlob.boundUAVTextureSpaces[uavIndex] != 0)
        {
            if (staticTextureUavCount > 0)
                FFX_ASSERT(pPipelineLayout->staticTextureUavSet != shaderBlob.boundUAVTextureSpaces[uavIndex]);

            staticTextureUavCount += shaderBlob.boundUAVTextureCounts[uavIndex];
            pPipelineLayout->staticTextureUavSet = shaderBlob.boundUAVTextureSpaces[uavIndex];
            continue;
        }

        layoutBindings[numLayoutBindings++] = {
            shaderBlob.boundUAVTextures[uavIndex], VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, shaderBlob.boundUAVTextureCounts[uavIndex], shaderStageFlags, nullptr};
    }

    // Buffer UAVs
    for (uint32_t uavIndex = 0; uavIndex < shaderBlob.uavBufferCount; ++uavIndex)
    {
        // count static uavs separately.
        if (shaderBlob.boundUAVBufferSpaces[uavIndex] != 0)
        {
            if (staticBufferUavCount > 0)
                FFX_ASSERT(pPipelineLayout->staticBufferUavSet != shaderBlob.boundUAVBufferSpaces[uavIndex]);

            staticBufferUavCount += shaderBlob.boundUAVBufferCounts[uavIndex];
            pPipelineLayout->staticBufferUavSet = shaderBlob.boundUAVBufferSpaces[uavIndex];
            continue;
        }

        layoutBindings[numLayoutBindings++] = {
            shaderBlob.boundUAVBuffers[uavIndex], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, shaderBlob.boundUAVBufferCounts[uavIndex], shaderStageFlags, nullptr};
    }

    // Constant buffers (uniforms)
    for (uint32_t cbIndex = 0; cbIndex < shaderBlob.cbvCount; ++cbIndex)
    {
        layoutBindings[numLayoutBindings++] = {shaderBlob.boundConstantBuffers[cbIndex],
                                               VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                               shaderBlob.boundConstantBufferCounts[cbIndex],
                                               shaderStageFlags,
                                               nullptr};
    }

    // Tensor SRVs
    for (uint32_t tensorIndex = 0; tensorIndex < shaderBlob.srvTensorCount; ++tensorIndex)
    {
        layoutBindings[numLayoutBindings++] = {
            shaderBlob.boundSRVTensors[tensorIndex], VK_DESCRIPTOR_TYPE_TENSOR_ARM, shaderBlob.boundSRVTensorCounts[tensorIndex], shaderStageFlags, nullptr};
    }

    // Tensor UAVs
    for (uint32_t tensorIndex = 0; tensorIndex < shaderBlob.uavTensorCount; ++tensorIndex)
    {
        layoutBindings[numLayoutBindings++] = {
            shaderBlob.boundUAVTensors[tensorIndex], VK_DESCRIPTOR_TYPE_TENSOR_ARM, shaderBlob.boundUAVTensorCounts[tensorIndex], shaderStageFlags, nullptr};
    }

    // Create the descriptor layout
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount                    = numLayoutBindings;
    layoutInfo.pBindings                       = layoutBindings;

    if (VulkanWrapper().vkCreateDescriptorSetLayout(backendContext->device, &layoutInfo, nullptr, &pPipelineLayout->descriptorSetLayout) != VK_SUCCESS)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // allocate descriptor sets
    pPipelineLayout->descriptorSetIndex = 0;
    for (uint32_t i = 0; i < (FFX_MAX_QUEUED_FRAMES * MAX_PIPELINE_USAGE_PER_FRAME); i++)
    {
        VkDescriptorSetAllocateInfo allocateInfo = {};
        allocateInfo.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool              = backendContext->descriptorPool;
        allocateInfo.descriptorSetCount          = 1;
        allocateInfo.pSetLayouts                 = &pPipelineLayout->descriptorSetLayout;

        if (VulkanWrapper().vkAllocateDescriptorSets(backendContext->device, &allocateInfo, &pPipelineLayout->descriptorSets[i]) != VK_SUCCESS)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }
    }

    uint32_t setCount = 0;

    VkDescriptorSetLayout layouts[5];

    layouts[setCount++] = pPipelineLayout->descriptorSetLayout;

    if (staticTextureSrvCount > 0)
    {
        layouts[pPipelineLayout->staticTextureSrvSet] = effectContext.bindlessTextureSrvDescriptorSetLayout;
        setCount++;
    }

    if (staticBufferSrvCount > 0)
    {
        layouts[pPipelineLayout->staticBufferSrvSet] = effectContext.bindlessBufferSrvDescriptorSetLayout;
        setCount++;
    }

    if (staticTextureUavCount > 0)
    {
        layouts[pPipelineLayout->staticTextureUavSet] = effectContext.bindlessTextureUavDescriptorSetLayout;
        setCount++;
    }

    if (staticBufferUavCount > 0)
    {
        layouts[pPipelineLayout->staticBufferUavSet] = effectContext.bindlessBufferUavDescriptorSetLayout;
        setCount++;
    }

    // create the pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount         = setCount;
    pipelineLayoutInfo.pSetLayouts            = layouts;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges    = nullptr;

    if (VulkanWrapper().vkCreatePipelineLayout(backendContext->device, &pipelineLayoutInfo, nullptr, &pPipelineLayout->pipelineLayout) != VK_SUCCESS)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // set the root signature to pipeline
    outPipeline->rootSignature = reinterpret_cast<FfxRootSignature>(pPipelineLayout);

    // Only set the command signature if this is setup as an indirect workload
    if (pipelineDescription->indirectWorkload)
    {
        // Only need the stride ahead of time in Vulkan
        outPipeline->cmdSignature = reinterpret_cast<FfxCommandSignature>(sizeof(VkDispatchIndirectCommand));
    }
    else
    {
        outPipeline->cmdSignature = nullptr;
    }

    uint32_t flattenedSrvTextureCount = 0;

    for (uint32_t srvIndex = 0; srvIndex < shaderBlob.srvTextureCount; ++srvIndex)
    {
        uint32_t slotIndex = shaderBlob.boundSRVTextures[srvIndex];
        uint32_t bindCount = shaderBlob.boundSRVTextureCounts[srvIndex];

        for (uint32_t arrayIndex = 0; arrayIndex < bindCount; arrayIndex++)
        {
            uint32_t bindingIndex = flattenedSrvTextureCount++;

            outPipeline->srvTextureBindings[bindingIndex].slotIndex  = slotIndex;
            outPipeline->srvTextureBindings[bindingIndex].arrayIndex = arrayIndex;
            strncpy(outPipeline->srvTextureBindings[bindingIndex].name, shaderBlob.boundSRVTextureNames[srvIndex], FFX_RESOURCE_NAME_SIZE - 1);
            outPipeline->srvTextureBindings[bindingIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
        }
    }

    outPipeline->srvTextureCount = flattenedSrvTextureCount;
    FFX_ASSERT(outPipeline->srvTextureCount < FFX_MAX_NUM_SRVS);

    uint32_t flattenedUavTextureCount = 0;

    for (uint32_t uavIndex = 0; uavIndex < shaderBlob.uavTextureCount; ++uavIndex)
    {
        uint32_t slotIndex = shaderBlob.boundUAVTextures[uavIndex];
        uint32_t bindCount = shaderBlob.boundUAVTextureCounts[uavIndex];

        for (uint32_t arrayIndex = 0; arrayIndex < bindCount; arrayIndex++)
        {
            uint32_t bindingIndex = flattenedUavTextureCount++;

            outPipeline->uavTextureBindings[bindingIndex].slotIndex  = slotIndex;
            outPipeline->uavTextureBindings[bindingIndex].arrayIndex = arrayIndex;
            strncpy(outPipeline->uavTextureBindings[bindingIndex].name, shaderBlob.boundUAVTextureNames[uavIndex], FFX_RESOURCE_NAME_SIZE - 1);
            outPipeline->uavTextureBindings[bindingIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
        }
    }

    outPipeline->uavTextureCount = flattenedUavTextureCount;
    FFX_ASSERT(outPipeline->uavTextureCount < FFX_MAX_NUM_UAVS);

    uint32_t flattenedSrvBufferCount = 0;

    for (uint32_t srvIndex = 0; srvIndex < shaderBlob.srvBufferCount; ++srvIndex)
    {
        uint32_t slotIndex  = shaderBlob.boundSRVBuffers[srvIndex];
        uint32_t spaceIndex = shaderBlob.boundSRVBufferSpaces[srvIndex];
        uint32_t bindCount  = shaderBlob.boundSRVBufferCounts[srvIndex];

        // Skip static resources
        if (spaceIndex == 1)
            continue;

        for (uint32_t arrayIndex = 0; arrayIndex < bindCount; arrayIndex++)
        {
            uint32_t bindingIndex = flattenedSrvBufferCount++;

            outPipeline->srvBufferBindings[bindingIndex].slotIndex  = slotIndex;
            outPipeline->srvBufferBindings[bindingIndex].arrayIndex = arrayIndex;
            strncpy(outPipeline->srvBufferBindings[bindingIndex].name, shaderBlob.boundSRVBufferNames[srvIndex], FFX_RESOURCE_NAME_SIZE - 1);
            outPipeline->srvBufferBindings[bindingIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
        }
    }

    outPipeline->srvBufferCount = flattenedSrvBufferCount;
    FFX_ASSERT(outPipeline->srvBufferCount < FFX_MAX_NUM_SRVS);

    uint32_t flattenedUavBufferCount = 0;

    for (uint32_t uavIndex = 0; uavIndex < shaderBlob.uavBufferCount; ++uavIndex)
    {
        uint32_t slotIndex = shaderBlob.boundUAVBuffers[uavIndex];
        uint32_t bindCount = shaderBlob.boundUAVBufferCounts[uavIndex];

        for (uint32_t arrayIndex = 0; arrayIndex < bindCount; arrayIndex++)
        {
            uint32_t bindingIndex = flattenedUavBufferCount++;

            outPipeline->uavBufferBindings[bindingIndex].slotIndex  = slotIndex;
            outPipeline->uavBufferBindings[bindingIndex].arrayIndex = arrayIndex;
            strncpy(outPipeline->uavBufferBindings[bindingIndex].name, shaderBlob.boundUAVBufferNames[uavIndex], FFX_RESOURCE_NAME_SIZE - 1);
            outPipeline->uavBufferBindings[bindingIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
        }
    }

    outPipeline->uavBufferCount = flattenedUavBufferCount;
    FFX_ASSERT(outPipeline->uavBufferCount < FFX_MAX_NUM_UAVS);

    {
        uint32_t flattenedSrvTensorCount = 0;

        for (uint32_t tensorIndex = 0; tensorIndex < shaderBlob.srvTensorCount; ++tensorIndex)
        {
            uint32_t slotIndex = shaderBlob.boundSRVTensors[tensorIndex];
            uint32_t bindCount = shaderBlob.boundSRVTensorCounts[tensorIndex];

            for (uint32_t arrayIndex = 0; arrayIndex < bindCount; arrayIndex++)
            {
                uint32_t bindingIndex = flattenedSrvTensorCount++;

                outPipeline->srvTensorBindings[bindingIndex].slotIndex  = slotIndex;
                outPipeline->srvTensorBindings[bindingIndex].arrayIndex = arrayIndex;
                strncpy(outPipeline->srvTensorBindings[bindingIndex].name, shaderBlob.boundSRVTensorNames[tensorIndex], FFX_RESOURCE_NAME_SIZE - 1);
                outPipeline->srvTensorBindings[bindingIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
            }
        }

        outPipeline->srvTensorCount = flattenedSrvTensorCount;
        FFX_ASSERT(outPipeline->srvTensorCount < FFX_MAX_NUM_TENSORS);
    }

    {
        uint32_t flattenedUavTensorCount = 0;

        for (uint32_t tensorIndex = 0; tensorIndex < shaderBlob.uavTensorCount; ++tensorIndex)
        {
            uint32_t slotIndex = shaderBlob.boundUAVTensors[tensorIndex];
            uint32_t bindCount = shaderBlob.boundUAVTensorCounts[tensorIndex];

            for (uint32_t arrayIndex = 0; arrayIndex < bindCount; arrayIndex++)
            {
                uint32_t bindingIndex = flattenedUavTensorCount++;

                outPipeline->uavTensorBindings[bindingIndex].slotIndex  = slotIndex;
                outPipeline->uavTensorBindings[bindingIndex].arrayIndex = arrayIndex;
                strncpy(outPipeline->uavTensorBindings[bindingIndex].name, shaderBlob.boundUAVTensorNames[tensorIndex], FFX_RESOURCE_NAME_SIZE - 1);
                outPipeline->uavTensorBindings[bindingIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
            }
        }

        outPipeline->uavTensorCount = flattenedUavTensorCount;
        FFX_ASSERT(outPipeline->uavTensorCount < FFX_MAX_NUM_TENSORS);
    }

    for (uint32_t cbIndex = 0; cbIndex < shaderBlob.cbvCount; ++cbIndex)
    {
        outPipeline->constantBufferBindings[cbIndex].slotIndex  = shaderBlob.boundConstantBuffers[cbIndex];
        outPipeline->constantBufferBindings[cbIndex].arrayIndex = 1;
        strncpy(outPipeline->constantBufferBindings[cbIndex].name, shaderBlob.boundConstantBufferNames[cbIndex], FFX_RESOURCE_NAME_SIZE - 1);
        outPipeline->constantBufferBindings[cbIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
    }

    outPipeline->constCount = shaderBlob.cbvCount;
    FFX_ASSERT(outPipeline->constCount < FFX_MAX_NUM_CONST_BUFFERS);

    outPipeline->staticTextureSrvCount = staticTextureSrvCount;
    FFX_ASSERT(outPipeline->staticTextureSrvCount <= effectContext.bindlessTextureSrvHeapSize);

    outPipeline->staticBufferSrvCount = staticBufferSrvCount;
    FFX_ASSERT(outPipeline->staticBufferSrvCount <= effectContext.bindlessBufferSrvHeapSize);

    outPipeline->staticTextureUavCount = staticTextureUavCount;
    FFX_ASSERT(outPipeline->staticTextureUavCount <= effectContext.bindlessTextureUavHeapSize);

    outPipeline->staticBufferUavCount = staticBufferUavCount;
    FFX_ASSERT(outPipeline->staticBufferUavCount <= effectContext.bindlessBufferUavHeapSize);

    // Todo when needed
    //outPipeline->samplerCount      = shaderBlob.samplerCount;
    //outPipeline->rtAccelStructCount= shaderBlob.rtAccelStructCount;

    //////////////////////////////////////////////////////////////////////////
    // pipeline creation
    FfxDeviceCapabilities capabilities;
    FFX_VALIDATE(backendInterface->fpGetDeviceCapabilities(backendInterface, &capabilities));
    FFX_ASSERT(backendInterface->devCapInitialized);

    // shader module
    VkShaderModule           shaderModule           = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.pCode                    = (uint32_t*)shaderBlob.data;
    shaderModuleCreateInfo.codeSize                 = shaderBlob.size;

    if (VulkanWrapper().vkCreateShaderModule(backendContext->device, &shaderModuleCreateInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // fill out shader stage create info
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage                           = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.pName                           = "main";
    shaderStageCreateInfo.module                          = shaderModule;

    // check if wave64 is requested
    bool isWave64 = false;
    ffxIsWave64(effect, permutationOptions, isWave64);
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroupSizeCreateInfo = {};
    if (isWave64 && (capabilities.waveLaneCountMin <= 64 && capabilities.waveLaneCountMax >= 64))
    {
        subgroupSizeCreateInfo.sType                = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
        subgroupSizeCreateInfo.requiredSubgroupSize = 64;
        shaderStageCreateInfo.pNext                 = &subgroupSizeCreateInfo;
    }

    // create the compute pipeline
    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType                       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage                       = shaderStageCreateInfo;
    pipelineCreateInfo.layout                      = pPipelineLayout->pipelineLayout;

    VkPipeline computePipeline = VK_NULL_HANDLE;
    if (VulkanWrapper().vkCreateComputePipelines(backendContext->device, nullptr, 1, &pipelineCreateInfo, nullptr, &computePipeline) != VK_SUCCESS)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // done with shader module, so clean up
    VulkanWrapper().vkDestroyShaderModule(backendContext->device, shaderModule, nullptr);

    // set the pipeline
    outPipeline->pipeline = reinterpret_cast<FfxPipeline>(computePipeline);

    // Setup the pipeline name
    strncpy(outPipeline->name, pipelineDescription->name, FFX_RESOURCE_NAME_SIZE - 1);
    outPipeline->name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';

    return FFX_OK;
}

FfxErrorCode CreateGraphicsPipelineVK(FfxInterface*                 backendInterface,
                                      FfxEffect                     effect,
                                      FfxPass                       pass,
                                      uint32_t                      permutationOptions,
                                      const FfxPipelineDescription* pipelineDescription,
                                      FfxUInt32                     effectContextId,
                                      FfxPipelineState*             outPipeline)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != pipelineDescription);

    BackendContext_VK*                backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;
    BackendContext_VK::EffectContext& effectContext  = backendContext->pEffectContexts[effectContextId];

    // start by fetching the shader blob
    FfxShaderBlob shaderBlob     = {};
    FfxShaderBlob vertShaderBlob = {};

    FFX_VALIDATE(backendInterface->fpGetPermutationBlobByIndex(effect, pass, permutationOptions, &shaderBlob, &vertShaderBlob, nullptr));
    FFX_ASSERT(shaderBlob.data && shaderBlob.size);
    FFX_ASSERT(vertShaderBlob.data && vertShaderBlob.size);

    //////////////////////////////////////////////////////////////////////////
    // One root signature (or pipeline layout) per pipeline
    FFX_ASSERT_MESSAGE(effectContext.nextPipelineLayout < (effectContextId * FFX_MAX_PASS_COUNT) + FFX_MAX_PASS_COUNT,
                       "FFXInterface: Vulkan: Ran out of pipeline layouts. Please increase FFX_MAX_PASS_COUNT");
    BackendContext_VK::PipelineLayout* pPipelineLayout = &backendContext->pPipelineLayouts[effectContext.nextPipelineLayout++];

    // Start by creating samplers
    FFX_ASSERT(pipelineDescription->samplerCount <= FFX_MAX_SAMPLERS);
    const size_t samplerCount = pipelineDescription->samplerCount;
    for (uint32_t currentSamplerIndex = 0; currentSamplerIndex < samplerCount; ++currentSamplerIndex)
    {
        VkSamplerCreateInfo createInfo = {};

        createInfo.maxLod                  = pipelineDescription->samplers[currentSamplerIndex].unnormalizedCoordinates ? 0.f : VK_LOD_CLAMP_NONE;
        createInfo.unnormalizedCoordinates = pipelineDescription->samplers[currentSamplerIndex].unnormalizedCoordinates ? VK_TRUE : VK_FALSE;

        createInfo.sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        createInfo.pNext            = nullptr;
        createInfo.flags            = 0;
        createInfo.minLod           = 0.f;
        createInfo.anisotropyEnable = false;
        createInfo.compareEnable    = false;
        createInfo.compareOp        = VK_COMPARE_OP_NEVER;
        createInfo.borderColor      = VK_BORDER_COLOR_INT_TRANSPARENT_BLACK;
        createInfo.addressModeU     = FfxGetAddressModeVK(pipelineDescription->samplers[currentSamplerIndex].addressModeU);
        createInfo.addressModeV     = FfxGetAddressModeVK(pipelineDescription->samplers[currentSamplerIndex].addressModeV);
        createInfo.addressModeW     = FfxGetAddressModeVK(pipelineDescription->samplers[currentSamplerIndex].addressModeW);

        // Set the right filter
        switch (pipelineDescription->samplers[currentSamplerIndex].filter)
        {
        case FFX_FILTER_TYPE_MINMAGMIP_POINT:
            createInfo.minFilter  = VK_FILTER_NEAREST;
            createInfo.magFilter  = VK_FILTER_NEAREST;
            createInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            break;
        case FFX_FILTER_TYPE_MINMAGMIP_LINEAR:
            createInfo.minFilter  = VK_FILTER_LINEAR;
            createInfo.magFilter  = VK_FILTER_LINEAR;
            createInfo.mipmapMode = createInfo.unnormalizedCoordinates ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR;
            break;
        case FFX_FILTER_TYPE_MINMAGLINEARMIP_POINT:
            createInfo.minFilter  = VK_FILTER_LINEAR;
            createInfo.magFilter  = VK_FILTER_LINEAR;
            createInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            break;

        default:
            FFX_ASSERT_MESSAGE(false, "FFXInterface: Vulkan: Unsupported filter type requested. Please implement");
            break;
        }

        if (VulkanWrapper().vkCreateSampler(backendContext->device, &createInfo, nullptr, &pPipelineLayout->samplers[currentSamplerIndex]) != VK_SUCCESS)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }
    }

    int32_t staticTextureSrvCount = 0;
    int32_t staticBufferSrvCount  = 0;
    int32_t staticTextureUavCount = 0;
    int32_t staticBufferUavCount  = 0;

    pPipelineLayout->staticTextureSrvSet = -1;
    pPipelineLayout->staticBufferSrvSet  = -1;
    pPipelineLayout->staticTextureUavSet = -1;
    pPipelineLayout->staticBufferUavSet  = -1;

    // Setup descriptor sets
    VkDescriptorSetLayoutBinding layoutBindings[MAX_DESCRIPTOR_SET_LAYOUTS];
    uint32_t                     numLayoutBindings = 0;

    // Support more when needed
    VkShaderStageFlags shaderStageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Samplers - just the static ones for now
    for (uint32_t currentSamplerIndex = 0; currentSamplerIndex < samplerCount; ++currentSamplerIndex)
        layoutBindings[numLayoutBindings++] = {
            currentSamplerIndex + SAMPLER_BINDING_SHIFT, VK_DESCRIPTOR_TYPE_SAMPLER, 1, shaderStageFlags, &pPipelineLayout->samplers[currentSamplerIndex]};

    // Texture SRVs
    for (uint32_t srvIndex = 0; srvIndex < shaderBlob.srvTextureCount; ++srvIndex)
    {
        // count static srvs separately.
        if (shaderBlob.boundSRVTextureSpaces[srvIndex] != 0)
        {
            if (staticTextureSrvCount > 0)
                FFX_ASSERT(pPipelineLayout->staticTextureSrvSet != shaderBlob.boundSRVTextureSpaces[srvIndex]);

            staticTextureSrvCount += shaderBlob.boundSRVTextureCounts[srvIndex];
            pPipelineLayout->staticTextureSrvSet = shaderBlob.boundSRVTextureSpaces[srvIndex];
            continue;
        }

        layoutBindings[numLayoutBindings++] = {
            shaderBlob.boundSRVTextures[srvIndex], VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, shaderBlob.boundSRVTextureCounts[srvIndex], shaderStageFlags, nullptr};
    }

    // Buffer SRVs
    for (uint32_t srvIndex = 0; srvIndex < shaderBlob.srvBufferCount; ++srvIndex)
    {
        // count static srvs separately.
        if (shaderBlob.boundSRVBufferSpaces[srvIndex] != 0)
        {
            if (staticBufferSrvCount > 0)
                FFX_ASSERT(pPipelineLayout->staticBufferSrvSet != shaderBlob.boundSRVBufferSpaces[srvIndex]);

            staticBufferSrvCount += shaderBlob.boundSRVBufferCounts[srvIndex];
            pPipelineLayout->staticBufferSrvSet = shaderBlob.boundSRVBufferSpaces[srvIndex];
            continue;
        }

        layoutBindings[numLayoutBindings++] = {
            shaderBlob.boundSRVBuffers[srvIndex], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, shaderBlob.boundSRVBufferCounts[srvIndex], shaderStageFlags, nullptr};
    }

    // Texture UAVs
    for (uint32_t uavIndex = 0; uavIndex < shaderBlob.uavTextureCount; ++uavIndex)
    {
        // count static uavs separately.
        if (shaderBlob.boundUAVTextureSpaces[uavIndex] != 0)
        {
            if (staticTextureUavCount > 0)
                FFX_ASSERT(pPipelineLayout->staticTextureUavSet != shaderBlob.boundUAVTextureSpaces[uavIndex]);

            staticTextureUavCount += shaderBlob.boundUAVTextureCounts[uavIndex];
            pPipelineLayout->staticTextureUavSet = shaderBlob.boundUAVTextureSpaces[uavIndex];
            continue;
        }

        layoutBindings[numLayoutBindings++] = {
            shaderBlob.boundUAVTextures[uavIndex], VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, shaderBlob.boundUAVTextureCounts[uavIndex], shaderStageFlags, nullptr};
    }

    // Buffer UAVs
    for (uint32_t uavIndex = 0; uavIndex < shaderBlob.uavBufferCount; ++uavIndex)
    {
        // count static uavs separately.
        if (shaderBlob.boundUAVBufferSpaces[uavIndex] != 0)
        {
            if (staticBufferUavCount > 0)
                FFX_ASSERT(pPipelineLayout->staticBufferUavSet != shaderBlob.boundUAVBufferSpaces[uavIndex]);

            staticBufferUavCount += shaderBlob.boundUAVBufferCounts[uavIndex];
            pPipelineLayout->staticBufferUavSet = shaderBlob.boundUAVBufferSpaces[uavIndex];
            continue;
        }

        layoutBindings[numLayoutBindings++] = {
            shaderBlob.boundUAVBuffers[uavIndex], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, shaderBlob.boundUAVBufferCounts[uavIndex], shaderStageFlags, nullptr};
    }

    // Tensor SRVs
    for (uint32_t tensorIndex = 0; tensorIndex < shaderBlob.srvTensorCount; ++tensorIndex)
    {
        layoutBindings[numLayoutBindings++] = {
            shaderBlob.boundSRVTensors[tensorIndex], VK_DESCRIPTOR_TYPE_TENSOR_ARM, shaderBlob.boundSRVTensorCounts[tensorIndex], shaderStageFlags, nullptr};
    }

    // Tensor UAVs
    for (uint32_t tensorIndex = 0; tensorIndex < shaderBlob.uavTensorCount; ++tensorIndex)
    {
        layoutBindings[numLayoutBindings++] = {
            shaderBlob.boundUAVTensors[tensorIndex], VK_DESCRIPTOR_TYPE_TENSOR_ARM, shaderBlob.boundUAVTensorCounts[tensorIndex], shaderStageFlags, nullptr};
    }

    // Constant buffers (uniforms)
    for (uint32_t cbIndex = 0; cbIndex < shaderBlob.cbvCount; ++cbIndex)
    {
        layoutBindings[numLayoutBindings++] = {shaderBlob.boundConstantBuffers[cbIndex],
                                               VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                               shaderBlob.boundConstantBufferCounts[cbIndex],
                                               shaderStageFlags,
                                               nullptr};
    }

    // Create the descriptor layout
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount                    = numLayoutBindings;
    layoutInfo.pBindings                       = layoutBindings;

    if (VulkanWrapper().vkCreateDescriptorSetLayout(backendContext->device, &layoutInfo, nullptr, &pPipelineLayout->descriptorSetLayout) != VK_SUCCESS)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // allocate descriptor sets
    pPipelineLayout->descriptorSetIndex = 0;
    for (uint32_t i = 0; i < (FFX_MAX_QUEUED_FRAMES * MAX_PIPELINE_USAGE_PER_FRAME); i++)
    {
        VkDescriptorSetAllocateInfo allocateInfo = {};
        allocateInfo.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool              = backendContext->descriptorPool;
        allocateInfo.descriptorSetCount          = 1;
        allocateInfo.pSetLayouts                 = &pPipelineLayout->descriptorSetLayout;

        if (VulkanWrapper().vkAllocateDescriptorSets(backendContext->device, &allocateInfo, &pPipelineLayout->descriptorSets[i]) != VK_SUCCESS)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }
    }

    uint32_t setCount = 0;

    VkDescriptorSetLayout layouts[5];

    layouts[setCount++] = pPipelineLayout->descriptorSetLayout;

    if (staticTextureSrvCount > 0)
    {
        layouts[pPipelineLayout->staticTextureSrvSet] = effectContext.bindlessTextureSrvDescriptorSetLayout;
        setCount++;
    }

    if (staticBufferSrvCount > 0)
    {
        layouts[pPipelineLayout->staticBufferSrvSet] = effectContext.bindlessBufferSrvDescriptorSetLayout;
        setCount++;
    }

    if (staticTextureUavCount > 0)
    {
        layouts[pPipelineLayout->staticTextureUavSet] = effectContext.bindlessTextureUavDescriptorSetLayout;
        setCount++;
    }

    if (staticBufferUavCount > 0)
    {
        layouts[pPipelineLayout->staticBufferUavSet] = effectContext.bindlessBufferUavDescriptorSetLayout;
        setCount++;
    }

    // create the pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount         = setCount;
    pipelineLayoutInfo.pSetLayouts            = layouts;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges    = nullptr;

    if (VulkanWrapper().vkCreatePipelineLayout(backendContext->device, &pipelineLayoutInfo, nullptr, &pPipelineLayout->pipelineLayout) != VK_SUCCESS)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // set the root signature to pipeline
    outPipeline->rootSignature = reinterpret_cast<FfxRootSignature>(pPipelineLayout);
    // Make sure textures are explicitly set to 0, as optical flow pipelines and generic data graph pipelines share the execute functions
    outPipeline->uavTextureCount = 0;
    outPipeline->srvTextureCount = 0;
    outPipeline->rtCount         = 0;

    // Only set the command signature if this is setup as an indirect workload
    if (pipelineDescription->indirectWorkload)
    {
        // Only need the stride ahead of time in Vulkan
        outPipeline->cmdSignature = reinterpret_cast<FfxCommandSignature>(sizeof(VkDispatchIndirectCommand));
    }
    else
    {
        outPipeline->cmdSignature = nullptr;
    }

    uint32_t flattenedSrvTextureCount = 0;

    for (uint32_t srvIndex = 0; srvIndex < shaderBlob.srvTextureCount; ++srvIndex)
    {
        uint32_t slotIndex = shaderBlob.boundSRVTextures[srvIndex];
        uint32_t bindCount = shaderBlob.boundSRVTextureCounts[srvIndex];

        for (uint32_t arrayIndex = 0; arrayIndex < bindCount; arrayIndex++)
        {
            uint32_t bindingIndex = flattenedSrvTextureCount++;

            outPipeline->srvTextureBindings[bindingIndex].slotIndex  = slotIndex;
            outPipeline->srvTextureBindings[bindingIndex].arrayIndex = arrayIndex;
            strncpy(outPipeline->srvTextureBindings[bindingIndex].name, shaderBlob.boundSRVTextureNames[srvIndex], FFX_RESOURCE_NAME_SIZE - 1);
            outPipeline->srvTextureBindings[bindingIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
        }
    }

    outPipeline->srvTextureCount = flattenedSrvTextureCount;
    FFX_ASSERT(outPipeline->srvTextureCount < FFX_MAX_NUM_SRVS);

    uint32_t flattenedUavTextureCount = 0;

    for (uint32_t uavIndex = 0; uavIndex < shaderBlob.uavTextureCount; ++uavIndex)
    {
        uint32_t slotIndex = shaderBlob.boundUAVTextures[uavIndex];
        uint32_t bindCount = shaderBlob.boundUAVTextureCounts[uavIndex];

        for (uint32_t arrayIndex = 0; arrayIndex < bindCount; arrayIndex++)
        {
            uint32_t bindingIndex = flattenedUavTextureCount++;

            outPipeline->uavTextureBindings[bindingIndex].slotIndex  = slotIndex;
            outPipeline->uavTextureBindings[bindingIndex].arrayIndex = arrayIndex;
            strncpy(outPipeline->uavTextureBindings[bindingIndex].name, shaderBlob.boundUAVTextureNames[uavIndex], FFX_RESOURCE_NAME_SIZE - 1);
            outPipeline->uavTextureBindings[bindingIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
        }
    }

    outPipeline->uavTextureCount = flattenedUavTextureCount;
    FFX_ASSERT(outPipeline->uavTextureCount < FFX_MAX_NUM_UAVS);

    uint32_t flattenedRtTextureCount = 0;

    for (uint32_t rtIndex = 0; rtIndex < shaderBlob.rtTextureCount; ++rtIndex)
    {
        uint32_t slotIndex = shaderBlob.boundRTTextures[rtIndex];
        uint32_t bindCount = shaderBlob.boundRTTextureCounts[rtIndex];

        for (uint32_t arrayIndex = 0; arrayIndex < bindCount; arrayIndex++)
        {
            uint32_t bindingIndex = flattenedRtTextureCount++;

            outPipeline->rtBindings[bindingIndex].slotIndex  = slotIndex;
            outPipeline->rtBindings[bindingIndex].arrayIndex = arrayIndex;
            strncpy(outPipeline->rtBindings[bindingIndex].name, shaderBlob.boundRTTextureNames[rtIndex], FFX_RESOURCE_NAME_SIZE - 1);
            outPipeline->rtBindings[bindingIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
        }
    }

    outPipeline->rtCount = flattenedRtTextureCount;
    FFX_ASSERT(outPipeline->rtCount < FFX_MAX_NUM_RTS);

    uint32_t flattenedSrvBufferCount = 0;

    for (uint32_t srvIndex = 0; srvIndex < shaderBlob.srvBufferCount; ++srvIndex)
    {
        uint32_t slotIndex  = shaderBlob.boundSRVBuffers[srvIndex];
        uint32_t spaceIndex = shaderBlob.boundSRVBufferSpaces[srvIndex];
        uint32_t bindCount  = shaderBlob.boundSRVBufferCounts[srvIndex];

        // Skip static resources
        if (spaceIndex == 1)
            continue;

        for (uint32_t arrayIndex = 0; arrayIndex < bindCount; arrayIndex++)
        {
            uint32_t bindingIndex = flattenedSrvBufferCount++;

            outPipeline->srvBufferBindings[bindingIndex].slotIndex  = slotIndex;
            outPipeline->srvBufferBindings[bindingIndex].arrayIndex = arrayIndex;
            strncpy(outPipeline->srvBufferBindings[bindingIndex].name, shaderBlob.boundSRVBufferNames[srvIndex], FFX_RESOURCE_NAME_SIZE - 1);
            outPipeline->srvBufferBindings[bindingIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
        }
    }

    outPipeline->srvBufferCount = flattenedSrvBufferCount;
    FFX_ASSERT(outPipeline->srvBufferCount < FFX_MAX_NUM_SRVS);

    uint32_t flattenedUavBufferCount = 0;

    for (uint32_t uavIndex = 0; uavIndex < shaderBlob.uavBufferCount; ++uavIndex)
    {
        uint32_t slotIndex = shaderBlob.boundUAVBuffers[uavIndex];
        uint32_t bindCount = shaderBlob.boundUAVBufferCounts[uavIndex];

        for (uint32_t arrayIndex = 0; arrayIndex < bindCount; arrayIndex++)
        {
            uint32_t bindingIndex = flattenedUavBufferCount++;

            outPipeline->uavBufferBindings[bindingIndex].slotIndex  = slotIndex;
            outPipeline->uavBufferBindings[bindingIndex].arrayIndex = arrayIndex;
            strncpy(outPipeline->uavBufferBindings[bindingIndex].name, shaderBlob.boundUAVBufferNames[uavIndex], FFX_RESOURCE_NAME_SIZE - 1);
            outPipeline->uavBufferBindings[bindingIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
        }
    }

    outPipeline->uavBufferCount = flattenedUavBufferCount;
    FFX_ASSERT(outPipeline->uavBufferCount < FFX_MAX_NUM_UAVS);

    {
        uint32_t flattenedSrvTensorCount = 0;

        for (uint32_t tensorIndex = 0; tensorIndex < shaderBlob.srvTensorCount; ++tensorIndex)
        {
            uint32_t slotIndex = shaderBlob.boundSRVTensors[tensorIndex];
            uint32_t bindCount = shaderBlob.boundSRVTensorCounts[tensorIndex];

            for (uint32_t arrayIndex = 0; arrayIndex < bindCount; arrayIndex++)
            {
                uint32_t bindingIndex = flattenedSrvTensorCount++;

                outPipeline->srvTensorBindings[bindingIndex].slotIndex  = slotIndex;
                outPipeline->srvTensorBindings[bindingIndex].arrayIndex = arrayIndex;
                strncpy(outPipeline->srvTensorBindings[bindingIndex].name, shaderBlob.boundSRVTensorNames[tensorIndex], FFX_RESOURCE_NAME_SIZE - 1);
                outPipeline->srvTensorBindings[bindingIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
            }
        }

        outPipeline->srvTensorCount = flattenedSrvTensorCount;
        FFX_ASSERT(outPipeline->srvTensorCount < FFX_MAX_NUM_TENSORS);
    }

    {
        uint32_t flattenedUavTensorCount = 0;

        for (uint32_t tensorIndex = 0; tensorIndex < shaderBlob.uavTensorCount; ++tensorIndex)
        {
            uint32_t slotIndex = shaderBlob.boundUAVTensors[tensorIndex];
            uint32_t bindCount = shaderBlob.boundUAVTensorCounts[tensorIndex];

            for (uint32_t arrayIndex = 0; arrayIndex < bindCount; arrayIndex++)
            {
                uint32_t bindingIndex = flattenedUavTensorCount++;

                outPipeline->uavTensorBindings[bindingIndex].slotIndex  = slotIndex;
                outPipeline->uavTensorBindings[bindingIndex].arrayIndex = arrayIndex;
                strncpy(outPipeline->uavTensorBindings[bindingIndex].name, shaderBlob.boundUAVTensorNames[tensorIndex], FFX_RESOURCE_NAME_SIZE - 1);
                outPipeline->uavTensorBindings[bindingIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
            }
        }

        outPipeline->uavTensorCount = flattenedUavTensorCount;
        FFX_ASSERT(outPipeline->uavTensorCount < FFX_MAX_NUM_TENSORS);
    }

    for (uint32_t cbIndex = 0; cbIndex < shaderBlob.cbvCount; ++cbIndex)
    {
        outPipeline->constantBufferBindings[cbIndex].slotIndex  = shaderBlob.boundConstantBuffers[cbIndex];
        outPipeline->constantBufferBindings[cbIndex].arrayIndex = 1;
        strncpy(outPipeline->constantBufferBindings[cbIndex].name, shaderBlob.boundConstantBufferNames[cbIndex], FFX_RESOURCE_NAME_SIZE - 1);
        outPipeline->constantBufferBindings[cbIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
    }

    outPipeline->constCount = shaderBlob.cbvCount;
    FFX_ASSERT(outPipeline->constCount < FFX_MAX_NUM_CONST_BUFFERS);

    outPipeline->staticTextureSrvCount = staticTextureSrvCount;
    FFX_ASSERT(outPipeline->staticTextureSrvCount <= effectContext.bindlessTextureSrvHeapSize);

    outPipeline->staticBufferSrvCount = staticBufferSrvCount;
    FFX_ASSERT(outPipeline->staticBufferSrvCount <= effectContext.bindlessBufferSrvHeapSize);

    outPipeline->staticTextureUavCount = staticTextureUavCount;
    FFX_ASSERT(outPipeline->staticTextureUavCount <= effectContext.bindlessTextureUavHeapSize);

    outPipeline->staticBufferUavCount = staticBufferUavCount;
    FFX_ASSERT(outPipeline->staticBufferUavCount <= effectContext.bindlessBufferUavHeapSize);

    // Todo when needed
    //outPipeline->samplerCount      = shaderBlob.samplerCount;
    //outPipeline->rtAccelStructCount= shaderBlob.rtAccelStructCount;

    //////////////////////////////////////////////////////////////////////////
    // pipeline creation

    // shader module
    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.pCode                    = (uint32_t*)shaderBlob.data;
    shaderModuleCreateInfo.codeSize                 = shaderBlob.size;

    if (VulkanWrapper().vkCreateShaderModule(backendContext->device, &shaderModuleCreateInfo, nullptr, &pPipelineLayout->fragShaderModule) != VK_SUCCESS)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    shaderModuleCreateInfo          = {};
    shaderModuleCreateInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.pCode    = (uint32_t*)vertShaderBlob.data;
    shaderModuleCreateInfo.codeSize = vertShaderBlob.size;

    if (VulkanWrapper().vkCreateShaderModule(backendContext->device, &shaderModuleCreateInfo, nullptr, &pPipelineLayout->vertShaderModule) != VK_SUCCESS)
    {
        VulkanWrapper().vkDestroyShaderModule(backendContext->device, pPipelineLayout->fragShaderModule, nullptr);
        pPipelineLayout->fragShaderModule = VK_NULL_HANDLE;
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // Setup the pipeline name
    strncpy(outPipeline->name, pipelineDescription->name, FFX_RESOURCE_NAME_SIZE - 1);
    outPipeline->name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';

    return FFX_OK;
}

template <size_t SIZE, typename T>
int8_t findObject(T (&objs)[SIZE], uint64_t hash)
{
    int8_t idx = -1;

    for (int8_t i = 0; i < SIZE; i++)
    {
        objs[i].visitedFlag++;

        if (objs[i].hash == hash)
        {
            objs[i].visitedFlag = 0;
            idx                 = i;
        }
    }

    return idx;
}

template <size_t SIZE, typename T>
uint8_t getLRUIndex(T (&objs)[SIZE])
{
    FfxUInt32 max = 0;
    uint8_t   idx = 0;

    for (uint8_t i = 0; i < SIZE; i++)
    {
        if (objs[i].visitedFlag > max)
        {
            max = objs[i].visitedFlag;
            idx = i;
        }
    }

    objs[idx].visitedFlag = 0;

    return idx;
}

FfxErrorCode getOrCreateFrameBuffer(BackendContext_VK* backendContext, FfxGpuJobDescription* job)
{
    FFX_ASSERT(NULL != backendContext);
    BackendContext_VK::PipelineLayout* pipelineLayout = reinterpret_cast<BackendContext_VK::PipelineLayout*>(job->fragmentJobDescriptor.pipeline.rootSignature);
    FfxPipelineState*                  pipeline       = &job->fragmentJobDescriptor.pipeline;

    std::array<VkImageView, FFX_MAX_NUM_RTS> attachments;
    for (FfxUInt32 rtIndex = 0; rtIndex < pipeline->rtCount; ++rtIndex)
    {
        const FfxUInt32       resourceIndex    = job->fragmentJobDescriptor.rtTextures[rtIndex].resource.internalIndex;
        VkImageViewCreateInfo createInfo       = {};
        createInfo.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image                       = backendContext->pResources[resourceIndex].imageResource;
        createInfo.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format                      = getVkFormatFromSurfaceFormatAndUsage(backendContext->pResources[resourceIndex].resourceDescription.format,
                                                                 backendContext->pResources[resourceIndex].resourceDescription.usage);
        createInfo.components.r                = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g                = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b                = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a                = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = (backendContext->pResources[resourceIndex].resourceDescription.usage & FFX_RESOURCE_USAGE_DEPTHTARGET)
                                                     ? VK_IMAGE_ASPECT_DEPTH_BIT
                                                     : VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel   = 0;
        createInfo.subresourceRange.levelCount     = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount     = 1;

        uint64_t hash = arm::computeHash(&createInfo, sizeof(createInfo));

        int8_t idx = findObject(pipelineLayout->imageView, hash);

        // find image view
        if (idx != -1)
        {
            pipelineLayout->imageViewIndex = idx;
        }
        else
        {
            // not find, get lru index in the array
            pipelineLayout->imageViewIndex = getLRUIndex(pipelineLayout->imageView);

            if (pipelineLayout->imageView[pipelineLayout->imageViewIndex].handle != VK_NULL_HANDLE)
            {
                VulkanWrapper().vkDestroyImageView(backendContext->device, pipelineLayout->imageView[pipelineLayout->imageViewIndex].handle, VK_NULL_HANDLE);
            }

            if (VulkanWrapper().vkCreateImageView(
                    backendContext->device, &createInfo, nullptr, &pipelineLayout->imageView[pipelineLayout->imageViewIndex].handle) != VK_SUCCESS)
            {
                return FFX_ERROR_BACKEND_API_ERROR;
            }

            pipelineLayout->imageView[pipelineLayout->imageViewIndex].hash = hash;
        }

        attachments[rtIndex] = pipelineLayout->imageView[pipelineLayout->imageViewIndex].handle;
    }

    uint64_t hash = arm::computeHash(attachments.data(), pipeline->rtCount * sizeof(attachments[0]));

    VkFramebufferCreateInfo fbufCreateInfo = {};
    fbufCreateInfo.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbufCreateInfo.renderPass              = pipelineLayout->renderPass[pipelineLayout->renderPassIndex].handle;
    fbufCreateInfo.attachmentCount         = pipeline->rtCount;
    fbufCreateInfo.width                   = job->fragmentJobDescriptor.viewport[0];
    fbufCreateInfo.height                  = job->fragmentJobDescriptor.viewport[1];
    fbufCreateInfo.layers                  = 1;

    hash = arm::appendHash(&fbufCreateInfo, sizeof(fbufCreateInfo), hash);

    fbufCreateInfo.pAttachments = attachments.data();

    int8_t idx = findObject(pipelineLayout->frameBuffer, hash);

    // find frame buffer
    if (idx != -1)
    {
        pipelineLayout->frameBufferIndex = idx;
    }
    else
    {
        // not find, get lru index in the array
        pipelineLayout->frameBufferIndex = getLRUIndex(pipelineLayout->frameBuffer);

        if (pipelineLayout->frameBuffer[pipelineLayout->frameBufferIndex].handle != VK_NULL_HANDLE)
        {
            VulkanWrapper().vkDestroyFramebuffer(backendContext->device, pipelineLayout->frameBuffer[pipelineLayout->frameBufferIndex].handle, VK_NULL_HANDLE);
        }

        if (VulkanWrapper().vkCreateFramebuffer(
                backendContext->device, &fbufCreateInfo, nullptr, &pipelineLayout->frameBuffer[pipelineLayout->frameBufferIndex].handle) != VK_SUCCESS)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }

        pipelineLayout->frameBuffer[pipelineLayout->frameBufferIndex].hash = hash;
    }

    return FFX_OK;
}

FfxErrorCode getOrCreateRenderPass(BackendContext_VK* backendContext, FfxGpuJobDescription* job)
{
    FFX_ASSERT(NULL != backendContext);
    BackendContext_VK::PipelineLayout* pipelineLayout = reinterpret_cast<BackendContext_VK::PipelineLayout*>(job->fragmentJobDescriptor.pipeline.rootSignature);
    FfxPipelineState*                  pipeline       = &job->fragmentJobDescriptor.pipeline;

    std::array<VkAttachmentDescription, FFX_MAX_NUM_RTS> attachmentDescriptions;
    std::array<VkAttachmentReference, FFX_MAX_NUM_RTS>   colorAttachmentReferences;

    for (FfxUInt32 rtIndex = 0; rtIndex < pipeline->rtCount; ++rtIndex)
    {
        const FfxUInt32         resourceIndex         = job->fragmentJobDescriptor.rtTextures[rtIndex].resource.internalIndex;
        VkAttachmentDescription attachmentDescription = {};
        attachmentDescription.format         = getVkFormatFromSurfaceFormatAndUsage(backendContext->pResources[resourceIndex].resourceDescription.format,
                                                                            backendContext->pResources[resourceIndex].resourceDescription.usage);
        attachmentDescription.samples        = VK_SAMPLE_COUNT_1_BIT;
        attachmentDescription.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentDescription.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachmentDescription.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescription.initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachmentDescription.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        attachmentDescriptions[rtIndex] = attachmentDescription;

        VkAttachmentReference colorAttachmentReference = {};
        colorAttachmentReference.attachment            = rtIndex;
        colorAttachmentReference.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        colorAttachmentReferences[rtIndex] = colorAttachmentReference;
    }

    uint64_t hash = arm::computeHash(attachmentDescriptions.data(), pipeline->rtCount * sizeof(attachmentDescriptions[0]));
    hash          = arm::appendHash(colorAttachmentReferences.data(), pipeline->rtCount * sizeof(colorAttachmentReferences[0]), hash);

    // Note: this is a description of how the attachments of the render pass will be used in this sub pass
    // e.g. if they will be read in shaders and/or drawn to
    VkSubpassDescription subPassDescription = {};
    subPassDescription.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subPassDescription.colorAttachmentCount = pipeline->rtCount;
    subPassDescription.pColorAttachments    = colorAttachmentReferences.data();

    // Create the render pass
    VkRenderPassCreateInfo renderPassCreateInfo = {};
    renderPassCreateInfo.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassCreateInfo.attachmentCount        = pipeline->rtCount;
    renderPassCreateInfo.pAttachments           = attachmentDescriptions.data();
    renderPassCreateInfo.subpassCount           = 1;
    renderPassCreateInfo.pSubpasses             = &subPassDescription;

    int8_t idx = findObject(pipelineLayout->renderPass, hash);

    // find render pass
    if (idx != -1)
    {
        pipelineLayout->renderPassIndex = idx;
    }
    else
    {
        // not find, get lru index in the array
        pipelineLayout->renderPassIndex = getLRUIndex(pipelineLayout->renderPass);

        if (pipelineLayout->renderPass[pipelineLayout->renderPassIndex].handle != VK_NULL_HANDLE)
        {
            VulkanWrapper().vkDestroyRenderPass(backendContext->device, pipelineLayout->renderPass[pipelineLayout->renderPassIndex].handle, VK_NULL_HANDLE);
        }

        if (VulkanWrapper().vkCreateRenderPass(
                backendContext->device, &renderPassCreateInfo, nullptr, &pipelineLayout->renderPass[pipelineLayout->renderPassIndex].handle) != VK_SUCCESS)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }

        pipelineLayout->renderPass[pipelineLayout->renderPassIndex].hash = hash;
    }

    return FFX_OK;
}

FfxErrorCode getOrCreateGraphicsPipeline(BackendContext_VK* backendContext, FfxGpuJobDescription* job)
{
    FFX_ASSERT(NULL != backendContext);
    BackendContext_VK::PipelineLayout* pipelineLayout = reinterpret_cast<BackendContext_VK::PipelineLayout*>(job->fragmentJobDescriptor.pipeline.rootSignature);
    FfxPipelineState*                  pipeline       = &job->fragmentJobDescriptor.pipeline;

    // pipeline only depends on render pass, so compute hash first
    uint64_t hash = arm::computeHash(&pipelineLayout->renderPass[pipelineLayout->renderPassIndex].handle,
                                     sizeof(pipelineLayout->renderPass[pipelineLayout->renderPassIndex].handle));

    int8_t idx = findObject(pipelineLayout->graphicsPipeline, hash);

    // find graphics pipeline
    if (idx != -1)
    {
        pipelineLayout->graphicsPipelineIndex = idx;

        // set the pipeline
        pipeline->pipeline = reinterpret_cast<FfxPipeline>(pipelineLayout->graphicsPipeline[pipelineLayout->graphicsPipelineIndex].handle);

        return FFX_OK;
    }

    // fill out shader stage create info
    VkPipelineShaderStageCreateInfo fragShaderStageCreateInfo = {};
    fragShaderStageCreateInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageCreateInfo.stage                           = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageCreateInfo.pName                           = "main";
    fragShaderStageCreateInfo.module                          = pipelineLayout->fragShaderModule;

    VkPipelineShaderStageCreateInfo vertShaderStageCreateInfo = {};
    vertShaderStageCreateInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageCreateInfo.stage                           = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageCreateInfo.pName                           = "main";
    vertShaderStageCreateInfo.module                          = pipelineLayout->vertShaderModule;

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageCreateInfo, fragShaderStageCreateInfo};

    // Describe vertex input
    VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo = {};
    vertexInputCreateInfo.sType                                = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputCreateInfo.vertexBindingDescriptionCount        = 0;
    vertexInputCreateInfo.pVertexBindingDescriptions           = nullptr;
    vertexInputCreateInfo.vertexAttributeDescriptionCount      = 0;
    vertexInputCreateInfo.pVertexAttributeDescriptions         = nullptr;

    // Describe input assembly
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyCreateInfo = {};
    inputAssemblyCreateInfo.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyCreateInfo.topology                               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssemblyCreateInfo.primitiveRestartEnable                 = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportCreateInfo = {};
    viewportCreateInfo.sType                             = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportCreateInfo.viewportCount                     = 1;
    viewportCreateInfo.scissorCount                      = 1;

    VkDynamicState                   dynamicState[2]        = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo = {};
    dynamicStateCreateInfo.sType                            = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateCreateInfo.dynamicStateCount                = 2;
    dynamicStateCreateInfo.pDynamicStates                   = dynamicState;

    VkPipelineRasterizationStateCreateInfo rasterizationCreateInfo = {};
    rasterizationCreateInfo.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizationCreateInfo.depthClampEnable                       = VK_FALSE;
    rasterizationCreateInfo.rasterizerDiscardEnable                = VK_FALSE;
    rasterizationCreateInfo.polygonMode                            = VK_POLYGON_MODE_FILL;
    rasterizationCreateInfo.cullMode                               = VK_CULL_MODE_BACK_BIT;
    rasterizationCreateInfo.frontFace                              = VK_FRONT_FACE_CLOCKWISE;
    rasterizationCreateInfo.depthBiasEnable                        = VK_FALSE;
    rasterizationCreateInfo.depthBiasConstantFactor                = 0.0f;
    rasterizationCreateInfo.depthBiasClamp                         = 0.0f;
    rasterizationCreateInfo.depthBiasSlopeFactor                   = 0.0f;
    rasterizationCreateInfo.lineWidth                              = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampleCreateInfo = {};
    multisampleCreateInfo.sType                                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampleCreateInfo.rasterizationSamples                 = VK_SAMPLE_COUNT_1_BIT;
    multisampleCreateInfo.sampleShadingEnable                  = VK_FALSE;
    multisampleCreateInfo.minSampleShading                     = 1.0f;  // 0.f
    multisampleCreateInfo.alphaToCoverageEnable                = VK_FALSE;
    multisampleCreateInfo.alphaToOneEnable                     = VK_FALSE;

    std::array<VkPipelineColorBlendAttachmentState, FFX_MAX_NUM_RTS> colorBlendAttachmentStates;
    for (FfxUInt32 rtIndex = 0; rtIndex < pipeline->rtCount; ++rtIndex)
    {
        VkPipelineColorBlendAttachmentState colorBlendAttachmentState = {};
        colorBlendAttachmentState.blendEnable                         = VK_FALSE;
        colorBlendAttachmentState.srcColorBlendFactor                 = VK_BLEND_FACTOR_ONE;
        colorBlendAttachmentState.dstColorBlendFactor                 = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachmentState.colorBlendOp                        = VK_BLEND_OP_ADD;
        colorBlendAttachmentState.srcAlphaBlendFactor                 = VK_BLEND_FACTOR_ONE;
        colorBlendAttachmentState.dstAlphaBlendFactor                 = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachmentState.alphaBlendOp                        = VK_BLEND_OP_ADD;
        colorBlendAttachmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        colorBlendAttachmentStates[rtIndex] = colorBlendAttachmentState;
    }

    VkPipelineColorBlendStateCreateInfo colorBlendCreateInfo = {};
    colorBlendCreateInfo.sType                               = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendCreateInfo.logicOpEnable                       = VK_FALSE;
    colorBlendCreateInfo.logicOp                             = VK_LOGIC_OP_COPY;
    colorBlendCreateInfo.attachmentCount                     = pipeline->rtCount;
    colorBlendCreateInfo.pAttachments                        = colorBlendAttachmentStates.data();
    colorBlendCreateInfo.blendConstants[0]                   = 0.0f;
    colorBlendCreateInfo.blendConstants[1]                   = 0.0f;
    colorBlendCreateInfo.blendConstants[2]                   = 0.0f;
    colorBlendCreateInfo.blendConstants[3]                   = 0.0f;

    // Create the graphics pipeline
    VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType                        = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stageCount                   = 2;
    pipelineCreateInfo.pStages                      = shaderStages;
    pipelineCreateInfo.pVertexInputState            = &vertexInputCreateInfo;
    pipelineCreateInfo.pInputAssemblyState          = &inputAssemblyCreateInfo;
    pipelineCreateInfo.pViewportState               = &viewportCreateInfo;
    pipelineCreateInfo.pRasterizationState          = &rasterizationCreateInfo;
    pipelineCreateInfo.pMultisampleState            = &multisampleCreateInfo;
    pipelineCreateInfo.pColorBlendState             = &colorBlendCreateInfo;
    pipelineCreateInfo.pDynamicState                = &dynamicStateCreateInfo;
    pipelineCreateInfo.layout                       = pipelineLayout->pipelineLayout;
    pipelineCreateInfo.renderPass                   = pipelineLayout->renderPass[pipelineLayout->renderPassIndex].handle;
    pipelineCreateInfo.subpass                      = 0;
    pipelineCreateInfo.basePipelineHandle           = VK_NULL_HANDLE;
    pipelineCreateInfo.basePipelineIndex            = -1;

    // not find, get lru index in the array
    pipelineLayout->graphicsPipelineIndex = getLRUIndex(pipelineLayout->graphicsPipeline);

    if (pipelineLayout->graphicsPipeline[pipelineLayout->graphicsPipelineIndex].handle != VK_NULL_HANDLE)
    {
        VulkanWrapper().vkDestroyPipeline(
            backendContext->device, pipelineLayout->graphicsPipeline[pipelineLayout->graphicsPipelineIndex].handle, VK_NULL_HANDLE);
    }

    if (VulkanWrapper().vkCreateGraphicsPipelines(backendContext->device,
                                                  VK_NULL_HANDLE,
                                                  1,
                                                  &pipelineCreateInfo,
                                                  nullptr,
                                                  &pipelineLayout->graphicsPipeline[pipelineLayout->graphicsPipelineIndex].handle) != VK_SUCCESS)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    pipelineLayout->graphicsPipeline[pipelineLayout->graphicsPipelineIndex].hash = hash;

    // set the pipeline
    pipeline->pipeline = reinterpret_cast<FfxPipeline>(pipelineLayout->graphicsPipeline[pipelineLayout->graphicsPipelineIndex].handle);

    return FFX_OK;
}

FfxErrorCode CreateDataGraphPipelineVK(FfxInterface*                 backendInterface,
                                       FfxEffect                     effect,
                                       FfxPass                       passId,
                                       uint32_t                      permutationOptions,
                                       const FfxPipelineDescription* desc,
                                       FfxUInt32                     effectContextId,
                                       FfxUInt32                     dataGraphWidth,
                                       FfxUInt32                     dataGraphHeight,
                                       FfxPipelineState*             outPipeline)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(desc != nullptr);

    BackendContext_VK*                backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;
    BackendContext_VK::EffectContext& effectContext  = backendContext->pEffectContexts[effectContextId];

    // start by fetching the shader blob
    FfxDataGraphBlob dataGraphBlob = {};
    FFX_VALIDATE(backendInterface->fpGetPermutationBlobByIndex(effect, passId, permutationOptions, nullptr, nullptr, &dataGraphBlob));

    //////////////////////////////////////////////////////////////////////////
    // One root signature (or pipeline layout) per pipeline
    FFX_ASSERT_MESSAGE(effectContext.nextPipelineLayout < (effectContextId * FFX_MAX_PASS_COUNT) + FFX_MAX_PASS_COUNT,
                       "FFXInterface: Vulkan: Ran out of pipeline layouts. Please increase FFX_MAX_PASS_COUNT");
    BackendContext_VK::PipelineLayout* pPipelineLayout = &backendContext->pPipelineLayouts[effectContext.nextPipelineLayout++];

    // Setup descriptor sets
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

    // Support more when needed
    VkShaderStageFlags shaderStageFlags = VK_SHADER_STAGE_ALL;

    // Input Tensors
    for (uint32_t tensorIndex = 0; tensorIndex < dataGraphBlob.inputTensorNums; ++tensorIndex)
    {
        VkDescriptorSetLayoutBinding binding = {dataGraphBlob.inputTensorBindings[tensorIndex], VK_DESCRIPTOR_TYPE_TENSOR_ARM, 1, shaderStageFlags, nullptr};
        layoutBindings.emplace_back(binding);
    }

    // Output Tensors
    for (uint32_t tensorIndex = 0; tensorIndex < dataGraphBlob.outputTensorNums; ++tensorIndex)
    {
        VkDescriptorSetLayoutBinding binding = {dataGraphBlob.outputTensorBindings[tensorIndex], VK_DESCRIPTOR_TYPE_TENSOR_ARM, 1, shaderStageFlags, nullptr};
        layoutBindings.emplace_back(binding);
    }

    FFX_ASSERT(layoutBindings.size() <= MAX_DESCRIPTOR_SET_LAYOUTS);

    // Create the descriptor layout
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount                    = static_cast<uint32_t>(layoutBindings.size());
    layoutInfo.pBindings                       = layoutBindings.data();

    if (VulkanWrapper().vkCreateDescriptorSetLayout(backendContext->device, &layoutInfo, nullptr, &pPipelineLayout->descriptorSetLayout) != VK_SUCCESS)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // allocate descriptor sets
    pPipelineLayout->descriptorSetIndex = 0;
    for (uint32_t i = 0; i < (FFX_MAX_QUEUED_FRAMES * MAX_PIPELINE_USAGE_PER_FRAME); i++)
    {
        VkDescriptorSetAllocateInfo allocateInfo = {};
        allocateInfo.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool              = backendContext->descriptorPool;
        allocateInfo.descriptorSetCount          = 1;
        allocateInfo.pSetLayouts                 = &pPipelineLayout->descriptorSetLayout;

        if (VulkanWrapper().vkAllocateDescriptorSets(backendContext->device, &allocateInfo, &pPipelineLayout->descriptorSets[i]) != VK_SUCCESS)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }
    }

    // create the pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount         = 1;
    pipelineLayoutInfo.pSetLayouts            = &pPipelineLayout->descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges    = nullptr;

    if (VulkanWrapper().vkCreatePipelineLayout(backendContext->device, &pipelineLayoutInfo, nullptr, &pPipelineLayout->pipelineLayout) != VK_SUCCESS)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    pPipelineLayout->effectContextId = effectContextId;

    // set the root signature to pipeline
    outPipeline->rootSignature = reinterpret_cast<FfxRootSignature>(pPipelineLayout);

    outPipeline->cmdSignature = nullptr;

    outPipeline->uavTensorCount = dataGraphBlob.outputTensorNums;
    FFX_ASSERT(outPipeline->uavTensorCount < FFX_MAX_NUM_TENSORS);
    for (uint32_t tensorIndex = 0; tensorIndex < dataGraphBlob.outputTensorNums; ++tensorIndex)
    {
        outPipeline->uavTensorBindings[tensorIndex].slotIndex  = dataGraphBlob.outputTensorBindings[tensorIndex];
        outPipeline->uavTensorBindings[tensorIndex].arrayIndex = 0;
        strncpy(outPipeline->uavTensorBindings[tensorIndex].name, dataGraphBlob.outputTensorNames[tensorIndex], FFX_RESOURCE_NAME_SIZE - 1);
        outPipeline->uavTensorBindings[tensorIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
    }

    outPipeline->srvTensorCount = dataGraphBlob.inputTensorNums;
    FFX_ASSERT(outPipeline->srvTensorCount < FFX_MAX_NUM_TENSORS);
    for (uint32_t tensorIndex = 0; tensorIndex < dataGraphBlob.inputTensorNums; ++tensorIndex)
    {
        outPipeline->srvTensorBindings[tensorIndex].slotIndex  = dataGraphBlob.inputTensorBindings[tensorIndex];
        outPipeline->srvTensorBindings[tensorIndex].arrayIndex = 0;
        strncpy(outPipeline->srvTensorBindings[tensorIndex].name, dataGraphBlob.inputTensorNames[tensorIndex], FFX_RESOURCE_NAME_SIZE - 1);
        outPipeline->srvTensorBindings[tensorIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
    }

    DescriptorSetBindingToShapeMap inputShapes = GetInputShapes(dataGraphBlob, dataGraphWidth, dataGraphHeight);

    ShapeInferenceResults ShapeInferenceResults =
        RunShapeInference(reinterpret_cast<const uint32_t*>(dataGraphBlob.graphData), dataGraphBlob.graphDataSize / 4, inputShapes);

    if (!ShapeInferenceResults.Success)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // shader module
    VkShaderModule           shaderModule           = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.pCode                    = ShapeInferenceResults.NewCode.data();
    shaderModuleCreateInfo.codeSize                 = ShapeInferenceResults.NewCode.size() * sizeof(ShapeInferenceResults.NewCode[0]);

    if (VulkanWrapper().vkCreateShaderModule(backendContext->device, &shaderModuleCreateInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    std::vector<VkTensorDescriptionARM>         pipelineTensorConstantDescs;
    std::vector<VkDataGraphPipelineConstantARM> pipelineConstants;

    pipelineTensorConstantDescs.resize(dataGraphBlob.constantNums);

    for (FfxUInt32 constantIndex = 0; constantIndex < dataGraphBlob.constantNums; ++constantIndex)
    {
        // VUID-VkDataGraphPipelineConstantARM-pNext-09917 requires the tiling format must be linear.
        VkTensorDescriptionARM tensorDescription = {VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
                                                    nullptr,
                                                    VK_TENSOR_TILING_LINEAR_ARM,
                                                    (VkFormat)dataGraphBlob.constantFormats[constantIndex],
                                                    dataGraphBlob.constantShapeSize[constantIndex],
                                                    dataGraphBlob.constantShapes[constantIndex],
                                                    nullptr,  // pStrides
                                                    VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM};

        pipelineTensorConstantDescs[constantIndex] = tensorDescription;

        VkDataGraphPipelineConstantARM pipelineConstant = {VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CONSTANT_ARM,
                                                           &pipelineTensorConstantDescs[constantIndex],
                                                           dataGraphBlob.constantIds[constantIndex],
                                                           dataGraphBlob.constantDatas[constantIndex]};

        pipelineConstants.push_back(pipelineConstant);
    }

    VkDataGraphPipelineShaderModuleCreateInfoARM dataGraphPipelineShaderModuleCreateInfo = {VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM,
                                                                                            nullptr,
                                                                                            shaderModule,
                                                                                            dataGraphBlob.graphEntryPoint,
                                                                                            nullptr,
                                                                                            dataGraphBlob.constantNums,
                                                                                            pipelineConstants.data()};

    std::vector<VkTensorDescriptionARM>             tensorDescs;
    std::vector<VkDataGraphPipelineResourceInfoARM> resourceInfos;

    tensorDescs.resize(dataGraphBlob.inputTensorNums + dataGraphBlob.outputTensorNums);
    const auto isTensorBufferAliased = [&](const char* tensorName) -> bool {
        if (desc->dataGraphTensorInfo == nullptr || desc->dataGraphTensorInfoCount == 0 || tensorName == nullptr)
        {
            return false;
        }

        for (FfxUInt32 tensorInfoIndex = 0; tensorInfoIndex < desc->dataGraphTensorInfoCount; ++tensorInfoIndex)
        {
            const FfxDataGraphTensorInfo& tensorInfo = desc->dataGraphTensorInfo[tensorInfoIndex];
            if (tensorInfo.resourceName != nullptr && 0 == strcmp(tensorInfo.resourceName, tensorName))
            {
                return tensorInfo.bufferAliased;
            }
        }

        return false;
    };

    constexpr uint32_t MESSAGE_BUFFER_SIZE = 256;
    char               message[MESSAGE_BUFFER_SIZE];
    for (FfxUInt32 tensorIndex = 0; tensorIndex < dataGraphBlob.inputTensorNums; ++tensorIndex)
    {
        std::pair<uint32_t, uint32_t> binding{0, dataGraphBlob.inputTensorBindings[tensorIndex]};  // [set, binding]
        const auto&                   tensorShape       = ShapeInferenceResults.Shapes[binding];
        const uint32_t                dimensionCount    = tensorShape.size();
        const bool                    bufferAliased     = isTensorBufferAliased(dataGraphBlob.inputTensorNames[tensorIndex]);
        VkTensorDescriptionARM        tensorDescription = {VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
                                                    nullptr,
                                                    GetTensorTiling(bufferAliased),
                                                    (VkFormat)dataGraphBlob.inputTensorFormats[tensorIndex],
                                                    static_cast<uint32_t>(dimensionCount),
                                                    tensorShape.data(),
                                                    nullptr,  // pStrides
                                                    VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM};

        tensorDescs[tensorIndex] = tensorDescription;

        VkDataGraphPipelineResourceInfoARM resourceInfo = {
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM,
            &tensorDescs[tensorIndex],
            binding.first,   // descriptorSet
            binding.second,  // binding
            0                // array element
        };

        resourceInfos.push_back(resourceInfo);

        snprintf(message,
                 MESSAGE_BUFFER_SIZE,
                 "Input tensor %u: set=%u, binding=%u, rank=%u, buffer_aliased=%u\n",
                 tensorIndex,
                 binding.first,
                 binding.second,
                 dimensionCount,
                 (uint32_t)bufferAliased);
        PrintMessage(backendInterface, FFX_MESSAGE_TYPE_WARNING, message);
        for (uint32_t i = 0; i < dimensionCount; ++i)
        {
            snprintf(message, MESSAGE_BUFFER_SIZE, "\tDimension[%u]: %u\n", i, tensorShape[i]);
            PrintMessage(backendInterface, FFX_MESSAGE_TYPE_WARNING, message);
        }
    }

    // output tensors infos
    for (FfxUInt32 tensorIndex = 0; tensorIndex < dataGraphBlob.outputTensorNums; ++tensorIndex)
    {
        std::pair<uint32_t, uint32_t> binding{0, dataGraphBlob.outputTensorBindings[tensorIndex]};  // [set, binding]
        const auto&                   tensorShape    = ShapeInferenceResults.Shapes[binding];
        const uint32_t                dimensionCount = tensorShape.size();
        const bool                    bufferAliased  = isTensorBufferAliased(dataGraphBlob.outputTensorNames[tensorIndex]);

        VkTensorDescriptionARM tensorDescription = {VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
                                                    nullptr,
                                                    GetTensorTiling(bufferAliased),
                                                    (VkFormat)dataGraphBlob.outputTensorFormats[tensorIndex],
                                                    static_cast<uint32_t>(dimensionCount),
                                                    tensorShape.data(),
                                                    nullptr,  // pStrides
                                                    VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM};

        tensorDescs[tensorIndex + dataGraphBlob.inputTensorNums] = tensorDescription;

        VkDataGraphPipelineResourceInfoARM resourceInfo = {
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM,
            &tensorDescs[tensorIndex + dataGraphBlob.inputTensorNums],
            binding.first,   // descriptorSet
            binding.second,  // binding
            0                // array element
        };

        resourceInfos.push_back(resourceInfo);

        snprintf(message,
                 MESSAGE_BUFFER_SIZE,
                 "Output tensor %u: set=%u, binding=%u, rank=%u, buffer_aliased=%u\n",
                 tensorIndex,
                 binding.first,
                 binding.second,
                 dimensionCount,
                 (uint32_t)bufferAliased);
        PrintMessage(backendInterface, FFX_MESSAGE_TYPE_WARNING, message);
        for (uint32_t i = 0; i < dimensionCount; ++i)
        {
            snprintf(message, MESSAGE_BUFFER_SIZE, "\tDimension[%u]: %lld\n", i, tensorShape[i]);
            PrintMessage(backendInterface, FFX_MESSAGE_TYPE_WARNING, message);
        }
    }

    // create the data graph pipeline
    VkDataGraphPipelineCreateInfoARM pipelineCreateInfo = {
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM,
        &dataGraphPipelineShaderModuleCreateInfo,
        0,                                            // flags
        pPipelineLayout->pipelineLayout,              // layout
        static_cast<uint32_t>(resourceInfos.size()),  // resourceInfoCount
        resourceInfos.data(),                         // pResourceInfos
    };

    VkPipeline dataGraphPipeline = VK_NULL_HANDLE;
    if (VulkanWrapper().vkCreateDataGraphPipelinesARM(
            backendContext->device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &dataGraphPipeline) != VK_SUCCESS)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // done with shader module, so clean up
    VulkanWrapper().vkDestroyShaderModule(backendContext->device, shaderModule, nullptr);

    // set the pipeline
    outPipeline->pipeline = reinterpret_cast<FfxPipeline>(dataGraphPipeline);

    // Setup the pipeline name
    strncpy(outPipeline->name, desc->name, FFX_RESOURCE_NAME_SIZE - 1);
    outPipeline->name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';

    const VkDataGraphPipelineSessionCreateInfoARM sessionCreateInfo = {
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM, nullptr, 0, dataGraphPipeline};

    VkDataGraphPipelineSessionARM session;
    if (VulkanWrapper().vkCreateDataGraphPipelineSessionARM(backendContext->device, &sessionCreateInfo, nullptr, &session))
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // Use cpp standard way (eg.reinterpret_cast) to store opaque vulkan handles in generic pointers
    outPipeline->session = reinterpret_cast<FfxDataGraphPipelineSession>(session);

    // Query the number of bind points to which memory must be bound
    uint32_t                                               numRequiredBindPoints = 0;
    VkDataGraphPipelineSessionBindPointRequirementsInfoARM bindPointReqsInfo     = {
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO_ARM, nullptr, session};

    if (VulkanWrapper().vkGetDataGraphPipelineSessionBindPointRequirementsARM(backendContext->device, &bindPointReqsInfo, &numRequiredBindPoints, nullptr))
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // Query the list of bind points to which memory must be bound
    std::vector<VkDataGraphPipelineSessionBindPointRequirementARM> bindPointRequirements(
        numRequiredBindPoints, VkDataGraphPipelineSessionBindPointRequirementARM{VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT_ARM});
    if (VulkanWrapper().vkGetDataGraphPipelineSessionBindPointRequirementsARM(
            backendContext->device, &bindPointReqsInfo, &numRequiredBindPoints, bindPointRequirements.data()))
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    FFX_ASSERT(numRequiredBindPoints >= 1);  // Transient bind point is required

    VkMemoryRequirements2 memreqs = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, nullptr, {}};
    for (uint32_t i = 0; i < numRequiredBindPoints; ++i)
    {
        if (bindPointRequirements[i].bindPoint == VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM)
        {
            FFX_ASSERT(bindPointRequirements[i].bindPointType == VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM);

            VkDataGraphPipelineSessionMemoryRequirementsInfoARM info = {
                VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_MEMORY_REQUIREMENTS_INFO_ARM, nullptr, session, bindPointRequirements[i].bindPoint, 0};

            VulkanWrapper().vkGetDataGraphPipelineSessionMemoryRequirementsARM(backendContext->device, &info, &memreqs);
            break;
        }
    }

    BackendContext_VK::Resource sessionMemoryResource = {};
    sessionMemoryResource.memoryProperties            = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if (allocateDeviceMemory(backendContext, memreqs.memoryRequirements, sessionMemoryResource.memoryProperties, &sessionMemoryResource))
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    pPipelineLayout->dataGraphSessionMemory     = sessionMemoryResource.deviceMemory;
    pPipelineLayout->dataGraphSessionMemorySize = memreqs.memoryRequirements.size;

    const VkBindDataGraphPipelineSessionMemoryInfoARM bindInfo = {
        VK_STRUCTURE_TYPE_BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO_ARM,
        nullptr,
        session,
        VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM,  // binding point
        0,                                                        // resource index
        pPipelineLayout->dataGraphSessionMemory,
        0  // memoryOffset
    };

    if (VulkanWrapper().vkBindDataGraphPipelineSessionMemoryARM(backendContext->device, 1, &bindInfo))
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    return FFX_OK;
}

FfxErrorCode CreateOpticalFlowPipelineVK(
    FfxInterface* backendInterface, const char* name, const FfxOpticalFlowDescription& ofDesc, FfxUInt32 effectContextId, FfxPipelineState* outPipeline)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != outPipeline);

    BackendContext_VK*                backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;
    BackendContext_VK::EffectContext& effectContext  = backendContext->pEffectContexts[effectContextId];

    //////////////////////////////////////////////////////////////////////////
    // One root signature (or pipeline layout) per pipeline
    FFX_ASSERT_MESSAGE(effectContext.nextPipelineLayout < (effectContextId * FFX_MAX_PASS_COUNT) + FFX_MAX_PASS_COUNT,
                       "FFXInterface: Vulkan: Ran out of pipeline layouts. Please increase FFX_MAX_PASS_COUNT");
    BackendContext_VK::PipelineLayout* pPipelineLayout = &backendContext->pPipelineLayouts[effectContext.nextPipelineLayout++];

    // Setup descriptor sets
    VkDescriptorSetLayoutBinding layoutBindings[MAX_DESCRIPTOR_SET_LAYOUTS];
    uint32_t                     numLayoutBindings = 0;

    VkShaderStageFlags shaderStageFlags = VK_SHADER_STAGE_ALL;

    const uint32_t srvBindingCount = ofDesc.srvTextureCount;
    const uint32_t uavBindingCount = ofDesc.uavTextureCount;

    const char**                  srvNames           = ofDesc.boundSRVTextureNames;
    uint32_t*                     srvBindings        = ofDesc.boundSRVTextures;
    uint32_t*                     srvBindCounts      = ofDesc.boundSRVTextureCounts;
    uint32_t*                     srvSets            = ofDesc.boundSRVTextureSpaces;
    FfxOpticalFlowConnectionType* srvConnectionTypes = ofDesc.boundSRVTextureConnectionType;
    FfxSurfaceFormat*             srvFormats         = ofDesc.boundSRVTextureFormats;

    const char**                  uavNames           = ofDesc.boundUAVTextureNames;
    uint32_t*                     uavBindings        = ofDesc.boundUAVTextures;
    uint32_t*                     uavBindCounts      = ofDesc.boundUAVTextureCounts;
    uint32_t*                     uavSets            = ofDesc.boundUAVTextureSpaces;
    FfxOpticalFlowConnectionType* uavConnectionTypes = ofDesc.boundUAVTextureConnectionType;
    FfxSurfaceFormat*             uavFormats         = ofDesc.boundUAVTextureFormats;

    // Texture SRVs
    for (uint32_t srvIndex = 0; srvIndex < srvBindingCount; ++srvIndex)
    {
        layoutBindings[numLayoutBindings++] = {srvBindings[srvIndex], VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, srvBindCounts[srvIndex], shaderStageFlags, nullptr};
    }

    // Texture UAVs
    for (uint32_t uavIndex = 0; uavIndex < uavBindingCount; ++uavIndex)
    {
        layoutBindings[numLayoutBindings++] = {uavBindings[uavIndex], VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, uavBindCounts[uavIndex], shaderStageFlags, nullptr};
    }

    // Create the descriptor layout
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount                    = numLayoutBindings;
    layoutInfo.pBindings                       = &layoutBindings[0];

    if (VulkanWrapper().vkCreateDescriptorSetLayout(backendContext->device, &layoutInfo, nullptr, &pPipelineLayout->descriptorSetLayout) != VK_SUCCESS)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // allocate descriptor sets
    pPipelineLayout->descriptorSetIndex = 0;
    for (uint32_t i = 0; i < (FFX_MAX_QUEUED_FRAMES * MAX_PIPELINE_USAGE_PER_FRAME); i++)
    {
        VkDescriptorSetAllocateInfo allocateInfo = {};
        allocateInfo.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool              = backendContext->descriptorPool;
        allocateInfo.descriptorSetCount          = 1;
        allocateInfo.pSetLayouts                 = &pPipelineLayout->descriptorSetLayout;

        // If you are getting a crash here, your bindings don't match the ones declared in the descriptor set.
        // Check your sampler descriptions and if they correctly match the ones in your shader.
        if (VulkanWrapper().vkAllocateDescriptorSets(backendContext->device, &allocateInfo, &pPipelineLayout->descriptorSets[i]) != VK_SUCCESS)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }
    }

    VkDescriptorSetLayout layouts[MAX_DESCRIPTOR_SET_LAYOUTS];
    uint32_t              layoutCount = 0;
    layouts[layoutCount++]            = pPipelineLayout->descriptorSetLayout;

    // TODO: do not consider bindless resources

    // create the pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount         = layoutCount;
    pipelineLayoutInfo.pSetLayouts            = layouts;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges    = nullptr;

    if (VulkanWrapper().vkCreatePipelineLayout(backendContext->device, &pipelineLayoutInfo, nullptr, &pPipelineLayout->pipelineLayout) != VK_SUCCESS)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // set the root signature to pipeline
    outPipeline->rootSignature = reinterpret_cast<FfxRootSignature>(pPipelineLayout);
    outPipeline->cmdSignature  = nullptr;

    // Create the optical flow connectivity map
    //
    // A node connection specifies the role of a binding (eg. search, reference, flow etc.)
    std::vector<VkDataGraphPipelineSingleNodeConnectionARM> opticalFlowConnections;
    // A resource info specifies the descriptor set / binding of a resource
    std::vector<VkDataGraphPipelineResourceInfoARM> resourceInfos;
    // An image resource info specifies image layout, attached to the corresponding resource info
    std::vector<VkDataGraphPipelineResourceInfoImageLayoutARM> resourceImageInfos;

    const uint32_t totalTextures = srvBindingCount + uavBindingCount;
    opticalFlowConnections.reserve(totalTextures);
    resourceInfos.reserve(totalTextures);
    resourceImageInfos.reserve(totalTextures);

    uint32_t flattenedSrvTextureCount = 0;

    for (uint32_t srvIndex = 0; srvIndex < srvBindingCount; ++srvIndex)
    {
        const uint32_t binding   = srvBindings[srvIndex];
        const uint32_t bindCount = srvBindCounts[srvIndex];
        const uint32_t set       = srvSets[srvIndex];

        // Note: for now we only support 1 descriptor set
        FFX_ASSERT(set == 0);

        // Note: we currently only support 1 resource per VkDataGraphPipelineResourceInfoARM, so make sure it is only 1.
        // We might support arrays in the future so please check the latest spec if you need this functionality.
        FFX_ASSERT(bindCount == 1);

        // Set up internal state so we know how to bind the textures in the pipeline later...

        uint32_t idx                                    = flattenedSrvTextureCount++;
        outPipeline->srvTextureBindings[idx].slotIndex  = binding;
        outPipeline->srvTextureBindings[idx].arrayIndex = 0;
        outPipeline->srvTextureBindings[idx].bindSet    = set;
        strncpy(outPipeline->srvTextureBindings[idx].name, srvNames[srvIndex], FFX_RESOURCE_NAME_SIZE - 1);
        outPipeline->srvTextureBindings[idx].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';

        // ... Then set up the optical flow connectivity map...

        opticalFlowConnections.emplace_back(VkDataGraphPipelineSingleNodeConnectionARM{
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SINGLE_NODE_CONNECTION_ARM,
            nullptr,
            set,
            binding,
            ffxGetVkOpticalFlowConnectionType(srvConnectionTypes[srvIndex]),  // connection type
        });

        // ... And finally the data graph resource infos

        resourceImageInfos.emplace_back(VkDataGraphPipelineResourceInfoImageLayoutARM{
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_IMAGE_LAYOUT_ARM,
            nullptr,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,  // srv
        });

        resourceInfos.emplace_back(VkDataGraphPipelineResourceInfoARM{
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM,
            nullptr,
            set,
            binding,
            0,  // arrayElement, we already know that bindCount == 1 so we hardcode 0 here
        });
    }

    outPipeline->srvTextureCount = flattenedSrvTextureCount;
    FFX_ASSERT(outPipeline->srvTextureCount < FFX_MAX_NUM_SRVS);

    uint32_t flattenedUavTextureCount = 0;

    for (uint32_t uavIndex = 0; uavIndex < uavBindingCount; ++uavIndex)
    {
        const uint32_t binding   = uavBindings[uavIndex];
        const uint32_t bindCount = uavBindCounts[uavIndex];
        const uint32_t set       = uavSets[uavIndex];
        // Note: for now we only support 1 descriptor set
        FFX_ASSERT(set == 0);

        // Note: we currently only support 1 resource per VkDataGraphPipelineResourceInfoARM, so make sure it is only 1.
        // We might support arrays in the future so please check the latest spec if you need this functionality.
        FFX_ASSERT(bindCount == 1);

        // Set up internal state so we know how to bind the textures in the pipeline later...

        uint32_t idx                                    = flattenedUavTextureCount++;
        outPipeline->uavTextureBindings[idx].slotIndex  = binding;
        outPipeline->uavTextureBindings[idx].arrayIndex = 0;
        outPipeline->uavTextureBindings[idx].bindSet    = set;
        strncpy(outPipeline->uavTextureBindings[idx].name, uavNames[uavIndex], FFX_RESOURCE_NAME_SIZE - 1);
        outPipeline->uavTextureBindings[idx].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';

        // ... Then set up the optical flow connectivity map...

        opticalFlowConnections.emplace_back(VkDataGraphPipelineSingleNodeConnectionARM{
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SINGLE_NODE_CONNECTION_ARM,
            nullptr,
            set,
            binding,
            ffxGetVkOpticalFlowConnectionType(uavConnectionTypes[uavIndex]),  // connection type
        });

        // ... And finally the data graph resource infos

        resourceImageInfos.emplace_back(VkDataGraphPipelineResourceInfoImageLayoutARM{
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_IMAGE_LAYOUT_ARM,
            nullptr,
            VK_IMAGE_LAYOUT_GENERAL,  // uav
        });

        resourceInfos.emplace_back(VkDataGraphPipelineResourceInfoARM{
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM,
            nullptr,
            set,
            binding,
            0,  // arrayElement, we already know that bindCount == 1 so we hardcode 0 here
        });
    }

    // Pushing back into resourceImageInfo even if it has been .reserve()'d results in a reallocation, therefore
    // the pointers get invalidated...
    // TODO: There should be a way to guarantee we are not ever reallocating and the .pNext values stay valid,
    // so we don't need to do these second loops.
    for (uint32_t srvIndex = 0; srvIndex < srvBindingCount; ++srvIndex)
    {
        resourceInfos[srvIndex].pNext = &resourceImageInfos[srvIndex];
    }

    for (uint32_t uavIndex = 0; uavIndex < uavBindingCount; ++uavIndex)
    {
        resourceInfos[uavIndex + flattenedSrvTextureCount].pNext = &resourceImageInfos[uavIndex + flattenedSrvTextureCount];
    }

    outPipeline->uavTextureCount = flattenedUavTextureCount;
    FFX_ASSERT(outPipeline->uavTextureCount < FFX_MAX_NUM_UAVS);
    outPipeline->srvTensorCount = 0;
    outPipeline->uavTensorCount = 0;

    pPipelineLayout->effectContextId = effectContextId;

    VkDataGraphPipelineSingleNodeCreateInfoARM connectivityMap{};
    connectivityMap.sType           = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SINGLE_NODE_CREATE_INFO_ARM;
    connectivityMap.pNext           = nullptr;
    connectivityMap.nodeType        = VK_DATA_GRAPH_PIPELINE_NODE_TYPE_OPTICAL_FLOW_ARM;
    connectivityMap.connectionCount = static_cast<uint32_t>(opticalFlowConnections.size());
    connectivityMap.pConnections    = opticalFlowConnections.data();

    // Optical flow configuration
    VkDataGraphOpticalFlowCreateFlagsARM ofFlags = 0;
    VkFormat                             image_format{VK_FORMAT_UNDEFINED};
    // cost map is optional, so default to valid format incase no cost map is specified
    VkFormat cost_format{VK_FORMAT_R16_UINT};
    VkFormat flow_format{VK_FORMAT_UNDEFINED};

    for (uint32_t i = 0; i < outPipeline->srvTextureCount; i++)
    {
        switch (srvConnectionTypes[i])
        {
        case FFX_OPTICAL_FLOW_CONNECTION_INPUT:
        {
            image_format = ffxGetVkFormatFromSurfaceFormat(srvFormats[i]);
            break;
        }
        case FFX_OPTICAL_FLOW_CONNECTION_HINT:
            ofFlags |= VK_DATA_GRAPH_OPTICAL_FLOW_CREATE_ENABLE_HINT_BIT_ARM;
            break;
        default:
            break;
        }
    }

    for (uint32_t i = 0; i < outPipeline->uavTextureCount; i++)
    {
        switch (uavConnectionTypes[i])
        {
        case FFX_OPTICAL_FLOW_CONNECTION_COST:
            cost_format = ffxGetVkFormatFromSurfaceFormat(uavFormats[i]);
            ofFlags |= VK_DATA_GRAPH_OPTICAL_FLOW_CREATE_ENABLE_COST_BIT_ARM;
            break;
        case FFX_OPTICAL_FLOW_CONNECTION_FLOW_VECTOR:
            flow_format = ffxGetVkFormatFromSurfaceFormat(uavFormats[i]);
            break;
        case FFX_OPTICAL_FLOW_CONNECTION_BACKWARD_FLOW_VECTOR:
        case FFX_OPTICAL_FLOW_CONNECTION_BACKWARD_COST:
            FFX_ASSERT_MESSAGE(false, "Backward flow/cost are no longer supported by VK_ARM_data_graph_optical_flow");
            break;
        default:
            break;
        }
    }

    VkDataGraphPipelineOpticalFlowCreateInfoARM ofCreateInfo{};
    ofCreateInfo.sType            = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_OPTICAL_FLOW_CREATE_INFO_ARM;
    ofCreateInfo.pNext            = &connectivityMap;
    ofCreateInfo.width            = ofDesc.dimensions.width;
    ofCreateInfo.height           = ofDesc.dimensions.height;
    ofCreateInfo.imageFormat      = image_format;
    ofCreateInfo.flowVectorFormat = flow_format;
    ofCreateInfo.costFormat       = cost_format;
    ofCreateInfo.outputGridSize   = ffxGetVkOpticalFlowGridSize(ofDesc.gridSize);
    ofCreateInfo.hintGridSize     = ffxGetVkOpticalFlowGridSize(ofDesc.gridSize);
    ofCreateInfo.performanceLevel = ffxGetVkOpticalFlowPerformanceLevel(ofDesc.performanceLevel);
    ofCreateInfo.flags            = ofFlags;

    // create the data graph pipeline
    VkDataGraphPipelineCreateInfoARM pipelineCreateInfo = {
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM,
        &ofCreateInfo,
        0,                                            // flags
        pPipelineLayout->pipelineLayout,              // layout
        static_cast<uint32_t>(resourceInfos.size()),  // resourceInfoCount
        resourceInfos.data(),                         // pResourceInfos
    };

    VkPipeline dataGraphPipeline = VK_NULL_HANDLE;
    if (VulkanWrapper().vkCreateDataGraphPipelinesARM(
            backendContext->device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &dataGraphPipeline) != VK_SUCCESS)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // set the pipeline
    outPipeline->pipeline = reinterpret_cast<FfxPipeline>(dataGraphPipeline);

    // Setup the pipeline name
    strncpy(outPipeline->name, name, FFX_RESOURCE_NAME_SIZE - 1);
    outPipeline->name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';

    const VkDataGraphPipelineSessionCreateInfoARM sessionCreateInfo = {VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM,
                                                                       nullptr,
                                                                       VK_DATA_GRAPH_PIPELINE_SESSION_CREATE_OPTICAL_FLOW_CACHE_BIT_ARM,
                                                                       dataGraphPipeline};

    VkDataGraphPipelineSessionARM session;
    if (VulkanWrapper().vkCreateDataGraphPipelineSessionARM(backendContext->device, &sessionCreateInfo, nullptr, &session))
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    outPipeline->session = reinterpret_cast<FfxDataGraphPipelineSession>(session);

    uint32_t                                               numRequiredBindPoints = 0;
    VkDataGraphPipelineSessionBindPointRequirementsInfoARM bindPointReqsInfo     = {
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO_ARM, nullptr, session};

    if (VulkanWrapper().vkGetDataGraphPipelineSessionBindPointRequirementsARM(backendContext->device, &bindPointReqsInfo, &numRequiredBindPoints, nullptr))
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    std::vector<VkDataGraphPipelineSessionBindPointRequirementARM> bindPointRequirements(
        numRequiredBindPoints, VkDataGraphPipelineSessionBindPointRequirementARM{VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT_ARM});
    if (VulkanWrapper().vkGetDataGraphPipelineSessionBindPointRequirementsARM(
            backendContext->device, &bindPointReqsInfo, &numRequiredBindPoints, bindPointRequirements.data()))
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    std::vector<SessionMemoryObjectRequirement> memoryObjectRequirements{};
    memoryObjectRequirements.reserve(numRequiredBindPoints);

    VkMemoryRequirements unionMemReq{};
    unionMemReq.size           = 0;
    unionMemReq.alignment      = 1;
    unionMemReq.memoryTypeBits = 0xFFFFFFFFu;
    bool hasMemoryBindPoint    = false;

    for (uint32_t i = 0; i < numRequiredBindPoints; ++i)
    {
        const auto& req = bindPointRequirements[i];

        if (req.bindPointType != VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM)
            continue;

        for (uint32_t objectIndex = 0; objectIndex < req.numObjects; ++objectIndex)
        {
            VkDataGraphPipelineSessionMemoryRequirementsInfoARM memReqInfo = {
                VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_MEMORY_REQUIREMENTS_INFO_ARM, nullptr, session, req.bindPoint, objectIndex};
            VkMemoryRequirements2 memreqs = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, nullptr, {}};

            VulkanWrapper().vkGetDataGraphPipelineSessionMemoryRequirementsARM(backendContext->device, &memReqInfo, &memreqs);

            hasMemoryBindPoint    = true;
            unionMemReq.size      = (std::max)(unionMemReq.size, memreqs.memoryRequirements.size);
            unionMemReq.alignment = (std::max)(unionMemReq.alignment, memreqs.memoryRequirements.alignment);
            unionMemReq.memoryTypeBits &= memreqs.memoryRequirements.memoryTypeBits;
            memoryObjectRequirements.push_back(SessionMemoryObjectRequirement{req.bindPoint, objectIndex, memreqs.memoryRequirements, 0});
        }
    }

    if (!hasMemoryBindPoint)
    {
        FFX_ASSERT_MESSAGE(false, "Optical flow pipeline session has no memory bind points");
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    if (unionMemReq.memoryTypeBits == 0)
    {
        FFX_ASSERT_MESSAGE(false, "Optical flow pipeline session memory bind points have incompatible memoryTypeBits");
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    auto alignUp = [](VkDeviceSize value, VkDeviceSize alignment) -> VkDeviceSize {
        if (alignment <= 1)
        {
            return value;
        }
        const VkDeviceSize remainder = value % alignment;
        return remainder == 0 ? value : (value + (alignment - remainder));
    };

    VkDeviceSize totalSessionMemorySize = 0;
    VkDeviceSize maxSessionAlignment    = 1;
    for (auto& req : memoryObjectRequirements)
    {
        totalSessionMemorySize = alignUp(totalSessionMemorySize, req.memoryRequirements.alignment);
        req.offset             = totalSessionMemorySize;
        totalSessionMemorySize += req.memoryRequirements.size;
        maxSessionAlignment = maxSessionAlignment > req.memoryRequirements.alignment ? maxSessionAlignment : req.memoryRequirements.alignment;
    }

    unionMemReq.size      = totalSessionMemorySize;
    unionMemReq.alignment = maxSessionAlignment;

    BackendContext_VK::Resource sessionMemoryResource = {};
    sessionMemoryResource.memoryProperties            = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (allocateDeviceMemory(backendContext, unionMemReq, sessionMemoryResource.memoryProperties, &sessionMemoryResource))
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    pPipelineLayout->dataGraphSessionMemory     = sessionMemoryResource.deviceMemory;
    pPipelineLayout->dataGraphSessionMemorySize = unionMemReq.size;

    for (const auto& req : memoryObjectRequirements)
    {
        const VkBindDataGraphPipelineSessionMemoryInfoARM bindInfo = {VK_STRUCTURE_TYPE_BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO_ARM,
                                                                      nullptr,
                                                                      session,
                                                                      req.bindPoint,
                                                                      req.objectIndex,
                                                                      pPipelineLayout->dataGraphSessionMemory,
                                                                      req.offset};
        if (VulkanWrapper().vkBindDataGraphPipelineSessionMemoryARM(backendContext->device, 1, &bindInfo))
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }
    }

    return FFX_OK;
}

FfxErrorCode DestroyPipelineVK(FfxInterface* backendInterface, FfxPipelineState* pipeline, FfxUInt32 effectContextId)
{
    FFX_ASSERT(backendInterface != nullptr);
    BackendContext_VK*                backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;
    BackendContext_VK::EffectContext& effectContext  = backendContext->pEffectContexts[effectContextId];

    if (!pipeline)
        return FFX_OK;

    // DATA GRAPH: `FfxPipelineState::session` is the Data Graph session handle (see FfxPipelineState in ffx_types.h).
    // Evidence in this backend:
    // - Set in CreateDataGraphPipelineVK via vkCreateDataGraphPipelineSessionARM.
    // - Used at dispatch time: vkCmdDispatchDataGraphARM(..., (VkDataGraphPipelineSessionARM)pipeline.session, ...).
    // Therefore it must be explicitly destroyed; otherwise internal allocations can survive until vkDestroyDevice.
    if (pipeline->session != VK_NULL_HANDLE)
    {
        VkDataGraphPipelineSessionARM session = reinterpret_cast<VkDataGraphPipelineSessionARM>(pipeline->session);
        VulkanWrapper().vkDestroyDataGraphPipelineSessionARM(backendContext->device, session, nullptr);
        pipeline->session = VK_NULL_HANDLE;
    }

    // Destroy the pipeline
    VkPipeline vkPipeline = reinterpret_cast<VkPipeline>(pipeline->pipeline);
    if (vkPipeline != VK_NULL_HANDLE)
    {
        VulkanWrapper().vkDestroyPipeline(backendContext->device, vkPipeline, VK_NULL_HANDLE);
        pipeline->pipeline = VK_NULL_HANDLE;
    }

    // Zero out the cmd signature
    pipeline->cmdSignature = nullptr;

    // Destroy the pipeline layout
    BackendContext_VK::PipelineLayout* pPipelineLayout = reinterpret_cast<BackendContext_VK::PipelineLayout*>(pipeline->rootSignature);

    if (pPipelineLayout)
    {
        if (pPipelineLayout->dataGraphSessionMemory != VK_NULL_HANDLE)
        {
            VulkanWrapper().vkFreeMemory(backendContext->device, pPipelineLayout->dataGraphSessionMemory, nullptr);
            pPipelineLayout->dataGraphSessionMemory     = VK_NULL_HANDLE;
            pPipelineLayout->dataGraphSessionMemorySize = 0;
        }

        // Descriptor set layout
        if (pPipelineLayout->pipelineLayout != VK_NULL_HANDLE)
        {
            VulkanWrapper().vkDestroyPipelineLayout(backendContext->device, pPipelineLayout->pipelineLayout, VK_NULL_HANDLE);
            pPipelineLayout->pipelineLayout = VK_NULL_HANDLE;
        }

        // Descriptor sets
        for (uint32_t i = 0; i < FFX_MAX_QUEUED_FRAMES * MAX_PIPELINE_USAGE_PER_FRAME; i++)
        {
            VulkanWrapper().vkFreeDescriptorSets(backendContext->device, backendContext->descriptorPool, 1, &pPipelineLayout->descriptorSets[i]);
            pPipelineLayout->descriptorSets[i] = VK_NULL_HANDLE;
        }

        // Descriptor set layout
        if (pPipelineLayout->descriptorSetLayout != VK_NULL_HANDLE)
        {
            VulkanWrapper().vkDestroyDescriptorSetLayout(backendContext->device, pPipelineLayout->descriptorSetLayout, VK_NULL_HANDLE);
            pPipelineLayout->descriptorSetLayout = VK_NULL_HANDLE;
        }

        // Samplers
        for (uint32_t currentSamplerIndex = 0; currentSamplerIndex < FFX_MAX_SAMPLERS; ++currentSamplerIndex)
        {
            if (pPipelineLayout->samplers[currentSamplerIndex] != VK_NULL_HANDLE)
            {
                VulkanWrapper().vkDestroySampler(backendContext->device, pPipelineLayout->samplers[currentSamplerIndex], VK_NULL_HANDLE);
                pPipelineLayout->samplers[currentSamplerIndex] = VK_NULL_HANDLE;
            }
        }

        for (FfxUInt32 i = 0; i < MAX_IMAGE_VIEW_COUNT; i++)
        {
            if (pPipelineLayout->imageView[i].handle != VK_NULL_HANDLE)
            {
                VulkanWrapper().vkDestroyImageView(backendContext->device, pPipelineLayout->imageView[i].handle, VK_NULL_HANDLE);
            }
        }

        for (FfxUInt32 i = 0; i < MAX_FRAME_BUFFER_COUNT; i++)
        {
            if (pPipelineLayout->frameBuffer[i].handle != VK_NULL_HANDLE)
            {
                VulkanWrapper().vkDestroyFramebuffer(backendContext->device, pPipelineLayout->frameBuffer[i].handle, VK_NULL_HANDLE);
            }
        }

        for (FfxUInt32 i = 0; i < MAX_RENDER_PASS_COUNT; i++)
        {
            if (pPipelineLayout->renderPass[i].handle != VK_NULL_HANDLE)
            {
                VulkanWrapper().vkDestroyRenderPass(backendContext->device, pPipelineLayout->renderPass[i].handle, VK_NULL_HANDLE);
            }
        }

        for (FfxUInt32 i = 0; i < MAX_GRAPHICS_PIPELINE_COUNT; i++)
        {
            if (pPipelineLayout->graphicsPipeline[i].handle != VK_NULL_HANDLE)
            {
                VulkanWrapper().vkDestroyPipeline(backendContext->device, pPipelineLayout->graphicsPipeline[i].handle, VK_NULL_HANDLE);
            }
        }

        if (pPipelineLayout->fragShaderModule != VK_NULL_HANDLE)
        {
            VulkanWrapper().vkDestroyShaderModule(backendContext->device, pPipelineLayout->fragShaderModule, nullptr);
        }

        if (pPipelineLayout->vertShaderModule != VK_NULL_HANDLE)
        {
            VulkanWrapper().vkDestroyShaderModule(backendContext->device, pPipelineLayout->vertShaderModule, nullptr);
        }
    }

    return FFX_OK;
}

FfxErrorCode ScheduleGpuJobVK(FfxInterface* backendInterface, const FfxGpuJobDescription* job)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != job);

    BackendContext_VK* backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;

    FFX_ASSERT(backendContext->gpuJobCount < FFX_MAX_GPU_JOBS);

    backendContext->pGpuJobs[backendContext->gpuJobCount] = *job;
    backendContext->gpuJobCount++;

    return FFX_OK;
}

static FfxErrorCode executeGpuJobCompute(BackendContext_VK*    backendContext,
                                         FfxGpuJobDescription* job,
                                         VkCommandBuffer       vkCommandBuffer,
                                         FfxUInt32             effectContextId)
{
    BackendContext_VK::PipelineLayout* pipelineLayout = reinterpret_cast<BackendContext_VK::PipelineLayout*>(job->computeJobDescriptor.pipeline.rootSignature);

    // bind texture & buffer UAVs (note the binding order here MUST match the root signature mapping order from CreatePipeline!)
    uint32_t             descriptorWriteIndex = 0;
    VkWriteDescriptorSet writeDescriptorSets[FFX_MAX_RESOURCE_COUNT];

    // These MUST be initialized
    uint32_t              imageDescriptorIndex = 0;
    VkDescriptorImageInfo imageDescriptorInfos[FFX_MAX_RESOURCE_COUNT];
    for (int i = 0; i < FFX_MAX_RESOURCE_COUNT; ++i)
        imageDescriptorInfos[i] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    // These MUST be initialized
    uint32_t               bufferDescriptorIndex = 0;
    VkDescriptorBufferInfo bufferDescriptorInfos[FFX_MAX_RESOURCE_COUNT];
    for (int i = 0; i < FFX_MAX_RESOURCE_COUNT; ++i)
        bufferDescriptorInfos[i] = {VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};

    // bind texture UAVs
    for (uint32_t currentPipelineUavIndex = 0; currentPipelineUavIndex < job->computeJobDescriptor.pipeline.uavTextureCount; ++currentPipelineUavIndex)
    {
        FfxTextureUAV& textureUAV = job->computeJobDescriptor.uavTextures[currentPipelineUavIndex];

        // continue if this is a null resource.
        if (job->computeJobDescriptor.uavTextures[currentPipelineUavIndex].resource.internalIndex == 0)
            continue;

        addBarrier(backendContext, &textureUAV.resource, FFX_RESOURCE_STATE_COMPUTE_UAV);

        const FfxResourceBinding binding = job->computeJobDescriptor.pipeline.uavTextureBindings[currentPipelineUavIndex];

        // where to bind it
        const uint32_t currentUavResourceIndex = job->computeJobDescriptor.pipeline.uavTextureBindings[currentPipelineUavIndex].slotIndex;

        // source: UAV of resource to bind
        const uint32_t resourceIndex = textureUAV.resource.internalIndex;
        uint32_t       mipOffset     = textureUAV.mip;
        if (textureUAV.mip >= backendContext->pResources[resourceIndex].resourceDescription.mipCount)
            mipOffset = backendContext->pResources[resourceIndex].resourceDescription.mipCount - 1;
        const uint32_t uavViewIndex = backendContext->pResources[resourceIndex].uavViewIndex + mipOffset;

        writeDescriptorSets[descriptorWriteIndex]                 = {};
        writeDescriptorSets[descriptorWriteIndex].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[descriptorWriteIndex].dstSet          = pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex];
        writeDescriptorSets[descriptorWriteIndex].descriptorCount = 1;
        writeDescriptorSets[descriptorWriteIndex].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writeDescriptorSets[descriptorWriteIndex].pImageInfo      = &imageDescriptorInfos[imageDescriptorIndex];
        writeDescriptorSets[descriptorWriteIndex].dstBinding      = binding.slotIndex;
        writeDescriptorSets[descriptorWriteIndex].dstArrayElement = binding.arrayIndex;

        imageDescriptorInfos[imageDescriptorIndex]             = {};
        imageDescriptorInfos[imageDescriptorIndex].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageDescriptorInfos[imageDescriptorIndex].imageView   = backendContext->pResourceViews[uavViewIndex].imageView;

        imageDescriptorIndex++;
        descriptorWriteIndex++;
    }

    // bind buffer UAVs
    for (uint32_t currentPipelineUavIndex = 0; currentPipelineUavIndex < job->computeJobDescriptor.pipeline.uavBufferCount; ++currentPipelineUavIndex)
    {
        FfxBufferUAV& bufferUAV = job->computeJobDescriptor.uavBuffers[currentPipelineUavIndex];

        // continue if this is a null resource.
        if (job->computeJobDescriptor.uavBuffers[currentPipelineUavIndex].resource.internalIndex == 0)
            continue;

        addBarrier(backendContext, &bufferUAV.resource, FFX_RESOURCE_STATE_COMPUTE_UAV);

        const FfxResourceBinding binding = job->computeJobDescriptor.pipeline.uavBufferBindings[currentPipelineUavIndex];

        // source: UAV of buffer to bind
        const uint32_t resourceIndex = bufferUAV.resource.internalIndex;

        // where to bind it
        const uint32_t currentUavResourceIndex = job->computeJobDescriptor.pipeline.uavBufferBindings[currentPipelineUavIndex].slotIndex;

        writeDescriptorSets[descriptorWriteIndex]                 = {};
        writeDescriptorSets[descriptorWriteIndex].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[descriptorWriteIndex].dstSet          = pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex];
        writeDescriptorSets[descriptorWriteIndex].descriptorCount = 1;
        writeDescriptorSets[descriptorWriteIndex].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSets[descriptorWriteIndex].pBufferInfo     = &bufferDescriptorInfos[bufferDescriptorIndex];
        writeDescriptorSets[descriptorWriteIndex].dstBinding      = binding.slotIndex;
        writeDescriptorSets[descriptorWriteIndex].dstArrayElement = binding.arrayIndex;

        bufferDescriptorInfos[bufferDescriptorIndex] = {};

        const auto& backendResource = backendContext->pResources[resourceIndex];
        const bool  isTensor        = (backendResource.resourceDescription.type == FFX_RESOURCE_TYPE_TENSOR);
        const bool  isBufferAliased = ((backendResource.resourceDescription.flags & FFX_RESOURCE_FLAGS_BUFFER_ALIASED) == FFX_RESOURCE_FLAGS_BUFFER_ALIASED);
        if (isBufferAliased)
        {
            bufferDescriptorInfos[bufferDescriptorIndex].buffer =
                isTensor ? backendResource.aliasedTensorBufferResource : backendResource.aliasedBufferResource;
            bufferDescriptorInfos[bufferDescriptorIndex].range = VK_WHOLE_SIZE;
        }
        else
        {
            bufferDescriptorInfos[bufferDescriptorIndex].buffer = backendResource.bufferResource;
            bufferDescriptorInfos[bufferDescriptorIndex].range  = bufferUAV.size > 0 ? bufferUAV.size : backendResource.resourceDescription.size;
        }
        bufferDescriptorInfos[bufferDescriptorIndex].offset = bufferUAV.offset;

        bufferDescriptorIndex++;
        descriptorWriteIndex++;
    }

    // bind texture SRVs
    for (uint32_t currentPipelineSrvIndex = 0; currentPipelineSrvIndex < job->computeJobDescriptor.pipeline.srvTextureCount; ++currentPipelineSrvIndex)
    {
        FfxTextureSRV& textureSRV = job->computeJobDescriptor.srvTextures[currentPipelineSrvIndex];

        // continue if this is a null resource.
        if (job->computeJobDescriptor.srvTextures[currentPipelineSrvIndex].resource.internalIndex == 0)
            continue;

        addBarrier(backendContext, &textureSRV.resource, FFX_RESOURCE_STATE_COMPUTE_READ);

        const FfxResourceBinding binding = job->computeJobDescriptor.pipeline.srvTextureBindings[currentPipelineSrvIndex];

        // where to bind it
        const uint32_t currentSrvResourceIndex = job->computeJobDescriptor.pipeline.srvTextureBindings[currentPipelineSrvIndex].slotIndex;

        writeDescriptorSets[descriptorWriteIndex]                 = {};
        writeDescriptorSets[descriptorWriteIndex].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[descriptorWriteIndex].dstSet          = pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex];
        writeDescriptorSets[descriptorWriteIndex].descriptorCount = 1;
        writeDescriptorSets[descriptorWriteIndex].descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writeDescriptorSets[descriptorWriteIndex].pImageInfo      = &imageDescriptorInfos[imageDescriptorIndex];
        writeDescriptorSets[descriptorWriteIndex].dstBinding      = binding.slotIndex;
        writeDescriptorSets[descriptorWriteIndex].dstArrayElement = binding.arrayIndex;

        const uint32_t resourceIndex = textureSRV.resource.internalIndex;
        const uint32_t srvViewIndex  = backendContext->pResources[resourceIndex].srvViewIndex;

        imageDescriptorInfos[imageDescriptorIndex]             = {};
        imageDescriptorInfos[imageDescriptorIndex].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageDescriptorInfos[imageDescriptorIndex].imageView   = backendContext->pResourceViews[srvViewIndex].imageView;

        imageDescriptorIndex++;
        descriptorWriteIndex++;
    }

    // bind buffer SRVs
    for (uint32_t currentPipelineSrvIndex = 0; currentPipelineSrvIndex < job->computeJobDescriptor.pipeline.srvBufferCount; ++currentPipelineSrvIndex)
    {
        FfxBufferSRV& bufferSRV = job->computeJobDescriptor.srvBuffers[currentPipelineSrvIndex];

        // continue if this is a null resource.
        if (job->computeJobDescriptor.srvBuffers[currentPipelineSrvIndex].resource.internalIndex == 0)
            continue;

        addBarrier(backendContext, &bufferSRV.resource, FFX_RESOURCE_STATE_COMPUTE_READ);

        const FfxResourceBinding binding = job->computeJobDescriptor.pipeline.srvBufferBindings[currentPipelineSrvIndex];

        // source: SRV of buffer to bind
        const uint32_t resourceIndex = bufferSRV.resource.internalIndex;

        // where to bind it
        const uint32_t currentSrvResourceIndex = job->computeJobDescriptor.pipeline.srvBufferBindings[currentPipelineSrvIndex].slotIndex;

        writeDescriptorSets[descriptorWriteIndex]                 = {};
        writeDescriptorSets[descriptorWriteIndex].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[descriptorWriteIndex].dstSet          = pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex];
        writeDescriptorSets[descriptorWriteIndex].descriptorCount = 1;
        writeDescriptorSets[descriptorWriteIndex].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSets[descriptorWriteIndex].pBufferInfo     = &bufferDescriptorInfos[bufferDescriptorIndex];
        writeDescriptorSets[descriptorWriteIndex].dstBinding      = binding.slotIndex;
        writeDescriptorSets[descriptorWriteIndex].dstArrayElement = binding.arrayIndex;

        bufferDescriptorInfos[bufferDescriptorIndex] = {};

        const auto& backendResource = backendContext->pResources[resourceIndex];
        const bool  isTensor        = (backendResource.resourceDescription.type == FFX_RESOURCE_TYPE_TENSOR);
        const bool  isBufferAliased = ((backendResource.resourceDescription.flags & FFX_RESOURCE_FLAGS_BUFFER_ALIASED) == FFX_RESOURCE_FLAGS_BUFFER_ALIASED);

        if (isBufferAliased)
        {
            bufferDescriptorInfos[bufferDescriptorIndex].buffer =
                isTensor ? backendResource.aliasedTensorBufferResource : backendResource.aliasedBufferResource;
            bufferDescriptorInfos[bufferDescriptorIndex].range = VK_WHOLE_SIZE;
        }
        else
        {
            bufferDescriptorInfos[bufferDescriptorIndex].buffer = backendResource.bufferResource;
            bufferDescriptorInfos[bufferDescriptorIndex].range  = bufferSRV.size > 0 ? bufferSRV.size : backendResource.resourceDescription.size;
        }
        bufferDescriptorInfos[bufferDescriptorIndex].offset = bufferSRV.offset;

        bufferDescriptorIndex++;
        descriptorWriteIndex++;
    }

    uint32_t                      tensorDescriptorIndex = 0;
    VkWriteDescriptorSetTensorARM tensorDescriptorInfos[FFX_MAX_RESOURCE_COUNT];
    for (int i = 0; i < FFX_MAX_RESOURCE_COUNT; ++i)
        tensorDescriptorInfos[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM, nullptr};

    // bind tensor UAVs
    for (uint32_t currentPipelineUavIndex = 0; currentPipelineUavIndex < job->computeJobDescriptor.pipeline.uavTensorCount; ++currentPipelineUavIndex)
    {
        FfxTensor& tensor = job->computeJobDescriptor.uavTensors[currentPipelineUavIndex];

        // continue if this is a null resource.
        if (tensor.resource.internalIndex == 0)
            continue;

        addBarrier(backendContext, &tensor.resource, FFX_RESOURCE_STATE_COMPUTE_UAV);

        const FfxResourceBinding binding = job->computeJobDescriptor.pipeline.uavTensorBindings[currentPipelineUavIndex];

        const uint32_t resourceIndex   = job->computeJobDescriptor.uavTensors[currentPipelineUavIndex].resource.internalIndex;
        const uint32_t tensorViewIndex = backendContext->pResources[resourceIndex].tensorViewIndex;

        writeDescriptorSets[descriptorWriteIndex]                 = {};
        writeDescriptorSets[descriptorWriteIndex].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[descriptorWriteIndex].pNext           = &tensorDescriptorInfos[tensorDescriptorIndex];
        writeDescriptorSets[descriptorWriteIndex].dstSet          = pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex];
        writeDescriptorSets[descriptorWriteIndex].descriptorCount = 1;
        writeDescriptorSets[descriptorWriteIndex].descriptorType  = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
        writeDescriptorSets[descriptorWriteIndex].dstBinding      = binding.slotIndex;
        writeDescriptorSets[descriptorWriteIndex].dstArrayElement = binding.arrayIndex;

        tensorDescriptorInfos[tensorDescriptorIndex].tensorViewCount = 1;
        tensorDescriptorInfos[tensorDescriptorIndex].pTensorViews    = &backendContext->pResourceViews[tensorViewIndex].tensorView;

        tensorDescriptorIndex++;
        descriptorWriteIndex++;
    }

    // bind tensor SRVs
    for (uint32_t currentPipelineSrvIndex = 0; currentPipelineSrvIndex < job->computeJobDescriptor.pipeline.srvTensorCount; ++currentPipelineSrvIndex)
    {
        FfxTensor& tensor = job->computeJobDescriptor.srvTensors[currentPipelineSrvIndex];

        // continue if this is a null resource.
        if (tensor.resource.internalIndex == 0)
            continue;

        addBarrier(backendContext, &tensor.resource, FFX_RESOURCE_STATE_COMPUTE_READ);

        const FfxResourceBinding binding = job->computeJobDescriptor.pipeline.srvTensorBindings[currentPipelineSrvIndex];

        const uint32_t resourceIndex   = job->computeJobDescriptor.srvTensors[currentPipelineSrvIndex].resource.internalIndex;
        const uint32_t tensorViewIndex = backendContext->pResources[resourceIndex].tensorViewIndex;

        writeDescriptorSets[descriptorWriteIndex]                 = {};
        writeDescriptorSets[descriptorWriteIndex].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[descriptorWriteIndex].pNext           = &tensorDescriptorInfos[tensorDescriptorIndex];
        writeDescriptorSets[descriptorWriteIndex].dstSet          = pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex];
        writeDescriptorSets[descriptorWriteIndex].descriptorCount = 1;
        writeDescriptorSets[descriptorWriteIndex].descriptorType  = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
        writeDescriptorSets[descriptorWriteIndex].dstBinding      = binding.slotIndex;
        writeDescriptorSets[descriptorWriteIndex].dstArrayElement = binding.arrayIndex;

        tensorDescriptorInfos[tensorDescriptorIndex].tensorViewCount = 1;
        tensorDescriptorInfos[tensorDescriptorIndex].pTensorViews    = &backendContext->pResourceViews[tensorViewIndex].tensorView;

        tensorDescriptorIndex++;
        descriptorWriteIndex++;
    }

    // update uniform buffers
    for (uint32_t currentRootConstantIndex = 0; currentRootConstantIndex < job->computeJobDescriptor.pipeline.constCount; ++currentRootConstantIndex)
    {
        uint32_t dataSize = job->computeJobDescriptor.cbs[currentRootConstantIndex].num32BitEntries * sizeof(uint32_t);

        // If we have a constant buffer allocator, use that, otherwise use the default backend allocator
        FfxConstantAllocation allocation;
        if (s_fpConstantAllocator)
            allocation = s_fpConstantAllocator(job->computeJobDescriptor.cbs[currentRootConstantIndex].data, dataSize);
        else
            allocation = backendContext->FallbackConstantAllocator(job->computeJobDescriptor.cbs[currentRootConstantIndex].data, dataSize);

        writeDescriptorSets[descriptorWriteIndex]                 = {};
        writeDescriptorSets[descriptorWriteIndex].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[descriptorWriteIndex].dstSet          = pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex];
        writeDescriptorSets[descriptorWriteIndex].descriptorCount = 1;
        writeDescriptorSets[descriptorWriteIndex].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[descriptorWriteIndex].pBufferInfo     = &bufferDescriptorInfos[bufferDescriptorIndex];
        writeDescriptorSets[descriptorWriteIndex].dstBinding = job->computeJobDescriptor.pipeline.constantBufferBindings[currentRootConstantIndex].slotIndex;
        writeDescriptorSets[descriptorWriteIndex].dstArrayElement = 0;

        bufferDescriptorInfos[bufferDescriptorIndex].buffer = static_cast<VkBuffer>(allocation.resource.resource);
        bufferDescriptorInfos[bufferDescriptorIndex].offset = static_cast<VkDeviceSize>(allocation.handle);
        bufferDescriptorInfos[bufferDescriptorIndex].range  = dataSize;

        bufferDescriptorIndex++;
        descriptorWriteIndex++;
    }

    // If we are dispatching indirectly, transition the argument resource to indirect argument
    if (job->computeJobDescriptor.pipeline.cmdSignature)
    {
        addBarrier(backendContext, &job->computeJobDescriptor.cmdArgument, FFX_RESOURCE_STATE_INDIRECT_ARGUMENT);
    }

    // insert all the barriers
    flushBarriers(backendContext, vkCommandBuffer);

    // update all uavs and srvs
    VulkanWrapper().vkUpdateDescriptorSets(backendContext->device, descriptorWriteIndex, writeDescriptorSets, 0, nullptr);

    // bind pipeline
    VulkanWrapper().vkCmdBindPipeline(
        vkCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, reinterpret_cast<VkPipeline>(job->computeJobDescriptor.pipeline.pipeline));

    // bind descriptor sets
    {
        VulkanWrapper().vkCmdBindDescriptorSets(vkCommandBuffer,
                                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                                pipelineLayout->pipelineLayout,
                                                0,
                                                1,
                                                &pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex],
                                                0,
                                                nullptr);

        BackendContext_VK::EffectContext& effectContext = backendContext->pEffectContexts[effectContextId];

        if (job->computeJobDescriptor.pipeline.staticTextureSrvCount > 0)
            VulkanWrapper().vkCmdBindDescriptorSets(vkCommandBuffer,
                                                    VK_PIPELINE_BIND_POINT_COMPUTE,
                                                    pipelineLayout->pipelineLayout,
                                                    pipelineLayout->staticTextureSrvSet,
                                                    1,
                                                    &effectContext.bindlessTextureSrvDescriptorSet,
                                                    0,
                                                    nullptr);

        if (job->computeJobDescriptor.pipeline.staticBufferSrvCount > 0)
            VulkanWrapper().vkCmdBindDescriptorSets(vkCommandBuffer,
                                                    VK_PIPELINE_BIND_POINT_COMPUTE,
                                                    pipelineLayout->pipelineLayout,
                                                    pipelineLayout->staticBufferSrvSet,
                                                    1,
                                                    &effectContext.bindlessBufferSrvDescriptorSet,
                                                    0,
                                                    nullptr);

        if (job->computeJobDescriptor.pipeline.staticTextureUavCount > 0)
            VulkanWrapper().vkCmdBindDescriptorSets(vkCommandBuffer,
                                                    VK_PIPELINE_BIND_POINT_COMPUTE,
                                                    pipelineLayout->pipelineLayout,
                                                    pipelineLayout->staticTextureUavSet,
                                                    1,
                                                    &effectContext.bindlessTextureUavDescriptorSet,
                                                    0,
                                                    nullptr);

        if (job->computeJobDescriptor.pipeline.staticBufferUavCount > 0)
            VulkanWrapper().vkCmdBindDescriptorSets(vkCommandBuffer,
                                                    VK_PIPELINE_BIND_POINT_COMPUTE,
                                                    pipelineLayout->pipelineLayout,
                                                    pipelineLayout->staticBufferUavSet,
                                                    1,
                                                    &effectContext.bindlessBufferUavDescriptorSet,
                                                    0,
                                                    nullptr);
    }

    // Dispatch (or dispatch indirect)
    if (job->computeJobDescriptor.pipeline.cmdSignature)
    {
        const uint32_t resourceIndex = job->computeJobDescriptor.cmdArgument.internalIndex;
        VkBuffer       buffer        = backendContext->pResources[resourceIndex].bufferResource;
        VulkanWrapper().vkCmdDispatchIndirect(vkCommandBuffer, buffer, job->computeJobDescriptor.cmdArgumentOffset);
    }
    else
    {
        VulkanWrapper().vkCmdDispatch(
            vkCommandBuffer, job->computeJobDescriptor.dimensions[0], job->computeJobDescriptor.dimensions[1], job->computeJobDescriptor.dimensions[2]);
    }

    // move to another descriptor set for the next compute render job so that we don't overwrite descriptors in-use
    ++pipelineLayout->descriptorSetIndex;
    if (pipelineLayout->descriptorSetIndex >= (FFX_MAX_QUEUED_FRAMES * MAX_PIPELINE_USAGE_PER_FRAME))
        pipelineLayout->descriptorSetIndex = 0;

    return FFX_OK;
}

static FfxErrorCode executeGpuJobFragment(BackendContext_VK* backendContext, FfxGpuJobDescription* job, VkCommandBuffer vkCommandBuffer)
{
    BackendContext_VK::PipelineLayout* pipelineLayout = reinterpret_cast<BackendContext_VK::PipelineLayout*>(job->fragmentJobDescriptor.pipeline.rootSignature);

    // bind texture & buffer UAVs (note the binding order here MUST match the root signature mapping order from CreatePipeline!)
    uint32_t             descriptorWriteIndex = 0;
    VkWriteDescriptorSet writeDescriptorSets[FFX_MAX_RESOURCE_COUNT];

    // These MUST be initialized
    uint32_t              imageDescriptorIndex = 0;
    VkDescriptorImageInfo imageDescriptorInfos[FFX_MAX_RESOURCE_COUNT];
    for (int i = 0; i < FFX_MAX_RESOURCE_COUNT; ++i)
        imageDescriptorInfos[i] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    // These MUST be initialized
    uint32_t               bufferDescriptorIndex = 0;
    VkDescriptorBufferInfo bufferDescriptorInfos[FFX_MAX_RESOURCE_COUNT];
    for (int i = 0; i < FFX_MAX_RESOURCE_COUNT; ++i)
        bufferDescriptorInfos[i] = {VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};

    uint32_t                      tensorDescriptorIndex = 0;
    VkWriteDescriptorSetTensorARM tensorDescriptorInfos[FFX_MAX_RESOURCE_COUNT];
    for (int i = 0; i < FFX_MAX_RESOURCE_COUNT; ++i)
        tensorDescriptorInfos[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM, nullptr};

    // bind texture UAVs
    for (uint32_t currentPipelineUavIndex = 0; currentPipelineUavIndex < job->fragmentJobDescriptor.pipeline.uavTextureCount; ++currentPipelineUavIndex)
    {
        FfxTextureUAV& textureUAV = job->fragmentJobDescriptor.uavTextures[currentPipelineUavIndex];

        // continue if this is a null resource.
        if (job->fragmentJobDescriptor.uavTextures[currentPipelineUavIndex].resource.internalIndex == 0)
            continue;

        addBarrier(backendContext, &textureUAV.resource, FFX_RESOURCE_STATE_PIXEL_UAV);

        const FfxResourceBinding binding = job->fragmentJobDescriptor.pipeline.uavTextureBindings[currentPipelineUavIndex];

        // where to bind it
        const uint32_t currentUavResourceIndex = job->fragmentJobDescriptor.pipeline.uavTextureBindings[currentPipelineUavIndex].slotIndex;

        // source: UAV of resource to bind
        const uint32_t resourceIndex = textureUAV.resource.internalIndex;
        uint32_t       mipOffset     = textureUAV.mip;
        if (textureUAV.mip >= backendContext->pResources[resourceIndex].resourceDescription.mipCount)
            mipOffset = backendContext->pResources[resourceIndex].resourceDescription.mipCount - 1;
        const uint32_t uavViewIndex = backendContext->pResources[resourceIndex].uavViewIndex + mipOffset;

        writeDescriptorSets[descriptorWriteIndex]                 = {};
        writeDescriptorSets[descriptorWriteIndex].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[descriptorWriteIndex].dstSet          = pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex];
        writeDescriptorSets[descriptorWriteIndex].descriptorCount = 1;
        writeDescriptorSets[descriptorWriteIndex].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writeDescriptorSets[descriptorWriteIndex].pImageInfo      = &imageDescriptorInfos[imageDescriptorIndex];
        writeDescriptorSets[descriptorWriteIndex].dstBinding      = binding.slotIndex;
        writeDescriptorSets[descriptorWriteIndex].dstArrayElement = binding.arrayIndex;

        imageDescriptorInfos[imageDescriptorIndex]             = {};
        imageDescriptorInfos[imageDescriptorIndex].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageDescriptorInfos[imageDescriptorIndex].imageView   = backendContext->pResourceViews[uavViewIndex].imageView;

        imageDescriptorIndex++;
        descriptorWriteIndex++;
    }

    // bind buffer UAVs
    for (uint32_t currentPipelineUavIndex = 0; currentPipelineUavIndex < job->fragmentJobDescriptor.pipeline.uavBufferCount; ++currentPipelineUavIndex)
    {
        FfxBufferUAV& bufferUAV = job->fragmentJobDescriptor.uavBuffers[currentPipelineUavIndex];

        // continue if this is a null resource.
        if (job->fragmentJobDescriptor.uavBuffers[currentPipelineUavIndex].resource.internalIndex == 0)
            continue;

        addBarrier(backendContext, &bufferUAV.resource, FFX_RESOURCE_STATE_PIXEL_UAV);

        const FfxResourceBinding binding = job->fragmentJobDescriptor.pipeline.uavBufferBindings[currentPipelineUavIndex];

        // source: UAV of buffer to bind
        const uint32_t resourceIndex = bufferUAV.resource.internalIndex;

        // where to bind it
        const uint32_t currentUavResourceIndex = job->fragmentJobDescriptor.pipeline.uavBufferBindings[currentPipelineUavIndex].slotIndex;

        writeDescriptorSets[descriptorWriteIndex]                 = {};
        writeDescriptorSets[descriptorWriteIndex].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[descriptorWriteIndex].dstSet          = pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex];
        writeDescriptorSets[descriptorWriteIndex].descriptorCount = 1;
        writeDescriptorSets[descriptorWriteIndex].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSets[descriptorWriteIndex].pBufferInfo     = &bufferDescriptorInfos[bufferDescriptorIndex];
        writeDescriptorSets[descriptorWriteIndex].dstBinding      = binding.slotIndex;
        writeDescriptorSets[descriptorWriteIndex].dstArrayElement = binding.arrayIndex;

        bufferDescriptorInfos[bufferDescriptorIndex] = {};

        const auto& backendResource = backendContext->pResources[resourceIndex];
        const bool  isTensor        = (backendResource.resourceDescription.type == FFX_RESOURCE_TYPE_TENSOR);
        const bool  isBufferAliased = ((backendResource.resourceDescription.flags & FFX_RESOURCE_FLAGS_BUFFER_ALIASED) == FFX_RESOURCE_FLAGS_BUFFER_ALIASED);

        if (isBufferAliased)
        {
            bufferDescriptorInfos[bufferDescriptorIndex].buffer =
                isTensor ? backendResource.aliasedTensorBufferResource : backendResource.aliasedBufferResource;
            bufferDescriptorInfos[bufferDescriptorIndex].range = VK_WHOLE_SIZE;
        }
        else
        {
            bufferDescriptorInfos[bufferDescriptorIndex].buffer = backendResource.bufferResource;
            bufferDescriptorInfos[bufferDescriptorIndex].range  = bufferUAV.size > 0 ? bufferUAV.size : backendResource.resourceDescription.size;
        }

        bufferDescriptorInfos[bufferDescriptorIndex].offset = bufferUAV.offset;

        bufferDescriptorIndex++;
        descriptorWriteIndex++;
    }

    // bind texture SRVs
    for (uint32_t currentPipelineSrvIndex = 0; currentPipelineSrvIndex < job->fragmentJobDescriptor.pipeline.srvTextureCount; ++currentPipelineSrvIndex)
    {
        FfxTextureSRV& textureSRV = job->fragmentJobDescriptor.srvTextures[currentPipelineSrvIndex];

        // continue if this is a null resource.
        if (job->fragmentJobDescriptor.srvTextures[currentPipelineSrvIndex].resource.internalIndex == 0)
            continue;

        addBarrier(backendContext, &textureSRV.resource, FFX_RESOURCE_STATE_PIXEL_READ);

        const FfxResourceBinding binding = job->fragmentJobDescriptor.pipeline.srvTextureBindings[currentPipelineSrvIndex];

        // where to bind it
        const uint32_t currentSrvResourceIndex = job->fragmentJobDescriptor.pipeline.srvTextureBindings[currentPipelineSrvIndex].slotIndex;

        writeDescriptorSets[descriptorWriteIndex]                 = {};
        writeDescriptorSets[descriptorWriteIndex].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[descriptorWriteIndex].dstSet          = pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex];
        writeDescriptorSets[descriptorWriteIndex].descriptorCount = 1;
        writeDescriptorSets[descriptorWriteIndex].descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writeDescriptorSets[descriptorWriteIndex].pImageInfo      = &imageDescriptorInfos[imageDescriptorIndex];
        writeDescriptorSets[descriptorWriteIndex].dstBinding      = binding.slotIndex;
        writeDescriptorSets[descriptorWriteIndex].dstArrayElement = binding.arrayIndex;

        const uint32_t resourceIndex = textureSRV.resource.internalIndex;
        const uint32_t srvViewIndex  = backendContext->pResources[resourceIndex].srvViewIndex;

        imageDescriptorInfos[imageDescriptorIndex]             = {};
        imageDescriptorInfos[imageDescriptorIndex].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageDescriptorInfos[imageDescriptorIndex].imageView   = backendContext->pResourceViews[srvViewIndex].imageView;

        imageDescriptorIndex++;
        descriptorWriteIndex++;
    }

    // bind buffer SRVs
    for (uint32_t currentPipelineSrvIndex = 0; currentPipelineSrvIndex < job->fragmentJobDescriptor.pipeline.srvBufferCount; ++currentPipelineSrvIndex)
    {
        FfxBufferSRV& bufferSRV = job->fragmentJobDescriptor.srvBuffers[currentPipelineSrvIndex];

        // continue if this is a null resource.
        if (job->fragmentJobDescriptor.srvBuffers[currentPipelineSrvIndex].resource.internalIndex == 0)
            continue;

        addBarrier(backendContext, &bufferSRV.resource, FFX_RESOURCE_STATE_PIXEL_READ);

        const FfxResourceBinding binding = job->fragmentJobDescriptor.pipeline.srvBufferBindings[currentPipelineSrvIndex];

        // source: SRV of buffer to bind
        const uint32_t resourceIndex = bufferSRV.resource.internalIndex;

        // where to bind it
        const uint32_t currentSrvResourceIndex = job->fragmentJobDescriptor.pipeline.srvBufferBindings[currentPipelineSrvIndex].slotIndex;

        writeDescriptorSets[descriptorWriteIndex]                 = {};
        writeDescriptorSets[descriptorWriteIndex].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[descriptorWriteIndex].dstSet          = pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex];
        writeDescriptorSets[descriptorWriteIndex].descriptorCount = 1;
        writeDescriptorSets[descriptorWriteIndex].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSets[descriptorWriteIndex].pBufferInfo     = &bufferDescriptorInfos[bufferDescriptorIndex];
        writeDescriptorSets[descriptorWriteIndex].dstBinding      = binding.slotIndex;
        writeDescriptorSets[descriptorWriteIndex].dstArrayElement = binding.arrayIndex;

        bufferDescriptorInfos[bufferDescriptorIndex] = {};

        const auto& backendResource = backendContext->pResources[resourceIndex];
        const bool  isTensor        = (backendResource.resourceDescription.type == FFX_RESOURCE_TYPE_TENSOR);
        const bool  isBufferAliased = ((backendResource.resourceDescription.flags & FFX_RESOURCE_FLAGS_BUFFER_ALIASED) == FFX_RESOURCE_FLAGS_BUFFER_ALIASED);

        if (isBufferAliased)
        {
            bufferDescriptorInfos[bufferDescriptorIndex].buffer =
                isTensor ? backendResource.aliasedTensorBufferResource : backendResource.aliasedBufferResource;
            bufferDescriptorInfos[bufferDescriptorIndex].range = VK_WHOLE_SIZE;
        }
        else
        {
            bufferDescriptorInfos[bufferDescriptorIndex].buffer = backendResource.bufferResource;
            bufferDescriptorInfos[bufferDescriptorIndex].range  = bufferSRV.size > 0 ? bufferSRV.size : backendResource.resourceDescription.size;
        }
        bufferDescriptorInfos[bufferDescriptorIndex].offset = bufferSRV.offset;

        bufferDescriptorIndex++;
        descriptorWriteIndex++;
    }

    // bind tensor UAVs
    for (uint32_t currentPipelineUavIndex = 0; currentPipelineUavIndex < job->fragmentJobDescriptor.pipeline.uavTensorCount; ++currentPipelineUavIndex)
    {
        FfxTensor& tensor = job->fragmentJobDescriptor.uavTensors[currentPipelineUavIndex];

        if (tensor.resource.internalIndex == 0)
            continue;

        addBarrier(backendContext, &tensor.resource, FFX_RESOURCE_STATE_PIXEL_UAV);

        const FfxResourceBinding binding = job->fragmentJobDescriptor.pipeline.uavTensorBindings[currentPipelineUavIndex];

        const uint32_t resourceIndex   = job->fragmentJobDescriptor.uavTensors[currentPipelineUavIndex].resource.internalIndex;
        const uint32_t tensorViewIndex = backendContext->pResources[resourceIndex].tensorViewIndex;

        writeDescriptorSets[descriptorWriteIndex]                 = {};
        writeDescriptorSets[descriptorWriteIndex].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[descriptorWriteIndex].pNext           = &tensorDescriptorInfos[tensorDescriptorIndex];
        writeDescriptorSets[descriptorWriteIndex].dstSet          = pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex];
        writeDescriptorSets[descriptorWriteIndex].descriptorCount = 1;
        writeDescriptorSets[descriptorWriteIndex].descriptorType  = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
        writeDescriptorSets[descriptorWriteIndex].dstBinding      = binding.slotIndex;
        writeDescriptorSets[descriptorWriteIndex].dstArrayElement = binding.arrayIndex;

        tensorDescriptorInfos[tensorDescriptorIndex].tensorViewCount = 1;
        tensorDescriptorInfos[tensorDescriptorIndex].pTensorViews    = &backendContext->pResourceViews[tensorViewIndex].tensorView;

        tensorDescriptorIndex++;
        descriptorWriteIndex++;
    }

    // bind tensor SRVs
    for (uint32_t currentPipelineSrvIndex = 0; currentPipelineSrvIndex < job->fragmentJobDescriptor.pipeline.srvTensorCount; ++currentPipelineSrvIndex)
    {
        FfxTensor& tensor = job->fragmentJobDescriptor.srvTensors[currentPipelineSrvIndex];

        if (tensor.resource.internalIndex == 0)
            continue;

        addBarrier(backendContext, &tensor.resource, FFX_RESOURCE_STATE_PIXEL_READ);

        const FfxResourceBinding binding = job->fragmentJobDescriptor.pipeline.srvTensorBindings[currentPipelineSrvIndex];

        const uint32_t resourceIndex   = job->fragmentJobDescriptor.srvTensors[currentPipelineSrvIndex].resource.internalIndex;
        const uint32_t tensorViewIndex = backendContext->pResources[resourceIndex].tensorViewIndex;

        writeDescriptorSets[descriptorWriteIndex]                 = {};
        writeDescriptorSets[descriptorWriteIndex].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[descriptorWriteIndex].pNext           = &tensorDescriptorInfos[tensorDescriptorIndex];
        writeDescriptorSets[descriptorWriteIndex].dstSet          = pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex];
        writeDescriptorSets[descriptorWriteIndex].descriptorCount = 1;
        writeDescriptorSets[descriptorWriteIndex].descriptorType  = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
        writeDescriptorSets[descriptorWriteIndex].dstBinding      = binding.slotIndex;
        writeDescriptorSets[descriptorWriteIndex].dstArrayElement = binding.arrayIndex;

        tensorDescriptorInfos[tensorDescriptorIndex].tensorViewCount = 1;
        tensorDescriptorInfos[tensorDescriptorIndex].pTensorViews    = &backendContext->pResourceViews[tensorViewIndex].tensorView;

        tensorDescriptorIndex++;
        descriptorWriteIndex++;
    }

    // update uniform buffers
    for (uint32_t currentRootConstantIndex = 0; currentRootConstantIndex < job->fragmentJobDescriptor.pipeline.constCount; ++currentRootConstantIndex)
    {
        uint32_t dataSize = job->fragmentJobDescriptor.cbs[currentRootConstantIndex].num32BitEntries * sizeof(uint32_t);

        // If we have a constant buffer allocator, use that, otherwise use the default backend allocator
        FfxConstantAllocation allocation;
        if (s_fpConstantAllocator)
            allocation = s_fpConstantAllocator(job->fragmentJobDescriptor.cbs[currentRootConstantIndex].data, dataSize);
        else
            allocation = backendContext->FallbackConstantAllocator(job->fragmentJobDescriptor.cbs[currentRootConstantIndex].data, dataSize);

        writeDescriptorSets[descriptorWriteIndex]                 = {};
        writeDescriptorSets[descriptorWriteIndex].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[descriptorWriteIndex].dstSet          = pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex];
        writeDescriptorSets[descriptorWriteIndex].descriptorCount = 1;
        writeDescriptorSets[descriptorWriteIndex].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[descriptorWriteIndex].pBufferInfo     = &bufferDescriptorInfos[bufferDescriptorIndex];
        writeDescriptorSets[descriptorWriteIndex].dstBinding = job->fragmentJobDescriptor.pipeline.constantBufferBindings[currentRootConstantIndex].slotIndex;
        writeDescriptorSets[descriptorWriteIndex].dstArrayElement = 0;

        bufferDescriptorInfos[bufferDescriptorIndex].buffer = static_cast<VkBuffer>(allocation.resource.resource);
        bufferDescriptorInfos[bufferDescriptorIndex].offset = static_cast<VkDeviceSize>(allocation.handle);
        bufferDescriptorInfos[bufferDescriptorIndex].range  = dataSize;

        bufferDescriptorIndex++;
        descriptorWriteIndex++;
    }

    // Transition RTs
    for (FfxUInt32 rt = 0; rt < job->fragmentJobDescriptor.pipeline.rtCount; ++rt)
    {
        addBarrier(backendContext, &job->fragmentJobDescriptor.rtTextures[rt].resource, FFX_RESOURCE_STATE_RENDER_TARGET);
    }

    // insert all the barriers
    flushBarriers(backendContext, vkCommandBuffer);

    // update all uavs and srvs
    VulkanWrapper().vkUpdateDescriptorSets(backendContext->device, descriptorWriteIndex, writeDescriptorSets, 0, nullptr);

    getOrCreateRenderPass(backendContext, job);

    getOrCreateFrameBuffer(backendContext, job);

    getOrCreateGraphicsPipeline(backendContext, job);

    constexpr VkClearValue clearColors[FFX_MAX_NUM_RTS] = {};
    const VkExtent2D       extent                       = {job->fragmentJobDescriptor.viewport[0], job->fragmentJobDescriptor.viewport[1]};

    VkRenderPassBeginInfo renderPassBeginInfo = {};
    renderPassBeginInfo.sType                 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass            = pipelineLayout->renderPass[pipelineLayout->renderPassIndex].handle;
    renderPassBeginInfo.framebuffer           = pipelineLayout->frameBuffer[pipelineLayout->frameBufferIndex].handle;
    renderPassBeginInfo.renderArea.offset.x   = 0;
    renderPassBeginInfo.renderArea.offset.y   = 0;
    renderPassBeginInfo.renderArea.extent     = extent;
    renderPassBeginInfo.clearValueCount       = job->fragmentJobDescriptor.pipeline.rtCount;
    renderPassBeginInfo.pClearValues          = (job->fragmentJobDescriptor.pipeline.rtCount > 0) ? clearColors : nullptr;

    VulkanWrapper().vkCmdBeginRenderPass(vkCommandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    // bind pipeline
    VulkanWrapper().vkCmdBindPipeline(
        vkCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, reinterpret_cast<VkPipeline>(job->fragmentJobDescriptor.pipeline.pipeline));

    // bind descriptor sets
    VulkanWrapper().vkCmdBindDescriptorSets(vkCommandBuffer,
                                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            pipelineLayout->pipelineLayout,
                                            0,
                                            1,
                                            &pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex],
                                            0,
                                            nullptr);

    const VkViewport viewport = {0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height)};
    VulkanWrapper().vkCmdSetViewport(vkCommandBuffer, 0, 1, &viewport);

    const VkRect2D scissor = {{0, 0}, extent};
    VulkanWrapper().vkCmdSetScissor(vkCommandBuffer, 0, 1, &scissor);

    VulkanWrapper().vkCmdDraw(vkCommandBuffer, 3, 1, 0, 0);

    VulkanWrapper().vkCmdEndRenderPass(vkCommandBuffer);

    // move to another descriptor set for the next compute render job so that we don't overwrite descriptors in-use
    ++pipelineLayout->descriptorSetIndex;
    if (pipelineLayout->descriptorSetIndex >= (FFX_MAX_QUEUED_FRAMES * MAX_PIPELINE_USAGE_PER_FRAME))
        pipelineLayout->descriptorSetIndex = 0;

    return FFX_OK;
}

static FfxErrorCode executeGpuJobDataGraph(BackendContext_VK* backendContext, FfxGpuJobDescription* job, VkCommandBuffer vkCommandBuffer)
{
    BackendContext_VK::PipelineLayout* pipelineLayout =
        reinterpret_cast<BackendContext_VK::PipelineLayout*>(job->dataGraphJobDescription.pipeline.rootSignature);
    FFX_ASSERT(pipelineLayout->effectContextId < backendContext->maxEffectContexts);

    const BackendContext_VK::EffectContext& effectContext = backendContext->pEffectContexts[pipelineLayout->effectContextId];

    uint32_t             descriptorWriteIndex = 0;
    VkWriteDescriptorSet writeDescriptorSets[FFX_MAX_RESOURCE_COUNT];

    uint32_t                      tensorDescriptorIndex = 0;
    VkWriteDescriptorSetTensorARM tensorDescriptorInfos[FFX_MAX_RESOURCE_COUNT];
    for (int i = 0; i < FFX_MAX_RESOURCE_COUNT; ++i)
        tensorDescriptorInfos[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM, nullptr};

    // These MUST be initialized
    uint32_t              imageDescriptorIndex = 0;
    VkDescriptorImageInfo imageDescriptorInfos[FFX_MAX_RESOURCE_COUNT];
    for (int i = 0; i < FFX_MAX_RESOURCE_COUNT; ++i)
        imageDescriptorInfos[i] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    // The data graph pipeline only bind the tensor resources and potentially texture resoruces for data graph optical flow
    FFX_ASSERT(job->dataGraphJobDescription.pipeline.srvBufferCount == 0);
    FFX_ASSERT(job->dataGraphJobDescription.pipeline.uavBufferCount == 0);
    FFX_ASSERT(job->dataGraphJobDescription.pipeline.staticTextureSrvCount == 0);
    FFX_ASSERT(job->dataGraphJobDescription.pipeline.staticBufferSrvCount == 0);
    FFX_ASSERT(job->dataGraphJobDescription.pipeline.staticTextureUavCount == 0);
    FFX_ASSERT(job->dataGraphJobDescription.pipeline.staticBufferUavCount == 0);
    FFX_ASSERT(job->dataGraphJobDescription.pipeline.rtCount == 0);

    // bind texture UAVs
    for (uint32_t currentPipelineUavIndex = 0; currentPipelineUavIndex < job->dataGraphJobDescription.pipeline.uavTextureCount; ++currentPipelineUavIndex)
    {
        FfxTextureUAV& textureUAV = job->dataGraphJobDescription.uavTextures[currentPipelineUavIndex];

        // continue if this is a null resource.
        if (job->dataGraphJobDescription.uavTextures[currentPipelineUavIndex].resource.internalIndex == 0)
            continue;

        addBarrier(backendContext, &textureUAV.resource, FFX_RESOURCE_STATE_DATA_GRAPH_WRITE);

        const FfxResourceBinding binding = job->dataGraphJobDescription.pipeline.uavTextureBindings[currentPipelineUavIndex];

        // where to bind it
        const uint32_t currentUavResourceIndex = job->dataGraphJobDescription.pipeline.uavTextureBindings[currentPipelineUavIndex].slotIndex;

        // source: UAV of resource to bind
        const uint32_t resourceIndex = textureUAV.resource.internalIndex;
        uint32_t       mipOffset     = textureUAV.mip;
        if (textureUAV.mip >= backendContext->pResources[resourceIndex].resourceDescription.mipCount)
            mipOffset = backendContext->pResources[resourceIndex].resourceDescription.mipCount - 1;
        const uint32_t uavViewIndex = backendContext->pResources[resourceIndex].uavViewIndex + mipOffset;

        writeDescriptorSets[descriptorWriteIndex]        = {};
        writeDescriptorSets[descriptorWriteIndex].sType  = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[descriptorWriteIndex].dstSet = pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex];

        writeDescriptorSets[descriptorWriteIndex].descriptorCount = 1;
        writeDescriptorSets[descriptorWriteIndex].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writeDescriptorSets[descriptorWriteIndex].pImageInfo      = &imageDescriptorInfos[imageDescriptorIndex];
        writeDescriptorSets[descriptorWriteIndex].dstBinding      = binding.slotIndex;
        writeDescriptorSets[descriptorWriteIndex].dstArrayElement = binding.arrayIndex;

        imageDescriptorInfos[imageDescriptorIndex]             = {};
        imageDescriptorInfos[imageDescriptorIndex].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageDescriptorInfos[imageDescriptorIndex].imageView   = backendContext->pResourceViews[uavViewIndex].imageView;

        imageDescriptorIndex++;
        descriptorWriteIndex++;
    }

    // bind texture SRVs
    for (uint32_t currentPipelineSrvIndex = 0; currentPipelineSrvIndex < job->dataGraphJobDescription.pipeline.srvTextureCount; ++currentPipelineSrvIndex)
    {
        FfxTextureSRV& textureSRV = job->dataGraphJobDescription.srvTextures[currentPipelineSrvIndex];

        // continue if this is a null resource.
        if (job->dataGraphJobDescription.srvTextures[currentPipelineSrvIndex].resource.internalIndex == 0)
            continue;

        addBarrier(backendContext, &textureSRV.resource, FFX_RESOURCE_STATE_DATA_GRAPH_READ);

        const FfxResourceBinding binding = job->dataGraphJobDescription.pipeline.srvTextureBindings[currentPipelineSrvIndex];

        // where to bind it
        const uint32_t currentSrvResourceIndex = job->dataGraphJobDescription.pipeline.srvTextureBindings[currentPipelineSrvIndex].slotIndex;

        writeDescriptorSets[descriptorWriteIndex]                 = {};
        writeDescriptorSets[descriptorWriteIndex].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[descriptorWriteIndex].dstSet          = pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex];
        writeDescriptorSets[descriptorWriteIndex].descriptorCount = 1;
        writeDescriptorSets[descriptorWriteIndex].descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writeDescriptorSets[descriptorWriteIndex].pImageInfo      = &imageDescriptorInfos[imageDescriptorIndex];
        writeDescriptorSets[descriptorWriteIndex].dstBinding      = binding.slotIndex;
        writeDescriptorSets[descriptorWriteIndex].dstArrayElement = binding.arrayIndex;

        const uint32_t resourceIndex = textureSRV.resource.internalIndex;
        const uint32_t srvViewIndex  = backendContext->pResources[resourceIndex].srvViewIndex;

        imageDescriptorInfos[imageDescriptorIndex]             = {};
        imageDescriptorInfos[imageDescriptorIndex].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageDescriptorInfos[imageDescriptorIndex].imageView   = backendContext->pResourceViews[srvViewIndex].imageView;

        imageDescriptorIndex++;
        descriptorWriteIndex++;
    }

    // bind tensor UAVs
    for (uint32_t currentPipelineUavIndex = 0; currentPipelineUavIndex < job->dataGraphJobDescription.pipeline.uavTensorCount; ++currentPipelineUavIndex)
    {
        FfxTensor& tensor = job->dataGraphJobDescription.uavTensors[currentPipelineUavIndex];

        // continue if this is a null resource.
        if (tensor.resource.internalIndex == 0)
            continue;

        addBarrier(backendContext, &tensor.resource, FFX_RESOURCE_STATE_DATA_GRAPH_WRITE);

        const FfxResourceBinding binding = job->dataGraphJobDescription.pipeline.uavTensorBindings[currentPipelineUavIndex];

        const uint32_t resourceIndex   = job->dataGraphJobDescription.uavTensors[currentPipelineUavIndex].resource.internalIndex;
        const uint32_t tensorViewIndex = backendContext->pResources[resourceIndex].tensorViewIndex;

        writeDescriptorSets[descriptorWriteIndex]                 = {};
        writeDescriptorSets[descriptorWriteIndex].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[descriptorWriteIndex].pNext           = &tensorDescriptorInfos[tensorDescriptorIndex];
        writeDescriptorSets[descriptorWriteIndex].dstSet          = pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex];
        writeDescriptorSets[descriptorWriteIndex].descriptorCount = 1;
        writeDescriptorSets[descriptorWriteIndex].descriptorType  = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
        writeDescriptorSets[descriptorWriteIndex].dstBinding      = binding.slotIndex;
        writeDescriptorSets[descriptorWriteIndex].dstArrayElement = binding.arrayIndex;

        tensorDescriptorInfos[tensorDescriptorIndex].tensorViewCount = 1;
        tensorDescriptorInfos[tensorDescriptorIndex].pTensorViews    = &backendContext->pResourceViews[tensorViewIndex].tensorView;

        tensorDescriptorIndex++;
        descriptorWriteIndex++;
    }

    // bind tensor SRVs
    for (uint32_t currentPipelineSrvIndex = 0; currentPipelineSrvIndex < job->dataGraphJobDescription.pipeline.srvTensorCount; ++currentPipelineSrvIndex)
    {
        FfxTensor& tensor = job->dataGraphJobDescription.srvTensors[currentPipelineSrvIndex];

        // continue if this is a null resource.
        if (tensor.resource.internalIndex == 0)
            continue;

        addBarrier(backendContext, &tensor.resource, FFX_RESOURCE_STATE_DATA_GRAPH_READ);

        const FfxResourceBinding binding = job->dataGraphJobDescription.pipeline.srvTensorBindings[currentPipelineSrvIndex];

        const uint32_t resourceIndex   = job->dataGraphJobDescription.srvTensors[currentPipelineSrvIndex].resource.internalIndex;
        const uint32_t tensorViewIndex = backendContext->pResources[resourceIndex].tensorViewIndex;

        writeDescriptorSets[descriptorWriteIndex]                 = {};
        writeDescriptorSets[descriptorWriteIndex].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[descriptorWriteIndex].pNext           = &tensorDescriptorInfos[tensorDescriptorIndex];
        writeDescriptorSets[descriptorWriteIndex].dstSet          = pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex];
        writeDescriptorSets[descriptorWriteIndex].descriptorCount = 1;
        writeDescriptorSets[descriptorWriteIndex].descriptorType  = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
        writeDescriptorSets[descriptorWriteIndex].dstBinding      = binding.slotIndex;
        writeDescriptorSets[descriptorWriteIndex].dstArrayElement = binding.arrayIndex;

        tensorDescriptorInfos[tensorDescriptorIndex].tensorViewCount = 1;
        tensorDescriptorInfos[tensorDescriptorIndex].pTensorViews    = &backendContext->pResourceViews[tensorViewIndex].tensorView;

        tensorDescriptorIndex++;
        descriptorWriteIndex++;
    }

    // insert all the barriers
    flushBarriers(backendContext, vkCommandBuffer);

    // update all uavs and srvs
    VulkanWrapper().vkUpdateDescriptorSets(backendContext->device, descriptorWriteIndex, writeDescriptorSets, 0, nullptr);

    // bind pipeline
    VulkanWrapper().vkCmdBindPipeline(
        vkCommandBuffer, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, reinterpret_cast<VkPipeline>(job->dataGraphJobDescription.pipeline.pipeline));

    // bind descriptor sets
    VulkanWrapper().vkCmdBindDescriptorSets(vkCommandBuffer,
                                            VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM,
                                            pipelineLayout->pipelineLayout,
                                            0,
                                            1,
                                            &pipelineLayout->descriptorSets[pipelineLayout->descriptorSetIndex],
                                            0,
                                            nullptr);

    VkDataGraphPipelineDispatchInfoARM dispatch_info = {};
    dispatch_info.sType                              = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_DISPATCH_INFO_ARM;
    dispatch_info.flags                              = 0;

    if (effectContext.effectId == ARM_EFFECT_OPTICALFLOW)
    {
        VkDataGraphPipelineOpticalFlowDispatchInfoARM dispatch_optical_flow_info = {};
        dispatch_optical_flow_info.sType                                         = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_OPTICAL_FLOW_DISPATCH_INFO_ARM;
        dispatch_optical_flow_info.pNext                                         = nullptr;
        dispatch_optical_flow_info.flags              = ffxGetVkOpticalFlowExecuteFlags(job->dataGraphJobDescription.opticalFlowExecuteFlags);
        dispatch_optical_flow_info.meanFlowL1NormHint = job->dataGraphJobDescription.meanFlowL1NormHint;
        dispatch_info.pNext                           = &dispatch_optical_flow_info;

        VulkanWrapper().vkCmdDispatchDataGraphARM(
            vkCommandBuffer, reinterpret_cast<VkDataGraphPipelineSessionARM>(job->dataGraphJobDescription.pipeline.session), &dispatch_info);
    }
    else
    {
        VulkanWrapper().vkCmdDispatchDataGraphARM(
            vkCommandBuffer, reinterpret_cast<VkDataGraphPipelineSessionARM>(job->dataGraphJobDescription.pipeline.session), &dispatch_info);
    }

    // move to another descriptor set for the next compute render job so that we don't overwrite descriptors in-use
    ++pipelineLayout->descriptorSetIndex;
    if (pipelineLayout->descriptorSetIndex >= (FFX_MAX_QUEUED_FRAMES * MAX_PIPELINE_USAGE_PER_FRAME))
        pipelineLayout->descriptorSetIndex = 0;

    return FFX_OK;
}

static FfxErrorCode executeGpuJobCopy(BackendContext_VK* backendContext, FfxGpuJobDescription* job, VkCommandBuffer vkCommandBuffer)
{
    BackendContext_VK::Resource ffxResourceSrc = backendContext->pResources[job->copyJobDescriptor.src.internalIndex];
    BackendContext_VK::Resource ffxResourceDst = backendContext->pResources[job->copyJobDescriptor.dst.internalIndex];

    addBarrier(backendContext, &job->copyJobDescriptor.src, FFX_RESOURCE_STATE_COPY_SRC);
    addBarrier(backendContext, &job->copyJobDescriptor.dst, FFX_RESOURCE_STATE_COPY_DEST);
    flushBarriers(backendContext, vkCommandBuffer);

    if (ffxResourceSrc.resourceDescription.type == FFX_RESOURCE_TYPE_BUFFER && ffxResourceDst.resourceDescription.type == FFX_RESOURCE_TYPE_BUFFER)
    {
        VkBuffer vkResourceSrc = ffxResourceSrc.bufferResource;
        VkBuffer vkResourceDst = ffxResourceDst.bufferResource;

        VkBufferCopy bufferCopy = {};

        bufferCopy.dstOffset = job->copyJobDescriptor.dstOffset;
        bufferCopy.srcOffset = job->copyJobDescriptor.srcOffset;
        bufferCopy.size      = job->copyJobDescriptor.size > 0 ? job->copyJobDescriptor.size : ffxResourceSrc.resourceDescription.width;

        VulkanWrapper().vkCmdCopyBuffer(vkCommandBuffer, vkResourceSrc, vkResourceDst, 1, &bufferCopy);
    }
    else if (ffxResourceSrc.resourceDescription.type == FFX_RESOURCE_TYPE_BUFFER && ffxResourceDst.resourceDescription.type != FFX_RESOURCE_TYPE_BUFFER)
    {
        VkBuffer vkResourceSrc = ffxResourceSrc.bufferResource;
        VkImage  vkResourceDst = ffxResourceDst.imageResource;

        VkImageSubresourceLayers subresourceLayers = {};

        subresourceLayers.aspectMask     = getImageAspect(ffxResourceDst.resourceDescription.usage);
        subresourceLayers.baseArrayLayer = 0;
        subresourceLayers.layerCount     = 1;
        subresourceLayers.mipLevel       = 0;

        VkOffset3D offset = {};

        offset.x = 0;
        offset.y = 0;
        offset.z = 0;

        VkExtent3D extent = {};

        extent.width  = ffxResourceDst.resourceDescription.width;
        extent.height = ffxResourceDst.resourceDescription.height;
        extent.depth  = ffxResourceDst.resourceDescription.depth;

        // TODO: account for source buffer offset
        VkBufferImageCopy bufferImageCopy = {};

        bufferImageCopy.bufferOffset      = 0;
        bufferImageCopy.bufferRowLength   = 0;
        bufferImageCopy.bufferImageHeight = 0;
        bufferImageCopy.imageSubresource  = subresourceLayers;
        bufferImageCopy.imageOffset       = offset;
        bufferImageCopy.imageExtent       = extent;

        VulkanWrapper().vkCmdCopyBufferToImage(vkCommandBuffer, vkResourceSrc, vkResourceDst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bufferImageCopy);
    }
    else
    {
#define FFX_MAX_IMAGE_COPY_MIPS 14  // Will handle 4k down to 1x1
        VkImageCopy imageCopies[FFX_MAX_IMAGE_COPY_MIPS];
        VkImage     vkResourceSrc = ffxResourceSrc.imageResource;
        VkImage     vkResourceDst = ffxResourceDst.imageResource;

        uint32_t numMipsToCopy = FFX_MINIMUM(ffxResourceSrc.resourceDescription.mipCount, ffxResourceDst.resourceDescription.mipCount);

        for (uint32_t mip = 0; mip < numMipsToCopy; mip++)
        {
            VkImageSubresourceLayers srcSubresourceLayers = {};

            srcSubresourceLayers.aspectMask     = getImageAspect(ffxResourceSrc.resourceDescription.usage);
            srcSubresourceLayers.baseArrayLayer = 0;
            srcSubresourceLayers.layerCount     = 1;
            srcSubresourceLayers.mipLevel       = mip;

            VkImageSubresourceLayers dstSubresourceLayers = {};

            dstSubresourceLayers.aspectMask     = getImageAspect(ffxResourceDst.resourceDescription.usage);
            dstSubresourceLayers.baseArrayLayer = 0;
            dstSubresourceLayers.layerCount     = 1;
            dstSubresourceLayers.mipLevel       = mip;

            VkOffset3D offset = {};

            offset.x = 0;
            offset.y = 0;
            offset.z = 0;

            VkExtent3D extent = {};

            switch (job->copyJobDescriptor.copyMode)
            {
            case FFX_GPU_COPY_SRC_EXTENT:
                extent.width  = ffxResourceSrc.resourceDescription.width / (mip + 1);
                extent.height = ffxResourceSrc.resourceDescription.height / (mip + 1);
                extent.depth  = ffxResourceSrc.resourceDescription.depth / (mip + 1);
                break;
            case FFX_GPU_COPY_DST_EXTENT:
                extent.width  = ffxResourceDst.resourceDescription.width / (mip + 1);
                extent.height = ffxResourceDst.resourceDescription.height / (mip + 1);
                extent.depth  = ffxResourceDst.resourceDescription.depth / (mip + 1);
                break;
            case FFX_GPU_COPY_MIN_EXTENT:
                extent.width  = FFX_MINIMUM(ffxResourceSrc.resourceDescription.width, ffxResourceDst.resourceDescription.width) / (mip + 1);
                extent.height = FFX_MINIMUM(ffxResourceSrc.resourceDescription.height, ffxResourceDst.resourceDescription.height) / (mip + 1);
                extent.depth  = FFX_MINIMUM(ffxResourceSrc.resourceDescription.depth, ffxResourceDst.resourceDescription.depth) / (mip + 1);
                break;
            default:
                FFX_ASSERT_MESSAGE(false, "Unsupported copy mode in vulkan backend of FFX SDK.");
                break;
            }

            VkImageCopy& copyRegion = imageCopies[mip];

            copyRegion.srcSubresource = srcSubresourceLayers;
            copyRegion.srcOffset      = offset;
            copyRegion.dstSubresource = dstSubresourceLayers;
            copyRegion.dstOffset      = offset;
            copyRegion.extent         = extent;
        }

        VulkanWrapper().vkCmdCopyImage(vkCommandBuffer,
                                       vkResourceSrc,
                                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       vkResourceDst,
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                       numMipsToCopy,
                                       imageCopies);
    }

    return FFX_OK;
}

static FfxErrorCode executeGpuJobBarrier(BackendContext_VK* backendContext, FfxGpuJobDescription* job, VkCommandBuffer vkCommandBuffer)
{
    addBarrier(backendContext, &job->barrierDescriptor.resource, job->barrierDescriptor.newState);
    flushBarriers(backendContext, vkCommandBuffer);

    return FFX_OK;
}

static FfxErrorCode executeGpuJobTimestamp(BackendContext_VK* backendContext, FfxGpuJobDescription* job, VkCommandBuffer vkCommandBuffer)
{
    return FFX_OK;
}

static FfxErrorCode executeGpuJobClearFloat(BackendContext_VK* backendContext, FfxGpuJobDescription* job, VkCommandBuffer vkCommandBuffer)
{
    uint32_t idx = job->clearJobDescriptor.target.internalIndex;

    BackendContext_VK::Resource ffxResource = backendContext->pResources[idx];

    if (ffxResource.resourceDescription.type == FFX_RESOURCE_TYPE_BUFFER)
    {
        addBarrier(backendContext, &job->clearJobDescriptor.target, FFX_RESOURCE_STATE_COPY_DEST);
        flushBarriers(backendContext, vkCommandBuffer);

        VkBuffer vkResource = ffxResource.bufferResource;

        VulkanWrapper().vkCmdFillBuffer(vkCommandBuffer, vkResource, 0, VK_WHOLE_SIZE, (uint32_t)job->clearJobDescriptor.color[0]);
    }
    else if (ffxResource.resourceDescription.type == FFX_RESOURCE_TYPE_TENSOR &&
             (ffxResource.resourceDescription.flags & FFX_RESOURCE_FLAGS_IMAGE_ALIASED) != FFX_RESOURCE_FLAGS_IMAGE_ALIASED)
    {
        // no vk function for clearing a tensor - must use a shader, which is not an effect agnostic job.
    }
    else
    {
        addBarrier(backendContext, &job->clearJobDescriptor.target, FFX_RESOURCE_STATE_COPY_DEST);
        flushBarriers(backendContext, vkCommandBuffer);

        VkImage vkResource = ffxResource.imageResource;

        VkClearColorValue clearColorValue = {};

        clearColorValue.float32[0] = job->clearJobDescriptor.color[0];
        clearColorValue.float32[1] = job->clearJobDescriptor.color[1];
        clearColorValue.float32[2] = job->clearJobDescriptor.color[2];
        clearColorValue.float32[3] = job->clearJobDescriptor.color[3];

        VkImageSubresourceRange range;
        range.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        range.baseMipLevel   = 0;
        range.levelCount     = ffxResource.resourceDescription.mipCount;
        range.baseArrayLayer = 0;
        range.layerCount     = VK_REMAINING_ARRAY_LAYERS;

        // For aliased tensors, we need to clear as SINT
        if (ffxResource.resourceDescription.type == FFX_RESOURCE_TYPE_TENSOR)
        {
            vkResource                            = ffxResource.aliasedTensorImageResource;
            VkClearColorValue clearColorValueSint = {};
            clearColorValueSint.int32[0]          = static_cast<int32_t>(job->clearJobDescriptor.color[0] * 255 - 128);
            clearColorValueSint.int32[1]          = static_cast<int32_t>(job->clearJobDescriptor.color[1] * 255 - 128);
            clearColorValueSint.int32[2]          = static_cast<int32_t>(job->clearJobDescriptor.color[2] * 255 - 128);
            clearColorValueSint.int32[3]          = static_cast<int32_t>(job->clearJobDescriptor.color[3] * 255 - 128);
            clearColorValue                       = clearColorValueSint;
        }
        VulkanWrapper().vkCmdClearColorImage(vkCommandBuffer, vkResource, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColorValue, 1, &range);
    }

    return FFX_OK;
}

static FfxErrorCode executeGpuJobClearUint(BackendContext_VK* backendContext, FfxGpuJobDescription* job, VkCommandBuffer vkCommandBuffer)
{
    uint32_t idx = job->clearUintJobDescriptor.target.internalIndex;

    BackendContext_VK::Resource ffxResource = backendContext->pResources[idx];

    if (ffxResource.resourceDescription.type == FFX_RESOURCE_TYPE_BUFFER)
    {
        addBarrier(backendContext, &job->clearUintJobDescriptor.target, FFX_RESOURCE_STATE_COPY_DEST);
        flushBarriers(backendContext, vkCommandBuffer);

        VkBuffer vkResource = ffxResource.bufferResource;

        VulkanWrapper().vkCmdFillBuffer(vkCommandBuffer, vkResource, 0, VK_WHOLE_SIZE, job->clearUintJobDescriptor.color[0]);
    }
    else if (ffxResource.resourceDescription.type == FFX_RESOURCE_TYPE_TENSOR &&
             (ffxResource.resourceDescription.flags & FFX_RESOURCE_FLAGS_IMAGE_ALIASED) != FFX_RESOURCE_FLAGS_IMAGE_ALIASED)
    {
        // no vk function for clearing a tensor - must use a shader, which is not an effect agnostic job.
    }
    else
    {
        addBarrier(backendContext, &job->clearUintJobDescriptor.target, FFX_RESOURCE_STATE_COPY_DEST);
        flushBarriers(backendContext, vkCommandBuffer);

        VkImage vkResource = ffxResource.imageResource;

        VkClearColorValue clearColorValue = {};

        clearColorValue.uint32[0] = job->clearUintJobDescriptor.color[0];
        clearColorValue.uint32[1] = job->clearUintJobDescriptor.color[1];
        clearColorValue.uint32[2] = job->clearUintJobDescriptor.color[2];
        clearColorValue.uint32[3] = job->clearUintJobDescriptor.color[3];

        VkImageSubresourceRange range;
        range.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        range.baseMipLevel   = 0;
        range.levelCount     = ffxResource.resourceDescription.mipCount;
        range.baseArrayLayer = 0;
        range.layerCount     = VK_REMAINING_ARRAY_LAYERS;
        VulkanWrapper().vkCmdClearColorImage(vkCommandBuffer, vkResource, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColorValue, 1, &range);
    }

    return FFX_OK;
}

FfxErrorCode ExecuteGpuJobsVK(FfxInterface* backendInterface, FfxCommandList commandList, FfxUInt32 effectContextId)
{
    FFX_ASSERT(nullptr != backendInterface);
    BackendContext_VK* backendContext = (BackendContext_VK*)backendInterface->scratchBuffer;

    FFX_ASSERT(nullptr != commandList);
    VkCommandBuffer vkCommandBuffer = reinterpret_cast<VkCommandBuffer>(commandList);

    FfxErrorCode errorCode = FFX_OK;

    // execute all renderjobs
    for (uint32_t i = 0; i < backendContext->gpuJobCount; ++i)
    {
        FfxGpuJobDescription* gpuJob = &backendContext->pGpuJobs[i];

        // If we have a label for the job, drop a marker for it
#ifdef FFX_DEBUG
        if (gpuJob->jobLabel[0])
        {
            beginMarkerVK(backendContext, vkCommandBuffer, gpuJob->jobLabel);
        }
#endif

        switch (gpuJob->jobType)
        {
        case FFX_GPU_JOB_CLEAR_FLOAT:
        {
            errorCode = executeGpuJobClearFloat(backendContext, gpuJob, vkCommandBuffer);
            break;
        }
        case FFX_GPU_JOB_COPY:
        {
            errorCode = executeGpuJobCopy(backendContext, gpuJob, vkCommandBuffer);
            break;
        }
        case FFX_GPU_JOB_COMPUTE:
        {
            errorCode = executeGpuJobCompute(backendContext, gpuJob, vkCommandBuffer, effectContextId);
            break;
        }
        case FFX_GPU_JOB_BARRIER:
        {
            errorCode = executeGpuJobBarrier(backendContext, gpuJob, vkCommandBuffer);
            break;
        }
        case FFX_GPU_JOB_FRAGMENT:
        {
            errorCode = executeGpuJobFragment(backendContext, gpuJob, vkCommandBuffer);
            break;
        }
        case FFX_GPU_JOB_DATA_GRAPH:
        {
            errorCode = executeGpuJobDataGraph(backendContext, gpuJob, vkCommandBuffer);
            break;
        }
        case FFX_GPU_JOB_CLEAR_UINT:
        {
            errorCode = executeGpuJobClearUint(backendContext, gpuJob, vkCommandBuffer);
            break;
        }
        default:;
        }

#ifdef FFX_DEBUG
        if (gpuJob->jobLabel[0])
        {
            endMarkerVK(backendContext, vkCommandBuffer);
        }
#endif
    }

    // check the execute function returned cleanly.
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, FFX_ERROR_BACKEND_API_ERROR);

    backendContext->gpuJobCount = 0;

    return FFX_OK;
}

void RegisterConstantBufferAllocatorVK(FfxInterface*, FfxConstantBufferAllocator fpConstantAllocator)
{
    s_fpConstantAllocator = fpConstantAllocator;
}

FfxCommandQueue ffxGetCommandQueueVK(VkQueue commandQueue)
{
    FFX_ASSERT(commandQueue != VK_NULL_HANDLE);
    return reinterpret_cast<FfxCommandQueue>(commandQueue);
}

FfxSwapchain ffxGetSwapchainVK(VkSwapchainKHR swapchain)
{
    FFX_ASSERT(swapchain != VK_NULL_HANDLE);
    return reinterpret_cast<FfxSwapchain>(swapchain);
}

VkSwapchainKHR ffxGetVKSwapchain(FfxSwapchain ffxSwapchain)
{
    return reinterpret_cast<VkSwapchainKHR>(ffxSwapchain);
}
