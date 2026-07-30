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
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#pragma once

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif  // _WIN32

#include <vulkan/vulkan.h>

#include <atomic>
#include <cstdint>

#include "FrameInterpolationSwapchainVK_Helpers.h"

#include <FidelityFX/host/backends/vk/ffx_vk.h>

// NOTES regarding using win32 objects:
//   - On Windows, critical section and events are faster than their std counterparts
//   - using Win32 threads to set the priorities
//   - this needs to be ported to standard C++ or other platform if necessary
#ifdef FFX_PLATFORM_WINDOWS
#include <Windows.h>
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
#include <pthread.h>
#endif

#define FFX_FRAME_INTERPOLATION_SWAP_CHAIN_VERSION                     1
#define FFX_FRAME_INTERPOLATION_SWAP_CHAIN_MAX_BUFFER_COUNT            6
#define FFX_FRAME_INTERPOLATION_SWAP_CHAIN_MAX_ACQUIRE_SEMAPHORE_COUNT 8

//////////////////////////////////////////////
/// Vulkan API overridden functions
//////////////////////////////////////////////

#ifdef __cplusplus
extern "C" {
#endif

// Provided by VK_KHR_swapchain
VkResult vkAcquireNextImageFFX(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore, VkFence fence, uint32_t* pImageIndex);

VkResult vkCreateSwapchainFFX(VkDevice                        device,
                              const VkSwapchainCreateInfoKHR* pCreateInfo,
                              const VkAllocationCallbacks*    pAllocator,
                              VkSwapchainKHR*                 pSwapchain);

void vkDestroySwapchainFFX(VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks* pAllocator);

VkResult vkGetSwapchainImagesFFX(VkDevice device, VkSwapchainKHR swapchain, uint32_t* pSwapchainImageCount, VkImage* pSwapchainImages);

VkResult vkQueuePresentFFX(VkQueue queue, const VkPresentInfoKHR* pPresentInfo);

// Provided by VK_EXT_hdr_metadata
void vkSetHdrMetadataFFX(VkDevice device, uint32_t swapchainCount, const VkSwapchainKHR* pSwapchains, const VkHdrMetadataEXT* pMetadata);

uint64_t getLastPresentCountFFX(VkSwapchainKHR swapchain);

#ifdef __cplusplus
}
#endif

typedef struct PacingData
{
    FfxPresentCallbackFunc presentCallback        = nullptr;
    void*                  presentCallbackContext = nullptr;
    FfxResource            uiSurface;

    VkPresentModeKHR presentMode;
    bool             usePremulAlphaComposite;
    bool             drawDebugPacingLines;

    uint64_t gameSemaphoreValue;
    uint64_t replacementBufferSemaphoreSignal;
    uint64_t numFramesSentForPresentationBase;
    uint32_t numFramesToPresent;
    uint64_t currentFrameID;

    typedef enum FrameType
    {
        Interpolated_1,
        Real,
        Count
    } FrameType;

    struct FrameInfo
    {
        uint32_t    realImageIndex;
        bool        doPresent;
        FfxResource resource;
        uint64_t    interpolationCompletedSemaphoreValue;
        uint64_t    presentIndex;
        uint64_t    presentQpcDelta;
    };

    FrameInfo frames[FrameType::Count];

    void invalidate()
    {
        memset(this, 0, sizeof(PacingData));
    }
} PacingData;

typedef struct ReplacementResource
{
    VkImage                image                      = VK_NULL_HANDLE;
    VkDeviceMemory         memory                     = VK_NULL_HANDLE;
    VkDeviceSize           allocationSize             = 0;
    uint64_t               availabilitySemaphoreValue = 0;
    FfxResourceDescription description                = {};
} ReplacementResource;

enum class FGSwapchainCompositionMode
{
    eNone,
    eComposeOnPresentQueue,  ///< optimal behavior
    eComposeOnGameQueue      ///< legacy behavior
};

struct FrameinterpolationPresentInfo
{
    VulkanCommandPool<3, 8> commandPool;  // at most 3 families: game, asyncCompute, present

    PacingData scheduledInterpolations;
    PacingData scheduledPresents;

    std::atomic<VkResult> lastPresentResult = VK_SUCCESS;

    FfxResource currentUiSurface   = {};
    uint32_t    uiCompositionFlags = 0;

    VkDevice         device         = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkSwapchainKHR   realSwapchain  = VK_NULL_HANDLE;

    uint32_t               realSwapchainImageCount = 0;
    VkImage                realSwapchainImages[FFX_FRAME_INTERPOLATION_SWAP_CHAIN_MAX_BUFFER_COUNT];
    VkSemaphore            frameRenderedSemaphores[FFX_FRAME_INTERPOLATION_SWAP_CHAIN_MAX_BUFFER_COUNT];
    FfxResourceDescription realSwapchainImageDescription;

    ReplacementResource compositionOutput;

    VulkanQueue interpolationQueue = {};
    VulkanQueue asyncComputeQueue  = {};
    VulkanQueue gameQueue          = {};
    VulkanQueue presentQueue       = {};

    VkSemaphore gameSemaphore              = VK_NULL_HANDLE;
    VkSemaphore interpolationSemaphore     = VK_NULL_HANDLE;
    VkSemaphore presentSemaphore           = VK_NULL_HANDLE;  // TODO: probably delete this one
    VkSemaphore replacementBufferSemaphore = VK_NULL_HANDLE;
    VkSemaphore compositionSemaphore       = VK_NULL_HANDLE;

    uint64_t lastPresentSemaphoreValue = 0;

    VkSemaphore acquireSemaphores[FFX_FRAME_INTERPOLATION_SWAP_CHAIN_MAX_ACQUIRE_SEMAPHORE_COUNT] = {};
    uint32_t    nextAcquireSemaphoreIndex                                                         = 0;

    uint64_t realPresentCount = 0;

    // using win32 threads to set the priorities
#ifdef FFX_PLATFORM_WINDOWS
    HANDLE           presenterThreadHandle         = NULL;
    CRITICAL_SECTION scheduledFrameCriticalSection = {};
    HANDLE           presentEvent                  = NULL;
    HANDLE           interpolationEvent            = NULL;
    HANDLE           pacerEvent                    = NULL;
    CRITICAL_SECTION swapchainCriticalSection;
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_t       presenterThreadHandle = 0;
    pthread_mutex_t presentMutex;
    pthread_cond_t  presentCond;
    bool            presentEventSignaled;
    pthread_mutex_t interpolationMutex;
    pthread_cond_t  interpolationCond;
    bool            interpolationEventSignaled;
    pthread_mutex_t scheduledFrameMutex;
    pthread_mutex_t pacerMutex;
    pthread_cond_t  pacerCond;
    bool            pacerEventSignaled;
    pthread_mutex_t swapchainMutex;
#endif

    FGSwapchainCompositionMode compositionMode = FGSwapchainCompositionMode::eNone;
    volatile bool              resetTimer      = false;
    volatile bool              shutdown        = false;

    VkResult acquireNextRealImage(uint32_t& imageIndex, VkSemaphore& acquireSemaphore);

    // small helpers for queue ownership transfer
    VkImageMemoryBarrier queueFamilyOwnershipTransferGameToPresent(const FfxResource& resource) const;

    volatile double safetyMarginInSec = 0.0001;  //0.1ms
    volatile double varianceFactor    = 0.1;

    FfxWaitCallbackFunc waitCallback = nullptr;
};

//////////////////////////////////////////////
/// FrameInterpolationSwapChainVK
//////////////////////////////////////////////

class FrameInterpolationSwapChainVK
{
public:
    FrameInterpolationSwapChainVK();
    virtual ~FrameInterpolationSwapChainVK();

public:  // vulkan functions reimplementation
    VkResult acquireNextImage(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore, VkFence fence, uint32_t* pImageIndex);

    VkResult init(const VkSwapchainCreateInfoKHR* pCreateInfo, const VkFrameInterpolationInfoFFX* pFrameInterpolationInfo);

    void destroySwapchain(VkDevice device, const VkAllocationCallbacks* pAllocator);

    VkResult getSwapchainImages(VkDevice device, uint32_t* pSwapchainImageCount, VkImage* pSwapchainImages);

    VkResult queuePresent(VkQueue queue, const VkPresentInfoKHR* pPresentInfo);

    void setHdrMetadata(VkDevice device, const VkHdrMetadataEXT* pMetadata);

    uint64_t getLastPresentCount();

public:
    void            setFrameGenerationConfig(FfxFrameGenerationConfig const* config);
    void            setFramePacingTuning(const FfxSwapchainFramePacingTuning* framePacingTuning);
    bool            waitForPresents();
    FfxResource     interpolationOutput(int index = 0);
    VkCommandBuffer getInterpolationCommandList();
    void            registerUiResource(const FfxResource& uiResource, uint32_t flags);
    void            setWaitCallback(FfxWaitCallbackFunc waitCallbackFunc);
    void            getGpuMemoryUsage(FfxEffectMemoryUsage* vramUsage);

private:
    bool spawnPresenterThread();
    bool killPresenterThread();

    VkResult presentInterpolated(const VkPresentInfoKHR* pPresentInfo, uint32_t currentBackBufferIndex, bool needUICopy);
    VkResult presentPassthrough(uint32_t              imageIndex,
                                SubmissionSemaphores& gameQueueWait,
                                SubmissionSemaphores& gameQueueSignal,
                                SubmissionSemaphores& presentQueueWait);
    VkResult presentNonInterpolatedWithUiCompositionOnGameQueue(uint32_t              imageIndex,
                                                                SubmissionSemaphores& gameQueueWait,
                                                                SubmissionSemaphores& gameQueueSignal,
                                                                SubmissionSemaphores& presentQueueWait,
                                                                bool                  needUICopy);
    VkResult presentNonInterpolatedWithUiCompositionOnPresentQueue(uint32_t              imageIndex,
                                                                   SubmissionSemaphores& gameQueueWait,
                                                                   SubmissionSemaphores& gameQueueSignal,
                                                                   SubmissionSemaphores& presentQueueWait,
                                                                   bool                  needUICopy);
    VkResult queuePresentNonInterpolated(VkCommands* pCommands, uint32_t imageIndex, SubmissionSemaphores& semaphoresToWait);

    void     dispatchInterpolationCommands(uint32_t              currentBackBufferIndex,
                                           FfxResource*          pInterpolatedFrame,
                                           FfxResource*          pRealFrame,
                                           SubmissionSemaphores& semaphoresToWait);
    void     discardOutstandingInterpolationCommandLists();
    VkResult submitCompositionOnGameQueue(const PacingData& entry);

    bool                 verifyUiDuplicateResource();
    VkImageMemoryBarrier copyUiResource(VkCommandBuffer commandBuffer, SubmissionSemaphores& gameQueueWait, bool transferToPresentQueue);

    VkResult createImage(ReplacementResource&                    resource,
                         VkImageCreateInfo&                      info,
                         FfxSurfaceFormat                        format,
                         const char*                             name,
                         const VkPhysicalDeviceMemoryProperties& memProperties,
                         const VkAllocationCallbacks*            pAllocator);
    VkResult createImage(ReplacementResource&                    resource,
                         VkImageCreateInfo&                      info,
                         FfxSurfaceFormat                        format,
                         const char*                             name,
                         uint32_t                                index,
                         const VkPhysicalDeviceMemoryProperties& memProperties,
                         const VkAllocationCallbacks*            pAllocator);
    void     destroyImage(ReplacementResource& resource, const VkAllocationCallbacks* pAllocator);

private:
    // swapchain settings
    VkPresentModeKHR presentMode       = VK_PRESENT_MODE_IMMEDIATE_KHR;
    VulkanQueue      imageAcquireQueue = {};

    // for framepacing
    FrameinterpolationPresentInfo presentInfo               = {};
    FfxFrameGenerationConfig      nextFrameGenerationConfig = {};

    uint64_t interpolationSemaphoreValue      = 0;
    uint64_t gameSemaphoreValue               = 0;
    bool     frameInterpolationResetCondition = false;

    uint32_t            gameBufferCount                                                             = 0;
    ReplacementResource replacementSwapBuffers[FFX_FRAME_INTERPOLATION_SWAP_CHAIN_MAX_BUFFER_COUNT] = {};
    ReplacementResource interpolationOutputs[2]                                                     = {};
    ReplacementResource uiReplacementBuffer                                                         = {};
    uint32_t            replacementSwapBufferIndex                                                  = 0;
    uint32_t            interpolationBufferIndex                                                    = 0;
    uint64_t            presentCount                                                                = 0;
    uint64_t            acquiredCount                                                               = 0;

    VkDeviceSize totalUsageInBytes     = 0;
    VkDeviceSize aliasableUsageInBytes = 0;

    bool        tearingSupported             = false;
    bool        interpolationEnabled         = false;
    bool        presentInterpolatedOnly      = false;
    bool        previousFrameWasInterpolated = false;
    bool        drawDebugPacingLines         = false;
    bool        dumpGeneratedFrame           = false;
    const char* dumpGeneratedFramePath       = nullptr;

    uint64_t currentFrameID = 0;

    double perfCountFreq = 0.0;

    uint64_t framesSentForPresentation = 0;

#ifdef FFX_PLATFORM_WINDOWS
    CRITICAL_SECTION criticalSection             = {};
    CRITICAL_SECTION criticalSectionUpdateConfig = {};
    HANDLE           interpolationThreadHandle   = NULL;
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_t criticalSection             = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t criticalSectionUpdateConfig = PTHREAD_MUTEX_INITIALIZER;
    pthread_t       interpolationThreadHandle   = 0;
#endif

    FfxPresentCallbackFunc         presentCallback                = nullptr;
    void*                          presentCallbackContext         = nullptr;
    FfxFrameGenerationDispatchFunc frameGenerationCallback        = nullptr;
    void*                          frameGenerationCallbackContext = nullptr;

    VkCommands* registeredInterpolationCommandLists[FFX_FRAME_INTERPOLATION_SWAP_CHAIN_MAX_BUFFER_COUNT] = {};

    // extension functions
    PFN_vkSetHdrMetadataEXT vkSetHdrMetadataEXTProc = nullptr;
};

VkFormat ffxGetVkFormatFromSurfaceFormat(FfxSurfaceFormat fmt);
