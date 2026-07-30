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

#include <vulkan/vulkan.h>

#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <FidelityFX/host/ffx_assert.h>
#include <FidelityFX/host/backends/vk/vk_wrapper.h>

#ifdef FFX_PLATFORM_WINDOWS
#include <Windows.h>
#include <synchapi.h>
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
#include <pthread.h>
#include <math.h>
#endif

void waitForPerformanceCount(const int64_t targetCount);

struct SubmissionSemaphores
{
    static const uint32_t Capacity = 6;

    VkSemaphore          semaphores[Capacity];
    uint64_t             values[Capacity];
    VkPipelineStageFlags waitStages[Capacity];
    uint32_t             count;

    SubmissionSemaphores()
    {
        reset();
    }

    bool isEmpty()
    {
        return count == 0;
    }

    void reset()
    {
        count = 0;
        memset(values, 0, sizeof(values));
        memset(waitStages, 0, sizeof(waitStages));
    }

    void add(VkSemaphore semaphore, uint64_t value = 0)
    {
        if (semaphore != VK_NULL_HANDLE)
        {
            FFX_ASSERT_MESSAGE(count < Capacity, "[FrameInterpolationSwapchainVK] SubmissionSemaphores capacity exceeded. Please increase it.");
            if (count < Capacity)
            {
                semaphores[count] = semaphore;
                values[count]     = value;
                waitStages[count] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                ++count;
            }
        }
    }
};

struct VulkanQueue : public VkQueueInfoFFX
{
    void operator=(const VkQueueInfoFFX& info)
    {
        queue       = info.queue;
        familyIndex = info.familyIndex;
        submitFunc  = info.submitFunc;
    }

    void reset()
    {
        queue       = VK_NULL_HANDLE;
        familyIndex = 0;
        submitFunc  = nullptr;
    }

    VkResult submit(VkCommandBuffer       commandBuffer,
                    SubmissionSemaphores& semaphoresToWait,
                    SubmissionSemaphores& semaphoresToSignal,
                    VkFence               fence = VK_NULL_HANDLE);
    VkResult submit(VkCommandBuffer commandBuffer, VkSemaphore timelineSemaphore, uint64_t signalValue);
};

class VkCommands
{
    VkDevice        device                  = VK_NULL_HANDLE;
    VulkanQueue     queue                   = {};
    VkCommandPool   commandPool             = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer           = VK_NULL_HANDLE;
    VkSemaphore     semaphore               = VK_NULL_HANDLE;
    uint64_t        availableSemaphoreValue = 0;

public:
    void release(const VkAllocationCallbacks* pAllocator = nullptr)
    {
        if (device != VK_NULL_HANDLE)
        {
            VulkanWrapper().vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
            VulkanWrapper().vkDestroyCommandPool(device, commandPool, pAllocator);
            VulkanWrapper().vkDestroySemaphore(device, semaphore, pAllocator);
        }

        device        = VK_NULL_HANDLE;
        commandBuffer = VK_NULL_HANDLE;
        commandPool   = VK_NULL_HANDLE;
        semaphore     = VK_NULL_HANDLE;
    }

public:
    ~VkCommands()
    {
        release();
    }

    bool initiated()
    {
        return commandPool != VK_NULL_HANDLE;
    }

    bool verify(VkDevice inputDevice, uint32_t queueFamily)
    {
        VkResult res = initiated() ? VK_SUCCESS : VK_ERROR_UNKNOWN;

        // TODO: set names

        if (res != VK_SUCCESS)
        {
            device = inputDevice;

            // command pools
            VkCommandPoolCreateInfo poolInfo = {};
            poolInfo.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.pNext                   = nullptr;
            poolInfo.flags                   = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            poolInfo.queueFamilyIndex        = queueFamily;

            res = VulkanWrapper().vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);

            if (res == VK_SUCCESS)
            {
                // command buffers
                VkCommandBufferAllocateInfo allocInfo = {};
                allocInfo.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                allocInfo.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                allocInfo.commandPool                 = commandPool;
                allocInfo.commandBufferCount          = 1;

                res = VulkanWrapper().vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
            }
            if (res == VK_SUCCESS)
            {
                // create timeline semaphore
                VkSemaphoreTypeCreateInfo semaphoreTypeCreateInfo = {};
                semaphoreTypeCreateInfo.sType                     = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
                semaphoreTypeCreateInfo.pNext                     = nullptr;
                semaphoreTypeCreateInfo.semaphoreType             = VK_SEMAPHORE_TYPE_TIMELINE;
                semaphoreTypeCreateInfo.initialValue              = availableSemaphoreValue;

                VkSemaphoreCreateInfo semaphoreCreateInfo = {};
                semaphoreCreateInfo.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
                semaphoreCreateInfo.pNext                 = &semaphoreTypeCreateInfo;
                semaphoreCreateInfo.flags                 = 0;

                res = VulkanWrapper().vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphore);
            }
        }
        if (res != VK_SUCCESS)
        {
            release();
        }

        return res == VK_SUCCESS;
    }

    void occupy(VulkanQueue inputQueue, const char* name)
    {
        availableSemaphoreValue++;
        queue = inputQueue;

        // TODO: set names
    }

    VkCommandBuffer reset()
    {
        VkResult res = VulkanWrapper().vkResetCommandPool(device, commandPool, 0);
        if (res == VK_SUCCESS)
        {
            res = VulkanWrapper().vkResetCommandBuffer(commandBuffer, 0);
            if (res == VK_SUCCESS)
            {
            }
        }

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.pNext                    = nullptr;
        beginInfo.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        beginInfo.pInheritanceInfo         = nullptr;

        res = VulkanWrapper().vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    VkCommandBuffer list()
    {
        return commandBuffer;
    }

    VkResult execute()
    {
        VkResult res = VulkanWrapper().vkEndCommandBuffer(commandBuffer);
        if (res != VK_SUCCESS)
            return res;

        return queue.submit(commandBuffer, semaphore, availableSemaphoreValue);
    }

    VkResult execute(SubmissionSemaphores& semaphoresToWait, SubmissionSemaphores& semaphoresToSignal)
    {
        VkResult res = VulkanWrapper().vkEndCommandBuffer(commandBuffer);
        if (res != VK_SUCCESS)
            return res;

        semaphoresToSignal.add(semaphore, availableSemaphoreValue);
        return queue.submit(commandBuffer, semaphoresToWait, semaphoresToSignal);
    }

    VkResult drop()
    {
        VkResult res = VulkanWrapper().vkEndCommandBuffer(commandBuffer);
        if (res != VK_SUCCESS)
            return res;

        return queue.submit(VK_NULL_HANDLE, semaphore, availableSemaphoreValue);
    }

    bool available()
    {
        uint64_t value = 0;
        if (VulkanWrapper().vkGetSemaphoreCounterValue(device, semaphore, &value) == VK_SUCCESS)
            return value >= availableSemaphoreValue;
        return false;
    }
};

template <size_t NumFamilies, size_t Capacity>
class VulkanCommandPool
{
public:
private:
#ifdef FFX_PLATFORM_WINDOWS
    CRITICAL_SECTION criticalSection = {};
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
    pthread_mutex_t criticalSection = PTHREAD_MUTEX_INITIALIZER;
#endif
    uint32_t   queueFamilyIndices[NumFamilies] = {};
    VkCommands buffer[NumFamilies][Capacity]   = {};

public:
    VulkanCommandPool()
    {
#ifdef FFX_PLATFORM_WINDOWS
        InitializeCriticalSection(&criticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
        pthread_mutex_init(&criticalSection, nullptr);
#endif
        for (size_t familyIndex = 0; familyIndex < NumFamilies; familyIndex++)
            queueFamilyIndices[familyIndex] = UINT32_MAX;
    }

    ~VulkanCommandPool()
    {
#ifdef FFX_PLATFORM_WINDOWS
        EnterCriticalSection(&criticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
        pthread_mutex_lock(&criticalSection);
#endif

        for (size_t familyIndex = 0; familyIndex < NumFamilies; familyIndex++)
        {
            for (size_t idx = 0; idx < Capacity; idx++)
            {
                auto& cmds = buffer[familyIndex][idx];
                while (cmds.initiated() && !cmds.available())
                {
                    // wait for list to be idling
                }
                cmds.release();
            }

            queueFamilyIndices[familyIndex] = UINT32_MAX;
        }
#ifdef FFX_PLATFORM_WINDOWS
        LeaveCriticalSection(&criticalSection);

        DeleteCriticalSection(&criticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
        pthread_mutex_unlock(&criticalSection);
        pthread_mutex_destroy(&criticalSection);
#endif
    }

    VkCommands* get(VkDevice device, VulkanQueue queue, const char* name)
    {
#ifdef FFX_PLATFORM_WINDOWS
        EnterCriticalSection(&criticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
        pthread_mutex_lock(&criticalSection);
#endif

        uint32_t familyIndex = 0;
        // find family index
        for (; familyIndex < NumFamilies; familyIndex++)
        {
            if (queueFamilyIndices[familyIndex] == queue.familyIndex)
            {
                break;
            }
            else if (queueFamilyIndices[familyIndex] == UINT32_MAX)
            {
                queueFamilyIndices[familyIndex] = queue.familyIndex;
                break;
            }
        }

        if (familyIndex >= NumFamilies)
        {
            FFX_ASSERT(false);
            return nullptr;
        }

        VkCommands* pCommands = nullptr;

        for (size_t idx = 0; idx < Capacity && (pCommands == nullptr); idx++)
        {
            auto& cmds = buffer[familyIndex][idx];
            if (cmds.verify(device, queue.familyIndex) && cmds.available())
            {
                pCommands = &cmds;
            }
        }

        FFX_ASSERT(pCommands);

        pCommands->occupy(queue, name);

#ifdef FFX_PLATFORM_WINDOWS
        LeaveCriticalSection(&criticalSection);
#elif FFX_PLATFORM_ANDROID || FFX_PLATFORM_LINUX
        pthread_mutex_unlock(&criticalSection);
#endif
        return pCommands;
    }
};

template <const int Size, typename Type = double>
struct SimpleMovingAverage
{
    Type         history[Size] = {};
    unsigned int idx           = 0;
    unsigned int updateCount   = 0;

    Type getAverage()
    {
        if (updateCount < Size)
            return 0.0;

        Type         average    = 0.f;
        unsigned int iterations = (updateCount >= Size) ? Size : updateCount;

        if (iterations > 0)
        {
            for (size_t i = 0; i < iterations; i++)
            {
                average += history[i];
            }
            average /= iterations;
        }

        return average;
    }

    Type getVariance()
    {
        if (updateCount < Size)
            return 0.0;

        Type         average    = getAverage();
        Type         variance   = 0.f;
        unsigned int iterations = (updateCount >= Size) ? Size : updateCount;

        if (iterations > 0)
        {
            for (size_t i = 0; i < iterations; i++)
            {
                variance += (history[i] - average) * (history[i] - average);
            }
            variance /= iterations;
        }

        return sqrt(variance);
    }

    void reset()
    {
        updateCount = 0;
        idx         = 0;
    }

    void update(Type newValue)
    {
        history[idx] = newValue;
        idx          = (idx + 1) % Size;
        updateCount++;
    }
};

bool     LoadVulkanFunctions(VkDeviceContext* vkDeviceContext);
VkResult CreateShaderModule(VkDevice device, size_t codeSize, const uint32_t* pCode, VkShaderModule* pModule, const VkAllocationCallbacks* pAllocator);
