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

#include "backends.h"

#ifdef FFX_BACKEND_VK
#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <ffx_api/vk/ffx_api_vk.h>
#endif  // #ifdef FFX_BACKEND_VK

ffxReturnCode_t CreateBackend(const ffxCreateContextDescHeader* desc, bool& backendFound, FfxInterface* iface, size_t contexts, Allocator& alloc)
{
    for (const auto* it = desc->pNext; it; it = it->pNext)
    {
        switch (it->type)
        {
#ifdef FFX_BACKEND_VK
        case FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK:
        {
            // check for double backend just to make sure.
            if (backendFound)
                return FFX_API_RETURN_ERROR;
            backendFound = true;

            const auto*     backendDesc       = reinterpret_cast<const ffxCreateBackendVKDesc*>(it);
            VkDeviceContext deviceContext     = {backendDesc->vkDevice,
                                             backendDesc->vkPhysicalDevice,
                                             backendDesc->vkDeviceProcAddr,
                                             backendDesc->vkInstance,
                                             backendDesc->vkGetInstanceProcAddr};
            FfxDevice       device            = ffxGetDeviceVK(&deviceContext);
            size_t          scratchBufferSize = ffxGetScratchMemorySizeVK(deviceContext, contexts);
            void*           scratchBuffer     = alloc.alloc(scratchBufferSize);
            memset(scratchBuffer, 0, scratchBufferSize);
            TRY2(ffxGetInterfaceVK(iface, device, scratchBuffer, scratchBufferSize, contexts));
            break;
        }
#endif  // FFX_BACKEND_VK
        }
    }
    return FFX_API_RETURN_OK;
}

void* GetDevice(const ffxApiHeader* desc)
{
    for (const auto* it = desc; it; it = it->pNext)
    {
        switch (it->type)
        {
        case FFX_API_QUERY_DESC_TYPE_GET_VERSIONS:
        {
            return reinterpret_cast<const ffxQueryDescGetVersions*>(it)->device;
        }
#ifdef FFX_BACKEND_VK
        case FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK:
        {
            return reinterpret_cast<const ffxCreateBackendVKDesc*>(it)->vkDevice;
        }
#endif
        }
    }
    return nullptr;
}
