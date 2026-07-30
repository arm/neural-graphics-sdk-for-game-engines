/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

// Prevent vulkan.h from declaring Vulkan API functions as prototypes;
// the layer itself exports them, so duplicates would cause redefinition errors.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include <assert.h>
#include <cstring>

#include <cstdarg>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
// NOMINMAX prevents windows.h from defining min/max as macros, which would
// break std::min / std::max usage later in this file.
#define NOMINMAX
#include <windows.h>
#endif

#if defined(__ANDROID__)
#include <android/log.h>
#include <sys/system_properties.h>
#include <unistd.h>
#endif

#undef VK_LAYER_EXPORT
#if defined(_WIN32)
#define VK_LAYER_EXPORT extern "C" __declspec(dllexport)
#else
#define VK_LAYER_EXPORT extern "C"
#endif

#if defined(__ANDROID__) && !defined(NDEBUG)
__attribute__((constructor)) static void NGLayer_OnLibraryLoaded()
{
    __android_log_print(ANDROID_LOG_ERROR, "NGLAYER", "Arm NG Vulkan layer loaded pid=%d", getpid());
}
#endif

static void LayerLogImplV(const char* level, const char* fmt, va_list ap)
{
    char buf[2048];
    int  n = std::snprintf(buf, sizeof(buf), "[NG][%s] ", level);
    if (n < 0)
        return;
    std::vsnprintf(buf + n, sizeof(buf) - (size_t)n, fmt, ap);

    size_t len = std::strlen(buf);
    if (len == 0 || buf[len - 1] != '\n')
    {
        if (len + 1 < sizeof(buf))
        {
            buf[len]     = '\n';
            buf[len + 1] = '\0';
        }
    }

#if defined(_WIN32)
    OutputDebugStringA(buf);
#endif

#if defined(__ANDROID__)
    if (strncmp(level, "E", 1) == 0)
        __android_log_print(ANDROID_LOG_ERROR, "NGLAYER", "%s", buf);
    else
        __android_log_print(ANDROID_LOG_INFO, "NGLAYER", "%s", buf);
#endif

    std::fwrite(buf, 1, std::strlen(buf), stderr);
    std::fflush(stderr);
}

static inline void LayerLog(const char* level, const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    LayerLogImplV(level, fmt, ap);
    va_end(ap);
}

#define LAYER_LOGI(fmt, ...)               \
    do                                     \
    {                                      \
        LayerLog("I", fmt, ##__VA_ARGS__); \
    } while (0)
#define LAYER_LOGE(fmt, ...)               \
    do                                     \
    {                                      \
        LayerLog("E", fmt, ##__VA_ARGS__); \
    } while (0)

// Custom defination of DipatchableTable
struct VkLayerInstanceDispatchTable
{
    PFN_vkGetInstanceProcAddr                GetInstanceProcAddr                = nullptr;
    PFN_vkCreateDevice                       CreateDevice                       = nullptr;
    PFN_vkDestroyInstance                    DestroyInstance                    = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties = nullptr;
    PFN_vkGetPhysicalDeviceProperties        GetPhysicalDeviceProperties        = nullptr;
    PFN_vkGetPhysicalDeviceFeatures2         GetPhysicalDeviceFeatures2         = nullptr;
    PFN_vkGetPhysicalDeviceFeatures2KHR      GetPhysicalDeviceFeatures2KHR      = nullptr;
};

struct VkLayerDispatchTable
{
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr = nullptr;
    PFN_vkDestroyDevice     DestroyDevice     = nullptr;
};

std::mutex                          g_Lock;
typedef std::lock_guard<std::mutex> ScopedLock;

template <typename DispatchableType>
void* GetKey(DispatchableType inst)
{
    return *(void**)inst;
}

std::map<void*, VkLayerInstanceDispatchTable> g_instanceDispatchTable;
std::map<void*, VkLayerDispatchTable>         g_deviceDispatchTable;

static constexpr const char* kLayerName = "VK_LAYER_ARM_NG";

// ARM extension feature nodes used to query / inject the ARM ML features
// plus their sync2 interaction feature.
//
// VK_ARM_data_graph spec lists these "Extension and Version Dependencies":
//   - Vulkan 1.3
//   - VK_KHR_extended_flags OR VK_KHR_maintenance5
//   - VK_KHR_deferred_host_operations
struct NGFeatureChainNodes
{
    VkPhysicalDeviceFeatures2                       features2;
    VkPhysicalDeviceTensorFeaturesARM               tensor;
    VkPhysicalDeviceDataGraphFeaturesARM            graph;
    VkPhysicalDeviceDataGraphOpticalFlowFeaturesARM opticalFlow;
    VkPhysicalDeviceSynchronization2Features        sync2;

    NGFeatureChainNodes()
    {
        std::memset(&features2, 0, sizeof(features2));
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = nullptr;

        std::memset(&tensor, 0, sizeof(tensor));
        tensor.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM;
        tensor.pNext = nullptr;

        std::memset(&graph, 0, sizeof(graph));
        graph.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM;
        graph.pNext = nullptr;

        std::memset(&opticalFlow, 0, sizeof(opticalFlow));
        opticalFlow.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_OPTICAL_FLOW_FEATURES_ARM;
        opticalFlow.pNext = nullptr;

        std::memset(&sync2, 0, sizeof(sync2));
        sync2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
        sync2.pNext = nullptr;
    }

    void LinkForQuery()
    {
        features2.pNext   = &tensor;
        tensor.pNext      = &graph;
        graph.pNext       = &opticalFlow;
        opticalFlow.pNext = &sync2;
    }
};

// VK_KHR_extended_flags is brand-new (drafted 2025-12) and may not yet have a
// macro in the SDK headers we build against; use the literal name string.
static constexpr const char* kExtendedFlagsExtName = "VK_KHR_extended_flags";

/**
 * @brief Per-feature yes/no flags for the ARM ML features, their interaction
 *        feature (sync2), and the spec-required dependencies of
 *        VK_ARM_data_graph.
 *
 * The same shape is used in three roles:
 *  - device support (return value of AreExtensionsSupported): does the device
 *    advertise this extension / expose this feature bit?
 *  - already provided by app (return value of AreExtensionsProvidedByApp):
 *    is the feature struct already in pNext, or is the extension name already
 *    in ppEnabledExtensionNames?
 *  - injection request (return value of DecideInjection): should the layer
 *    push this feature struct or extension name into vkCreateDevice?
 *
 * `tensors` / `data_graph` / `sync2` are feature-struct kinds: chain presence
 * is checked in pNext.
 * `extended_flags` / `maintenance5` / `deferred_host_operations` are
 * extension-name kinds: chain presence is checked in ppEnabledExtensionNames.
 */
struct ExtensionFlags
{
    bool tensors;
    bool data_graph;
    bool data_graph_optical_flow;
    bool sync2;
    bool extended_flags;
    bool maintenance5;
    bool deferred_host_operations;

    bool AnyArmFeature() const
    {
        return tensors || data_graph || data_graph_optical_flow;
    }
    bool AnyMutation() const
    {
        return tensors || data_graph || data_graph_optical_flow || sync2 || extended_flags || maintenance5 || deferred_host_operations;
    }
};

VK_LAYER_EXPORT VkResult VKAPI_CALL NGLayer_CreateInstance(const VkInstanceCreateInfo*  pCreateInfo,
                                                           const VkAllocationCallbacks* pAllocator,
                                                           VkInstance*                  pInstance)
{
    LAYER_LOGI("CreateInstance enter: pCreateInfo=%p, pNext=%p", (void*)pCreateInfo, pCreateInfo->pNext);

    VkLayerInstanceCreateInfo* createInfo = (VkLayerInstanceCreateInfo*)pCreateInfo->pNext;

    while (createInfo && (createInfo->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO || createInfo->function != VK_LAYER_LINK_INFO))
    {
        createInfo = (VkLayerInstanceCreateInfo*)createInfo->pNext;
    }

    if (createInfo == NULL)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    PFN_vkGetInstanceProcAddr gpa = createInfo->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    createInfo->u.pLayerInfo      = createInfo->u.pLayerInfo->pNext;

    PFN_vkCreateInstance createFunc = (PFN_vkCreateInstance)gpa(VK_NULL_HANDLE, "vkCreateInstance");
    if (!createFunc)
    {
        LAYER_LOGE("Failed to resolve vkCreateInstance");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkResult ret = createFunc(pCreateInfo, pAllocator, pInstance);
    if (ret != VK_SUCCESS)
    {
        return ret;
    }

    VkLayerInstanceDispatchTable dispatchTable;
    dispatchTable.GetInstanceProcAddr                = (PFN_vkGetInstanceProcAddr)gpa(*pInstance, "vkGetInstanceProcAddr");
    dispatchTable.CreateDevice                       = (PFN_vkCreateDevice)gpa(*pInstance, "vkCreateDevice");
    dispatchTable.DestroyInstance                    = (PFN_vkDestroyInstance)gpa(*pInstance, "vkDestroyInstance");
    dispatchTable.EnumerateDeviceExtensionProperties = (PFN_vkEnumerateDeviceExtensionProperties)gpa(*pInstance, "vkEnumerateDeviceExtensionProperties");
    dispatchTable.GetPhysicalDeviceProperties        = (PFN_vkGetPhysicalDeviceProperties)gpa(*pInstance, "vkGetPhysicalDeviceProperties");
    dispatchTable.GetPhysicalDeviceFeatures2         = (PFN_vkGetPhysicalDeviceFeatures2)gpa(*pInstance, "vkGetPhysicalDeviceFeatures2");
    dispatchTable.GetPhysicalDeviceFeatures2KHR      = (PFN_vkGetPhysicalDeviceFeatures2KHR)gpa(*pInstance, "vkGetPhysicalDeviceFeatures2KHR");

    {
        ScopedLock l(g_Lock);
        g_instanceDispatchTable[GetKey(*pInstance)] = dispatchTable;
    }

    return ret;
}

VK_LAYER_EXPORT void VKAPI_CALL NGLayer_DestroyInstance(VkInstance instance, const VkAllocationCallbacks* pAllocator)
{
    PFN_vkDestroyInstance destroy = nullptr;
    {
        ScopedLock l(g_Lock);
        auto       it = g_instanceDispatchTable.find(GetKey(instance));
        if (it != g_instanceDispatchTable.end())
        {
            destroy = it->second.DestroyInstance;
            g_instanceDispatchTable.erase(it);
        }
    }

    if (destroy)
    {
        destroy(instance, pAllocator);
    }
}

static bool HasExtension(const std::vector<VkExtensionProperties>& extensions, const char* name)
{
    for (auto const& extension : extensions)
        if (strcmp(extension.extensionName, name) == 0)
            return true;
    return false;
}

static bool HasExtension(const std::vector<const char*>& extensions, const char* name)
{
    for (const char* extensionName : extensions)
        if (extensionName && strcmp(extensionName, name) == 0)
            return true;
    return false;
}

static const VkBaseInStructure* FindTypedStructure(const void* pNext, VkStructureType target)
{
    for (const VkBaseInStructure* it = reinterpret_cast<const VkBaseInStructure*>(pNext); it != nullptr; it = it->pNext)
    {
        if (it->sType == target)
            return it;
    }

    return nullptr;
}

/**
 * @brief Query which of the layer-known features/extensions the device exposes.
 *
 * @param physicalDevice Target physical device.
 * @param table Instance dispatch table used for downstream Vulkan calls.
 * @param deviceApiVersion VkPhysicalDeviceProperties::apiVersion of the device.
 * @return ExtensionFlags  Per-feature device-support flags.
 */
static ExtensionFlags AreExtensionsSupported(VkPhysicalDevice physicalDevice, const VkLayerInstanceDispatchTable& table, uint32_t deviceApiVersion)
{
    NGFeatureChainNodes features;
    features.LinkForQuery();

    if (table.GetPhysicalDeviceFeatures2)
    {
        table.GetPhysicalDeviceFeatures2(physicalDevice, &features.features2);
    }
    else if (table.GetPhysicalDeviceFeatures2KHR)
    {
        table.GetPhysicalDeviceFeatures2KHR(physicalDevice, &features.features2);
    }

    std::vector<VkExtensionProperties> extensions;
    if (table.EnumerateDeviceExtensionProperties)
    {
        uint32_t count = 0;
        table.EnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr);
        extensions.resize(count);
        table.EnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, extensions.data());
    }

    ExtensionFlags out{};
    out.tensors =
        HasExtension(extensions, VK_ARM_TENSORS_EXTENSION_NAME) && features.tensor.tensors == VK_TRUE && features.tensor.shaderTensorAccess == VK_TRUE;
    out.data_graph = HasExtension(extensions, VK_ARM_DATA_GRAPH_EXTENSION_NAME) && features.graph.dataGraph == VK_TRUE;
    out.data_graph_optical_flow =
        HasExtension(extensions, VK_ARM_DATA_GRAPH_OPTICAL_FLOW_EXTENSION_NAME) && features.opticalFlow.dataGraphOpticalFlow == VK_TRUE;
    out.sync2                    = (deviceApiVersion >= VK_API_VERSION_1_3) || HasExtension(extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    out.extended_flags           = HasExtension(extensions, kExtendedFlagsExtName);
    out.maintenance5             = HasExtension(extensions, VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    out.deferred_host_operations = HasExtension(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    return out;
}

/**
 * @brief Has the application already provided each known feature/extension?
 *
 * For feature-struct kinds (tensors, data_graph): true iff the
 * matching `VkStructureType` is present in the incoming pNext chain.
 * 
 * For sync2 we accept either `VkPhysicalDeviceSynchronization2Features` or
 * `VkPhysicalDeviceVulkan13Features` (the latter subsumes sync2; per
 * VUID-VkDeviceCreateInfo-pNext-06532 the two structs cannot coexist).
 * 
 * For extension-name kinds (extended_flags, maintenance5,
 * deferred_host_operations): true iff the name is already in
 * ppEnabledExtensionNames.
 */
static ExtensionFlags AreExtensionsProvidedByApp(const VkDeviceCreateInfo* pCreateInfo)
{
    ExtensionFlags out{};

    // Feature-struct kinds: scan the whole pNext chain. Vulkan allows feature
    // structs to live either under VkPhysicalDeviceFeatures2 or directly off
    // VkDeviceCreateInfo.pNext, so anchoring the search under features2 only
    // would miss valid placements and let us emit a duplicate sType.
    out.tensors                 = FindTypedStructure(pCreateInfo->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM) != nullptr;
    out.data_graph              = FindTypedStructure(pCreateInfo->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM) != nullptr;
    out.data_graph_optical_flow = FindTypedStructure(pCreateInfo->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_OPTICAL_FLOW_FEATURES_ARM) != nullptr;

    // Tracks mere CARRIER PRESENCE on purpose: when a carrier is present we
    // cannot append a duplicate sType (single-sType rule + VUID-06532 forbidding
    // Sync2Features alongside Vulkan13Features), so even a carrier with the bit
    // set to VK_FALSE must be treated as "provided" by the inject decision.
    // The actual bit is forced to VK_TRUE separately by ForceEnableSync2InAppChain.
    // That's why we still mark it provided even when the carrier's bit is VK_FALSE.
    out.sync2 = FindTypedStructure(pCreateInfo->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES) != nullptr ||
                FindTypedStructure(pCreateInfo->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES) != nullptr;

    // Extension-name kinds.
    const std::vector<const char*> extensions(pCreateInfo->ppEnabledExtensionNames, pCreateInfo->ppEnabledExtensionNames + pCreateInfo->enabledExtensionCount);
    out.extended_flags           = HasExtension(extensions, kExtendedFlagsExtName);
    out.maintenance5             = HasExtension(extensions, VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    out.deferred_host_operations = HasExtension(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    return out;
}

// When the app already has a sync2 carrier struct in the pNext chain, ensure
// its synchronization2 bit is VK_TRUE.
//
// Pre-condition: at least one sync2 carrier struct is in the chain.
static void ForceEnableSync2InAppChain(const VkDeviceCreateInfo* pCreateInfo)
{
    // The const cast here should be safe because we only mutate the synchronization2 bits of existing carrier structs.
    // We also throw an warning later.
    auto* sync2 = const_cast<VkPhysicalDeviceSynchronization2Features*>(reinterpret_cast<const VkPhysicalDeviceSynchronization2Features*>(
        FindTypedStructure(pCreateInfo->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES)));
    auto* v13   = const_cast<VkPhysicalDeviceVulkan13Features*>(reinterpret_cast<const VkPhysicalDeviceVulkan13Features*>(
        FindTypedStructure(pCreateInfo->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)));

    if (sync2 && sync2->synchronization2 != VK_TRUE)
    {
        LAYER_LOGE("CreateDevice: app-provided VkPhysicalDeviceSynchronization2Features had synchronization2=VK_FALSE; forcing to VK_TRUE for ARM ML paths");
        sync2->synchronization2 = VK_TRUE;
    }

    // There should be only one path reached.
    if (v13 && v13->synchronization2 != VK_TRUE)
    {
        LAYER_LOGE("CreateDevice: app-provided VkPhysicalDeviceVulkan13Features had synchronization2=VK_FALSE; forcing to VK_TRUE for ARM ML paths");
        v13->synchronization2 = VK_TRUE;
    }
}

/**
 * @brief Decide whether ARM feature structs and the spec-required dependency
 *        extension names should be injected into vkCreateDevice.
 *
 * VK_ARM_data_graph "Extension and Version Dependencies" require:
 *   - Vulkan 1.3
 *   - VK_KHR_extended_flags OR VK_KHR_maintenance5
 *   - VK_KHR_deferred_host_operations
 *
 * If the device's apiVersion is < 1.3 the spec is not satisfied even if the
 * device advertises the ARM extensions, so this function returns all-false
 * and CreateDevice falls back to passthrough.
 *
 * tensor / data_graph: injected when both ARM features are device-supported
 * and the matching feature struct is not already in the incoming pNext chain.
 * The Vulkan pNext chain is a flat singly-linked list; tensor / data_graph /
 * sync2 each carry a unique sType and may live anywhere in the chain (with
 * or without a sibling VkPhysicalDeviceFeatures2). CreateDevice splices our
 * standalone nodes onto pCreateInfo->pNext directly when the app already
 * supplied a features2, and only prepends its own features2 (to absorb
 * pEnabledFeatures) when the app did not.
 *
 * sync2 / extended_flags / maintenance5 / deferred_host_operations: injected when at
 * least one ARM feature is being injected, the device advertises the extension,
 * and the application has not already enabled it via ppEnabledExtensionNames.
 * Between extended_flags and maintenance5 we prefer maintenance5 (broader,
 * widely deployed) and only fall back to extended_flags when only the latter
 * is available; if the app already enables either, no injection happens.
 *
 * @param pCreateInfo  Incoming VkDeviceCreateInfo.
 * @param physicalDevice  Target physical device.
 * @param table  Instance dispatch table for feature queries.
 * @param deviceApiVersion  VkPhysicalDeviceProperties::apiVersion of the device.
 * @return ExtensionFlags  Injection flags.
 */
static ExtensionFlags DecideInjection(const VkDeviceCreateInfo*           pCreateInfo,
                                      VkPhysicalDevice                    physicalDevice,
                                      const VkLayerInstanceDispatchTable& table,
                                      uint32_t                            deviceApiVersion)
{
    ExtensionFlags out{};

    // Hard spec gate: VK_ARM_data_graph requires Vulkan 1.3.
    if (deviceApiVersion < VK_API_VERSION_1_3)
    {
        LAYER_LOGI("CreateDevice: device apiVersion=0x%08x < 1.3; skip ARM injection per VK_ARM_data_graph dep", deviceApiVersion);
        return out;
    }

    const ExtensionFlags supported = AreExtensionsSupported(physicalDevice, table, deviceApiVersion);
    const ExtensionFlags provided  = AreExtensionsProvidedByApp(pCreateInfo);

    // Keep passthrough behavior on unsupported platforms: if either ARM feature
    // is unsupported, do not add any ARM extension/feature chain nodes.
    if (!supported.tensors || !supported.data_graph)
    {
        return out;
    }

    // Do not inject duplicates if the feature structs are already in the chain.
    out.tensors    = !provided.tensors;
    out.data_graph = !provided.data_graph;

    // Optional advanced ARM features: only inject when the device supports them
    // AND the app has not already supplied the feature struct. Not injecting
    // them does not affect the base tensor/data_graph path.
    out.data_graph_optical_flow = supported.data_graph_optical_flow && !provided.data_graph_optical_flow;

    if (!out.AnyArmFeature())
    {
        return out;
    }

    // Interaction feature: synchronization2.
    // - No carrier struct in chain: ask BuildEnabledFeaturesChain to inject
    //   a fresh VkPhysicalDeviceSynchronization2Features (bit=VK_TRUE).
    // - Carrier struct already in chain: ensure
    //   the existing bit is VK_TRUE in place (warn on flip).
    if (supported.sync2)
    {
        if (provided.sync2)
        {
            ForceEnableSync2InAppChain(pCreateInfo);
        }
        else
        {
            out.sync2 = true;
        }
    }

    // Spec-required dependency extension names. Prefer maintenance5 over
    // extended_flags (the spec allows either, maintenance5 is broader and more
    // widely deployed). If the app already enables either one, skip both.
    if (!provided.maintenance5 && !provided.extended_flags)
    {
        if (supported.maintenance5)
        {
            out.maintenance5 = true;
        }
        else if (supported.extended_flags)
        {
            out.extended_flags = true;
        }
        else
        {
            LAYER_LOGE("CreateDevice: device supports neither VK_KHR_maintenance5 nor VK_KHR_extended_flags; VK_ARM_data_graph dep unsatisfiable");
        }
    }

    out.deferred_host_operations = supported.deferred_host_operations && !provided.deferred_host_operations;
    if (!supported.deferred_host_operations && !provided.deferred_host_operations)
    {
        LAYER_LOGE("CreateDevice: device does not advertise VK_KHR_deferred_host_operations; VK_ARM_data_graph dep unsatisfiable");
    }

    return out;
}

/**
 * @brief Build the new pNext chain head used for the cloned VkDeviceCreateInfo.
 *
 * The Vulkan pNext chain is a flat singly-linked list: every feature struct
 * carries a unique sType and the loader/driver scans the whole list. There
 * is no requirement that ARM feature structs sit "under" features2.
 *
 * Two cases:
 *  - `hasFeatures2 == false`: the layer prepends its own local features2
 *    (which the caller has populated from `pCreateInfo->pEnabledFeatures`),
 *    then prepends each requested ARM feature struct on top. The caller is
 *    responsible for clearing `pEnabledFeatures` on the cloned create-info
 *    afterwards (since features2 supersedes it).
 *  - `hasFeatures2 == true`: the app already owns the features2 in its
 *    pNext chain. We must NOT add a second one (single-sType rule), so we
 *    simply prepend the requested ARM feature structs onto pCreateInfo->pNext
 *    directly. The app's features2 + (per spec VUID-pNext-00373) NULL
 *    `pEnabledFeatures` are preserved untouched.
 *
 * Spec dependency extensions (extended_flags / maintenance5 /
 * deferred_host_operations) are extension-name only and are appended to
 * ppEnabledExtensionNames in CreateDevice; they do not need feature structs
 * here.
 *
 * @param originalPNext Original device-create pNext chain head (read-only).
 * @param inject        Requested injection flags.
 * @param hasFeatures2  True iff the app already has a features2 in its chain.
 * @param nodes         Local storage for feature structs used to build the chain.
 * @return const void*  Pointer to the new pNext head.
 */
static const void* BuildEnabledFeaturesChain(const void* originalPNext, const ExtensionFlags& inject, bool hasFeatures2, NGFeatureChainNodes& nodes)
{
    assert(inject.AnyArmFeature() || inject.sync2);

    // Start with the original chain as the tail; the const_cast is safe
    // because we never mutate any node we did not allocate ourselves.
    VkBaseOutStructure* head = reinterpret_cast<VkBaseOutStructure*>(const_cast<void*>(originalPNext));

    auto prepend = [&head](VkBaseOutStructure* node) {
        node->pNext = head;
        head        = node;
    };

    if (inject.tensors)
    {
        nodes.tensor.tensors            = VK_TRUE;
        nodes.tensor.shaderTensorAccess = VK_TRUE;
        prepend(reinterpret_cast<VkBaseOutStructure*>(&nodes.tensor));
        LAYER_LOGI("CreateDevice: injecting VkPhysicalDeviceTensorFeaturesARM");
    }

    if (inject.data_graph)
    {
        nodes.graph.dataGraph = VK_TRUE;
        prepend(reinterpret_cast<VkBaseOutStructure*>(&nodes.graph));
        LAYER_LOGI("CreateDevice: injecting VkPhysicalDeviceDataGraphFeaturesARM");
    }

    if (inject.data_graph_optical_flow)
    {
        nodes.opticalFlow.dataGraphOpticalFlow = VK_TRUE;
        prepend(reinterpret_cast<VkBaseOutStructure*>(&nodes.opticalFlow));
        LAYER_LOGI("CreateDevice: injecting VkPhysicalDeviceDataGraphOpticalFlowFeaturesARM");
    }

    if (inject.sync2)
    {
        nodes.sync2.synchronization2 = VK_TRUE;
        prepend(reinterpret_cast<VkBaseOutStructure*>(&nodes.sync2));
        LAYER_LOGI("CreateDevice: injecting VkPhysicalDeviceSynchronization2Features");
    }

    // Only prepend our own features2 head when the app didn't already supply
    // one. Single-sType rule forbids two features2 in the same chain.
    if (!hasFeatures2)
    {
        nodes.features2.pNext = head;
        head                  = reinterpret_cast<VkBaseOutStructure*>(&nodes.features2);
    }

    return head;
}

VK_LAYER_EXPORT VkResult VKAPI_CALL NGLayer_CreateDevice(VkPhysicalDevice             physicalDevice,
                                                         const VkDeviceCreateInfo*    pCreateInfo,
                                                         const VkAllocationCallbacks* pAllocator,
                                                         VkDevice*                    pDevice)
{
    LAYER_LOGI("CreateDevice enter: phys=%p, pNext=%p, extCount=%u", (void*)physicalDevice, pCreateInfo->pNext, pCreateInfo->enabledExtensionCount);

    VkLayerDeviceCreateInfo* createInfo = (VkLayerDeviceCreateInfo*)pCreateInfo->pNext;

    while (createInfo && (createInfo->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO || createInfo->function != VK_LAYER_LINK_INFO))
    {
        createInfo = (VkLayerDeviceCreateInfo*)createInfo->pNext;
    }

    if (createInfo == NULL)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    PFN_vkGetDeviceProcAddr gdpa = createInfo->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    createInfo->u.pLayerInfo     = createInfo->u.pLayerInfo->pNext;  // move to next layer

    VkLayerInstanceDispatchTable idt;
    {
        ScopedLock l(g_Lock);
        auto       it = g_instanceDispatchTable.find(GetKey(physicalDevice));
        if (it == g_instanceDispatchTable.end())
        {
            LAYER_LOGE("No instance dispatch table for physicalDevice=%p", (void*)physicalDevice);
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        idt = it->second;
    }

    // Query device apiVersion once: needed below to gate ARM injection per
    // VK_ARM_data_graph spec dependency on Vulkan 1.3.
    VkPhysicalDeviceProperties deviceProps{};
    if (idt.GetPhysicalDeviceProperties)
    {
        idt.GetPhysicalDeviceProperties(physicalDevice, &deviceProps);
    }
    const uint32_t deviceApiVersion = deviceProps.apiVersion;

    auto injectIt = [](bool flag, const char* name, std::vector<const char*>& list) {
        if (flag && !HasExtension(list, name))
        {
            list.push_back(name);
            LAYER_LOGI("CreateDevice: will inject %s into enabled extensions list", name);
        }
    };

    std::vector<const char*> extensions(pCreateInfo->ppEnabledExtensionNames, pCreateInfo->ppEnabledExtensionNames + pCreateInfo->enabledExtensionCount);
    ExtensionFlags           inject = DecideInjection(pCreateInfo, physicalDevice, idt, deviceApiVersion);
    injectIt(inject.tensors, VK_ARM_TENSORS_EXTENSION_NAME, extensions);
    injectIt(inject.data_graph, VK_ARM_DATA_GRAPH_EXTENSION_NAME, extensions);
    injectIt(inject.data_graph_optical_flow, VK_ARM_DATA_GRAPH_OPTICAL_FLOW_EXTENSION_NAME, extensions);
    // Need not to inject VK_KHR_synchronization2 because we hard-gate to deviceApiVersion >= 1.3 above, where sync2 is core and does not require an extension name.

    // DecideInjection has already chosen at most one of the (extended_flags, maintenance5)
    // pair (preferring maintenance5) and only set the bits the device supports.
    injectIt(inject.maintenance5, VK_KHR_MAINTENANCE_5_EXTENSION_NAME, extensions);
    injectIt(inject.extended_flags, kExtendedFlagsExtName, extensions);
    injectIt(inject.deferred_host_operations, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, extensions);

    PFN_vkCreateDevice createFunc = idt.CreateDevice;
    if (!createFunc)
    {
        LAYER_LOGE("Failed to resolve vkCreateDevice from instance dispatch table");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // If no ARM extension/feature mutation is requested, preserve app behavior by
    // passing VkDeviceCreateInfo through untouched.
    if (!inject.AnyMutation())
    {
        LAYER_LOGI("CreateDevice passthrough: no ARM extension or feature mutation");
        VkResult ret = createFunc(physicalDevice, pCreateInfo, pAllocator, pDevice);
        if (ret != VK_SUCCESS)
        {
            LAYER_LOGE("Failed to create device");
            return ret;
        }

        VkLayerDispatchTable table;
        table.GetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)gdpa(*pDevice, "vkGetDeviceProcAddr");
        table.DestroyDevice     = (PFN_vkDestroyDevice)gdpa(*pDevice, "vkDestroyDevice");

        {
            ScopedLock l(g_Lock);
            g_deviceDispatchTable[GetKey(*pDevice)] = table;
        }
        return VK_SUCCESS;
    }

    // Build a local feature chain that can inject requested ARM feature structs.
    // If the app already owns a features2 in its pNext chain we must not add
    // another one (single-sType rule); we then splice our standalone nodes
    // onto pCreateInfo->pNext directly and leave pEnabledFeatures alone
    // (per VUID-VkDeviceCreateInfo-pNext-00373 it is required to be NULL
    // whenever a features2 is in pNext).
    const bool hasFeatures2 = FindTypedStructure(pCreateInfo->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) != nullptr;

    NGFeatureChainNodes createFeatures;
    if (!hasFeatures2 && pCreateInfo->pEnabledFeatures != nullptr)
    {
        createFeatures.features2.features = *pCreateInfo->pEnabledFeatures;
    }

    const void* head = BuildEnabledFeaturesChain(pCreateInfo->pNext, inject, hasFeatures2, createFeatures);

    VkDeviceCreateInfo cloned      = *pCreateInfo;
    cloned.pNext                   = head;
    cloned.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    cloned.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();
    if (!hasFeatures2)
    {
        // pEnabledFeatures was absorbed into our local features2 head.
        cloned.pEnabledFeatures = nullptr;
    }
    // else: app's features2 dictates feature enablement; pEnabledFeatures
    //       is required to be NULL by VUID-pNext-00373 in that case, so
    //       keep it untouched.

    VkResult ret = createFunc(physicalDevice, &cloned, pAllocator, pDevice);
    if (ret != VK_SUCCESS)
    {
        LAYER_LOGE("Failed to create device");
        return ret;
    }

    VkLayerDispatchTable table;
    table.GetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)gdpa(*pDevice, "vkGetDeviceProcAddr");
    table.DestroyDevice     = (PFN_vkDestroyDevice)gdpa(*pDevice, "vkDestroyDevice");

    {
        ScopedLock l(g_Lock);
        g_deviceDispatchTable[GetKey(*pDevice)] = table;
    }

    return ret;
}

VK_LAYER_EXPORT void VKAPI_CALL NGLayer_DestroyDevice(VkDevice device, const VkAllocationCallbacks* pAllocator)
{
    PFN_vkDestroyDevice destroy = nullptr;
    {
        ScopedLock l(g_Lock);
        auto       it = g_deviceDispatchTable.find(GetKey(device));

        if (it != g_deviceDispatchTable.end())
        {
            destroy = it->second.DestroyDevice;
            g_deviceDispatchTable.erase(it);
        }
    }

    if (destroy)
    {
        destroy(device, pAllocator);
    }
}

VK_LAYER_EXPORT VkResult VKAPI_CALL NGLayer_EnumerateInstanceLayerProperties(uint32_t* pPropertyCount, VkLayerProperties* pProperties)
{
    if (pPropertyCount)
        *pPropertyCount = 1;

    if (pProperties)
    {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)  // 'strncpy': MSVC-only deprecation; usage here is bounded + explicitly null-terminated.
#endif
        std::strncpy(pProperties->layerName, kLayerName, sizeof(pProperties->layerName) - 1);
        pProperties->layerName[sizeof(pProperties->layerName) - 1] = '\0';
        std::strncpy(pProperties->description, "Arm Neural Graphics Vulkan layer", sizeof(pProperties->description) - 1);
        pProperties->description[sizeof(pProperties->description) - 1] = '\0';
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
        pProperties->implementationVersion = 1;
        pProperties->specVersion           = VK_API_VERSION_1_0;
    }

    return VK_SUCCESS;
}

VK_LAYER_EXPORT VkResult VKAPI_CALL NGLayer_EnumerateDeviceLayerProperties(VkPhysicalDevice   physicalDevice,
                                                                           uint32_t*          pPropertyCount,
                                                                           VkLayerProperties* pProperties)
{
    return NGLayer_EnumerateInstanceLayerProperties(pPropertyCount, pProperties);
}

VK_LAYER_EXPORT VkResult VKAPI_CALL NGLayer_EnumerateInstanceExtensionProperties(const char*            pLayerName,
                                                                                 uint32_t*              pPropertyCount,
                                                                                 VkExtensionProperties* pProperties)
{
    if (pLayerName && strcmp(pLayerName, kLayerName) != 0)
    {
        return VK_ERROR_LAYER_NOT_PRESENT;
    }

    // If the pLayerName is NULL, it means a global query for all extensions.
    // If it's not NULL and matches our layer name, it means a query for our layer's extensions.
    if (pPropertyCount)
        *pPropertyCount = 0;

    return VK_SUCCESS;
}

VK_LAYER_EXPORT VkResult VKAPI_CALL NGLayer_EnumerateDeviceExtensionProperties(VkPhysicalDevice       physicalDevice,
                                                                               const char*            pLayerName,
                                                                               uint32_t*              pPropertyCount,
                                                                               VkExtensionProperties* pProperties)
{
    // Return empty for two cases:
    //   - own-layer query: we expose no device extensions.
    //   - VK_NULL_HANDLE: no valid device to query regardless of pLayerName.
    if (physicalDevice == VK_NULL_HANDLE || (pLayerName != nullptr && strcmp(pLayerName, kLayerName) == 0))
    {
        if (pPropertyCount)
            *pPropertyCount = 0;
        return VK_SUCCESS;
    }

    PFN_vkEnumerateDeviceExtensionProperties enumerateDeviceExtensionProperties = nullptr;
    {
        ScopedLock l(g_Lock);
        auto       it = g_instanceDispatchTable.find(GetKey(physicalDevice));
        if (it == g_instanceDispatchTable.end() || !it->second.EnumerateDeviceExtensionProperties)
            return VK_ERROR_INITIALIZATION_FAILED;
        enumerateDeviceExtensionProperties = it->second.EnumerateDeviceExtensionProperties;
    }
    return enumerateDeviceExtensionProperties(physicalDevice, pLayerName, pPropertyCount, pProperties);
}

#define GETPROCADDR(func)           \
    if (!strcmp(pName, "vk" #func)) \
        return (PFN_vkVoidFunction)&NGLayer_##func;
#define GETPROCADDR_INSTANCE(func)                  \
    if (!strcmp(pName, "vk" #func))                 \
    {                                               \
        return (PFN_vkVoidFunction)&NGLayer_##func; \
    }

VK_LAYER_EXPORT PFN_vkVoidFunction VKAPI_CALL NGLayer_GetDeviceProcAddr(VkDevice device, const char* pName)
{
    GETPROCADDR(GetDeviceProcAddr);
    GETPROCADDR(EnumerateDeviceLayerProperties);
    GETPROCADDR(EnumerateDeviceExtensionProperties);
    GETPROCADDR(CreateDevice);
    GETPROCADDR(DestroyDevice);

    if (device == VK_NULL_HANDLE)
    {
        LAYER_LOGE("GetDeviceProcAddr called with VK_NULL_HANDLE device");
        return nullptr;
    }

    PFN_vkGetDeviceProcAddr getDeviceProcAddr = nullptr;
    {
        ScopedLock l(g_Lock);
        auto       it = g_deviceDispatchTable.find(GetKey(device));
        if (it == g_deviceDispatchTable.end() || !it->second.GetDeviceProcAddr)
            return nullptr;
        getDeviceProcAddr = it->second.GetDeviceProcAddr;
    }
    return getDeviceProcAddr(device, pName);
}

VK_LAYER_EXPORT PFN_vkVoidFunction VKAPI_CALL NGLayer_GetInstanceProcAddr(VkInstance instance, const char* pName)
{
    // instance chain functions we intercept
    GETPROCADDR_INSTANCE(GetInstanceProcAddr);
    GETPROCADDR_INSTANCE(EnumerateInstanceLayerProperties);
    GETPROCADDR_INSTANCE(EnumerateInstanceExtensionProperties);
    GETPROCADDR_INSTANCE(CreateInstance);
    GETPROCADDR_INSTANCE(DestroyInstance);

    // device chain functions we intercept
    GETPROCADDR_INSTANCE(GetDeviceProcAddr);
    GETPROCADDR_INSTANCE(EnumerateDeviceLayerProperties);
    GETPROCADDR_INSTANCE(EnumerateDeviceExtensionProperties);
    GETPROCADDR_INSTANCE(CreateDevice);
    GETPROCADDR_INSTANCE(DestroyDevice);

    // when instance == VK_NULL_HANDLE the loader is querying for global-scope entry points.
    // We have already returned every global entry point we own above.
    if (instance == VK_NULL_HANDLE)
        return nullptr;

    PFN_vkGetInstanceProcAddr getInstanceProcAddr = nullptr;
    {
        ScopedLock l(g_Lock);
        auto       it = g_instanceDispatchTable.find(GetKey(instance));
        if (it == g_instanceDispatchTable.end() || !it->second.GetInstanceProcAddr)
            return nullptr;
        getInstanceProcAddr = it->second.GetInstanceProcAddr;
    }
    return getInstanceProcAddr(instance, pName);
}

// Standard loader-visible aliases for Android/desktop loaders that resolve
// layer entrypoints by canonical Vulkan symbol names.
VK_LAYER_EXPORT PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* pName)
{
    return NGLayer_GetInstanceProcAddr(instance, pName);
}

VK_LAYER_EXPORT PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* pName)
{
    return NGLayer_GetDeviceProcAddr(device, pName);
}

VK_LAYER_EXPORT VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t* pPropertyCount, VkLayerProperties* pProperties)
{
    return NGLayer_EnumerateInstanceLayerProperties(pPropertyCount, pProperties);
}

VK_LAYER_EXPORT VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkLayerProperties* pProperties)
{
    return NGLayer_EnumerateDeviceLayerProperties(physicalDevice, pPropertyCount, pProperties);
}

VK_LAYER_EXPORT VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char* pLayerName, uint32_t* pPropertyCount, VkExtensionProperties* pProperties)
{
    return NGLayer_EnumerateInstanceExtensionProperties(pLayerName, pPropertyCount, pProperties);
}

VK_LAYER_EXPORT VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice       physicalDevice,
                                                                         const char*            pLayerName,
                                                                         uint32_t*              pPropertyCount,
                                                                         VkExtensionProperties* pProperties)
{
    return NGLayer_EnumerateDeviceExtensionProperties(physicalDevice, pLayerName, pPropertyCount, pProperties);
}

// On Windows/MSVC, vk_layer.h already declares vkNegotiateLoaderLayerInterfaceVersion
// inside an extern "C" block without __declspec(dllexport). Repeating extern "C"
// with __declspec(dllexport) in the definition (via VK_LAYER_EXPORT) causes
// C2375 "redefinition; different linkage". Export the C-linkage symbol via a
// linker pragma instead; on non-Windows platforms use the normal VK_LAYER_EXPORT.
#if defined(_WIN32)
#pragma comment(linker, "/EXPORT:vkNegotiateLoaderLayerInterfaceVersion")
VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* pVersionStruct)
#else
VK_LAYER_EXPORT VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* pVersionStruct)
#endif
{
    if (pVersionStruct == nullptr)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    if (pVersionStruct->loaderLayerInterfaceVersion < 2)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    if (pVersionStruct->loaderLayerInterfaceVersion > 2)
    {
        pVersionStruct->loaderLayerInterfaceVersion = 2;
    }

    pVersionStruct->pfnGetInstanceProcAddr       = NGLayer_GetInstanceProcAddr;
    pVersionStruct->pfnGetDeviceProcAddr         = NGLayer_GetDeviceProcAddr;
    pVersionStruct->pfnGetPhysicalDeviceProcAddr = nullptr;

    LAYER_LOGI("Negotiated loader-layer interface version=%u", pVersionStruct->loaderLayerInterfaceVersion);
    return VK_SUCCESS;
}
