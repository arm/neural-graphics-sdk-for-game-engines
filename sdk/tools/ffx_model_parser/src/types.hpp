/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */
#pragma once

//#include "nlohmann/json.hpp"

//#include "vulkan/vulkan_raii.hpp"

//#include <variant>

#include <vector>
#include <string>
#include <optional>

#include <vulkan/vulkan.h>

//using json = nlohmann::json;

namespace arm
{
    enum class ModuleType
    {
        SHADER,
        GRAPH
    };

    /// \brief Structure that describes 1-dimensional buffer data
    ///
    /// \note We don't account for any specialized meta-data like
    /// for example padding or stride information. We assume that
    /// the provided data are packed in a linear manner
    struct BufferInfo
    {
        uint32_t size;
    };

    /// \brief Structure that describes N-dimensional data
    ///
    /// \note We don't account for any specialized meta-data like
    /// for example padding or stride information. We assume that
    /// the provided data are packed in a linear manner
    struct TensorInfo
    {
        std::vector<uint64_t> shape;
        VkFormat              format;
        bool                  isAliased{false};
        int64_t               sparsityDimension;
    };

    struct ConstantsInfo
    {
        int                         constantIdx;
        TensorInfo                  tensorInfo;
        mlsdk_decoder_constant_data constantData;
    };

    struct BindingDesc
    {
        BindingDesc() = default;
        BindingDesc(int set, int id)
            : set(set)
            , id(id)
        {
        }

        int set;
        int id;
        // Guid resourceRef;
        std::optional<int> lod = std::nullopt;
    };

    struct ResourceInfo
    {
        ResourceInfo() = default;
        ResourceInfo(std::string name, int set, int id, int format, mlsdk_decoder_tensor_dimensions dims)
            : name(name)
            , set(set)
            , id(id)
            , format(format)
            , dims(dims)
        {
        }

        std::string                     name;
        int                             set;
        int                             id;
        int                             format;
        mlsdk_decoder_tensor_dimensions dims;
    };

    struct VgfModule
    {
        VgfModule(std::string entryPoint, std::vector<uint8_t> spv, ModuleType type, std::vector<BindingDesc> bindings)
            : entryPoint(entryPoint)
            , spv(spv)
            , type(type)
            , bindings(bindings)
        {
        }
        VgfModule(std::string entryPoint, std::string moduleName, std::vector<uint32_t> dispatchShape, ModuleType type, std::vector<BindingDesc> bindings)
            : entryPoint(entryPoint)
            , moduleName(moduleName)
            , dispatchShape(dispatchShape)
            , type(type)
            , bindings(bindings)
        {
        }
        VgfModule(std::string              entryPoint,
                  std::string              moduleName,
                  std::vector<uint8_t>     spv,
                  std::vector<uint32_t>    dispatchShape,
                  ModuleType               type,
                  std::vector<BindingDesc> bindings)
            : entryPoint(entryPoint)
            , moduleName(moduleName)
            , spv(spv)
            , dispatchShape(dispatchShape)
            , type(type)
            , bindings(bindings)
        {
        }

        VgfModule() = default;

        std::string                                                      entryPoint;
        std::string                                                      moduleName;
        std::vector<uint8_t>                                             spv;
        std::vector<uint32_t>                                            dispatchShape;
        std::vector<std::tuple<uint32_t, TensorInfo, std::vector<char>>> constants;
        ModuleType                                                       type;
        std::vector<BindingDesc>                                         bindings;
    };

    /// \brief Constant structure
    ///
    /// Used for push and specialization constants
    union Constant
    {
        int32_t  i;   ///< Signed 32-bit integer value
        uint32_t ui;  ///< Unsigned 32-bit integer value
        float    f;   ///< 32-bit floating point value
    };

}  // namespace arm
