/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "decoder.h"
#include "types.hpp"

#if defined(_WIN32)
#include <windows.h>
#include <fileapi.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <algorithm>
#endif

#include <vector>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <filesystem>
#include <locale>
#include <codecvt>
#include <string>

static std::wstring UTF8ToWChar(const std::string& str)
{
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
    return conv.from_bytes(str);
}

namespace arm
{
    class MemoryMap
    {
    public:
        explicit MemoryMap(const std::string& filename)
        {
#ifdef _WIN32
            _hFile = CreateFile(filename.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);

            if (_hFile == INVALID_HANDLE_VALUE)
            {
                throw std::runtime_error("Could not open file " + filename);
            }

            _hMap = CreateFileMapping(_hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (_hMap == INVALID_HANDLE_VALUE)
            {
                throw std::runtime_error("Failed to create file mapping for file " + filename);
            }

            _addr = MapViewOfFile(_hMap, FILE_MAP_READ, 0, 0, 0);
            if (_addr == nullptr)
            {
                throw std::runtime_error("MapViewOfFile failed for file " + filename);
            }
#else
            _fd = open(filename.c_str(), O_RDONLY);
            if (_fd < 0)
            {
                throw std::runtime_error("Could not open file " + filename);
            }
            struct stat st = {};
            if (fstat(_fd, &st) == -1)
            {
                throw std::runtime_error("Could not read attributes of file " + filename);
            }
            size = st.st_size;

            _addr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, _fd, 0);
            if (_addr == MAP_FAILED)
            {  // NOLINT
                throw std::runtime_error("Failed to memory map the file " + filename);
            }
#endif
        }
        MemoryMap(const MemoryMap&) = delete;
        MemoryMap& operator=(const MemoryMap&) = delete;
        MemoryMap(const MemoryMap&&)           = delete;
        MemoryMap& operator=(const MemoryMap&&) = delete;

        ~MemoryMap()
        {
#ifdef _WIN32
            UnmapViewOfFile(_addr);
            CloseHandle(_hMap);
            CloseHandle(_hFile);
#else
            if (_fd > 0)
            {
                munmap(_addr, size);
                close(_fd);
            }
#endif
        }

        const void* ptr(const size_t offset = 0)
        {
            return reinterpret_cast<const void*>(static_cast<char*>(_addr) + offset);
        }

        size_t size;

    private:
#ifdef _WIN32
        HANDLE _hFile;
        HANDLE _hMap;
#else
        int _fd;
#endif
        void* _addr;
    };

    std::vector<uint32_t> getDispatchShape(mlsdk_decoder_model_sequence_decoder* sequenceDecoder)
    {
        mlsdk_decoder_dispatch_shape dispatchShapePtr;
        for (int seqIdx = 0; seqIdx < mlsdk_decoder_get_model_sequence_table_size(sequenceDecoder); seqIdx++)
        {
            if (0 == mlsdk_decoder_model_sequence_get_segment_module_index(sequenceDecoder, seqIdx))
            {
                mlsdk_decoder_model_sequence_get_segment_dispatch_shape(sequenceDecoder, 0, &dispatchShapePtr);
                return {dispatchShapePtr.data[0], dispatchShapePtr.data[1], dispatchShapePtr.data[2]};
            }
        }
        throw std::runtime_error("[Scenario-Runner] Could not find dispatchShape");
    }

    std::vector<BindingDesc> getBindings(mlsdk_decoder_model_sequence_decoder*       sequenceDecoder,
                                         mlsdk_decoder_model_resource_table_decoder* resourceTableDecoder,
                                         int                                         moduleIdx)
    {
        // Get segment binding infos
        std::vector<BindingDesc> bindings{};
        auto                     descSetSize = mlsdk_decoder_model_sequence_get_segment_descriptorset_info_size(sequenceDecoder, moduleIdx);
        // For each segment descriptor set:
        for (int set = 0; set < descSetSize; ++set)
        {
            auto handle = mlsdk_decoder_model_sequence_get_segment_descriptor_binding_slot(sequenceDecoder, moduleIdx, set);
            // For each descriptor set binding:
            for (int slot = 0; slot < mlsdk_decoder_binding_slot_size(sequenceDecoder, handle); ++slot)
            {
                int bindingId = mlsdk_decoder_binding_slot_binding_id(sequenceDecoder, handle, slot);
                int mrtIdx    = mlsdk_decoder_binding_slot_mrt_index(sequenceDecoder, handle, slot);

                std::string guidStr = "Resource_" + std::to_string(bindingId);
                if (mlsdk_decoder_model_resource_table_get_category(resourceTableDecoder, mrtIdx) == mlsdk_decoder_mrt_category_input)
                {
                    guidStr.append("_input");
                }
                else if (mlsdk_decoder_model_resource_table_get_category(resourceTableDecoder, mrtIdx) == mlsdk_decoder_mrt_category_output)
                {
                    guidStr.append("_output");
                }
                else if (mlsdk_decoder_model_resource_table_get_category(resourceTableDecoder, mrtIdx) == mlsdk_decoder_mrt_category_intermediate)
                {
                    guidStr.append("_intermediate");
                }
                else if (mlsdk_decoder_model_resource_table_get_category(resourceTableDecoder, mrtIdx) == mlsdk_decoder_mrt_category_constant)
                {
                    guidStr.append("_constant");
                }
                else
                {
                    throw std::runtime_error("Unknown resource category");
                }

                bindings.emplace_back(BindingDesc(set, bindingId /*, Guid(guidStr)*/));
            }
        }
        return bindings;
    }

    std::vector<ResourceInfo> getResourceInfos(mlsdk_decoder_model_sequence_decoder*       sequenceDecoder,
                                               mlsdk_decoder_model_resource_table_decoder* resourceTableDecoder,
                                               int                                         moduleIdx)
    {
        // Get segment binding infos
        std::vector<ResourceInfo> infos{};
        auto                      descSetSize = mlsdk_decoder_model_sequence_get_segment_descriptorset_info_size(sequenceDecoder, moduleIdx);
        // For each segment descriptor set:
        for (int set = 0; set < descSetSize; ++set)
        {
            auto handle = mlsdk_decoder_model_sequence_get_segment_descriptor_binding_slot(sequenceDecoder, moduleIdx, set);
            // For each descriptor set binding:
            for (int slot = 0; slot < mlsdk_decoder_binding_slot_size(sequenceDecoder, handle); ++slot)
            {
                int bindingId = mlsdk_decoder_binding_slot_binding_id(sequenceDecoder, handle, slot);
                int mrtIdx    = mlsdk_decoder_binding_slot_mrt_index(sequenceDecoder, handle, slot);

                std::string guidStr = "Resource_" + std::to_string(bindingId);
                if (mlsdk_decoder_model_resource_table_get_category(resourceTableDecoder, mrtIdx) == mlsdk_decoder_mrt_category_input)
                {
                    guidStr.append("_input");
                }
                else if (mlsdk_decoder_model_resource_table_get_category(resourceTableDecoder, mrtIdx) == mlsdk_decoder_mrt_category_output)
                {
                    guidStr.append("_output");
                }
                else if (mlsdk_decoder_model_resource_table_get_category(resourceTableDecoder, mrtIdx) == mlsdk_decoder_mrt_category_intermediate)
                {
                    guidStr.append("_intermediate");
                }
                else if (mlsdk_decoder_model_resource_table_get_category(resourceTableDecoder, mrtIdx) == mlsdk_decoder_mrt_category_constant)
                {
                    guidStr.append("_constant");
                }
                else
                {
                    throw std::runtime_error("Unknown resource category");
                }

                mlsdk_decoder_tensor_dimensions dims;
                mlsdk_decoder_model_resource_table_get_tensor_shape(resourceTableDecoder, mrtIdx, &dims);
                mlsdk_vk_format format = mlsdk_decoder_get_vk_format(resourceTableDecoder, mrtIdx);

                infos.emplace_back(ResourceInfo(guidStr, set, bindingId, format, dims));
            }
        }
        return infos;
    }

    void writeConstants(std::vector<ConstantsInfo>& constantInfos,
                        std::vector<std::string>&   constantHeaderFiles,
                        const std::wstring&         outputPath,
                        const std::string&          vgfFileName)
    {
        FILE* fp = NULL;

        std::string fileName = vgfFileName + "_constants";

        constantHeaderFiles.push_back(fileName + ".h");

#if defined(_WIN32)
        const std::wstring wFileName(fileName.begin(), fileName.end());
        std::wstring       outputFile = outputPath;
        outputFile += wFileName + L".h";
        _wfopen_s(&fp, outputFile.c_str(), L"wb");
#else
        std::string outputFile(outputPath.begin(), outputPath.end());
        outputFile += fileName + ".h";
        fp = fopen(outputFile.c_str(), "wb");
        if (fp == nullptr)
        {
            throw std::runtime_error("Could not open file " + outputFile);
        }
#endif

        std::string varName = fileName;

        fprintf(fp, "static const uint32_t g_%s_id[] = { ", varName.c_str());
        for (uint32_t i = 0; i < constantInfos.size(); i++)
        {
            fprintf(fp, " %d,", constantInfos[i].constantIdx);
        }
        fprintf(fp, " };\n\n");

        fprintf(fp, "static const uint32_t g_%s_format[] = { ", varName.c_str());
        for (uint32_t i = 0; i < constantInfos.size(); i++)
        {
            fprintf(fp, " %d,", (int)constantInfos[i].tensorInfo.format);
        }
        fprintf(fp, " };\n\n");

        fprintf(fp, "static const uint32_t g_%s_shape_size[] = {", varName.c_str());
        for (uint32_t i = 0; i < constantInfos.size(); i++)
        {
            fprintf(fp, " %d,", constantInfos[i].tensorInfo.shape.size());
        }
        fprintf(fp, " };\n\n");

        for (uint32_t i = 0; i < constantInfos.size(); i++)
        {
            fprintf(fp, "static const uint64_t g_%s_shape_%d[] = { ", varName.c_str(), i);

            for (int j = 0; j < constantInfos[i].tensorInfo.shape.size(); j++)
            {
                fprintf(fp, " %i,", constantInfos[i].tensorInfo.shape[j]);
            }

            fprintf(fp, " };\n\n");
        }

        fprintf(fp, "static const uint64_t* g_%s_shape[] = { ", varName.c_str());
        for (int i = 0; i < constantInfos.size(); i++)
        {
            fprintf(fp, " g_%s_shape_%d,", varName.c_str(), i);
        }
        fprintf(fp, " };\n\n");

        fprintf(fp, "static const int64_t g_%s_sparsity_dimension[] = {", varName.c_str());
        for (uint32_t i = 0; i < constantInfos.size(); i++)
        {
            fprintf(fp, " %d,", (int)constantInfos[i].tensorInfo.sparsityDimension);
        }
        fprintf(fp, " };\n\n");

        fprintf(fp, "static const uint32_t g_%s_data_size[] = {", varName.c_str());
        for (uint32_t i = 0; i < constantInfos.size(); i++)
        {
            fprintf(fp, " %d,", constantInfos[i].constantData.size);
        }
        fprintf(fp, " };\n\n");

        for (uint32_t i = 0; i < constantInfos.size(); i++)
        {
            fprintf(fp, "static const unsigned char g_%s_data_%d[] = {\n", varName.c_str(), i);

            for (int32_t j = 0; j < constantInfos[i].constantData.size; ++j)
                fprintf(fp,
                        "0x%02x%s",
                        constantInfos[i].constantData.data[j],
                        j == constantInfos[i].constantData.size - 1 ? "" : ((j + 1) % 16 == 0 ? ",\n" : ","));

            fprintf(fp, "\n};\n\n");
        }

        fprintf(fp, "static const unsigned char* g_%s_data[] = { ", varName.c_str());
        for (int i = 0; i < constantInfos.size(); i++)
        {
            fprintf(fp, " g_%s_data_%d,", varName.c_str(), i);
        }
        fprintf(fp, " };\n\n");

        fclose(fp);
    }

    void writeGraph(int                               moduleIdx,
                    const std::string&                entryPoint,
                    const std::vector<unsigned char>& spirv,
                    const std::vector<ResourceInfo>&  resourceInfos,
                    std::vector<std::string>&         graphHeaderFiles,
                    const std::wstring&               outputPath,
                    const std::string&                vgfFileName)
    {
        FILE* fp = NULL;

        std::string fileName = vgfFileName + "_graph_";

        graphHeaderFiles.push_back(fileName + std::to_string(moduleIdx) + ".h");

#if defined(_WIN32)
        const std::wstring wFileName(fileName.begin(), fileName.end());
        std::wstring       outputFile = outputPath;
        outputFile += wFileName + std::to_wstring(moduleIdx) + L".h";
        _wfopen_s(&fp, outputFile.c_str(), L"wb");
#else
        std::string outputFile(outputPath.begin(), outputPath.end());
        outputFile += fileName + std::to_string(moduleIdx) + ".h";
        fp = fopen(outputFile.c_str(), "wb");
        if (fp == nullptr)
        {
            throw std::runtime_error("Could not open file " + outputFile);
        }
#endif

        std::string varName = fileName + std::to_string(moduleIdx);

        fprintf(fp, "static const char g_%s_entry_point[] = \"%s\";\n\n", varName.c_str(), entryPoint.c_str());

        fprintf(fp, "static const uint32_t g_%s_tensor_nums = %d;\n\n", varName.c_str(), resourceInfos.size());

        fprintf(fp, "static const char* g_%s_tensor_names[] = { ", varName.c_str());

        for (int i = 0; i < resourceInfos.size(); i++)
        {
            fprintf(fp, " \"%s\",", resourceInfos[i].name.c_str());
        }

        fprintf(fp, " };\n\n");

        fprintf(fp, "static const uint32_t g_%s_tensor_sets[] = { ", varName.c_str());

        for (int i = 0; i < resourceInfos.size(); i++)
        {
            fprintf(fp, " %d,", resourceInfos[i].set);
        }

        fprintf(fp, " };\n\n");

        fprintf(fp, "static const uint32_t g_%s_tensor_bindings[] = { ", varName.c_str());

        for (int i = 0; i < resourceInfos.size(); i++)
        {
            fprintf(fp, " %d,", resourceInfos[i].id);
        }

        fprintf(fp, " };\n\n");

        fprintf(fp, "static const uint32_t g_%s_tensor_formats[] = { ", varName.c_str());

        for (int i = 0; i < resourceInfos.size(); i++)
        {
            fprintf(fp, " %d,", resourceInfos[i].format);
        }

        fprintf(fp, " };\n\n");

        for (int i = 0; i < resourceInfos.size(); i++)
        {
            fprintf(fp, "static const uint32_t g_%s_tensor_dim_size_%d = %d;\n\n", varName.c_str(), i, resourceInfos[i].dims.size);
        }

        for (int i = 0; i < resourceInfos.size(); i++)
        {
            fprintf(fp, "static const uint64_t g_%s_tensor_dims_%d[] = { ", varName.c_str(), i);
            for (int j = 0; j < resourceInfos[i].dims.size; j++)
            {
                fprintf(fp, " %d,", resourceInfos[i].dims.data[j]);
            }
            fprintf(fp, " };\n\n");
        }

        fprintf(fp, "static const uint32_t g_%s_tensor_dim_size[] = { ", varName.c_str());

        for (int i = 0; i < resourceInfos.size(); i++)
        {
            fprintf(fp, " g_%s_tensor_dim_size_%d,", varName.c_str(), i);
        }

        fprintf(fp, " };\n\n");

        fprintf(fp, "static const uint64_t* g_%s_tensor_dims[] = { ", varName.c_str());

        for (int i = 0; i < resourceInfos.size(); i++)
        {
            fprintf(fp, " g_%s_tensor_dims_%d,", varName.c_str(), i);
        }

        fprintf(fp, " };\n\n");

        fprintf(fp, "static const uint32_t g_%s_data_size = %d;\n\n", varName.c_str(), spirv.size());

        fprintf(fp, "static const unsigned char g_%s_data[] = {\n", varName.c_str());

        for (int32_t i = 0; i < spirv.size(); ++i)
            fprintf(fp, "0x%02x%s", spirv[i], i == spirv.size() - 1 ? "" : ((i + 1) % 16 == 0 ? ",\n" : ","));

        fprintf(fp, "\n};\n\n");

        fclose(fp);
    }

    void writeHeaderFile(const std::vector<std::string>& moduleHeaderFiles,
                         const std::vector<std::string>& constantHeaderFiles,
                         uint32_t                        constantNums,
                         const std::wstring&             outputPath,
                         const std::string&              vgfFileName)
    {
        FILE* fp = NULL;

        const std::string& fileName = vgfFileName;

#if defined(_WIN32) || defined(_WIN64)
        const std::wstring wFileName(fileName.begin(), fileName.end());
        std::wstring       outputFile = outputPath;
        outputFile += wFileName + L".h";
        _wfopen_s(&fp, outputFile.c_str(), L"wb");
#else
        std::string outputFile(outputPath.begin(), outputPath.end());
        outputFile += fileName + ".h";
        fp = fopen(outputFile.c_str(), "wb");
        if (fp == nullptr)
        {
            throw std::runtime_error("Could not open file " + outputFile);
        }
#endif

        for (int i = 0; i < moduleHeaderFiles.size(); i++)
        {
            fprintf(fp, "#include \"%s\"\n", moduleHeaderFiles[i].c_str());
        }

        for (int i = 0; i < constantHeaderFiles.size(); i++)
        {
            fprintf(fp, "#include \"%s\"\n", constantHeaderFiles[i].c_str());
        }

        fprintf(fp, "\n");

        std::string varName = fileName;

        fprintf(fp, "typedef struct %s_Info {\n", varName.c_str());
        fprintf(fp, "    const uint32_t       constantNums;\n");
        fprintf(fp, "    const uint32_t*      constantIds;\n");
        fprintf(fp, "    const uint32_t*      constantFormats;\n");
        fprintf(fp, "    const uint32_t*      constantShapeSize;\n");
        fprintf(fp, "    const uint64_t**     constantShapes;\n");
        fprintf(fp, "    const int64_t*       constantSparsityDimensions;\n");
        fprintf(fp, "    const uint32_t*      constantDataSize;\n");
        fprintf(fp, "    const unsigned char** constantDatas;\n\n");
        fprintf(fp, "    const char*          graphEntryPoint;\n");
        fprintf(fp, "    const uint32_t       graphDataSize;\n");
        fprintf(fp, "    const unsigned char* graphData;\n\n");
        fprintf(fp, "    const uint32_t       tensorNums;\n");
        fprintf(fp, "    const char**         tensorNames;\n");
        fprintf(fp, "    const uint32_t*      tensorSets;\n");
        fprintf(fp, "    const uint32_t*      tensorBindings;\n");
        fprintf(fp, "    const uint32_t*      tensorFormats;\n");
        fprintf(fp, "    const uint32_t*      tensorDimSize;\n");
        fprintf(fp, "    const uint64_t**     tensorDims;\n");

        fprintf(fp, "} %s_Info;\n\n", varName.c_str());

        fprintf(fp, "static const %s_Info g_%s_Info = {\n", varName.c_str(), varName.c_str());

        fprintf(fp, "    %d, \n", constantNums);
        fprintf(fp, "    g_%s_constants_id, \n", varName.c_str());
        fprintf(fp, "    g_%s_constants_format, \n", varName.c_str());
        fprintf(fp, "    g_%s_constants_shape_size, \n", varName.c_str());
        fprintf(fp, "    g_%s_constants_shape, \n", varName.c_str());
        fprintf(fp, "    g_%s_constants_sparsity_dimension, \n", varName.c_str());
        fprintf(fp, "    g_%s_constants_data_size, \n", varName.c_str());
        fprintf(fp, "    g_%s_constants_data, \n", varName.c_str());

        fprintf(fp, "    g_%s_graph_%d_entry_point, \n", varName.c_str(), 0);
        fprintf(fp, "    g_%s_graph_%d_data_size, \n", varName.c_str(), 0);
        fprintf(fp, "    g_%s_graph_%d_data, \n", varName.c_str(), 0);
        fprintf(fp, "    g_%s_graph_%d_tensor_nums, \n", varName.c_str(), 0);
        fprintf(fp, "    g_%s_graph_%d_tensor_names, \n", varName.c_str(), 0);
        fprintf(fp, "    g_%s_graph_%d_tensor_sets, \n", varName.c_str(), 0);
        fprintf(fp, "    g_%s_graph_%d_tensor_bindings, \n", varName.c_str(), 0);
        fprintf(fp, "    g_%s_graph_%d_tensor_formats, \n", varName.c_str(), 0);
        fprintf(fp, "    g_%s_graph_%d_tensor_dim_size, \n", varName.c_str(), 0);
        fprintf(fp, "    g_%s_graph_%d_tensor_dims \n", varName.c_str(), 0);

        fprintf(fp, "};\n\n");
    }

    void parseVgf(std::wstring& wvgfFile, std::wstring& outputPath)
    {
        std::string vgfFile(wvgfFile.begin(), wvgfFile.end());

        std::string vgfFileName = std::filesystem::path(vgfFile).stem().generic_string();
        std::replace(vgfFileName.begin(), vgfFileName.end(), '-', '_');

        MemoryMap mapped(vgfFile);

        // Create header decoder
        std::vector<uint8_t> headerDecoderMemory;
        headerDecoderMemory.resize(mlsdk_decoder_header_decoder_mem_reqs());
        mlsdk_decoder_header_decoder* headerDecoder = mlsdk_decoder_create_header_decoder(mapped.ptr(), headerDecoderMemory.data());
        if (!mlsdk_decoder_is_header_valid(headerDecoder))
        {
            throw std::runtime_error("Invalid vgf header");
        }
        if (!mlsdk_decoder_is_header_compatible(headerDecoder))
        {
            throw std::runtime_error("Incompatible vgf header");
        }

        // Create module section decoder
        mlsdk_decoder_vgf_section_info moduleSection;
        mlsdk_decoder_get_header_section_info(headerDecoder, mlsdk_decoder_section_modules, &moduleSection);

        std::vector<uint8_t> decoderMemory;
        decoderMemory.resize(mlsdk_decoder_module_table_decoder_mem_reqs());
        mlsdk_decoder_module_table_decoder* moduleDecoder =
            mlsdk_decoder_create_module_table_decoder((char*)mapped.ptr() + moduleSection.offset, decoderMemory.data());

        // Create model sequence decoder
        mlsdk_decoder_vgf_section_info modelSequenceSection;
        mlsdk_decoder_get_header_section_info(headerDecoder, mlsdk_decoder_section_model_sequence, &modelSequenceSection);
        std::vector<uint8_t> sequenceDecoderMemory;
        sequenceDecoderMemory.resize(mlsdk_decoder_model_sequence_decoder_mem_reqs());
        mlsdk_decoder_model_sequence_decoder* sequenceDecoder =
            mlsdk_decoder_create_model_sequence_decoder((char*)mapped.ptr() + modelSequenceSection.offset, sequenceDecoderMemory.data());

        // Create model resource table decoder
        mlsdk_decoder_vgf_section_info modelResourceSection;
        mlsdk_decoder_get_header_section_info(headerDecoder, mlsdk_decoder_section_resources, &modelResourceSection);
        std::vector<uint8_t> resourceTableDecoderMemory;
        resourceTableDecoderMemory.resize(mlsdk_decoder_model_resource_table_decoder_mem_reqs());
        mlsdk_decoder_model_resource_table_decoder* resourceTableDecoder =
            mlsdk_decoder_create_model_resource_table_decoder((char*)mapped.ptr() + modelResourceSection.offset, resourceTableDecoderMemory.data());

        // Needed to transfer info from module handling to constant handling sections
        // Module Id -> {constant Id}
        std::map<int, std::set<int>> segmentConstantRefs;

        // Get module info / spirv code
        std::vector<VgfModule> vgfModules;
        size_t                 num_module_table_entries = mlsdk_decoder_get_module_table_num_entries(moduleDecoder);

        // Model sequence section
        mlsdk_decoder_vgf_section_info sequenceSection;
        mlsdk_decoder_get_header_section_info(headerDecoder, mlsdk_decoder_section_model_sequence, &sequenceSection);

        std::vector<std::string> graphHeaderFiles;
        std::vector<std::string> constantHeaderFiles;

        for (int moduleIdx = 0; moduleIdx < num_module_table_entries; ++moduleIdx)
        {
            switch (mlsdk_decoder_get_module_type(moduleDecoder, moduleIdx))
            {
            case mlsdk_decoder_module_type_graph:
            {
                mlsdk_decoder_spirv_code spirv;
                mlsdk_decoder_get_module_code(moduleDecoder, moduleIdx, &spirv);
                if (spirv.code == nullptr)
                {
                    throw std::runtime_error("No spirv code found in graph module");
                }

                // Get constant ids for each module
                mlsdk_decoder_constant_indexes constantIdxs{};
                mlsdk_decoder_model_sequence_get_segment_constant_indexes(sequenceDecoder, moduleIdx, &constantIdxs);
                for (int constantIdx = 0; constantIdx < constantIdxs.size; ++constantIdx)
                {
                    segmentConstantRefs[moduleIdx].insert(constantIdxs.data[constantIdx]);
                }

                auto castSpv = reinterpret_cast<const uint8_t*>(spirv.code);
                writeGraph(moduleIdx,
                           mlsdk_decoder_get_module_entry_point(moduleDecoder, moduleIdx),
                           std::vector(castSpv, castSpv + 4 * spirv.words),
                           getResourceInfos(sequenceDecoder, resourceTableDecoder, moduleIdx),
                           graphHeaderFiles,
                           outputPath,
                           vgfFileName);
                break;
            }
            case mlsdk_decoder_module_type_compute:
            {
                std::vector<uint32_t>    dispatchShape = getDispatchShape(sequenceDecoder);
                mlsdk_decoder_spirv_code spirv;
                mlsdk_decoder_get_module_code(moduleDecoder, moduleIdx, &spirv);

                const char* moduleName = mlsdk_decoder_get_module_name(moduleDecoder, moduleIdx);
                const char* entryPoint = mlsdk_decoder_get_module_entry_point(moduleDecoder, moduleIdx);

                std::vector<uint8_t> spvData = {};
                if (spirv.code)
                {
                    auto castSpv = reinterpret_cast<const uint8_t*>(spirv.code);
                    spvData      = std::vector(castSpv, castSpv + 4 * spirv.words);
                }

                break;
            }
            default:
                throw std::runtime_error("[Scenario-Runner]: Module is of unknown type");
            }
        }

        // Iterate over all vgf Resources, create intermediates
        const mlsdk_vk_descriptor_type DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT = 6;
        const mlsdk_vk_descriptor_type DESCRIPTOR_TYPE_STORAGE_TENSOR_EXT = 1000460000;

        size_t num_resources = mlsdk_decoder_get_model_resource_table_num_entries(resourceTableDecoder);
        for (int resourceIdx = 0; resourceIdx < num_resources; ++resourceIdx)
        {
            const std::string guidStr = "Resource_" + std::to_string(resourceIdx) + "_intermediate";
            if (mlsdk_decoder_model_resource_table_get_category(resourceTableDecoder, resourceIdx) == mlsdk_decoder_mrt_category_intermediate)
            {
                auto type = mlsdk_decoder_get_vk_descriptor_type(resourceTableDecoder, resourceIdx);
                switch (type.value)
                {
                case (DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT):
                {
                    mlsdk_decoder_tensor_dimensions dims;
                    mlsdk_decoder_model_resource_table_get_tensor_shape(resourceTableDecoder, resourceIdx, &dims);
                    auto bufferSize = static_cast<uint32_t>(std::accumulate(dims.data, dims.data + dims.size, 1, std::multiplies<int64_t>()));

                    // Create scenario runner buffer resource
                    BufferInfo        info{bufferSize};
                    std::vector<char> values(bufferSize, 0);
                    //createBuffer(Guid(guidStr), info, values);
                }
                break;
                case (DESCRIPTOR_TYPE_STORAGE_TENSOR_EXT):
                {
                    mlsdk_decoder_tensor_dimensions dims;
                    mlsdk_decoder_model_resource_table_get_tensor_shape(resourceTableDecoder, resourceIdx, &dims);
                    mlsdk_vk_format format = mlsdk_decoder_get_vk_format(resourceTableDecoder, resourceIdx);

                    auto info = TensorInfo{
                        std::vector<uint64_t>(dims.data, dims.data + dims.size),
                        VkFormat(format),
                        false,
                        -1,
                    };
                }
                break;
                default:
                    throw std::runtime_error("Unknown resource type read from VGF file");
                }
            }
        }

        // Create constant decoder
        mlsdk_decoder_vgf_section_info modelConstantsSection;
        mlsdk_decoder_get_header_section_info(headerDecoder, mlsdk_decoder_section_constants, &modelConstantsSection);

        std::vector<uint8_t> constantDecoderMemory;
        constantDecoderMemory.resize(mlsdk_decoder_constant_table_decoder_mem_reqs());
        mlsdk_decoder_constant_table_decoder* constantDecoder =
            mlsdk_decoder_create_constant_table_decoder((char*)mapped.ptr() + modelConstantsSection.offset, constantDecoderMemory.data());

        std::vector<ConstantsInfo> constantInfos;
        // Iterate over all vgf constants
        size_t num_constants = mlsdk_decoder_get_constant_table_num_entries(constantDecoder);
        for (int constantIdx = 0; constantIdx < num_constants; ++constantIdx)
        {
            mlsdk_decoder_constant_data constantData;
            mlsdk_decoder_constant_table_get_data(constantDecoder, constantIdx, &constantData);
            if (constantData.data == nullptr)
            {
                throw std::runtime_error("Invalid constant retrieved");
            }

            auto mrt_idx = mlsdk_decoder_constant_table_get_mrt_index(constantDecoder, constantIdx);
            if (mlsdk_decoder_model_resource_table_get_category(resourceTableDecoder, mrt_idx) != mlsdk_decoder_mrt_category_constant)
            {
                throw std::runtime_error("VGF constant retrieved not marked as such");
            }

            // get constant shape
            mlsdk_decoder_tensor_dimensions constantDims;
            mlsdk_decoder_model_resource_table_get_tensor_shape(resourceTableDecoder, mrt_idx, &constantDims);

            // Create scenario-runner constant tensor object
            TensorInfo tensorInfo;
            tensorInfo.shape             = std::vector<uint64_t>(constantDims.data, constantDims.data + constantDims.size);
            tensorInfo.format            = static_cast<VkFormat>(mlsdk_decoder_get_vk_format(resourceTableDecoder, mrt_idx));
            tensorInfo.sparsityDimension = mlsdk_decoder_constant_table_get_sparsity_dimension(constantDecoder, constantIdx);

            // Check each module to see if requires the current constant
            for (auto ref : segmentConstantRefs)
            {
                if (ref.second.count(constantIdx))
                {
                    constantInfos.push_back({constantIdx, tensorInfo, constantData});
                }
            }
        }

        writeConstants(constantInfos, constantHeaderFiles, outputPath, vgfFileName);
        writeHeaderFile(graphHeaderFiles, constantHeaderFiles, constantInfos.size(), outputPath, vgfFileName);
    }

    static const wchar_t* const APP_NAME    = L"Arm_Model_Parser";
    static const wchar_t* const APP_VERSION = L"1.0.0";

    struct LaunchParameters
    {
        std::wstring outputPath;
        std::wstring inputFile;
    };

    void printCommandLineSyntax()
    {
        wprintf(L"%ls %ls\n", APP_NAME, APP_VERSION);
        wprintf(L"Command line syntax:\n");
        wprintf(L"  %ls.exe [Options] <InputFile>\n", APP_NAME);
        wprintf(
            L"Options:\n"
            L"-output=<Path>\n"
            L"  Path to where the shader permutations should be output to.\n");
    }

    bool startsWith(const wchar_t* s, const wchar_t* subS)
    {
        size_t sLen    = wcslen(s);
        size_t subSLen = wcslen(subS);
        if (sLen >= subSLen)
        {
            size_t p = std::wstring(s).find(std::wstring(subS));
            return p == 0;
        }
        return false;
    }

    void parseString(std::wstring& outCompilerArg, const wchar_t* arg)
    {
        std::wstring argStr   = std::wstring(arg);
        size_t       equalPos = argStr.find_first_of(L"=", 0);
        outCompilerArg        = argStr.substr(equalPos + 1, argStr.length() - equalPos);
    }

    void parseCommandLine(int argCount, const wchar_t* const* args, LaunchParameters& params)
    {
        int i = 0;

        for (; i < argCount; ++i)
        {
            if (startsWith(args[i], L"-output"))
                parseString(params.outputPath, args[i]);
            else
                params.inputFile = args[i];
        }
    }

}  // namespace arm

#ifdef _WIN32
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
    if (argc <= 1)
    {
        arm::printCommandLineSyntax();
        return 1;
    }

    arm::LaunchParameters params;
#ifdef _WIN32
    arm::parseCommandLine(argc - 1, argv + 1, params);
#else
    std::vector<std::wstring> wargs(argc - 1);
    for (int i = 1; i < argc; ++i)
    {
        wargs[i - 1] = UTF8ToWChar(argv[i]);
    }

    std::vector<const wchar_t*> wargv;
    for (const auto& w : wargs)
    {
        wargv.push_back(w.c_str());
    }
    arm::parseCommandLine(static_cast<int>(wargv.size()), wargv.data(), params);
#endif
    arm::parseVgf(params.inputFile, params.outputPath);
    return 0;
}
