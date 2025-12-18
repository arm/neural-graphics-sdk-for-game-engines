/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ML_SDK_VGF_LIB_DECODER_API_H
#define ML_SDK_VGF_LIB_DECODER_API_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif          // __cplusplus
#ifdef __GNUC__ // GCC/CLANG
#    define MLSDK_EXPORT __attribute__((visibility("default")))
#    define MLSDK_IMPORT
#else
#    ifdef _MSC_VER
#        define MLSDK_EXPORT __declspec(dllexport)
#        define MLSDK_IMPORT __declspec(dllimport)
#    else
#        define MLSDK_EXPORT
#        define MLSDK_IMPORT
#        pragma warning Undefined dynamic library export / import semantic for detected toolchain.
#    endif
#endif
#ifdef MLSDK_DYNAMIC_LIB    // Building for a dynamic lib
#    ifdef MLSDK_EXPORT_API // Building the library for export
#        define MLSDKAPI MLSDK_EXPORT
#    else
#        define MLSDKAPI MLSDK_IMPORT
#    endif
#else
#    define MLSDKAPI
#endif

typedef struct mlsdk_decoder_header_decoder_s *mlsdk_decoder_header_decoder;
typedef struct mlsdk_decoder_module_table_decoder_s *mlsdk_decoder_module_table_decoder;
typedef struct mlsdk_decoder_push_constants_range_decoder_s *mlsdk_decoder_push_constants_range_decoder;
typedef struct mlsdk_decoder_model_sequence_decoder_s *mlsdk_decoder_model_sequence_decoder;
typedef struct mlsdk_decoder_model_resource_table_decoder_s *mlsdk_decoder_model_resource_table_decoder;
typedef struct mlsdk_decoder_constant_table_decoder_s *mlsdk_decoder_constant_table_decoder;

/**
 * \defgroup VGFCAPI Decoder C API
 * @{
 */

/**
 * @brief Type for VK_HEADER_VERSION
 *
 */
typedef uint16_t mlsdk_vk_header_version;

/**
 * @brief Enum for index of sections in the VGF
 *
 */
typedef enum {
    mlsdk_decoder_section_modules = 0,
    mlsdk_decoder_section_model_sequence = 1,
    mlsdk_decoder_section_resources = 2,
    mlsdk_decoder_section_constants = 3,
} mlsdk_decoder_section;

/**
 * @brief Type for VkDescriptorType enum
 *
 */
typedef int32_t mlsdk_vk_descriptor_type;

/**
 * @brief Optional mlsdk_vk_descriptor_type
 *
 */
typedef struct {
    mlsdk_vk_descriptor_type value;
    bool has_value;
} mlsdk_vk_descriptor_type_optional;

/**
 * @brief Carrier type for VkFormat
 *
 */
typedef int32_t mlsdk_vk_format;

/**
 * @brief return the value for an mlsdk_vk_format corresponding to VK_FORMAT_UNDEFINED.
 *
 */
MLSDKAPI mlsdk_vk_format mlsdk_vk_format_undefined();

/**
 * @brief Enum for module types
 *
 */
typedef enum {
    mlsdk_decoder_module_type_compute = 0,
    mlsdk_decoder_module_type_graph = 1,
} mlsdk_decoder_module_type;

/**
 * @brief Enum for model resource table category
 *
 */
typedef enum {
    mlsdk_decoder_mrt_category_input,
    mlsdk_decoder_mrt_category_output,
    mlsdk_decoder_mrt_category_intermediate,
    mlsdk_decoder_mrt_category_constant,
} mlsdk_decoder_mrt_category;

/**
 * @brief Semantic version of the VGF format.
 * Compatibility is ensured if the file uses the same major version wrt to this library and
 * a minor version which is lesser or equal wrt this library.
 */
typedef struct {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
} mlsdk_decoder_vgf_version;

/**
 * @brief Section info describing the location and size of a VGF section.
 *
 */
typedef struct {
    uint64_t offset; ///< offset from the beginning of file the start of the section.
    uint64_t size;   ///< size (in bytes) of the section.
} mlsdk_decoder_vgf_section_info;

/**
 * @brief The SPIR-V code view
 */
typedef struct {
    const uint32_t *code;
    size_t words;
} mlsdk_decoder_spirv_code;

/**
 * @brief The constant data view
 */
typedef struct {
    const uint8_t *data;
    size_t size;
} mlsdk_decoder_constant_data;

/**
 * @brief The dispatch shape view
 */
typedef struct {
    uint32_t data[3];
} mlsdk_decoder_dispatch_shape;

/**
 * @brief The constant view
 */
typedef struct {
    const uint32_t *data;
    size_t size;
} mlsdk_decoder_constant_indexes;

/**
 * @brief The dimensions view
 * values of "-1" represent unshaped dimensions
 */
typedef struct {
    const int64_t *data;
    size_t size;
} mlsdk_decoder_tensor_dimensions;

/**
 * @brief Gets the library version
 *
 * @param version The pointer to the struct to write the version information
 */
MLSDKAPI void mlsdk_decoder_get_version(mlsdk_decoder_vgf_version *version);

/**
 * @brief Returns the size in bytes of the VGF header on disk
 * @return The size in bytes of the VGF header data
 */
MLSDKAPI size_t mlsdk_decoder_header_size();

/**
 * @brief Returns the memory requirements in bytes to allocate memory for creating the header decoder
 * @return The size in bytes of the memory needed to create the header decoder
 */
MLSDKAPI size_t mlsdk_decoder_header_decoder_mem_reqs();

/**
 * @brief Creates the header decoder
 *
 * @param headerData The pointer to the header data
 * @param decoderMemory Memory allocated to be used to create the decoder
 * @return The pointer to the newly created decoder
 */
MLSDKAPI mlsdk_decoder_header_decoder *mlsdk_decoder_create_header_decoder(const void *const headerData,
                                                                           void *decoderMemory);

/**
 * @brief Checks if the header is valid
 *
 * @param decoder The header decoder associated to the header data
 * @return True if the parsed data is a valid VGF header, false otherwise
 */
MLSDKAPI bool mlsdk_decoder_is_header_valid(const mlsdk_decoder_header_decoder *const decoder);

/**
 * @brief Checks if the VGF file is compatible with this library version
 *
 * @param decoder The header decoder associated to the header data
 * @return True if the VGF file is compatible with the library, false otherwise
 */
MLSDKAPI bool mlsdk_decoder_is_header_compatible(const mlsdk_decoder_header_decoder *const decoder);

/**
 * @brief Returns the value of VK_HEADERS_VERSION found in the VGF file
 *
 * @param decoder The header decoder associated to the header data
 * @param vkHeaderVersion Return value of the VK_HEADERS_VERSION used while encoding the VGF
 */
MLSDKAPI void mlsdk_decoder_get_encoder_vk_header_version(const mlsdk_decoder_header_decoder *const decoder,
                                                          mlsdk_vk_header_version *vkHeaderVersion);

/**
 * @brief Gets the VGF file version
 *
 * @param decoder The header decoder associated to the header data
 * @param version The pointer to the struct to write the result to
 */
MLSDKAPI void mlsdk_decoder_get_header_version(const mlsdk_decoder_header_decoder *const decoder,
                                               mlsdk_decoder_vgf_version *version);

/**
 * @brief Gets the VGF file section info for the given section name
 *
 * @param decoder The header decoder associated to the header data
 * @param sectionName Section name
 * @param section The pointer to the VGF section info
 *
 */
MLSDKAPI void mlsdk_decoder_get_header_section_info(const mlsdk_decoder_header_decoder *const decoder,
                                                    mlsdk_decoder_section sectionName,
                                                    mlsdk_decoder_vgf_section_info *section);

/**
 * @brief Returns the memory requirements in bytes to allocate memory for creating the module table decoder
 * @return The size in bytes of the memory needed to create the module table decoder
 */
MLSDKAPI size_t mlsdk_decoder_module_table_decoder_mem_reqs();

/**
 * @brief Checks if pointer points to valid module table data
 *
 * @param moduleTableData The pointer to the data
 * @param size The size in bytes of the data
 * @return True if the data is a valid section, false otherwise
 */
MLSDKAPI bool mlsdk_decoder_is_valid_module_table(const void *moduleTableData, uint64_t size);

/**
 * @brief Creates the module table decoder
 *
 * @param moduleTableData The pointer to the module table data
 * @param decoderMemory Memory allocated to be used to create the decoder
 * @return The pointer to the newly created decoder
 */
MLSDKAPI mlsdk_decoder_module_table_decoder *
mlsdk_decoder_create_module_table_decoder(const void *const moduleTableData, void *decoderMemory);

/**
 * @brief Returns the number of entries in the module table
 *
 * @param decoder The pointer to the module table decoder
 * @return The number of entries in the table
 */
MLSDKAPI size_t mlsdk_decoder_get_module_table_num_entries(const mlsdk_decoder_module_table_decoder *const decoder);

/**
 * @brief Returns the module type of the idx-entry
 *
 * @param decoder The pointer to the module table decoder
 * @param idx The index for the entry in the module table
 * @return The module type of the entry
 */
MLSDKAPI mlsdk_decoder_module_type
mlsdk_decoder_get_module_type(const mlsdk_decoder_module_table_decoder *const decoder, uint32_t idx);

/**
 * @brief Returns the module name of the idx-entry
 *
 * @param decoder The pointer to the module table decoder
 * @param idx The index for the entry in the module table
 * @return Char pointer to the module name
 */
MLSDKAPI const char *mlsdk_decoder_get_module_name(const mlsdk_decoder_module_table_decoder *const decoder,
                                                   uint32_t idx);

/**
 * @brief Returns the SPIR-V entry_point name stored in the module
 *
 * @param decoder The pointer to the module table decoder
 * @param idx The index for the entry in the module table
 * @return Char pointer to the entry_point name
 *
 * If no entry_point is stored in the module a null pointer is returned.
 */
MLSDKAPI const char *mlsdk_decoder_get_module_entry_point(const mlsdk_decoder_module_table_decoder *const decoder,
                                                          uint32_t idx);

/**
 * @brief Gets the SPIR-V code stored in the module
 *
 * @param decoder The pointer to the module table decoder
 * @param idx The index for the entry in the module table
 * @param spirvCode The place where to store the SPIR-V code
 *
 * If no code is stored in the module the SPIR-V code pointer will be set to null and the words to zero.
 */
MLSDKAPI void mlsdk_decoder_get_module_code(const mlsdk_decoder_module_table_decoder *const decoder, uint32_t idx,
                                            mlsdk_decoder_spirv_code *spirvCode);

/**********************************************************************************************************************/

/**
 * @brief Handle to refer to a specific view of a bundle of binding slots
 *
 */
typedef struct mlsdk_decoder_binding_slots_handle_s const *mlsdk_decoder_binding_slots_handle;

/**
 * @brief Returns the number of binding slots
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @param handle The handle to the binding slots array
 * @return Number of binding slots in model sequence
 */
MLSDKAPI size_t mlsdk_decoder_binding_slot_size(const mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder,
                                                mlsdk_decoder_binding_slots_handle handle);

/**
 * @brief Returns the binding id of a given binding slot
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @param handle The handle to the binding slots array
 * @param slotIdx The index of the binding slot in the binding slots array
 * @return Binding id of binding slot with index slotIdx
 */
MLSDKAPI uint32_t
mlsdk_decoder_binding_slot_binding_id(const mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder,
                                      mlsdk_decoder_binding_slots_handle handle, uint32_t slotIdx);

/**
 * @brief Returns the mrt index of a given binding slot
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @param handle The handle to the binding slots array
 * @param slotIdx The index of the binding slot in the model sequence binding
 * @return Mrt index of binding slot with index slotIdx
 */
MLSDKAPI uint32_t
mlsdk_decoder_binding_slot_mrt_index(const mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder,
                                     mlsdk_decoder_binding_slots_handle handle, uint32_t slotIdx);

/**********************************************************************************************************************/

/**
 * @brief Handle to refer to an array of push constant ranges
 *
 */
typedef struct mlsdk_decoder_push_constant_ranges_handle_s const *mlsdk_decoder_push_constant_ranges_handle;

/**
 * @brief Returns the number of push constant ranges
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @param handle The handle to the push constant ranges
 * @return Number of push constant ranges
 */
MLSDKAPI size_t
mlsdk_decoder_get_push_constant_ranges_size(const mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder,
                                            mlsdk_decoder_push_constant_ranges_handle handle);

/**
 * @brief Returns the stage flags of a given push constant range
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @param handle The handle to the push constant ranges
 * @param rangeIdx The index of the push constant range in the model sequence push constant ranges
 * @return Stage flags of push constant range with index rangeIdx
 */
MLSDKAPI uint32_t mlsdk_decoder_get_push_constant_range_stage_flags(
    const mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder,
    mlsdk_decoder_push_constant_ranges_handle handle, uint32_t rangeIdx);

/**
 * @brief Returns the offset of a given push constant range
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @param handle The handle to the push constant ranges
 * @param rangeIdx The index of the push constant range in the model sequence push constant ranges
 * @return Offset of push constant range with index rangeIdx
 */
MLSDKAPI uint32_t
mlsdk_decoder_get_push_constant_range_offset(const mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder,
                                             mlsdk_decoder_push_constant_ranges_handle handle, uint32_t rangeIdx);

/**
 * @brief Returns the number of push constants in a given push constant range
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @param handle The handle to the push constant ranges
 * @param rangeIdx The index of the push constant range in the model sequence push constant ranges
 * @return Number of push constants in push constant range with index rangeIdx
 */
MLSDKAPI uint32_t
mlsdk_decoder_get_push_constant_range_size(const mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder,
                                           mlsdk_decoder_push_constant_ranges_handle handle, uint32_t rangeIdx);

/**********************************************************************************************************************/

/**
 * @brief Checks if pointer points to valid model sequence data
 *
 * @param modelSequenceData The pointer to the data
 * @param size The size in bytes of the data
 * @return True if the data is a valid section, false otherwise
 */
MLSDKAPI bool mlsdk_decoder_is_valid_model_sequence(const void *modelSequenceData, uint64_t size);

/**
 * @brief Create the model sequence decoder
 *
 * @param modelSequenceData The pointer to the model sequence data
 * @param modelSequenceDecoderMemory Memory allocated to be used to create the decoder
 * @return The pointer to the newly created model sequence decoder
 */
MLSDKAPI mlsdk_decoder_model_sequence_decoder *
mlsdk_decoder_create_model_sequence_decoder(const void *const modelSequenceData, void *modelSequenceDecoderMemory);

/**
 * @brief Returns the memory requirements in bytes to allocate memory for creating the model sequence decoder
 *
 * @return The size in bytes of the memory needed to create the model sequence decoder
 */
MLSDKAPI size_t mlsdk_decoder_model_sequence_decoder_mem_reqs();

/**
 * @brief Returns the number of segments in a model sequence table
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @return Number of segments in a model sequence table
 */
MLSDKAPI size_t
mlsdk_decoder_get_model_sequence_table_size(const mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder);

/**
 * @brief Returns the number of descriptorset infos in a given segment of model sequence
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @param segIdx The index for the segment in the model sequence
 * @return Number of descriptorset infos in segment with index segIdx
 */
MLSDKAPI size_t mlsdk_decoder_model_sequence_get_segment_descriptorset_info_size(
    const mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder, uint32_t segIdx);

/**
 * @brief Gets the constant indexes for the given segment into the constant section
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @param segIdx The index of segment in the model sequence
 * @param constant Constant struct pointer where to store the data
 */
MLSDKAPI void mlsdk_decoder_model_sequence_get_segment_constant_indexes(
    const mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder, uint32_t segIdx,
    mlsdk_decoder_constant_indexes *constant);

/**
 * @brief Returns the type of a given segment of model sequence
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @param segIdx The index for the segment in the model sequence
 * @return Type of segment with index segIdx
 */
MLSDKAPI mlsdk_decoder_module_type mlsdk_decoder_model_sequence_get_segment_type(
    const mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder, uint32_t segIdx);

/**
 * @brief Returns the name of given segment of model sequence
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @param segIdx The index for the segment in the model sequence
 * @return Name of segment with index segIdx
 */
MLSDKAPI const char *
mlsdk_decoder_model_sequence_get_segment_name(const mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder,
                                              uint32_t segIdx);

/**
 * @brief Returns the module index into the ModuleTable to access the associated Segment's Module
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @param segIdx The index for the segment in the model sequence
 * @return The module index into the ModuleTable of the assocated Segment's Module
 */
MLSDKAPI uint32_t mlsdk_decoder_model_sequence_get_segment_module_index(
    const mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder, uint32_t segIdx);

/**
 * @brief Gets the dispatch shape of given segment of model sequence
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @param segIdx The index for the segment in the model sequence
 * @param dispatchShape Pointer to a dispatch shape struct where to save the data
 */
MLSDKAPI void mlsdk_decoder_model_sequence_get_segment_dispatch_shape(
    mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder, uint32_t segIdx,
    mlsdk_decoder_dispatch_shape *dispatchShape);

/**
 * @brief Gets the push constant range of a given segment of model sequence
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @param segIdx The index for the segment in the model sequence
 * @return Handle to the push constant ranges
 */
MLSDKAPI mlsdk_decoder_push_constant_ranges_handle mlsdk_decoder_model_sequence_get_segment_push_constant_range(
    mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder, uint32_t segIdx);

/**
 * @brief Gets the binding slot of a given descriptor of a given segment of model sequence
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @param segIdx The index for the segment in the model sequence
 * @param descIdx The index for the descriptor in the segment
 * @return Handle to the binding slots array
 * descInd in the segment with index segIdx
 */
MLSDKAPI mlsdk_decoder_binding_slots_handle mlsdk_decoder_model_sequence_get_segment_descriptor_binding_slot(
    mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder, uint32_t segIdx, uint32_t descIdx);

/**
 * @brief Gets the input binding slot of a given segment of model sequence
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @param segIdx The index for the segment in the model sequence
 * @return Handle to the binding slots array
 * */
MLSDKAPI mlsdk_decoder_binding_slots_handle mlsdk_decoder_model_sequence_get_segment_input_binding_slot(
    mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder, uint32_t segIdx);

/**
 * @brief Gets the output binding slot of a given segment of model sequence
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @param segIdx The index for the segment in the model sequence
 * @return Handle to the binding slots array
 */
MLSDKAPI
mlsdk_decoder_binding_slots_handle mlsdk_decoder_model_sequence_get_segment_output_binding_slot(
    mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder, uint32_t segIdx);

/**
 * @brief Gets the input binding slot of model sequence
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @return Handle to the binding slots array
 */

MLSDKAPI mlsdk_decoder_binding_slots_handle
mlsdk_decoder_model_sequence_get_input_binding_slot(mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder);
/**
 * @brief Gets the output binding slot of model sequence
 *
 * @param modelSequenceDecoder The pointer to the model sequence decoder
 * @return Handle to the binding slots array
 */

MLSDKAPI mlsdk_decoder_binding_slots_handle
mlsdk_decoder_model_sequence_get_output_binding_slot(mlsdk_decoder_model_sequence_decoder *const modelSequenceDecoder);
/**
 * @brief Returns the memory requirements in bytes to allocate memory for creating the model resource table decoder
 * @return The size in bytes of the memory needed to create the model resource table decoder
 */
MLSDKAPI size_t mlsdk_decoder_model_resource_table_decoder_mem_reqs();

/**
 * @brief Checks if pointer points to valid model resource table data
 *
 * @param modelResourceTableData The pointer to the data
 * @param size The size in bytes of the data
 * @return True if the data is a valid section, false otherwise
 */
MLSDKAPI bool mlsdk_decoder_is_valid_model_resource_table(const void *modelResourceTableData, uint64_t size);

/**
 * @brief Create the model resource table decoder
 *
 * @param modelResourceTableData The pointer to the module resource table data
 * @param decoderMemory Memory allocated to be used to create the decoder
 * @return The pointer to the newly created decoder
 */
MLSDKAPI mlsdk_decoder_model_resource_table_decoder *
mlsdk_decoder_create_model_resource_table_decoder(const void *const modelResourceTableData, void *decoderMemory);

/**
 * @brief Returns the number of entries in the model resource table
 *
 * @param modelResourceTableDecoder The pointer to the model resource table decoder
 * @return The number of entries in the table
 */
MLSDKAPI size_t mlsdk_decoder_get_model_resource_table_num_entries(
    const mlsdk_decoder_model_resource_table_decoder *const modelResourceTableDecoder);

/**
 * @brief Returns the vk_descriptor_type of the idx-entry in the model resource table
 * or mlsdk_vk_descriptor_type_none for types with no corresponding descriptor type
 *
 * @param modelResourceTableDecoder The pointer to the model resource table decoder
 * @param idx The index for the entry in the model resource table
 * @return The model resource vk_descriptor_type of the entry
 */
MLSDKAPI mlsdk_vk_descriptor_type_optional mlsdk_decoder_get_vk_descriptor_type(
    const mlsdk_decoder_model_resource_table_decoder *const modelResourceTableDecoder, uint32_t idx);

/**
 * @brief Returns the vk_format of the idx-entry in the model resource table
 *
 * @param modelResourceTableDecoder The pointer to the model resource table decoder
 * @param idx The index for the entry in the model resource table
 * @return The model resource vk_format of the entry
 */
MLSDKAPI mlsdk_vk_format mlsdk_decoder_get_vk_format(
    const mlsdk_decoder_model_resource_table_decoder *const modelResourceTableDecoder, uint32_t idx);

/**
 * @brief Returns the category of the idx-entry in the model resource table
 *
 * @param modelResourceTableDecoder The pointer to the model resource table decoder
 * @param mrtIdx The index for the entry in the model resource table
 * @return The category of the entry
 */
MLSDKAPI mlsdk_decoder_mrt_category mlsdk_decoder_model_resource_table_get_category(
    const mlsdk_decoder_model_resource_table_decoder *const modelResourceTableDecoder, uint32_t mrtIdx);

/**
 * @brief Gets the shape of the idx-entry in the model resource table
 *
 * @param modelResourceTableDecoder The pointer to the model resource table decoder
 * @param mrtIdx The index for the entry in the model resource table
 * @param dimensions The pointer to the shape of the model resource table entry
 */
MLSDKAPI void mlsdk_decoder_model_resource_table_get_tensor_shape(
    const mlsdk_decoder_model_resource_table_decoder *const modelResourceTableDecoder, uint32_t mrtIdx,
    mlsdk_decoder_tensor_dimensions *dimensions);

/**
 * @brief Gets the strides of the idx-entry in the model resource table
 *
 * @param modelResourceTableDecoder The pointer to the model resource table decoder
 * @param mrtIdx The index for the entry in the model resource table
 * @param dimensions The pointer to the strides of the model resource table entry
 */
MLSDKAPI void mlsdk_decoder_model_resource_table_get_tensor_strides(
    const mlsdk_decoder_model_resource_table_decoder *const modelResourceTableDecoder, uint32_t mrtIdx,
    mlsdk_decoder_tensor_dimensions *dimensions);

/**********************************************************************************************************************/

/**
 * @brief Checks if pointer points to valid constant table data
 *
 * @param constantTableData The pointer to the data
 * @param size The size in bytes of the data
 * @return True if the data is a valid section, false otherwise
 */
MLSDKAPI bool mlsdk_decoder_is_valid_constant_table(const void *constantTableData, uint64_t size);

/**
 * @brief Create the constant table decoder
 *
 * @param constantTableData The pointer to the constant table data
 * @param constantDecoderMemory Memory allocated to be used to create the decoder
 * @return The pointer to the newly created decoder
 */
MLSDKAPI mlsdk_decoder_constant_table_decoder *
mlsdk_decoder_create_constant_table_decoder(const void *const constantTableData, void *constantDecoderMemory);

/**
 * @brief Returns the memory requirements in bytes to allocate memory for creating the constant tabledecoder
 * @return The size in bytes of the memory needed to create the constant table decoder
 */
MLSDKAPI size_t mlsdk_decoder_constant_table_decoder_mem_reqs();

/**
 * @brief Gets the data for the constant stored in the idx-entry in the constant table
 *
 * @param constantDecoder The pointer to the constant table decoder
 * @param idx The index for the entry in the constant table
 * @param constantData The place where to store the constant data
 *
 * If no code is stored in the module the constantData pointer will be set to null
 */
MLSDKAPI void mlsdk_decoder_constant_table_get_data(const mlsdk_decoder_constant_table_decoder *const constantDecoder,
                                                    uint32_t idx, mlsdk_decoder_constant_data *constantData);

/**
 * @brief Returns the index into the module resource table associated to the constant
 *
 * @param constantDecoder The pointer to the constant table decoder
 * @param constidx The index for the entry in the constant table
 * @return The index into the model resource table
 */
MLSDKAPI uint32_t mlsdk_decoder_constant_table_get_mrt_index(
    const mlsdk_decoder_constant_table_decoder *const constantDecoder, uint32_t constidx);

/**
 * @brief Returns true if the constant is sparse on one dimension
 *
 * @param constantDecoder The pointer to the constant table decoder
 * @param constidx The index for the entry in the constant table
 * @return true if constant is sparse or false oherwise
 */
MLSDKAPI bool mlsdk_decoder_constant_table_is_sparse(const mlsdk_decoder_constant_table_decoder *const constantDecoder,
                                                     uint32_t constidx);

/**
 * @brief Returns the dimension on which the constant is sparse
 *
 * @param constantDecoder The pointer to the constant table decoder
 * @param constidx The index for the entry in the constant table
 * @return The dimension on which the constant is sparse
 */
MLSDKAPI int64_t mlsdk_decoder_constant_table_get_sparsity_dimension(
    const mlsdk_decoder_constant_table_decoder *const constantDecoder, uint32_t constidx);

/**
 * @brief Returns the number of entries in the constant table
 *
 * @param constantDecoder The pointer to the constant table decoder
 * @return The number of entries in the table
 */
MLSDKAPI size_t
mlsdk_decoder_get_constant_table_num_entries(const mlsdk_decoder_constant_table_decoder *const constantDecoder);

/**@}*/

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // ML_SDK_VGF_LIB_DECODER_API_H
