# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

project(Bin2Exr)

find_package(Vulkan REQUIRED)

# Bin2Exr parses capture metadata with nlohmann's single-header library. It is
# not distributed with the SDK; retain the samples default only when that tree
# is present and otherwise require the public external-dependency contract.
set(_ffx_capture_json_default "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../samples/third_party/tinygltf/json.hpp")
    set(_ffx_capture_json_default "${CMAKE_CURRENT_SOURCE_DIR}/../../../samples/third_party/tinygltf")
endif()
set(FFX_CAPTURE_JSON_INCLUDE_DIR
    "${_ffx_capture_json_default}"
    CACHE PATH "Directory containing nlohmann/json.hpp for Bin2Exr")
if(NOT EXISTS "${FFX_CAPTURE_JSON_INCLUDE_DIR}/json.hpp")
    message(FATAL_ERROR
        "Bin2Exr requires nlohmann/json.hpp. Set FFX_CAPTURE_JSON_INCLUDE_DIR "
        "to the directory containing json.hpp.")
endif()

add_executable(Bin2Exr
    bin2exr.cpp)

target_compile_features(Bin2Exr PRIVATE cxx_std_17)
target_include_directories(Bin2Exr PRIVATE
    ${FFX_INCLUDE_PATH}
    ${FFX_INCLUDE_PATH}/vulkan-headers
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${FFX_CAPTURE_JSON_INCLUDE_DIR})
target_link_libraries(Bin2Exr PRIVATE
    arm_capture_${FFX_PLATFORM_NAME}
    ffx_backend_vk_${FFX_PLATFORM_NAME})
set_target_properties(Bin2Exr PROPERTIES FOLDER Tools)
