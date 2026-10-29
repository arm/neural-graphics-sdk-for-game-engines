# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)

project(NGLayer LANGUAGES CXX)

include("${CMAKE_CURRENT_SOURCE_DIR}/../sdk/cmake/release_build.cmake")

# Canonical short name for the layer. The Vulkan layer name is derived from
# this via "VK_LAYER_ARM_<name>"; platform-specific binary/manifest filenames
# keep Linux and Android layer libraries from colliding in release packages.
set(NGLAYER_SHORT_NAME      "NG")
set(NGLAYER_NAME            "VK_LAYER_ARM_${NGLAYER_SHORT_NAME}")
set(NGLAYER_OUTPUT_NAME     "VkLayer_arm_${NGLAYER_SHORT_NAME}")
set(NGLAYER_MANIFEST_LINUX   "VkLayer_arm_${NGLAYER_SHORT_NAME}_linux.json")
set(NGLAYER_MANIFEST_ANDROID "VkLayer_arm_${NGLAYER_SHORT_NAME}_android.json")
set(NGLAYER_MANIFEST_WINDOWS "VkLayer_arm_${NGLAYER_SHORT_NAME}_windows.json")

if(ANDROID)
    set(NGLAYER_OUTPUT_NAME "VkLayer_arm_${NGLAYER_SHORT_NAME}_android")
elseif(WIN32)
    set(NGLAYER_OUTPUT_NAME "VkLayer_arm_${NGLAYER_SHORT_NAME}_windows")
elseif(UNIX)
    set(NGLAYER_OUTPUT_NAME "VkLayer_arm_${NGLAYER_SHORT_NAME}_linux")
endif()

add_library(NGLayer SHARED
    NGLayer.cpp
)

# Use the SDK's bundled Vulkan headers so the layer always builds against the
# same vulkan_core.h that ships with NG-SDK (including the VK_ARM_tensors and
# VK_ARM_data_graph extension definitions). Allow overriding for out-of-tree
# builds via -DNGLAYER_VULKAN_INCLUDE_DIR=<path>.
if(NOT NGLAYER_VULKAN_INCLUDE_DIR)
    set(NGLAYER_VULKAN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../sdk/include/vulkan-headers")
endif()
if(NOT EXISTS "${NGLAYER_VULKAN_INCLUDE_DIR}/vulkan/vk_layer.h")
    message(FATAL_ERROR
        "NGLayer: Vulkan headers not found at '${NGLAYER_VULKAN_INCLUDE_DIR}/vulkan'. "
        "Pass -DNGLAYER_VULKAN_INCLUDE_DIR=<path-to-vulkan-headers> to override.")
endif()
target_include_directories(NGLayer PRIVATE "${NGLAYER_VULKAN_INCLUDE_DIR}")

target_compile_features(NGLayer PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(NGLayer PRIVATE /W3)
else()
    target_compile_options(NGLayer PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# Android loader auto-discovers libVkLayer_*.so from the app nativeLibraryDir;
# the binary name must match the layer manifest's library_path field.
set_target_properties(NGLayer PROPERTIES
    OUTPUT_NAME "${NGLAYER_OUTPUT_NAME}"
)

if(ANDROID)
    target_link_options(NGLayer PRIVATE
        LINKER:-z,max-page-size=16384
        LINKER:-z,common-page-size=16384
    )
    target_link_libraries(NGLayer PRIVATE log) # logcat support
endif()

# --- Output placement ------------------------------------------------------
# When built as part of NG-SDK (i.e. add_subdirectory(nglayer) from the SDK
# root CMakeLists), place the produced shared library next to the SDK
# binaries in ${CMAKE_HOME_DIRECTORY}/bin/ so that BuildSDK / packaging
# scripts can ferry both ngsdk and the layer with a single copy.
#
# When built standalone (e.g. via the local build.py), leave the artifact
# in the default location — build.py uses --output-dir to stage it.
if(DEFINED FFX_BUILD_NGLAYER OR EXISTS "${CMAKE_HOME_DIRECTORY}/bin")
    set_target_properties(NGLayer PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY               "${CMAKE_HOME_DIRECTORY}/bin"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG         "${CMAKE_HOME_DIRECTORY}/bin"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE       "${CMAKE_HOME_DIRECTORY}/bin"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_HOME_DIRECTORY}/bin"
        LIBRARY_OUTPUT_DIRECTORY               "${CMAKE_HOME_DIRECTORY}/bin"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG         "${CMAKE_HOME_DIRECTORY}/bin"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE       "${CMAKE_HOME_DIRECTORY}/bin"
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_HOME_DIRECTORY}/bin"
        ARCHIVE_OUTPUT_DIRECTORY               "${CMAKE_HOME_DIRECTORY}/bin"
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG         "${CMAKE_HOME_DIRECTORY}/bin"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE       "${CMAKE_HOME_DIRECTORY}/bin"
        ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_HOME_DIRECTORY}/bin"
    )

    # Copy the platform-appropriate manifest into ${CMAKE_HOME_DIRECTORY}/bin
    # alongside the freshly built layer so that consumers picking up the bin/
    # snapshot get a self-contained Vulkan layer drop.
    if(ANDROID)
        set(_nglayer_manifest "${CMAKE_CURRENT_SOURCE_DIR}/${NGLAYER_MANIFEST_ANDROID}")
        set(_nglayer_manifest_output "${CMAKE_HOME_DIRECTORY}/bin/${NGLAYER_MANIFEST_ANDROID}")
    elseif(WIN32)
        set(_nglayer_manifest "${CMAKE_CURRENT_SOURCE_DIR}/${NGLAYER_MANIFEST_WINDOWS}")
        set(_nglayer_manifest_output "${CMAKE_HOME_DIRECTORY}/bin/${NGLAYER_MANIFEST_WINDOWS}")
    else()
        set(_nglayer_manifest "${CMAKE_CURRENT_SOURCE_DIR}/${NGLAYER_MANIFEST_LINUX}")
        set(_nglayer_manifest_output "${CMAKE_HOME_DIRECTORY}/bin/${NGLAYER_MANIFEST_LINUX}")
    endif()
    add_custom_command(TARGET NGLayer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_nglayer_manifest}"
                "${_nglayer_manifest_output}"
        COMMENT "NGLayer: staging Vulkan layer manifest into bin/"
        VERBATIM
    )
endif()
