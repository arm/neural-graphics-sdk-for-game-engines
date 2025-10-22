# This file is part of the FidelityFX SDK.
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.23)
message(STATUS "Start configure sdk-core")
message(STATUS "FFX_BUILD_AS_DLL = ${FFX_BUILD_AS_DLL}")

if(POLICY CMP0147)
	# parallel custom command builds (used for shader compilation)
	cmake_policy(SET CMP0147 NEW)
endif()

message(STATUS "FFX_API_BACKEND is ${FFX_API_BACKEND}")

if (FFX_GRAPHICS_API STREQUAL "dx12")
	message(FATAL_ERROR "DX12 backend is currently not supported. Please use Vulkan (vk) as the graphics API.")
elseif(FFX_GRAPHICS_API STREQUAL "vk")
	message(STATUS "Building FidelityFX SDK with VK backend")
else()
	message(FATAL_ERROR "Unsupported graphics API!")
endif()

# Pre-compile shaders
set(FFX_AUTO_COMPILE_SHADERS ON CACHE BOOL "Compile shaders automatically as a prebuild step.")

if(CMAKE_GENERATOR STREQUAL "Ninja")
    set(USE_DEPFILE TRUE)
else()
    set(USE_DEPFILE FALSE)
endif()

# Set Visual Studio version
if(CMAKE_GENERATOR STREQUAL "Visual Studio 14 2015")
    set(FFX_VS_VERSION 2015)
elseif(CMAKE_GENERATOR STREQUAL "Visual Studio 15 2017")
	set(FFX_VS_VERSION 2017)
elseif(CMAKE_GENERATOR STREQUAL "Visual Studio 16 2019")
    set(FFX_VS_VERSION 2019)
elseif(CMAKE_GENERATOR STREQUAL "Visual Studio 17 2022")
	set(FFX_VS_VERSION 2022)
endif()

if (NOT FFX_FSR3_AS_LIBRARY)
	# Set configuration types based on what we are building for
	if(MSVC)
		# Embed PDBs in the debug versions of the libs
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Z7 /Od")
	endif()

	# Write both debug and release versions of the static libs to the /bin folder as they are uniquely named
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY   ${CMAKE_HOME_DIRECTORY}/bin/ffx_sdk/)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_HOME_DIRECTORY}/bin/ffx_sdk/)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_HOME_DIRECTORY}/bin/ffx_sdk/)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_HOME_DIRECTORY}/bin/ffx_sdk/)

	if (FFX_BUILD_AS_DLL)
		# Write both debug and release versions of the dlls to the /bin folder as they are uniquely named
		set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin/ffx_sdk/)
		set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_HOME_DIRECTORY}/bin/ffx_sdk/)
		set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_HOME_DIRECTORY}/bin/ffx_sdk/)
		set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_HOME_DIRECTORY}/bin/ffx_sdk/)
		set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS FFX_BUILD_AS_DLL)
	endif()
endif()

# Add requred compile definitions
add_compile_definitions(_UNICODE)
add_compile_definitions(UNICODE)

if(FFX_VS_VERSION STREQUAL 2015 OR FFX_VS_VERSION STREQUAL 2017)
    message(NOTICE "Forcing the SDK path for VS 2015 and VS 2017")
    set(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION "10.0.18362.0")
endif()

# Setup common variables
if(WIN32)
	set(FFX_SC_EXECUTABLE ${CMAKE_CURRENT_SOURCE_DIR}/tools/binary_store/FidelityFX_SC.exe)
	set(VGF_PARSER_EXECUTABLE ${CMAKE_CURRENT_SOURCE_DIR}/tools/binary_store/Model_Parser.exe)
else()
	set(FFX_SC_EXECUTABLE ${CMAKE_CURRENT_SOURCE_DIR}/tools/binary_store/FidelityFX_SC)
	# we need to use the glslangValidator that supports the VulkanML
	set(FFX_GLSLANG_EXE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tools/binary_store/glslangValidator)
	set(VGF_PARSER_EXECUTABLE ${CMAKE_CURRENT_SOURCE_DIR}/tools/binary_store/Model_Parser)
endif()

set(FFX_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(FFX_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libs)
set(FFX_BIN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin/ffx_sdk)
set(FFX_SHARED_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src/shared)
set(FFX_HOST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include/FidelityFX/host)
set(FFX_GPU_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include/FidelityFX/gpu)
set(FFX_COMPONENTS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src/components)
set(FFX_HOST_BACKENDS_PATH ${FFX_HOST_PATH}/backends)
set(FFX_SRC_BACKENDS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src/backends)

# ------------------------------------------------------
# Define common source groupings used by all effects

# Set shared and global public sources
file(GLOB FFX_SHARED_SOURCES
    "${FFX_SHARED_PATH}/*.cpp"
    "${FFX_SHARED_PATH}/*.h")

set(FFX_PUBLIC_SOURCES
    "${FFX_HOST_PATH}/ffx_assert.h"
    "${FFX_HOST_PATH}/ffx_error.h"
	"${FFX_HOST_PATH}/ffx_interface.h"
    "${FFX_HOST_PATH}/ffx_types.h"
    "${FFX_HOST_PATH}/ffx_util.h")

# Init shader sources (for easy viewing)
set(FFX_PUBLIC_SHADER_SOURCES
    "${FFX_GPU_PATH}/ffx_common_types.h"
    "${FFX_GPU_PATH}/ffx_core.h"
    "${FFX_GPU_PATH}/ffx_core_cpu.h"
    "${FFX_GPU_PATH}/ffx_core_glsl.h"
    "${FFX_GPU_PATH}/ffx_core_gpu_common.h"
    "${FFX_GPU_PATH}/ffx_core_gpu_common_half.h"
    "${FFX_GPU_PATH}/ffx_core_hlsl.h"
    "${FFX_GPU_PATH}/ffx_core_portability.h")

if(NOT FFX_FSR3_AS_LIBRARY)
	set(PROJ_NAME "FidelityFX-SDK_${FFX_API_BACKEND}")
	project(PROJ_NAME)
	message(STATUS "Creating project ${PROJ_NAME}")
endif()

# Components
if(FFX_NSS OR FFX_ALL)
	add_subdirectory(${FFX_COMPONENTS_PATH}/nss)
endif()

# Add appropriate graphics backend if requested
if (FFX_GRAPHICS_API STREQUAL "vk")
	add_subdirectory(${FFX_SRC_BACKENDS_PATH}/vk)
endif()

option(BUILD_TOOLS "Build the tools" OFF)
message(STATUS "Build tools: ${BUILD_TOOLS}")
if(BUILD_TOOLS)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tools/ffx_shader_compiler)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tools/ffx_model_parser)
endif()
