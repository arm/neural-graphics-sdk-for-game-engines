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

if(NOT ${FFX_API_VK})
    return()
endif()

message(STATUS "Configure sdk/src/backends/vk")
message(STATUS "FFX_BUILD_BACKEND_AS_DLL =${FFX_BUILD_BACKEND_AS_DLL}")

#include CMake custom command generator functions for compiling shaders.
include("${FFX_GPU_PATH}/CMakeCompileShaders.txt")

function (CheckVulkanSDKVersion)
    # set the desired version
    set(vulkan_min_version "1.3.250")
    message(STATUS "Vulkan_VERSION=${Vulkan_VERSION}")

    if (Vulkan_VERSION VERSION_LESS ${vulkan_min_version})
        message(WARNING "Vulkan SDK ${vulkan_min_version} or above is required. Found Vulkan SDK ${Vulkan_VERSION}")
    endif()
endfunction()

# Add your local Vulkan headers directory to the search path
set(Vulkan_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/vulkan-headers")
set(Vulkan_INCLUDE_DIRS "${Vulkan_INCLUDE_DIR}")

message(STATUS "Vulkan_INCLUDE_DIR=${Vulkan_INCLUDE_DIR}")
message(STATUS "Vulkan_INCLUDE_DIRS=${Vulkan_INCLUDE_DIRS}")

find_package(Vulkan REQUIRED)
CheckVulkanSDKVersion()

set(PUBLIC_SHADERS "")
set(PRIVATE_SHADERS "")

file(GLOB PRIVATE_SOURCE
    "${FFX_SHARED_PATH}/ffx_assert.cpp"
	"${FFX_SRC_BACKENDS_PATH}/shared/*.h"
	"${FFX_SRC_BACKENDS_PATH}/shared/*.cpp"
	"${FFX_SRC_BACKENDS_PATH}/shared/blob_accessors/"
	"${FFX_SRC_BACKENDS_PATH}/shared/blob_accessors/"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
	"${FFX_INCLUDE_PATH}/spirv-tools/*.h"
	"${FFX_INCLUDE_PATH}/spirv-tools/*.cpp"
)

if (FFX_NSS OR FFX_ALL)
	set(FFX_NSS_PRIVATE_SOURCE
		"${FFX_SRC_BACKENDS_PATH}/shared/blob_accessors/ffx_nss_shaderblobs.h"
		"${FFX_SRC_BACKENDS_PATH}/shared/blob_accessors/ffx_nss_shaderblobs.cpp")
	list(APPEND PRIVATE_SOURCE ${FFX_NSS_PRIVATE_SOURCE})
	
	# Add shaders for nss
	
	file(GLOB FFX_NSS_PRIVATE_SHADERS
		"${CMAKE_CURRENT_SOURCE_DIR}/shaders/nss/*.glsl"
		"${CMAKE_CURRENT_SOURCE_DIR}/shaders/nss/*.comp")
	list(APPEND PRIVATE_SHADERS ${FFX_NSS_PRIVATE_SHADERS})
	
	file(GLOB FFX_NSS_PUBLIC_SHADERS
		"${FFX_GPU_PATH}/nss/*.h"
		"${FFX_GPU_PATH}/nss/*.glsl"
		"${FFX_GPU_PATH}/nss/*.comp")
	list(APPEND PUBLIC_SHADERS ${FFX_NSS_PUBLIC_SHADERS})
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../../../..")

file(GLOB_RECURSE PUBLIC_SOURCE
    "${FFX_HOST_BACKENDS_PATH}/vk/*.h")

if (FFX_BUILD_BACKEND_AS_DLL)
    add_library(ffx_backend_vk_${FFX_PLATFORM_NAME} SHARED ${PRIVATE_SOURCE} ${PUBLIC_SOURCE} ${PRIVATE_SHADERS} ${PUBLIC_SHADERS})
else()
    add_library(ffx_backend_vk_${FFX_PLATFORM_NAME} STATIC ${PRIVATE_SOURCE} ${PUBLIC_SOURCE} ${PRIVATE_SHADERS} ${PUBLIC_SHADERS})
endif()

if(FFX_OS_NAME STREQUAL "windows")
	if(CMAKE_BUILD_TYPE STREQUAL "Debug")
		message(STATUS "Building in Debug mode")
		target_link_libraries(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE "${FFX_LIB_PATH}/Debug/SPIRV-Tools-opt.lib")
		target_link_libraries(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE "${FFX_LIB_PATH}/Debug/SPIRV-Tools.lib")
	else()
		message(STATUS "Building in Release mode")
		target_link_libraries(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE "${FFX_LIB_PATH}/SPIRV-Tools-opt.lib")
		target_link_libraries(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE "${FFX_LIB_PATH}/SPIRV-Tools.lib")
	endif()
elseif(FFX_OS_NAME STREQUAL "linux")
	if(FFX_ARCH_NAME STREQUAL "x64")
		target_link_libraries(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE "${FFX_LIB_PATH}/libSPIRV-Tools-opt.a")
		target_link_libraries(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE "${FFX_LIB_PATH}/libSPIRV-Tools.a")
	else()
		target_link_libraries(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE "${FFX_LIB_PATH}/libSPIRV-Tools-opt-aarch64.a")
		target_link_libraries(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE "${FFX_LIB_PATH}/libSPIRV-Tools-aarch64.a")
	endif()
elseif(FFX_OS_NAME STREQUAL "android")
	target_link_libraries(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE "${FFX_LIB_PATH}/libSPIRV-Tools-opt-android-arm.a")
	target_link_libraries(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE "${FFX_LIB_PATH}/libSPIRV-Tools-android-arm.a")
endif()

# Depend on vulkan-headers in the sdk/include folder to ensure that the Vulkan headers are available.
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PUBLIC "${Vulkan_INCLUDE_DIR}")
target_link_libraries(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE "Vulkan::Vulkan")

# vk backend source
source_group("private_source"  FILES ${PRIVATE_SOURCE})
source_group("public_source"   FILES ${PUBLIC_SOURCE})
source_group("private_shaders" FILES ${PRIVATE_SHADERS})
source_group("public_shaders"  FILES ${PUBLIC_SHADERS})

get_filename_component(FFX_PASS_SHADER_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../shaders/vk ABSOLUTE)
get_filename_component(FFX_PASS_DATA_GRAPH_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../data_graphs/vk ABSOLUTE)

target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PUBLIC ${FFX_INCLUDE_PATH})
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PUBLIC ${FFX_LIB_PATH})
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PUBLIC ${FFX_PASS_SHADER_OUTPUT_PATH})
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PUBLIC ${FFX_PASS_DATA_GRAPH_OUTPUT_PATH})
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE ${FFX_COMPONENTS_PATH})
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE ${FFX_SHARED_PATH})
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE "${FFX_SRC_BACKENDS_PATH}/shared")
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE ${FFX_PASS_SHADER_OUTPUT_PATH})
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE ${FFX_PASS_DATA_GRAPH_OUTPUT_PATH})

set_source_files_properties(${PRIVATE_SHADERS} PROPERTIES HEADER_FILE_ONLY TRUE)
set_source_files_properties(${PUBLIC_SHADERS} PROPERTIES HEADER_FILE_ONLY TRUE)

file(MAKE_DIRECTORY ${FFX_PASS_SHADER_OUTPUT_PATH})
file(MAKE_DIRECTORY ${FFX_PASS_DATA_GRAPH_OUTPUT_PATH})

if (FFX_AUTO_COMPILE_SHADERS)
    set(FFX_SC_DEPENDENT_TARGET ffx_backend_vk_${FFX_PLATFORM_NAME})
else()
    set(FFX_SC_DEPENDENT_TARGET ffx_backend_vk_shaders_${FFX_PLATFORM_NAME})
    add_custom_target(${FFX_SC_DEPENDENT_TARGET})
endif()

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20.0")
	cmake_policy(SET CMP0116 OLD)
endif()

if (FFX_NSS OR FFX_ALL)
	target_compile_definitions(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE FFX_NSS)
	include (CMakeShadersNSS.txt)
endif()

add_custom_target(ffx_shader_permutations_vk DEPENDS ${FFX_SC_PERMUTATION_OUTPUTS})
add_dependencies(${FFX_SC_DEPENDENT_TARGET} ffx_shader_permutations_vk)

# Make sure shader builds are a dependency of the backend
add_dependencies(ffx_backend_vk_${FFX_PLATFORM_NAME} ffx_shader_permutations_vk)

# Add to solution folder.
set_target_properties(ffx_backend_vk_${FFX_PLATFORM_NAME} PROPERTIES FOLDER Backends)
set_target_properties(ffx_shader_permutations_vk PROPERTIES FOLDER Backends)
