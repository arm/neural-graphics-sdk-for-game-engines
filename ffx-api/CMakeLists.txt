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

cmake_minimum_required(VERSION 3.21)
message(STATUS "Start configure ffx-api")
message(STATUS "FFX_API_BACKEND=${FFX_API_BACKEND}")

if(POLICY CMP0147)
	# parallel custom command builds (used for shader compilation)
	cmake_policy(SET CMP0147 NEW)
endif()

project(FFX_API_${FFX_API_BACKEND})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W3")
endif()

if(NOT FFX_FSR3_AS_LIBRARY)
	set(CMAKE_CONFIGURATION_TYPES Debug Release RelWithDebInfo)
	set(CMAKE_DEBUG_POSTFIX d)
	set(CMAKE_RELWITHDEBINFO_POSTFIX drel)

	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_HOME_DIRECTORY}/bin)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_HOME_DIRECTORY}/bin)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_HOME_DIRECTORY}/bin)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_HOME_DIRECTORY}/bin)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_HOME_DIRECTORY}/bin)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_HOME_DIRECTORY}/bin)
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_HOME_DIRECTORY}/bin)
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_HOME_DIRECTORY}/bin)
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_HOME_DIRECTORY}/bin)
endif()

add_compile_definitions(FFXDLL_VERSION_MAJOR=1)
add_compile_definitions(FFXDLL_VERSION_MINOR=0)
add_compile_definitions(FFXDLL_VERSION_PATCH=0)

if (DEFINED ENV{CI_PIPELINE_ID})
add_compile_definitions(CI_PIPELINE_ID=$ENV{CI_PIPELINE_ID})
else()
add_compile_definitions(CI_PIPELINE_ID=0)
endif()

file(GLOB private_ffx_api
	${CMAKE_CURRENT_SOURCE_DIR}/src/resource/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/backends.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/backends.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/ffx_api.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/ffx_provider_external.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/ffx_provider.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/ffx_provider.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/validation.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/validation.h
	)

if(FFX_NSS OR FFX_ALL)
	set(private_ffx_api_nss
		${CMAKE_CURRENT_SOURCE_DIR}/src/ffx_provider_nss.h
		${CMAKE_CURRENT_SOURCE_DIR}/src/ffx_provider_nss.cpp)
	list(APPEND private_ffx_api ${private_ffx_api_nss})
endif()

file(GLOB public_ffx_api
	${CMAKE_CURRENT_SOURCE_DIR}/include/ffx_api/*.h
	${CMAKE_CURRENT_SOURCE_DIR}/include/ffx_api/*.hpp)

if (FFX_GRAPHICS_API STREQUAL vk)
	message(WARNING "Setting FFX_BACKEND_VK")
	add_compile_definitions(FFXDLL_VERSION_API=VK)
	set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS FFX_BACKEND_VK)

	file(GLOB public_ffx_api_backend_vk
		${CMAKE_CURRENT_SOURCE_DIR}/include/ffx_api/vk/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/include/ffx_api/vk/*.hpp)
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../..")
# Use operating system and architecture as the postfix for the library name
# We might consider add graphics api when we start supporting other than vulkan(for example dx12)
set(SDK_LIB_NAME ngsdk_${FFX_PLATFORM_NAME})
if(FFX_BUILD_AS_DLL)
	add_library(${SDK_LIB_NAME} SHARED ${private_ffx_api} ${private_ffx_api_backend_vk} ${public_ffx_api} ${public_ffx_api_backend_vk})
else()
	add_library(${SDK_LIB_NAME} STATIC ${private_ffx_api} ${private_ffx_api_backend_vk} ${public_ffx_api} ${public_ffx_api_backend_vk})
endif()

source_group("private_source" FILES ${private_ffx_api})
source_group("public_source"  FILES ${public_ffx_api})

if (FFX_GRAPHICS_API STREQUAL "vk")
	source_group("private_source\\vk" FILES ${private_ffx_api_backend_vk})
	source_group("public_source\\vk"  FILES ${public_ffx_api_backend_vk})
endif()

target_include_directories(${SDK_LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${SDK_LIB_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../sdk/include)

# FSR dependency, must be built before using BuildFidelityFXSDK.bat

set(FFX_AUTO_COMPILE_SHADERS ON)
add_subdirectory(../sdk sdk)

if(FFX_NSS OR FFX_ALL)
	target_compile_definitions(${SDK_LIB_NAME} PRIVATE FFX_NSS)
    target_link_libraries(${SDK_LIB_NAME} PRIVATE ffx_nss_${FFX_PLATFORM_NAME})
endif()

if (FFX_GRAPHICS_API STREQUAL vk)
	find_package(Vulkan REQUIRED)
	target_link_libraries(${SDK_LIB_NAME} PRIVATE Vulkan::Headers ffx_backend_vk_${FFX_PLATFORM_NAME})
else()
	message(FATAL_ERROR "For now we only support vulkan backend.")
endif()
