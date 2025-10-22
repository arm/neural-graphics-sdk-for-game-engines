# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.17)
message(STATUS "Configure ffx_model_parser")

# General language options (require language standards specified)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Get warnings for everything
if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall")
endif()
if (MSVC)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W3")
endif()

project(Model_Parser)

if(WIN32)
	# Check for Visual Studio 2019's build tooling
	if(MSVC_TOOLSET_VERSION VERSION_LESS 142)
		message(FATAL_ERROR "Cannot find MSVC toolset version 142 or greater. Please make sure Visual Studio 2019 or newer installed")
	endif()
endif()

find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIRS})

# Generate the output binary in the /bin directory
if(WIN32)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)
elseif(UNIX)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)
endif()

# Set sources
file(GLOB sources
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp")

# Setup target binary
add_executable(${PROJECT_NAME} ${sources})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Add link time dependencies
target_link_libraries(${PROJECT_NAME} ${Vulkan_LIBRARIES})
if(WIN32)
	target_link_libraries(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/libs/vgf/vgf.lib)
elseif(UNIX)
	target_link_libraries(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/libs/vgf/libvgf.a)
endif()

target_include_directories (${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libs/vgf)
