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

cmake_minimum_required(VERSION 3.17)
message(STATUS "Configure ffx_shader_compiler")

project(FidelityFX_SC)

# Enable multi-threaded compilation
if(MSVC)
    add_compile_options(/MP)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W3")
    # Check for Visual Studio 2019's build tooling
    if(MSVC_TOOLSET_VERSION VERSION_LESS 142)
        message(FATAL_ERROR "Cannot find MSVC toolset version 142 or greater. Please make sure Visual Studio 2019 or newer installed")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-pthread)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

if(WIN32)
    # add_definitions(-DPLATFORM_WINDOWS=1)
    if (CMAKE_GENERATOR_PLATFORM STREQUAL x64)
        #Set so taken by default when creating custom configs
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    endif()
    message("++ Building for Windows")
elseif(APPLE)
    # add_definitions(-DPLATFORM_MACOS=1)
    message("++ Building for MacOS")
elseif(UNIX)
    # add_definitions(-DPLATFORM_LINUX=1)
    message("++ Building for Linux")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
else()
    message(FATAL_ERROR "Unknown or unsupported platform")
endif()

# General language options (require language standards specified)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate the output binary in the /bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)

# Copies a set of files to a directory (only if they already don't exist or are different)
# Usage example:
#     set(media_src ${CMAKE_CURRENT_SOURCE_DIR}/../../media/brdfLut.dds )
#     copyCommand("${media_src}" ${CMAKE_HOME_DIRECTORY}/bin)
function(copyCommand list dest)
    foreach(fullFileName ${list})
        get_filename_component(file ${fullFileName} NAME)
        message("Generating custom command for ${fullFileName}")
        add_custom_command(
            OUTPUT   ${dest}/${file}
            PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${dest}
            COMMAND ${CMAKE_COMMAND} -E copy ${fullFileName}  ${dest}
            MAIN_DEPENDENCY  ${fullFileName}
            COMMENT "Updating ${file} into ${dest}"
        )
    endforeach()
endfunction()

#
# Same as copyCommand() but you can give a target name
# This custom target will depend on all those copied files
# Then the target can be properly set as a dependency of other executable or library
# Usage example:
#     add_library(my_lib ...)
#     set(media_src ${CMAKE_CURRENT_SOURCE_DIR}/../../media/brdfLut.dds )
#     copyTargetCommand("${media_src}" ${CMAKE_HOME_DIRECTORY}/bin copied_media_src)
#     add_dependencies(my_lib copied_media_src)
#
function(copyTargetCommand list dest returned_target_name)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    foreach(fullFileName ${list})
        get_filename_component(file ${fullFileName} NAME)
        message("Generating custom command for ${fullFileName}")
        add_custom_command(
            OUTPUT   ${dest}/${file}
            PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${dest}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${fullFileName} ${dest}
            MAIN_DEPENDENCY  ${fullFileName}
            COMMENT "Updating ${file} into ${dest}"
        )
        list(APPEND dest_list ${dest}/${file})
    endforeach()

    add_custom_target(${returned_target_name} DEPENDS "${dest_list}")
    set_target_properties(${returned_target_name} PROPERTIES FOLDER CopyTargets)
endfunction()

# Set sources
file(GLOB sources
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp")

foreach(src ${sources})
    message(STATUS "Source: ${src}")
endforeach()

# Append lib sources	
list(APPEND sources 
	${CMAKE_CURRENT_SOURCE_DIR}/libs/MD5/md5.h 
	${CMAKE_CURRENT_SOURCE_DIR}/libs/MD5/md5.cpp  
	${CMAKE_CURRENT_SOURCE_DIR}/libs/SPIRV-Reflect/spirv_reflect.h 
	${CMAKE_CURRENT_SOURCE_DIR}/libs/SPIRV-Reflect/spirv_reflect.c)


# Setup target binary
add_executable(${PROJECT_NAME} ${sources})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_subdirectory(libs/glslangValidator)
add_subdirectory(libs/tiny-process-library)

set(SPIRV_CROSS_ENABLE_C_API OFF CACHE BOOL "")
set(SPIRV_CROSS_CLI OFF CACHE BOOL "")
add_subdirectory(libs/SPIRV-Cross)

# Add link time dependencies
target_link_libraries (${PROJECT_NAME} glslangValidator tiny-process-library spirv-cross-glsl)
target_include_directories (${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libs/MD5
                                                   ${CMAKE_CURRENT_SOURCE_DIR}/libs/SPIRV-Reflect
                                                   ${CMAKE_CURRENT_SOURCE_DIR}/libs/SPIRV-Cross
                                                   ${CMAKE_CURRENT_SOURCE_DIR}/libs/tiny-process-library)
