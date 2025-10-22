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
message(STATUS "Start configure arm-ng-sdk")
project (neural_graphics_sdk)

# FFX_GRAPHICS_API is the graphics API: VK, DX12, etc.
# FFX_OS_NAME is the operation system name: LINUX, WINDOWS, ANDROID, etc.
# FFX_ARCH_NAME is the architecture: X64, ARM, etc.
# FFX_PLATFORM_NAME is {FFX_OS_NAME}_{FFX_ARCH_NAME}: LINUX_X64, ANROID_ARM, etc.
# FFX_API_BACKEND is {FFX_GRAPHICS_API}_{FFX_PLATFORM_NAME}: VK_LINUX_X64, VK_ANROID_ARM, DX12_WINDOWS_X64, etc.
set(FFX_GRAPHICS_API "Unknown")
set(FFX_OS_NAME "Unknown")
set(FFX_ARCH_NAME "Unknown")
if(NOT DEFINED FFX_API_BACKEND)
    message(WARNING "The FFX_API_BACKEND is not provided, using vk_windows_x64 as default value.")
    set(FFX_API_BACKEND vk_windows_x64)
endif()

string(TOLOWER "${FFX_API_BACKEND}" FFX_API_BACKEND)
message(STATUS "FFX_API_BACKEND=${FFX_API_BACKEND}")
message(STATUS "Parse the FFX_GRAPHICS_API, FFX_OS_NAME, and FFX_ARCH_NAME from the FFX_API_BACKEND")
string(REGEX MATCH "^([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)$" _match "${FFX_API_BACKEND}")
if(_match)
    string(REGEX REPLACE "^([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)$" "\\1" FFX_GRAPHICS_API "${FFX_API_BACKEND}")
    string(REGEX REPLACE "^([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)$" "\\2" FFX_OS_NAME "${FFX_API_BACKEND}")
    string(REGEX REPLACE "^([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)$" "\\3" FFX_ARCH_NAME "${FFX_API_BACKEND}")
    set(FFX_PLATFORM_NAME "${FFX_OS_NAME}_${FFX_ARCH_NAME}")
else()
    message(FATAL_ERROR "Unsupported FFX_API_BACKEND: ${FFX_API_BACKEND}")
endif()

message(STATUS "API: ${FFX_GRAPHICS_API}, OS: ${FFX_OS_NAME}, ARCH: ${FFX_ARCH_NAME}")

# some architecture verification
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(STATUS "Detected x64 architecture")
    if(NOT FFX_ARCH_NAME STREQUAL "x64")
        message(FATAL_ERROR "The target architecture is ${CMAKE_SYSTEM_PROCESSOR} which does not match FFX_ARCH_NAME")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|armv7l|aarch64|arm64)$")
    message(STATUS "Detected arm architecture")
    if(NOT FFX_ARCH_NAME STREQUAL "arm")
        message(FATAL_ERROR "The target architecture is ${CMAKE_SYSTEM_PROCESSOR} which does not match FFX_ARCH_NAME")
    endif()
else()
    message(STATUS "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# some operating system verfication
if(WIN32)
    set(WINDOWS TRUE)
    message("++ Building for Windows")
    if(NOT FFX_OS_NAME STREQUAL "windows")
        message(FATAL_ERROR "The targeted OS does not match FFX_API_BACKEND")
    endif()

    # unusual os + arch warning
    if(NOT FFX_ARCH_NAME STREQUAL "x64")
		message(WARNING "The architecture of windows is usually 'x64', while FFX_ARCH_NAME=${FFX_ARCH_NAME}")
	endif()
elseif(APPLE)
    set(MACOS TRUE)
    message("++ Building for MacOS")
    if(NOT FFX_OS_NAME STREQUAL "macos")
        message(FATAL_ERROR "The targeted OS does not match FFX_API_BACKEND")
    endif()
elseif(ANDROID)
    set(ANDROID TRUE)
    message("++ Building for Android")
    if(NOT FFX_OS_NAME STREQUAL "android")
        message(FATAL_ERROR "The targeted OS does not match FFX_API_BACKEND")
    endif()

    # unusual os + arch warning
    if(NOT FFX_ARCH_NAME STREQUAL "arm")
		message(WARNING "The architecture of android is usually 'arm', while FFX_ARCH_NAME=${FFX_ARCH_NAME}")
	endif()
elseif(UNIX)
    set(LINUX TRUE)
    message("++ Building for Linux")
    if(NOT FFX_OS_NAME STREQUAL "linux")
        message(FATAL_ERROR "The targeted OS does not match FFX_API_BACKEND")
    endif()
else()
    message(FATAL_ERROR "Unknown or unsupported platform")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(_DEBUG)
endif()

if(FFX_FSR3_AS_LIBRARY)
    message(STATUS "Building FSR3 as a third-party library")
endif()

# The FFX_BUILD_AS_DLL will be used to control whether to build the armng sdk as a shared library.
# The FFX_BUILD_BACKEND_AS_DLL and FFX_BUILD_COMPONENT_AS_DLL control how the backend and components are linked.
if(NOT DEFINED FFX_BUILD_AS_DLL)
    set(FFX_BUILD_AS_DLL OFF)
endif()

if(NOT DEFINED FFX_BUILD_BACKEND_AS_DLL)
    set(FFX_BUILD_BACKEND_AS_DLL OFF)
endif()

if(NOT DEFINED FFX_BUILD_COMPONENT_AS_DLL)
    set(FFX_BUILD_COMPONENT_AS_DLL OFF)
endif()

if(NOT FFX_BUILD_AS_DLL)
    set(FFX_BUILD_BACKEND_AS_DLL OFF)
    set(FFX_BUILD_COMPONENT_AS_DLL OFF)
endif()

set(FFX_GLSL ON)
set(FFX_NSS ON)

add_subdirectory(./ffx-api)
