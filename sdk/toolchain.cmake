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

mark_as_advanced(CMAKE_TOOLCHAIN_FILE)
message(STATUS "PULLING IN TOOLCHAIN_FILE")

if(_TOOLCHAIN_)
  return()
endif()

# Set configuration options
set(CMAKE_CONFIGURATION_TYPES Debug Release RelWithDebInfo)
set(CMAKE_DEBUG_POSTFIX d)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Get warnings for everything
if (CMAKE_COMPILER_IS_GNUCC)
	set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall")
endif()

if (MSVC)
	# Recommended compiler switches:
	# /fp:fast (Fast floating-point optimizations)
	# /GR (RTTI): if not using typeid or dynamic_cast, we can disable RTTI to save binary size using /GR-
	# /GS (buffer security check)
	# /Gy (enable function-level linking)
	set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W3 /GR- /fp:fast /GS /Gy")
endif()

if (FFX_OS_NAME STREQUAL "windows")
	if(not WIN32)
		message(FATAL_ERROR "The targeted os is not windows, while FFX_OS_NAME=${FFX_OS_NAME}")
	endif()
	set(CMAKE_RELWITHDEBINFO_POSTFIX drel)
	set(CMAKE_SYSTEM_NAME WINDOWS)
	set(CMAKE_SYSTEM_VERSION 10.0)

	set(CMAKE_GENERATOR_PLATFORM "x64" CACHE STRING "" FORCE)
	set(CMAKE_VS_PLATFORM_NAME "x64" CACHE STRING "" FORCE)
	
	# Ensure our platform toolset is x64
	set(CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE "x64" CACHE STRING "" FORCE)
endif()

set(_TOOLCHAIN_ ON)