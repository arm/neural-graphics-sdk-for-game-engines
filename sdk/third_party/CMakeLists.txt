# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

find_package(Threads REQUIRED)

add_library(ffx_spdlog STATIC
    spdlog/async.cpp
    spdlog/bundled_fmtlib_format.cpp
    spdlog/color_sinks.cpp
    spdlog/spdlog.cpp
    spdlog/stdout_sinks.cpp)
add_library(ffx::spdlog ALIAS ffx_spdlog)

target_compile_features(ffx_spdlog PUBLIC cxx_std_11)
# Samples link a different static spdlog revision into the same executable.
target_compile_definitions(ffx_spdlog PUBLIC SPDLOG_COMPILED_LIB spdlog=ffx_sdk_spdlog)
target_include_directories(ffx_spdlog SYSTEM PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/spdlog/include")
target_link_libraries(ffx_spdlog PUBLIC Threads::Threads)
set_target_properties(ffx_spdlog PROPERTIES FOLDER "SDK/ThirdParty" POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(ffx_spdlog PRIVATE /Zc:__cplusplus)
    target_compile_options(ffx_spdlog PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

if(ANDROID)
    target_link_libraries(ffx_spdlog PUBLIC log)
endif()

target_compile_definitions(ffx_spdlog PUBLIC FMT_USE_EXCEPTIONS=0 SPDLOG_NO_EXCEPTIONS)
if(MSVC)
    target_compile_options(ffx_spdlog PRIVATE /EHs-c-)
    target_compile_definitions(ffx_spdlog PRIVATE _HAS_EXCEPTIONS=0)
else()
    target_compile_options(ffx_spdlog PRIVATE -fno-exceptions)
endif()
