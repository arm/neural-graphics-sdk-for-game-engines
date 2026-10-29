# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

get_filename_component(_ngsdk_source_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(_ngsdk_release_compile "$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>")

# Apply before adding targets so statically linked dependencies also map __FILE__.
if(MSVC)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options(
            "$<${_ngsdk_release_compile}:/clang:-ffile-prefix-map=${_ngsdk_source_root}=.>"
            "$<${_ngsdk_release_compile}:/clang:-ffile-prefix-map=${CMAKE_BINARY_DIR}=./build>")
    else()
        file(TO_NATIVE_PATH "${_ngsdk_source_root}" _ngsdk_native_source_root)
        file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}" _ngsdk_native_build_root)
        add_compile_options(
            "$<${_ngsdk_release_compile}:/d1trimfile:${_ngsdk_native_source_root}\\>"
            "$<${_ngsdk_release_compile}:/d1trimfile:${_ngsdk_native_build_root}\\>")
    endif()
    add_link_options("$<$<CONFIG:Release>:/DEBUG:NONE>" "$<$<CONFIG:Release>:/INCREMENTAL:NO>")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        "$<${_ngsdk_release_compile}:-g0>"
        "$<${_ngsdk_release_compile}:-ffile-prefix-map=${_ngsdk_source_root}=.>"
        "$<${_ngsdk_release_compile}:-ffile-prefix-map=${CMAKE_BINARY_DIR}=./build>")
    add_link_options("$<$<CONFIG:Release>:LINKER:--strip-all>")
endif()

unset(_ngsdk_native_source_root)
unset(_ngsdk_native_build_root)
unset(_ngsdk_release_compile)
unset(_ngsdk_source_root)
