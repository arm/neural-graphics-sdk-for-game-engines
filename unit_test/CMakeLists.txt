#[[
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT
]]

cmake_minimum_required(VERSION 3.21)

project(ngsdk_unit_test LANGUAGES CXX)

find_package(Vulkan REQUIRED)

set(UNIT_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/framework/test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/framework/vk_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend_suites/backend_vk_suite.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/component_suites/nss_suite.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/component_suites/nfru_suite.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/component_suites/nfru_resolution_suite.cpp
)

if(FFX_CAPTURE_ENABLED)
    list(APPEND UNIT_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/component_suites/capture_suite.cpp)
endif()

add_executable(sdk_unit_test ${UNIT_TEST_SOURCES})

set(UNIT_TEST_SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/framework/shader")
add_custom_command(TARGET sdk_unit_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:sdk_unit_test>/shader"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${UNIT_TEST_SHADER_SOURCE_DIR}/triangle.vert.spv"
        "$<TARGET_FILE_DIR:sdk_unit_test>/shader/triangle.vert.spv"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${UNIT_TEST_SHADER_SOURCE_DIR}/triangle.frag.spv"
        "$<TARGET_FILE_DIR:sdk_unit_test>/shader/triangle.frag.spv"
)

target_include_directories(sdk_unit_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/ffx-api/include
        ${CMAKE_SOURCE_DIR}/ffx-api/src
        ${CMAKE_SOURCE_DIR}/sdk/include
        ${CMAKE_SOURCE_DIR}/sdk/src
        ${CMAKE_SOURCE_DIR}/sdk/src/shared
        ${CMAKE_SOURCE_DIR}/sdk/include/vulkan-headers
        ${CMAKE_SOURCE_DIR}/sdk/src/components/nss
)

target_compile_definitions(sdk_unit_test
    PRIVATE
        FFX_BACKEND_VK
        FFX_UNIT_TEST
        FFX_CAPTURE_ENABLED=$<BOOL:${FFX_CAPTURE_ENABLED}>
)

if(FFX_FI OR FFX_ALL)
    target_compile_definitions(sdk_unit_test PRIVATE FFX_FI)
endif()

if(FFX_NSS OR FFX_ALL)
    target_compile_definitions(sdk_unit_test PRIVATE FFX_NSS)
endif()

set(CAPTURE_UNIT_TEST_LIBRARIES)
if(FFX_CAPTURE_ENABLED)
    list(APPEND CAPTURE_UNIT_TEST_LIBRARIES arm_capture_${FFX_PLATFORM_NAME})
endif()

if(MSVC)
    target_link_libraries(sdk_unit_test
        PRIVATE
            "$<LINK_LIBRARY:WHOLE_ARCHIVE,ffx_backend_vk_${FFX_PLATFORM_NAME}>"
            # Resolve capture locally before the DLL import of its EXR converter.
            ${CAPTURE_UNIT_TEST_LIBRARIES}
            ${SDK_LIB_NAME}
            ffx_logging
            ffx_frameinterpolation_${FFX_PLATFORM_NAME}
            ffx_nss_${FFX_PLATFORM_NAME}
            Vulkan::Vulkan
    )
else()
    target_link_libraries(sdk_unit_test
        PRIVATE
            ${SDK_LIB_NAME}
            ffx_logging
            ffx_frameinterpolation_${FFX_PLATFORM_NAME}
            ffx_nss_${FFX_PLATFORM_NAME}
            ${CAPTURE_UNIT_TEST_LIBRARIES}
            Vulkan::Vulkan
    )
endif()

set_target_properties(sdk_unit_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_HOME_DIRECTORY}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_HOME_DIRECTORY}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_HOME_DIRECTORY}/bin
)
