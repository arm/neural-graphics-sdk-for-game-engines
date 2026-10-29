# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

# Shared capture implementation. This stays below providers and components:
# algorithms supply semantic/timing hooks; this target owns readback mechanics.
include(${CMAKE_CURRENT_LIST_DIR}/sources.cmake)
add_library(arm_capture_${FFX_PLATFORM_NAME} STATIC ${ARM_CAPTURE_SOURCES})

target_include_directories(arm_capture_${FFX_PLATFORM_NAME}
    PUBLIC ${FFX_INCLUDE_PATH}
    PRIVATE
        # Give the SDK's own ARM-enabled Vulkan headers priority so vk_wrapper.h
        # resolves <vulkan/vulkan_core.h> to the bundled copy (which declares
        # VK_ARM_data_graph) instead of a host engine's stock Vulkan headers.
        ${FFX_INCLUDE_PATH}/vulkan-headers
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../ffx-api/include)

find_package(Vulkan REQUIRED)
target_link_libraries(arm_capture_${FFX_PLATFORM_NAME} PRIVATE Vulkan::Headers ffx_logging)
target_link_libraries(arm_capture_${FFX_PLATFORM_NAME} PRIVATE ffx_tinyexr)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(arm_capture_${FFX_PLATFORM_NAME} PRIVATE stdc++fs)
endif()
target_compile_definitions(arm_capture_${FFX_PLATFORM_NAME}
    PRIVATE
        FFX_CAPTURE_ENABLED=$<BOOL:${FFX_CAPTURE_ENABLED}>)

set_target_properties(arm_capture_${FFX_PLATFORM_NAME} PROPERTIES FOLDER Components)
