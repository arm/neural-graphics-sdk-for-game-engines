# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

add_library(sample_common OBJECT
	exr_image.cpp
	exr_image.h)

target_include_directories(sample_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(sample_common PRIVATE framework)
