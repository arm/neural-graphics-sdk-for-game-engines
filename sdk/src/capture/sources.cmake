# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

set(ARM_CAPTURE_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/worker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/worker.h
    ${CMAKE_CURRENT_LIST_DIR}/session.cpp
    ${CMAKE_CURRENT_LIST_DIR}/session.h
    ${CMAKE_CURRENT_LIST_DIR}/transaction.cpp
    ${CMAKE_CURRENT_LIST_DIR}/transaction.h
    ${CMAKE_CURRENT_LIST_DIR}/execution.h
    ${CMAKE_CURRENT_LIST_DIR}/config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/config.h
    ${CMAKE_CURRENT_LIST_DIR}/file.cpp
    ${CMAKE_CURRENT_LIST_DIR}/file.h
    ${CMAKE_CURRENT_LIST_DIR}/json.cpp
    ${CMAKE_CURRENT_LIST_DIR}/json.hpp
    ${CMAKE_CURRENT_LIST_DIR}/format.cpp
    ${CMAKE_CURRENT_LIST_DIR}/format.h
    ${CMAKE_CURRENT_LIST_DIR}/meta.h
    ${CMAKE_CURRENT_LIST_DIR}/log.cpp
    ${CMAKE_CURRENT_LIST_DIR}/log.h
    ${CMAKE_CURRENT_LIST_DIR}/utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils.h
    ${CMAKE_CURRENT_LIST_DIR}/allocator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/staging_pool.cpp
    ${CMAKE_CURRENT_LIST_DIR}/staging_pool.h
    ${CMAKE_CURRENT_LIST_DIR}/allocator.h
    ${CMAKE_CURRENT_LIST_DIR}/payload.cpp
    ${CMAKE_CURRENT_LIST_DIR}/payload.h
    ${CMAKE_CURRENT_LIST_DIR}/readback.cpp
    ${CMAKE_CURRENT_LIST_DIR}/readback.h
)
