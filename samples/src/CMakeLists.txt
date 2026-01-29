# Copyright (c) 2019-2026, Arm Limited and Contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 the "License";
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.16)

# Creates a list of the subdirectories within this folder
scan_dirs(
        DIR ${CMAKE_CURRENT_SOURCE_DIR}
        LIST SUB_DIRS)

# List of all samples
set(TOTAL_SAMPLE_ID_LIST)

# Add all sample subdirectories directly (no category folders)
foreach (SUB_DIR ${SUB_DIRS})
    add_subdirectory(${SUB_DIR})
    list(APPEND TOTAL_SAMPLE_ID_LIST ${SUB_DIR})
endforeach ()

# Order of the sample ids
set(ORDER_LIST
    #Performance Samples
    "nss"
    )

# Orders the sample ids by the order list above
set(ORDERED_LIST)
list(REMOVE_DUPLICATES ORDER_LIST)

# Add samples to ordered list based on the a predefined order
foreach (SAMPLE_ID ${ORDER_LIST})
    list(FIND ORDERED_LIST ${SAMPLE_ID} FOUND_SAMPLE)
    if (NOT ${FOUND_SAMPLE} LESS 0)
        list(APPEND ORDERED_LIST ${SAMPLE_ID})
    endif ()
endforeach ()

# Add the remaining samples to the ordered list that are not in the predefined order
foreach (SAMPLE_ID ${TOTAL_SAMPLE_ID_LIST})
    list(FIND ORDERED_LIST ${SAMPLE_ID} FOUND_SAMPLE)
    if (${FOUND_SAMPLE} LESS 0)
        list(APPEND ORDERED_LIST ${SAMPLE_ID})
    endif ()
endforeach ()

# Make sample list visible parent scope (required by vulkan_samples project)
set(TOTAL_SAMPLE_ID_LIST ${ORDERED_LIST} PARENT_SCOPE)
