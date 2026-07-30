# Copyright (c) 2026, Arm Limited and Contributors
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

set(SAMPLE_DEPENDENCY_MANIFEST "${SAMPLES_ROOT}/third_party/manifest.tsv")

function(require_sample_dependencies)
    file(STRINGS "${SAMPLE_DEPENDENCY_MANIFEST}" _rows)

    set(_missing "")
    foreach(_row IN LISTS _rows)
        string(REPLACE "\t" ";" _columns "${_row}")
        list(GET _columns 0 _path)
        if(_path STREQUAL "path" OR _path STREQUAL "")
            continue()
        endif()
        if(NOT EXISTS "${SAMPLES_ROOT}/../${_path}")
            list(APPEND _missing "  - ${_path}")
        endif()
    endforeach()

    if(_missing)
        string(JOIN "\n" _missing_text ${_missing})
        message(FATAL_ERROR
            "Sample external dependencies are missing:\n${_missing_text}\n\n"
            "Fetch them before building the samples:\n"
            "  Windows: powershell -ExecutionPolicy Bypass -File samples/scripts/fetch_sample_dependencies.ps1\n"
            "  Linux:   bash samples/scripts/fetch_sample_dependencies.sh")
    endif()
endfunction()