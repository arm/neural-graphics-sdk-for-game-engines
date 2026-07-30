#!/usr/bin/env bash
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

# Fetch the external repositories required to build the samples.
#
# Each dependency listed in samples/third_party/manifest.tsv is partially
# cloned and sparse-checked-out to the directories listed in
# samples/third_party/sparse.tsv, so only the files needed by the sample
# build are materialized.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
manifest_path="${repo_root}/samples/third_party/manifest.tsv"
sparse_config_path="${repo_root}/samples/third_party/sparse.tsv"

get_sparse_paths() {
    awk -F '\t' -v target="$1" 'NR > 1 && $1 == target { print $2; exit }' "${sparse_config_path}"
}

# Third-party libraries are CMake sub-projects, so their repository-root build
# files are always kept; the asset payload only needs the listed subdirectories.
build_patterns() {
    local rel_path="$1" sparse_csv="$2" dir
    if [[ "${rel_path}" == samples/third_party/* ]]; then
        printf '/*\n!/*/\n'
    fi
    [[ -z "${sparse_csv}" ]] && return 0
    local IFS=';'
    for dir in ${sparse_csv}; do
        dir="${dir#/}"; dir="${dir%/}"
        [[ -n "${dir}" ]] && printf '/%s/\n' "${dir}"
    done
}

# The `|| [[ -n ... ]]` keeps the final row even when the manifest has no
# trailing newline.
tail -n +2 "${manifest_path}" | while IFS=$'\t' read -r rel_path url commit version || [[ -n "${rel_path}" ]]; do
    [[ -z "${rel_path}" ]] && continue

    target_path="${repo_root}/${rel_path}"
    echo "==> ${rel_path} @ ${commit}"

    if [[ ! -e "${target_path}/.git" ]]; then
        # Remove any leftovers from an interrupted clone before fetching again.
        rm -rf "${target_path}"
        git clone --filter=blob:none --no-checkout "${url}" "${target_path}"
    fi

    build_patterns "${rel_path}" "$(get_sparse_paths "${rel_path}")" \
        | git -C "${target_path}" sparse-checkout set --no-cone --stdin
    git -C "${target_path}" fetch --tags origin
    git -C "${target_path}" checkout --force --detach "${commit}"
done

echo "Sample dependencies are ready."
