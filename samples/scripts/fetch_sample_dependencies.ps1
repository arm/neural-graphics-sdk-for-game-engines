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

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot     = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$manifestPath = Join-Path $repoRoot "samples\third_party\manifest.tsv"
$sparsePath   = Join-Path $repoRoot "samples\third_party\sparse.tsv"

function Invoke-Git {
    & git @args
    if ($LASTEXITCODE -ne 0) {
        throw "git $($args -join ' ') failed with exit code $LASTEXITCODE"
    }
}

# Third-party libraries are CMake sub-projects, so their repository-root build
# files are always kept; the asset payload only needs the listed subdirectories.
function Get-SparsePatterns {
    param([string]$RelativePath, [string]$SparsePaths)

    $patterns = @()
    if ($RelativePath.StartsWith("samples/third_party/")) {
        $patterns += "/*", "!/*/"
    }
    if ($SparsePaths) {
        foreach ($dir in $SparsePaths.Split(';', [StringSplitOptions]::RemoveEmptyEntries)) {
            $patterns += "/$($dir.Trim('/'))/"
        }
    }
    return $patterns
}

$sparsePaths = @{}
foreach ($row in Import-Csv -LiteralPath $sparsePath -Delimiter "`t") {
    $sparsePaths[$row.path] = $row.sparse_paths
}

foreach ($entry in Import-Csv -LiteralPath $manifestPath -Delimiter "`t") {
    $target = Join-Path $repoRoot $entry.path
    Write-Host "==> $($entry.path) @ $($entry.commit)"

    if (-not (Test-Path -LiteralPath (Join-Path $target ".git"))) {
        # Remove any leftovers from an interrupted clone before fetching again.
        if (Test-Path -LiteralPath $target) {
            Remove-Item -LiteralPath $target -Recurse -Force
        }
        Invoke-Git clone --filter=blob:none --no-checkout $entry.url $target
    }

    $patterns = Get-SparsePatterns -RelativePath $entry.path -SparsePaths $sparsePaths[$entry.path]
    Invoke-Git -C $target sparse-checkout set --no-cone @patterns
    Invoke-Git -C $target fetch --tags origin
    Invoke-Git -C $target checkout --force --detach $entry.commit
}

Write-Host "Sample dependencies are ready."
