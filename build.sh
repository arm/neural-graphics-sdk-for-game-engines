#!/bin/bash

set -e

# ------------------------------------------------------
#       helper functions
# ------------------------------------------------------
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
function print_title()
{
    content=$1
    echo ""
    echo "============================================"
    echo "\t${content}"
    echo "============================================"
    echo ""
}

function build_x64_helper()
{
    build_type=$1
    print_title "Build x64 ${build_type}"

    out="build_x64/${build_type}"
    mkdir -p ${out}
    cmake -B ${out} -DFFX_API_BACKEND=vk_linux_x64 -DFFX_FSR3_AS_LIBRARY=OFF
    cmake --build ${out} --verbose -- -j
}

# Build debug
function build_x64_debug()
{
    build_x64_helper "Debug"
}

# Build release
function build_x64_release()
{
    build_x64_helper "Release"
}

# ------------------------------------------------------
#       invoke the build function here
# ------------------------------------------------------
build_x64_release

