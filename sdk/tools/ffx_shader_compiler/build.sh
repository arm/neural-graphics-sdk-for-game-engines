#!/bin/bash

set -e

echo "Checking pre-requisites..."

# Check if CMake is installed
if ! command -v cmake &> /dev/null; then
    echo "Cannot find path to cmake. Is CMake installed? Exiting..."
    exit 1
else
    echo "   CMake      - Ready."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd ${SCRIPT_DIR}

echo
echo "Building FidelityFX Shader Compiler Tool"
echo

# Check if build directory exists, create if not
rm -rf build
rm -rf bin
mkdir -p build

# Clear out CMakeCache
rm -rf CMakeFiles
rm -f CMakeCache.txt

pushd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    # Build the project with parallel jobs
    cmake --build . --verbose --parallel 4
popd

echo ""
echo " Build finished"

echo "Copy the FidelityFX_SC executable to ../binary_store/"
rsync -avzh --progress bin/FidelityFX_SC ../binary_store/
rsync -avzh --progress ${SCRIPT_DIR}/libs/glslangValidator/bin/x64/glslangValidator ../binary_store/
