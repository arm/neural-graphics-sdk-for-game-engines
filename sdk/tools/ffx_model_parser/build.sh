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

echo
echo "Building SDK Model Parser Tool"
echo

# Check if build directory exists, create if not
rm -rf build
rm -rf bin
mkdir -p build

# Clear out CMakeCache
rm -rf CMakeFiles
rm -f CMakeCache.txt

cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
# Build the project with parallel jobs
cmake --build . --verbose --parallel 4

cd ..

echo ""
echo " Build finished"

echo "Copy the Model_Parser executable to ../binary_store/"
rsync -avzh --progress bin/Model_Parser ../binary_store/