@echo off
setlocal enabledelayedexpansion

echo Checking pre-requisites...

:: Check if CMake is installed
where cmake >nul 2>&1
if errorlevel 1 (
    echo Cannot find path to cmake. Is CMake installed? Exiting...
    exit /b 1
) else (
    echo    CMake      - Ready.
)

echo.
echo Building SDK Model Parser Tool
echo.

:: Remove build and bin directories if they exist
if exist build rmdir /s /q build
if exist bin rmdir /s /q bin
mkdir build

:: Remove CMake cache files if they exist
if exist CMakeFiles rmdir /s /q CMakeFiles
if exist CMakeCache.txt del /f /q CMakeCache.txt

:: The vgf.lib added in the repo is built with releaes mode, so we need to build the model parser with release mode.
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build ./ --config Release --verbose --parallel 4

cd ..

echo.
echo  Build finished

echo Copy the Model_Parser executable to ..\binary_store\
if not exist ..\binary_store mkdir ..\binary_store
xcopy /y /v bin\Release\Model_Parser.exe ..\binary_store\