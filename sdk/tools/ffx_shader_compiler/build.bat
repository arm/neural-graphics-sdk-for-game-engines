@echo off
setlocal enabledelayedexpansion

echo Checking pre-requisites...

:: Check if CMake is installed
where cmake >nul 2>nul
if errorlevel 1 (
    echo Cannot find path to cmake. Is CMake installed? Exiting...
    exit /b 1
) else (
    echo    CMake      - Ready.
)

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo.
echo Building FidelityFX Shader Compiler Tool
echo.

:: Remove old build and bin directories if they exist
if exist build rmdir /s /q build
if exist bin rmdir /s /q bin
mkdir build

:: Clear out CMakeCache
if exist CMakeFiles rmdir /s /q CMakeFiles
if exist CMakeCache.txt del /f /q CMakeCache.txt

set BUILD_TYPE=Release
pushd build
    cmake .. -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
    cmake --build . --verbose --config %BUILD_TYPE% -- /m:4
popd

echo.
echo  Build finished

echo Copy the FidelityFX_SC executable to ..\binary_store\
@REM if not exist ..\binary_store mkdir ..\binary_store
echo %SCRIPT_DIR%bin\%BUILD_TYPE%\FidelityFX_SC.exe
echo %SCRIPT_DIR%..\binary_store\
if exist %SCRIPT_DIR%bin\%BUILD_TYPE%\FidelityFX_SC.exe (
    copy /y %SCRIPT_DIR%bin\%BUILD_TYPE%\FidelityFX_SC.exe %SCRIPT_DIR%..\binary_store\
)
if exist "%SCRIPT_DIR%libs\glslangValidator\bin\x64\glslangValidator.exe" (
    copy /y "%SCRIPT_DIR%libs\glslangValidator\bin\x64\glslangValidator.exe" ..\binary_store\
)
