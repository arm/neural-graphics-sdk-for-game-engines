#!/usr/bin/env python3
"""
Cross-platform build script for ARM NG SDK.
Supports Windows, Linux x64, Linux ARM64, and Android builds.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def printf(*args, **kwargs):
    """Print with automatic flushing for proper output ordering in redirected streams."""
    kwargs['flush'] = True
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()


def get_default_backend():
    """Determine default backend based on host OS and architecture."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        return "vk_windows_x64"
    elif system == "linux":
        if machine in ["aarch64", "arm64"]:
            return "vk_linux_arm"
        else:
            return "vk_linux_x64"
    elif system == "darwin":
        return "vk_macos"
    else:
        return "vk_linux_x64"


def get_build_dir(backend, build_type):
    """Generate build directory name based on backend and build type."""
    if "android" in backend:
        return f"build_android/{build_type}"
    elif "linux_arm" in backend:
        return f"build_aarch64/{build_type}"
    elif "linux_x64" in backend:
        return f"build_x64/{build_type}"
    elif "windows" in backend:
        return f"build/{build_type}"
    else:
        return f"build_{backend}/{build_type}"


def run_command(cmd, description):
    """Run a command and print it before execution."""
    printf(f"\n{'='*60}")
    printf(f"  {description}")
    printf(f"{'='*60}")
    printf(f"CMake command:\n")
    printf(f"{' '.join(cmd)}\n")
    printf(f"{'='*60}\n")
    
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        printf(f"\nError: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def build(args):
    """Execute the build process."""
    script_dir = Path(__file__).parent.resolve()
    build_dir = script_dir / get_build_dir(args.backend, args.build_type)
    
    # Clean build if requested
    if args.clean:
        if build_dir.exists():
            printf(f"Cleaning build directory: {build_dir}")
            
            # Remove CMake-specific files/folders
            cmake_files_dir = build_dir / "CMakeFiles"
            cmake_cache_file = build_dir / "CMakeCache.txt"
            
            if cmake_files_dir.exists():
                printf(f"  Removing CMakeFiles directory...")
                shutil.rmtree(cmake_files_dir)
            
            if cmake_cache_file.exists():
                printf(f"  Removing CMakeCache.txt...")
                cmake_cache_file.unlink()
            
            # Also remove the entire build directory for a complete clean
            printf(f"  Removing entire build directory...")
            shutil.rmtree(build_dir)
            printf(f"Build directory cleaned")
        else:
            printf(f"Build directory does not exist, skipping clean: {build_dir}")
    
    # Create build directory
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare CMake configure command
    cmake_configure = ["cmake"]
    
    # Add generator if needed
    if "android" in args.backend:
        cmake_configure.extend(["-G", "Ninja"])
    elif platform.system().lower() == "windows" and "windows" in args.backend:
        cmake_configure.extend(["-A", "x64"])
    
    # Add build directory
    cmake_configure.extend(["-B", str(build_dir)])
    
    # Add backend
    cmake_configure.append(f"-DFFX_API_BACKEND={args.backend}")
    
    # Add common options (defaults that can be overridden by cmake_args)
    cmake_configure.append("-DFFX_FSR3_AS_LIBRARY=OFF")
    cmake_configure.append("-DFFX_BUILD_AS_DLL=ON")
    
    # Platform-specific options
    if "android" in args.backend:
        android_ndk = os.environ.get("ANDROID_NDK") or os.environ.get("NDK_ROOT")
        if not android_ndk:
            printf("Error: ANDROID_NDK or NDK_ROOT environment variable not set")
            sys.exit(1)
        
        api_level = args.android_api or "33"
        cmake_configure.extend([
            f"-DANDROID_ABI=arm64-v8a",
            f"-DANDROID_PLATFORM=android-{api_level}",
            f"-DANDROID_NDK={android_ndk}",
            f"-DCMAKE_TOOLCHAIN_FILE={android_ndk}/build/cmake/android.toolchain.cmake",
            f"-DCMAKE_BUILD_TYPE={args.build_type}"
        ])
    elif "linux_arm" in args.backend:
        toolchain_file = script_dir / "aarch64_toolchain.cmake"
        cmake_configure.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}")
        
        if args.vulkan_lib:
            cmake_configure.append(f"-DCMAKE_PREFIX_PATH={args.vulkan_lib}")
        
        # Add build type for Linux
        cmake_configure.append(f"-DCMAKE_BUILD_TYPE={args.build_type}")
    elif "linux" in args.backend:
        # Add build type for Linux
        cmake_configure.append(f"-DCMAKE_BUILD_TYPE={args.build_type}")
    
    # Add extra CMake arguments
    if args.cmake_args:
        cmake_configure.extend(args.cmake_args)
    
    # Run CMake configure
    run_command(cmake_configure, f"Configuring {args.backend} {args.build_type}")
    
    # Prepare CMake build command
    cmake_build = [
        "cmake",
        "--build", str(build_dir),
        "--verbose",
        "--config", args.build_type
    ]
    
    # Add parallel jobs
    if args.jobs:
        if "android" in args.backend:
            cmake_build.extend(["--parallel", str(args.jobs)])
        else:
            cmake_build.extend(["--parallel", str(args.jobs)])
            if platform.system().lower() == "windows":
                cmake_build.extend(["--", f"/p:CL_MPcount={args.jobs}"])
            else:
                cmake_build.extend(["--", f"-j{args.jobs}"])
    
    # Run CMake build
    run_command(cmake_build, f"Building {args.backend} {args.build_type}")
    
    printf(f"\nBuild completed successfully!")
    printf(f"Build directory: {build_dir}")
    printf(f"Output binaries: {script_dir / 'bin'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-platform build script for ARM NG SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default values (auto-detects backend based on host OS)
  python build.py
  # On Linux x64, equivalent to:
  #   python build.py --build-type=Release --backend=vk_linux_x64
  # On Windows x64, equivalent to:
  #   python build.py --build-type=Release --backend=vk_windows_x64
  
Some common build examples:
  Windows:   python build.py -t Debug -b vk_windows_x64
  Linux x64: python build.py -t Release -b vk_linux_x64
  Linux arm: python build.py -t Release -b vk_linux_arm --vulkan-lib=[path/to/vulkan/lib/for/aarch64]
  Android:   python build.py -t Debug -b vk_android_arm

More custom examples:
  # Basic usage with named arguments
  python build.py --build-type=Debug --backend=vk_android_arm
  python build.py -t Release -b vk_linux_x64
  
  # With additional CMake options
  python build.py -t Release -b vk_linux_x64 --cmake-args="-DFFX_FSR3_AS_LIBRARY=OFF"
  python build.py -t Debug -b vk_android_arm --cmake-args="-DFFX_BUILD_AS_DLL=ON"
  python build.py -t Release -b vk_linux_x64 --cmake-args="-DFFX_API_AS_DLL=OFF -DFFX_FSR3_AS_LIBRARY=OFF"
  
  # With parallel jobs
  python build.py --build-type=RelWithDebInfo --backend=vk_windows_x64 --jobs=8
  python build.py -t Release -b vk_linux_x64 -j 4
  
  # Incremental build (default behavior)
  python build.py -t Release -b vk_linux_x64
  python build.py -t Debug -b vk_android_arm
  
  # Clean build (removes build directory and CMake cache)
  python build.py --clean -t Release -b vk_linux_x64
  python build.py -t Debug -b vk_android_arm --clean
  
  # Combined example with all options
  python build.py --clean -t Release -b vk_linux_x64 -j 8 --cmake-args="-DFFX_API_AS_DLL=OFF"

Available backends:
  vk_windows_x64    - Windows x64 (Vulkan)
  vk_linux_x64      - Linux x64 (Vulkan)
  vk_linux_arm      - Linux ARM64/aarch64 (Vulkan, requires toolchain)
  vk_android_arm    - Android ARM64 (Vulkan, requires ANDROID_NDK)

Build output:
  Build artifacts are located in: bin/
  Build directories follow pattern: build_<platform>/<BuildType>/
        """
    )
    
    # Named arguments
    parser.add_argument(
        "--build-type", "-t",
        dest="build_type",
        choices=["Debug", "Release", "RelWithDebInfo"],
        default="Release",
        help="Build type (default: Release)"
    )
    
    parser.add_argument(
        "--backend", "-b",
        dest="backend",
        default=get_default_backend(),
        help=f"Backend (default: auto-detect, currently: {get_default_backend()})"
    )
    
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        help="Number of parallel jobs (default: system default)"
    )
    
    parser.add_argument(
        "--android-api",
        type=str,
        help="Android API level (default: 33)"
    )
    
    parser.add_argument(
        "--vulkan-lib",
        type=str,
        help="Path to Vulkan library for aarch64 Linux builds"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean build directory and CMake cache before building (default: False)"
    )
    
    parser.add_argument(
        "--cmake-args",
        type=str,
        help='Additional CMake arguments as a space-separated string (e.g., --cmake-args="-DFFX_API_AS_DLL=OFF -DFFX_FSR3_AS_LIBRARY=OFF")'
    )
    
    args = parser.parse_args()
    
    # Parse cmake_args string into a list
    if args.cmake_args:
        args.cmake_args = args.cmake_args.split()
    else:
        args.cmake_args = []
    
    printf(f"ARM NG SDK Build Script")
    printf(f"   Build Type: {args.build_type}")
    printf(f"   Backend:    {args.backend}")
    printf(f"   Host OS:    {platform.system()} {platform.machine()}")
    printf(f"   Clean:      {'Yes (full rebuild)' if args.clean else 'No (incremental build)'}")
    printf(f"   Jobs:       {args.jobs if args.jobs else 'Auto (system default)'}")
    
    build(args)


if __name__ == "__main__":
    main()
