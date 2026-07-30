#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

"""
Cross-platform build script for Neural Graphics SDK for Game Engines.
Supports Windows, Linux x64, Linux ARM64, and Android builds.
"""

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


DOCKER_SUPPORTED_BACKENDS = {"vk_linux_x64", "vk_linux_arm", "vk_android_arm"}


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


def run_command(cmd, description, cwd=None):
    """Run a command and print it before execution."""
    printf(f"\n{'='*60}")
    printf(f"  {description}")
    printf(f"{'='*60}")
    printf(f"Command:\n")
    printf(f"{' '.join(str(c) for c in cmd)}\n")
    if cwd:
        printf(f"Working directory: {cwd}")
    printf(f"{'='*60}\n")

    result = subprocess.run(cmd, check=False, cwd=cwd)
    if result.returncode != 0:
        printf(f"\nError: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def build_android_samples(args, script_dir):
    """Build Android samples using generate.py + Gradle.

    Called automatically from build() when --build-samples is used with an
    Android backend.  The two steps (Gradle project generation + Gradle build)
    are hidden from the user — they just pass --build-samples and both happen.
    """
    samples_dir = script_dir / "samples"
    gradle_dir = samples_dir / "build" / "android_gradle"

    # Wipe stale Gradle project to avoid cached cmake/ninja paths from a previous build.
    if gradle_dir.exists():
        shutil.rmtree(gradle_dir)

    # generate.py uses relative paths so it must run from samples/
    run_command(
        [sys.executable, str(samples_dir / "scripts" / "generate.py"), "android"],
        "Generating Android Gradle project",
        cwd=str(samples_dir),
    )

    build_type_lower = args.build_type.lower()
    if build_type_lower == "debug":
        gradle_task = "assembleDebug"
    elif build_type_lower == "relwithdebinfo":
        gradle_task = "assembleRelWithDebInfo"
    else:
        gradle_task = "assembleRelease"

    if platform.system() == "Windows":
        # cmd /c is needed because .bat files are not directly executable by CreateProcess.
        gradlew_cmd = ["cmd", "/c", str(gradle_dir / "gradlew.bat"), "clean", gradle_task]
    else:
        gradlew_path = gradle_dir / "gradlew"
        # Ensure execute permission (generate.py may not set it).
        gradlew_path.chmod(gradlew_path.stat().st_mode | 0o111)
        gradlew_cmd = [str(gradlew_path), "clean", gradle_task]

    run_command(
        gradlew_cmd,
        f"Building Android samples ({gradle_task})",
        cwd=str(gradle_dir),
    )

    printf(f"\nAndroid samples build completed!")
    printf(f"APK: {gradle_dir}/app/build/outputs/apk/")
    printf("")
    printf("To install and run on a connected device:")
    apk_type = "debug" if build_type_lower == "debug" else "release"
    printf(f"  adb install {gradle_dir}/app/build/outputs/apk/{apk_type}/vulkan_samples-{apk_type}.apk")
    printf("")
    printf("If assets/shaders were not synced automatically (no device connected during build):")
    printf(f"  adb push {samples_dir}/assets/scenes/sponza "
           "/sdcard/Android/data/com.khronos.vulkan_samples/files/assets/scenes/sponza")
    printf(f"  adb push {samples_dir}/assets/fonts "
           "/sdcard/Android/data/com.khronos.vulkan_samples/files/assets/fonts")
    printf(f"  adb push {samples_dir}/shaders "
           "/sdcard/Android/data/com.khronos.vulkan_samples/files/shaders")
    printf("  adb shell \"chmod -R 777 /sdcard/Android/data/com.khronos.vulkan_samples/files/assets\"")
    printf("  adb shell \"chmod -R 777 /sdcard/Android/data/com.khronos.vulkan_samples/files/shaders\"")
    printf("  adb shell am start-activity -n com.khronos.vulkan_samples/"
           "com.khronos.vulkan_samples.SampleLauncherActivity -e sample nss")


def image_exists(image_name):
    """Check if a Docker image exists locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", image_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def ensure_docker_image(args, script_dir):
    """Ensure Docker image exists or rebuild when requested."""
    dockerfile_path = Path(args.dockerfile)
    if not dockerfile_path.is_absolute():
        dockerfile_path = script_dir / dockerfile_path

    if not dockerfile_path.exists():
        printf(f"Error: Dockerfile not found: {dockerfile_path}")
        sys.exit(1)

    should_build = args.docker_rebuild or not image_exists(args.docker_image)
    if should_build:
        reason = "Rebuilding" if args.docker_rebuild else "Building"
        printf(f"{reason} Docker image: {args.docker_image}")
        docker_build_cmd = [
            "docker", "build",
            "-f", str(dockerfile_path),
            "-t", args.docker_image,
            str(script_dir),
        ]
        run_command(docker_build_cmd, f"Docker image build ({args.docker_image})")


def get_docker_inner_command(args):
    """Build command arguments for invoking this script inside Docker."""
    inner_cmd = [
        "python3", "build.py",
        "--build-type", args.build_type,
        "--backend", args.backend,
    ]

    if args.jobs:
        inner_cmd.extend(["--jobs", str(args.jobs)])
    if args.android_api:
        inner_cmd.extend(["--android-api", str(args.android_api)])
    if args.vulkan_lib:
        inner_cmd.extend(["--vulkan-lib", args.vulkan_lib])
    if args.clean:
        inner_cmd.append("--clean")
    if args.build_samples:
        inner_cmd.append("--build-samples")
    if args.cmake_args:
        inner_cmd.append(f"--cmake-args={' '.join(args.cmake_args)}")

    return inner_cmd


def get_container_workspace_path(host_workspace_path):
    """Compute container workspace path from current host workspace path."""
    if platform.system().lower() == "windows":
        workspace_name = Path(host_workspace_path).name or "workspace"
        return f"/{workspace_name}"

    return host_workspace_path


def ensure_linux_workspace_permissions(args, script_dir, host_mount, container_mount):
    """Repair ownership of existing output paths when needed on Linux hosts."""
    if platform.system().lower() != "linux":
        return

    uid = os.getuid()
    gid = os.getgid()
    target_build_dir = script_dir / get_build_dir(args.backend, args.build_type)
    candidate_paths = [target_build_dir, script_dir / "bin"]

    paths_to_fix = []
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            owner_uid = path.stat().st_uid
            if owner_uid != uid:
                paths_to_fix.append(path)
        except OSError:
            continue

    if not paths_to_fix:
        return

    relative_paths = [str(path.relative_to(script_dir)) for path in paths_to_fix]
    quoted_targets = " ".join(shlex.quote(p) for p in relative_paths)
    repair_cmd = (
        f"cd {shlex.quote(container_mount)} && "
        f"chown -R {uid}:{gid} {quoted_targets}"
    )

    docker_fix_cmd = [
        "docker", "run", "--rm",
        "-v", f"{host_mount}:{container_mount}",
        "-w", container_mount,
        args.docker_image,
        "sh", "-c", repair_cmd,
    ]
    run_command(docker_fix_cmd, "Repairing ownership of existing build artifacts")


def build_with_docker(args):
    """Execute build inside Docker container."""
    if args.backend not in DOCKER_SUPPORTED_BACKENDS:
        printf(
            "Error: Docker mode supports only vk_linux_x64, vk_linux_arm, vk_android_arm. "
            f"Got: {args.backend}"
        )
        sys.exit(1)

    if shutil.which("docker") is None:
        printf("Error: Docker CLI not found. Please install Docker and ensure it is in PATH.")
        sys.exit(1)

    script_dir = Path(__file__).parent.resolve()
    ensure_docker_image(args, script_dir)

    host_mount = str(script_dir)
    if platform.system().lower() == "windows":
        host_mount = host_mount.replace("\\", "/")

    container_mount = get_container_workspace_path(host_mount)

    ensure_linux_workspace_permissions(args, script_dir, host_mount, container_mount)

    docker_run_cmd = [
        "docker", "run", "--rm",
        "-v", f"{host_mount}:{container_mount}",
        "-w", container_mount,
    ]

    if platform.system().lower() == "linux":
        docker_run_cmd.extend([
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", f"HOME={container_mount}",
        ])

    docker_run_cmd.append(args.docker_image)
    docker_run_cmd.extend(get_docker_inner_command(args))

    run_command(docker_run_cmd, f"Docker build for {args.backend} {args.build_type}")


def build_native(args):
    """Execute the build process."""
    # Validate: --build-samples requires static SDK libraries
    if args.build_samples:
        if any("-DFFX_BUILD_AS_DLL=ON" in a for a in args.cmake_args):
            printf("\nError: --build-samples is incompatible with -DFFX_BUILD_AS_DLL=ON.")
            printf("  Static-library mode is set automatically when --build-samples is used.")
            sys.exit(1)

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

    if args.build_samples:
        cmake_configure.append("-DFFX_BUILD_AS_DLL=OFF")
    else:
        cmake_configure.append("-DFFX_BUILD_AS_DLL=ON")

    # Include samples in the CMake build tree (non-Android only).
    # Android samples use a separate Gradle-based build invoked after the SDK
    if args.build_samples and "android" not in args.backend:
        cmake_configure.append("-DFFX_BUILD_SAMPLES=ON")

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
    
    # On Linux with samples, default to Direct-to-Display (no X11/XCB/Wayland libs needed).
    # Users can override via --cmake-args="-DVKB_WSI_SELECTION=XCB" (or XLIB/WAYLAND).
    if args.build_samples and "linux" in args.backend:
        if not any("VKB_WSI_SELECTION" in a for a in args.cmake_args):
            cmake_configure.append("-DVKB_WSI_SELECTION=D2D")

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

    printf(f"\nSDK build completed successfully!")
    printf(f"Build directory: {build_dir}")
    printf(f"Output binaries: {script_dir / 'bin'}")

    # Android samples: two-step separate build after SDK
    if args.build_samples and "android" in args.backend:
        build_android_samples(args, script_dir)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-platform build script for Neural Graphics SDK for Game Engines.",
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

  With samples:
    Fetch sample dependencies first:
    Windows:   powershell -ExecutionPolicy Bypass -File samples/scripts/fetch_sample_dependencies.ps1
    Linux:     bash samples/scripts/fetch_sample_dependencies.sh
  Windows:   python build.py -t Debug -b vk_windows_x64 --build-samples
  Linux x64: python build.py -t Release -b vk_linux_x64 --build-samples
  Android:   python build.py -t Release -b vk_android_arm --build-samples

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

  # Build SDK + samples together (Windows)
  python build.py -t Debug -b vk_windows_x64 --build-samples

  # Build SDK + samples together (Linux)
  # Select WSI backend via VKB_WSI_SELECTION:
  #   XCB (X11) | XLIB (X11) | WAYLAND | D2D (Direct-to-Display, VK_KHR_DISPLAY, default)
  python build.py -t Release -b vk_linux_x64 --build-samples
  python build.py -t Release -b vk_linux_x64 --build-samples --cmake-args="-DVKB_WSI_SELECTION=XCB"

  # Build SDK + Android samples
  # JDK 21 is required
  $env:NDK_ROOT="path/to/ndk"
  python build.py --clean -t Release -b vk_android_arm --build-samples

Docker examples:
    # Build Linux x64 on Linux or Windows host via Docker
    python build.py --docker -t Release -b vk_linux_x64

    # Build Linux ARM64 using Docker image (auto-build image when missing)
    python build.py --docker -t Release -b vk_linux_arm

    # Build Android ARM64 in Docker with specific API level
    python build.py --docker -t Debug -b vk_android_arm --android-api 33

    # Force Docker image rebuild from Dockerfile before build
    python build.py --docker --docker-rebuild -b vk_linux_x64

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
        "--build-samples",
        action="store_true",
        default=False,
        dest="build_samples",
        help="Also build the Sample code together with the SDK. Fetch sample dependencies first with samples/scripts/fetch_sample_dependencies.ps1 or .sh."
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

    parser.add_argument(
        "--docker",
        action="store_true",
        default=False,
        help="Run build in Docker (supported backends: vk_linux_x64, vk_linux_arm, vk_android_arm)"
    )

    parser.add_argument(
        "--docker-image",
        type=str,
        default="ngsdk-build:latest",
        help="Docker image name/tag to use for --docker builds (default: ngsdk-build:latest)"
    )

    parser.add_argument(
        "--dockerfile",
        type=str,
        default="Dockerfile",
        help="Dockerfile path used to build image when needed (default: Dockerfile)"
    )

    parser.add_argument(
        "--docker-rebuild",
        action="store_true",
        default=False,
        help="Rebuild Docker image before --docker build"
    )
    
    args = parser.parse_args()
    
    # Parse cmake_args string into a list
    if args.cmake_args:
        args.cmake_args = args.cmake_args.split()
    else:
        args.cmake_args = []
    
    printf(f"Neural Graphics SDK for Game Engines Build Script")
    printf(f"   Build Type: {args.build_type}")
    printf(f"   Backend:    {args.backend}")
    printf(f"   Host OS:    {platform.system()} {platform.machine()}")
    printf(f"   Clean:      {'Yes (full rebuild)' if args.clean else 'No (incremental build)'}")
    printf(f"   Jobs:       {args.jobs if args.jobs else 'Auto (system default)'}")
    printf(f"   Docker:     {'Yes' if args.docker else 'No'}")
    printf(f"   Build Samples:  {'Yes' if args.build_samples else 'No'}")

    if args.docker:
        build_with_docker(args)
    else:
        build_native(args)


if __name__ == "__main__":
    main()
