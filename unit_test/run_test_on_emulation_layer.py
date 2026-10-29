#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

"""Run SDK unit tests with the compatible Vulkan ML emulation layers."""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


def prepend_path(path, existing_value):
    """Prepend a path to an environment-variable path list."""
    return str(path) if not existing_value else f"{path}{os.pathsep}{existing_value}"


def find_layer_directories(path, library_subdirectory, layer_libraries):
    """Return the manifest and library directories below an extracted package."""
    if all((path / library).is_file() for library in layer_libraries):
        library_directory = path
        package_directory = path.parent if path.name in ("bin", "lib") else path
    else:
        package_directory = path
        library_directory = path / library_subdirectory

    missing_libraries = [
        library for library in layer_libraries if not (library_directory / library).is_file()
    ]
    if missing_libraries:
        raise RuntimeError(f"Vulkan ML layer directory is incomplete: {path}")

    manifest_names = ("VkLayer_Graph.json", "VkLayer_Tensor.json")
    manifest_candidates = (
        library_directory,
        package_directory / "share" / "vulkan" / "explicit_layer.d",
    )
    for manifest_directory in manifest_candidates:
        if all((manifest_directory / manifest).is_file() for manifest in manifest_names):
            return manifest_directory, library_directory

    raise RuntimeError(f"Vulkan ML layer manifests not found below: {path}")


def validate_test_arguments(arguments):
    """Reject test-runner modes unsupported by the emulation layers."""
    if "--no-isolation" in arguments:
        raise RuntimeError(
            "Vulkan ML emulation-layer execution requires subprocess isolation; "
            "remove --no-isolation."
        )


def main():
    system_name = platform.system()
    if system_name == "Windows":
        if platform.machine().lower() not in ("amd64", "x86_64"):
            print(f"Unsupported Windows architecture: {platform.machine()}", file=sys.stderr)
            return 1
        executable_name = "sdk_unit_test.exe"
        layer_libraries = ("VkLayer_Graph.dll", "VkLayer_Tensor.dll")
        artifact_platform = "Windows_AMD64"
        library_subdirectory = "bin"
    elif system_name == "Linux":
        executable_name = "sdk_unit_test"
        layer_libraries = ("libVkLayer_Graph.so", "libVkLayer_Tensor.so")
        machine = platform.machine().lower()
        if machine in ("arm64", "aarch64"):
            artifact_platform = "Linux_aarch64"
        elif machine in ("amd64", "x86_64"):
            artifact_platform = "Linux_x86_64"
        else:
            print(f"Unsupported Linux architecture: {platform.machine()}", file=sys.stderr)
            return 1
        library_subdirectory = "lib"
    else:
        print(f"Unsupported operating system: {system_name}", file=sys.stderr)
        return 1

    script_directory = Path(__file__).resolve().parent
    repository_root = script_directory.parent
    default_layer_directory = (
        repository_root
        / "deps"
        / "emulation_layer"
        / artifact_platform
    )

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--vk-emul-layer-path",
        type=Path,
        default=default_layer_directory,
    )
    helper_arguments, arguments = parser.parse_known_args(sys.argv[1:])
    if arguments and arguments[0] in ("--native", "native"):
        arguments = arguments[1:]

    try:
        validate_test_arguments(arguments)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    test_binary = repository_root / "bin" / executable_name
    if not test_binary.is_file() or (system_name != "Windows" and not os.access(test_binary, os.X_OK)):
        print(f"Unit-test binary not found or not executable: {test_binary}", file=sys.stderr)
        return 1

    if "--help" in arguments or "-h" in arguments:
        return subprocess.run([str(test_binary), *arguments], check=False).returncode

    environment = os.environ.copy()
    layer_directory = helper_arguments.vk_emul_layer_path
    try:
        manifest_directory, library_directory = find_layer_directories(
            layer_directory,
            library_subdirectory,
            layer_libraries,
        )
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    environment["VK_ADD_LAYER_PATH"] = prepend_path(
        manifest_directory,
        environment.get("VK_ADD_LAYER_PATH"),
    )
    environment["VK_LAYER_PATH"] = prepend_path(
        manifest_directory,
        environment.get("VK_LAYER_PATH"),
    )
    environment["VK_INSTANCE_LAYERS"] = os.pathsep.join(
        ("VK_LAYER_ML_Graph_Emulation", "VK_LAYER_ML_Tensor_Emulation")
    )
    if system_name == "Windows":
        environment["PATH"] = prepend_path(library_directory, environment.get("PATH"))
    else:
        environment["LD_LIBRARY_PATH"] = prepend_path(
            library_directory,
            environment.get("LD_LIBRARY_PATH"),
        )

    print("\nRunning sdk_unit_test\n")
    try:
        return subprocess.run([str(test_binary), *arguments], env=environment, check=False).returncode
    except OSError as error:
        print(f"Failed to run {test_binary}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())