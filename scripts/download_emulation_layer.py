#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

"""Download the compatible Vulkan ML emulation layer release."""

import argparse
import os
import platform
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
COMPATIBILITY_MATRIX = REPOSITORY_ROOT / "docs" / "compatibility_matrix.md"
RELEASE_URL = "https://github.com/arm/ai-ml-emulation-layer-for-vulkan/releases/download"
PLATFORMS = {
    "Darwin_arm64": ("Darwin", {"arm64", "aarch64"}, ".tar.gz"),
    "Linux_aarch64": ("Linux", {"arm64", "aarch64"}, ".tar.gz"),
    "Linux_x86_64": ("Linux", {"amd64", "x86_64"}, ".tar.gz"),
    "Windows_AMD64": ("Windows", {"amd64", "x86_64"}, ".zip"),
}


def read_version():
    text = COMPATIBILITY_MATRIX.read_text(encoding="utf-8")
    match = re.search(
        r"^\| Arm ML Emulation Layer for Vulkan \| `(v\d+\.\d+\.\d+)` \|",
        text,
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError(f"Emulation-layer version not found in {COMPATIBILITY_MATRIX}")
    return match.group(1)


def detect_platform():
    system_name = platform.system()
    machine = platform.machine().lower()
    for artifact_platform, (system, machines, _suffix) in PLATFORMS.items():
        if system_name == system and machine in machines:
            return artifact_platform
    raise RuntimeError(f"Unsupported platform: {system_name} {platform.machine()}")


def validate_archive_path(root, member_name):
    try:
        (root / member_name).resolve().relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"Archive contains an unsafe path: {member_name}") from error


def extract_archive(archive, destination):
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                validate_archive_path(destination, member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise RuntimeError(f"Archive contains a symbolic link: {member.filename}")
            package.extractall(destination)
        return

    with tarfile.open(archive, "r:gz") as package:
        for member in package.getmembers():
            validate_archive_path(destination, member.name)
            if member.issym() or member.islnk():
                raise RuntimeError(f"Archive contains a link: {member.name}")
        package.extractall(destination)


def download(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "arm-ng-sdk"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=PLATFORMS)
    args = parser.parse_args()

    try:
        version = read_version()
        artifact_platform = args.platform or detect_platform()
        suffix = PLATFORMS[artifact_platform][2]
        package_name = f"ai_ml_emulation_layer_for_vulkan-{version[1:]}-{artifact_platform}"
        archive_name = package_name + suffix
        url = f"{RELEASE_URL}/release/{version}/{archive_name}"

        output_root = REPOSITORY_ROOT / "deps" / "emulation_layer"
        target = output_root / artifact_platform
        output_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=output_root) as temporary_directory:
            temporary_path = Path(temporary_directory)
            archive = temporary_path / archive_name
            extracted = temporary_path / "extracted"
            extracted.mkdir()

            print(f"Downloading {url}")
            download(url, archive)
            extract_archive(archive, extracted)

            if target.exists():
                shutil.rmtree(target)
            os.replace(extracted, target)

        print(f"Emulation layer {version} extracted to {target}")
        return 0
    except (OSError, RuntimeError, tarfile.TarError, urllib.error.URLError, zipfile.BadZipFile) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
