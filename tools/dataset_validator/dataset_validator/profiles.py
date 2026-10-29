# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROFILE_ROOT = Path(__file__).resolve().parents[1] / "profiles"


class ProfileError(ValueError):
    pass


def load_profile(name: str) -> dict[str, Any]:
    if not name or Path(name).name != name:
        raise ProfileError(f"Invalid profile name: {name!r}")

    profile_path = PROFILE_ROOT / f"{name}.json"
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        available = ", ".join(sorted(path.stem for path in PROFILE_ROOT.glob("*.json")))
        raise ProfileError(f"Unknown profile {name!r}. Available profiles: {available}") from error
    except json.JSONDecodeError as error:
        raise ProfileError(f"Profile is not valid JSON: {profile_path}: {error}") from error

    if profile.get("name") != name:
        raise ProfileError(f"Profile name mismatch in {profile_path}")
    return profile
