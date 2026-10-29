#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an NSS or NFRU dataset without modifying it.",
    )
    parser.add_argument("dataset_path", type=Path, help="Dataset root containing numeric sequence JSON files.")
    parser.add_argument("--profile", required=True, help="Validation profile name.")
    parser.add_argument(
        "--use-case",
        choices=("debug", "replay"),
        default="replay",
        help="Required-signal policy (default: replay).",
    )
    parser.add_argument(
        "--scale",
        default="x2",
        help="Dataset scale directory, for example x2 or x1_7 (default: x2).",
    )
    parser.add_argument(
        "--warning-policy",
        choices=("allow", "fail"),
        default="allow",
        help="Whether warnings are allowed or make validation fail (default: allow).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path for the complete JSON report.",
    )
    return parser


def _print_report(payload: dict) -> None:
    summary = payload["summary"]
    findings = summary["findings"]
    print(f"{payload['status']}: {payload['dataset']}")
    print(
        f"Profile: {payload['profile']}  "
        f"Use case: {payload['use_case']}  "
        f"Scale: {payload['scale']}"
    )
    print(
        f"Sequences: {summary['sequences']}  "
        f"EXR files: {summary['files_scanned']}  "
        f"Errors: {findings['errors']}  "
        f"Warnings: {findings['warnings']}"
    )
    for finding in payload["findings"]:
        location = finding["path"] or "."
        if finding["frame"] is not None:
            location = f"{location}:{finding['frame']}"
        print(
            f"[{finding['severity'].upper()}] "
            f"{finding['check_id']} ({location}): {finding['message']}"
        )
        print(f"  Expected: {finding['expected']}")
        print(f"  Observed: {finding['observed']}")
        print(f"  Remediation: {finding['remediation']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        from dataset_validator import ValidationError, validate_dataset
    except (ImportError, OSError) as error:
        print(f"ERROR: validator dependencies could not be loaded: {error}", file=sys.stderr)
        return 2

    try:
        report = validate_dataset(
            args.dataset_path,
            args.profile,
            use_case=args.use_case,
            scale=args.scale,
            warning_policy=args.warning_policy,
        )
    except (ValidationError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if args.json_out is not None:
        json_output_path = args.json_out.expanduser().resolve()
        dataset_root = Path(report.dataset)
        if json_output_path == dataset_root or dataset_root in json_output_path.parents:
            print(
                "ERROR: --json-out must be outside the dataset root to preserve read-only validation.",
                file=sys.stderr,
            )
            return 2
        try:
            json_output_path.write_text(report.to_json(), encoding="utf-8")
        except OSError as error:
            print(f"ERROR: could not write JSON report {json_output_path}: {error}", file=sys.stderr)
            return 2

    payload = report.to_dict()
    _print_report(payload)
    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
