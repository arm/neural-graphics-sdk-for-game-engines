#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


TOOL_ROOT = Path(__file__).resolve().parent
DATASET_VALIDATOR_ROOT = TOOL_ROOT.parent / "dataset_validator"
sys.path.insert(0, str(TOOL_ROOT))
sys.path.insert(0, str(DATASET_VALIDATOR_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate offline NSS integration diagnostics from a captured dataset.",
    )
    parser.add_argument("dataset_path", type=Path, help="NSS dataset root.")
    parser.add_argument(
        "--profile",
        choices=("nss-replay-v1",),
        default="nss-replay-v1",
        help="Dataset contract (default: nss-replay-v1).",
    )
    parser.add_argument("--sequence", default="0000", help="Numeric sequence identifier (default: 0000).")
    parser.add_argument("--scale", default="x2", help="Dataset scale directory (default: x2).")
    parser.add_argument(
        "--frames",
        default="auto",
        help=(
            "Frame selection: auto, all, a number, or comma-separated ranges "
            "such as 0,4-8 (maximum 256 selected frames)."
        ),
    )
    parser.add_argument(
        "--diagnostic",
        action="append",
        choices=("all", "jitter", "histogram", "motion"),
        default=[],
        help="Diagnostic to generate; repeat to select several (default: all).",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="New output directory outside the dataset.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        from nss_integration_debug import (
            DiagnosticError,
            run_diagnostics,
        )
    except (ImportError, OSError) as error:
        print(f"ERROR: diagnostic dependencies could not be loaded: {error}", file=sys.stderr)
        return 2

    try:
        report = run_diagnostics(
            args.dataset_path,
            profile=args.profile,
            sequence=args.sequence,
            scale=args.scale,
            frames=args.frames,
            diagnostics=args.diagnostic,
            output_dir=args.output_dir,
        )
    except (DiagnosticError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    report_path = args.output_dir.expanduser().resolve() / "report.json"
    interpretation_path = (
        args.output_dir.expanduser().resolve()
        / report["interpretation"]["human_readable_report"]
    )
    if report["status"] != "PASS":
        print("FAIL: dataset validation found input errors.", file=sys.stderr)
        for finding in report["preflight"]["findings"]:
            if finding["severity"] == "error":
                print(
                    f"  {finding['check_id']}: {finding['message']} "
                    f"({finding['path'] or '.'})",
                    file=sys.stderr,
                )
        print(
            f"Assessment: {report['interpretation']['overall']['assessment']}",
            file=sys.stderr,
        )
        print(f"Report: {report_path}", file=sys.stderr)
        print(f"Interpretation: {interpretation_path}", file=sys.stderr)
        return 1

    summary = report["summary"]
    print(f"PASS: {report['dataset']} sequence {report['sequence']} scale {report['scale']}")
    print(
        f"Frames: {summary['frames']}  "
        f"PNG artifacts: {summary['artifacts']}  "
        f"Warnings: {summary['warnings']}"
    )
    print(f"Assessment: {report['interpretation']['overall']['assessment']}")
    print(f"Report: {report_path}")
    print(f"Interpretation: {interpretation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
