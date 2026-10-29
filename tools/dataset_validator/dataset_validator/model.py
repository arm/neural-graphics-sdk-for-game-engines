# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any


VALIDATOR_VERSION = "0.1.0"
SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class Finding:
    severity: str
    check_id: str
    message: str
    remediation: str
    sequence: str | None = None
    frame: int | None = None
    path: str | None = None
    observed: Any = None
    expected: Any = None

    def sort_key(self) -> tuple[Any, ...]:
        return (
            SEVERITY_ORDER[self.severity],
            self.check_id,
            self.sequence or "",
            self.frame if self.frame is not None else -1,
            self.path or "",
            self.message,
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe({
            "severity": self.severity,
            "check_id": self.check_id,
            "message": self.message,
            "sequence": self.sequence,
            "frame": self.frame,
            "path": self.path,
            "observed": self.observed,
            "expected": self.expected,
            "remediation": self.remediation,
        })


@dataclass
class ValidationReport:
    dataset: str
    profile: str
    use_case: str
    scale: str
    warning_policy: str
    sequences: list[dict[str, Any]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    complete_scan: bool = True
    files_scanned: int = 0

    @property
    def status(self) -> str:
        has_errors = any(finding.severity == "error" for finding in self.findings)
        warnings_fail = self.warning_policy == "fail" and any(
            finding.severity == "warning" for finding in self.findings
        )
        return "FAIL" if has_errors or warnings_fail else "PASS"

    def to_dict(self) -> dict[str, Any]:
        ordered_findings = sorted(self.findings, key=Finding.sort_key)
        finding_counts = {
            severity: sum(finding.severity == severity for finding in ordered_findings)
            for severity in ("error", "warning", "info")
        }
        return _json_safe({
            "schema_version": 1,
            "validator_version": VALIDATOR_VERSION,
            "dataset": self.dataset,
            "profile": self.profile,
            "use_case": self.use_case,
            "scale": self.scale,
            "warning_policy": self.warning_policy,
            "complete_scan": self.complete_scan,
            "status": self.status,
            "summary": {
                "sequences": len(self.sequences),
                "files_scanned": self.files_scanned,
                "findings": {
                    "errors": finding_counts["error"],
                    "warnings": finding_counts["warning"],
                    "info": finding_counts["info"],
                },
            },
            "sequences": self.sequences,
            "findings": [finding.to_dict() for finding in ordered_findings],
        })

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
