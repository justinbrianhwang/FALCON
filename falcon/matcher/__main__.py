"""Command-line interface for paired-run validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .matcher import validate_pair


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a FALCON run pair")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args()

    report = validate_pair(args.reference, args.failure)
    print(f"Status: {report.status}")
    print("Checks:")
    width = max(map(len, report.checks))
    for name, passed in report.checks.items():
        print(f"  {name:<{width}}  {'PASS' if passed else 'FAIL'}")
    if report.first_divergence_round is None:
        print("First divergence: none")
    else:
        print(
            "First divergence: "
            f"round {report.first_divergence_round}, "
            f"stage {report.first_divergence_stage}"
        )
    for warning in report.warnings:
        print(f"Warning: {warning}")

    if args.json_path is not None:
        args.json_path.write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    return 1 if report.status == "INVALID_PAIR" else 0


if __name__ == "__main__":
    raise SystemExit(main())
