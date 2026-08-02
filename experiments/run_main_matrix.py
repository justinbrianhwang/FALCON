"""Run the declared main experiment matrix and build Table 1."""

from __future__ import annotations

import argparse
import copy
import csv
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.e1_equivalence import (  # noqa: E402
    UnmatchableSeverityError,
    run_experiment,
)

CONFIG_DIR = REPO / "configs" / "experiments" / "main"
DEFAULT_MATRIX = CONFIG_DIR / "matrix.yaml"
DEFAULT_OUTPUT = REPO / "results" / "main_matrix"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_COLUMNS = (
    "case",
    "seed",
    "ground_truth",
    "falcon",
    "passive",
    "terminal",
    "gap",
    "notes",
)
_METHODS = (
    ("FALCON", "falcon"),
    ("Passive", "passive"),
    ("Terminal-only", "terminal"),
)


def _load_matrix(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = raw.get("experiments") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} must contain a non-empty experiments list")

    names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each matrix entry must be a mapping")
        name = str(entry.get("name", ""))
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError(f"matrix name must be a safe identifier, got {name!r}")
        if name in names:
            raise ValueError(f"duplicate matrix name {name!r}")
        names.add(name)
        if not isinstance(entry.get("config"), str) or not entry["config"]:
            raise ValueError(f"matrix entry {name!r} needs a config path")
        seeds = entry.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"matrix entry {name!r} needs a non-empty seeds list")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise ValueError(f"matrix entry {name!r} seeds must be integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"matrix entry {name!r} contains duplicate seeds")
    return entries


def _load_experiment(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("experiment"), dict):
        raise ValueError(f"{path} must contain an experiment mapping")
    return raw["experiment"]


def _markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _totals(rows: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    return [
        (
            label,
            sum(row[key] == row["ground_truth"] for row in rows),
            len(rows),
        )
        for label, key in _METHODS
    ]


def _write_tables(
    output_root: Path,
    rows: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "table1.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Main experiment matrix - Table 1",
        "",
        "Top-1 totals use matched cases only.",
        "",
        "| Method | Top-1 |",
        "|---|---:|",
    ]
    lines.extend(f"| {label} | {correct}/{total} |" for label, correct, total in _totals(rows))
    lines.extend(
        [
            "",
            "## Matched cases",
            "",
            "| Case | Seed | Ground truth | FALCON | Passive | Terminal-only | Gap | Notes |",
            "|---|---:|---|---|---|---|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {_markdown(row['case'])} | {row['seed']} | {row['ground_truth']} | "
            f"{row['falcon']} | {row['passive']} | {row['terminal']} | "
            f"{float(row['gap']):.6g} | {_markdown(row['notes'])} |"
        )
    lines.extend(
        [
            "",
            "## Exclusions",
            "",
            "| Case | Seed | Status | Reason |",
            "|---|---:|---|---|",
        ]
    )
    if exclusions:
        lines.extend(
            f"| {_markdown(item['case'])} | {item['seed']} | EXCLUDED | "
            f"{_markdown(item['reason'])} |"
            for item in exclusions
        )
    else:
        lines.append("| None | - | - | - |")
    (output_root / "table1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_matrix(matrix_path: Path, output_root: Path) -> dict[str, Any]:
    entries = _load_matrix(matrix_path)
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    runs_root = output_root / "runs"

    with tempfile.TemporaryDirectory(prefix="falcon_main_matrix_") as temp_dir:
        for entry in entries:
            name = str(entry["name"])
            config_path = Path(entry["config"])
            if not config_path.is_absolute():
                config_path = matrix_path.parent / config_path
            base = _load_experiment(config_path)
            base_case_id = str(base.get("case_id", name))
            for seed in entry["seeds"]:
                spec = copy.deepcopy(base)
                spec.update({"case_id": f"{base_case_id}_s{seed}", "seed": seed})
                spec_path = Path(temp_dir) / f"{name}_s{seed}.yaml"
                spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
                try:
                    report = run_experiment(spec_path, runs_root / name)
                except UnmatchableSeverityError as exc:
                    exclusions.append({"case": name, "seed": seed, "reason": str(exc)})
                    continue

                for case in report["cases"]:
                    predictions = case["predictions"]
                    rows.append(
                        {
                            "case": f"{name}/{case['id']}",
                            "seed": seed,
                            "ground_truth": case["ground_truth"],
                            "falcon": predictions["falcon"],
                            "passive": predictions["passive"],
                            "terminal": predictions["terminal_only"],
                            "gap": case["gap"],
                            "notes": "; ".join(case["falcon"].get("notes", [])),
                        }
                    )

    _write_tables(output_root, rows, exclusions)
    return {"rows": rows, "exclusions": exclusions, "totals": _totals(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = run_matrix(args.matrix.resolve(), args.output.resolve())
    except Exception as exc:
        print(f"Main matrix failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.output / 'table1.csv'} and {args.output / 'table1.md'}")
    print(f"Matched cases: {len(result['rows'])}; exclusions: {len(result['exclusions'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
