"""Run the complete co-author experiment suite and collect its outputs."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.e0_replay_validation import run_experiment as run_e0  # noqa: E402
from experiments.e1_equivalence import run_experiment as run_e1  # noqa: E402

CONFIG_DIR = REPO / "configs" / "experiments" / "coauthor"
COLLECTOR = REPO / "scripts" / "collect_output.py"


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_matrix(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("experiment"), dict):
        raise ValueError(f"{path} must contain an experiment mapping")
    return raw


def _run_e1_matrix(
    name: str, config_path: Path, output_root: Path
) -> dict[str, Any]:
    matrix = _load_matrix(config_path)
    base = matrix["experiment"]
    base_case_id = str(base.get("case_id", name))
    specs: list[dict[str, Any]] = []

    if name == "e1_seeds":
        seeds = matrix.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("e1_seeds config needs a non-empty seeds list")
        for seed in seeds:
            spec = copy.deepcopy(base)
            spec.update({"case_id": f"{base_case_id}_s{int(seed)}", "seed": int(seed)})
            specs.append(spec)
    else:
        levels = matrix.get("heterogeneity_levels")
        if not isinstance(levels, list) or not levels:
            raise ValueError(
                "e1_heterogeneity config needs a non-empty heterogeneity_levels list"
            )
        seed = int(matrix["seed"])
        for level in levels:
            value = float(level)
            spec = copy.deepcopy(base)
            spec.update(
                {
                    "case_id": f"{base_case_id}_h{value:g}".replace(".", "p"),
                    "seed": seed,
                }
            )
            spec["reference"]["dataset"]["heterogeneity"] = value
            specs.append(spec)

    runs = []
    with tempfile.TemporaryDirectory(prefix=f"falcon_{name}_") as temp_dir:
        for index, spec in enumerate(specs):
            spec_path = Path(temp_dir) / f"{index}.yaml"
            spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
            try:
                report = run_e1(spec_path, output_root)
                runs.append(
                    {
                        "case_id": spec["case_id"],
                        "status": report["status"],
                    }
                )
            except Exception as exc:  # continue collecting independent evidence
                runs.append(
                    {
                        "case_id": spec["case_id"],
                        "status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    return {
        "name": name,
        "status": "PASS" if all(run["status"] == "PASS" for run in runs) else "FAILED",
        "runs": runs,
    }


def _collect(collector: Path) -> Path:
    completed = subprocess.run(
        [sys.executable, str(collector)],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode:
        raise RuntimeError(f"collector exited with status {completed.returncode}")
    match = re.search(r"->\s*(.+\.zip)\s*$", completed.stdout)
    if not match:
        raise RuntimeError("collector did not report an output zip")
    output_zip = Path(match.group(1))
    if not output_zip.is_file():
        raise RuntimeError(f"collector output does not exist: {output_zip}")
    return output_zip


def run_suite(
    *,
    e0_config: Path,
    e1_seeds_config: Path,
    e1_heterogeneity_config: Path,
    output_root: Path,
    collector: Path = COLLECTOR,
) -> tuple[dict[str, Any], Path | None]:
    output_root.mkdir(parents=True, exist_ok=True)
    experiments = []

    try:
        report = run_e0(e0_config, output_root / "e0_crossmachine")
        experiments.append(
            {"name": "e0_crossmachine", "status": report["status"]}
        )
    except Exception as exc:
        experiments.append(
            {
                "name": "e0_crossmachine",
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    for name, config in (
        ("e1_seeds", e1_seeds_config),
        ("e1_heterogeneity", e1_heterogeneity_config),
    ):
        try:
            experiments.append(_run_e1_matrix(name, config, output_root / name))
        except Exception as exc:
            experiments.append(
                {
                    "name": name,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "runs": [],
                }
            )

    summary = {
        "suite": "coauthor",
        "status": (
            "PASS"
            if all(experiment["status"] == "PASS" for experiment in experiments)
            else "FAILED"
        ),
        "experiments": experiments,
        "collector": {"status": "PENDING"},
    }
    summary_path = output_root / "summary.json"
    _write_summary(summary_path, summary)

    output_zip = None
    try:
        output_zip = _collect(collector)
        summary["collector"] = {"status": "PASS", "output_zip": str(output_zip)}
    except Exception as exc:
        summary["status"] = "FAILED"
        summary["collector"] = {
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }
    _write_summary(summary_path, summary)
    return summary, output_zip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e0-config", type=Path, default=CONFIG_DIR / "e0_crossmachine.yaml"
    )
    parser.add_argument(
        "--e1-seeds-config", type=Path, default=CONFIG_DIR / "e1_seeds.yaml"
    )
    parser.add_argument(
        "--e1-heterogeneity-config",
        type=Path,
        default=CONFIG_DIR / "e1_heterogeneity.yaml",
    )
    parser.add_argument("--output", type=Path, default=REPO / "results" / "coauthor")
    parser.add_argument("--collector", type=Path, default=COLLECTOR)
    args = parser.parse_args(argv)

    summary, output_zip = run_suite(
        e0_config=args.e0_config,
        e1_seeds_config=args.e1_seeds_config,
        e1_heterogeneity_config=args.e1_heterogeneity_config,
        output_root=args.output,
        collector=args.collector,
    )
    if output_zip is not None:
        print(f"SEND THIS FILE: {output_zip}")
    else:
        print("NO OUTPUT ZIP CREATED; see the suite summary for the collector error.")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
