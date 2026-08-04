"""Synthetic Tier-0 benchmark for two simultaneous pipeline failures.

    python experiments/run_compound.py
    python experiments/run_compound.py --smoke
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from falcon.pipeline.runner import run  # noqa: E402
from falcon.recorder import Recorder  # noqa: E402
from falcon.replay.rng import Rng  # noqa: E402
from falcon.reporting.analyze import analyze_pair  # noqa: E402
from falcon.reporting.report import render_markdown  # noqa: E402
from falcon.schema import FailureSpecification, RunConfig, RunMetadata  # noqa: E402

CASES = (
    (
        "selection_compression",
        ("synthetic_selection_failure.yaml", 2),
        ("synthetic_compression_failure.yaml", 2),
    ),
    (
        "selection_aggregation",
        ("synthetic_selection_failure.yaml", 2),
        ("synthetic_aggregation_biased.yaml", None),
    ),
)


def _load_spec(filename: str, severity: int | None) -> FailureSpecification:
    payload = yaml.safe_load(
        (REPO / "configs" / "cases" / filename).read_text(encoding="utf-8")
    )["failure"]
    if severity is not None:
        payload["severity"] = severity
    return FailureSpecification.model_validate(payload)


def _record(root: Path, cfg: RunConfig) -> None:
    run_dir = root / "runs" / cfg.run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    recorder = Recorder(root, cfg.run_id)
    recorder.save_metadata(
        RunMetadata(
            run_id=cfg.run_id,
            seed=cfg.seed,
            rounds=cfg.rounds,
            config=cfg.model_dump(mode="json", exclude={"run_id"}),
            failure=cfg.failure,
            failures=cfg.failures,
        )
    )
    run(cfg, recorder=recorder, rng=Rng(cfg.seed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run four rounds")
    args = parser.parse_args()

    base = yaml.safe_load(
        (REPO / "configs" / "cases" / "synthetic_reference.yaml").read_text(
            encoding="utf-8"
        )
    )
    if args.smoke:
        base["rounds"] = 4

    out = REPO / "results" / "compound"
    out.mkdir(parents=True, exist_ok=True)
    reference = RunConfig.model_validate(
        {**base, "run_id": "ref", "failure": None, "failures": []}
    )
    print(f"[compound] reference ({reference.rounds} rounds)...", flush=True)
    _record(out, reference)

    summary = []
    for name, *sources in CASES:
        specs = [_load_spec(filename, severity) for filename, severity in sources]
        if args.smoke:
            specs = [
                spec.model_copy(
                    update={
                        "active_rounds": (
                            min(spec.active_rounds[0], reference.rounds - 1),
                            min(spec.active_rounds[1], reference.rounds - 1),
                        )
                    }
                )
                for spec in specs
            ]
        run_id = f"compound_{name}"
        cfg = RunConfig.model_validate(
            {**base, "run_id": run_id, "failure": None, "failures": specs}
        )
        print(f"[compound] run: {name}...", flush=True)
        _record(out, cfg)
        print(f"[compound] attribution: {name}...", flush=True)
        report, interventions = analyze_pair(
            out,
            "ref",
            run_id,
            metric="accuracy",
            higher_is_better=True,
            min_gap=0.005,
            sham_tolerance=1e-9,
        )
        (out / f"report_{name}.md").write_text(
            render_markdown(report, interventions, ground_truth=specs),
            encoding="utf-8",
        )
        row = {
            "case": name,
            "outcome": report.outcome,
            "origin_ranking": report.origin_ranking,
            "origin_set": report.origin_set,
            "notes": report.notes,
        }
        summary.append(row)
        print(f"  -> {row}", flush=True)

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
