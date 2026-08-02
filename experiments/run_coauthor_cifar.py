"""Co-author 실험3 — CIFAR-10 stage localization (fixed severities, no bisection).

    python experiments/run_coauthor_cifar.py            # full run (several hours, CPU)
    python experiments/run_coauthor_cifar.py --smoke    # tiny MNIST mechanics check (~2 min)

Runs the committed CIFAR-10 reference once, then four fixed-severity failures
(one per stage), validates each pair, and runs the full FALCON attribution
(restore / inject / sham, window-aware) per pair. Ends with collect_output —
send the printed zip back.
"""
import argparse
import json
import subprocess
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

FAILURES = [
    ("selection", FailureSpecification(
        stage="selection", type="minority_exclusion", active_rounds=(10, 39), severity=2,
        parameters={"target_class": 1, "exclusion_probability": 0.9})),
    ("local", FailureSpecification(
        stage="local", type="lr_misconfig", active_rounds=(10, 39), severity=2,
        parameters={"fraction": 0.5, "lr_multiplier": -1.0})),
    ("compression", FailureSpecification(
        stage="compression", type="aggressive_topk", active_rounds=(10, 39), severity=2,
        parameters={"k_ratio": 0.05})),
    ("aggregation", FailureSpecification(
        stage="aggregation", type="wrong_sample_weights", active_rounds=(10, 39), severity=2,
        parameters={"mode": "biased", "target_class": 1, "weight_multiplier": 0.1})),
]


def _record(root: Path, cfg: RunConfig) -> None:
    rec = Recorder(root, cfg.run_id)
    rec.save_metadata(RunMetadata(
        run_id=cfg.run_id, seed=cfg.seed, rounds=cfg.rounds,
        config=cfg.model_dump(mode="json", exclude={"run_id"}), failure=cfg.failure))
    run(cfg, recorder=rec, rng=Rng(cfg.seed))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny MNIST mechanics check")
    args = ap.parse_args()

    base = yaml.safe_load((REPO / "configs" / "cases" / "cifar10_reference.yaml").read_text(encoding="utf-8"))
    out = REPO / "results" / ("coauthor_cifar_smoke" if args.smoke else "coauthor_cifar")
    if args.smoke:
        base = yaml.safe_load((REPO / "configs" / "cases" / "mnist_reference.yaml").read_text(encoding="utf-8"))
        base["rounds"] = 4
        active = (1, 3)
    else:
        active = (10, 39)
    out.mkdir(parents=True, exist_ok=True)

    ref_cfg = RunConfig(**{**base, "run_id": "ref", "failure": None})
    print(f"[cifar-suite] reference run ({ref_cfg.rounds} rounds)...", flush=True)
    _record(out, ref_cfg)

    summary = []
    for name, spec in FAILURES:
        spec = spec.model_copy(update={"active_rounds": active})
        fail_id = f"fail_{name}"
        cfg = RunConfig(**{**base, "run_id": fail_id, "failure": spec})
        print(f"[cifar-suite] failure run: {name} ...", flush=True)
        _record(out, cfg)
        print(f"[cifar-suite] attribution: {name} (interventions replay, slow)...", flush=True)
        try:
            report, interventions = analyze_pair(
                out, "ref", fail_id, metric="accuracy", higher_is_better=True,
                min_gap=0.005, sham_tolerance=1e-9)
            (out / f"report_{name}.md").write_text(
                render_markdown(report, interventions, ground_truth=spec), encoding="utf-8")
            row = {"failure": name, "ground_truth": spec.stage, "outcome": report.outcome,
                   "prediction": report.origin_ranking[0] if report.origin_ranking else None,
                   "origin_set": report.origin_set, "gap": report.failure_gap,
                   "notes": report.notes}
        except Exception as e:  # keep collecting independent evidence
            row = {"failure": name, "ground_truth": spec.stage, "error": f"{type(e).__name__}: {e}"}
        summary.append(row)
        print(f"  -> {row}", flush=True)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "collect_output.py")],
                          capture_output=True, text=True)
    print(proc.stdout.strip())
    zip_line = [l for l in proc.stdout.splitlines() if "Output_" in l]
    print("\n>>> 이 파일을 보내주세요:", zip_line[-1].split("-> ")[-1] if zip_line else "tmp/Output_*.zip")


if __name__ == "__main__":
    main()
