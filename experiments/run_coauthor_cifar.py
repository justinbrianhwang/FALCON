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
    # target_class 5: co-author run 2 (2026-08-03) showed the weak-regime reference learns ONLY
    # classes 0 (0.585) and 5 (0.953) — every other class sits at ~0 accuracy, so failures
    # targeting them have nothing to damage (selection gap was 0.0001 even at p=1.0). Minority-
    # targeted failures must aim at a class the reference actually learned (Plan §14.10).
    #
    # Round-3 lessons (local retry 2026-08-04): (1) the whipsaw regime has no memory — a window
    # ending before the final round is erased by the clean tail (fail_selection r59 per-class
    # was identical to ref), so the window must reach the last round; (2) global accuracy stays
    # ~constant across collapse modes, so class-targeted failures are attributed on the recorded
    # per-class metric of the targeted class (Plan §14.10 outcome vector).
    ("selection", FailureSpecification(
        stage="selection", type="minority_exclusion", active_rounds=(10, 59), severity=3,
        parameters={"target_class": 5, "exclusion_probability": 1.0})),
    ("local", FailureSpecification(
        stage="local", type="lr_misconfig", active_rounds=(10, 49), severity=2,
        parameters={"fraction": 0.5, "lr_multiplier": -1.0})),
    ("compression", FailureSpecification(
        stage="compression", type="aggressive_topk", active_rounds=(10, 49), severity=2,
        parameters={"k_ratio": 0.05})),
    ("aggregation", FailureSpecification(
        stage="aggregation", type="wrong_sample_weights", active_rounds=(10, 59), severity=2,
        parameters={"mode": "biased", "target_class": 5, "weight_multiplier": 0.1})),
]

# Class-targeted failures are judged on the targeted class's accuracy; others on global.
CASE_METRIC = {"selection": "class_5_accuracy", "aggregation": "class_5_accuracy"}


def _record(root: Path, cfg: RunConfig) -> None:
    rec = Recorder(root, cfg.run_id)
    rec.save_metadata(RunMetadata(
        run_id=cfg.run_id, seed=cfg.seed, rounds=cfg.rounds,
        config=cfg.model_dump(mode="json", exclude={"run_id"}), failure=cfg.failure))
    run(cfg, recorder=rec, rng=Rng(cfg.seed))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny MNIST mechanics check")
    ap.add_argument("--cases", default="all",
                    help="comma list of selection,local,compression,aggregation (default all)")
    args = ap.parse_args()
    wanted = None if args.cases == "all" else {s.strip() for s in args.cases.split(",")}

    base = yaml.safe_load((REPO / "configs" / "cases" / "cifar10_reference.yaml").read_text(encoding="utf-8"))
    out = REPO / "results" / ("coauthor_cifar_smoke" if args.smoke else "coauthor_cifar")
    active = None  # full run: each spec keeps its own window
    if args.smoke:
        base = yaml.safe_load((REPO / "configs" / "cases" / "mnist_reference.yaml").read_text(encoding="utf-8"))
        base["rounds"] = 4
        active = (1, 3)
    out.mkdir(parents=True, exist_ok=True)

    ref_cfg = RunConfig(**{**base, "run_id": "ref", "failure": None})
    print(f"[cifar-suite] reference run ({ref_cfg.rounds} rounds)...", flush=True)
    _record(out, ref_cfg)

    summary = []
    for name, spec in FAILURES:
        if wanted is not None and name not in wanted:
            continue
        if active is not None:
            spec = spec.model_copy(update={"active_rounds": active})
        metric = CASE_METRIC.get(name, "accuracy")
        fail_id = f"fail_{name}"
        cfg = RunConfig(**{**base, "run_id": fail_id, "failure": spec})
        print(f"[cifar-suite] failure run: {name} ...", flush=True)
        _record(out, cfg)
        print(f"[cifar-suite] attribution: {name} on {metric} (interventions replay, slow)...", flush=True)
        try:
            report, interventions = analyze_pair(
                out, "ref", fail_id, metric=metric, higher_is_better=True,
                min_gap=0.005, sham_tolerance=1e-9)
            (out / f"report_{name}.md").write_text(
                render_markdown(report, interventions, ground_truth=spec), encoding="utf-8")
            row = {"failure": name, "ground_truth": spec.stage, "metric": metric,
                   "outcome": report.outcome,
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
