"""Co-author suite 2 (실험2) — severity calibration sweep + scale/cost profile.

One command, no arguments:

    python experiments/run_coauthor_suite2.py

Part A (Plan §17.3): for each of the four failure cases, run the stripped
reference once and the failure at three severities (mild/moderate/severe),
recording final-metric gaps. This produces the severity-vs-gap curves
FALCON-Bench needs to declare valid severity tiers.

Part B (Plan §20 E8-lite): reference runs at 10/25/50 clients with full
recording, measuring wall-clock time and recorded-state storage.

Ends by invoking scripts/collect_output.py — send the printed zip back.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from falcon.pipeline.runner import run  # noqa: E402
from falcon.recorder import Recorder  # noqa: E402
from falcon.replay.rng import Rng  # noqa: E402
from falcon.schema import RunConfig  # noqa: E402

CASES = REPO / "configs" / "cases"
OUT = REPO / "results" / "coauthor2"

# (case yaml, severity knob inside failure.parameters, [mild, moderate, severe])
SEVERITY_GRID = [
    ("synthetic_selection_failure.yaml", "exclusion_probability", [0.3, 0.6, 0.9]),
    # lr INCREASE improves this task (measured: 3x/10x give negative gaps), so the
    # severity axis runs through too-SMALL lr toward the sign flip.
    ("synthetic_local_failure.yaml", "lr_multiplier", [0.1, 0.01, -1.0]),
    ("synthetic_compression_failure.yaml", "k_ratio", [0.5, 0.2, 0.05]),
    ("synthetic_aggregation_failure.yaml", "mode", ["uniform", "swapped", "corrupted"]),
]
SCALE_CLIENTS = [10, 25, 50]


def _load(name: str) -> dict:
    return yaml.safe_load((CASES / name).read_text(encoding="utf-8"))


def _final_metrics(cfg: RunConfig) -> dict:
    outcomes = run(cfg, rng=Rng(cfg.seed))
    last = outcomes[-1]
    flat = dict(last.metrics)
    for cls, m in last.per_class.items():
        for k, v in m.items():
            flat[f"class_{cls}_{k}"] = v
    return flat


def severity_sweep() -> list[dict]:
    results = []
    for yaml_name, knob, values in SEVERITY_GRID:
        raw = _load(yaml_name)
        ref_raw = {k: v for k, v in raw.items() if k != "failure"}
        ref_raw["run_id"] = raw["run_id"] + "_ref"
        ref = _final_metrics(RunConfig(**ref_raw))
        for level, value in zip(("mild", "moderate", "severe"), values):
            fail_raw = json.loads(json.dumps(raw))  # deep copy
            fail_raw["failure"]["parameters"][knob] = value
            fail_raw["failure"]["severity"] = {"mild": 1, "moderate": 2, "severe": 3}[level]
            fail_raw["run_id"] = f"{raw['run_id']}_{level}"
            fail = _final_metrics(RunConfig(**fail_raw))
            gaps = {k: ref[k] - fail[k] for k in ref if k in fail}
            entry = {
                "case": yaml_name,
                "stage": raw["failure"]["stage"],
                "severity": level,
                "knob": knob,
                "value": value,
                "reference": ref,
                "failure": fail,
                "gaps": gaps,
            }
            results.append(entry)
            print(f"[severity] {raw['failure']['stage']:12s} {level:8s} "
                  f"{knob}={value!s:<10s} acc_gap={gaps.get('accuracy', float('nan')):+.4f}")
    return results


def scale_profile() -> list[dict]:
    raw = _load("synthetic_reference.yaml")
    results = []
    for n in SCALE_CLIENTS:
        r = json.loads(json.dumps(raw))
        r["dataset"]["num_clients"] = n
        r["selection"]["clients_per_round"] = max(2, n // 2)
        r["run_id"] = f"scale_{n}"
        cfg = RunConfig(**r)
        root = OUT / "scale_runs"
        rec = Recorder(root, cfg.run_id)
        t0 = time.perf_counter()
        run(cfg, recorder=rec, rng=Rng(cfg.seed))
        elapsed = time.perf_counter() - t0
        run_dir = root / "runs" / cfg.run_id
        if not run_dir.exists():  # recorder layout fallback
            run_dir = next(p for p in root.rglob(cfg.run_id) if p.is_dir())
        size = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
        results.append({"num_clients": n, "clients_per_round": r["selection"]["clients_per_round"],
                        "rounds": r["rounds"], "wall_seconds": round(elapsed, 3),
                        "recorded_bytes": size})
        print(f"[scale] clients={n:3d} wall={elapsed:.2f}s recorded={size/1e6:.2f}MB")
    return results


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"suite": "coauthor2", "parts": {}}
    try:
        sev = severity_sweep()
        (OUT / "severity_sweep.json").write_text(json.dumps(sev, indent=2), encoding="utf-8")
        summary["parts"]["severity_sweep"] = "PASS"
    except Exception as e:  # keep going — partial results still useful
        summary["parts"]["severity_sweep"] = f"FAILED: {e!r}"
    try:
        scale = scale_profile()
        (OUT / "scale_cost.json").write_text(json.dumps(scale, indent=2), encoding="utf-8")
        summary["parts"]["scale_cost"] = "PASS"
    except Exception as e:
        summary["parts"]["scale_cost"] = f"FAILED: {e!r}"
    summary["status"] = "PASS" if all(v == "PASS" for v in summary["parts"].values()) else "PARTIAL"
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[suite2] {summary['status']}")
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "collect_output.py")],
                          capture_output=True, text=True)
    print(proc.stdout.strip())
    zip_line = [l for l in proc.stdout.splitlines() if "Output_" in l]
    print("\n>>> 이 파일을 보내주세요:", zip_line[-1].split("-> ")[-1] if zip_line else "tmp/Output_*.zip")


if __name__ == "__main__":
    main()
