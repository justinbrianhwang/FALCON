"""E5 - aggregator-role matrix (H4, task T22).

    python experiments/run_e5_aggregators.py            # full: 4 rules x 4 failures, 10 rounds
    python experiments/run_e5_aggregators.py --smoke    # 2 rules x 1 failure, 4 rounds

H4 (Plan section 8): the aggregation stage's propagation role depends on the
aggregation rule. Under weighted_mean every E1 run showed aggregation as a
faithful carrier (CARRIER_TIE_RESOLVED:aggregation); robust rules (median,
trimmed_mean) may instead suppress upstream damage, changing both the failure
gap and the tie structure.

Base config is configs/cases/synthetic_reference.yaml with aggregation.rule
overridden per rule; failure specs keep the calibrated parameters and
active_rounds of configs/cases/synthetic_*_failure.yaml. One reference run per
rule (run_id ref_<rule>); each failure pair goes through the same
_record/analyze_pair flow as experiments/run_coauthor_cifar.py. Writes
summary.json + e5_table.md (rows = failure, cols = rule) under results/.
"""
import argparse
import json
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

# Matrix (T22): trimmed_mean keeps beta 0.2; median/trimmed_mean are
# coordinate-wise and unweighted by definition (stages.aggregate docstring).
RULES = {
    "weighted_mean": {},
    "median": {},
    "trimmed_mean": {"beta": 0.2},
    "krum": {"byzantine_f": 1},
}

# Failure specs: parameter shapes and active_rounds copied from
# configs/cases/synthetic_{selection,local,compression}_failure.yaml
# (severity 2 per the T22 matrix).
FAILURES = [
    ("selection", FailureSpecification(
        stage="selection", type="minority_exclusion", active_rounds=(2, 9), severity=2,
        parameters={"target_class": 1, "exclusion_probability": 1.0})),
    ("local", FailureSpecification(
        stage="local", type="lr_misconfig", active_rounds=(2, 9), severity=2,
        parameters={"affected_clients": ["client_1", "client_3", "client_5", "client_7"],
                    "lr_multiplier": -1.0})),
    ("compression", FailureSpecification(
        stage="compression", type="aggressive_topk", active_rounds=(2, 9), severity=2,
        parameters={"k_ratio": 0.05})),
    ("poisoning", FailureSpecification(
        stage="local", type="model_poisoning", active_rounds=(2, 9), severity=2,
        parameters={"fraction_clients": 0.2})),
]

# Smoke mode: the H4 core contrast (faithful carrier vs robust rule) on the
# mid-pipeline failure, 4 rounds with the failure active on rounds 1-3.
SMOKE_RULES = ("weighted_mean", "median")
SMOKE_FAILURES = ("local",)
SMOKE_ACTIVE = (1, 3)


def _record(root: Path, cfg: RunConfig) -> None:
    rec = Recorder(root, cfg.run_id)
    rec.save_metadata(RunMetadata(
        run_id=cfg.run_id, seed=cfg.seed, rounds=cfg.rounds,
        config=cfg.model_dump(mode="json", exclude={"run_id"}), failure=cfg.failure))
    run(cfg, recorder=rec, rng=Rng(cfg.seed))


def _carrier_tie_on_aggregation(notes: list[str]) -> bool:
    """True when a CARRIER_TIE_RESOLVED note names the aggregation stage."""
    for note in notes:
        if note.startswith("CARRIER_TIE_RESOLVED:"):
            carriers = note.split(":", 1)[1].split(",")
            if "aggregation" in carriers:
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 rules x 1 failure, 4 rounds")
    args = ap.parse_args()

    base = yaml.safe_load(
        (REPO / "configs" / "cases" / "synthetic_reference.yaml").read_text(encoding="utf-8"))
    out = REPO / "results" / ("e5_aggregators_smoke" if args.smoke else "e5_aggregators")
    rules = {k: v for k, v in RULES.items() if not args.smoke or k in SMOKE_RULES}
    failures = [(n, s) for n, s in FAILURES if not args.smoke or n in SMOKE_FAILURES]
    if args.smoke:
        base["rounds"] = 4
        failures = [(n, s.model_copy(update={"active_rounds": SMOKE_ACTIVE}))
                    for n, s in failures]
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    for rule, params in rules.items():
        cfg_dict = {**base, "aggregation": {"rule": rule, "parameters": params}}
        ref_id = f"ref_{rule}"
        ref_cfg = RunConfig(**{**cfg_dict, "run_id": ref_id, "failure": None})
        print(f"[e5] reference run: {rule} ({ref_cfg.rounds} rounds)...", flush=True)
        _record(out, ref_cfg)

        for name, spec in failures:
            fail_id = f"fail_{rule}_{name}"
            cfg = RunConfig(**{**cfg_dict, "run_id": fail_id, "failure": spec})
            print(f"[e5] failure run: {rule} x {name} ...", flush=True)
            _record(out, cfg)
            print(f"[e5] attribution: {rule} x {name} ...", flush=True)
            try:
                report, interventions = analyze_pair(
                    out, ref_id, fail_id, metric="accuracy", higher_is_better=True,
                    min_gap=0.005, sham_tolerance=1e-9)
                (out / f"report_{rule}_{name}.md").write_text(
                    render_markdown(report, interventions, ground_truth=spec),
                    encoding="utf-8")
                agg_effects = report.stage_effects.get("aggregation", {})
                agg_nsre = agg_effects.get("nSRE")
                agg_nsie = agg_effects.get("nSIE")
                row = {"rule": rule, "failure": name, "ground_truth": spec.stage,
                       "outcome": report.outcome,
                       "prediction": report.origin_ranking[0] if report.origin_ranking else None,
                       "origin_set": report.origin_set,
                       "gap": report.failure_gap,
                       "aggregation_role": report.roles.get("aggregation"),
                       "carrier_tie_aggregation": _carrier_tie_on_aggregation(report.notes),
                       "aggregation_nSRE": agg_nsre,
                       "aggregation_nSIE": agg_nsie,
                       "aggregation_suppressor_evidence": bool(
                           (agg_nsre is not None and agg_nsre < 0)
                           or (agg_nsie is not None and agg_nsie < 0)),
                       "notes": report.notes}
            except Exception as e:  # keep collecting independent evidence
                row = {"rule": rule, "failure": name, "ground_truth": spec.stage,
                       "error": f"{type(e).__name__}: {e}"}
            summary.append(row)
            print(f"  -> {row}", flush=True)

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = [
        "# E5 aggregator-role matrix (H4)",
        "",
        "Cell: outcome / top-1 prediction / failure gap / aggregation-stage role.",
        "",
        "| failure | " + " | ".join(rules) + " |",
        "|---|" + "---:|" * len(rules),
    ]
    for name, _spec in failures:
        cells = []
        for rule in rules:
            row = next((r for r in summary
                        if r["rule"] == rule and r["failure"] == name), None)
            if row is None:
                cells.append("missing")
            elif "error" in row:
                cells.append(f"ERROR: {row['error']}")
            else:
                gap = row["gap"].get("accuracy")
                gap_text = "n/a" if gap is None else f"{gap:+.4f}"
                cells.append(f"{row['outcome']} / {row['prediction']} / "
                             f"{gap_text} / {row['aggregation_role']}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (out / "e5_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[e5] wrote {out / 'summary.json'} and {out / 'e5_table.md'}", flush=True)


if __name__ == "__main__":
    main()
