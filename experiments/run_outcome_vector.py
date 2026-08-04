"""Plan 14.10 outcome-vector attribution over the recorded CIFAR-10 pairs.

    python experiments/run_outcome_vector.py            # full CIFAR (records any missing runs)
    python experiments/run_outcome_vector.py --smoke    # reuse the MNIST smoke recordings
    python experiments/run_outcome_vector.py --cases selection,local

Each failure case gets ONE intervention replay set, analyzed under every
outcome metric: metric-specific attribution is reported instead of a single
scalar, including where the metrics disagree. Reuses the runs recorded by
run_coauthor_cifar.py and records any that are missing (e.g. local and
compression, which round 1 ran on the co-author machine only).
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from run_coauthor_cifar import FAILURES, _record  # noqa: E402

import yaml  # noqa: E402

from falcon.reporting.analyze import analyze_pair_vector  # noqa: E402
from falcon.reporting.report import render_markdown  # noqa: E402
from falcon.schema import RunConfig  # noqa: E402

# The recorded evaluations support these vector components (Plan 14.10);
# worst-client accuracy and ASR need per-client eval / attack labels the
# recorder does not carry, so they are out of scope without a format change.
VECTOR = {
    "accuracy": {"higher_is_better": True},
    "loss": {"higher_is_better": False, "min_gap": 0.01},
    "macro_recall": {"higher_is_better": True},
    "worst_class_accuracy": {"higher_is_better": True},
    "fairness_dispersion": {"higher_is_better": False},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="reuse tiny MNIST recordings")
    ap.add_argument("--cases", default="all",
                    help="comma list of selection,local,compression,aggregation (default all)")
    args = ap.parse_args()
    wanted = None if args.cases == "all" else {s.strip() for s in args.cases.split(",")}

    base = yaml.safe_load((REPO / "configs" / "cases" / "cifar10_reference.yaml").read_text(encoding="utf-8"))
    out = REPO / "results" / ("coauthor_cifar_smoke" if args.smoke else "coauthor_cifar")
    active = None
    if args.smoke:
        base = yaml.safe_load((REPO / "configs" / "cases" / "mnist_reference.yaml").read_text(encoding="utf-8"))
        base["rounds"] = 4
        active = (1, 3)
    out.mkdir(parents=True, exist_ok=True)

    if not (out / "runs" / "ref" / "metadata.json").exists():
        print("[vector] reference run missing, recording...", flush=True)
        _record(out, RunConfig(**{**base, "run_id": "ref", "failure": None}))

    summary = []
    for name, spec in FAILURES:
        if wanted is not None and name not in wanted:
            continue
        if active is not None:
            spec = spec.model_copy(update={"active_rounds": active})
        fail_id = f"fail_{name}"
        if not (out / "runs" / fail_id / "metadata.json").exists():
            print(f"[vector] failure run missing: {name}, recording...", flush=True)
            _record(out, RunConfig(**{**base, "run_id": fail_id, "failure": spec}))

        metrics = dict(VECTOR)
        target = spec.parameters.get("target_class")
        if target is not None:
            metrics[f"class_{target}_accuracy"] = {"higher_is_better": True}
        print(f"[vector] attribution: {name} on {len(metrics)} metrics "
              "(one shared intervention replay, slow)...", flush=True)
        try:
            reports, interventions = analyze_pair_vector(
                out, "ref", fail_id, metrics=metrics, min_gap=0.005,
                sham_tolerance=1e-9)
        except Exception as e:
            summary.append({"failure": name, "error": f"{type(e).__name__}: {e}"})
            print(f"  -> ERROR {summary[-1]['error']}", flush=True)
            continue
        (out / f"interventions_{name}.json").write_text(json.dumps(
            [r.model_dump(mode="json") for r in interventions], indent=2),
            encoding="utf-8")
        for metric, report in reports.items():
            (out / f"report_vector_{name}_{metric}.md").write_text(
                render_markdown(report, interventions, ground_truth=spec),
                encoding="utf-8")
        row = {"failure": name, "ground_truth": spec.stage, "metrics": {
            metric: {
                "outcome": report.outcome,
                "prediction": report.origin_ranking[0] if report.origin_ranking else None,
                "gap": report.failure_gap.get(metric),
            } for metric, report in reports.items()}}
        summary.append(row)
        print(f"  -> " + ", ".join(
            f"{m}: {v['outcome']}/{v['prediction']}" for m, v in row["metrics"].items()),
            flush=True)

    (out / "vector_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # metric-disagreement table (Plan 14.10: report, don't average away)
    metric_names = sorted({m for row in summary for m in row.get("metrics", {})})
    lines = ["# Outcome-vector attribution (Plan 14.10)", "",
             "| Failure (gt) | " + " | ".join(metric_names) + " |",
             "|---|" + "---|" * len(metric_names)]
    for row in summary:
        if "error" in row:
            lines.append(f"| {row['failure']} | ERROR: {row['error']} |")
            continue
        cells = []
        for metric in metric_names:
            v = row["metrics"].get(metric)
            if v is None:
                cells.append("-")
            elif v["outcome"] == "unique_origin":
                mark = "V" if v["prediction"] == row["ground_truth"] else "X"
                cells.append(f"{v['prediction']} {mark} ({v['gap']:+.3f})")
            else:
                gap = f" ({v['gap']:+.3f})" if v["gap"] is not None else ""
                cells.append(f"{v['outcome']}{gap}")
        lines.append(f"| {row['failure']} ({row['ground_truth']}) | " + " | ".join(cells) + " |")
    (out / "vector_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[vector] wrote {out / 'vector_summary.json'} and {out / 'vector_table.md'}", flush=True)


if __name__ == "__main__":
    main()
