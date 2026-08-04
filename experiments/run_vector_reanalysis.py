"""T25 / Plan 14.10: outcome-vector reanalysis of every recorded matched pair.

    python experiments/run_vector_reanalysis.py              # all discovered pairs
    python experiments/run_vector_reanalysis.py --limit 2    # smoke: first 2 pairs
    python experiments/run_vector_reanalysis.py --roots results/compound,results/coauthor_cifar_smoke

Walks the discovery roots (default: results/e1_main_seeds, results/main_matrix,
results/coauthor_cifar_smoke -- skipped silently when absent) for run roots (a
directory with a runs/ child). In each run root, a run with failure: null (and
empty failures when the field exists) is a reference; every other run pairs with
the reference that shares its seed. Roots where pairing is ambiguous are skipped
and the reason is recorded in the summary (no guessing).

Each pair gets ONE shared intervention replay via
falcon.reporting.analyze.analyze_pair_vector, re-attributed under every metric
of VECTOR (plus class_<target>_accuracy for failures with a target_class).

Writes results/vector_reanalysis/summary.json and vector_matrix.md.
"""
import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_outcome_vector import VECTOR  # noqa: E402

from falcon.reporting.analyze import analyze_pair_vector  # noqa: E402

DEFAULT_ROOTS = (
    "results/e1_main_seeds",
    "results/main_matrix",
    "results/coauthor_cifar_smoke",
)
MIN_GAP = 0.005
SHAM_TOLERANCE = 1e-9
OUT = REPO / "results" / "vector_reanalysis"


def _rel(path: Path) -> str:
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def find_run_roots(base: Path) -> list[Path]:
    """Directories under base (base included) that have a runs/ child."""
    if not base.is_dir():
        return []  # absent roots are skipped silently
    found = []
    stack = [base]
    while stack:
        current = stack.pop()
        children = [p for p in current.iterdir() if p.is_dir()]
        if (current / "runs").is_dir():
            found.append(current)
            children = [p for p in children if p.name != "runs"]
        stack.extend(children)
    return sorted(found)


def pair_root(root: Path):
    """Return (pairs, skip_reason) for one run root.

    pairs is a list of {"ref_id", "fail_id", "ground_truth", "targets"}.
    skip_reason is None unless the root has runs but pairing is ambiguous.
    """
    metas = {}
    for run_dir in sorted((root / "runs").iterdir()):
        meta_path = run_dir / "metadata.json"
        if not run_dir.is_dir() or not meta_path.exists():
            continue
        try:
            metas[run_dir.name] = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return [], f"unreadable metadata {run_dir.name}: {type(exc).__name__}"
    refs = {rid: m for rid, m in metas.items()
            if m.get("failure") is None and not m.get("failures")}
    fails = {rid: m for rid, m in metas.items() if rid not in refs}
    if not refs or not fails:
        return [], None  # no recorded matched pairs here
    problems = []
    for fid, m in fails.items():
        matches = [rid for rid, r in refs.items() if r.get("seed") == m.get("seed")]
        if len(matches) != 1:
            problems.append(f"{fid}: {len(matches)} references with seed {m.get('seed')}")
    if problems:
        return [], "ambiguous pairing: " + "; ".join(sorted(problems))
    pairs = []
    for fid, m in sorted(fails.items()):
        ref_id = next(rid for rid, r in refs.items() if r.get("seed") == m.get("seed"))
        specs = ([m["failure"]] if m.get("failure") else []) + list(m.get("failures") or [])
        ground_truth = "+".join(str(s.get("stage", "?")) for s in specs)
        targets = sorted({s["parameters"]["target_class"] for s in specs
                          if isinstance(s.get("parameters"), dict)
                          and s["parameters"].get("target_class") is not None})
        pairs.append({"ref_id": ref_id, "fail_id": fid,
                      "ground_truth": ground_truth, "targets": targets})
    return pairs, None


def _correct(top1, ground_truth) -> bool:
    return top1 is not None and top1 in ground_truth.split("+")


def _cell(row) -> str:
    if row["outcome"] == "error":
        return f"ERROR: {row['error']}"
    gap = row["gap"]
    suffix = (f" ({gap:+.3f})" if isinstance(gap, (int, float)) and math.isfinite(gap)
              else "")
    if row["outcome"] == "unique_origin":
        return f"{row['top1']} {'V' if _correct(row['top1'], row['ground_truth']) else 'X'}{suffix}"
    return f"{row['outcome']}{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="analyze only the first N pairs (default all)")
    ap.add_argument("--roots", default=None,
                    help="comma list of discovery roots (default: "
                         + ",".join(DEFAULT_ROOTS) + ")")
    args = ap.parse_args()

    if args.roots:
        bases = [Path(r.strip()) for r in args.roots.split(",") if r.strip()]
        bases = [p if p.is_absolute() else REPO / p for p in bases]
    else:
        bases = [REPO / r for r in DEFAULT_ROOTS]

    run_roots = sorted({root for base in bases for root in find_run_roots(base)})

    pairs = []
    skipped = []
    for root in run_roots:
        found, reason = pair_root(root)
        if reason:
            skipped.append({"root": _rel(root), "reason": reason})
            print(f"[t25] SKIP {_rel(root)}: {reason}", flush=True)
            continue
        for pair in found:
            pair["root"] = _rel(root)
            pairs.append(pair)
    pairs.sort(key=lambda p: (p["root"], p["fail_id"]))
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"[t25] {len(pairs)} matched pairs from {len(run_roots)} run roots", flush=True)

    rows = []
    for pair in pairs:
        metrics = dict(VECTOR)
        for target in pair["targets"]:
            metrics[f"class_{target}_accuracy"] = {"higher_is_better": True}
        print(f"[t25] attribution: {pair['root']}/{pair['fail_id']} on {len(metrics)} "
              "metrics (one shared intervention replay)...", flush=True)
        try:
            reports, _interventions = analyze_pair_vector(
                REPO / pair["root"], pair["ref_id"], pair["fail_id"],
                metrics=metrics, min_gap=MIN_GAP, sham_tolerance=SHAM_TOLERANCE)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"  -> ERROR {error}", flush=True)
            for metric in metrics:
                rows.append({"root": pair["root"], "ref_id": pair["ref_id"],
                             "fail_id": pair["fail_id"],
                             "ground_truth": pair["ground_truth"], "metric": metric,
                             "outcome": "error", "top1": None, "gap": None,
                             "error": error})
            continue
        for metric, report in reports.items():
            rows.append({"root": pair["root"], "ref_id": pair["ref_id"],
                         "fail_id": pair["fail_id"],
                         "ground_truth": pair["ground_truth"], "metric": metric,
                         "outcome": report.outcome,
                         "top1": report.origin_ranking[0] if report.origin_ranking else None,
                         "gap": report.failure_gap.get(metric)})
        print("  -> " + ", ".join(
            f"{r['metric']}: {r['outcome']}/{r['top1']}"
            for r in rows if r["fail_id"] == pair["fail_id"] and r["root"] == pair["root"]),
            flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(
        {"rows": rows, "skipped": skipped}, indent=2, default=str), encoding="utf-8")

    # metric-disagreement matrix (Plan 14.10: report, don't average away)
    metric_names = sorted({r["metric"] for r in rows})
    by_pair = {}
    for row in rows:
        by_pair.setdefault((row["root"], row["fail_id"]), {})[row["metric"]] = row
    lines = ["# Outcome-vector reanalysis (T25 / Plan 14.10)", "",
             "| Root | Pair (gt) | " + " | ".join(metric_names) + " |",
             "|---|---|" + "---|" * len(metric_names)]
    for (root, fail_id), per_metric in by_pair.items():
        cells = [_cell(per_metric[m]) if m in per_metric else "-" for m in metric_names]
        gt = per_metric[next(iter(per_metric))]["ground_truth"]
        lines.append(f"| {root} | {fail_id} ({gt}) | " + " | ".join(cells) + " |")
    lines += ["", "## Accuracy of attribution", ""]
    for metric in metric_names:
        eff_min_gap = VECTOR.get(metric, {}).get("min_gap", MIN_GAP)
        sub = [r for r in rows if r["metric"] == metric]
        sufficient = [r for r in sub
                      if isinstance(r["gap"], (int, float)) and math.isfinite(r["gap"])
                      and r["gap"] >= eff_min_gap]
        correct = [r for r in sufficient if r["outcome"] == "unique_origin"
                   and _correct(r["top1"], r["ground_truth"])]
        lines.append(f"- ACCURACY-OF-ATTRIBUTION {metric}: {len(correct)}/{len(sufficient)} "
                     "correct unique_origin over sufficient-gap pairs")
    if skipped:
        lines += ["", "## Skipped roots", ""]
        lines += [f"- {s['root']}: {s['reason']}" for s in skipped]
    (OUT / "vector_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[t25] wrote {_rel(OUT / 'summary.json')} and {_rel(OUT / 'vector_matrix.md')}",
          flush=True)


if __name__ == "__main__":
    main()
