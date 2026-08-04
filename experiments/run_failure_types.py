"""Ko co-author suite for T24 failure-type broadening on MNIST.

    python experiments/run_failure_types.py
    python experiments/run_failure_types.py --smoke
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from run_outcome_vector import VECTOR  # noqa: E402

from falcon.intervention import apply_intervention  # noqa: E402
from falcon.pipeline.runner import run  # noqa: E402
from falcon.recorder import Recorder  # noqa: E402
from falcon.replay.rng import Rng  # noqa: E402
from falcon.reporting.analyze import analyze_pair, analyze_pair_vector  # noqa: E402
from falcon.reporting.report import render_markdown  # noqa: E402
from falcon.schema import (  # noqa: E402
    FailureSpecification,
    InterventionSpecification,
    RunConfig,
    RunMetadata,
)

CASES = (
    (
        "label_corruption",
        FailureSpecification(
            stage="local",
            type="label_corruption",
            active_rounds=(1, 4),
            severity=2,
            parameters={"fraction_clients": 0.3, "flip_probability": 0.5},
        ),
    ),
    (
        "aggressive_clipping",
        FailureSpecification(
            stage="aggregation",
            type="aggressive_clipping",
            active_rounds=(1, 4),
            severity=2,
        ),
    ),
    (
        "aggressive_quantization",
        FailureSpecification(
            stage="compression",
            type="aggressive_quantization",
            active_rounds=(1, 4),
            severity=2,
        ),
    ),
)


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


def _localize_clients(
    root: Path, base: dict, spec: FailureSpecification, failure_run_id: str
) -> dict:
    round_id = spec.active_rounds[0]
    candidates = Recorder(root, failure_run_id).load(
        round_id, "selection"
    ).selected_ids
    all_clients = sorted(f"client_{i}" for i in range(base["dataset"]["num_clients"]))
    n_affected = math.ceil(
        spec.parameters["fraction_clients"] * len(all_clients)
    )
    affected = set(all_clients[:n_affected])
    known_candidates = affected.intersection(candidates)
    failed_accuracy = Recorder(root, failure_run_id).load(
        base["rounds"] - 1, "evaluation"
    ).metrics["accuracy"]

    ranking = []
    for client_id in candidates:
        result = apply_intervention(
            InterventionSpecification(
                target_run_id=failure_run_id,
                source_run_id="ref",
                round_id=round_id,
                stage="local",
                mode="restore",
                scope={"client_ids": [client_id]},
            ),
            root,
        )
        ranking.append(
            {
                "client_id": client_id,
                "affected": client_id in affected,
                "valid": result.valid,
                "improvement": (
                    result.outcome_metrics["accuracy"] - failed_accuracy
                    if result.valid
                    else None
                ),
                "reason": result.reason,
            }
        )
    ranking.sort(
        key=lambda row: (
            not row["valid"],
            -(row["improvement"] if row["improvement"] is not None else 0.0),
            row["client_id"],
        )
    )
    top_k = len(known_candidates)
    predicted = [row["client_id"] for row in ranking[:top_k]]
    precision = (
        len(set(predicted).intersection(known_candidates)) / top_k
        if top_k
        else 0.0
    )
    return {
        "round_id": round_id,
        "candidate_clients": candidates,
        "corrupted_clients": sorted(affected),
        "corrupted_candidates": sorted(known_candidates),
        "top_k": top_k,
        "top_k_precision": precision,
        "ranking": ranking,
    }


def _write_table(root: Path, cases: list[dict], localization: dict | None) -> None:
    lines = [
        "# T24 failure-type attribution",
        "",
        "| Failure | Ground truth | Scalar outcome | Scalar prediction | Gap |",
        "|---|---|---|---|---:|",
    ]
    for row in cases:
        if "error" in row:
            lines.append(
                f"| {row['failure']} | {row['ground_truth']} | ERROR | - | - |"
            )
        else:
            lines.append(
                f"| {row['failure']} | {row['ground_truth']} | "
                f"{row['scalar']['outcome']} | {row['scalar']['prediction']} | "
                f"{row['scalar']['gap']:+.6f} |"
            )
    if localization is not None:
        lines.extend(
            [
                "",
                "## Client localization",
                "",
                f"Top-{localization['top_k']} precision: "
                f"{localization['top_k_precision']:.3f}",
                "",
                "| Rank | Client | Known corrupted | Accuracy improvement |",
                "|---:|---|---|---:|",
            ]
        )
        for rank, row in enumerate(localization["ranking"], 1):
            improvement = (
                f"{row['improvement']:+.6f}" if row["improvement"] is not None else "invalid"
            )
            lines.append(
                f"| {rank} | {row['client_id']} | {row['affected']} | {improvement} |"
            )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="four rounds, label corruption only")
    parser.add_argument(
        "--clients",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="run label-corruption client localization",
    )
    args = parser.parse_args()
    run_clients = not args.smoke if args.clients is None else args.clients

    base = yaml.safe_load(
        (REPO / "configs" / "cases" / "mnist_reference.yaml").read_text(
            encoding="utf-8"
        )
    )
    if args.smoke:
        base["rounds"] = 4
    root = REPO / "results" / (
        "failure_types_smoke" if args.smoke else "failure_types"
    )
    root.mkdir(parents=True, exist_ok=True)

    reference = RunConfig.model_validate(
        {**base, "run_id": "ref", "failure": None, "failures": []}
    )
    print(f"[failure-types] reference ({reference.rounds} rounds)...", flush=True)
    _record(root, reference)

    rows = []
    localization = None
    cases = CASES[:1] if args.smoke else CASES
    for name, original_spec in cases:
        spec = original_spec.model_copy(
            update={"active_rounds": (1, reference.rounds - 1)}
        )
        failure_run_id = f"fail_{name}"
        cfg = RunConfig.model_validate(
            {**base, "run_id": failure_run_id, "failure": spec, "failures": []}
        )
        print(f"[failure-types] failure run: {name}...", flush=True)
        _record(root, cfg)
        print(f"[failure-types] attribution: {name}...", flush=True)
        try:
            report, interventions = analyze_pair(
                root,
                "ref",
                failure_run_id,
                metric="accuracy",
                higher_is_better=True,
                min_gap=0.005,
                sham_tolerance=1e-9,
            )
            (root / f"report_{name}.md").write_text(
                render_markdown(report, interventions, ground_truth=spec),
                encoding="utf-8",
            )
            vector_reports, vector_interventions = analyze_pair_vector(
                root,
                "ref",
                failure_run_id,
                metrics=VECTOR,
                min_gap=0.005,
                sham_tolerance=1e-9,
            )
            for metric, vector_report in vector_reports.items():
                (root / f"report_vector_{name}_{metric}.md").write_text(
                    render_markdown(
                        vector_report, vector_interventions, ground_truth=spec
                    ),
                    encoding="utf-8",
                )
            row = {
                "failure": name,
                "ground_truth": spec.stage,
                "scalar": {
                    "outcome": report.outcome,
                    "prediction": (
                        report.origin_ranking[0] if report.origin_ranking else None
                    ),
                    "gap": report.failure_gap.get("accuracy", 0.0),
                },
                "vector": {
                    metric: {
                        "outcome": vector_report.outcome,
                        "prediction": (
                            vector_report.origin_ranking[0]
                            if vector_report.origin_ranking
                            else None
                        ),
                        "gap": vector_report.failure_gap.get(metric),
                    }
                    for metric, vector_report in vector_reports.items()
                },
            }
            if name == "label_corruption" and run_clients:
                print("[failure-types] client localization...", flush=True)
                localization = _localize_clients(root, base, spec, failure_run_id)
        except Exception as exc:
            row = {
                "failure": name,
                "ground_truth": spec.stage,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        print(f"  -> {row}", flush=True)

    summary = {"cases": rows, "client_localization": localization}
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    _write_table(root, rows, localization)
    print(f"[failure-types] wrote {root / 'summary.json'} and {root / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
