"""Run the co-author 3 E3, E8, and dataset-replication suite.

    python experiments/run_coauthor3_suite.py
    python experiments/run_coauthor3_suite.py --smoke
    python experiments/run_coauthor3_suite.py --parts e3,e8,datasets
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.run_coauthor_cifar import CASE_METRIC, FAILURES  # noqa: E402
from falcon.baselines import (  # noqa: E402
    NearestCentroidStageClassifier,
    passive_localize,
    passive_stage_scores,
    terminal_features,
)
from falcon.data_paths import processed_path  # noqa: E402
from falcon.intervention import apply_intervention  # noqa: E402
from falcon.pipeline.runner import run  # noqa: E402
from falcon.recorder import Recorder  # noqa: E402
from falcon.replay.rng import Rng  # noqa: E402
from falcon.reporting.analyze import analyze_pair  # noqa: E402
from falcon.reporting.report import render_markdown  # noqa: E402
from falcon.schema import (  # noqa: E402
    FailureSpecification,
    InterventionSpecification,
    RunConfig,
    RunMetadata,
)

CASES = REPO / "configs" / "cases"
E3_ALPHAS = [0.1, 0.5, 1.0, 10.0]
E8_CLIENTS = [10, 25, 50, 100]
PARTS = {"e3", "e8", "datasets"}

E3_FAILURES = [
    (
        "selection",
        FailureSpecification(
            stage="selection",
            type="minority_exclusion",
            active_rounds=(1, 4),
            severity=3,
            parameters={"target_class": 5, "exclusion_probability": 1.0},
        ),
    ),
    (
        "local",
        FailureSpecification(
            stage="local",
            type="lr_misconfig",
            active_rounds=(1, 4),
            severity=2,
            parameters={"fraction": 0.5, "lr_multiplier": -1.0},
        ),
    ),
]

# Match the main-matrix terminal-only protocol without training on evaluated runs.
TERMINAL_TRAINING_FAILURES = [
    FailureSpecification(
        stage="selection",
        type="minority_exclusion",
        active_rounds=(1, 4),
        severity=1,
        parameters={"target_class": 5, "exclusion_probability": 0.75},
    ),
    FailureSpecification(
        stage="local",
        type="lr_misconfig",
        active_rounds=(1, 4),
        severity=1,
        parameters={"fraction": 0.5, "lr_multiplier": 0.0},
    ),
    FailureSpecification(
        stage="compression",
        type="aggressive_topk",
        active_rounds=(1, 4),
        severity=1,
        parameters={"k_ratio": 0.05},
    ),
    FailureSpecification(
        stage="aggregation",
        type="wrong_sample_weights",
        active_rounds=(1, 4),
        severity=1,
        parameters={"mode": "corrupted"},
    ),
]

def _load_case(name: str) -> dict[str, Any]:
    return yaml.safe_load((CASES / name).read_text(encoding="utf-8"))


def _record(root: Path, cfg: RunConfig) -> None:
    recorder = Recorder(root, cfg.run_id)
    recorder.save_metadata(
        RunMetadata(
            run_id=cfg.run_id,
            seed=cfg.seed,
            rounds=cfg.rounds,
            config=cfg.model_dump(mode="json", exclude={"run_id"}),
            failure=cfg.failure,
        )
    )
    run(cfg, recorder=recorder, rng=Rng(cfg.seed))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _falcon_prediction(report: Any) -> str | None:
    return report.origin_ranking[0] if report.origin_ranking else None


def _fit_terminal(
    root: Path, base: dict[str, Any]
) -> NearestCentroidStageClassifier:
    configs = []
    for index, original_spec in enumerate(TERMINAL_TRAINING_FAILURES):
        spec = original_spec.model_copy(
            update={"active_rounds": (1, base["rounds"] - 1)}
        )
        cfg = RunConfig(
            **{
                **base,
                "run_id": f"e3_terminal_{index}_{spec.stage}",
                "failure": spec,
            }
        )
        _record(root, cfg)
        configs.append(cfg)
    return NearestCentroidStageClassifier().fit(
        [terminal_features(root, cfg.run_id) for cfg in configs],
        [cfg.failure.stage for cfg in configs if cfg.failure is not None],
    )


def _e3_table(rows: list[dict[str, Any]], alphas: list[float]) -> str:
    by_case = {(row["alpha"], row["failure"]): row for row in rows}
    lines = [
        "# E3 heterogeneity stress",
        "",
        "H5 comparison: FALCON stability and passive degradation are shown side by side.",
        "",
        "| alpha | selection FALCON | selection passive | selection terminal | "
        "local FALCON | local passive | local terminal |",
        "|---:|---|---|---|---|---|---|",
    ]
    for alpha in alphas:
        cells = []
        for failure in ("selection", "local"):
            row = by_case.get((alpha, failure))
            if row is None:
                cells.extend(["-", "-", "-"])
            else:
                cells.extend(
                    [
                        f"{row['falcon_outcome']} / {row['falcon_prediction'] or '-'}",
                        row["passive_prediction"],
                        row["terminal_prediction"],
                    ]
                )
        lines.append(f"| {alpha:g} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run_e3(out: Path, smoke: bool) -> list[dict[str, Any]]:
    print("[e3] starting heterogeneity stress", flush=True)
    root = out / "e3_runs"
    alphas = [0.5] if smoke else E3_ALPHAS
    failures = E3_FAILURES[:1] if smoke else E3_FAILURES
    rows = []
    training_base = copy.deepcopy(_load_case("mnist_reference.yaml"))
    training_base["rounds"] = 4 if smoke else 5
    print("[e3] terminal baseline training", flush=True)
    classifier = _fit_terminal(root, training_base)
    for alpha in alphas:
        base = copy.deepcopy(_load_case("mnist_reference.yaml"))
        base["rounds"] = 4 if smoke else 5
        base["dataset"]["dirichlet_alpha"] = alpha
        tag = f"e3_a{str(alpha).replace('.', 'p')}"
        reference = RunConfig(**{**base, "run_id": f"{tag}_ref", "failure": None})
        print(f"[e3] alpha={alpha:g} reference", flush=True)
        _record(root, reference)

        failure_configs = []
        for name, original_spec in failures:
            spec = original_spec.model_copy(update={"active_rounds": (1, base["rounds"] - 1)})
            failure = RunConfig(
                **{**base, "run_id": f"{tag}_fail_{name}", "failure": spec}
            )
            print(f"[e3] alpha={alpha:g} failure={name}", flush=True)
            _record(root, failure)
            failure_configs.append(failure)

        for failure in failure_configs:
            assert failure.failure is not None
            spec = failure.failure
            name = spec.stage
            metric = "class_5_accuracy" if name == "selection" else "accuracy"
            report, interventions = analyze_pair(
                root,
                reference.run_id,
                failure.run_id,
                metric=metric,
                higher_is_better=True,
                min_gap=0.005,
                sham_tolerance=1e-9,
            )
            (out / f"e3_report_a{str(alpha).replace('.', 'p')}_{name}.md").write_text(
                render_markdown(report, interventions, ground_truth=spec), encoding="utf-8"
            )
            scores = passive_stage_scores(root, root, reference.run_id, failure.run_id)
            row = {
                "alpha": alpha,
                "failure": name,
                "ground_truth": spec.stage,
                "metric": metric,
                "falcon_outcome": report.outcome,
                "falcon_prediction": _falcon_prediction(report),
                "passive_prediction": passive_localize(scores),
                "terminal_prediction": classifier.predict(terminal_features(root, failure.run_id)),
                "passive_scores": scores,
            }
            rows.append(row)
            print(
                f"[e3] alpha={alpha:g} failure={name} "
                f"falcon={row['falcon_prediction'] or 'unresolved'} "
                f"passive={row['passive_prediction']} terminal={row['terminal_prediction']}",
                flush=True,
            )

    _write_json(out / "e3_summary.json", rows)
    (out / "e3_table.md").write_text(_e3_table(rows, alphas), encoding="utf-8")
    print("[e3] complete", flush=True)
    return rows


def _run_size(root: Path, run_id: str) -> int:
    return sum(
        path.stat().st_size
        for path in (root / "runs" / run_id).rglob("*")
        if path.is_file()
    )


def _e8_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# E8 cost profile",
        "",
        "| clients | record_s | intervention_s | bytes |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['clients']} | {row['record_s']:.3f} | "
        f"{row['intervention_s']:.3f} | {row['bytes']} |"
        for row in rows
    )
    return "\n".join(lines) + "\n"


def run_e8(out: Path, smoke: bool) -> list[dict[str, Any]]:
    print("[e8] starting cost profile", flush=True)
    root = out / "e8_runs"
    clients = [10] if smoke else E8_CLIENTS
    rows = []
    for count in clients:
        base = copy.deepcopy(_load_case("synthetic_reference.yaml"))
        base["rounds"] = 10
        base["dataset"]["num_clients"] = count
        base["selection"]["clients_per_round"] = round(0.3 * count)
        cfg = RunConfig(**{**base, "run_id": f"e8_clients_{count}", "failure": None})

        started = time.perf_counter()
        _record(root, cfg)
        record_s = time.perf_counter() - started
        midpoint = cfg.rounds // 2
        spec = InterventionSpecification(
            target_run_id=cfg.run_id,
            source_run_id=cfg.run_id,
            round_id=midpoint,
            round_window=(midpoint, cfg.rounds - 1),
            stage="aggregation",
            mode="restore",
        )
        started = time.perf_counter()
        result = apply_intervention(spec, root)
        intervention_s = time.perf_counter() - started
        if not result.valid:
            raise RuntimeError(f"E8 intervention invalid: {result.reason}")
        row = {
            "clients": count,
            "record_s": round(record_s, 3),
            "intervention_s": round(intervention_s, 3),
            "bytes": _run_size(root, cfg.run_id),
        }
        rows.append(row)
        print(
            f"[e8] clients={count} record_s={record_s:.3f} "
            f"intervention_s={intervention_s:.3f} bytes={row['bytes']}",
            flush=True,
        )

    _write_json(out / "e8_cost.json", rows)
    (out / "e8_table.md").write_text(_e8_table(rows), encoding="utf-8")
    print("[e8] complete", flush=True)
    return rows


def _datasets_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# FMNIST and SVHN stage localization",
        "",
        "| dataset | failure | ground truth | metric | FALCON outcome | FALCON prediction |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {row['dataset']} | {row['failure']} | {row['ground_truth']} | "
        f"{row['metric']} | {row['outcome']} | {row['prediction'] or '-'} |"
        for row in rows
    )
    return "\n".join(lines) + "\n"


def run_datasets(out: Path, smoke: bool) -> list[dict[str, Any]]:
    names = ["fmnist", "svhn"]
    missing = [name for name in names if not processed_path(name).exists()]
    if missing:
        print(
            "[datasets] SKIP: missing processed files: "
            + ", ".join(f"processed/{name}.pkl" for name in missing),
            flush=True,
        )
        return []

    print("[datasets] starting FMNIST and SVHN localization", flush=True)
    root = out / "dataset_runs"
    rows = []
    for name in names:
        base = copy.deepcopy(_load_case("mnist_reference.yaml"))
        base["rounds"] = 5
        base["dataset"]["name"] = name
        reference = RunConfig(**{**base, "run_id": f"{name}_ref", "failure": None})
        print(f"[datasets] dataset={name} reference", flush=True)
        _record(root, reference)
        for failure_name, original_spec in FAILURES:
            spec = original_spec.model_copy(update={"active_rounds": (1, 4)})
            failure = RunConfig(
                **{
                    **base,
                    "run_id": f"{name}_fail_{failure_name}",
                    "failure": spec,
                }
            )
            metric = CASE_METRIC.get(failure_name, "accuracy")
            print(f"[datasets] dataset={name} failure={failure_name}", flush=True)
            _record(root, failure)
            report, interventions = analyze_pair(
                root,
                reference.run_id,
                failure.run_id,
                metric=metric,
                higher_is_better=True,
                min_gap=0.005,
                sham_tolerance=1e-9,
            )
            (out / f"datasets_report_{name}_{failure_name}.md").write_text(
                render_markdown(report, interventions, ground_truth=spec), encoding="utf-8"
            )
            row = {
                "dataset": name,
                "failure": failure_name,
                "ground_truth": spec.stage,
                "metric": metric,
                "outcome": report.outcome,
                "prediction": _falcon_prediction(report),
                "origin_set": report.origin_set,
                "gap": report.failure_gap,
                "notes": report.notes,
            }
            rows.append(row)
            print(
                f"[datasets] dataset={name} failure={failure_name} "
                f"falcon={row['prediction'] or 'unresolved'}",
                flush=True,
            )

    _write_json(out / "datasets_summary.json", rows)
    (out / "datasets_table.md").write_text(_datasets_table(rows), encoding="utf-8")
    print("[datasets] complete", flush=True)
    return rows


def _parse_parts(value: str) -> set[str]:
    if value == "all":
        return set(PARTS)
    selected = {part.strip() for part in value.split(",") if part.strip()}
    unknown = selected - PARTS
    if not selected or unknown:
        raise argparse.ArgumentTypeError(
            "parts must be a comma list of e3,e8,datasets"
            + (f"; unknown: {','.join(sorted(unknown))}" if unknown else "")
        )
    return selected


def _collect_output() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "collect_output.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    if proc.stdout.strip():
        print(proc.stdout.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--parts", type=_parse_parts, default=set(PARTS), help="comma list of e3,e8,datasets"
    )
    args = parser.parse_args(argv)
    out = REPO / "results" / ("coauthor3_smoke" if args.smoke else "coauthor3")
    out.mkdir(parents=True, exist_ok=True)

    if "e3" in args.parts:
        run_e3(out, args.smoke)
    if "e8" in args.parts:
        run_e8(out, args.smoke)
    if "datasets" in args.parts:
        run_datasets(out, args.smoke)
    _collect_output()
    print("[suite] complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
