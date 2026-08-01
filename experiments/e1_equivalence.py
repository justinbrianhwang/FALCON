"""E1 terminal observational-equivalence experiment harness.

The YAML schema is intentionally small: one reference ``RunConfig`` (without
``run_id``, ``seed``, or ``failure``), two stage-distinct ``failures`` with a
numeric ``severity`` search, and four ``terminal_training_failures``.

Terminal-only protocol: fit one deterministic labeled run for each of the
four built-in failure types (selection/minority_exclusion,
local/lr_misconfig, compression/aggressive_topk, and
aggregation/wrong_sample_weights).  Every training run uses the experiment's
reference config and seed.  Labels are injected stages, and the classifier
sees only ``terminal_features``; matched-pair states and interventions are
never training features.  This is deliberately the small, weak E1 baseline
specified in Plan section 20.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from falcon.baselines import (  # noqa: E402
    NearestCentroidStageClassifier,
    passive_localize,
    passive_stage_scores,
    terminal_features,
)
from falcon.matcher.matcher import validate_pair  # noqa: E402
from falcon.pipeline import run  # noqa: E402
from falcon.recorder import Recorder  # noqa: E402
from falcon.replay import Rng  # noqa: E402
from falcon.reporting import analyze_pair  # noqa: E402
from falcon.schema import FailureSpecification, RunConfig, RunMetadata  # noqa: E402

_INTERVENABLE_STAGES = {"selection", "local", "compression", "aggregation"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TRAINING_PROTOCOL = (
    "One labeled run per built-in failure type, all generated from the same "
    "reference config and experiment seed; labels are injected stages and "
    "NearestCentroidStageClassifier receives terminal_features only."
)


class UnmatchableSeverityError(ValueError):
    """The requested terminal gap is not attainable within severity bounds."""

    def __init__(self, message: str, trace: list[dict[str, Any]]):
        super().__init__(message)
        self.trace = trace


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _identifier(value: Any, label: str) -> str:
    value = str(value)
    if value in {".", ".."} or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} must be a safe identifier, got {value!r}")
    return value


def _number(raw: dict[str, Any], key: str, *, minimum: float = 0.0) -> float:
    value = float(raw[key])
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{key} must be finite and >= {minimum}, got {value}")
    return value


def _failure_config(
    reference: RunConfig,
    run_id: str,
    failure: dict[str, Any],
) -> RunConfig:
    return reference.model_copy(
        update={
            "run_id": run_id,
            "failure": FailureSpecification.model_validate(failure),
        },
        deep=True,
    )


def _record(root: Path, cfg: RunConfig) -> Recorder:
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
    return recorder


def _gap(reference_value: float, failure_value: float, higher_is_better: bool) -> float:
    return (
        reference_value - failure_value
        if higher_is_better
        else failure_value - reference_value
    )


def _default_pair_failures(reference: RunConfig) -> list[dict[str, Any]]:
    target_class = reference.dataset.minority_class
    if target_class is None:
        raise ValueError(
            "default minority_exclusion pair requires reference.dataset.minority_class"
        )
    active_rounds = [1 if reference.rounds > 1 else 0, reference.rounds - 1]
    return [
        {
            "id": "selection_minority_exclusion",
            "stage": "selection",
            "type": "minority_exclusion",
            "active_rounds": active_rounds,
            "parameters": {"target_class": target_class},
            "severity": {
                "parameter": "exclusion_probability",
                "bounds": [0.0, 1.0],
                "higher_is_more_severe": True,
            },
        },
        {
            "id": "compression_aggressive_topk",
            "stage": "compression",
            "type": "aggressive_topk",
            "active_rounds": active_rounds,
            "parameters": {},
            "severity": {
                "parameter": "k_ratio",
                "bounds": [0.01, 1.0],
                "higher_is_more_severe": False,
            },
        },
    ]


def _bisect_match(
    reference: RunConfig,
    reference_value: float,
    failure_case: dict[str, Any],
    *,
    metric: str,
    higher_is_better: bool,
    target_gap: float,
    gap_tolerance: float,
    max_iterations: int,
) -> tuple[RunConfig, list[dict[str, Any]]]:
    case_id = _identifier(failure_case["id"], "failure id")
    severity = failure_case.get("severity")
    if not isinstance(severity, dict):
        raise ValueError(f"failure {case_id!r} needs a severity mapping")
    parameter = str(severity["parameter"])
    bounds = severity.get("bounds")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or any(not math.isfinite(float(value)) for value in bounds)
    ):
        raise ValueError(f"failure {case_id!r} severity.bounds needs two finite values")
    lower, upper = map(float, bounds)
    if lower >= upper:
        raise ValueError(f"failure {case_id!r} severity bounds must be increasing")
    higher_is_more_severe = severity.get("higher_is_more_severe")
    if not isinstance(higher_is_more_severe, bool):
        raise ValueError(
            f"failure {case_id!r} severity.higher_is_more_severe must be boolean"
        )

    base_failure = {
        key: copy.deepcopy(value)
        for key, value in failure_case.items()
        if key not in {"id", "severity"}
    }
    if base_failure.get("stage") not in _INTERVENABLE_STAGES:
        raise ValueError(f"failure {case_id!r} has invalid stage {base_failure.get('stage')!r}")
    base_failure.setdefault("severity", 1)
    parameters = base_failure.setdefault("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError(f"failure {case_id!r} parameters must be a mapping")

    trace: list[dict[str, Any]] = []
    def evaluate(value: float, phase: str, iteration: int) -> tuple[RunConfig, float]:
        failure = copy.deepcopy(base_failure)
        failure["parameters"][parameter] = value
        cfg = _failure_config(reference, f"e1_search_{case_id}_{len(trace)}", failure)
        terminal_value = float(run(cfg, rng=Rng(cfg.seed))[-1].metrics[metric])
        gap = _gap(reference_value, terminal_value, higher_is_better)
        if not math.isfinite(gap):
            raise UnmatchableSeverityError(
                f"failure {case_id!r} produced a non-finite {metric} gap at {parameter}={value}",
                trace,
            )
        trace.append(
            {
                "phase": phase,
                "iteration": iteration,
                "parameter": parameter,
                "value": value,
                "terminal_metric": terminal_value,
                "gap": gap,
                "target_gap": target_gap,
                "absolute_error": abs(gap - target_gap),
                "matched": abs(gap - target_gap) <= gap_tolerance,
            }
        )
        return cfg, gap

    low_cfg, low_gap = evaluate(lower, "bound", 0)
    high_cfg, high_gap = evaluate(upper, "bound", 0)
    endpoint_matches = [
        (cfg, gap)
        for cfg, gap in ((low_cfg, low_gap), (high_cfg, high_gap))
        if abs(gap - target_gap) <= gap_tolerance
    ]
    if endpoint_matches:
        chosen, _ = min(endpoint_matches, key=lambda item: abs(item[1] - target_gap))
        return chosen.model_copy(update={"run_id": f"e1_{case_id}"}), trace

    if higher_is_more_severe:
        mild_value, mild_gap = lower, low_gap
        severe_value, severe_gap = upper, high_gap
    else:
        mild_value, mild_gap = upper, high_gap
        severe_value, severe_gap = lower, low_gap
    if mild_gap > severe_gap:
        raise UnmatchableSeverityError(
            f"failure {case_id!r} gap is not monotone in the declared severity "
            f"direction: mild={mild_gap:.6g}, severe={severe_gap:.6g}",
            trace,
        )
    if not mild_gap <= target_gap <= severe_gap:
        raise UnmatchableSeverityError(
            f"failure {case_id!r} cannot match target gap {target_gap:.6g} within "
            f"bounds {bounds}: attainable endpoint gaps are {mild_gap:.6g} to "
            f"{severe_gap:.6g}",
            trace,
        )

    for iteration in range(1, max_iterations + 1):
        midpoint = (mild_value + severe_value) / 2.0
        cfg, gap = evaluate(midpoint, "bisection", iteration)
        if abs(gap - target_gap) <= gap_tolerance:
            return cfg.model_copy(update={"run_id": f"e1_{case_id}"}), trace
        if gap < target_gap:
            mild_value, mild_gap = midpoint, gap
        else:
            severe_value, severe_gap = midpoint, gap

    best = min(trace, key=lambda item: item["absolute_error"])
    raise UnmatchableSeverityError(
        f"failure {case_id!r} did not match target gap {target_gap:.6g} +/- "
        f"{gap_tolerance:.6g} in {max_iterations} bisection iterations; best "
        f"gap was {best['gap']:.6g} at {parameter}={best['value']:.6g}",
        trace,
    )


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# E1 terminal observational equivalence",
        "",
        f"Status: **{summary['status']}**. Target gap: "
        f"{summary['target_gap']:.6g} +/- {summary['gap_tolerance']:.6g} "
        f"({summary['metric']}).",
        "",
        "| Case | Truth | Gap | Terminal-only | Passive | FALCON | FALCON outcome |",
        "|---|---|---:|---|---|---|---|",
    ]
    for case in summary.get("cases", []):
        predictions = case["predictions"]
        lines.append(
            f"| {case['id']} | {case['ground_truth']} | {case['gap']:.6g} | "
            f"{predictions['terminal_only']} | {predictions['passive']} | "
            f"{predictions['falcon']} | {case['falcon']['outcome']} |"
        )
    lines.extend(["", f"Terminal-only training protocol: {_TRAINING_PROTOCOL}", ""])
    return "\n".join(lines)


def _load_spec(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("E1 config must be a YAML mapping")
    return raw


def run_experiment(config_path: Path, output_root: Path) -> dict[str, Any]:
    raw = _load_spec(config_path)
    case_id = _identifier(raw["case_id"], "case_id")
    seed = int(raw["seed"])
    reference_raw = raw.get("reference")
    if not isinstance(reference_raw, dict):
        raise ValueError("reference must be a RunConfig mapping")
    if reference_raw.get("failure") is not None:
        raise ValueError("reference.failure must be null or omitted")
    reference_payload = copy.deepcopy(reference_raw)
    reference_payload.update(
        {"run_id": f"e1_{case_id}_reference", "seed": seed, "failure": None}
    )
    reference = RunConfig.model_validate(reference_payload)

    metric = str(raw.get("metric", "accuracy"))
    higher_is_better = raw.get("higher_is_better", True)
    if not isinstance(higher_is_better, bool):
        raise ValueError("higher_is_better must be boolean")
    target_gap = _number(raw, "target_gap")
    gap_tolerance = _number(raw, "gap_tolerance")
    max_iterations = int(raw.get("max_iterations", 12))
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    min_gap = float(raw.get("min_gap", max(1e-9, target_gap - gap_tolerance)))
    sham_tolerance = float(raw.get("sham_tolerance", 1e-9))
    if min_gap < 0 or not math.isfinite(min_gap):
        raise ValueError("min_gap must be finite and nonnegative")
    if sham_tolerance <= 0 or not math.isfinite(sham_tolerance):
        raise ValueError("sham_tolerance must be finite and positive")

    pair_failures = raw.get("failures") or _default_pair_failures(reference)
    if not isinstance(pair_failures, list) or len(pair_failures) != 2:
        raise ValueError("failures must contain exactly two failure cases")
    pair_stages = [failure.get("stage") for failure in pair_failures]
    if len(set(pair_stages)) != 2:
        raise ValueError("the two E1 failures must have different origin stages")

    case_dir = Path(output_root) / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    reference_value = float(run(reference, rng=Rng(reference.seed))[-1].metrics[metric])
    if not math.isfinite(reference_value):
        raise ValueError(f"reference produced a non-finite terminal metric {metric!r}")

    matched: list[RunConfig] = []
    traces: dict[str, Any] = {
        "metric": metric,
        "higher_is_better": higher_is_better,
        "reference_terminal_metric": reference_value,
        "target_gap": target_gap,
        "gap_tolerance": gap_tolerance,
        "max_iterations": max_iterations,
        "failures": {},
    }
    try:
        for failure in pair_failures:
            failure_id = _identifier(failure["id"], "failure id")
            cfg, trace = _bisect_match(
                reference,
                reference_value,
                failure,
                metric=metric,
                higher_is_better=higher_is_better,
                target_gap=target_gap,
                gap_tolerance=gap_tolerance,
                max_iterations=max_iterations,
            )
            matched.append(cfg)
            traces["failures"][failure_id] = trace
    except UnmatchableSeverityError as exc:
        failure_id = _identifier(failure["id"], "failure id")
        traces["failures"][failure_id] = exc.trace
        _write_json(case_dir / "severity_traces.json", traces)
        failed_summary = {
            "experiment": "E1",
            "case_id": case_id,
            "status": "UNMATCHABLE",
            "metric": metric,
            "target_gap": target_gap,
            "gap_tolerance": gap_tolerance,
            "error": str(exc),
            "cases": [],
        }
        _write_json(case_dir / "summary.json", failed_summary)
        (case_dir / "summary.md").write_text(
            f"# E1 terminal observational equivalence\n\n"
            f"Status: **UNMATCHABLE**. {exc}\n",
            encoding="utf-8",
        )
        raise

    training_failures = raw.get("terminal_training_failures")
    if not isinstance(training_failures, list) or not training_failures:
        raise ValueError("terminal_training_failures must be a non-empty list")
    training_stages = [item.get("stage") for item in training_failures]
    if set(training_stages) != _INTERVENABLE_STAGES:
        raise ValueError(
            "terminal_training_failures must cover selection, local, compression, "
            "and aggregation exactly"
        )

    with tempfile.TemporaryDirectory(prefix=f"falcon_e1_{case_id}_") as temp:
        runs_root = Path(temp)
        _record(runs_root, reference)
        for cfg in matched:
            _record(runs_root, cfg)

        training_configs = [
            _failure_config(
                reference,
                f"e1_training_{index}_{item['stage']}",
                item,
            )
            for index, item in enumerate(training_failures)
        ]
        for cfg in training_configs:
            _record(runs_root, cfg)
        classifier = NearestCentroidStageClassifier().fit(
            [terminal_features(runs_root, cfg.run_id) for cfg in training_configs],
            [cfg.failure.stage for cfg in training_configs if cfg.failure is not None],
        )

        cases = []
        predictions_output = {
            "terminal_training_protocol": _TRAINING_PROTOCOL,
            "cases": [],
        }
        for cfg in matched:
            assert cfg.failure is not None
            pair = validate_pair(
                runs_root / "runs" / reference.run_id,
                runs_root / "runs" / cfg.run_id,
            )
            if pair.status == "INVALID_PAIR":
                raise ValueError(f"constructed pair for {cfg.run_id!r} is invalid")
            terminal_prediction = classifier.predict(terminal_features(runs_root, cfg.run_id))
            passive_scores = passive_stage_scores(
                runs_root, runs_root, reference.run_id, cfg.run_id
            )
            passive_prediction = passive_localize(passive_scores)
            falcon_report, _ = analyze_pair(
                runs_root,
                reference.run_id,
                cfg.run_id,
                metric=metric,
                higher_is_better=higher_is_better,
                min_gap=min_gap,
                sham_tolerance=sham_tolerance,
            )
            falcon_prediction = (
                falcon_report.origin_ranking[0]
                if falcon_report.outcome == "unique_origin"
                and falcon_report.origin_ranking
                else "unresolved"
            )
            failure_value = float(
                Recorder(runs_root, cfg.run_id)
                .load(cfg.rounds - 1, "evaluation")
                .metrics[metric]
            )
            case = {
                "id": cfg.run_id.removeprefix("e1_"),
                "ground_truth": cfg.failure.stage,
                "failure_type": cfg.failure.type,
                "severity_parameters": cfg.failure.parameters,
                "terminal_metric": failure_value,
                "gap": _gap(reference_value, failure_value, higher_is_better),
                "pair_status": pair.status,
                "predictions": {
                    "terminal_only": terminal_prediction,
                    "passive": passive_prediction,
                    "falcon": falcon_prediction,
                },
                "passive_scores": passive_scores,
                "falcon": {
                    "prediction": falcon_prediction,
                    "outcome": falcon_report.outcome,
                    "origin_ranking": falcon_report.origin_ranking,
                    "origin_set": falcon_report.origin_set,
                    "stage_effects": falcon_report.stage_effects,
                    "notes": falcon_report.notes,
                },
            }
            cases.append(case)
            predictions_output["cases"].append(
                {
                    "id": case["id"],
                    "ground_truth": case["ground_truth"],
                    "predictions": case["predictions"],
                    "falcon_outcome": case["falcon"]["outcome"],
                }
            )

    matched_configs = {
        "reference": reference.model_dump(mode="json"),
        "failures": [cfg.model_dump(mode="json") for cfg in matched],
    }
    (case_dir / "matched_configs.yaml").write_text(
        yaml.safe_dump(matched_configs, sort_keys=False), encoding="utf-8"
    )
    _write_json(case_dir / "severity_traces.json", traces)
    _write_json(case_dir / "predictions.json", predictions_output)
    summary = {
        "experiment": "E1",
        "case_id": case_id,
        "status": "PASS",
        "seed": seed,
        "metric": metric,
        "higher_is_better": higher_is_better,
        "reference_terminal_metric": reference_value,
        "target_gap": target_gap,
        "gap_tolerance": gap_tolerance,
        "terminal_training_protocol": _TRAINING_PROTOCOL,
        "terminal_training_labels": training_stages,
        "cases": cases,
    }
    _write_json(case_dir / "summary.json", summary)
    (case_dir / "summary.md").write_text(_render_markdown(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/e1"))
    args = parser.parse_args(argv)
    try:
        run_experiment(args.config, args.output)
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"E1 harness refused experiment: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
