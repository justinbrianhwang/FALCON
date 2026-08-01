"""E0 deterministic replay and sham-validation harness."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from falcon.intervention import apply_intervention  # noqa: E402
from falcon.pipeline import run  # noqa: E402
from falcon.recorder import Recorder  # noqa: E402
from falcon.replay import Rng  # noqa: E402
from falcon.schema import (  # noqa: E402
    STAGES,
    InterventionSpecification,
    RunConfig,
    RunMetadata,
)

SHAM_TOLERANCE = 1e-12


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


def _boundary_results(first: Recorder, second: Recorder, rounds: int) -> list[dict]:
    first_hashes = first.stage_hashes()
    second_hashes = second.stage_hashes()
    expected = {(round_id, stage) for round_id in range(rounds) for stage in STAGES}
    stage_order = {stage: index for index, stage in enumerate(STAGES)}
    boundaries = sorted(
        expected | set(first_hashes) | set(second_hashes),
        key=lambda boundary: (boundary[0], stage_order[boundary[1]]),
    )
    return [
        {
            "round": round_id,
            "stage": stage,
            "match": (
                boundary in expected
                and first_hashes.get(boundary) is not None
                and first_hashes.get(boundary) == second_hashes.get(boundary)
            ),
            "first_hash": first_hashes.get(boundary),
            "second_hash": second_hashes.get(boundary),
        }
        for boundary in boundaries
        for round_id, stage in [boundary]
    ]


def _checkpoint_check(recorder: Recorder, round_id: int, seed: int) -> dict:
    """Validate the public RNG restore API and report the suffix-runner gap."""
    recorded_streams = recorder.load(round_id, "selection").rng_state
    snapshot = {"root_seed": seed, "streams": recorded_streams}
    restored = Rng(0)
    restored.load_state_dict(snapshot)

    def encoded(state: dict) -> str:
        return json.dumps(state, sort_keys=True, separators=(",", ":"))

    snapshot_matches = encoded(restored.state_dict()) == encoded(snapshot)
    left, right = Rng(0), Rng(0)
    left.load_state_dict(snapshot)
    right.load_state_dict(snapshot)
    continuation_matches = all(
        np.array_equal(
            left.stream(name).integers(0, 2**63, size=4, dtype=np.int64),
            right.stream(name).integers(0, 2**63, size=4, dtype=np.int64),
        )
        for name in snapshot.get("streams", {})
    )
    restored_ok = snapshot_matches and continuation_matches
    return {
        "status": "API_GAP" if restored_ok else "FAIL",
        "round": round_id,
        "rng_state_restored": restored_ok,
        "restored_streams": sorted(recorded_streams),
        "suffix_hashes_compared": False,
        "reason": (
            "The recorded per-stream RNG snapshot restores and continues "
            "equivalently, but recordings do not contain a complete RNG registry "
            "and public run() has no model-checkpoint/start-round input for suffix replay."
            if restored_ok
            else "The recorded RNG snapshot did not restore equivalently."
        ),
    }


def _validate_config(raw: dict, index: int, root: Path) -> dict:
    source_cfg = RunConfig.model_validate(raw)
    first_cfg = source_cfg.model_copy(update={"run_id": f"e0_{index}_first"})
    second_cfg = source_cfg.model_copy(update={"run_id": f"e0_{index}_second"})
    first = _record(root, first_cfg)
    second = _record(root, second_cfg)

    boundaries = _boundary_results(first, second, source_cfg.rounds)
    mismatches = [boundary for boundary in boundaries if not boundary["match"]]
    replay_level = "bitwise" if not mismatches else "mismatch"
    mid_round = source_cfg.rounds // 2

    sham_results = []
    deviations = []
    for stage in STAGES:
        result = apply_intervention(
            InterventionSpecification(
                target_run_id=first_cfg.run_id,
                source_run_id=first_cfg.run_id,
                round_id=mid_round,
                stage=stage,
                mode="sham",
            ),
            root,
        )
        stage_deviations = {
            key.removeprefix("sham_deviation_"): value
            for key, value in result.outcome_metrics.items()
            if key.startswith("sham_deviation_")
        }
        deviations.extend(abs(value) for value in stage_deviations.values())
        sham_results.append(
            {
                "stage": stage,
                "valid": result.valid,
                "reason": result.reason,
                "deviations": stage_deviations,
            }
        )

    max_deviation = max(deviations, default=math.inf)
    sham_ok = (
        all(result["valid"] and result["deviations"] for result in sham_results)
        and math.isfinite(max_deviation)
        and max_deviation <= SHAM_TOLERANCE
    )
    checkpoint = _checkpoint_check(first, mid_round, source_cfg.seed)
    passed = replay_level == "bitwise" and sham_ok and checkpoint["rng_state_restored"]
    return {
        "config_id": source_cfg.run_id,
        "failure_specified": source_cfg.failure is not None,
        "status": "PASS" if passed else "FAIL",
        "replay_level": replay_level,
        "boundaries_checked": len(boundaries),
        "boundary_agreement": boundaries,
        "mismatched_boundaries": mismatches,
        "sham_round": mid_round,
        "sham_tolerance": SHAM_TOLERANCE,
        "sham_results": sham_results,
        "max_abs_sham_deviation": max_deviation,
        "checkpoint_restore": checkpoint,
    }


def _render_markdown(report: dict) -> str:
    lines = [
        "# E0 replay validation",
        "",
        f"Overall: **{report['status']}**; replay level: "
        f"**{report['replay_level']}**.",
        "",
        "| Config | Status | Replay | Mismatches | Max sham deviation | Checkpoint |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config in report["configs"]:
        lines.append(
            f"| {config['config_id']} | {config['status']} | "
            f"{config['replay_level']} | {len(config['mismatched_boundaries'])} | "
            f"{config['max_abs_sham_deviation']:.3g} | "
            f"{config['checkpoint_restore']['status']} |"
        )
    lines.extend(
        [
            "",
            "Checkpoint note: RNG snapshots restore equivalently, but suffix hashes "
            "cannot be checked until the public runner accepts a model checkpoint and "
            "start round.",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment(config_path: Path, output_dir: Path) -> dict:
    raw_configs = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw_configs, list) or not raw_configs:
        raise ValueError("E0 config must be a non-empty YAML list of run configs")

    with tempfile.TemporaryDirectory(prefix="falcon_e0_") as temp_dir:
        configs = [
            _validate_config(raw, index, Path(temp_dir))
            for index, raw in enumerate(raw_configs)
        ]

    report = {
        "experiment": "E0",
        "status": "PASS" if all(cfg["status"] == "PASS" for cfg in configs) else "FAIL",
        "replay_level": (
            "bitwise"
            if all(cfg["replay_level"] == "bitwise" for cfg in configs)
            else "mismatch"
        ),
        "configs": configs,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_experiment(args.config, args.output)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
