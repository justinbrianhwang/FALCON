"""Integration tests for the T17 deterministic biased-weights failure case.

``configs/cases/synthetic_aggregation_biased.yaml`` runs
``wrong_sample_weights/biased``: minority-heavy clients (same targeting rule
as selection's minority_exclusion) get their aggregation weight multiplied by
``weight_multiplier`` before normalization, with no RNG at all. Measured on
the committed config (seed 42, reference final accuracy 0.824):

- multiplier 1.0: bit-identical stage hashes vs the reference (gap +0.000);
- multiplier 0.5: accuracy gap +0.038, loss gap +0.033;
- multiplier 0.1 (committed): accuracy gap +0.128, loss gap +0.094,
  minority-class accuracy gap +0.229.
"""
import copy
from pathlib import Path

import pytest
import yaml

from falcon.pipeline.runner import run
from falcon.recorder.recorder import Recorder
from falcon.replay.rng import Rng
from falcon.schema import RunConfig

CASE_FILE = (
    Path(__file__).resolve().parents[2]
    / "configs" / "cases" / "synthetic_aggregation_biased.yaml"
)
STAGE_ORDER = ("selection", "local", "compression", "aggregation", "evaluation")


def _load_case(weight_multiplier: float | None = None) -> RunConfig:
    payload = yaml.safe_load(CASE_FILE.read_text(encoding="utf-8"))
    if weight_multiplier is not None:
        payload["failure"]["parameters"]["weight_multiplier"] = weight_multiplier
    return RunConfig(**payload)


def _reference_cfg(case: RunConfig) -> RunConfig:
    payload = case.model_dump()
    payload["failure"] = None
    payload["run_id"] = "synthetic_aggregation_biased_reference"
    return RunConfig(**payload)


def _run_recorded(cfg: RunConfig, root: Path, run_id: str):
    recorder = Recorder(root, run_id)
    outcomes = run(cfg, recorder=recorder, rng=Rng(cfg.seed))
    return recorder.stage_hashes(), outcomes


def test_multiplier_1_is_bit_identical_to_reference(tmp_path):
    """T17: weight_multiplier=1.0 is a provable no-op on the full pipeline."""
    reference_cfg = _reference_cfg(_load_case())
    noop_cfg = _load_case(weight_multiplier=1.0)

    reference_hashes, reference = _run_recorded(reference_cfg, tmp_path, "reference")
    noop_hashes, noop = _run_recorded(noop_cfg, tmp_path, "noop")

    for round_id in range(noop_cfg.rounds):
        for stage in STAGE_ORDER:
            assert reference_hashes[(round_id, stage)] == noop_hashes[(round_id, stage)], (
                f"divergence at round {round_id} stage {stage}"
            )
    assert noop[-1].metrics == reference[-1].metrics


def test_accuracy_gap_monotone_in_weight_multiplier(tmp_path):
    """Measured gaps: 0.000 (1.0), +0.038 (0.5), +0.128 (0.1) — direction
    asserted with generous margins."""
    reference_cfg = _reference_cfg(_load_case())
    _, reference = _run_recorded(reference_cfg, tmp_path, "reference")
    reference_accuracy = reference[-1].metrics["accuracy"]

    gaps = {}
    for multiplier in (1.0, 0.5, 0.1):
        _, failure = _run_recorded(
            _load_case(weight_multiplier=multiplier), tmp_path, f"failure-{multiplier}"
        )
        gaps[multiplier] = reference_accuracy - failure[-1].metrics["accuracy"]

    assert gaps[1.0] == 0.0  # exact no-op
    assert gaps[0.5] >= 0.02, f"gap at 0.5 too small: {gaps[0.5]}"
    assert gaps[0.1] >= gaps[0.5] + 0.03, (
        f"gap not monotone: 0.5 -> {gaps[0.5]}, 0.1 -> {gaps[0.1]}"
    )
    assert gaps[0.1] >= 0.08, f"gap at 0.1 too small: {gaps[0.1]}"


def test_biased_failure_run_is_deterministic(tmp_path):
    """Two runs of the committed config (multiplier 0.1) are bit-identical."""
    hashes_a, _ = _run_recorded(_load_case(), tmp_path, "det-a")
    hashes_b, _ = _run_recorded(_load_case(), tmp_path, "det-b")
    assert hashes_a == hashes_b
