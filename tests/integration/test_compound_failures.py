from pathlib import Path

import yaml

from falcon.pipeline.runner import run
from falcon.recorder import Recorder
from falcon.replay.rng import Rng
from falcon.reporting.analyze import analyze_pair
from falcon.schema import FailureSpecification, RunConfig, RunMetadata

CASES = Path(__file__).resolve().parents[2] / "configs" / "cases"


def _payload(filename: str) -> dict:
    return yaml.safe_load((CASES / filename).read_text(encoding="utf-8"))


def _spec(filename: str) -> FailureSpecification:
    spec = _payload(filename)["failure"]
    spec["active_rounds"] = [1, 3]
    spec["severity"] = 2
    return FailureSpecification.model_validate(spec)


def _record(tmp_path, run_id: str, cfg: RunConfig):
    recorder = Recorder(tmp_path, run_id)
    recorder.save_metadata(
        RunMetadata(
            run_id=run_id,
            seed=cfg.seed,
            rounds=cfg.rounds,
            config=cfg.model_dump(mode="json", exclude={"run_id"}),
            failure=cfg.failure,
            failures=cfg.failures,
        )
    )
    run(cfg, recorder=recorder, rng=Rng(cfg.seed))
    return recorder.stage_hashes()


def test_compound_run_diverges_at_both_injected_stages(tmp_path):
    base = _payload("synthetic_selection_failure.yaml")
    base["rounds"] = 4
    reference = RunConfig.model_validate(
        {**base, "run_id": "reference", "failure": None, "failures": []}
    )
    compound = RunConfig.model_validate(
        {
            **base,
            "run_id": "compound",
            "failure": None,
            "failures": [
                _spec("synthetic_selection_failure.yaml"),
                _spec("synthetic_compression_failure.yaml"),
            ],
        }
    )

    reference_hashes = _record(tmp_path, "reference", reference)
    compound_hashes = _record(tmp_path, "compound", compound)

    for stage in ("selection", "compression"):
        assert any(
            reference_hashes[(round_id, stage)]
            != compound_hashes[(round_id, stage)]
            for round_id in range(1, 4)
        )

    report, interventions = analyze_pair(
        tmp_path,
        "reference",
        "compound",
        metric="accuracy",
        higher_is_better=True,
        min_gap=0.005,
        sham_tolerance=1e-9,
    )
    assert report.outcome == "unresolved"
    assert report.origin_set == report.origin_ranking
    assert "COMPOUND_FAILURE_AMBIGUITY" in report.notes
    assert all(result.spec.round_window == (1, 3) for result in interventions)


def test_single_failure_list_matches_legacy_single_failure_hashes(tmp_path):
    base = _payload("synthetic_selection_failure.yaml")
    base["rounds"] = 4
    spec = _spec("synthetic_selection_failure.yaml")
    legacy = RunConfig.model_validate(
        {**base, "run_id": "legacy", "failure": spec, "failures": []}
    )
    listed = RunConfig.model_validate(
        {**base, "run_id": "listed", "failure": None, "failures": [spec]}
    )

    assert _record(tmp_path, "legacy", legacy) == _record(tmp_path, "listed", listed)
