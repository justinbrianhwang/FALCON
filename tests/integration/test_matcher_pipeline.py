import numpy as np

from falcon.matcher.matcher import validate_pair
from falcon.pipeline.runner import run
from falcon.recorder import Recorder
from falcon.replay import Rng
from falcon.schema import (
    AggregationConfig,
    CompressionConfig,
    DatasetConfig,
    FailureSpecification,
    LocalConfig,
    RunConfig,
    RunMetadata,
    SelectionConfig,
)


def _config(run_id, failure):
    return RunConfig(
        run_id=run_id,
        seed=42,
        rounds=2,
        dataset=DatasetConfig(
            num_clients=4,
            num_features=5,
            samples_per_client=20,
        ),
        selection=SelectionConfig(clients_per_round=2),
        local=LocalConfig(lr=0.25, local_steps=2, batch_size=5),
        compression=CompressionConfig(kind="identity"),
        aggregation=AggregationConfig(rule="weighted_mean"),
        failure=failure,
    )


def _record_run(tmp_path, cfg):
    recorder = Recorder(tmp_path, cfg.run_id)
    metadata_config = cfg.model_dump(mode="json", exclude={"run_id"})
    recorder.save_metadata(
        RunMetadata(
            run_id=cfg.run_id,
            seed=cfg.seed,
            rounds=cfg.rounds,
            config=metadata_config,
            failure=cfg.failure,
        )
    )
    run(cfg, recorder=recorder, rng=Rng(cfg.seed))
    return recorder


def test_real_pipeline_inert_failure_matches_with_warning(tmp_path):
    # real failure type, but its window (5, 5) is beyond rounds=2 → never active
    inert = FailureSpecification(
        stage="compression",
        type="aggressive_topk",
        active_rounds=(5, 5),
        parameters={"k_ratio": 0.5},
    )
    reference = _record_run(tmp_path, _config("reference", None))
    failure = _record_run(tmp_path, _config("failure", inert))

    report = validate_pair(reference.run_dir, failure.run_dir)

    assert report.status == "MATCHED_WITH_WARNINGS"
    assert all(report.checks.values())
    assert report.first_divergence_round is None
    assert report.first_divergence_stage is None
    assert "identical" in report.warnings[0]


def test_real_pipeline_array_tamper_finds_first_divergence(tmp_path):
    # real failure type made a no-op (lr * 1.0) so the run stays byte-identical;
    # the divergence below comes from the manual on-disk tamper only
    spec = FailureSpecification(
        stage="local",
        type="lr_misconfig",
        active_rounds=(1, 1),
        parameters={"lr_multiplier": 1.0, "affected_clients": ["client_0"]},
    )
    reference = _record_run(tmp_path, _config("reference", None))
    failure = _record_run(tmp_path, _config("failure", spec))
    path = next((failure.run_dir / "round_1" / "local").glob("*.npz"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["array_0000"][0] += 1.0
    np.savez(path, **arrays)

    report = validate_pair(reference.run_dir, failure.run_dir)

    assert report.status == "MATCHED"
    assert all(report.checks.values())
    assert (report.first_divergence_round, report.first_divergence_stage) == (
        1,
        "local",
    )
