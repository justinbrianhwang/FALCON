import json

import pytest

from falcon.pipeline.runner import run
from falcon.recorder import Recorder
from falcon.replay import Rng
from falcon.reporting import analyze_pair, render_markdown
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


def _config(run_id, failure=None):
    return RunConfig(
        run_id=run_id,
        seed=42,
        rounds=5,
        dataset=DatasetConfig(
            num_clients=10,
            num_features=20,
            samples_per_client=100,
        ),
        selection=SelectionConfig(clients_per_round=5),
        local=LocalConfig(lr=0.05, local_steps=3, batch_size=32),
        compression=CompressionConfig(kind="identity"),
        aggregation=AggregationConfig(rule="weighted_mean"),
        failure=failure,
    )


def _record(root, cfg):
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


def test_analyze_pair_end_to_end(tmp_path):
    reference = _record(tmp_path, _config("reference"))
    selected = reference.load(2, "selection").selected_ids
    ground_truth = FailureSpecification(
        stage="compression",
        type="aggressive_topk",
        active_rounds=(2, 2),
        parameters={"k_ratio": 0.02, "affected_clients": selected[:4]},
    )
    failure = _record(tmp_path, _config("failure", ground_truth))

    report, interventions = analyze_pair(
        tmp_path,
        "reference",
        "failure",
        metric="loss",
        higher_is_better=False,
        min_gap=0.005,
        sham_tolerance=1e-9,
    )

    assert report.origin_ranking[0] == ground_truth.stage
    assert len(interventions) == 12
    assert all(result.valid for result in interventions)
    assert all(
        result.outcome_metrics["sham_deviation_loss"] == pytest.approx(0.0)
        for result in interventions
        if result.spec.mode == "sham"
    )

    markdown = render_markdown(report, interventions, ground_truth=ground_truth)
    for section in (
        "## Pair validity",
        "## Terminal failure summary",
        "## Measured evidence — intervention effects",
        "## Inferred origin ranking and roles",
        "## Counterfactual explanation",
        "## Warnings and assumptions",
        "## Ground truth (benchmark)",
    ):
        assert section in markdown
    assert "Restoring compression closes 100.0% of the gap" in markdown
    assert "injecting reproduces 100.0%." in markdown

    metadata_path = failure.run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["seed"] += 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    invalid_report, invalid_interventions = analyze_pair(
        tmp_path,
        "reference",
        "failure",
        metric="loss",
        higher_is_better=False,
        min_gap=0.005,
        sham_tolerance=1e-9,
    )
    assert invalid_report.pair.status == "INVALID_PAIR"
    assert "INVALID_PAIR" in invalid_report.notes
    assert invalid_interventions == []
