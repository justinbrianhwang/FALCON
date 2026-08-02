import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from experiments import e1_equivalence
from falcon.schema import RunConfig


def test_e1_smoke(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "e1"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "experiments" / "e1_equivalence.py"),
            "--config",
            str(root / "configs" / "experiments" / "e1_smoke.yaml"),
            "--output",
            str(output_root),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    output = output_root / "e1_smoke"
    expected = {
        "matched_configs.yaml",
        "severity_traces.json",
        "predictions.json",
        "summary.json",
        "summary.md",
    }
    assert expected <= {path.name for path in output.iterdir()}

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert len(summary["cases"]) == 2
    assert {case["ground_truth"] for case in summary["cases"]} == {
        "selection",
        "compression",
    }
    for case in summary["cases"]:
        assert abs(case["gap"] - summary["target_gap"]) <= summary["gap_tolerance"]
        assert case["pair_status"] != "INVALID_PAIR"
        assert case["predictions"]["falcon"] in {
            case["ground_truth"],
            "unresolved",
        }

    traces = json.loads(
        (output / "severity_traces.json").read_text(encoding="utf-8")
    )
    assert set(traces["failures"]) == {
        "selection_minority_exclusion",
        "compression_aggressive_topk",
    }
    assert all(
        any(step["matched"] for step in trace)
        for trace in traces["failures"].values()
    )


def test_accuracy_matching_accepts_nearest_quantized_step(monkeypatch):
    reference = RunConfig.model_validate(
        {
            "run_id": "reference",
            "seed": 1,
            "rounds": 1,
            "dataset": {"num_clients": 2, "num_features": 2},
            "selection": {"clients_per_round": 1},
            "local": {"lr": 0.1, "local_steps": 1, "batch_size": 1},
        }
    )
    calls = []

    def quantized_run(cfg, *, rng):
        value = cfg.failure.parameters["lr_multiplier"]
        calls.append(value)
        accuracy = 1.0 - round(3 * value) / 500
        return [SimpleNamespace(metrics={"accuracy": accuracy})]

    monkeypatch.setattr(e1_equivalence, "run", quantized_run)
    _, trace = e1_equivalence._bisect_match(
        reference,
        1.0,
        {
            "id": "quantized_accuracy",
            "stage": "local",
            "type": "lr_misconfig",
            "active_rounds": [0, 0],
            "parameters": {"fraction": 1.0},
            "severity": {
                "parameter": "lr_multiplier",
                "bounds": [0.0, 1.0],
                "higher_is_more_severe": True,
            },
        },
        metric="accuracy",
        higher_is_better=True,
        target_gap=0.003,
        gap_tolerance=1e-6,
        max_iterations=8,
    )

    assert calls == [0.0, 1.0, 0.5]
    assert abs(trace[-1]["gap"] - 0.004) < 1e-12
    assert trace[-1]["match_tolerance"] == 0.001
    assert trace[-1]["matched"] is True
