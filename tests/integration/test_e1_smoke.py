import json
import subprocess
import sys
from pathlib import Path


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
