import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_e0_smoke(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "e0"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "experiments" / "e0_replay_validation.py"),
            "--config",
            str(root / "configs" / "experiments" / "e0_smoke.yaml"),
            "--output",
            str(output),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["replay_level"] == "bitwise"
    assert len(report["configs"]) == 2
    assert {config["failure_specified"] for config in report["configs"]} == {
        False,
        True,
    }
    for config in report["configs"]:
        assert config["status"] == "PASS"
        assert config["replay_level"] == "bitwise"
        assert config["mismatched_boundaries"] == []
        assert all(boundary["match"] for boundary in config["boundary_agreement"])
        assert config["max_abs_sham_deviation"] == pytest.approx(0.0, abs=1e-12)
        assert all(sham["valid"] for sham in config["sham_results"])
        assert config["checkpoint_restore"]["rng_state_restored"]
        assert not config["checkpoint_restore"]["suffix_hashes_compared"]
    assert (output / "report.md").is_file()
