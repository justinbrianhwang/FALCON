import csv
import subprocess
import sys
from pathlib import Path

import yaml


def test_main_matrix_smoke(tmp_path):
    root = Path(__file__).resolve().parents[2]
    smoke = yaml.safe_load(
        (root / "configs" / "experiments" / "e1_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    matched_config = tmp_path / "matched.yaml"
    matched_config.write_text(
        yaml.safe_dump({"experiment": smoke}, sort_keys=False), encoding="utf-8"
    )

    excluded = dict(smoke)
    excluded["case_id"] = "excluded_smoke"
    excluded["target_gap"] = 10.0
    excluded_config = tmp_path / "excluded.yaml"
    excluded_config.write_text(
        yaml.safe_dump({"experiment": excluded}, sort_keys=False), encoding="utf-8"
    )

    matrix = {
        "experiments": [
            {"name": "matched", "config": "matched.yaml", "seeds": [42]},
            {"name": "excluded", "config": "excluded.yaml", "seeds": [42]},
        ]
    }
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
    output = tmp_path / "results" / "main_matrix"

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "experiments" / "run_main_matrix.py"),
            "--matrix",
            str(matrix_path),
            "--output",
            str(output),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    with (output / "table1.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert {row["ground_truth"] for row in rows} == {"selection", "compression"}
    assert all(row["case"].startswith("matched/") for row in rows)

    markdown = (output / "table1.md").read_text(encoding="utf-8")
    assert "| FALCON | 2/2 |" in markdown
    assert (
        "| excluded | 42 | EXCLUDED | failure 'selection_minority_exclusion' cannot match"
        in markdown
    )
    assert "Matched cases: 2; exclusions: 1" in completed.stdout
