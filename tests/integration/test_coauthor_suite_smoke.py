import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


def test_coauthor_suite_smoke(tmp_path):
    root = Path(__file__).resolve().parents[2]
    configs = root / "configs" / "experiments" / "coauthor"

    seeds = yaml.safe_load((configs / "e1_seeds.yaml").read_text(encoding="utf-8"))
    seeds["seeds"] = [101]
    seeds_path = tmp_path / "e1_seeds.yaml"
    seeds_path.write_text(yaml.safe_dump(seeds, sort_keys=False), encoding="utf-8")

    heterogeneity = yaml.safe_load(
        (configs / "e1_heterogeneity.yaml").read_text(encoding="utf-8")
    )
    heterogeneity["heterogeneity_levels"] = [0.5]
    heterogeneity_path = tmp_path / "e1_heterogeneity.yaml"
    heterogeneity_path.write_text(
        yaml.safe_dump(heterogeneity, sort_keys=False), encoding="utf-8"
    )

    output = tmp_path / "results" / "coauthor"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "experiments" / "run_coauthor_suite.py"),
            "--e1-seeds-config",
            str(seeds_path),
            "--e1-heterogeneity-config",
            str(heterogeneity_path),
            "--output",
            str(output),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert [item["name"] for item in summary["experiments"]] == [
        "e0_crossmachine",
        "e1_seeds",
        "e1_heterogeneity",
    ]
    assert all(item["status"] == "PASS" for item in summary["experiments"])

    match = re.search(r"^SEND THIS FILE: (.+\.zip)$", completed.stdout, re.MULTILINE)
    assert match, completed.stdout
    assert Path(match.group(1)).is_file()
