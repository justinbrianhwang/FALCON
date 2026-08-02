"""Compare a co-author E0 report in an Output zip with the local golden hashes."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "fixtures" / "golden_stage_hashes.json"
REPORT = "results/coauthor/e0_crossmachine/report.json"


def _load_report(archive: zipfile.ZipFile) -> dict[str, Any]:
    matches = [name for name in archive.namelist() if name.replace("\\", "/").endswith(REPORT)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {REPORT} in the zip, found {len(matches)}")
    return json.loads(archive.read(matches[0]))


def compare(output_zip: Path, golden_path: Path = GOLDEN) -> bool:
    golden = json.loads(golden_path.read_text(encoding="utf-8"))["stage_hashes"]
    with zipfile.ZipFile(output_zip) as archive:
        report = _load_report(archive)
    configs = report.get("configs")
    if not isinstance(configs, list) or len(configs) != 1:
        raise ValueError("E0 cross-machine report must contain exactly one config")
    boundaries = configs[0].get("boundary_agreement")
    if not isinstance(boundaries, list):
        raise ValueError("E0 cross-machine report has no boundary_agreement list")
    remote = {
        f"{item['round']}/{item['stage']}": (item.get("first_hash"), item.get("second_hash"))
        for item in boundaries
    }

    portable = set(remote) == set(golden)
    for boundary in sorted(set(golden) | set(remote), key=lambda key: (int(key.split("/", 1)[0]), key)):
        expected = golden.get(boundary)
        first, second = remote.get(boundary, (None, None))
        match = expected is not None and first == expected and second == expected
        portable &= match
        print(
            f"{boundary}: {'MATCH' if match else 'MISMATCH'} "
            f"local={expected or 'MISSING'} "
            f"remote_first={first or 'MISSING'} remote_second={second or 'MISSING'}"
        )
    print(f"VERDICT: {'bitwise-portable' if portable else 'machine-dependent'}")
    return portable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_zip", type=Path)
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    args = parser.parse_args(argv)
    try:
        return 0 if compare(args.output_zip, args.golden) else 1
    except (KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"compare_crossmachine refused input: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
