"""Bundle experiment outputs into tmp/Output_<YYYY-MM-DD_HH-MM-SS>.zip.

Run after experiments; send the zip back for analysis.

    python scripts/collect_output.py            # light: metrics/hashes/reports/configs
    python scripts/collect_output.py --full     # also raw recorded arrays (.npz) — large

Light mode collects:
- results/ and figures/ entirely;
- runs/**: metadata.json and every stage .json (metrics, hashes, ids) — no .npz tensors;
- configs/cases/*.yaml (what was actually run);
- environment snapshot (platform, python, installed packages).
"""
import argparse
import datetime
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _env_snapshot() -> str:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    ).stdout
    return json.dumps(
        {
            "platform": platform.platform(),
            "python": sys.version,
            "machine": platform.node(),
            "packages": freeze.splitlines(),
        },
        indent=2,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="include raw .npz tensors from runs/")
    args = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = REPO / "tmp"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"Output_{stamp}.zip"

    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("environment.json", _env_snapshot())
        for base, pattern in [
            ("results", "**/*"),
            ("figures", "**/*"),
            ("configs/cases", "*.yaml"),
        ]:
            root = REPO / base
            if root.exists():
                for p in sorted(root.glob(pattern)):
                    if p.is_file():
                        z.write(p, p.relative_to(REPO))
                        n += 1
        runs = REPO / "runs"
        if runs.exists():
            for p in sorted(runs.rglob("*")):
                if not p.is_file():
                    continue
                if p.suffix == ".npz" and not args.full:
                    continue
                z.write(p, p.relative_to(REPO))
                n += 1
    print(f"[collect_output] {n} files -> {out}")


if __name__ == "__main__":
    main()
