"""Build a co-author bundle: runnable code only, a run guide as its README.md.

Excluded: Plan.md, docs/, assets/, paper/, figures/ — nothing needed to run.

    python scripts/make_coauthor_bundle.py                # legacy: docs/COAUTHOR.md
    python scripts/make_coauthor_bundle.py --variant raf  # Italian guide, synthetic suite
    python scripts/make_coauthor_bundle.py --variant ko   # Korean guide, failure-type suite
    -> tmp/FALCON-<variant>-<YYYY-MM-DD>.zip
"""
import argparse
import datetime
import io
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

GUIDES = {
    "coauthor": "COAUTHOR.md",
    "raf": "COAUTHOR_RAF.md",
    "ko": "COAUTHOR_KO.md",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(GUIDES), default="coauthor")
    args = ap.parse_args()

    stamp = datetime.date.today().isoformat()
    out = REPO / "tmp" / f"FALCON-{args.variant}-{stamp}.zip"
    out.parent.mkdir(exist_ok=True)

    archive = subprocess.run(
        ["git", "archive", "--format=zip", "HEAD"], cwd=REPO, capture_output=True, check=True
    ).stdout
    guide = (REPO / "docs" / GUIDES[args.variant]).read_bytes()

    with zipfile.ZipFile(io.BytesIO(archive)) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        skip = ("README.md", "Plan.md")
        skip_dirs = ("docs/", "assets/", "paper/", "figures/")
        for item in src.infolist():
            if item.filename in skip or item.filename.startswith(skip_dirs):
                continue
            dst.writestr(item, src.read(item))
        dst.writestr("README.md", guide)
    print(f"[make_coauthor_bundle] -> {out}")


if __name__ == "__main__":
    main()
