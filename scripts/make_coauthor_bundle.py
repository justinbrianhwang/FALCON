"""Build the co-author bundle: repo snapshot with docs/COAUTHOR.md as its README.md.

    python scripts/make_coauthor_bundle.py
    -> tmp/FALCON-coauthor-<YYYY-MM-DD>.zip
"""
import datetime
import io
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    stamp = datetime.date.today().isoformat()
    out = REPO / "tmp" / f"FALCON-coauthor-{stamp}.zip"
    out.parent.mkdir(exist_ok=True)

    archive = subprocess.run(
        ["git", "archive", "--format=zip", "HEAD"], cwd=REPO, capture_output=True, check=True
    ).stdout
    guide = (REPO / "docs" / "COAUTHOR.md").read_bytes()

    with zipfile.ZipFile(io.BytesIO(archive)) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            if item.filename == "README.md":
                continue  # replaced by the co-author guide below
            dst.writestr(item, src.read(item))
        dst.writestr("README.md", guide)
    print(f"[make_coauthor_bundle] -> {out}")


if __name__ == "__main__":
    main()
