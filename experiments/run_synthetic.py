"""Run the synthetic reference case and print per-round accuracy.

Usage (from repo root):  python experiments/run_synthetic.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from falcon.pipeline.runner import run  # noqa: E402
from falcon.schema import RunConfig  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "cases" / "synthetic_reference.yaml"


def _make_rng(seed: int):
    """Use falcon.replay.rng.Rng when available (T3); otherwise a minimal
    stand-in with the same stream(name) -> Generator interface (CONTRACTS §3)."""
    try:
        from falcon.replay.rng import Rng

        return Rng(seed)
    except ImportError:
        pass

    class _StubRng:
        def __init__(self, root_seed: int):
            self._seed = root_seed
            self._streams: dict[str, np.random.Generator] = {}

        def stream(self, name: str) -> np.random.Generator:
            if name not in self._streams:
                key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
                self._streams[name] = np.random.default_rng(
                    np.random.SeedSequence([self._seed, key])
                )
            return self._streams[name]

    return _StubRng(seed)


def main() -> None:
    cfg = RunConfig(**yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")))
    outcomes = run(cfg, rng=_make_rng(cfg.seed))
    for outcome in outcomes:
        print(
            f"round {outcome.round_id:2d}: "
            f"accuracy={outcome.metrics['accuracy']:.4f} "
            f"loss={outcome.metrics['loss']:.4f}"
        )


if __name__ == "__main__":
    main()
