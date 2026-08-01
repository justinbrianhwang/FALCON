"""Deterministic registry of named NumPy random streams."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from struct import unpack
from typing import Any

import numpy as np


class Rng:
    """Create independent, order-independent random streams from one seed.

    Standard names are ``global_init``, ``client_selection``,
    ``client.<id>.round.<t>.dataloader``, ``client.<id>.round.<t>.optimizer``,
    ``compression.<id>``, ``aggregation``, and ``evaluation``. Failure
    injectors use their own ``failure.<stage>`` stream. Per-client streams are
    keyed by (client, round) so a client's draws are independent of its
    participation history (CONTRACTS v0.2).
    """

    def __init__(self, root_seed: int):
        self.root_seed = root_seed
        self._streams: dict[str, np.random.Generator] = {}

    @staticmethod
    def _spawn_key(name: str) -> tuple[int, ...]:
        if not isinstance(name, str) or not name:
            raise ValueError("stream name must be a non-empty string")
        return unpack(">8I", sha256(name.encode("utf-8")).digest())

    def stream(self, name: str) -> np.random.Generator:
        """Return the persistent generator associated with *name*."""
        if name not in self._streams:
            seed = np.random.SeedSequence(
                self.root_seed, spawn_key=self._spawn_key(name)
            )
            self._streams[name] = np.random.default_rng(seed)
        return self._streams[name]

    def state_dict(self) -> dict[str, Any]:
        """Return an independent snapshot of every stream's state."""
        return {
            "root_seed": self.root_seed,
            "streams": {
                name: deepcopy(generator.bit_generator.state)
                for name, generator in sorted(self._streams.items())
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Replace this registry with a snapshot produced by ``state_dict``.

        Previously returned generators are stale after this call; re-fetch every
        stream from this registry before drawing more values.
        """
        try:
            root_seed = state["root_seed"]
            stream_states = state["streams"]
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid RNG state dictionary") from exc
        if not isinstance(stream_states, dict):
            raise ValueError("invalid RNG stream states")

        self.root_seed = root_seed
        self._streams.clear()
        for name, generator_state in stream_states.items():
            generator = self.stream(name)
            generator.bit_generator.state = deepcopy(generator_state)
