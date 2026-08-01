"""On-disk recorder for FALCON stage-boundary states."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from falcon.schema import (
    STAGES,
    AggregationState,
    ClientLocalState,
    CompressionState,
    OutcomeState,
    RunMetadata,
    SelectionState,
)

from .hashing import _ARRAY_MARKER, _split_model, hash_model

_MODEL_TYPES: dict[str, type[BaseModel]] = {
    model.__name__: model
    for model in (
        SelectionState,
        ClientLocalState,
        CompressionState,
        AggregationState,
        OutcomeState,
    )
}
_PER_CLIENT_STAGES = {"local", "compression"}


def _safe_component(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or any(character in value for character in "/\\:")
    ):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def _sidecar(path: Path, suffix: str) -> Path:
    return path.parent / f"{path.name}{suffix}"


class Recorder:
    """Persist metadata and stage states under ``root_dir/runs/run_id``."""

    def __init__(self, root_dir: Path, run_id: str):
        self.run_id = _safe_component(run_id, "run_id")
        self.run_dir = Path(root_dir) / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, meta: RunMetadata) -> None:
        if not isinstance(meta, RunMetadata):
            raise TypeError("meta must be RunMetadata")
        self._write_json(
            self.run_dir / "metadata.json", meta.model_dump(mode="json")
        )

    def record(
        self, round_id: int, stage: str, state: BaseModel | list[BaseModel]
    ) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage!r}")
        round_dir = self.run_dir / f"round_{round_id}"
        round_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(state, list):
            if stage not in _PER_CLIENT_STAGES:
                raise TypeError(f"stage {stage!r} does not accept a state list")
            _sidecar(round_dir / stage, ".json").unlink(missing_ok=True)
            _sidecar(round_dir / stage, ".npz").unlink(missing_ok=True)
            stage_dir = round_dir / stage
            stage_dir.mkdir(exist_ok=True)
            for old_file in (*stage_dir.glob("*.json"), *stage_dir.glob("*.npz")):
                old_file.unlink()

            seen: set[str] = set()
            for index, model in enumerate(state):
                client_id = _safe_component(
                    getattr(model, "client_id", None), "client_id"
                )
                if client_id in seen:
                    raise ValueError(f"duplicate client_id: {client_id!r}")
                seen.add(client_id)
                self._write_model(stage_dir / client_id, model, index=index)
            return

        if not isinstance(state, BaseModel):
            raise TypeError("state must be a BaseModel or list[BaseModel]")
        path = round_dir / stage
        if path.is_dir():
            for old_file in (*path.glob("*.json"), *path.glob("*.npz")):
                old_file.unlink()
            path.rmdir()
        self._write_model(path, state)

    def load(self, round_id: int, stage: str) -> BaseModel | list[BaseModel]:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage!r}")
        path = self.run_dir / f"round_{round_id}" / stage
        if path.is_dir():
            indexed = [
                self._read_model(file.with_suffix(""))
                for file in path.glob("*.json")
            ]
            indexed.sort(key=lambda pair: pair[0])
            return [model for _, model in indexed]
        _, model = self._read_model(path)
        return model

    def stage_hashes(self) -> dict[tuple[int, str], str]:
        """Return one stable hash for each recorded round/stage boundary."""
        hashes: dict[tuple[int, str], str] = {}
        round_dirs: list[tuple[int, Path]] = []
        for path in self.run_dir.glob("round_*"):
            try:
                round_dirs.append((int(path.name.removeprefix("round_")), path))
            except ValueError:
                continue

        for round_id, round_dir in sorted(round_dirs):
            for stage in STAGES:
                path = round_dir / stage
                if not _sidecar(path, ".json").exists() and not path.is_dir():
                    continue
                state = self.load(round_id, stage)
                if isinstance(state, list):
                    content_hashes = [model.content_hash for model in state]
                    encoded = json.dumps(
                        content_hashes, separators=(",", ":")
                    ).encode("ascii")
                    hashes[(round_id, stage)] = sha256(encoded).hexdigest()
                else:
                    hashes[(round_id, stage)] = state.content_hash or hash_model(state)
        return hashes

    def _write_model(
        self, path: Path, model: BaseModel, index: int | None = None
    ) -> None:
        if not isinstance(model, BaseModel):
            raise TypeError("state list items must be BaseModel instances")
        if "content_hash" not in type(model).model_fields:
            raise TypeError("recorded states must define content_hash")

        content_hash = hash_model(model)
        model.content_hash = content_hash
        data, arrays = _split_model(model)
        data["content_hash"] = content_hash
        if index is not None:
            data["__index__"] = index
        self._write_json(_sidecar(path, ".json"), data)

        npz_path = _sidecar(path, ".npz")
        if arrays:
            np.savez(npz_path, **arrays)
        else:
            npz_path.unlink(missing_ok=True)

    def _read_model(self, path: Path) -> tuple[int, BaseModel]:
        json_path = _sidecar(path, ".json")
        if not json_path.exists():
            raise FileNotFoundError(json_path)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        try:
            model_name = data.pop("__model__")
            stored_hash = data["content_hash"]
        except KeyError as exc:
            raise ValueError(f"invalid recorded state: {json_path}") from exc
        index = data.pop("__index__", 0)

        arrays: dict[str, np.ndarray] = {}
        npz_path = _sidecar(path, ".npz")
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
        restored = self._restore_value(data, arrays)

        try:
            model_type = _MODEL_TYPES[model_name]
        except KeyError as exc:
            raise ValueError(f"unknown recorded model type: {model_name!r}") from exc
        model = model_type.model_validate(restored)
        if hash_model(model) != stored_hash:
            raise ValueError(f"content hash mismatch: {json_path}")
        return index, model

    @classmethod
    def _restore_value(cls, value: Any, arrays: dict[str, np.ndarray]) -> Any:
        if isinstance(value, dict):
            if _ARRAY_MARKER in value:
                key = value[_ARRAY_MARKER]
                try:
                    array = arrays[key]
                except KeyError as exc:
                    raise ValueError(f"missing recorded array: {key}") from exc
                if (
                    array.dtype.str != value["dtype"]
                    or list(array.shape) != value["shape"]
                ):
                    raise ValueError(f"recorded array metadata mismatch: {key}")
                return array
            return {
                key: cls._restore_value(item, arrays)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._restore_value(item, arrays) for item in value]
        return value

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.write_text(
            json.dumps(
                data,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
