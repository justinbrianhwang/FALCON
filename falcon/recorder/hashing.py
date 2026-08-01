"""Canonical hashing helpers for recorded Pydantic models."""

from __future__ import annotations

import json
from enum import Enum
from hashlib import sha256
from typing import Any

import numpy as np
from pydantic import BaseModel

_ARRAY_MARKER = "__falcon_array__"


def hash_array(a: np.ndarray) -> str:
    """Hash an array's values as C-order raw bytes."""
    if not isinstance(a, np.ndarray):
        raise TypeError("hash_array expects a numpy.ndarray")
    return sha256(a.tobytes(order="C")).hexdigest()


def _split_value(
    value: Any, arrays: dict[str, np.ndarray], path: tuple[str, ...]
) -> Any:
    if isinstance(value, np.ndarray):
        key = f"array_{len(arrays):04d}"
        arrays[key] = value
        return {
            _ARRAY_MARKER: key,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "path": list(path),
        }
    if isinstance(value, BaseModel):
        return {
            name: _split_value(getattr(value, name), arrays, path + (name,))
            for name in type(value).model_fields
            if name != "content_hash"
        }
    if isinstance(value, dict):
        return {
            key: _split_value(item, arrays, path + (str(key),))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [
            _split_value(item, arrays, path + (str(index),))
            for index, item in enumerate(value)
        ]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _split_model(model: BaseModel) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not isinstance(model, BaseModel):
        raise TypeError("expected a pydantic BaseModel")
    arrays: dict[str, np.ndarray] = {}
    data = _split_value(model, arrays, ())
    data["__model__"] = type(model).__name__
    return data, arrays


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def hash_model(m: BaseModel) -> str:
    """Hash canonical model data and its arrays, ignoring ``content_hash``."""
    data, arrays = _split_model(m)
    digest = sha256(_canonical_json(data))
    for key in sorted(arrays):
        digest.update(arrays[key].tobytes(order="C"))
    return digest.hexdigest()
