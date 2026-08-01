"""Canonical hashing helpers for recorded Pydantic models."""

from __future__ import annotations

import json
from enum import Enum
from hashlib import sha256
from math import isfinite, isnan
from typing import Any, get_args

import numpy as np
from pydantic import BaseModel

_ARRAY_MARKER = "__falcon_array__"
_FLOAT_MARKER = "__falcon_float__"


def hash_array(a: np.ndarray) -> str:
    """Hash an array's dtype, shape, and C-order raw bytes."""
    if not isinstance(a, np.ndarray):
        raise TypeError("hash_array expects a numpy.ndarray")
    return sha256(
        a.dtype.str.encode() + str(a.shape).encode() + a.tobytes(order="C")
    ).hexdigest()


def _field_error(path: tuple[str, ...], detail: str) -> ValueError:
    field = path[0] if path else "<root>"
    return ValueError(f"field {field!r} contains {detail}")


def _contains_any(annotation: Any) -> bool:
    return annotation is Any or any(
        _contains_any(argument) for argument in get_args(annotation)
    )


def _validate_json_native(value: Any, path: tuple[str, ...]) -> None:
    if isinstance(value, Enum):
        raise _field_error(path, f"non-JSON-native Enum {value!r}")
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise _field_error(path, "a non-string dictionary key")
        for marker in (_ARRAY_MARKER, _FLOAT_MARKER):
            if marker in value:
                raise _field_error(path, f"reserved key {marker!r}")
        for item in value.values():
            _validate_json_native(item, path)
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_native(item, path)
        return
    if isinstance(value, tuple):
        raise _field_error(path, "a non-JSON-native tuple")
    if isinstance(value, np.generic):
        raise _field_error(path, f"non-JSON-native {type(value).__name__}")
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    raise _field_error(path, f"non-JSON-native {type(value).__name__}")


def _validate_any_fields(model: BaseModel) -> None:
    for name, field in type(model).model_fields.items():
        value = getattr(model, name)
        if _contains_any(field.annotation):
            _validate_json_native(value, (name,))
        elif isinstance(value, BaseModel):
            _validate_any_fields(value)


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
        if any(not isinstance(key, str) for key in value):
            raise _field_error(path, "a non-string dictionary key")
        if _ARRAY_MARKER in value:
            raise _field_error(path, f"reserved key {_ARRAY_MARKER!r}")
        if _FLOAT_MARKER in value:
            raise _field_error(path, f"reserved key {_FLOAT_MARKER!r}")
        return {
            key: _split_value(item, arrays, path + (key,))
            for key, item in sorted(value.items())
        }
    if isinstance(value, list):
        return [
            _split_value(item, arrays, path + (str(index),))
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
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
    _validate_any_fields(model)
    arrays: dict[str, np.ndarray] = {}
    data = _split_value(model, arrays, ())
    data["__model__"] = type(model).__name__
    return data, arrays


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _encode_non_finite(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _encode_non_finite(value: Any) -> Any:
    if isinstance(value, float) and not isfinite(value):
        encoded = (
            "NaN"
            if isnan(value)
            else "Infinity"
            if value > 0
            else "-Infinity"
        )
        return {_FLOAT_MARKER: encoded}
    if isinstance(value, dict):
        return {key: _encode_non_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_non_finite(item) for item in value]
    return value


def hash_model(m: BaseModel) -> str:
    """Hash canonical model data and its arrays, ignoring ``content_hash``."""
    data, arrays = _split_model(m)
    digest = sha256(_canonical_json(data))
    for key in sorted(arrays):
        digest.update(arrays[key].tobytes(order="C"))
    return digest.hexdigest()
