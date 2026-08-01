"""Deterministic synthetic FL pipeline (Task T2, contract v0.1)."""
from .runner import run
from .stages import (
    aggregate,
    compress,
    evaluate,
    init_params,
    local_train,
    select_clients,
)
from .synthetic_data import ClientData, EvalData, make_eval_data, make_partition

__all__ = [
    "ClientData",
    "EvalData",
    "aggregate",
    "compress",
    "evaluate",
    "init_params",
    "local_train",
    "make_eval_data",
    "make_partition",
    "run",
    "select_clients",
]
