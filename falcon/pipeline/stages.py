"""The five pipeline stage functions (CONTRACTS v0.1 §1).

Model: flat float64 param vector for multinomial logistic regression, layout
``[W (K, D) row-major | b (K,)]``, i.e. length ``K * (D + 1)`` including bias.
Pure numpy; all randomness comes from the named streams of the provided rng.
"""
from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

import numpy as np

from falcon.schema import (
    AggregationConfig,
    AggregationState,
    ClientLocalState,
    CompressionConfig,
    CompressionState,
    LocalConfig,
    OutcomeState,
    SelectionConfig,
    SelectionState,
)

from .synthetic_data import ClientData, EvalData

if TYPE_CHECKING:
    from falcon.replay.rng import Rng  # CONTRACTS §3; deferred to avoid import cycles


def _sha256(params: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(params, dtype=np.float64).tobytes()).hexdigest()


def _split_params(params: np.ndarray, num_features: int) -> tuple[np.ndarray, np.ndarray]:
    num_classes = params.shape[0] // (num_features + 1)
    w = params[: num_classes * num_features].reshape(num_classes, num_features)
    b = params[num_classes * num_features :]
    return w, b


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def _loss_and_grad(
    params: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[float, np.ndarray]:
    """Mean softmax cross-entropy and its gradient w.r.t. the flat params."""
    w, b = _split_params(params, x.shape[1])
    log_probs = _log_softmax(x @ w.T + b)
    n = x.shape[0]
    loss = -log_probs[np.arange(n), y].mean()
    dlogits = np.exp(log_probs)
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n
    grad = np.concatenate([(dlogits.T @ x).ravel(), dlogits.sum(axis=0)])
    return float(loss), grad.astype(np.float64)


def _predict(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    w, b = _split_params(params, x.shape[1])
    return np.argmax(x @ w.T + b, axis=1)


def init_params(num_features: int, num_classes: int, rng) -> np.ndarray:
    """Small random init from the ``global_init`` stream (CONTRACTS §3)."""
    gen = rng.stream("global_init")
    return gen.normal(0.0, 0.01, size=num_classes * (num_features + 1)).astype(np.float64)


def select_clients(
    pool: list[str], round_id: int, cfg: SelectionConfig, rng: "Rng"
) -> SelectionState:
    """Uniform sampling without replacement, stream ``client_selection``.

    Undersubscription is allowed (T4-F finding 7): when the candidate pool is
    smaller than ``cfg.clients_per_round`` — e.g. a selection failure excluded
    most clients — the round selects the whole pool instead of crashing, and
    the recorded state shows the shortfall (``len(selected_ids) <
    clients_per_round``, inclusion probability 1.0).
    """
    gen = rng.stream("client_selection")
    n_select = min(cfg.clients_per_round, len(pool))
    idx = gen.choice(len(pool), size=n_select, replace=False)
    selected = sorted(pool[int(i)] for i in idx)
    inclusion_prob = n_select / len(pool) if pool else 0.0
    return SelectionState(
        round_id=round_id,
        candidate_ids=list(pool),
        selected_ids=selected,
        sampling_probs={cid: inclusion_prob for cid in pool},
        rng_state={"client_selection": gen.bit_generator.state},
    )


def local_train(
    model_params: np.ndarray,
    client_id: str,
    data: ClientData,
    round_id: int,
    cfg: LocalConfig,
    rng: "Rng",
) -> ClientLocalState:
    """Plain minibatch SGD on softmax cross-entropy.

    Minibatches are drawn from stream ``client.<id>.round.<t>.dataloader``
    (CONTRACTS §3 v0.2): keying the stream by (client, round) makes a client's
    round-t draws independent of which rounds it participated in earlier, so a
    selection intervention cannot advance or freeze its stream. The returned
    ``update`` is the delta ``trained - global``, NOT the trained params.
    """
    gen = rng.stream(f"client.{client_id}.round.{round_id}.dataloader")
    params = model_params.astype(np.float64, copy=True)
    n = data.x.shape[0]
    batch_size = min(cfg.batch_size, n)
    loss_history: list[float] = []
    for _ in range(cfg.local_steps):
        idx = gen.integers(0, n, size=batch_size)
        loss, grad = _loss_and_grad(params, data.x[idx], data.y[idx])
        params -= cfg.lr * grad
        loss_history.append(loss)
    return ClientLocalState(
        round_id=round_id,
        client_id=client_id,
        base_model_hash=_sha256(model_params),
        update=params - model_params,
        num_examples=n,
        num_steps=cfg.local_steps,
        loss_history=loss_history,
        rng_state={
            f"client.{client_id}.round.{round_id}.dataloader": gen.bit_generator.state
        },
    )


def _topk(update: np.ndarray, k: int) -> np.ndarray:
    """Keep the ``k`` largest-|value| coordinates of ``update``, zero the rest.

    Deterministic tie-break: among tied magnitudes the LARGER index is kept —
    a stable ascending argsort orders ties by increasing index, so the kept
    suffix of the ordering holds the larger indices. Exact float64, no
    randomness.
    """
    order = np.argsort(np.abs(update), kind="stable")
    out = update.copy()
    out[order[: update.shape[0] - k]] = 0.0
    return out


def compress(local_state: ClientLocalState, cfg: CompressionConfig, rng: "Rng") -> CompressionState:
    """Compress one client's update for transmission.

    ``identity`` (default) copies the update and round-trips exactly.
    ``topk`` keeps the top ``ceil(k_ratio * n)`` coordinates by magnitude
    (``k_ratio`` in ``cfg.parameters``, in (0, 1]) and zeroes the rest, with
    the larger-index tie-break of ``_topk``; ``bytes_transmitted`` is
    ``nnz * 8 + nnz * 4`` (8 bytes per value, 4 per coordinate index) where
    ``nnz`` counts the nonzeros of the sparsified update actually sent.
    """
    update = np.array(local_state.update, dtype=np.float64, copy=True)
    uncompressed_hash = _sha256(update)
    if cfg.kind == "identity":
        compressed = update
        bytes_transmitted = update.nbytes
    elif cfg.kind == "topk":
        k_ratio = float(cfg.parameters["k_ratio"])
        if not math.isfinite(k_ratio) or not 0.0 < k_ratio <= 1.0:
            raise ValueError(f"topk k_ratio must be in (0, 1], got {k_ratio}")
        k = min(update.shape[0], math.ceil(k_ratio * update.shape[0]))
        compressed = _topk(update, k)
        nnz = int(np.count_nonzero(compressed))
        bytes_transmitted = nnz * 8 + nnz * 4
    else:
        raise NotImplementedError(f"compression kind {cfg.kind!r} not implemented yet")
    return CompressionState(
        round_id=local_state.round_id,
        client_id=local_state.client_id,
        uncompressed_hash=uncompressed_hash,
        update=compressed,
        compression_params=dict(cfg.parameters),
        bytes_transmitted=bytes_transmitted,
    )


def aggregate(
    compressed: list[CompressionState],
    weights: dict[str, float],
    cfg: AggregationConfig,
    rng: "Rng",
) -> AggregationState:
    """``weighted_mean`` by the weights arg (num_examples) or ``uniform_mean``."""
    if not compressed:
        raise ValueError("aggregate() needs at least one CompressionState")
    ids = [c.client_id for c in compressed]
    if len(set(ids)) != len(ids):
        raise ValueError(f"aggregate() got duplicate client ids: {ids}")
    updates = np.stack([c.update for c in compressed])
    if cfg.rule == "weighted_mean":
        missing = sorted(set(ids) - set(weights))
        extra = sorted(set(weights) - set(ids))
        if missing or extra:
            raise ValueError(
                "weights must cover exactly the received clients "
                f"(missing={missing}, extra={extra})"
            )
        raw = np.array([weights[cid] for cid in ids], dtype=np.float64)
        if not np.all(np.isfinite(raw)):
            raise ValueError(f"weights must be finite, got {raw.tolist()}")
        if np.any(raw < 0.0):
            raise ValueError(f"weights must be nonnegative, got {raw.tolist()}")
        total = float(raw.sum())
        if total <= 0.0:
            raise ValueError("total weight must be positive")
        coeffs = raw / total
    elif cfg.rule == "uniform_mean":
        coeffs = np.full(len(ids), 1.0 / len(ids), dtype=np.float64)
    else:
        raise NotImplementedError(f"aggregation rule {cfg.rule!r} not implemented yet")
    return AggregationState(
        round_id=compressed[0].round_id,
        received_ids=ids,
        accepted_ids=list(ids),
        rejected_ids=[],
        weights={cid: float(c) for cid, c in zip(ids, coeffs)},
        aggregate=(coeffs[:, None] * updates).sum(axis=0),
    )


def evaluate(model_params: np.ndarray, eval_data: EvalData) -> OutcomeState:
    """Accuracy + mean log-loss; per-class accuracy into ``per_class``.

    ``round_id`` is a sentinel here; the runner stamps the real round id.
    """
    w, b = _split_params(model_params, eval_data.x.shape[1])
    log_probs = _log_softmax(eval_data.x @ w.T + b)
    y = eval_data.y
    loss = float(-log_probs[np.arange(y.shape[0]), y].mean())
    pred = np.argmax(log_probs, axis=1)
    per_class = {
        str(c): {"accuracy": float((pred[y == c] == c).mean()) if np.any(y == c) else 0.0}
        for c in range(w.shape[0])
    }
    return OutcomeState(
        round_id=-1,
        model_hash=_sha256(model_params),
        metrics={"accuracy": float((pred == y).mean()), "loss": loss},
        per_class=per_class,
    )
