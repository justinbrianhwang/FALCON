"""Torch-local stages for Tier 1 (Task T18): local SGD training + evaluation.

Torch is confined to this module and ``models.py`` (T18 design decision):
only local training and the evaluation forward pass touch torch; the stage
boundaries stay flat numpy arrays — float32 here (torch-native), float64 in
the synthetic path. The recorder hashes dtype+bytes, so per-tier determinism
is well-defined (CONTRACTS-adjacent note in ``falcon/schema/states.py``).

Determinism (T18 / CONTRACTS §5): CPU only, deterministic algorithms, a
single torch thread, and every random draw seeded from the named Rng streams
— a seed integer is drawn from the numpy stream and fed to a
``torch.Generator`` (model init lives in ``models.build_model``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from falcon.schema import ClientLocalState, OutcomeState

from .models import build_model, flatten, load_flat
from .stages import _sha256
from .synthetic_data import ClientData, EvalData

if TYPE_CHECKING:
    from falcon.replay.rng import Rng  # CONTRACTS §3; deferred to avoid import cycles
    from falcon.schema.config import DatasetConfig, LocalConfig, ModelConfig

torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)

_EVAL_BATCH = 2048  # eval forward-pass chunk size; chunking does not affect results


def local_train(
    model_params: np.ndarray,
    client_id: str,
    data: ClientData,
    round_id: int,
    cfg: "LocalConfig",
    rng: "Rng",
    *,
    model_cfg: "ModelConfig",
    dataset_cfg: "DatasetConfig",
) -> ClientLocalState:
    """Minibatch SGD on softmax cross-entropy, SmallCNN torch implementation.

    Mirrors the synthetic stage contract (CONTRACTS §1): the returned
    ``update`` is the flat delta ``trained - global`` (float32), NOT the
    trained params; ``loss_history`` holds the pre-update loss of each step.
    Minibatch indices are drawn WITH replacement, one batch of
    ``min(batch_size, n)`` per local step — the torch analog of the synthetic
    stage — from a torch generator seeded by stream
    ``client.<id>.round.<t>.dataloader``; the snapshot of that stream's state
    goes into ``rng_state`` as today. Plain SGD has no optimizer randomness,
    so the ``...optimizer`` stream is never drawn.
    """
    stream_name = f"client.{client_id}.round.{round_id}.dataloader"
    gen = rng.stream(stream_name)
    torch_gen = torch.Generator().manual_seed(int(gen.integers(0, 2**31 - 1)))

    model = build_model(model_cfg, dataset_cfg)  # init irrelevant: load_flat overwrites
    load_flat(model, model_params)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    x_all = torch.from_numpy(np.ascontiguousarray(data.x))
    y_all = torch.from_numpy(np.ascontiguousarray(data.y))
    n = int(x_all.shape[0])
    batch_size = min(cfg.batch_size, n)
    loss_history: list[float] = []
    for _ in range(cfg.local_steps):
        idx = torch.randint(0, n, (batch_size,), generator=torch_gen)
        optimizer.zero_grad()
        logits = model(x_all[idx])
        loss = F.cross_entropy(logits, y_all[idx])
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach()))
    update = flatten(model) - np.asarray(model_params, dtype=np.float32)
    return ClientLocalState(
        round_id=round_id,
        client_id=client_id,
        base_model_hash=_sha256(model_params),
        update=update.astype(np.float32, copy=False),
        num_examples=n,
        num_steps=cfg.local_steps,
        loss_history=loss_history,
        rng_state={stream_name: gen.bit_generator.state},
    )


def evaluate(
    model_params: np.ndarray,
    eval_data: EvalData,
    *,
    model_cfg: "ModelConfig",
    dataset_cfg: "DatasetConfig",
) -> OutcomeState:
    """Forward pass over the full eval split: accuracy, mean CE loss, per-class accuracy.

    ``round_id`` is a sentinel here; the runner stamps the real round id.
    """
    model = build_model(model_cfg, dataset_cfg)  # init irrelevant: load_flat overwrites
    load_flat(model, model_params)
    model.eval()

    x = torch.from_numpy(np.ascontiguousarray(eval_data.x))
    y = eval_data.y
    total = int(y.shape[0])
    preds: list[np.ndarray] = []
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, total, _EVAL_BATCH):
            logits = model(x[start : start + _EVAL_BATCH])
            loss_sum += float(
                F.cross_entropy(
                    logits, torch.from_numpy(y[start : start + _EVAL_BATCH]), reduction="sum"
                )
            )
            preds.append(torch.argmax(logits, dim=1).numpy())
    pred = np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)
    per_class = {
        str(c): {"accuracy": float((pred[y == c] == c).mean()) if np.any(y == c) else 0.0}
        for c in range(dataset_cfg.num_classes)
    }
    return OutcomeState(
        round_id=-1,
        model_hash=_sha256(model_params),
        metrics={
            "accuracy": float((pred == y).mean()) if total else 0.0,
            "loss": loss_sum / total if total else 0.0,
        },
        per_class=per_class,
    )
