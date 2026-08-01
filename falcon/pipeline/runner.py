"""Round loop: select -> local -> compress -> aggregate -> apply -> evaluate."""
from __future__ import annotations

from typing import Callable

from falcon.failures import build_injector
from falcon.schema import OutcomeState, RunConfig

from .stages import aggregate, compress, evaluate, init_params, local_train, select_clients
from .synthetic_data import make_eval_data, make_partition


def _stage(name: str, fn: Callable):
    # T4 FAILURE HOOK: when cfg.failure is set, the FailureInjector built in
    # run() transforms this stage's inputs immediately before fn() runs
    # (candidate pool, per-client local/compression configs, aggregation
    # weights). Keep this the only place stages are invoked from.
    return fn()


def _record(recorder, round_id: int, stage: str, state) -> None:
    if recorder is not None:
        recorder.record(round_id, stage, state)


def run(cfg: RunConfig, recorder=None, rng=None) -> list[OutcomeState]:
    """Run ``cfg.rounds`` federated rounds and return per-round outcomes.

    ``recorder`` is duck-typed: any object with
    ``record(round_id, stage, state)``. ``rng`` must provide the named-stream
    interface of CONTRACTS §3; when omitted, ``falcon.replay.rng.Rng`` (T3)
    is imported lazily.
    """
    if rng is None:
        from falcon.replay.rng import Rng  # provided by T3 (Codex); CONTRACTS §3

        rng = Rng(cfg.seed)

    partition = make_partition(cfg.dataset)
    eval_data = make_eval_data(cfg.dataset)
    params = init_params(cfg.dataset.num_features, cfg.dataset.num_classes, rng)
    pool = sorted(partition)

    # T4: one injector per run; None for reference runs, which then execute
    # exactly as before (byte-identical stage hashes).
    injector = None
    if cfg.failure is not None:
        injector = build_injector(cfg.failure, partition, rng)

    outcomes: list[OutcomeState] = []
    for round_id in range(cfg.rounds):
        round_pool = pool if injector is None else injector.candidate_pool(pool, round_id)
        selection = _stage(
            "selection", lambda: select_clients(round_pool, round_id, cfg.selection, rng)
        )
        _record(recorder, round_id, "selection", selection)

        # Per-client stages are recorded ONCE per stage, as a list of states
        # (CONTRACTS §1), never once per client.
        local_states = [
            _stage(
                "local",
                lambda cid=cid: local_train(
                    params,
                    cid,
                    partition[cid],
                    round_id,
                    cfg.local
                    if injector is None
                    else injector.local_cfg(cid, cfg.local, round_id),
                    rng,
                ),
            )
            for cid in selection.selected_ids
        ]
        _record(recorder, round_id, "local", local_states)

        compressed = [
            _stage(
                "compression",
                lambda s=state: compress(
                    s,
                    cfg.compression
                    if injector is None
                    else injector.compression_cfg(
                        s.client_id, cfg.compression, round_id
                    ),
                    rng,
                ),
            )
            for state in local_states
        ]
        _record(recorder, round_id, "compression", compressed)

        weights = {s.client_id: float(s.num_examples) for s in local_states}
        if injector is not None:
            weights = injector.weights(weights, round_id)
        aggregation = _stage(
            "aggregation",
            lambda: aggregate(compressed, weights, cfg.aggregation, rng),
        )
        _record(recorder, round_id, "aggregation", aggregation)

        params = params + aggregation.aggregate

        outcome = _stage("evaluation", lambda: evaluate(params, eval_data))
        outcome.round_id = round_id
        _record(recorder, round_id, "evaluation", outcome)
        outcomes.append(outcome)

    return outcomes
