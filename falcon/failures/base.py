"""Failure-injector base class and dispatcher (Task T4, docs/tasks/T4).

An injector is built once per run from a ``FailureSpecification``. Every
transform is the identity outside ``spec.active_rounds`` (inclusive) and, for
client-scoped failures, for unaffected clients. Transforms return NEW objects
and never mutate their inputs.

All injector randomness comes from the injector's own stream
``failure.<stage>`` (CONTRACTS §3, Plan §12.2) — never a shared stream, so an
injector cannot desynchronize the pipeline's downstream streams. Streams are
requested lazily: an injector that needs no randomness (or is never active)
requests nothing at all.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from falcon.schema import (
    CompressionConfig,
    FailureSpecification,
    LocalConfig,
)

if TYPE_CHECKING:
    from falcon.pipeline.synthetic_data import ClientData
    from falcon.replay.rng import Rng  # CONTRACTS §3; deferred to avoid import cycles


class FailureInjector:
    """Built from a FailureSpecification; every transform is identity when inactive."""

    #: pipeline stage this injector attaches to (subclasses override)
    stage: ClassVar[str] = ""

    def __init__(
        self,
        spec: FailureSpecification,
        partition: dict[str, "ClientData"],
        rng: "Rng",
    ):
        if self.stage and spec.stage != self.stage:
            raise ValueError(
                f"{type(self).__name__} is a stage {self.stage!r} injector, "
                f"got spec.stage={spec.stage!r}"
            )
        self.spec = spec
        self.partition = partition
        self._rng = rng
        self._stream_name = f"failure.{spec.stage}"

    def _gen(self):
        """The injector's own ``failure.<stage>`` stream (its ONLY stream)."""
        return self._rng.stream(self._stream_name)

    def active(self, round_id: int) -> bool:
        """True while ``round_id`` lies inside ``spec.active_rounds`` (inclusive)."""
        start, end = self.spec.active_rounds
        return start <= round_id <= end

    # --- stage-specific transforms, called by the runner at the hook site ---

    def candidate_pool(self, pool: list[str], round_id: int) -> list[str]:
        return list(pool)

    def local_cfg(self, client_id: str, cfg: LocalConfig, round_id: int) -> LocalConfig:
        return cfg.model_copy()

    def compression_cfg(
        self, client_id: str, cfg: CompressionConfig, round_id: int
    ) -> CompressionConfig:
        return cfg.model_copy()

    def weights(self, weights: dict[str, float], round_id: int) -> dict[str, float]:
        return dict(weights)


def build_injector(
    spec: FailureSpecification,
    partition: dict[str, "ClientData"],
    rng: "Rng",
) -> FailureInjector:
    """Dispatch on ``spec.stage`` / ``spec.type`` to the matching injector."""
    if spec.stage == "selection" and spec.type == "minority_exclusion":
        from .selection.minority_exclusion import MinorityExclusionInjector

        return MinorityExclusionInjector(spec, partition, rng)
    if spec.stage == "local" and spec.type == "lr_misconfig":
        from .local.lr_misconfig import LrMisconfigInjector

        return LrMisconfigInjector(spec, partition, rng)
    if spec.stage == "compression" and spec.type == "aggressive_topk":
        from .compression.aggressive_topk import AggressiveTopKInjector

        return AggressiveTopKInjector(spec, partition, rng)
    if spec.stage == "aggregation" and spec.type == "wrong_sample_weights":
        from .aggregation.wrong_sample_weights import WrongSampleWeightsInjector

        return WrongSampleWeightsInjector(spec, partition, rng)
    raise ValueError(
        f"unknown failure injector: stage={spec.stage!r} type={spec.type!r}"
    )
