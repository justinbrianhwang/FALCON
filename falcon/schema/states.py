"""Stage-boundary state schemas (contract v0.1, see docs/CONTRACTS.md)."""
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

Stage = Literal["selection", "local", "compression", "aggregation", "evaluation"]

STAGES: tuple[Stage, ...] = ("selection", "local", "compression", "aggregation", "evaluation")


class _ArrayModel(BaseModel):
    """Base for states that carry numpy arrays (recorder serializes them to .npz)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    content_hash: Optional[str] = None  # filled by the recorder on save


class SelectionState(_ArrayModel):
    round_id: int
    candidate_ids: list[str]
    selected_ids: list[str]
    sampling_probs: dict[str, float]
    rng_state: dict[str, Any] = Field(default_factory=dict)


class ClientLocalState(_ArrayModel):
    round_id: int
    client_id: str
    base_model_hash: str
    update: np.ndarray  # delta = trained - global, flat float64
    num_examples: int
    num_steps: int
    loss_history: list[float]
    rng_state: dict[str, Any] = Field(default_factory=dict)


class CompressionState(_ArrayModel):
    round_id: int
    client_id: str
    uncompressed_hash: str
    update: np.ndarray  # decompressed update actually seen by the server
    compression_params: dict[str, Any] = Field(default_factory=dict)
    bytes_transmitted: int = 0


class AggregationState(_ArrayModel):
    round_id: int
    received_ids: list[str]
    accepted_ids: list[str]
    rejected_ids: list[str]
    weights: dict[str, float]
    aggregate: np.ndarray


class OutcomeState(_ArrayModel):
    round_id: int
    model_hash: str
    metrics: dict[str, float]  # e.g. {"accuracy": ..., "loss": ...}
    per_class: dict[str, dict[str, float]] = Field(default_factory=dict)


class FailureSpecification(BaseModel):
    stage: Stage
    type: str
    active_rounds: tuple[int, int]  # inclusive [start, end]
    severity: int = 1
    parameters: dict[str, Any] = Field(default_factory=dict)


class InterventionSpecification(BaseModel):
    target_run_id: str
    source_run_id: str
    round_id: int
    stage: Stage
    mode: Literal["restore", "inject", "sham"]
    scope: dict[str, Any] = Field(default_factory=dict)  # e.g. {"client_ids": [...]}


class InterventionResult(BaseModel):
    spec: InterventionSpecification
    valid: bool
    reason: Optional[str] = None
    outcome_metrics: dict[str, float] = Field(default_factory=dict)


class PairValidationReport(BaseModel):
    reference_run_id: str
    failure_run_id: str
    status: Literal["MATCHED", "MATCHED_WITH_WARNINGS", "INVALID_PAIR"]
    checks: dict[str, bool]
    warnings: list[str] = Field(default_factory=list)
    first_divergence_round: Optional[int] = None
    first_divergence_stage: Optional[Stage] = None


class AttributionReport(BaseModel):
    pair: PairValidationReport
    failure_gap: dict[str, float]
    stage_effects: dict[str, dict[str, float]]  # stage -> {"SRE":..,"SIE":..,"nSRE":..,"nSIE":..}
    origin_ranking: list[str]
    roles: dict[str, str] = Field(default_factory=dict)  # stage -> originator/amplifier/...
    notes: list[str] = Field(default_factory=list)


class RunMetadata(BaseModel):
    run_id: str
    seed: int
    rounds: int
    config: dict[str, Any]
    failure: Optional[FailureSpecification] = None
    code_version: Optional[str] = None
