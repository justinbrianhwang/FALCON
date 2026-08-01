"""Run configuration schemas (contract v0.1). YAML in configs/ maps 1:1 onto RunConfig."""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from .states import FailureSpecification


class SelectionConfig(BaseModel):
    clients_per_round: int
    policy: Literal["uniform"] = "uniform"


class LocalConfig(BaseModel):
    lr: float
    local_steps: int
    batch_size: int


class CompressionConfig(BaseModel):
    kind: Literal["identity", "topk", "quantization"] = "identity"
    parameters: dict[str, Any] = Field(default_factory=dict)


class AggregationConfig(BaseModel):
    rule: Literal["weighted_mean", "uniform_mean", "median", "trimmed_mean"] = "weighted_mean"
    parameters: dict[str, Any] = Field(default_factory=dict)


class DatasetConfig(BaseModel):
    name: Literal["synthetic"] = "synthetic"
    num_clients: int
    num_features: int = 20
    num_classes: int = 2
    samples_per_client: int = 100
    heterogeneity: float = 0.0  # 0 = IID; larger = more client shift
    minority_class: Optional[int] = None
    minority_client_fraction: float = 0.2
    seed: int = 1001  # partition seed, independent of run seed


class RunConfig(BaseModel):
    run_id: str
    seed: int
    rounds: int
    dataset: DatasetConfig
    selection: SelectionConfig
    local: LocalConfig
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    failure: Optional[FailureSpecification] = None
