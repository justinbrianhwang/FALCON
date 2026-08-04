"""Run configuration schemas (contract v0.1). YAML in configs/ maps 1:1 onto RunConfig."""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .states import FailureSpecification


class SelectionConfig(BaseModel):
    clients_per_round: int
    policy: Literal["uniform"] = "uniform"


class LocalConfig(BaseModel):
    lr: float
    local_steps: int
    batch_size: int
    algorithm: Literal["fedavg", "fedprox"] = "fedavg"
    prox_mu: float = 0.0


class CompressionConfig(BaseModel):
    kind: Literal["identity", "topk", "quantization"] = "identity"
    parameters: dict[str, Any] = Field(default_factory=dict)


class AggregationConfig(BaseModel):
    rule: Literal["weighted_mean", "uniform_mean", "median", "trimmed_mean"] = "weighted_mean"
    parameters: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    name: Literal["logistic_regression", "small_cnn"] = "logistic_regression"
    parameters: dict[str, Any] = Field(default_factory=dict)


class DatasetConfig(BaseModel):
    name: Literal["synthetic", "mnist", "fmnist", "cifar10", "cifar100", "svhn"] = "synthetic"
    num_clients: int
    num_features: int = 20  # synthetic only
    num_classes: int = 2
    samples_per_client: int = 100  # synthetic only
    dirichlet_alpha: Optional[float] = None  # real datasets: None = IID, else Dirichlet(alpha)
    heterogeneity: float = 0.0  # 0 = IID; larger = more client shift
    class_separation: float = 1.0  # cluster-mean distance / noise scale; lower = harder task
    label_noise: float = 0.0  # fraction of training labels flipped (deterministic from seed)
    minority_class: Optional[int] = None
    minority_client_fraction: float = 0.2
    seed: int = 1001  # partition seed, independent of run seed


class RunConfig(BaseModel):
    run_id: str
    seed: int
    rounds: int
    dataset: DatasetConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    selection: SelectionConfig
    local: LocalConfig
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    failure: Optional[FailureSpecification] = None
    failures: list[FailureSpecification] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_failures(self):
        if self.failure is not None and self.failures:
            raise ValueError("failure and failures cannot both be set")
        stages = [failure.stage for failure in self.failures]
        if len(stages) != len(set(stages)):
            raise ValueError("compound failures must target different stages")
        return self
