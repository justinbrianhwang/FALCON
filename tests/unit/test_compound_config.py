from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from falcon.schema import FailureSpecification, RunConfig

CASES = Path(__file__).resolve().parents[2] / "configs" / "cases"


def _base() -> dict:
    return yaml.safe_load(
        (CASES / "synthetic_reference.yaml").read_text(encoding="utf-8")
    )


def _failure(stage: str = "selection") -> FailureSpecification:
    return FailureSpecification(
        stage=stage,
        type="test",
        active_rounds=(1, 2),
    )


def test_run_config_rejects_single_and_compound_failures_together():
    with pytest.raises(ValidationError, match="cannot both be set"):
        RunConfig.model_validate(
            {**_base(), "failure": _failure(), "failures": [_failure("local")]}
        )


def test_run_config_rejects_two_compound_failures_at_the_same_stage():
    with pytest.raises(ValidationError, match="different stages"):
        RunConfig.model_validate(
            {**_base(), "failure": None, "failures": [_failure(), _failure()]}
        )
