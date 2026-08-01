"""Passive/terminal localization baselines (Plan §19.1–19.2, Task T8)."""
from .passive import INTERVENABLE_STAGES, passive_localize, passive_stage_scores
from .terminal import NearestCentroidStageClassifier, terminal_features

__all__ = [
    "INTERVENABLE_STAGES",
    "NearestCentroidStageClassifier",
    "passive_localize",
    "passive_stage_scores",
    "terminal_features",
]
