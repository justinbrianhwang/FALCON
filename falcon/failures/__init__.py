"""Failure injectors: one per pipeline stage (Task T4, docs/tasks/T4)."""
from .base import FailureInjector, build_injector

__all__ = ["FailureInjector", "build_injector"]
