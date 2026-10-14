"""Prompt optimizer resolution is independent of model-training resolution."""
from ..job import PromptJobSpec
from .base import OptimizerEngine
from .gepa import GEPAOptimizer


def resolve_optimizer(job, *, allow_unavailable=False) -> OptimizerEngine:
    if not isinstance(job, PromptJobSpec):
        raise ValueError("Optimizer resolver requires kind=prompt_optimization")
    optimizer = GEPAOptimizer()
    if not optimizer.supports(job):
        raise ValueError("Unsupported prompt optimizer")
    if not allow_unavailable and not optimizer.available():
        raise RuntimeError("Install the optional gepa extra (DSPy 3.4.1, GEPA 0.1.4) on Python 3.10–3.14")
    return optimizer


__all__ = ["OptimizerEngine", "GEPAOptimizer", "resolve_optimizer"]
