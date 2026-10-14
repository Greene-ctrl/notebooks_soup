"""A GEPA metric accepts predictor-level feedback arguments as well as examples."""
from typing import Any, Protocol


class FeedbackMetric(Protocol):
    def __call__(self, gold, pred, trace=None, pred_name=None, pred_trace=None) -> Any: ...


def numeric_score(value) -> float:
    score = value.score if hasattr(value, "score") else value
    score = float(score)
    if not 0 <= score <= 1:
        raise ValueError("Initial prompt metrics must return scores in [0, 1]")
    return score
