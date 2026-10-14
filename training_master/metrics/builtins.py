"""Deterministic text metrics and useful feedback without extra model calls."""
from __future__ import annotations

from .base import numeric_score


def _value(record, field):
    if isinstance(record, dict):
        return record.get(field)
    return getattr(record, field, None)


def _text(record, field):
    value = _value(record, field)
    return value.strip().casefold() if isinstance(value, str) else None


def exact_match(field):
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        expected, actual = _text(gold, field), _text(pred, field)
        return float(expected is not None and actual is not None and expected == actual)
    return metric


def contains(field):
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        expected, actual = _text(gold, field), _text(pred, field)
        return float(bool(expected) and actual is not None and expected in actual)
    return metric


def exact_match_with_feedback(field):
    score_metric = exact_match(field)

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        import dspy
        score = score_metric(gold, pred)
        feedback = "Correct: the answer matches after trimming whitespace and ignoring case." if score else (
            f"Expected {field}={_value(gold, field)!r}; received {_value(pred, field)!r}. "
            "Return the expected answer without unrelated text."
        )
        return dspy.Prediction(score=score, feedback=feedback)
    return metric


METRIC_REGISTRY = {
    "exact_match": exact_match,
    "exact_match_with_feedback": exact_match_with_feedback,
    "contains": contains,
}


def resolve_metric(spec, *, output_field="answer"):
    try:
        return METRIC_REGISTRY[spec.type](output_field)
    except KeyError as error:
        raise ValueError(f"Unknown metric: {spec.type}") from error


def evaluate(program, examples, metric) -> dict:
    """Evaluate once, sequentially and fail on provider errors rather than score 0."""
    rows = []
    for example in examples:
        prediction = program(**example.inputs())
        score = numeric_score(metric(example, prediction))
        rows.append({"score": score})
    if not rows:
        raise ValueError("Cannot evaluate an empty split")
    return {"score": sum(row["score"] for row in rows) / len(rows), "count": len(rows), "examples": rows}
