"""Prompt example loading, split isolation, and DSPy normalization."""
from __future__ import annotations

import json
from pathlib import Path

from ..job import HFDatasetSpec
from ..programs.registry import signature_fields


def read_examples(source) -> list[dict]:
    if isinstance(source, list):
        return source
    if isinstance(source, HFDatasetSpec):
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError("HF dataset identifiers require the optional 'datasets' package; inline/local examples need none") from error
        from itertools import islice
        return list(islice(load_dataset(source.id, source.config, split=source.split, streaming=True), source.limit))
    path = Path(source)
    if not path.is_file():
        raise ValueError("Prompt split must be inline examples, a local JSON/JSONL file, or an explicit HF dataset spec")
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Local prompt dataset exceeds 10 MiB; select a smaller explicit split")
    text = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(text)
    if not isinstance(rows, list):
        raise ValueError("Prompt JSON must contain a list of example objects")
    return rows


def load_splits(job) -> dict[str, list[dict]]:
    inputs, output = signature_fields(job.program.signature)
    result = {}
    seen = set()
    for name in ("train", "validation", "test"):
        rows = read_examples(getattr(job.dataset, name))
        if not rows:
            raise ValueError(f"dataset.{name} must not be empty")
        split_seen = set()
        normalized = []
        for row in rows:
            if not isinstance(row, dict) or any(not isinstance(row.get(field), str) for field in [*inputs, output]):
                raise ValueError(f"dataset.{name} rows require text fields: {', '.join([*inputs, output])}")
            key = tuple(row[field] for field in inputs)
            if key in seen:
                raise ValueError("Dataset splits overlap on program inputs; held-out examples must remain isolated")
            split_seen.add(key)
            normalized.append({field: row[field] for field in [*inputs, output]})
        seen.update(split_seen)
        result[name] = normalized
    return result


def dspy_examples(rows, signature, *, dspy_module=None):
    if dspy_module is None:
        import dspy as dspy_module
    inputs, _ = signature_fields(signature)
    return [dspy_module.Example(**row).with_inputs(*inputs) for row in rows]
