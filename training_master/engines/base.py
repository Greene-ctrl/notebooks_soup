"""A small engine contract and common text-SFT dataset preparation."""
from __future__ import annotations

import importlib.util
import json
from abc import ABC, abstractmethod
from pathlib import Path

from ..artifacts import RunPaths
from ..hardware import HardwareProfile
from ..job import JobSpec


class UnsupportedJob(ValueError):
    pass


def installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def hardware_errors(job: JobSpec, hardware: HardwareProfile) -> list[str]:
    errors = []
    first_vram = hardware.gpu_vram_gb[0] if hardware.gpu_vram_gb else 0
    if job.requirements.min_vram_gb > first_vram:
        errors.append(f"requirements.min_vram_gb={job.requirements.min_vram_gb} exceeds the first visible device (VRAM is not pooled)")
    if job.training.quantization == "4bit" and not hardware.cuda_available:
        errors.append("training.quantization=4bit requires working CUDA")
    return errors


class TrainingEngine(ABC):
    name: str

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def unsupported_fields(self, job: JobSpec) -> list[str]: ...

    def supports(self, job: JobSpec, hardware: HardwareProfile) -> bool:
        return not self.unsupported_fields(job) and not hardware_errors(job, hardware)

    def validate(self, job: JobSpec, hardware: HardwareProfile) -> None:
        errors = self.unsupported_fields(job) + hardware_errors(job, hardware)
        if errors:
            raise UnsupportedJob(f"{self.name}: " + "; ".join(errors))

    def prepare(self, job: JobSpec, hardware: HardwareProfile, paths: RunPaths | None = None) -> dict:
        self.validate(job, hardware)
        return {"engine": self.name, "run_id": job.run_id}

    def dry_run(self, job: JobSpec, hardware: HardwareProfile, paths: RunPaths | None = None) -> dict:
        self.validate(job, hardware)
        return {"engine": self.name, "valid": True, "weights_downloaded": False}

    @abstractmethod
    def run(self, job: JobSpec, hardware: HardwareProfile, paths: RunPaths, *, resume: Path | None = None, on_checkpoint=None) -> dict: ...


def materialize_dataset(job: JobSpec, paths: RunPaths) -> Path:
    """Resolve the selected split once into normalized text/chat/alpaca JSONL.

    HF streaming keeps a smoke limit small. Both engines consume these identical
    rows; no platform-specific data decisions and no model/tokenizer download.
    """
    spec = job.dataset
    from ..paths import REPO_ROOT, resolve_runtime_path
    source = Path(resolve_runtime_path(spec.source, repo_root=REPO_ROOT))
    if source.is_file():
        if spec.config is not None or spec.split != "train":
            raise UnsupportedJob("dataset.config/split cannot select a local JSONL split")
        def local_rows():
            with source.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)
        rows = local_rows()
    else:
        from datasets import load_dataset
        rows = load_dataset(spec.source, spec.config, split=spec.split, streaming=True)
    target = paths.root / "data.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            if spec.format == "plaintext":
                value = row.get(spec.text_field)
                if not isinstance(value, str):
                    raise ValueError(f"dataset.text_field={spec.text_field} must contain strings")
                if not value.strip():
                    continue
                record = {"text": value}
            elif spec.format == "alpaca":
                if not all(isinstance(row.get(k), str) for k in ("instruction", "output")):
                    raise ValueError("alpaca rows require instruction and output strings")
                if not isinstance(row.get("input", ""), str):
                    raise ValueError("alpaca input must be a string")
                record = {k: row.get(k, "") for k in ("instruction", "input", "output")}
            else:
                messages = row.get("messages")
                if not isinstance(messages, list) or not messages or any(
                    not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str)
                    for m in messages
                ):
                    raise ValueError("chatml rows require text-only messages with system/user/assistant roles")
                record = {"messages": messages}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if spec.limit and count >= spec.limit:
                break
    if count == 0:
        raise ValueError("Selected dataset has no nonempty training rows")
    return target
