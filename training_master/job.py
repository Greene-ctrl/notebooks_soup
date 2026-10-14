"""Backend-neutral, versioned jobs. Unknown settings are errors, not hints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit
from urllib.request import urlopen

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

PositiveInt = Annotated[StrictInt, Field(gt=0)]


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True, allow_inf_nan=False)


class ModelSpec(Spec):
    id: str = Field(min_length=1)
    revision: str = "main"


class DatasetSpec(Spec):
    source: str = Field(min_length=1)
    config: str | None = None
    split: str = "train"
    format: Literal["plaintext", "alpaca", "chatml"] = "plaintext"
    text_field: str = "text"
    limit: PositiveInt | None = None


class TaskSpec(Spec):
    type: Literal["sft", "dpo", "grpo", "ppo", "distillation"] = "sft"


class EngineSpec(Spec):
    type: Literal["auto", "soup", "unsloth", "nemo_automodel"] = "auto"
    backend: Literal["transformers", "unsloth"] | None = None


class LoraSpec(Spec):
    enabled: bool = True
    r: PositiveInt = 16
    alpha: PositiveInt = 32
    dropout: float = Field(default=0.0, ge=0, lt=1)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])

    @field_validator("target_modules")
    @classmethod
    def nonempty_targets(cls, value):
        if not value or any(not v.strip() for v in value):
            raise ValueError("lora.target_modules must contain nonempty module names")
        return value


class TrainingSpec(Spec):
    epochs: PositiveInt = 1
    max_steps: PositiveInt | None = None
    batch_size: PositiveInt | Literal["auto"] = "auto"
    max_length: Annotated[StrictInt, Field(ge=64)] = 512
    learning_rate: float = Field(default=2e-4, gt=0)
    gradient_accumulation_steps: PositiveInt = 1
    gradient_checkpointing: bool = True
    quantization: Literal["none", "4bit"] = "none"
    stream_layers: bool = False
    seed: Annotated[StrictInt, Field(ge=0)] = 42
    lora: LoraSpec = Field(default_factory=LoraSpec)


class ArtifactSpec(Spec):
    hf_repo: str | None = None
    checkpoint_interval: PositiveInt = 10

    @field_validator("hf_repo")
    @classmethod
    def repo_id(cls, value):
        if value is not None:
            import re
            if not re.fullmatch(r"[\w.-]+/[\w.-]+", value) or ".." in value:
                raise ValueError("artifacts.hf_repo must be an owner/repository ID")
        return value


class RequirementsSpec(Spec):
    min_vram_gb: float = Field(default=0, ge=0)


class ResumeSpec(Spec):
    mode: Literal["auto", "never", "required"] = "auto"


class JobSpec(Spec):
    version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")
    model: ModelSpec
    dataset: DatasetSpec
    task: TaskSpec = Field(default_factory=TaskSpec)
    engine: EngineSpec = Field(default_factory=EngineSpec)
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    artifacts: ArtifactSpec = Field(default_factory=ArtifactSpec)
    requirements: RequirementsSpec = Field(default_factory=RequirementsSpec)
    resume: ResumeSpec = Field(default_factory=ResumeSpec)

    @field_validator("version", mode="before")
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("version must be integer 1")
        return value


def load_job(source: JobSpec | dict | str | Path) -> JobSpec:
    """Load an injected dict, local YAML/JSON, HTTPS URL, or hf://repo/path.

    Remote jobs are data only: YAML safe_load, strict schema, no executable hooks.
    Never put credentials in JobSpec URLs; provider authentication is separate.
    """
    if isinstance(source, JobSpec):
        return source
    if isinstance(source, dict):
        return JobSpec.model_validate(source)
    text_source = str(source)
    if text_source.startswith("hf://"):
        from huggingface_hub import hf_hub_download
        from .secrets import get_secret
        parts = text_source[5:].split("/", 2)
        if len(parts) != 3:
            raise ValueError("Use hf://owner/repository/path/to/job.yaml (model repository)")
        filename = hf_hub_download(repo_id="/".join(parts[:2]), filename=parts[2], token=get_secret("HF_TOKEN"))
        if Path(filename).stat().st_size > 1024 * 1024:
            raise ValueError("JobSpec exceeds 1 MiB")
        text = Path(filename).read_text(encoding="utf-8")
    elif "://" in text_source:
        parsed = urlsplit(text_source)
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Job URL must be HTTPS without credentials, query, or fragment")
        with urlopen(text_source, timeout=30) as response:
            if urlsplit(response.url).scheme != "https":
                raise ValueError("Job URL redirected away from HTTPS")
            data = response.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError("JobSpec exceeds 1 MiB")
        text = data.decode("utf-8")
    else:
        path = Path(source)
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("JobSpec exceeds 1 MiB")
        text = path.read_text(encoding="utf-8")
    return JobSpec.model_validate(yaml.safe_load(text))


def job_digest(job: JobSpec) -> str:
    """Training identity for resume; requested engine/runtime and storage may vary."""
    import hashlib
    data = job.model_dump(mode="json", exclude={"engine", "artifacts", "resume", "requirements"})
    data["backend"] = job.engine.backend or "transformers"
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
