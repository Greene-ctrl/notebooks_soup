"""Resolve local resources against an explicit source root, never notebook cwd."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from .job import JobSpec

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_runtime_path(value: str, *, repo_root: Path, local: bool = False, must_exist: bool = True) -> str:
    """Keep remote IDs/URLs; canonicalize explicit or existing local resources.

    Local means absolute, ./, ../, ~/, file://, a known data suffix, an existing
    source-root entry, or local=True. Ambiguous owner/name remains a Hub ID.
    Bare missing local model directories must be written ./models/name.
    Relative resources are repository-relative, including those in YAML jobs.
    """
    parsed = urlsplit(value)
    if parsed.scheme and parsed.scheme != "file":
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Resource URLs must not contain credentials, query or fragment")
        return value
    if parsed.scheme == "file":
        if parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("file:// resources must be local paths")
        value = parsed.path
        local = True
    path = Path(value).expanduser()
    candidate = path if path.is_absolute() else repo_root / path
    explicit = value.startswith(("/", "./", "../", "~/")) or path.suffix.lower() in {".json", ".jsonl", ".csv", ".parquet", ".yaml", ".yml", ".py"}
    if local or explicit or candidate.exists():
        candidate = candidate.resolve()
        if must_exist and not candidate.exists():
            raise FileNotFoundError(f"Local resource does not exist: {candidate}")
        return str(candidate)
    return value


def resolve_job_paths(job: JobSpec, repo_root: Path) -> JobSpec:
    """Resolve every local resource in the supported training schema.

    Additional splits, custom metrics/rewards and checkpoint path overrides are
    not supported by JobSpec and remain strict validation errors.
    """
    data = job.model_dump(mode="json")
    data["dataset"]["source"] = resolve_runtime_path(job.dataset.source, repo_root=repo_root)
    data["model"]["id"] = resolve_runtime_path(job.model.id, repo_root=repo_root)
    return JobSpec.model_validate(data)
