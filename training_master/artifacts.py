"""Disposable run paths and optional explicit Hugging Face synchronization."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .job import JobSpec


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @classmethod
    def for_job(cls, job: JobSpec, output_dir: str | Path = "runs") -> RunPaths:
        # run_id is schema-validated, so it cannot escape the output directory.
        return cls(Path(output_dir).resolve() / job.run_id)

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def trainer(self) -> Path:
        return self.checkpoints / "trainer"

    @property
    def final(self) -> Path:
        return self.root / "final"

    def create(self) -> None:
        for name in ("logs", "checkpoints", "metrics", "final"):
            (self.root / name).mkdir(parents=True, exist_ok=True)


class HFArtifactStore:
    """An existing model repository; methods write only when explicitly called.

    upload_folder creates a commit, so checkpoint files and their completion
    manifest become visible together. Publication authorization belongs to the
    caller, never to a mere hf_repo field in a JobSpec.
    """

    def __init__(self, repo_id: str, token: str | None = None, *, api=None):
        from huggingface_hub import HfApi
        self.repo_id = repo_id
        self.api = api or HfApi(token=token)
        self.token = token

    def prefix(self, job: JobSpec) -> str:
        return f"runs/{job.run_id}"

    def check_access(self) -> None:
        self.api.repo_info(self.repo_id, repo_type="model")

    def publish_run(self, job: JobSpec, paths: RunPaths) -> None:
        self.api.upload_folder(
            repo_id=self.repo_id, repo_type="model", folder_path=str(paths.root),
            path_in_repo=self.prefix(job),
            ignore_patterns=["checkpoints/**", "data.jsonl", "*.tmp", "*.lock"],
            commit_message=f"Training Master {job.run_id}: run artifacts",
        )

    def publish_checkpoint(self, job: JobSpec, checkpoint: Path) -> None:
        self.api.upload_folder(
            repo_id=self.repo_id, repo_type="model", folder_path=str(checkpoint),
            path_in_repo=f"{self.prefix(job)}/checkpoints/{checkpoint.name}",
            commit_message=f"Training Master {job.run_id}: {checkpoint.name}",
        )

    def restore(self, job: JobSpec, paths: RunPaths) -> None:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import EntryNotFoundError
        prefix = self.prefix(job)
        try:
            snapshot = Path(snapshot_download(
                repo_id=self.repo_id, repo_type="model", token=self.token,
                allow_patterns=[f"{prefix}/checkpoints/**"],
            ))
        except EntryNotFoundError:
            return
        import shutil
        import tempfile
        import uuid
        from .checkpoint import validate_checkpoint
        source = snapshot / prefix / "checkpoints"
        if source.is_dir():
            # Hub checkpoint directories are immutable numbered snapshots.
            # Copy only missing candidates; never overwrite an in-progress local run.
            for entry in source.iterdir():
                destination = paths.checkpoints / entry.name
                if not entry.is_dir() or not entry.name.startswith("checkpoint-"):
                    continue
                paths.checkpoints.mkdir(parents=True, exist_ok=True)
                # Snapshot cache files can be symlinks; copy into a staging
                # directory first so checksum validation sees regular files.
                with tempfile.TemporaryDirectory(dir=paths.checkpoints) as staging:
                    candidate = Path(staging) / entry.name
                    shutil.copytree(entry, candidate)
                    try:
                        validate_checkpoint(job, candidate)
                    except (ValueError, OSError, KeyError, TypeError):
                        continue
                    if destination.exists():
                        try:
                            validate_checkpoint(job, destination)
                            continue
                        except (ValueError, OSError, KeyError, TypeError):
                            # Preserve partial local data, but let the intact
                            # durable checkpoint recover the runtime.
                            destination.rename(destination.with_name(f"{entry.name}.incomplete-{uuid.uuid4().hex}"))
                    candidate.rename(destination)
