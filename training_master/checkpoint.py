"""Completed Trainer checkpoints, identity checks, and optional durable hooks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .artifacts import HFArtifactStore, RunPaths, write_json
from .job import JobSpec, job_digest

MANIFEST = "training_master_checkpoint.json"


class IncompleteCheckpoint(ValueError):
    """A CLI checkpoint save has not finished yet."""


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seal_checkpoint(job: JobSpec, checkpoint: Path, engine: str) -> Path:
    """Call only after Trainer's save finishes (on_save / finalized CLI save).

    Adapters alone are not resumable checkpoints. Refuse missing optimizer,
    scheduler, RNG, or Trainer state and hash the complete immutable snapshot.
    """
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
    if any(not (checkpoint / name).is_file() for name in required):
        raise IncompleteCheckpoint(f"Incomplete Trainer checkpoint: {checkpoint.name}")
    if not list(checkpoint.glob("rng_state*.pth")):
        raise IncompleteCheckpoint("Checkpoint has no RNG state")
    if not any(checkpoint.glob("*.safetensors")) and not any(checkpoint.glob("*model*.bin")):
        raise IncompleteCheckpoint("Checkpoint has no model/adapter weights")
    try:
        state = json.loads((checkpoint / "trainer_state.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise IncompleteCheckpoint("Trainer state is still being written") from error
    step = state.get("global_step")
    if not isinstance(step, int) or step < 1 or checkpoint.name != f"checkpoint-{step}":
        raise ValueError("Checkpoint name and Trainer global_step disagree")
    files = {}
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise ValueError("Checkpoint symlinks are not supported")
        if path.is_file() and path.name != MANIFEST:
            try:
                files[path.relative_to(checkpoint).as_posix()] = _hash(path)
            except OSError as error:
                raise IncompleteCheckpoint("Checkpoint files changed during the save scan") from error
    write_json(checkpoint / MANIFEST, {"version": 1, "engine": engine, "job_digest": job_digest(job), "step": step, "files": files})
    return checkpoint


def validate_checkpoint(job: JobSpec, checkpoint: Path, engine: str | None = None) -> int:
    manifest = json.loads((checkpoint / MANIFEST).read_text())
    if manifest.get("version") != 1 or manifest.get("job_digest") != job_digest(job):
        raise ValueError("Checkpoint belongs to a different JobSpec")
    if engine and manifest.get("engine") != engine:
        raise ValueError("Checkpoint engine differs; cross-engine resume is not supported")
    files = manifest.get("files", {})
    if not isinstance(files, dict) or not {"trainer_state.json", "optimizer.pt", "scheduler.pt"} <= files.keys():
        raise ValueError("Checkpoint manifest lacks Trainer state")
    if not any(name.startswith("rng_state") and name.endswith(".pth") for name in files):
        raise ValueError("Checkpoint manifest lacks RNG state")
    if not any(name.endswith(".safetensors") or ("model" in name and name.endswith(".bin")) for name in files):
        raise ValueError("Checkpoint manifest lacks model weights")
    root = checkpoint.resolve()
    for name, digest in files.items():
        path = checkpoint / name
        if not path.resolve().is_relative_to(root) or path.is_symlink() or _hash(path) != digest:
            raise ValueError("Checkpoint is incomplete or corrupted")
    step = manifest["step"]
    if not isinstance(step, int) or step < 1 or checkpoint.name != f"checkpoint-{step}":
        raise ValueError("Checkpoint step is invalid")
    return step


def find_latest_checkpoint(job: JobSpec, paths: RunPaths | None = None, *, engine: str | None = None) -> Path | None:
    paths = paths or RunPaths.for_job(job)
    candidates = []
    for manifest in paths.checkpoints.rglob(MANIFEST):
        checkpoint = manifest.parent
        try:
            step = validate_checkpoint(job, checkpoint, engine)
        except (ValueError, OSError, KeyError, TypeError):
            continue  # Ignore incomplete uploads, partial saves and incompatible snapshots.
        candidates.append((step, str(checkpoint), checkpoint))
    return max(candidates)[2] if candidates else None


def restore_checkpoint(job: JobSpec, paths: RunPaths | None = None, *, engine: str | None = None, store: HFArtifactStore | None = None) -> Path | None:
    if job.resume.mode == "never":
        return None
    paths = paths or RunPaths.for_job(job)
    if store:
        store.restore(job, paths)
    checkpoint = find_latest_checkpoint(job, paths, engine=engine)
    if job.resume.mode == "required" and checkpoint is None:
        raise ValueError("resume.mode=required but no complete compatible checkpoint exists")
    return checkpoint


def publish_checkpoint(job: JobSpec, checkpoint: str | Path, *, engine: str, store: HFArtifactStore | None = None) -> Path:
    checkpoint = Path(checkpoint)
    seal_checkpoint(job, checkpoint, engine)
    if store:
        store.publish_checkpoint(job, checkpoint)
    return checkpoint
