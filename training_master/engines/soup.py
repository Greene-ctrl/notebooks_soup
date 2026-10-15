"""Soup 0.76.0 CLI/config adapter; no dependency on a private training API."""
from __future__ import annotations

import importlib.metadata
import hashlib
import shutil
from pathlib import Path

import yaml

from ..artifacts import RunPaths
from ..bootstrap import SOUP_VERSION, soup_executable
from ..checkpoint import IncompleteCheckpoint, MANIFEST
from ..hardware import HardwareProfile
from ..job import JobSpec
from ..process import run_process
from ..paths import REPO_ROOT, resolve_job_paths
from .base import TrainingEngine, UnsupportedJob, materialize_dataset


class SoupEngine(TrainingEngine):
    name = "soup"

    def __init__(self, *, tail_lines: int = 100):
        self.tail_lines = tail_lines

    def available(self) -> bool:
        try:
            executable = shutil.which("soup")
            expected = soup_executable()
            return (bool(executable) and Path(executable).resolve() == expected.resolve()
                    and importlib.metadata.version("soup-cli") == SOUP_VERSION)
        except importlib.metadata.PackageNotFoundError:
            return False

    def unsupported_fields(self, job: JobSpec) -> list[str]:
        errors = []
        t = job.training
        if job.task.type != "sft":
            errors.append(f"task.type={job.task.type}: initial adapter supports only text SFT")
        if t.max_steps is not None:
            errors.append("training.max_steps: Soup 0.76.0 has no such schema/CLI setting; use bounded data and epochs or request direct Unsloth")
        if job.model.revision != "main":
            errors.append("model.revision: Soup 0.76.0 has no model revision setting")
        if not t.lora.enabled:
            errors.append("training.lora.enabled=false: initial runner supports only LoRA/QLoRA")
        if job.dataset.format != "plaintext" and job.dataset.text_field != "text":
            errors.append("dataset.text_field: applies only to plaintext datasets")
        if t.stream_layers and ((job.engine.backend or "transformers") != "transformers" or t.quantization != "none" or t.batch_size != 1 or t.gradient_accumulation_steps != 1):
            errors.append("training.stream_layers requires transformers SFT, quantization=none, batch_size=1, gradient_accumulation_steps=1")
        return errors

    def config(self, job: JobSpec, paths: RunPaths, *, dataset_path: Path | None = None) -> dict:
        errors = self.unsupported_fields(job)
        if errors:
            raise UnsupportedJob("soup: " + "; ".join(errors))
        t = job.training
        return {
            "base": job.model.id, "task": job.task.type, "backend": job.engine.backend or "transformers",
            "data": {"train": str(dataset_path or paths.root / "data.jsonl"), "format": job.dataset.format, "max_length": t.max_length, "val_split": 0.0},
            # Soup defaults to 4bit: write "none" explicitly when unquantized.
            "training": {
                "epochs": t.epochs, "lr": t.learning_rate, "batch_size": t.batch_size,
                "gradient_accumulation_steps": t.gradient_accumulation_steps,
                "gradient_checkpointing": t.gradient_checkpointing,
                "quantization": t.quantization, "stream_layers": t.stream_layers,
                "seed": t.seed, "data_seed": t.seed,
                "save_steps": job.artifacts.checkpoint_interval, "logging_steps": 1,
                "lora": {"r": t.lora.r, "alpha": t.lora.alpha, "dropout": t.lora.dropout, "target_modules": t.lora.target_modules},
            },
            "output": str(paths.trainer),
        }

    def prepare(self, job: JobSpec, hardware: HardwareProfile, paths: RunPaths | None = None) -> dict:
        self.validate(job, hardware)
        paths = paths or RunPaths.for_job(job)
        paths.create()
        config_path = paths.root / "soup.yaml"
        job = resolve_job_paths(job, REPO_ROOT)
        config_path.write_text(yaml.safe_dump(self.config(job, paths), sort_keys=False), encoding="utf-8")
        return {"engine": self.name, "config_path": str(config_path)}

    def dry_run(self, job: JobSpec, hardware: HardwareProfile, paths: RunPaths | None = None, *, cli: bool = False) -> dict:
        self.validate(job, hardware)
        paths = paths or RunPaths.for_job(job)
        if not (paths.root / "soup.yaml").exists():
            self.prepare(job, hardware, paths)
        plan = {"engine": self.name, "config_path": str(paths.root / "soup.yaml"),
                "valid": True, "weights_downloaded": False, "cli_preflight": False}
        if cli:
            if not self.available():
                raise RuntimeError("Install the pinned Soup CLI in the kernel environment and selected training extra first")
            self._validated_config = None
            # Use the same materialized data and config for preflight and train.
            materialize_dataset(resolve_job_paths(job, REPO_ROOT), paths)
            self._command(["doctor", "--config", plan["config_path"]], paths, name="soup-doctor")
            self._command(["train", "--config", plan["config_path"], "--dry-run"], paths, name="soup-dry-run")
            self._validated_config = self._fingerprint(paths)
            plan["cli_preflight"] = True
        return plan

    def _fingerprint(self, paths: RunPaths):
        digest = hashlib.sha256()
        for path in (paths.root / "soup.yaml", paths.root / "data.jsonl"):
            if not path.exists():
                return None
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    def _command(self, args: list[str], paths: RunPaths, *, name: str, on_poll=None):
        # Keep every invocation, including retries, while the first uses the
        # simple documented filename. Diagnostics select the newest record.
        stem = name
        index = 1
        while (paths.root / "commands" / f"{stem}.json").exists():
            stem = f"{name}-{index:03d}"
            index += 1
        return run_process([str(soup_executable()), *args], cwd=paths.root,
                           log_path=paths.root / "logs" / f"{stem}.log",
                           record_path=paths.root / "commands" / f"{stem}.json", on_poll=on_poll, tail_lines=self.tail_lines)

    def run(self, job: JobSpec, hardware: HardwareProfile, paths: RunPaths, *, resume: Path | None = None, on_checkpoint=None) -> dict:
        # Direct engine callers also receive mandatory preflight. Execution
        # supplies an already validated config and marks that specific snapshot.
        self.validate(job, hardware)
        fingerprint = self._fingerprint(paths)
        if fingerprint is None or getattr(self, "_validated_config", None) != fingerprint:
            self.dry_run(job, hardware, paths, cli=True)
        command = ["train", "--config", str(paths.root / "soup.yaml"), "--yes"]
        if resume:
            command += ["--resume", str(resume.resolve())]
        # Never relabel or publish checkpoints from before this subprocess run.
        seen = {path for path in paths.trainer.glob("checkpoint-*") if (path / MANIFEST).is_file()}

        def scan():
            if on_checkpoint:
                for path in sorted(paths.trainer.glob("checkpoint-*")):
                    if path.is_dir() and path not in seen and (path / "trainer_state.json").is_file():
                        # Trainer writes state after weights/optimizer/RNG. Ignore
                        # incomplete files until the next scan; upload failures must
                        # propagate so durability is never reported falsely.
                        try:
                            on_checkpoint(path)
                        except IncompleteCheckpoint:
                            continue
                        seen.add(path)

        self._command(command, paths, name="soup-train", on_poll=scan)
        scan()
        # Soup saves the final adapter alongside its Trainer checkpoints.
        for file in paths.trainer.iterdir():
            if file.is_file() and file.name != MANIFEST:
                shutil.copy2(file, paths.final / file.name)
        return {"output": str(paths.final), "engine": self.name}
