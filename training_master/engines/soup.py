"""Soup 0.76.0 CLI/config adapter; no dependency on a private training API."""
from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
from pathlib import Path

import yaml

from ..artifacts import RunPaths
from ..bootstrap import SOUP_VERSION
from ..checkpoint import IncompleteCheckpoint, MANIFEST
from ..hardware import HardwareProfile
from ..job import JobSpec
from .base import TrainingEngine, UnsupportedJob, materialize_dataset


class SoupEngine(TrainingEngine):
    name = "soup"

    def available(self) -> bool:
        try:
            return bool(shutil.which("soup")) and importlib.metadata.version("soup-cli") == SOUP_VERSION
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
        config_path.write_text(yaml.safe_dump(self.config(job, paths), sort_keys=False), encoding="utf-8")
        return {"engine": self.name, "config_path": str(config_path)}

    def dry_run(self, job: JobSpec, hardware: HardwareProfile, paths: RunPaths | None = None, *, cli: bool = False) -> dict:
        plan = self.prepare(job, hardware, paths)
        plan.update(valid=True, weights_downloaded=False, cli_preflight=False)
        if cli:
            if not self.available():
                raise RuntimeError("Install the pinned Soup CLI and selected training extra first")
            paths = paths or RunPaths.for_job(job)
            # --dry-run validates data; it may fetch a small dataset/model config,
            # but does not instantiate/download model weights. doctor alone is not
            # sufficient: its unsupported-setting reports are partly advisory.
            materialize_dataset(job, paths)
            self._command(["doctor", "--config", plan["config_path"]], paths)
            self._command(["train", "--config", plan["config_path"], "--dry-run"], paths)
            plan["cli_preflight"] = True
        return plan

    def _command(self, args: list[str], paths: RunPaths) -> None:
        with (paths.root / "logs" / "soup.log").open("a", encoding="utf-8") as log:
            subprocess.run(["soup", *args], cwd=paths.root.parent.parent, stdout=log, stderr=subprocess.STDOUT, check=True)

    def run(self, job: JobSpec, hardware: HardwareProfile, paths: RunPaths, *, resume: Path | None = None, on_checkpoint=None) -> dict:
        self.dry_run(job, hardware, paths, cli=True)
        command = ["soup", "train", "--config", str(paths.root / "soup.yaml"), "--yes"]
        if resume:
            command += ["--resume", str(resume)]
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

        with (paths.root / "logs" / "soup.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=paths.root.parent.parent, stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    try:
                        code = process.wait(timeout=2)
                        break
                    except subprocess.TimeoutExpired:
                        scan()
                scan()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        # Soup saves the final adapter alongside its Trainer checkpoints.
        for file in paths.trainer.iterdir():
            if file.is_file() and file.name != MANIFEST:
                shutil.copy2(file, paths.final / file.name)
        return {"output": str(paths.final), "engine": self.name}
