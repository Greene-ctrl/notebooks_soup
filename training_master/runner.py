"""Disposable runtime orchestration and a CLI with an offline planning path."""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import yaml

from .artifacts import HFArtifactStore, RunPaths, write_json
from .checkpoint import find_latest_checkpoint, publish_checkpoint, restore_checkpoint
from .engines import TrainingEngine, resolve_engine
from .hardware import HardwareProfile, probe_hardware
from .job import JobSpec, PromptJobSpec, job_digest, load_job
from .platform import RuntimePlatform, detect_platform
from .secrets import get_secret, restore_secrets


@dataclass
class Execution:
    job: JobSpec
    hardware: HardwareProfile
    platform: RuntimePlatform
    engine: TrainingEngine
    paths: RunPaths
    planning_only: bool = False

    def dry_run(self, *, cli: bool = False) -> dict:
        from .engines import SoupEngine
        if isinstance(self.engine, SoupEngine):
            detail = self.engine.dry_run(self.job, self.hardware, self.paths, cli=cli)
        else:
            if cli:
                raise ValueError("CLI preflight is provided only for Soup")
            detail = self.engine.dry_run(self.job, self.hardware, self.paths)
        result = {
            "run_id": self.job.run_id, "platform": self.platform.name,
            "requested_engine": self.job.engine.type, "resolved_engine": self.engine.name,
            "hardware": self.hardware.to_dict(), "requirements": self.job.requirements.model_dump(),
            "planning_only": self.planning_only, "preflight": detail,
            "checkpoint_path": str(self.paths.checkpoints),
            "hf_repo": self.job.artifacts.hf_repo, "metrics_path": str(self.paths.root / "metrics"),
            "status": "planned" if self.planning_only else "validated",
        }
        try:
            source_root = Path(__file__).resolve().parents[1]
            result["runner_commit"] = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            result["runner_commit"] = None
        write_json(self.paths.root / "resolved.json", result)
        return result

    def execute(self, *, publish: bool = False) -> dict:
        if self.planning_only:
            raise RuntimeError("Planning-only execution cannot train; resolve again against real hardware and installed engines")
        if not self.engine.available():
            raise RuntimeError("Selected engine dependencies are missing; bootstrap only that engine first")
        restore_secrets(self.platform)
        store = None
        if self.job.artifacts.hf_repo:
            store = HFArtifactStore(self.job.artifacts.hf_repo, get_secret("HF_TOKEN", self.platform, required=publish))
        elif publish:
            raise ValueError("Explicit publication requires artifacts.hf_repo")
        status_path = self.paths.root / "status.json"
        write_json(status_path, {"status": "preparing", "run_id": self.job.run_id})
        try:
            if store:
                store.check_access()
            resume = restore_checkpoint(self.job, self.paths, engine=self.engine.name, store=store)
            # A previously completed run must not silently retrain the same ID.
            if resume is None and ((self.paths.final / "training_master_final.json").exists() or any(self.paths.trainer.glob("checkpoint-*"))):
                raise ValueError("Run has existing artifacts but no selected compatible checkpoint; use a new run_id to start fresh")
            self.dry_run()
            write_json(status_path, {"status": "running", "run_id": self.job.run_id, "engine": self.engine.name, "resume_checkpoint": str(resume) if resume else None})

            def saved(checkpoint):
                return publish_checkpoint(self.job, checkpoint, engine=self.engine.name, store=store if publish else None)

            result = self.engine.run(self.job, self.hardware, self.paths, resume=resume, on_checkpoint=saved)
            # Soup's trainer_state is useful even when no periodic save was due.
            trainer_state = self.paths.trainer / "trainer_state.json"
            metrics_source = "final_trainer_state"
            if not trainer_state.exists():
                latest = find_latest_checkpoint(self.job, self.paths, engine=self.engine.name)
                if latest:
                    trainer_state = latest / "trainer_state.json"
                    metrics_source = "latest_checkpoint"
            if trainer_state.exists():
                shutil.copy2(trainer_state, self.paths.root / "metrics" / "trainer_state.json")
            write_json(self.paths.root / "metrics" / "availability.json", {
                "trainer_state_available": trainer_state.exists(),
                "source": metrics_source if trainer_state.exists() else "logs_only",
                "final_metrics_guaranteed": metrics_source == "final_trainer_state" and trainer_state.exists(),
            })
            write_json(self.paths.final / "training_master_final.json", {"job_digest": job_digest(self.job), "engine": self.engine.name})
            result.update(run_id=self.job.run_id, checkpoint_path=str(self.paths.checkpoints), hf_repo=self.job.artifacts.hf_repo, metrics_path=str(self.paths.root / "metrics"), status="succeeded", published=False)
            write_json(status_path, result)
            if publish:
                store.publish_run(self.job, self.paths)
                result["published"] = True
                write_json(status_path, result)
            return result
        except BaseException as error:
            # Exception strings from third-party HTTP/CLI code can contain URLs
            # or credentials. Keep status safe; original traceback stays local.
            write_json(status_path, {"status": "failed", "run_id": self.job.run_id, "error_type": type(error).__name__, "engine": self.engine.name})
            raise


def prepare_execution(source: JobSpec | dict | str | Path, *, platform: RuntimePlatform | None = None, hardware: HardwareProfile | None = None, output_dir: str | Path | None = None, planning_only: bool = False, engines=None) -> Execution:
    job = load_job(source)
    if isinstance(job, PromptJobSpec):
        if engines is not None:
            raise ValueError("Training engine overrides do not apply to prompt optimization")
        from .prompt_runner import prepare_prompt_execution
        return prepare_prompt_execution(job, platform=platform, hardware=hardware, output_dir=output_dir, planning_only=planning_only)
    platform = platform or detect_platform()
    hardware = hardware or probe_hardware(platform)
    engine = resolve_engine(job, hardware, engines=engines, allow_unavailable=planning_only)
    paths = RunPaths.for_job(job, output_dir or platform.output_dir)
    existing = paths.root / "job.yaml"
    if existing.exists() and job_digest(load_job(existing)) != job_digest(job):
        raise ValueError("run_id already has a different JobSpec; choose a new run_id")
    paths.create()
    (paths.root / "job.yaml").write_text(yaml.safe_dump(job.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    write_json(paths.root / "hardware.json", hardware.to_dict())
    # Do not overwrite a running/completed status merely to inspect a dry run.
    if not (paths.root / "status.json").exists():
        write_json(paths.root / "status.json", {"status": "planned", "run_id": job.run_id})
    return Execution(job, hardware, platform, engine, paths, planning_only)


def main() -> None:
    parser = argparse.ArgumentParser(description="Training Master: backend-neutral disposable execution")
    parser.add_argument("job", help="YAML/JSON path, HTTPS URL, or hf://owner/repo/path")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hardware-profile", help="Recorded JSON profile for offline planning only")
    parser.add_argument("--plan", action="store_true", help="Allow an absent engine; generate config without training or downloading weights")
    parser.add_argument("--cli-preflight", action="store_true", help="Run Soup doctor + train --dry-run (requires installed training extras)")
    parser.add_argument("--execute", action="store_true", help="Explicitly start training")
    parser.add_argument("--publish", action="store_true", help="Explicitly enable uploads to artifacts.hf_repo during execution")
    args = parser.parse_args()
    if args.execute and (args.plan or args.hardware_profile):
        parser.error("--execute requires real probed hardware, not --plan/--hardware-profile")
    if args.publish and not args.execute:
        parser.error("--publish requires --execute")
    if args.hardware_profile and not args.plan:
        parser.error("--hardware-profile requires --plan")
    hardware = HardwareProfile(**json.loads(Path(args.hardware_profile).read_text())) if args.hardware_profile else None
    execution = prepare_execution(args.job, output_dir=args.output_dir, hardware=hardware, planning_only=args.plan)
    if args.execute:
        if isinstance(execution.job, PromptJobSpec):
            print(json.dumps({"preflight": execution.dry_run()}, indent=2), file=sys.stderr, flush=True)
        # Prompt publication also requires an explicit JobSpec setting. Preserve
        # the existing model-training CLI's upload behavior unchanged.
        publish = (True if args.publish else None) if isinstance(execution.job, PromptJobSpec) else args.publish
        result = execution.execute(publish=publish)
    else:
        result = execution.dry_run(cli=args.cli_preflight)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
