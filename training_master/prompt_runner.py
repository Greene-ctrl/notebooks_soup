"""CPU-compatible prompt workflow with stages, safe artifacts and explicit uploads."""
from __future__ import annotations

import json
from dataclasses import dataclass

import yaml

from .artifacts import HFArtifactStore, RunPaths, write_json
from .job import PromptJobSpec, job_digest, load_job
from .optimizers import resolve_optimizer
from .optimizers.lm import apply_lm_environment
from .platform import detect_platform
from .hardware import probe_hardware
from .secrets import get_secret


@dataclass
class PromptExecution:
    job: PromptJobSpec
    platform: object
    hardware: object
    optimizer: object
    paths: RunPaths
    rows: dict
    planning_only: bool = False

    def dry_run(self, *, cli=False) -> dict:
        if cli:
            raise ValueError("Soup CLI preflight does not apply to prompt optimization")
        plan = self.optimizer.dry_run(self.job, paths=self.paths, rows=self.rows)
        plan.update(platform=self.platform.name, hardware=self.hardware.to_dict(), planning_only=self.planning_only,
                    run_id=self.job.run_id, artifact_path=str(self.paths.root), hf_repo=self.job.artifacts.hf_repo,
                    publish=self.job.artifacts.publish)
        write_json(self.paths.root / "resolved.json", plan)
        return plan

    def _stage(self, stage, **extra):
        from datetime import datetime, timezone
        status = {"run_id": self.job.run_id, "kind": self.job.kind, "status": stage,
                  "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
        write_json(self.paths.root / "status.json", status)
        with (self.paths.root / "optimization" / "logs" / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(status) + "\n")

    def execute(self, *, publish: bool | None = None) -> dict:
        if publish and not self.job.artifacts.publish:
            raise ValueError("Set artifacts.publish=true explicitly before requesting upload")
        self.baseline()
        self.optimize()
        return self.evaluate(publish=publish)

    def baseline(self) -> dict:
        if self.planning_only:
            raise RuntimeError("Planning-only execution cannot call LMs; prepare again without --plan")
        if not self.optimizer.available():
            raise RuntimeError("Install the gepa optional dependencies first")
        if (self.paths.root / "baseline" / "program.json").exists() or (self.paths.root / "result.json").exists():
            raise ValueError("Prompt run already has execution artifacts; choose a new run_id (optimizer resume is not implemented)")
        self._stage("PREPARING")
        try:
            self.dry_run()
            self._steps = self.optimizer.steps(self.job, paths=self.paths, platform=self.platform, rows=self.rows, on_stage=self._stage)
            result = next(self._steps)
            self._phase = "baseline"
            return result
        except BaseException as error:
            self._stage("FAILED", error_type=type(error).__name__)
            raise

    def optimize(self) -> dict:
        return self._advance("baseline", "optimized")

    def _advance(self, expected, following):
        if getattr(self, "_phase", None) != expected:
            raise ValueError(f"Run the {expected} stage first; stages cannot be repeated")
        try:
            result = next(self._steps)
            self._phase = following
            return result
        except BaseException as error:
            self._stage("FAILED", error_type=type(error).__name__)
            self._phase = "failed"
            raise

    def evaluate(self, *, publish: bool | None = None) -> dict:
        if getattr(self, "_phase", None) != "optimized":
            raise ValueError("Run the optimize stage first; evaluation cannot be repeated")
        if publish and not self.job.artifacts.publish:
            raise ValueError("Set artifacts.publish=true explicitly before requesting upload")
        publish = self.job.artifacts.publish if publish is None else publish
        result = None
        try:
            result = self._advance("optimized", "completed")
            self._steps.close()
            result.update(status="COMPLETED", published=False)
            write_json(self.paths.root / "result.json", result)
            if publish:
                self._stage("PUBLISHING")
                store = HFArtifactStore(self.job.artifacts.hf_repo, get_secret("HF_TOKEN", self.platform, required=True))
                store.check_access()
                # Upload a consistent completed snapshot. The local status is
                # reverted to FAILED if the publication call fails.
                self._stage("COMPLETED", published=True)
                result["published"] = True
                write_json(self.paths.root / "result.json", result)
                store.publish_run(self.job, self.paths)
            else:
                self._stage("COMPLETED", published=False)
            return result
        except BaseException as error:
            if isinstance(result, dict):
                result.update(status="FAILED", published=False)
                write_json(self.paths.root / "result.json", result)
            self._stage("FAILED", error_type=type(error).__name__)
            raise


def prepare_prompt_execution(job, *, platform=None, hardware=None, output_dir=None, planning_only=False):
    job = apply_lm_environment(job)
    platform = platform or detect_platform()
    hardware = hardware or probe_hardware(platform)
    optimizer = resolve_optimizer(job, allow_unavailable=planning_only)
    paths = RunPaths.for_job(job, output_dir or platform.output_dir)
    existing = paths.root / "job.yaml"
    if existing.exists() and job_digest(load_job(existing)) != job_digest(job):
        raise ValueError("run_id already has a different JobSpec; choose a new run_id")
    rows = optimizer.prepare(job, paths=paths)
    paths.create()
    (paths.root / "job.yaml").write_text(yaml.safe_dump(job.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    write_json(paths.root / "hardware.json", hardware.to_dict())
    if not (paths.root / "status.json").exists():
        write_json(paths.root / "status.json", {"run_id": job.run_id, "status": "PLANNED", "kind": job.kind})
    return PromptExecution(job, platform, hardware, optimizer, paths, rows, planning_only)
