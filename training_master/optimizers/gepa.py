"""Thin adapter for DSPy 3.4.1 / GEPA 0.1.4; remote LMs are optional."""
from __future__ import annotations

import importlib.metadata
import os
import sys
import tempfile
from pathlib import Path

from ..artifacts import RunPaths, write_json
from ..job import PromptJobSpec
from ..metrics import evaluate, resolve_metric
from ..programs import build_program, save_program, signature_fields
from ..programs.registry import TOOL_REGISTRY
from .base import OptimizerEngine
from .data import dspy_examples, load_splits
from .lm import create_lm, lm_usage, validate_lm

DSPY_VERSION = "3.4.1"
GEPA_VERSION = "0.1.4"


def dspy_runtime():
    os.environ.setdefault("DSPY_CACHEDIR", str(Path(tempfile.gettempdir()) / "training-master-dspy"))
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import dspy
    return dspy


class GEPAOptimizer(OptimizerEngine):
    name = "gepa"

    def available(self) -> bool:
        try:
            return (3, 10) <= sys.version_info[:2] < (3, 15) and (
                importlib.metadata.version("dspy") == DSPY_VERSION
                and importlib.metadata.version("gepa") == GEPA_VERSION
            )
        except importlib.metadata.PackageNotFoundError:
            return False

    def supports(self, job) -> bool:
        return isinstance(job, PromptJobSpec) and job.optimizer.type == "gepa"

    def prepare(self, job, *, paths=None, rows=None) -> dict:
        if not self.supports(job):
            raise ValueError("GEPA accepts only kind=prompt_optimization, optimizer.type=gepa")
        validate_lm(job.student)
        validate_lm(job.reflection)
        if set(job.program.tools) - TOOL_REGISTRY.keys():
            raise ValueError("Unknown program tool registry entry")
        rows = rows if rows is not None else load_splits(job)
        if job.optimizer.max_metric_calls is not None and job.optimizer.max_metric_calls < len(rows["validation"]):
            raise ValueError("optimizer.max_metric_calls must cover the initial full validation evaluation")
        if paths:
            for name in ("baseline", "optimization/logs", "optimized", "test"):
                (paths.root / name).mkdir(parents=True, exist_ok=True)
        return rows

    def configuration(self, job) -> dict:
        cfg = {"auto": job.optimizer.budget, "max_metric_calls": job.optimizer.max_metric_calls,
               "num_threads": job.optimizer.num_threads, "track_stats": job.optimizer.track_stats,
               "seed": job.optimizer.seed, "reflection_minibatch_size": 2,
               "use_wandb": False, "use_mlflow": False, "log_dir": None}
        # log_dir=None avoids GEPA's implicit pickle checkpoint files. The runner
        # writes its own portable JSON stats/candidates/events instead.
        return cfg

    def dry_run(self, job, *, paths=None, rows=None) -> dict:
        rows = self.prepare(job, paths=paths, rows=rows)
        budget = job.optimizer.max_metric_calls
        if budget is None and self.available():
            dspy = dspy_runtime()
            from dspy.teleprompt.gepa.gepa import AUTO_RUN_SETTINGS
            program = build_program(job.program, dspy_module=dspy)
            budget = dspy.GEPA.auto_budget(
                None, num_preds=max(len(program.named_predictors()), 1),
                num_candidates=AUTO_RUN_SETTINGS[job.optimizer.budget]["n"], valset_size=len(rows["validation"]),
            )
        evaluation_examples = 2 * (len(rows["validation"]) + len(rows["test"]))
        return {
            "kind": job.kind, "optimizer": self.name, "program": job.program.model_dump(),
            "student": job.student.model_dump(), "reflection": job.reflection.model_dump(),
            "metric": job.metric.type, "dataset_sizes": {k: len(v) for k, v in rows.items()},
            "budget_mode": job.optimizer.budget or "explicit_metric_calls", "max_metric_calls": budget,
            "num_threads": job.optimizer.num_threads,
            "additional_evaluation_examples": evaluation_examples,
            "estimated_metric_evaluations": budget + evaluation_examples if budget is not None else None,
            "budget_note": "GEPA scheduling budget; in-flight batches can overshoot. Reflection, retries and ReAct steps add LM calls. Not a hard API/dollar quota.",
            "required_secrets": sorted({s.api_key_secret for s in (job.student, job.reflection) if s.api_key_secret}),
            "configuration": self.configuration(job), "api_calls_made": False,
            "dspy_version": DSPY_VERSION, "gepa_version": GEPA_VERSION,
        }

    def run(self, job, **kwargs) -> dict:
        result = None
        for result in self.steps(job, **kwargs):
            pass
        return result

    def steps(self, job, *, paths: RunPaths, platform=None, rows=None, on_stage=None):
        rows = self.prepare(job, paths=paths, rows=rows)
        dspy = dspy_runtime()
        stage = on_stage or (lambda value: None)
        # Both LMs are constructed/validated before the first paid request.
        student = create_lm(job.student, platform=platform)
        reflection = create_lm(job.reflection, platform=platform)
        program = build_program(job.program, dspy_module=dspy)
        program.set_lm(student)
        _, output = signature_fields(job.program.signature)
        metric = resolve_metric(job.metric, output_field=output)
        trainset = dspy_examples(rows["train"], job.program.signature, dspy_module=dspy)
        valset = dspy_examples(rows["validation"], job.program.signature, dspy_module=dspy)
        # Test rows are NOT converted/passed to compile or evaluated before it.
        stage("BASELINE_EVAL")
        save_program(program, paths.root / "baseline" / "program.json")
        with dspy.context(lm=student):
            baseline_validation = evaluate(program, valset, metric)
        write_json(paths.root / "baseline" / "metrics.json", {"validation": baseline_validation})
        baseline_program = program.deepcopy()
        yield {"baseline_validation": baseline_validation["score"]}
        stage("OPTIMIZING")
        optimizer = dspy.GEPA(metric=metric, reflection_lm=reflection, **self.configuration(job))
        with dspy.context(lm=student):
            optimized = optimizer.compile(program, trainset=trainset, valset=valset)
        save_program(optimized, paths.root / "optimized" / "program.json")
        self._save_stats(optimized, paths)
        yield {"stats_path": str(paths.root / "optimization" / "stats.json")}
        stage("TESTING")
        with dspy.context(lm=student):
            optimized_validation = evaluate(optimized, valset, metric)
        testset = dspy_examples(rows["test"], job.program.signature, dspy_module=dspy)
        # Baseline test is intentionally deferred too: test data stays out of
        # optimizer selection, reflection and the entire compile lifecycle.
        with dspy.context(lm=student):
            baseline_test = evaluate(baseline_program, testset, metric)
            optimized_test = evaluate(optimized, testset, metric)
        write_json(paths.root / "test" / "baseline.json", baseline_test)
        write_json(paths.root / "test" / "optimized.json", optimized_test)
        write_json(paths.root / "baseline" / "metrics.json", {"validation": baseline_validation, "test": baseline_test})
        write_json(paths.root / "optimized" / "metrics.json", {"validation": optimized_validation, "test": optimized_test})
        write_json(paths.root / "optimization" / "usage.json", {"student": lm_usage(student), "reflection": lm_usage(reflection)})
        result = {
            "run_id": job.run_id, "kind": job.kind, "optimizer": self.name,
            "baseline_validation": baseline_validation["score"], "optimized_validation": optimized_validation["score"],
            "baseline_test": baseline_test["score"], "optimized_test": optimized_test["score"],
            "delta_test": optimized_test["score"] - baseline_test["score"],
            "artifact_path": str(paths.root), "hf_repo": job.artifacts.hf_repo,
        }
        write_json(paths.root / "result.json", result)
        yield result

    def _save_stats(self, optimized, paths):
        details = getattr(optimized, "detailed_results", None)
        raw = details.to_dict() if details is not None else {}
        # Keep portable instruction candidates and numerical lineage/statistics;
        # never serialize candidate Module objects, LM histories or credentials.
        write_json(paths.root / "optimization" / "candidates.json", raw.get("candidates", []))
        keys = ("parents", "val_aggregate_scores", "val_subscores", "discovery_eval_counts", "total_metric_calls", "num_full_val_evals", "seed", "best_idx")
        write_json(paths.root / "optimization" / "stats.json", {"available": details is not None, **{k: raw[k] for k in keys if k in raw}})
