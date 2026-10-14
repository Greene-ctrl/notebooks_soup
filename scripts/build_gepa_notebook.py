"""Build the CPU GEPA entrypoint; reuse the existing source bootstrap/exporter."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_training_master_notebooks import BOOTSTRAP  # noqa: E402
from update_all_notebooks import convert_notebook_to_script  # noqa: E402


def build():
    md = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    cells = [
        md('# Training Master — DSPy GEPA\n\nCPU-compatible prompt optimization with independent student and reflection endpoints. GPU is optional for API-backed models. No model training stack is installed.\n\nSPDX-License-Identifier: LGPL-3.0-or-later. Existing Unsloth reference notebooks and attribution remain preserved.'),
        md('## Runner source\nUse a checkout containing the GEPA implementation, or set `TRAINING_MASTER_REF` to its published immutable commit. See `TRAINING_MASTER.md`; the older training-only commit does not contain GEPA.'),
        code(BOOTSTRAP.rstrip()),
        md('## 1 — Runtime\nGPU is optional. The LMs may run on remote OpenAI-compatible APIs while this notebook runs on Colab/Kaggle CPU.'),
        code('import json\nfrom training_master.platform import detect_platform\nfrom training_master.hardware import probe_hardware\nruntime = detect_platform()\nhardware = probe_hardware(runtime)\nprint(json.dumps({"python": sys.version.split()[0], "cpu_count": os.cpu_count(), **hardware.to_dict()}, indent=2))'),
        md('## 2 — Secrets\nUse Colab Secrets, Kaggle Secrets, or environment variables for `STUDENT_API_KEY` and `REFLECTION_API_KEY`. `HF_TOKEN` is needed only for explicitly requested uploads. Keys are retrieved lazily at execution, never printed. A keyless local endpoint may omit `api_key_secret`.'),
        code('RUN_OPTIMIZATION = False  # Explicitly opt in to LM/API usage.\nprint("No keys are loaded during planning. The execution stages load only their required secrets.")'),
        md('## 3 — Bootstrap\nInstall the fork package and pinned DSPy CPU dependencies. No Soup, Unsloth or NeMo installation. If pip changes modules already imported in this kernel, restart it before continuing.'),
        code('subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(ROOT)], check=True)\nsubprocess.run([sys.executable, "-m", "pip", "install", "dspy==3.4.1"], check=True)'),
        md('## 4 — JobSpec\nThe example has ten train, four validation and four held-out test examples. Its explicit GEPA scheduling budget is 16 metric calls; evaluation/reflection adds usage. Override `STUDENT_MODEL`, `STUDENT_API_BASE`, `REFLECTION_MODEL`, and `REFLECTION_API_BASE` in the environment or edit this dict. Uploads default to off.'),
        code('from training_master import load_job\nfrom training_master.runner import prepare_execution\n\nJOB_SPEC = load_job(ROOT / "examples/jobs/tiny-gepa-prompt.yaml").model_dump(mode="json")\nJOB_SPEC["run_id"] = "gepa-notebook-smoke-001"  # New ID for each independent optimization.\n# Edit JOB_SPEC here, or replace it with a local/remote JobSpec source.\njob = load_job(JOB_SPEC)'),
        md('## 5 — Resolve\nInspect student/reflection endpoints and token caps, optimizer, metric, dataset sizes, thread count, and estimated metric budget. Metric budget is not a hard API-call or dollar quota.'),
        code('execution = prepare_execution(job, platform=runtime, hardware=hardware, planning_only=not RUN_OPTIMIZATION)\nprint(json.dumps(execution.dry_run(), indent=2))'),
        md('## 6 — Baseline\nEvaluate only the starting program on validation. Both baseline and optimized test scores are deferred until GEPA compilation has finished.'),
        code('if RUN_OPTIMIZATION:\n    print(execution.baseline())\nelse:\n    print("Planning only. Set RUN_OPTIMIZATION=True and rerun Resolve to enable LM calls.")'),
        md('## 7 — Optimize\nGEPA receives train and validation examples only. Stop/restart with a new run ID after a lost runtime; optimizer checkpoint resume is not implemented.'),
        code('if RUN_OPTIMIZATION:\n    print(execution.optimize())'),
        md('## 8 — Evaluate\nEvaluate optimized validation, then both programs on the held-out test set. Publication requires explicit `artifacts.publish=True` and an HF target; the smoke has both disabled.'),
        code('if RUN_OPTIMIZATION:\n    result = execution.evaluate()\nelse:\n    result = {"run_id": job.run_id, "status": "NOT_EXECUTED"}\nprint(json.dumps(result, indent=2))'),
        md('## 9 — Artifacts\nPortable JSON programs, baseline/optimized metrics, held-out results, GEPA statistics/candidates and safe status events live under the run folder. No pickle is enabled; no provider credentials are saved.'),
        code('print(json.dumps({"local_run": str(execution.paths.root),\n                  "optimized_program": str(execution.paths.root / "optimized/program.json"),\n                  "hf_repo": execution.job.artifacts.hf_repo,\n                  "publish": execution.job.artifacts.publish}, indent=2))'),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'},
    })
    for index, cell in enumerate(notebook.cells):
        cell.id = f'gepa-{index:02d}'
    target = ROOT / 'nb/Prompt_Optimization_GEPA.ipynb'
    nbformat.write(notebook, target)
    convert_notebook_to_script(str(target), str(ROOT / 'python_scripts/Prompt_Optimization_GEPA.py'))


if __name__ == '__main__':
    build()
