"""Build only the two short bootloaders using the existing Python exporter."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from update_all_notebooks import convert_notebook_to_script  # noqa: E402


BOOTSTRAP = '''# Use a published immutable commit, or the existing checkout for local testing.
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/JsonLord/notebooks_soup.git"
REPO_REF = os.environ.get("TRAINING_MASTER_REF", "SET_TO_PUBLISHED_COMMIT_SHA")
local_checkout = os.environ.get("TRAINING_MASTER_CHECKOUT")
base = Path("/kaggle/working") if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE") else Path("/content") if Path("/content").is_dir() else Path.cwd()
ROOT = Path(local_checkout).resolve() if local_checkout else base / "training-master-source"
if local_checkout:
    if not (ROOT / "training_master" / "__init__.py").is_file():
        raise ValueError("TRAINING_MASTER_CHECKOUT must contain Training Master")
else:
    if len(REPO_REF) != 40 or any(c not in "0123456789abcdef" for c in REPO_REF):
        raise ValueError("Set TRAINING_MASTER_REF to the published runner commit SHA")
    if not ROOT.exists():
        subprocess.run(["git", "init", str(ROOT)], check=True)
        subprocess.run(["git", "-C", str(ROOT), "remote", "add", "origin", REPO_URL], check=True)
        subprocess.run(["git", "-C", str(ROOT), "fetch", "--depth", "1", "origin", REPO_REF], check=True)
        subprocess.run(["git", "-C", str(ROOT), "checkout", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if actual != REPO_REF:
        raise ValueError("Existing runner checkout differs; use a fresh source directory")
actual = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
print("Runner SHA:", actual)
print("Python executable:", sys.executable)
print("Python version:", sys.version.split()[0])
# Lightweight runner requirements only; training extras are selected later.
subprocess.run([sys.executable, "-m", "pip", "install", "pydantic>=2,<3", "pyyaml>=6"], check=True)
sys.path.insert(0, str(ROOT))
if "training_master" in sys.modules:
    loaded_root = Path(sys.modules["training_master"].__file__).resolve().parents[1]
    if loaded_root != ROOT:
        raise RuntimeError("Runner source changed in this kernel; restart and rerun")
'''


def build() -> None:
    for name, platform in (("Training_Master_Colab", "Colab"), ("Kaggle-Training_Master", "Kaggle")):
        cells = [
            nbformat.v4.new_markdown_cell(f"# Training Master — {platform}\n\nDisposable compute, one backend-neutral JobSpec. Local files are caches; enable explicit HF publication for durable checkpoints. See `TRAINING_MASTER.md`.\n\nSPDX-License-Identifier: LGPL-3.0-or-later. Upstream Unsloth notebooks and their copyright/attribution remain preserved as model-specific references."),
            nbformat.v4.new_markdown_cell("## Runner source\nPin a published fork commit in `TRAINING_MASTER_REF`. A local `TRAINING_MASTER_CHECKOUT` can use the existing checkout. This initial import bootstrap installs no ML framework."),
            nbformat.v4.new_code_cell((BOOTSTRAP + 'if not (3, 10) <= sys.version_info[:2] < (3, 13):\n    raise RuntimeError("Model-training bootstrap requires Python 3.10–3.12; select a Python 3.12 runtime")\n').rstrip()),
            nbformat.v4.new_markdown_cell("## 1 — Runtime information\nGPU names, per-device VRAM, RAM, free disk, CUDA and platform. Multiple GPUs do not imply pooled VRAM."),
            nbformat.v4.new_code_cell('import json\nfrom training_master.platform import detect_platform\nfrom training_master.hardware import probe_hardware\n\nruntime = detect_platform()\nhardware = probe_hardware(runtime)\nprint(json.dumps(hardware.to_dict(), indent=2))'),
            nbformat.v4.new_markdown_cell("## 2 — Secrets\nColab: grant notebook access to Secrets. Kaggle: attach secrets using Add-ons → Secrets. Environment variables are the fallback. Only presence is displayed."),
            nbformat.v4.new_code_cell('from training_master.secrets import restore_secrets\nprint(restore_secrets(runtime))'),
            nbformat.v4.new_markdown_cell("## 3 — Bootstrap Training Master runner\nSelect only the job's engine. Default bootstrap is Soup's lightweight CLI; training extras are installed only when `RUN_TRAINING=True` below. Soup requires Python 3.10–3.12. NeMo is an opt-in preconfigured environment."),
            nbformat.v4.new_code_cell('from training_master.bootstrap import bootstrap, verify_engine_environment\nfrom training_master.job import load_job\nfrom training_master.runner import prepare_execution\n\nRUN_TRAINING = False\nPUBLISH_ARTIFACTS = False'),
            nbformat.v4.new_markdown_cell("## 4 — Job\nThe default uses the bundled tiny dataset. Set `JOB_SPEC_PATH`/`JOB_SPEC_URL`/`HF_JOB_SPEC` to load another YAML/JSON job. OpenOperator can inject the same JobSpec. Never place tokens in it."),
            nbformat.v4.new_code_cell('JOB_SPEC_PATH = ROOT / "examples/jobs/tiny-soup-sft.yaml"\nJOB_SPEC_URL = None\nHF_JOB_SPEC = None  # hf://owner/repository/path/to/job.yaml\njob = load_job(JOB_SPEC_URL or HF_JOB_SPEC or JOB_SPEC_PATH)'),
            nbformat.v4.new_markdown_cell("## 5 — Resolve\nInspect the requested/resolved engine and requirements before fetching training dependencies. This first pass is planning only; it cannot execute training."),
            nbformat.v4.new_code_cell('execution = prepare_execution(job, platform=runtime, hardware=hardware, planning_only=True, repo_root=ROOT)\nexecution.show_preflight(publish=PUBLISH_ARTIFACTS)'),
            nbformat.v4.new_markdown_cell("## 6 — Dry run\nInstall only the selected engine. Offline config validation downloads no model weights. When training is enabled, this cell runs Soup doctor and data preflight before the execution cell. Full logs and exact commands are saved under the run root. Restart the kernel if installation changes frameworks already imported in this runtime."),
            nbformat.v4.new_code_cell('bootstrap(job, training=RUN_TRAINING)\nverify_engine_environment(job)\nhardware = probe_hardware(runtime)\nexecution = prepare_execution(job, platform=runtime, hardware=hardware, planning_only=not RUN_TRAINING, repo_root=ROOT)\nexecution.show_preflight(publish=PUBLISH_ARTIFACTS)\nexecution.validate()'),
            nbformat.v4.new_markdown_cell("## 7 — Execute\nOpt in with `RUN_TRAINING=True`, then rerun the bootstrap/resolve cells. `PUBLISH_ARTIFACTS=True` additionally authorizes uploads to the existing `artifacts.hf_repo`. Checkpoints publish as they complete; uploads default to off."),
            nbformat.v4.new_code_cell('if RUN_TRAINING:\n    result = execution.execute(publish=PUBLISH_ARTIFACTS)\nelse:\n    result = {"run_id": job.run_id, "status": "not_executed",\n              "checkpoint_path": str(execution.paths.checkpoints),\n              "hf_repo": job.artifacts.hf_repo,\n              "metrics_path": str(execution.paths.root / "metrics")}'),
            nbformat.v4.new_markdown_cell("## 8 — Artifacts\nA runtime may disappear after publishing a complete checkpoint. Resume uses the same JobSpec/run ID and engine; real cross-backend GPU resume has not been validated."),
            nbformat.v4.new_code_cell('print(json.dumps(result, indent=2))'),
            nbformat.v4.new_markdown_cell("## Diagnostics\nRun this cell after a failure to inspect status, resolved paths, environment, config, last command, log tail and checkpoints."),
            nbformat.v4.new_code_cell('execution.show_diagnostics()'),
        ]
        nb = nbformat.v4.new_notebook(cells=cells, metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        })
        for index, cell in enumerate(nb.cells):
            cell.id = f"tm-{index:02d}"
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        target = ROOT / "nb" / f"{name}.ipynb"
        nbformat.write(nb, target)
        convert_notebook_to_script(str(target), str(ROOT / "python_scripts" / f"{name}.py"))


if __name__ == "__main__":
    build()
