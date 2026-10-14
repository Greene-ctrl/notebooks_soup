#!/usr/bin/env python
# coding: utf-8

# # Training Master — DSPy GEPA
# 
# CPU-compatible prompt optimization with independent student and reflection endpoints. GPU is optional for API-backed models. No model training stack is installed.
# 
# SPDX-License-Identifier: LGPL-3.0-or-later. Existing Unsloth reference notebooks and attribution remain preserved.

# ## Runner source
# Use a checkout containing the GEPA implementation, or set `TRAINING_MASTER_REF` to its published immutable commit. See `TRAINING_MASTER.md`; the older training-only commit does not contain GEPA.

# In[ ]:


# Use a published immutable commit, or the existing checkout for local testing.
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/JsonLord/notebooks_soup.git"
REPO_REF = os.environ.get("TRAINING_MASTER_REF", "SET_TO_PUBLISHED_COMMIT_SHA")
ROOT = Path(os.environ.get("TRAINING_MASTER_CHECKOUT", str(Path.cwd()))).resolve()
if not (ROOT / "training_master" / "__init__.py").is_file():
    if len(REPO_REF) != 40 or any(c not in "0123456789abcdef" for c in REPO_REF):
        raise ValueError("Set TRAINING_MASTER_REF to the published runner commit SHA")
    base = Path("/kaggle/working") if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else Path("/content") if Path("/content").is_dir() else Path.cwd()
    ROOT = base / "training-master-source"
    if not ROOT.exists():
        subprocess.run(["git", "init", str(ROOT)], check=True)
        subprocess.run(["git", "-C", str(ROOT), "remote", "add", "origin", REPO_URL], check=True)
        subprocess.run(["git", "-C", str(ROOT), "fetch", "--depth", "1", "origin", REPO_REF], check=True)
        subprocess.run(["git", "-C", str(ROOT), "checkout", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if actual != REPO_REF:
        raise ValueError("Existing runner checkout differs; use a fresh source directory")
# Lightweight runner requirements only; training extras are selected later.
subprocess.run([sys.executable, "-m", "pip", "install", "pydantic>=2,<3", "pyyaml>=6"], check=True)
sys.path.insert(0, str(ROOT))
if "training_master" in sys.modules:
    loaded_root = Path(sys.modules["training_master"].__file__).resolve().parents[1]
    if loaded_root != ROOT:
        raise RuntimeError("Runner source changed in this kernel; restart and rerun")


# ## 1 — Runtime
# GPU is optional. The LMs may run on remote OpenAI-compatible APIs while this notebook runs on Colab/Kaggle CPU.

# In[ ]:


import json
from training_master.platform import detect_platform
from training_master.hardware import probe_hardware
runtime = detect_platform()
hardware = probe_hardware(runtime)
print(json.dumps({"python": sys.version.split()[0], "cpu_count": os.cpu_count(), **hardware.to_dict()}, indent=2))


# ## 2 — Secrets
# Use Colab Secrets, Kaggle Secrets, or environment variables for `STUDENT_API_KEY` and `REFLECTION_API_KEY`. `HF_TOKEN` is needed only for explicitly requested uploads. Keys are retrieved lazily at execution, never printed. A keyless local endpoint may omit `api_key_secret`.

# In[ ]:


RUN_OPTIMIZATION = False  # Explicitly opt in to LM/API usage.
print("No keys are loaded during planning. The execution stages load only their required secrets.")


# ## 3 — Bootstrap
# Install the fork package and pinned DSPy CPU dependencies. No Soup, Unsloth or NeMo installation. If pip changes modules already imported in this kernel, restart it before continuing.

# In[ ]:


subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(ROOT)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "dspy==3.4.1"], check=True)


# ## 4 — JobSpec
# The example has ten train, four validation and four held-out test examples. Its explicit GEPA scheduling budget is 16 metric calls; evaluation/reflection adds usage. Override `STUDENT_MODEL`, `STUDENT_API_BASE`, `REFLECTION_MODEL`, and `REFLECTION_API_BASE` in the environment or edit this dict. Uploads default to off.

# In[ ]:


from training_master import load_job
from training_master.runner import prepare_execution

JOB_SPEC = load_job(ROOT / "examples/jobs/tiny-gepa-prompt.yaml").model_dump(mode="json")
JOB_SPEC["run_id"] = "gepa-notebook-smoke-001"  # New ID for each independent optimization.
# Edit JOB_SPEC here, or replace it with a local/remote JobSpec source.
job = load_job(JOB_SPEC)


# ## 5 — Resolve
# Inspect student/reflection endpoints and token caps, optimizer, metric, dataset sizes, thread count, and estimated metric budget. Metric budget is not a hard API-call or dollar quota.

# In[ ]:


execution = prepare_execution(job, platform=runtime, hardware=hardware, planning_only=not RUN_OPTIMIZATION)
print(json.dumps(execution.dry_run(), indent=2))


# ## 6 — Baseline
# Evaluate only the starting program on validation. Both baseline and optimized test scores are deferred until GEPA compilation has finished.

# In[ ]:


if RUN_OPTIMIZATION:
    print(execution.baseline())
else:
    print("Planning only. Set RUN_OPTIMIZATION=True and rerun Resolve to enable LM calls.")


# ## 7 — Optimize
# GEPA receives train and validation examples only. Stop/restart with a new run ID after a lost runtime; optimizer checkpoint resume is not implemented.

# In[ ]:


if RUN_OPTIMIZATION:
    print(execution.optimize())


# ## 8 — Evaluate
# Evaluate optimized validation, then both programs on the held-out test set. Publication requires explicit `artifacts.publish=True` and an HF target; the smoke has both disabled.

# In[ ]:


if RUN_OPTIMIZATION:
    result = execution.evaluate()
else:
    result = {"run_id": job.run_id, "status": "NOT_EXECUTED"}
print(json.dumps(result, indent=2))


# ## 9 — Artifacts
# Portable JSON programs, baseline/optimized metrics, held-out results, GEPA statistics/candidates and safe status events live under the run folder. No pickle is enabled; no provider credentials are saved.

# In[ ]:


print(json.dumps({"local_run": str(execution.paths.root),
                  "optimized_program": str(execution.paths.root / "optimized/program.json"),
                  "hf_repo": execution.job.artifacts.hf_repo,
                  "publish": execution.job.artifacts.publish}, indent=2))

