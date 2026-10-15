# Training notebook hardening validation

Branch: `training-master-runner`. This work adds one focused commit after the
existing model-training and GEPA commits. Nothing is pushed by this task.

## Findings and fixes

The supplied `CalledProcessError` does not contain the original Soup stderr;
the exact cause of the original Colab **training** failure cannot be established
without that run's log. The old smoke used a remote Wikitext dataset. Its generated
Soup `data.train` and `output` paths were already absolute through `RunPaths`;
a relative Soup dataset path is **not** a proven cause of that particular failure.
Confirmed runtime defects were:

- Local dataset lookup used `Path(source).is_file()` against notebook cwd. A missing
  local resource could silently fall through to the Hub dataset loader.
- Soup stdout/stderr were redirected into `logs/soup.log` without live output;
  subprocess failure exposed only the wrapper exit status.
- Soup cwd was derived indirectly from two parent directories. Commands, durations,
  environment provenance and resolved resource roots were not fully recorded.
- Config generation was repeated across prepare/dry-run/run, and the notebook had
  no concise after-failure diagnostic helper.
- A shell `soup` on PATH could differ from the kernel's installed distribution.

Resources now resolve against the explicit checkout, all engine file paths are
absolute, and Soup runs with an explicit recorded `run_root` cwd. Remote IDs/URLs
are preserved. The original JobSpec/digest remains the checkpoint identity;
the separate resolved job is passed to engines. The bundled synthetic dataset
removes the remote dataset dependency from the smoke job.

Soup 0.76.0 was inspected at its pinned source commit
`d5cb1a7e87a0042de27b83947f6948c3ca9456b7` and exercised as an installed CLI.
Its supported checks are `doctor --config` and `train --config --dry-run`.
The actual CLI has **no `--version` option**; we report distribution metadata.
Its schema defaults to **4bit quantization**, so `quantization: none` must be
explicit for this unquantized job. Omitting it on GPU changes the requested job.

## Files changed

- `training_master/paths.py`: shared local resource resolution and resolved jobs.
- `training_master/process.py`: live combined UTF-8 output, full sanitized logs,
  bounded error tails, structured results/errors, command records and cancellation.
- `training_master/diagnostics.py`: environment provenance and run diagnostics.
- `training_master/runner.py`: resolved manifests, prepare/validate/execute stages,
  safe terminal failure state and concise notebook helpers.
- `training_master/engines/base.py`: deterministic direct dataset materialization.
- `training_master/engines/soup.py`: pinned executable, explicit cwd, one config,
  mandatory doctor/data preflight, validated config/data fingerprint and retry logs.
- `training_master/bootstrap.py`: verify the kernel's scripts directory and version.
- `examples/data/plaintext_tiny.jsonl`, `examples/jobs/tiny-soup-sft.yaml`: eight
  synthetic rows, public 135M model, no quantization/W&B/publication.
- `scripts/build_training_master_notebooks.py`, both canonical training `.ipynb`
  files and their `python_scripts/` mirrors: predictable source roots, immutable
  SHA checking, source/interpreter/engine provenance, summary, checks and diagnostics.
- `tests/test_training_master.py`, `tests/test_training_diagnostics.py`,
  `tests/test_soup_validation.py`, `.github/workflows/notebooks-tests-ci.yml`:
  new CPU regression coverage and pinned Soup core validation in CI.
- `TRAINING_MASTER.md` and this report: semantics, operational guidance and evidence.

GEPA modules, jobs, notebook and script remain unchanged. Shared-runner regression
checks include the real installed DSPy runtime with mocked LMs; no live API calls.

## Execution evidence

Actual machine: Linux, Python 3.12.15 for Soup, Soup 0.76.0, no GPU, no torch.
For the unmodified GPU smoke JobSpec, a recorded 15 GB profile is used **only for
planning/config generation**. Soup doctor and data dry-run run as real CPU child
processes and probe the actual machine. The planning object cannot start training.
Doctor exits 0 and correctly advises that the optional training stack is absent;
this is a core/config/data validation success, not GPU or training readiness.

| Check | Result |
| --- | --- |
| Current smoke JobSpec load through real runner | PASS |
| Source and generated local paths absolute, cwd independent | PASS |
| Current Soup schema loader accepts generated config | PASS |
| Real Soup doctor on generated config | PASS, exit 0 |
| Real Soup data dry-run | PASS, exit 0; 8 train samples |
| Real Soup bad-data failure exposes useful engine error/log | PASS |
| Child exit 1, long output, Unicode/invalid UTF-8, Ctrl+C | PASS |
| Real GPU training / model download | NOT EXECUTED |
| Colab/Kaggle live runtime | NOT EXECUTED |
| Artifact publication | NOT EXECUTED |

Persisted local validation run: `/tmp/tm-hardening-validation/tiny-soup-sft-001`.
Its resolved source is
`/workspace/notebooks_soup/examples/data/plaintext_tiny.jsonl`.
Exact generated Soup config used for successful CPU validation:

```yaml
base: HuggingFaceTB/SmolLM2-135M
task: sft
backend: transformers
data:
  train: /tmp/tm-hardening-validation/tiny-soup-sft-001/data.jsonl
  format: plaintext
  max_length: 128
  val_split: 0.0
training:
  epochs: 1
  lr: 0.0002
  batch_size: 1
  gradient_accumulation_steps: 1
  gradient_checkpointing: true
  quantization: none
  stream_layers: false
  seed: 42
  data_seed: 42
  save_steps: 2
  logging_steps: 1
  lora:
    r: 8
    alpha: 16
    dropout: 0.0
    target_modules:
    - q_proj
    - v_proj
output: /tmp/tm-hardening-validation/tiny-soup-sft-001/checkpoints/trainer
```

No unsupported `max_steps` or invented Soup settings are generated. Quantization
is only enabled for an explicit `4bit` job; explicit `none` disables Soup's default.

## Tests

- Targeted training/diagnostics/GEPA contracts plus notebook structural checks:
  **1,916 passed** (Python 3.13).
- Pinned real Soup schema/doctor/data/failure validation plus diagnostic tests:
  **22 passed** (Python 3.12, Soup core only).
- Training/diagnostics/GEPA with actual DSPy 3.4.1 and mocked LMs:
  **148 passed** (Python 3.12; no provider network calls).
- Broad CPU suite, same API-drift exclusion as baseline:
  **18 failed, 15,512 passed, 572 skipped**. All 18 failed test IDs exactly match
  `/tmp/gepa-full-cpu-tests.log`: inherited Qwen vLLM pin/selector and generated
  script parity defects. **Zero new failures.**

Commands/evidence logs in this workspace:

```text
/workspace/.venvs/notebooks_soup/bin/python -m pytest tests/test_training_master.py tests/test_training_diagnostics.py tests/test_prompt_optimization.py tests/test_notebook_json_validity.py -q --tb=short
  /tmp/tm-hardening-final-targeted.log
PATH=/workspace/.venvs/training-master-contract/bin:$PATH /workspace/.venvs/training-master-contract/bin/python -m pytest tests/test_soup_validation.py tests/test_training_diagnostics.py -q --tb=short
  /tmp/tm-hardening-soup.log
/workspace/.venvs/training-master-gepa/bin/python -m pytest tests/test_training_master.py tests/test_training_diagnostics.py tests/test_prompt_optimization.py tests/test_gepa_runtime.py -q --tb=short
  /tmp/tm-hardening-gepa.log
/workspace/.venvs/notebooks_soup/bin/python -m pytest tests --ignore=tests/test_notebook_api_drift.py -q --tb=short -p no:cacheprovider
  /tmp/tm-hardening-final-full-tests.log
```

Ruff's repository gate and full Ruff on changed runner/test/generator modules,
compileall, dynamic-execution lint and its self-test pass. No repository type checker
is configured. Notebook structural tests and live script-export parity pass for
both canonical training notebooks. The heavyweight upstream API-drift matrix
(`tests/test_notebook_api_drift.py`) requires a separate ML stack and was not run;
the broad CPU command uses the same exclusion as the recorded baseline.

## Rerun in Colab

1. Publish the new commit when desired; this task deliberately leaves it local.
   Use the full new SHA reported in chat as `TRAINING_MASTER_REF` after publication.
   The previous published SHA does not contain these fixes.
2. Start a fresh Colab runtime with a T4 GPU and Python 3.12. Open the updated
   `nb/Training_Master_Colab.ipynb`. A fresh runtime avoids old checkout SHA and
   old JobSpec/run-ID conflicts. Kaggle uses the corresponding canonical notebook,
   Internet enabled, GPU selected and `/kaggle/working` storage.
3. In the source cell set `REPO_REF` to the full published SHA (or set
   `TRAINING_MASTER_REF` before the cell). Leave `TRAINING_MASTER_CHECKOUT` unset
   on Colab/Kaggle. Confirm the printed source SHA and Python executable/version.
   Source roots are `/content/training-master-source` and
   `/kaggle/working/training-master-source` respectively.
4. Run Runtime information and Secrets. The public smoke needs no HF/W&B secret.
   Confirm working CUDA and the GPU's VRAM.
5. Leave `PUBLISH_ARTIFACTS=False`. The Job cell loads the repository's actual
   `examples/jobs/tiny-soup-sft.yaml`, identically on both platforms. For a reused
   runtime with previous run artifacts, choose a new `run_id` before Resolve.
6. Set `RUN_TRAINING=True` in the Bootstrap settings cell. Run Job, Resolve and
   Dry run. Only the selected engine extras install. If bootstrap reports changed
   already-imported frameworks, restart the kernel and rerun the cells.
7. Confirm the expected kernel's Soup executable, Soup 0.76.0, source SHA,
   absolute dataset/config/output roots and disabled publishing. Require PASS on
   path, dataset, engine and Soup doctor checks before running Execute.
8. Run Execute. Read live output. On failure, run the independent Diagnostics cell
   (`execution.show_diagnostics()`) and retain its output plus the named full log.
   If preparation failed before assignment, use
   `from training_master.diagnostics import show_run_diagnostics` and call it with
   the displayed absolute run root. No directory browsing is required.
9. Inspect the Artifacts result after success. A local checkpoint remains ephemeral
   until explicitly published to a configured existing HF repository.

Remaining risks: live hosted package/CUDA combinations and first model/tokenizer
network access still require the Colab/Kaggle test. Small-model training and actual
GPU checkpoint/resume were not exercised here. Doctor's optional-stack notices
are advisory, so a CPU doctor PASS alone never establishes training readiness.
A fresh run ID is required if an old run has a different JobSpec. Existing upstream
notebook pin/parity failures remain outside this change.
