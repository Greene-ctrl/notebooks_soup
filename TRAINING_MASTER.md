# Training Master execution repository

This fork keeps the upstream Unsloth notebook collection as model-specific reference
material. Its new entrypoints are **[Colab](nb/Training_Master_Colab.ipynb)** and
**[Kaggle](nb/Kaggle-Training_Master.ipynb)**. They run the same JobSpec. The existing
OpenOperator / Agent Zero → Training Master → BackendResolver control plane remains
separate; this repository owns execution, not scheduling or job issuance.

```text
backend-neutral JobSpec
  → platform + hardware + runtime secret providers
  → kind=model_training → TrainingEngine → Soup | Unsloth | opt-in NeMo boundary
      → complete checkpoint restore → training → checkpoint/artifact hooks
  → kind=prompt_optimization → OptimizerEngine → DSPy GEPA
      → baseline validation → compile(train, validation) → held-out evaluation
      → portable JSON programs + metrics/statistics + optional artifact hooks
```

Colab and Kaggle are disposable compute. Their notebook survives; their disk and
processes may not. `runs/` is a cache and staging area, never authoritative state.
HF Hub is the optional durable model-repository adapter. A run without explicit
publication has **no durable recovery guarantee**. DGX Spark uses the same local
runner in its already configured environment; this implementation does not change
its system stack or launch containers.

## Install and first smoke run

Notebook maintenance retains the repository's `.python-version` (3.13) and `uv.lock`.
**Soup execution requires Python 3.10–3.12**, per its current metadata. Use a separate
Python 3.12 runtime/venv rather than bypassing its Python bound. The runner itself
supports Python >=3.10. Example local installation, from this checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'pydantic>=2,<3' 'pyyaml>=6'
python -m pip install --no-deps -e .
```

First inspect a CPU-safe plan without installing ML frameworks or fetching weights:

```bash
python -m training_master.runner examples/jobs/tiny-soup-sft.yaml \
  --plan --hardware-profile examples/hardware/t4.json --output-dir /tmp/training-master-plan
```

The profile is **illustrative**, not evidence of hardware availability. The CLI
refuses `--execute` with `--plan` or `--hardware-profile`. A successful plan validates
translation and requirements only, not a training installation or model fit.

For the first actual tiny training run, in a **CUDA GPU Python 3.12 runtime**:

```bash
python -c 'from training_master import load_job; from training_master.bootstrap import bootstrap; bootstrap(load_job("examples/jobs/tiny-soup-sft.yaml"), training=True)'
python -m training_master.runner examples/jobs/tiny-soup-sft.yaml --cli-preflight
python -m training_master.runner examples/jobs/tiny-soup-sft.yaml --execute
```

This uses public `HuggingFaceTB/SmolLM2-135M`, streams eight nonempty Wikitext rows,
sequence length 128, LoRA rank 8, batch size 1, accumulation 1, one epoch. It has at
most eight optimizer steps. It proves wiring, not model quality. Uploads default to
off. `tiny-unsloth-sft.yaml` uses public `Qwen/Qwen3-0.6B`, four steps, and QLoRA.
Use a new `run_id` for each independent experiment.

In a notebook, pip may replace a framework already imported for hardware probing.
The bootstrap detects changed installed versions and requires a kernel restart;
rerun the cells after restarting. It never continues with stale imported wheels.

## JobSpec v1

`training_master.job.JobSpec` uses Pydantic 2, rejects unknown keys at every level,
and accepts dict injection, local YAML/JSON, HTTPS JobSpec URLs, or
`hf://owner/repository/path/to/job.yaml` from a model repository. HTTP, credentials
in URL userinfo, query strings, and fragments are refused for URL JobSpecs. There
are no executable remote Python/config hooks. `TRAINING_MASTER_TOKEN` is restored
for future control-plane integration; no remote control-plane protocol is invented.

```yaml
version: 1
run_id: smoke-qwen-001
model:
  id: Qwen/Qwen3-0.6B
  revision: main
dataset:
  source: Salesforce/wikitext
  config: wikitext-2-raw-v1
  split: train
  format: plaintext
  text_field: text
  limit: 8
task:
  type: sft
engine:
  type: auto
  backend: transformers
training:
  epochs: 1
  batch_size: 1
  max_length: 128
  quantization: none
  gradient_accumulation_steps: 1
  gradient_checkpointing: true
  learning_rate: 0.0002
  seed: 42
  lora:
    enabled: true
    r: 8
    alpha: 16
    dropout: 0
    target_modules: [q_proj, v_proj]
artifacts:
  hf_repo: null
  checkpoint_interval: 2
requirements:
  min_vram_gb: 4
resume:
  mode: auto
```

The spec contains no backend paths, device IDs, platform install choices, or secrets.
`engine.backend` is optional and applies only to Soup (default `transformers`);
an explicit backend setting is rejected by direct Unsloth rather than ignored.
Execution paths/hardware/engine choice are recorded separately in `resolved.json`.
The minimum VRAM requirement checks the first visible GPU; two T4s do not become
one 30 GB device. `batch_size: auto` is passed to Soup's current implementation and
resolves conservatively to **1** in the direct Unsloth engine. RAM/disk readings and
a VRAM floor are not a model-size estimator; Soup performs its own preflight.

Both runnable adapters initially support **text causal-LM SFT with LoRA/QLoRA**.
Other tasks can be expressed in v1 but produce a clear unsupported-task error.
Dataset formats are `plaintext`, `alpaca`, and text-only `chatml`. HF datasets are
streamed using the requested config/split; the optional row limit counts nonempty
rows. Local input is JSONL, resolved relative to the caller's current directory;
local config/split overrides are rejected. Both engines receive the same normalized
rows. Direct Unsloth uses the model's chat template for alpaca/chatml. Soup uses its
own supported formatting. Their numerical outputs and tokenization are not promised
to be identical.

## Engine policy and upstream contracts

Explicit engine requests either work as requested or fail. `auto` considers a
supported, installed NeMo adapter for Nemotron first, then Soup, then direct
Unsloth. It never changes the task or drops a requested field. Offline `--plan`
allows a supported adapter whose dependencies are absent and labels the result as
planning only. An explicit engine can be bootstrapped without installing alternatives.

| Use case | Preferred engine |
|---|---|
| Generic SFT/LoRA | Soup |
| Kaggle/Colab QLoRA | Soup (`backend: transformers` or `unsloth`) |
| Model-specific Unsloth optimization | Direct Unsloth / preserved reference notebook |
| Nemotron model-specific tuning | NeMo AutoModel, provisioned and validated separately |
| Unsupported/custom experiment | Explicit custom path later |

### Soup

Inspected `MakazhanAlpamys/Soup` commit
`d5cb1a7e87a0042de27b83947f6948c3ca9456b7` (version **0.76.0**):
`examples/README.md`, SFT/DPO/GRPO/RLHF examples, config schema and loader,
backend support, install extras, doctor/train/data/adapters commands, and SFT
checkpoint/output behavior. Bootstrap pins that Git commit instead of inventing a
Python training API. Lightweight `soup-cli` is separate from its `[train]` extra;
`backend: unsloth` adds `[fast]` only for actual selected-engine installation.

Verified command contracts:

```bash
soup doctor --config runs/RUN_ID/soup.yaml
soup train --config runs/RUN_ID/soup.yaml --dry-run
soup train --config runs/RUN_ID/soup.yaml --yes
soup train --config runs/RUN_ID/soup.yaml --yes --resume PATH
soup data inspect PATH
soup adapters check PATH --json
```

The adapter emits only supported fields: `base`, `task`, `backend`, `data.train`,
`data.format`, `data.max_length`, `data.val_split`, the mapped training/LoRA settings,
and `output`. Both doctor and train's data dry run must succeed before execution.
Their output and exit statuses are preserved in `logs/soup-*.log` and `commands/*.json`. The offline dry-run
path performs JobSpec/config translation without downloading data or weights;
`--cli-preflight` additionally fetches the small selected dataset and may fetch model
configuration, but does not instantiate/download model weights.

**There is no `training.max_steps` in this Soup schema/CLI.** An explicit Soup job
requesting it fails; an auto job can resolve to direct Unsloth if supported. Soup's
smoke uses a bounded dataset and epochs. Do not assume it implements a strict step
limit. `model.revision` other than `main` is also refused by this Soup adapter because
there is no corresponding upstream config field. Exact upstream model/data revision
pinning is therefore a current limitation for Soup.

`training.stream_layers` is only emitted for transformers SFT, quantization `none`,
batch size 1, accumulation 1. Soup's own preflight still checks architecture, RAM,
disk and model compatibility. It is not an Unsloth streaming option.

### Direct Unsloth

`FastLanguageModel`, `get_peft_model`, `SFTTrainer/SFTConfig`, adapter/tokenizer saving,
and Trainer resume follow the upstream Qwen SFT references. Imports load Unsloth
before TRL/Transformers. The bootstrap pins Unsloth/Unsloth Zoo **2026.10.5**,
TRL **0.22.2**, Transformers **4.57.6**, and their tokenizer window. Dependency
metadata resolution was checked; no GPU import/training success is claimed. It
preserves an installed torch version to let pip choose compatible kernels instead
of resolving blindly to a CUDA 13 wheel; without torch it requests 2.10.0.
Unsloth bootstrap requires torch >=2.8 and pins torchao 0.16.0 below torch 2.11,
or 0.18.0 on newer torch, because torchao's dependency metadata omits that
compatibility relationship. Soup's transformers backend allows torch >=2.6. Actual
CUDA driver, wheel and hardware compatibility must pass on the target runtime.

Supports LoRA/4-bit QLoRA, `max_steps`, seed, sequence length, checkpoints and
same-engine Trainer resume. No DPO/GRPO/distillation, vision/audio, distributed
launcher, merged model/GGUF export or notebook-specific model overrides are added.
Use the preserved upstream notebooks for those reference paths.

### NeMo AutoModel boundary

Inspected `NVIDIA-NeMo/Nemotron` commit
`ca8c409f08a9a5a5648d427383adea741b61966a`: steps overview, AutoModel SFT/PEFT
readmes/configs, shared recipe runner, hardware guidance and current AutoModel
cookbook notebooks. Upstream steps consume chat JSONL and produce HF/LoRA
checkpoints. Their "tiny" configuration can still require two nodes × eight GPUs;
it is not a small Kaggle recipe.

`NemoAutoModelEngine` detects `nemo_automodel` and reports a conservative CUDA/bf16/
VRAM candidate hint. **Recipe translation and execution are not implemented.** Its
`supports` returns false, so auto cannot select a detection-only adapter as a working
engine. The resolver's preference order is ready for a future executable adapter.
A candidate hint is not a claim a large Nemotron model fits. No NVIDIA stack,
Megatron, or NeMo-RL is installed by default; no NVIDIA implementation code is copied.

## Colab and Kaggle bootloaders

Both notebooks share the same small code cells and default JobSpec. Set
`TRAINING_MASTER_REF` to a **published immutable fork commit SHA**. The notebook
checks out that commit; it cannot fetch this local unpublished branch yet.
`TRAINING_MASTER_CHECKOUT` can select an existing checkout for local development.
Source bootstrap installs only Pydantic/PyYAML. Heavy dependencies are selected after
job/hardware inspection, only with `RUN_TRAINING=True`.

Colab detects its runtime environment, uses `/content/runs`, and prefers
`google.colab.userdata.get`. Add `HF_TOKEN`, optionally `WANDB_API_KEY` and
`TRAINING_MASTER_TOKEN` to Secrets and grant notebook access. The runner does not
print secret values or call a login that saves them to tracked files.

Kaggle detects its runtime environment, uses `/kaggle/working/runs`, and prefers
`kaggle_secrets.UserSecretsClient().get_secret`. Attach those names in notebook
Secrets and enable Internet/GPU as needed for installation and model/dataset reads.
Both providers fall back to environment variables. W&B and control-plane tokens
are restored for integrations; W&B logging is intentionally disabled in initial SFT.

The notebooks default to `RUN_TRAINING=False` and `PUBLISH_ARTIFACTS=False`. Set the
first to true to execute; set an existing `artifacts.hf_repo` and the second to true
only when you authorize Hub uploads. Choose a Python 3.12 runtime for Soup. If the
host offers a different interpreter, use its supported runtime selection rather
than ignoring Soup's version requirements.

## Artifact and checkpoint contract

```text
runs/<run_id>/
  job.yaml
  resolved.json
  hardware.json
  status.json
  soup.yaml                 # Soup adapter only
  data.jsonl                # normalized selected rows, created for execution
  logs/
  checkpoints/trainer/checkpoint-N/
  metrics/
  final/
```

HF uses `runs/<run_id>/checkpoints/checkpoint-N/` and the remaining run metadata,
metrics and final artifacts under `runs/<run_id>/`. The runner uses an **existing
model repo** and does not silently create repositories. Normalized dataset rows
are staged locally and excluded from run uploads.
Soup exposes metrics through logs and saved Trainer checkpoint state; the runner
copies the latest available state into `metrics/` and records whether it is partial
in `availability.json`. Without a checkpoint/final Trainer-state file, only the CLI
log is available. Direct Unsloth additionally writes its final training metrics.
HF authentication comes from the
secret provider. Reads/restoration are possible when `hf_repo` is set; writes require
`execute(publish=True)` or CLI `--execute --publish`. No test requires live Hub writes.

`find_latest_checkpoint`, `restore_checkpoint`, and `publish_checkpoint` require a
completion manifest, job fingerprint, engine identity, numeric step and SHA-256
file hashes. Adapter weights alone are insufficient: Trainer state, optimizer,
scheduler, RNG and weights must be present. Partial uploads, corrupted checkpoints,
other jobs, and other engines are ignored for automatic selection. `resume: required`
fails if no valid compatible checkpoint is found; `never` skips restoration.

Unsloth seals checkpoints in Trainer's `on_save`. Soup CLI checkpoints are monitored
until their completed Trainer state is present, then sealed/uploaded; process exits
also trigger a scan. Each Hub upload commits files and manifest together. A runtime
may still vanish **before** a save/upload completes; only a successfully published
checkpoint can be durable. Upload failures stop execution and leave a safe failed
status and local recovery files. Hugging Face commits are immutable, but restoring
larger checkpoints costs network/disk time. Complete Trainer checkpoints can still
be incompatible across library versions or hardware layouts; no multi-node/sharded
checkpoint conversion is provided.

Contract tests simulate restoration on another platform. **Real Kaggle → Colab GPU
resume has not been tested and is not claimed.** Use the same run ID, job and engine;
changing the engine is a new run. Model weights/optimizer state from checkpoints
must be trusted because Trainer resume loads framework checkpoint formats.

## Development, tests and notebook synchronization

```bash
uv sync --frozen --extra test
uv run --extra test python -m pytest tests/test_training_master.py -q
uv run --extra test python scripts/build_training_master_notebooks.py
```

The builder writes only the two canonical notebooks and uses the existing
`convert_notebook_to_script` for their Python mirrors. They are excluded from model
announcement/install mutations and AMD/Kaggle variant generation. The fork links
sit outside generated upstream README tables. Other notebook content and license
notices are left untouched. CPU tests cover JobSpec validation, providers, structured
hardware, resolver/fallback policy, supported/unsupported Soup fields, paths, dataset
selection, checkpoint integrity, artifact hooks, safe statuses and notebook parity.
A fixture engine exercises orchestration without training a model.

The inherited CPU suite has existing failures in Qwen Tool Use/Tool Calling
consistency/pin detectors and generated-script mirrors. New work does not regenerate
or suppress those failures. The heavyweight ML API-drift matrix was excluded from
the CPU run; existing image URL checks were included.

Validation for this change (baseline `31a8d597f678f54ea343368713d8db880af008a7`):

- Full CPU run: **15,410 passed, 569 skipped, 18 failed** with
  `python -m pytest tests --ignore=tests/test_notebook_api_drift.py -q --tb=short -p no:cacheprovider`.
  The 18 failing test IDs exactly match the pre-change baseline; they concern
  inherited Qwen generated-source parity and vLLM pin/selector detectors.
- Final runner contracts: **67 passed** on Python 3.13 and **67 passed** on Python 3.12.
- Both new notebooks validate as nbformat 4 JSON and match their generated scripts.
- The packaged CLI planning pipeline and generated YAML were checked against the
  actual Soup 0.76.0 config loader, doctor and current CLI help contracts.
- Dependency metadata resolution, lockfile consistency, runner Ruff checks and
  the repository's dynamic-execution gate passed.
- No GPU model load/training, real Colab/Kaggle launch, real cross-backend resume,
  or Hub upload was performed. The separate ML API-drift matrix was not run.

## Licensing and scope

Upstream notices and `LICENSE` remain intact. These new notebooks use
LGPL-3.0-or-later and acknowledge upstream Unsloth reference material. The Soup
adapter interoperates with its Apache-2.0 CLI/config contract rather than copying
its internal trainers. NVIDIA recipe references are attributed above; no recipes
are copied. This change does not push a branch, open a PR, train on paid GPU quota,
or upload artifacts automatically.

## Prompt optimization with DSPy GEPA

Prompt optimization is a first-class workflow, separate from model training. GEPA
implements `OptimizerEngine`; it does **not** subclass `TrainingEngine`, use Soup,
or require LoRA, CUDA or a locally loaded model. The same prompt JobSpec can run
on Colab CPU, Kaggle CPU, or a local/persistent machine using remote LMs.

| Workflow | Engine |
|---|---|
| SFT/LoRA | Soup |
| Optimized Unsloth training | Direct Unsloth |
| Nemotron-specific training | NeMo AutoModel boundary / separately validated NVIDIA recipe |
| Prompt optimization | DSPy GEPA |

### Current API verified

The adapter pins **DSPy 3.4.1**, whose GEPA dependency is **`gepa[dspy]==0.1.4`**.
Verified the installed release's `dspy.GEPA`, `compile`, `dspy.LM`, `Prediction`,
`Module.save/load`, program registry constructors, and `DspyGEPAResult.to_dict`.
The official getting-started documentation was inspected in the DSPy source at
`stanfordnlp/dspy` commit `1af633fecbb2b8acec7e3ed9acd6db1cde6d341c`, alongside
`docs/docs/getting-started/gepa-optimization.md`, the GEPA adapter and its tests.
The public `dspy.ai` URL was blocked by the cloud proxy; the release source and
installed API were checked directly instead of relying on remembered APIs.

The verified API accepts:

```python
optimizer = dspy.GEPA(
    metric=metric, reflection_lm=reflection_lm,
    max_metric_calls=16,  # OR auto="light"/"medium"/"heavy", never both
    num_threads=1, track_stats=True,
    reflection_minibatch_size=2, log_dir=None,
)
optimized = optimizer.compile(program, trainset=trainset, valset=valset)
optimized.save("program.json", save_program=False)
```

Exactly one budget is mandatory. The adapter deliberately keeps `log_dir=None`:
GEPA 0.1.4's on-disk optimizer checkpoint format uses pickle even though portable
DSPy program state can be JSON. Instead, the runner saves JSON program state,
candidates, statistics, usage counters and its own status event log. It does not
vendor DSPy or implement another optimizer. GEPA runtime support is Python
3.10–3.14; its dependency is omitted upstream on Python 3.15, so bootstrap rejects
that unsupported combination. Python 3.12 is recommended for this smoke workflow.

### Workflow JobSpecs and backward compatibility

`load_job` now dispatches a discriminated `WorkflowJobSpec`:

- `kind: model_training` uses the original `JobSpec` class and training resolver.
- `kind: prompt_optimization` uses `PromptJobSpec` and `resolve_optimizer`.
- Old jobs without `kind` retain their model-training interpretation. Training
  checkpoint digests explicitly exclude the new default field, preserving existing
  checkpoint identity rather than invalidating old manifests.

Unknown settings remain errors. Prompt jobs reject training/LoRA settings;
training jobs reject student/reflection/program/optimizer fields. A minimal prompt
job looks like this; see the complete [synthetic smoke](examples/jobs/tiny-gepa-prompt.yaml):

```yaml
version: 1
kind: prompt_optimization
run_id: ux-gepa-smoke
program:
  type: predict
  signature: "question -> answer"
student:
  model: openai/gpt-4.1-mini
  api_base: null
  api_key_secret: STUDENT_API_KEY
  temperature: 0
  max_tokens: 128
reflection:
  model: openai/gpt-4.1-mini
  api_base: null
  api_key_secret: REFLECTION_API_KEY
  temperature: 1
  max_tokens: 768
optimizer:
  type: gepa
  max_metric_calls: 16
  num_threads: 1
  track_stats: true
dataset:
  train: [{question: "Train question", answer: "gold"}]
  validation: [{question: "Validation question", answer: "gold"}]
  test: [{question: "Held-out question", answer: "gold"}]
metric:
  type: exact_match_with_feedback
artifacts:
  hf_repo: null
  publish: false
```

The explicit program registry supports `predict`, `chain_of_thought` and `react`.
Initial signatures are untyped text inputs with one text output; no executable
signature strings or arbitrary imports. ReAct defaults to three iterations and
an empty tool list (DSPy's built-in finish behavior). Named tools are a future
trusted-code registry boundary; unknown names fail instead of loading code.

### Student versus reflection LM and provider configuration

The student answers task examples. The reflection LM critiques optimization
traces and proposes instructions. Each has independent model, endpoint, secret
name, temperature and token cap. There is no global `dspy.configure` mutation;
short-lived `dspy.context` scopes end between staged notebook calls.

Use DSPy's `provider/model` convention. For OpenAI-compatible gateways, use
`openai/<served-model-name>` with the gateway's `/v1` base URL. The same interface
supports OpenAI cloud, FreeLLM/gateways, Helmholtz endpoints, local Gemma services,
Nemotron services and other OpenAI-compatible deployments. Model-specific API and
context-window compatibility still depends on the chosen provider.

These environment variables override the corresponding role independently:

```text
STUDENT_MODEL          STUDENT_API_BASE          STUDENT_API_KEY
REFLECTION_MODEL       REFLECTION_API_BASE       REFLECTION_API_KEY
```

For example, set `STUDENT_MODEL=openai/gemma` and
`STUDENT_API_BASE=http://localhost:8000/v1` while keeping reflection on a cloud
model, or set a different `REFLECTION_API_BASE` for a strong remote endpoint.
Credential values belong in Colab/Kaggle Secrets or environment settings, never
in YAML, notebook cells, URLs, result files or the command history. Existing
`get_secret` prefers the platform provider and falls back to environment variables.
The LM factory resolves only requested key names at execution; HF credentials are
not requested unless publication is enabled. A keyless explicit endpoint may omit
`api_key_secret`; authenticated endpoints should always name the required binding.

DSPy imposes special limits on reasoning families such as `openai/gpt-5.4`: the
caller must explicitly choose `max_tokens>=16000` and temperature `null` or `1`.
The adapter refuses an incompatible small cap rather than silently increasing it.
The smoke uses cheaper non-reasoning models with small explicit caps. Custom LM
providers and endpoint behavior remain unverified against live services here.

### Datasets, metrics and held-out isolation

Each split accepts inline example objects, a local JSON array, or local JSONL.
Relative paths are resolved from the caller's current directory. An optional
HF-backed source is supported with a bounded streaming read:

```yaml
train:
  type: hf
  id: owner/dataset
  config: optional-config
  split: train
  limit: 10
```

Only HF sources need the optional `datasets` package; bootstrap adds it when one
is selected. Inline/local datasets need neither datasets nor pandas. Rows must
contain string values for the signature's inputs and output. Extra source columns
are not DSPy inputs. Input duplicates across train/validation/test are rejected,
even when labels differ. Examples are normalized centrally using
`dspy.Example(...).with_inputs(...)`; output labels never become input fields.

Metric factories are an explicit extensible registry:

- `exact_match`: numeric score after trimming whitespace and ignoring case.
- `exact_match_with_feedback`: the same score as
  `dspy.Prediction(score=..., feedback=...)`, including the expected/received answer
  on mismatch. It accepts GEPA's five metric arguments and makes no extra LM calls.
- `contains`: whether a nonempty normalized gold answer is contained in prediction.

Baseline validation is evaluated before compilation. GEPA sees **only train and
validation**. Both baseline-test and optimized-test evaluations happen **after**
compilation, keeping held-out data away from proposal generation and candidate
selection. The result stores both comparisons and `delta_test`. The synthetic job
has 10 train, 4 validation and 4 test examples; it checks orchestration, not quality.

### Budgets, execution and usage

Install only the selected CPU optimizer dependencies:

```bash
python -m pip install -e '.[gepa,test]'
python -m training_master.runner examples/jobs/tiny-gepa-prompt.yaml --plan
```

Planning requires no API keys, makes no LM calls and needs no GPU/hardware profile.
The report shows both endpoint/model settings, token caps, split sizes, metric,
threads and scheduling budget. It also reports the additional baseline/final
validation/test evaluations: with four validation and four test rows, these add
16 metric evaluations beyond the optimization budget. For auto presets, the
estimate calls the installed DSPy `auto_budget` helper. `light` is already 396
optimization metric calls for a one-predictor program with four validation rows;
therefore the smoke uses **16 explicit metric calls**, not an implicit preset.

GEPA's metric-call cap is a **scheduling budget**, not a hard provider-call/dollar
quota: in-flight batches can overshoot; reflection calls, adapter formatting
retries and ReAct iterations add LM requests. Provider retries are set to zero,
and the LM cache is disabled. The runner persists observed student/reflection
history-entry counts and token usage, plus actual GEPA metric-call statistics
when `track_stats` is enabled. SDK histories may be truncated. No dollar price or
unavailable usage is invented.

After inspecting the plan and configuring the two secret bindings, actual API
usage is explicitly enabled by:

```bash
python -m training_master.runner examples/jobs/tiny-gepa-prompt.yaml --execute
```

Use a new `run_id` for every independent run. There is no optimizer-state resume
in this implementation. A lost runtime can recover saved optimized JSON programs,
but it cannot resume a partially completed GEPA search. Existing training
checkpoint/resume behavior is unchanged.

### Colab CPU and Kaggle CPU notebook

Open [Prompt_Optimization_GEPA.ipynb](nb/Prompt_Optimization_GEPA.ipynb).

1. Supply this GEPA-capable checkout or set `TRAINING_MASTER_REF` to its published
   immutable fork commit. These changes are local until you publish them yourself;
   the previously pushed training-only commit cannot bootstrap the GEPA package.
2. Choose a CPU Python 3.12 runtime. On Kaggle enable Internet for dependency
   installation and provider requests. GPU acceleration is optional.
3. Add `STUDENT_API_KEY` and `REFLECTION_API_KEY` in the platform's secret settings;
   grant/attach access to the notebook. Use environment variables for endpoint/model
   overrides as needed. Do not add HF credentials for the default local-only run.
4. Run source/bootstrap/JobSpec/Resolve cells with `RUN_OPTIMIZATION=False` first.
   Review endpoint destinations, token caps and the small metric-call budget.
5. Set `RUN_OPTIMIZATION=True` and rerun Resolve, Baseline, Optimize and Evaluate.
   The cells call reusable `PromptExecution.baseline()`, `.optimize()` and
   `.evaluate()`; the CLI's `.execute()` drives the same stages in one call.
6. Inspect Artifacts. To rerun, change `run_id`. If dependency installation changes
   an imported module, restart the kernel before running the execution cells.

### Prompt artifacts and statuses

```text
runs/<run_id>/
  job.yaml, hardware.json, resolved.json, status.json
  baseline/program.json, baseline/metrics.json
  optimization/stats.json, optimization/candidates.json, optimization/usage.json
  optimization/logs/events.jsonl
  optimized/program.json, optimized/metrics.json
  test/baseline.json, test/optimized.json
  result.json
```

Program JSON is saved through DSPy's state-only `save(..., save_program=False)`
after detaching bound LMs. Load it with
`training_master.programs.load_program(job.program, path, lm=student_lm)`, which
rebuilds the registered program and loads JSON with pickle and unsafe LM-state
loading disabled. The program definition lives in `job.yaml`; no Python program
pickle or credentials are embedded.

`status.json` and JSONL events expose `PREPARING`, `BASELINE_EVAL`, `OPTIMIZING`,
`TESTING`, optional `PUBLISHING`, `COMPLETED`, or `FAILED`. Failure records contain
the exception type, not provider response bodies/keys. These files can be polled
by a future OpenOperator integration. Local files remain disposable staging state.

Optional HF synchronization reuses the existing model-repository adapter and
`runs/<run_id>/...` layout. **`artifacts.publish: false` is the default**. An upload
requires explicit `artifacts.publish: true` and an existing `hf_repo`; CLI
`--publish` alone cannot override a job that forbids publication. Tests use fake Hub
clients only. No artifacts were uploaded during development.

### Future GEPA → fine-tune → GEPA loop

A later workflow may export optimized prompts and successful traces, fine-tune a
model through Soup/Unsloth, then run GEPA again against the adapted student. DSPy's
BetterTogether remains a relevant reference for combined prompt/weight optimization.
It is documentation only here: no BetterTogether implementation, automatic training
handoff, trace export contract or new model-training engine is added.

### GEPA verification performed

Against the preceding Training Master baseline `d20518f72485dae2d4f73cfb21e0ccfb9b8fc8a6`:

- Prompt unit tests + real DSPy runtime contracts + original Training Master tests:
  **128 passed** on Python 3.12. Real DSPy uses local `DummyLM`s; network connections
  are prohibited in the runtime tests. The real search produces no claimed quality
  improvement on the scripted perfect-answer fixture.
- Prompt/training/CI-coverage contracts on the default Python 3.13 environment:
  **224 passed**. The CPU-only CLI plan also works without installing DSPy or
  providing API credentials.
- Full inherited CPU suite: **15,488 passed, 571 skipped, 18 failed**. All 18 failing
  test IDs exactly match the pre-existing Qwen pin/selector and generated-script
  parity failures. Optional real-DSPy tests are exercised separately above; the
  heavyweight training API-drift matrix was not run.
- GEPA notebook JSON/schema, generated-script parity, source Ruff/narrow lint,
  compilation, dynamic-execution checks and frozen lockfile consistency passed.
- No real student/reflection endpoint, paid API optimization, GPU run, HF upload,
  Git push or external publication was performed. JSON recovery was tested locally;
  partially completed optimizer searches are not resumable.

Usage history counters are observed SDK history, not a complete billing ledger:
histories can be truncated or belong to deep-copied programs. GEPA's own metric-call
statistics are saved separately. Live endpoint compatibility, real optimization
quality and real provider token/cost totals remain unverified.

## Training notebook diagnostics and deterministic resources

Training resources resolve against `repo_root` (the runner checkout by default),
not the notebook working directory or the YAML file's directory. Pass an explicit
`repo_root=ROOT` for another checkout. `resolve_runtime_path` canonicalizes absolute
paths, `./`, `../`, `~/`, `file://`, known local file suffixes, and existing entries
under that root. A missing explicit local resource fails during preparation;
ambiguous `owner/name` strings remain Hub IDs. Use `./models/name` for a missing
local model directory so it cannot be mistaken for a Hub ID. HTTPS and `hf://`
identifiers stay remote; credentials/query strings in resource URLs are rejected.
The current training schema supports one dataset source; unsupported split paths,
custom reward/metric scripts and checkpoint path overrides remain schema errors.

The roots have distinct purposes:

| Root | Meaning |
| --- | --- |
| `repo_root` | Source checkout and base for local resources |
| `runtime_root` | Platform working directory (`/content`, `/kaggle/working`, or local root) |
| `run_root` | Absolute `<output root>/<run_id>` |
| `engine_workdir` | Explicit `run_root` cwd for Soup commands |
| `output_dir` | Absolute Trainer output, `run_root/checkpoints/trainer` |

Relative output overrides resolve against `runtime_root`. Generated configs contain
absolute local model/data/output paths. Dataset rows normalize into the absolute
`run_root/data.jsonl`. The original JobSpec remains in `job.yaml` and retains its
checkpoint digest; resolved engine inputs and roots are in `resolved.json`.
`environment.json` records Python/executable, source SHA, package versions, CUDA
and GPU metadata without environment-variable dumps.

`prepare()` writes engine files once. `dry_run()` validates the generated config;
CLI preflight materializes data, runs Soup doctor, then `train --dry-run`.
`Execution.validate()` performs that preflight for an executable Soup run.
`execute()` repeats validation before training, and `SoupEngine.run()` reuses the
validated config/data fingerprint. Direct engine calls also require preflight.
No training is launched after a failed check. Offline planning is explicitly
unverified for dataset/doctor checks and cannot execute training.

```python
job = load_job(ROOT / "examples/jobs/tiny-soup-sft.yaml")
execution = prepare_execution(job, repo_root=ROOT)
execution.show_preflight(publish=False)
execution.validate()
result = execution.execute(publish=False)
# Independently callable after a failed execution cell:
execution.show_diagnostics()
```

If preparation itself fails before assigning `execution`, use
`training_master.diagnostics.show_run_diagnostics(run_root)`. Diagnostics include
status/stage, paths, engine version, config, latest command/exit code, a bounded
log tail, checkpoint directories and final artifact state.

Soup stdout/stderr are merged in order, streamed live, and persisted in
`logs/soup-doctor.log`, `logs/soup-dry-run.log`, and `logs/soup-train.log`.
Matching `commands/*.json` files record exact argv, explicit cwd, timestamps,
duration, exit code and interruption state. Retries get numbered filenames,
preserving previous attempts. `EngineProcessError` includes the command, code,
log path and last 100 lines (also bounded to 24,000 characters). Configure
`SoupEngine(tail_lines=80)` or the reusable `run_process` tail limits as needed.
`stderr_tail` is empty because the combined ordered output is in `stdout_tail`.
Known environment secret values and common provider token forms are redacted
before persistence/display. Never put credentials into jobs or command arguments.
Ctrl+C terminates the child process group on POSIX, escalates to kill if necessary,
and persists command/failure state before re-raising. Complete logs/configs and
existing checkpoints remain available after failure.

The smoke job now uses eight bundled synthetic plaintext rows rather than a
remote dataset. Its public `SmolLM2-135M` model still requires a first model/tokenizer
download for actual training. It uses LoRA, one epoch, batch size 1, max length 128,
no W&B and no publication. Soup defaults to **4bit**: the adapter explicitly writes
`quantization: none` to honor an unquantized job; omitting that field would enable
quantization on GPU. Soup 0.76.0 has no `--version` flag, so bootstrap prints its
installed distribution version and verifies the executable belongs to the kernel's
environment. Python 3.10–3.12 is required for these training bootloaders.
