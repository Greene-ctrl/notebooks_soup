"""CPU-safe execution contracts: no model downloads, GPU work or Hub writes."""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import nbformat
import pytest
import yaml
from pydantic import ValidationError

from training_master.artifacts import HFArtifactStore, RunPaths, write_json
from training_master.bootstrap import install_command, selected_bootstrap_engine
from training_master.checkpoint import (
    IncompleteCheckpoint, MANIFEST, find_latest_checkpoint,
    publish_checkpoint, restore_checkpoint, seal_checkpoint, validate_checkpoint,
)
from training_master.engines import (
    NemoAutoModelEngine, SoupEngine, UnsupportedJob, resolve_engine,
)
from training_master.engines.base import TrainingEngine, materialize_dataset
from training_master.hardware import HardwareProfile, probe_hardware
from training_master.job import JobSpec, load_job
from training_master.platform import RuntimePlatform, detect_platform
from training_master.runner import main, prepare_execution
from training_master.secrets import get_secret, restore_secrets

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw_job():
    return {"version": 1, "run_id": "test-run", "model": {"id": "HuggingFaceTB/SmolLM2-135M"}, "dataset": {"source": "public/data", "limit": 8}, "training": {"batch_size": 1}}


@pytest.fixture
def job(raw_job):
    return JobSpec.model_validate(raw_job)


@pytest.fixture
def gpu():
    return HardwareProfile(platform="kaggle", gpu_count=2, gpu_names=["Tesla T4", "Tesla T4"], gpu_vram_gb=[15, 15], cuda_available=True, torch_version="2.6.0", cuda_version="12.4", disk_free_gb=50, ram_gb=30)


@pytest.mark.parametrize("mutation", [
    {"run_id": "../escape"}, {"version": 2}, {"unexpected": 1},
    {"training": {"max_steps": 0}}, {"training": {"batch_size": True}},
    {"training": {"max_length": 0}}, {"training": {"lora": {"r": -1}}},
    {"training": {"lora": {"target_modules": []}}},
    {"dataset": {"source": "x", "limit": -1}},
    {"artifacts": {"hf_repo": "../../outside"}},
    {"engine": {"type": "custom"}}, {"HF_TOKEN": "test-secret"},
])
def test_job_rejects_bad_or_unknown_fields(raw_job, mutation):
    raw_job.update(mutation)
    with pytest.raises(ValidationError):
        JobSpec.model_validate(raw_job)


@pytest.mark.parametrize("suffix", ["yaml", "json"])
def test_job_round_trip(job, tmp_path, suffix):
    path = tmp_path / f"job.{suffix}"
    path.write_text(yaml.safe_dump(job.model_dump()) if suffix == "yaml" else job.model_dump_json())
    assert load_job(path) == job
    assert load_job(job) is job
    assert load_job(job.model_dump()) == job


@pytest.mark.parametrize("url", ["http://example.org/job", "https://user:password@example.org/job", "https://example.org/job?token=secret"])
def test_job_urls_cannot_carry_credentials(url):
    with pytest.raises(ValueError, match="HTTPS"):
        load_job(url)


@pytest.mark.parametrize("env,name,provider", [
    ({"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, "kaggle", "kaggle"),
    ({"COLAB_RELEASE_TAG": "release"}, "colab", "colab"),
    ({"COLAB_GPU": "0"}, "colab", "colab"), ({}, "local", "environment"),
])
def test_platform_detection(env, name, provider):
    platform = detect_platform(env)
    assert platform.name == name and platform.secrets_provider == provider
    assert platform.persistent_notebook == (name != "local")
    assert platform.output_dir == platform.working_dir / "runs"


def test_cpu_probe_without_torch_or_driver(monkeypatch, tmp_path):
    monkeypatch.setattr("training_master.hardware.subprocess.run", Mock(side_effect=FileNotFoundError))
    monkeypatch.setattr("training_master.hardware.importlib.import_module", Mock(side_effect=ImportError))
    platform = RuntimePlatform("local", tmp_path, tmp_path / "runs", "environment", False)
    result = probe_hardware(platform)
    assert result.gpu_count == 0 and not result.cuda_available
    assert result.disk_free_gb > 0
    assert json.loads(json.dumps(result.to_dict()))["gpu_names"] == []


def test_torch_probe_records_each_gpu(monkeypatch, tmp_path):
    fake = types.SimpleNamespace(__version__="2.6", version=types.SimpleNamespace(cuda="12.4"), cuda=types.SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 2, get_device_name=lambda i: f"T4-{i}",
        get_device_properties=lambda i: types.SimpleNamespace(total_memory=15 * 1024**3), is_bf16_supported=lambda: False,
    ))
    monkeypatch.setattr("training_master.hardware.importlib.import_module", lambda name: fake)
    result = probe_hardware(RuntimePlatform("kaggle", tmp_path, tmp_path / "runs", "kaggle", True))
    assert result.gpu_vram_gb == [15, 15] and result.gpu_count == 2
    assert result.cuda_available and not result.bf16_supported


def test_driver_only_is_not_torch_cuda_readiness(monkeypatch, tmp_path):
    monkeypatch.setattr("training_master.hardware.subprocess.run", lambda *a, **k: types.SimpleNamespace(stdout="Tesla T4, 15360\n"))
    monkeypatch.setattr("training_master.hardware.importlib.import_module", Mock(side_effect=ImportError))
    result = probe_hardware(RuntimePlatform("kaggle", tmp_path, tmp_path / "runs", "kaggle", True))
    assert result.gpu_count == 1 and result.gpu_vram_gb == [15]
    assert not result.cuda_available


@pytest.mark.parametrize("provider", ["colab", "kaggle"])
def test_secret_provider_and_environment_fallback(monkeypatch, tmp_path, provider, capsys):
    monkeypatch.setenv("HF_TOKEN", "env-secret")
    platform = RuntimePlatform(provider, tmp_path, tmp_path / "runs", provider, True)
    if provider == "colab":
        module = types.ModuleType("google.colab")
        module.userdata = types.SimpleNamespace(get=lambda name: "provider-secret")
        monkeypatch.setitem(sys.modules, "google.colab", module)
    else:
        module = types.ModuleType("kaggle_secrets")
        module.UserSecretsClient = lambda: types.SimpleNamespace(get_secret=lambda name: "provider-secret")
        monkeypatch.setitem(sys.modules, "kaggle_secrets", module)
    assert get_secret("HF_TOKEN", platform) == "provider-secret"
    if provider == "colab":
        module.userdata.get = Mock(side_effect=RuntimeError("denied"))
    else:
        module.UserSecretsClient = Mock(side_effect=RuntimeError("denied"))
    assert get_secret("HF_TOKEN", platform) == "env-secret"
    assert restore_secrets(platform)["HF_TOKEN"] is True
    assert capsys.readouterr().out == ""


def test_missing_required_secret(monkeypatch):
    monkeypatch.delenv("TRAINING_MASTER_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="Missing TRAINING_MASTER_TOKEN"):
        get_secret("TRAINING_MASTER_TOKEN", detect_platform({}), required=True)


def test_soup_config_maps_all_supported_options(job, tmp_path):
    paths = RunPaths.for_job(job, tmp_path)
    config = SoupEngine().config(job, paths)
    assert config["base"] == job.model.id
    assert config["task"] == "sft" and config["backend"] == "transformers"
    assert config["data"]["max_length"] == 512 and config["data"]["val_split"] == 0
    assert config["training"]["lora"]["r"] == 16
    assert config["training"]["seed"] == 42
    assert config["training"]["save_steps"] == 10
    assert "max_steps" not in config["training"]
    assert config["output"] == str(paths.trainer)


@pytest.mark.parametrize("mutation,field", [
    ({"training": {"max_steps": 10}}, "max_steps"),
    ({"model": {"id": "x", "revision": "pinned"}}, "model.revision"),
    ({"task": {"type": "grpo"}}, "task.type"),
    ({"training": {"lora": {"enabled": False}}}, "lora.enabled"),
    ({"training": {"stream_layers": True, "batch_size": 2}}, "stream_layers"),
    ({"dataset": {"source": "x", "format": "chatml", "text_field": "other"}}, "text_field"),
])
def test_soup_rejects_unsupported_fields(raw_job, tmp_path, mutation, field):
    raw_job.update(mutation)
    with pytest.raises(UnsupportedJob, match=field):
        SoupEngine().config(load_job(raw_job), RunPaths.for_job(load_job(raw_job), tmp_path))


def test_supported_soup_unsloth_backend_and_streaming(raw_job, tmp_path):
    raw_job["engine"] = {"type": "soup", "backend": "unsloth"}
    raw_job["training"] = {"quantization": "4bit"}
    job = load_job(raw_job)
    assert SoupEngine().config(job, RunPaths.for_job(job, tmp_path))["backend"] == "unsloth"
    raw_job["engine"] = {"type": "soup"}
    raw_job["training"] = {"stream_layers": True, "batch_size": 1}
    job = load_job(raw_job)
    assert SoupEngine().config(job, RunPaths.for_job(job, tmp_path))["training"]["stream_layers"]


class FakeEngine(TrainingEngine):
    name = "soup"

    def __init__(self, available=True, errors=(), fail=False):
        self._available, self.errors, self.fail = available, list(errors), fail
        self.received_resume = None

    def available(self):
        return self._available

    def unsupported_fields(self, job):
        return self.errors

    def run(self, job, hardware, paths, *, resume=None, on_checkpoint=None):
        self.received_resume = resume
        checkpoint = make_checkpoint(paths.trainer / "checkpoint-2", 2)
        on_checkpoint(checkpoint)
        if self.fail:
            raise RuntimeError("failure-with-sensitive-details")
        (paths.final / "adapter_model.safetensors").write_text("fake-weights")
        return {"engine": self.name}


def test_engine_resolution_default_explicit_fallback_and_missing(job, gpu):
    soup, unsloth = FakeEngine(), FakeEngine()
    unsloth.name = "unsloth"
    registry = {"soup": soup, "unsloth": unsloth}
    assert resolve_engine(job, gpu, engines=registry) is soup
    soup.errors = ["unsupported field"]
    assert resolve_engine(job, gpu, engines=registry) is unsloth
    explicit = load_job({**job.model_dump(), "engine": {"type": "soup"}})
    with pytest.raises(UnsupportedJob, match="unsupported field"):
        resolve_engine(explicit, gpu, engines=registry)
    soup.errors = []
    soup._available = unsloth._available = False
    with pytest.raises(UnsupportedJob, match="missing"):
        resolve_engine(job, gpu, engines=registry)
    assert resolve_engine(job, gpu, engines=registry, allow_unavailable=True) is soup


def test_auto_max_steps_falls_back_without_changing_task(raw_job, gpu):
    raw_job["training"]["max_steps"] = 4
    job = load_job(raw_job)
    assert resolve_engine(job, gpu, allow_unavailable=True).name == "unsloth"
    assert selected_bootstrap_engine(job) == "unsloth"
    assert job.task.type == "sft"


def test_unsupported_task_never_silently_becomes_sft(raw_job, gpu):
    raw_job["task"] = {"type": "dpo"}
    with pytest.raises(UnsupportedJob, match="dpo"):
        resolve_engine(load_job(raw_job), gpu, allow_unavailable=True)


def test_vram_is_not_pooled(raw_job, gpu):
    raw_job["requirements"] = {"min_vram_gb": 20}
    with pytest.raises(UnsupportedJob, match="not pooled"):
        resolve_engine(load_job(raw_job), gpu, allow_unavailable=True)


def test_nemo_detection_boundary_does_not_advertise_execution(raw_job, gpu, monkeypatch):
    raw_job["model"] = {"id": "nvidia/Nemotron-3-Nano-30B-A3B"}
    job = load_job(raw_job)
    engine = NemoAutoModelEngine()
    monkeypatch.setattr(engine, "available", lambda: True)
    assert not engine.supports(job, gpu)
    assert not engine.dry_run(job, gpu)["executable"]
    assert not engine.candidate(job, gpu)
    powerful = HardwareProfile(gpu_count=1, gpu_vram_gb=[80], cuda_available=True, bf16_supported=True)
    assert engine.candidate(job, powerful)
    assert resolve_engine(job, powerful, engines={"nemo_automodel": engine, "soup": FakeEngine()}).name == "soup"
    with pytest.raises(UnsupportedJob, match="not implemented"):
        engine.run(job, powerful, None)


def test_bootstrap_only_installs_selected_engine(job, monkeypatch):
    monkeypatch.setattr("training_master.bootstrap.sys.version_info", (3, 12))
    core = install_command(job)
    train = install_command(job, training=True)
    assert "soup-cli @ git+" in core[-1] and any("soup-cli[train]" in argument for argument in train)
    assert "nemo" not in str(train) and "unsloth" not in str(train)
    monkeypatch.setattr("training_master.bootstrap.sys.version_info", (3, 13))
    with pytest.raises(RuntimeError, match="Python 3.10"):
        install_command(job)


def test_local_dataset_split_and_format_are_not_ignored(raw_job, tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text('\n'.join(json.dumps({"body": row}) for row in ("", "first", "second")))
    raw_job["dataset"] = {"source": str(source), "text_field": "body", "limit": 1}
    job = load_job(raw_job)
    target = materialize_dataset(job, RunPaths.for_job(job, tmp_path))
    assert [json.loads(line) for line in target.read_text().splitlines()] == [{"text": "first"}]
    raw_job["dataset"]["split"] = "validation"
    with pytest.raises(UnsupportedJob, match="split"):
        materialize_dataset(load_job(raw_job), RunPaths.for_job(job, tmp_path))


def test_hf_dataset_split_and_limit_are_honored(raw_job, tmp_path, monkeypatch):
    loader = Mock(return_value=iter([{"text": "first"}, {"text": "second"}]))
    module = types.ModuleType("datasets")
    module.load_dataset = loader
    monkeypatch.setitem(sys.modules, "datasets", module)
    raw_job["dataset"] = {"source": "public/data", "config": "small", "split": "train[:20]", "limit": 1}
    job = load_job(raw_job)
    target = materialize_dataset(job, RunPaths.for_job(job, tmp_path))
    loader.assert_called_once_with("public/data", "small", split="train[:20]", streaming=True)
    assert len(target.read_text().splitlines()) == 1


def make_checkpoint(path, step):
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "trainer_state.json", {"global_step": step})
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
        (path / name).write_text("fixture-data")
    return path


def test_checkpoint_selection_excludes_partial_and_corrupt(job, tmp_path):
    paths = RunPaths.for_job(job, tmp_path)
    two = make_checkpoint(paths.trainer / "checkpoint-2", 2)
    ten = make_checkpoint(paths.trainer / "checkpoint-10", 10)
    seal_checkpoint(job, two, "soup")
    seal_checkpoint(job, ten, "soup")
    (paths.checkpoints / "checkpoint-99").mkdir()
    assert find_latest_checkpoint(job, paths, engine="soup") == ten
    (ten / "optimizer.pt").write_text("corrupt")
    assert find_latest_checkpoint(job, paths, engine="soup") == two
    assert find_latest_checkpoint(job, paths, engine="unsloth") is None
    changed = load_job({**job.model_dump(), "model": {"id": "different/model"}})
    assert find_latest_checkpoint(changed, paths) is None


def test_incomplete_checkpoint_refused(job, tmp_path):
    incomplete = tmp_path / "checkpoint-2"
    incomplete.mkdir()
    (incomplete / "adapter_model.safetensors").write_text("adapter-only")
    with pytest.raises(IncompleteCheckpoint):
        publish_checkpoint(job, incomplete, engine="soup")
    assert not (incomplete / MANIFEST).exists()


def test_resume_never_required_and_hub_restore(job, tmp_path):
    paths = RunPaths.for_job(job, tmp_path)
    required = load_job({**job.model_dump(), "resume": {"mode": "required"}})
    with pytest.raises(ValueError, match="required"):
        restore_checkpoint(required, paths)
    store = Mock()
    never = load_job({**job.model_dump(), "resume": {"mode": "never"}})
    assert restore_checkpoint(never, paths, store=store) is None
    store.restore.assert_not_called()
    restore_checkpoint(job, paths, store=store)
    store.restore.assert_called_once_with(job, paths)


def test_checkpoint_manifest_cannot_escape_root(job, tmp_path):
    checkpoint = make_checkpoint(tmp_path / "checkpoint-2", 2)
    seal_checkpoint(job, checkpoint, "soup")
    manifest = json.loads((checkpoint / MANIFEST).read_text())
    manifest["files"]["../outside"] = "bad-digest"
    write_json(checkpoint / MANIFEST, manifest)
    with pytest.raises(ValueError, match="corrupted"):
        validate_checkpoint(job, checkpoint)


def test_hub_adapter_uses_atomic_checkpoint_layout(job, tmp_path):
    api = Mock()
    store = HFArtifactStore("test-owner/test-repo", api=api)
    paths = RunPaths.for_job(job, tmp_path)
    checkpoint = make_checkpoint(paths.trainer / "checkpoint-2", 2)
    publish_checkpoint(job, checkpoint, engine="soup", store=store)
    args = api.upload_folder.call_args.kwargs
    assert args["path_in_repo"] == "runs/test-run/checkpoints/checkpoint-2"
    assert args["repo_type"] == "model" and (checkpoint / MANIFEST).exists()
    store.publish_run(job, paths)
    assert "checkpoints/**" in api.upload_folder.call_args.kwargs["ignore_patterns"]


def test_job_to_hardware_resolver_soup_config_and_dry_run(job, gpu, tmp_path):
    execution = prepare_execution(job, hardware=gpu, output_dir=tmp_path, planning_only=True)
    result = execution.dry_run()
    assert result["resolved_engine"] == "soup" and result["planning_only"]
    assert not result["preflight"]["weights_downloaded"]
    assert yaml.safe_load((execution.paths.root / "soup.yaml").read_text())["base"] == job.model.id
    for name in ("job.yaml", "hardware.json", "resolved.json", "status.json", "logs", "checkpoints", "metrics"):
        assert (execution.paths.root / name).exists()
    with pytest.raises(RuntimeError, match="Planning-only"):
        execution.execute()


def test_run_id_cannot_overwrite_another_job(job, gpu, tmp_path):
    prepare_execution(job, hardware=gpu, output_dir=tmp_path, planning_only=True)
    other = load_job({**job.model_dump(), "model": {"id": "different/model"}})
    with pytest.raises(ValueError, match="different JobSpec"):
        prepare_execution(other, hardware=gpu, output_dir=tmp_path, planning_only=True)


def test_execute_hooks_and_local_contract_resume(job, gpu, tmp_path):
    engine = FakeEngine()
    execution = prepare_execution(job, hardware=gpu, output_dir=tmp_path, engines={"soup": engine})
    result = execution.execute()
    assert result["status"] == "succeeded" and not result["published"]
    metrics = json.loads((execution.paths.root / "metrics" / "availability.json").read_text())
    assert metrics["source"] == "latest_checkpoint" and not metrics["final_metrics_guaranteed"]
    latest = find_latest_checkpoint(job, execution.paths, engine="soup")
    assert latest and latest.name == "checkpoint-2"
    second_platform = RuntimePlatform("colab", tmp_path, tmp_path / "runs", "colab", True)
    second = prepare_execution(job, hardware=gpu, platform=second_platform, output_dir=tmp_path, engines={"soup": engine})
    second.execute()
    assert engine.received_resume == latest  # Contract fixture, not a real GPU resume claim.


def test_failure_records_safe_status_and_preserves_checkpoint(job, gpu, tmp_path):
    execution = prepare_execution(job, hardware=gpu, output_dir=tmp_path, engines={"soup": FakeEngine(fail=True)})
    with pytest.raises(RuntimeError):
        execution.execute()
    status = json.loads((execution.paths.root / "status.json").read_text())
    assert status["status"] == "failed" and "sensitive" not in json.dumps(status)
    assert find_latest_checkpoint(job, execution.paths)


def test_upload_requires_explicit_opt_in_and_target(job, gpu, tmp_path):
    execution = prepare_execution(job, hardware=gpu, output_dir=tmp_path, engines={"soup": FakeEngine()})
    with pytest.raises(ValueError, match="hf_repo"):
        execution.execute(publish=True)


def test_soup_cli_preflight_uses_current_commands(job, gpu, tmp_path, monkeypatch):
    paths = RunPaths.for_job(job, tmp_path)
    paths.create()
    monkeypatch.setattr(SoupEngine, "available", lambda self: True)
    monkeypatch.setattr("training_master.engines.soup.materialize_dataset", lambda *a: paths.root / "data.jsonl")
    run = Mock(return_value=types.SimpleNamespace(returncode=0))
    monkeypatch.setattr("training_master.engines.soup.subprocess.run", run)
    result = SoupEngine().dry_run(job, gpu, paths, cli=True)
    assert result["cli_preflight"]
    assert run.call_args_list[0].args[0][1:3] == ["doctor", "--config"]
    assert run.call_args_list[1].args[0][-1] == "--dry-run"
    assert all(call.kwargs["check"] for call in run.call_args_list)


def test_cli_cannot_execute_synthetic_hardware(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["training-master", "job.yaml", "--execute", "--hardware-profile", "fake.json"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


@pytest.mark.parametrize("name", ["Training_Master_Colab", "Kaggle-Training_Master"])
def test_canonical_notebook_structure_tokens_and_script_parity(name, tmp_path):
    path = ROOT / "nb" / f"{name}.ipynb"
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    markdown = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")
    for title in ("Runtime information", "Secrets", "Bootstrap Training Master runner", "Job", "Resolve", "Dry run", "Execute", "Artifacts"):
        assert title in markdown
    code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    assert "RUN_TRAINING = False" in code and "PUBLISH_ARTIFACTS = False" in code
    assert "userdata.get" not in code and "UserSecretsClient" not in code
    import re
    assert not re.search(r"hf_[A-Za-z0-9]{20,}|wandb_[A-Za-z0-9]{20,}", code)
    assert all(not cell.outputs and cell.execution_count is None for cell in nb.cells if cell.cell_type == "code")
    generator = importlib.import_module("update_all_notebooks")
    target = tmp_path / f"{name}.py"
    generator.convert_notebook_to_script(str(path), str(target))
    assert target.read_bytes() == (ROOT / "python_scripts" / target.name).read_bytes()
    assert path.name in generator.DONT_UPDATE_EXCEPTIONS
    before = path.read_bytes()
    generator._apply_global_fixes(str(path))
    assert not generator._update_news_only(str(path), "replacement")
    assert path.read_bytes() == before


@pytest.mark.parametrize("name", ["tiny-soup-sft", "tiny-unsloth-sft"])
def test_smoke_jobs_are_small(name, gpu, tmp_path):
    job = load_job(ROOT / "examples" / "jobs" / f"{name}.yaml")
    assert job.dataset.limit <= 10 and job.training.max_length <= 128
    assert job.training.lora.enabled
    assert not job.artifacts.hf_repo
    engine = resolve_engine(job, gpu, allow_unavailable=True)
    assert engine.dry_run(job, gpu, RunPaths.for_job(job, tmp_path))["weights_downloaded"] is False


def test_hub_restore_recovers_a_partial_local_copy(job, tmp_path, monkeypatch):
    snapshot = tmp_path / "hub-cache"
    source = make_checkpoint(snapshot / "runs" / job.run_id / "checkpoints" / "checkpoint-2", 2)
    seal_checkpoint(job, source, "soup")
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: str(snapshot))
    paths = RunPaths.for_job(job, tmp_path / "new-runtime")
    broken = paths.checkpoints / "checkpoint-2"
    broken.mkdir(parents=True)
    (broken / "partial").write_text("preserve-me")
    restored = restore_checkpoint(job, paths, engine="soup", store=HFArtifactStore("test-owner/test-repo", api=Mock()))
    assert restored == broken
    assert validate_checkpoint(job, restored, "soup") == 2
    assert list(paths.checkpoints.glob("checkpoint-2.incomplete-*/partial"))


def test_direct_unsloth_run_contract_without_real_frameworks(raw_job, gpu, tmp_path, monkeypatch):
    from training_master.engines import UnslothEngine
    source = tmp_path / "input.jsonl"
    source.write_text('{"text": "tiny training fixture"}\n')
    raw_job.update(dataset={"source": str(source)}, engine={"type": "unsloth"})
    raw_job["training"]["max_steps"] = 4
    job = load_job(raw_job)
    paths = RunPaths.for_job(job, tmp_path)
    paths.create()
    model = Mock()
    tokenizer = Mock()
    loader = types.SimpleNamespace(from_pretrained=Mock(return_value=(model, tokenizer)), get_peft_model=Mock(return_value=model))
    trainer = Mock()
    trainer.train.return_value = types.SimpleNamespace(metrics={"train_loss": 1.0})
    config = Mock(side_effect=lambda **kwargs: types.SimpleNamespace(**kwargs))
    sft_trainer = Mock(return_value=trainer)
    modules = {
        "unsloth": {"FastLanguageModel": loader},
        "torch": {"cuda": types.SimpleNamespace(is_bf16_supported=lambda: False)},
        "datasets": {"Dataset": types.SimpleNamespace(from_list=lambda rows: rows)},
        "transformers": {"TrainerCallback": type("TrainerCallback", (), {})},
        "trl": {"SFTConfig": config, "SFTTrainer": sft_trainer},
    }
    for name, values in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    resume = tmp_path / "checkpoint-2"
    result = UnslothEngine().run(job, gpu, paths, resume=resume)
    assert config.call_args.kwargs["max_steps"] == 4
    assert config.call_args.kwargs["max_length"] == 512
    assert config.call_args.kwargs["fp16"] and not config.call_args.kwargs["bf16"]
    trainer.train.assert_called_once_with(resume_from_checkpoint=str(resume))
    assert loader.get_peft_model.call_args.kwargs["r"] == 16
    assert sft_trainer.call_args.kwargs["train_dataset"] == [{"text": "tiny training fixture"}]
    model.save_pretrained.assert_called_once_with(str(paths.final))
    assert result["metrics"]["train_loss"] == 1.0


def test_boolean_version_and_backend_resume_identity(raw_job, job, tmp_path):
    from training_master.job import job_digest
    raw_job["version"] = True
    with pytest.raises(ValidationError):
        load_job(raw_job)
    other_backend = load_job({**job.model_dump(), "engine": {"type": "soup", "backend": "unsloth"}})
    assert job_digest(job) != job_digest(other_backend)


def test_t4_bf16_emulation_is_not_native_support(monkeypatch, tmp_path):
    fake = types.SimpleNamespace(__version__="2.6", version=types.SimpleNamespace(cuda="12.4"), cuda=types.SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda i: "Tesla T4",
        get_device_properties=lambda i: types.SimpleNamespace(total_memory=15 * 1024**3, major=7),
        is_bf16_supported=lambda: True,  # Some torch versions include software emulation.
    ))
    monkeypatch.setattr("training_master.hardware.importlib.import_module", lambda name: fake)
    result = probe_hardware(RuntimePlatform("kaggle", tmp_path, tmp_path / "runs", "kaggle", True))
    assert not result.bf16_supported


def test_future_executable_nemo_adapter_is_preferred(raw_job):
    raw_job["model"] = {"id": "nvidia/Nemotron-model"}
    job = load_job(raw_job)
    nemo, soup = FakeEngine(), FakeEngine()
    nemo.name = "nemo_automodel"
    hardware = HardwareProfile(gpu_count=1, gpu_vram_gb=[80], cuda_available=True, bf16_supported=True)
    assert resolve_engine(job, hardware, engines={"nemo_automodel": nemo, "soup": soup}) is nemo


def test_bootstrap_pins_torchao_to_the_installed_torch_window(raw_job, monkeypatch):
    from training_master.bootstrap import _torch_constraints
    monkeypatch.setattr("training_master.bootstrap.importlib.metadata.version", lambda name: "2.10.0")
    assert _torch_constraints(unsloth=True) == ["torch==2.10.0", "torchao==0.16.0"]
    monkeypatch.setattr("training_master.bootstrap.importlib.metadata.version", lambda name: "2.11.0")
    assert _torch_constraints(unsloth=True) == ["torch==2.11.0", "torchao==0.18.0"]
    monkeypatch.setattr("training_master.bootstrap.importlib.metadata.version", lambda name: "2.6.0")
    with pytest.raises(RuntimeError, match="supported torch"):
        _torch_constraints(unsloth=True)
    assert _torch_constraints(unsloth=False) == ["torch==2.6.0"]


def test_incomplete_trainer_json_is_retried(job, tmp_path):
    checkpoint = make_checkpoint(tmp_path / "checkpoint-2", 2)
    (checkpoint / "trainer_state.json").write_text('{"global_step":')
    with pytest.raises(IncompleteCheckpoint, match="still being written"):
        seal_checkpoint(job, checkpoint, "soup")


def test_soup_backend_option_is_not_ignored_by_direct_engine(raw_job, gpu):
    raw_job["engine"] = {"type": "unsloth", "backend": "transformers"}
    with pytest.raises(UnsupportedJob, match="engine.backend"):
        resolve_engine(load_job(raw_job), gpu, allow_unavailable=True)
