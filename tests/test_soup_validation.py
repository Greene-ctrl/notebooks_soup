"""Pinned real Soup CPU validation: no torch, model weights or provider calls."""
from __future__ import annotations

import importlib.metadata
import json
import shutil
from pathlib import Path

import pytest
import yaml

from training_master.bootstrap import SOUP_VERSION
from training_master.hardware import HardwareProfile
from training_master.paths import REPO_ROOT
from training_master.process import EngineProcessError
from training_master.runner import prepare_execution

pytest.importorskip('soup_cli')
from soup_cli.config.loader import load_config  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which('soup') or importlib.metadata.version('soup-cli') != SOUP_VERSION, reason='requires pinned Soup CLI')


def smoke(tmp_path):
    # Recorded hardware permits generating the GPU smoke config; actual CLI
    # commands probe this CPU themselves. This never authorizes real training.
    return prepare_execution(REPO_ROOT / 'examples/jobs/tiny-soup-sft.yaml',
                             hardware=HardwareProfile(gpu_count=1,gpu_vram_gb=[15],cuda_available=True),
                             output_dir=tmp_path, planning_only=True)


def test_real_soup_smoke_schema_doctor_and_data(tmp_path, monkeypatch):
    elsewhere = tmp_path / 'notebook-cwd'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    execution = smoke(tmp_path)
    path = execution.paths.root / 'soup.yaml'
    config = load_config(path)
    assert Path(config.data.train).is_absolute() and Path(config.output).is_absolute()
    assert config.base == 'HuggingFaceTB/SmolLM2-135M' and config.task == 'sft'
    assert config.training.lora.r == 8 and config.training.epochs == 1
    assert config.training.batch_size == 1 and config.data.max_length == 128
    assert config.training.gradient_checkpointing and config.training.quantization == 'none'
    assert 'max_steps' not in yaml.safe_load(path.read_text())['training']
    config_timestamp = path.stat().st_mtime_ns
    plan = execution.dry_run(cli=True)
    assert path.stat().st_mtime_ns == config_timestamp
    assert plan['preflight']['cli_preflight']
    for name in ['soup-doctor','soup-dry-run']:
        record = json.loads((execution.paths.root / 'commands' / f'{name}.json').read_text())
        assert record['exit_code'] == 0 and record['cwd'] == str(execution.paths.root)
    assert '8 train samples' in (execution.paths.root / 'logs/soup-dry-run.log').read_text()


def test_real_soup_failure_is_actionable(tmp_path):
    execution = smoke(tmp_path)
    path = execution.paths.root / 'soup.yaml'
    config = yaml.safe_load(path.read_text())
    config['data']['train'] = str(tmp_path / 'missing.jsonl')
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(EngineProcessError) as caught:
        execution.dry_run(cli=True)
    assert caught.value.exit_code != 0 and 'missing.jsonl' in str(caught.value)
    assert Path(caught.value.log_path).exists()
