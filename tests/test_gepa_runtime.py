"""Optional real DSPy/GEPA contract tests using local DummyLMs, never providers."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
os.environ.setdefault('DSPY_CACHEDIR', '/tmp/training-master-dspy-tests')
dspy = pytest.importorskip('dspy', reason='Install the optional gepa extra for runtime contract tests')

from training_master import load_job  # noqa: E402
from training_master.optimizers import GEPAOptimizer  # noqa: E402
from training_master.optimizers.lm import create_lm  # noqa: E402
from training_master.programs import build_program, load_program, save_program  # noqa: E402
from training_master.runner import prepare_execution  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def refused(*args, **kwargs):
        raise AssertionError("Real network/API calls are forbidden in DSPy runtime tests")
    monkeypatch.setattr(socket.socket, 'connect', refused)


def test_actual_dspy_gepa_compile_json_roundtrip_and_no_pickle(tmp_path, monkeypatch):
    from dspy.utils import DummyLM
    assert dspy.__version__ == '3.4.1' and importlib.metadata.version('gepa') == '0.1.4'
    data = load_job(ROOT / 'examples/jobs/tiny-gepa-prompt.yaml').model_dump(mode='json')
    data['run_id'] = 'real-gepa-dummy'
    for split in data['dataset'].values():
        for row in split:
            row['answer'] = 'ok'
    monkeypatch.setattr('training_master.optimizers.gepa.create_lm', lambda *a, **k: DummyLM([{'answer': 'ok'}] * 200))
    execution = prepare_execution(data, output_dir=tmp_path)
    result = execution.execute()
    assert result['baseline_test'] == result['optimized_test'] == 1
    assert result['delta_test'] == 0 and not result['published']
    stats = json.loads((execution.paths.root / 'optimization/stats.json').read_text())
    assert stats['available'] and stats['total_metric_calls'] >= 4
    assert not list(tmp_path.rglob('*.pkl')) and not list(tmp_path.rglob('*.pickle'))
    restored = load_program(execution.job.program, execution.paths.root / 'optimized/program.json', dspy_module=dspy)
    assert restored.signature.input_fields.keys() == {'question'}


@pytest.mark.parametrize('name', ['predict', 'chain_of_thought', 'react'])
def test_actual_program_registry(name):
    from training_master.job import ProgramSpec
    program = build_program(ProgramSpec(type=name, signature='question -> answer'), dspy_module=dspy)
    assert len(program.named_predictors()) >= 1


def test_actual_lm_configuration_and_credential_free_json(tmp_path, monkeypatch):
    from training_master.job import LMSpec, ProgramSpec
    monkeypatch.setenv('STUDENT_API_KEY', 'test-private-lm-key')
    lm = create_lm(LMSpec(model='openai/local-gemma', api_base='http://localhost:8000/v1', api_key_secret='STUDENT_API_KEY', temperature=0, max_tokens=64))
    assert lm.model == 'openai/local-gemma' and lm.kwargs['api_base'] == 'http://localhost:8000/v1'
    assert lm.kwargs['max_tokens'] == 64
    program = build_program(ProgramSpec(signature='question -> answer'))
    program.set_lm(lm)
    target = tmp_path / 'program.json'
    save_program(program, target)
    assert 'test-private-lm-key' not in target.read_text()
    assert json.loads(target.read_text())['lm'] is None


@pytest.mark.parametrize('mode', ['light', 'medium', 'heavy'])
def test_preset_budget_uses_current_dspy_auto_budget(mode):
    from dspy.teleprompt.gepa.gepa import AUTO_RUN_SETTINGS
    raw = load_job(ROOT / 'examples/jobs/tiny-gepa-prompt.yaml').model_dump(mode='json')
    raw['optimizer'] = {'type': 'gepa', 'budget': mode}
    job = load_job(raw)
    estimate = GEPAOptimizer().dry_run(job)
    assert estimate['max_metric_calls'] == dspy.GEPA.auto_budget(None, 1, AUTO_RUN_SETTINGS[mode]['n'], 4)
