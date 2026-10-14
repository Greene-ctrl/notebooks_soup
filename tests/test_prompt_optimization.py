"""Prompt workflow contracts: no real DSPy installation/provider/API calls needed."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from training_master import JobSpec, PromptJobSpec, load_job
from training_master.artifacts import HFArtifactStore, RunPaths
from training_master.bootstrap import install_command
from training_master.engines import TrainingEngine, resolve_engine
from training_master.hardware import HardwareProfile
from training_master.job import job_digest
from training_master.metrics import numeric_score, resolve_metric
from training_master.optimizers import GEPAOptimizer, OptimizerEngine, resolve_optimizer
from training_master.optimizers.data import dspy_examples, load_splits, read_examples
from training_master.optimizers.lm import apply_lm_environment, create_lm, validate_lm
from training_master.programs import build_program, load_program, signature_fields
from training_master.runner import prepare_execution

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw():
    data = load_job(ROOT / 'examples/jobs/tiny-gepa-prompt.yaml').model_dump(mode='json')
    data['run_id'] = 'prompt-unit'
    for split, rows in data['dataset'].items():
        for i, row in enumerate(rows):
            row.update(question=f'{split}-{i}', answer='ok')
    return data


@pytest.fixture
def job(raw):
    return load_job(raw)


@pytest.mark.parametrize('mutation', [
    {'training': {'lora': {'r': 8}}}, {'model': {'id': 'wrong'}},
    {'student': None}, {'reflection': None}, {'metric': None}, {'optimizer': None},
    {'optimizer': {'type': 'gepa'}},
    {'optimizer': {'type': 'gepa', 'budget': 'light', 'max_metric_calls': 16}},
    {'optimizer': {'type': 'unknown', 'budget': 'light'}},
    {'optimizer': {'type': 'gepa', 'max_metric_calls': 0}},
    {'optimizer': {'type': 'gepa', 'budget': 'light', 'num_threads': 0}},
    {'program': {'type': 'arbitrary_import', 'signature': 'question -> answer'}},
    {'program': {'type': 'predict', 'signature': '__import__(x) -> answer'}},
    {'program': {'type': 'predict', 'tools': ['untrusted']}},
    {'artifacts': {'publish': True}}, {'kind': 'unknown'}, {'version': True},
])
def test_invalid_prompt_combinations(raw, mutation):
    raw.update(mutation)
    with pytest.raises(ValidationError):
        load_job(raw)


def test_training_backward_compatibility_and_digest():
    old = {'run_id': 'legacy', 'model': {'id': 'test/model'}, 'dataset': {'source': 'data'}}
    job = load_job(old)
    assert isinstance(job, JobSpec) and job.kind == 'model_training'
    assert load_job({**old, 'kind': 'model_training'}) == job
    data = job.model_dump(mode='json', exclude={'kind', 'engine', 'artifacts', 'resume', 'requirements'})
    data['backend'] = job.engine.backend or 'transformers'
    assert job_digest(job) == hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValidationError):
        load_job({**old, 'reflection': {'model': 'openai/wrong'}})


def test_resolver_paths_are_separate(job):
    assert isinstance(job, PromptJobSpec)
    optimizer = resolve_optimizer(job, allow_unavailable=True)
    assert isinstance(optimizer, OptimizerEngine) and not isinstance(optimizer, TrainingEngine)
    with pytest.raises(ValueError, match='model_training'):
        resolve_engine(job, HardwareProfile())
    training = load_job({'run_id': 'x', 'model': {'id': 'x'}, 'dataset': {'source': 'x'}})
    with pytest.raises(ValueError, match='prompt_optimization'):
        resolve_optimizer(training)


def test_explicit_budget_and_no_pickled_gepa_logs(job):
    config = GEPAOptimizer().configuration(job)
    assert config['max_metric_calls'] == 16 and config['auto'] is None
    assert config['log_dir'] is None and config['track_stats']
    assert not config['use_wandb'] and not config['use_mlflow']
    assert config['num_threads'] == 1


def test_prepare_plan_needs_neither_gpu_nor_secrets(job, tmp_path, monkeypatch):
    monkeypatch.setattr('training_master.optimizers.gepa.create_lm', Mock(side_effect=AssertionError('no paid calls')))
    execution = prepare_execution(job, hardware=HardwareProfile(), planning_only=True, output_dir=tmp_path)
    plan = execution.dry_run()
    assert plan['dataset_sizes'] == {'train': 10, 'validation': 4, 'test': 4}
    assert plan['max_metric_calls'] == 16 and plan['estimated_metric_evaluations'] == 32
    assert plan['required_secrets'] == ['REFLECTION_API_KEY', 'STUDENT_API_KEY']
    assert not plan['api_calls_made']
    with pytest.raises(RuntimeError, match='Planning-only'):
        execution.execute()


def test_budget_must_cover_initial_validation(raw):
    raw['optimizer']['max_metric_calls'] = 1
    with pytest.raises(ValueError, match='initial full validation'):
        GEPAOptimizer().prepare(load_job(raw))


@pytest.mark.parametrize('signature', ['a -> a', 'a -> b, c', ' -> a', 'a: int -> b', 'a a -> b'])
def test_unsupported_signatures(signature):
    with pytest.raises(ValueError):
        signature_fields(signature)


def test_dataset_local_json_jsonl_and_empty_validation(raw, tmp_path):
    for suffix in ['json', 'jsonl']:
        path = tmp_path / f'train.{suffix}'
        rows = raw['dataset']['train']
        path.write_text(json.dumps(rows) if suffix == 'json' else '\n'.join(json.dumps(r) for r in rows))
        assert read_examples(str(path)) == rows
    raw['dataset']['validation'] = []
    with pytest.raises(ValueError, match='validation'):
        load_splits(load_job(raw))


def test_heldout_overlap_is_refused_even_with_changed_labels(raw):
    raw['dataset']['test'][0] = {**raw['dataset']['train'][0], 'answer': 'different'}
    with pytest.raises(ValueError, match='held-out'):
        load_splits(load_job(raw))


def test_dataset_missing_output_field_is_rejected(raw):
    del raw['dataset']['train'][0]['answer']
    with pytest.raises(ValueError, match='answer'):
        load_splits(load_job(raw))


def test_hf_dataset_is_streamed_and_limited(monkeypatch):
    from training_master.job import HFDatasetSpec
    loader = Mock(return_value=iter([{'q': 'one'}, {'q': 'two'}]))
    monkeypatch.setitem(sys.modules, 'datasets', types.SimpleNamespace(load_dataset=loader))
    spec = HFDatasetSpec(type='hf', id='public/data', config='small', split='validation', limit=1)
    assert read_examples(spec) == [{'q': 'one'}]
    loader.assert_called_once_with('public/data', 'small', split='validation', streaming=True)


def test_student_reflection_environment_settings_are_independent(job):
    resolved = apply_lm_environment(job, {
        'STUDENT_MODEL': 'openai/gemma', 'STUDENT_API_BASE': 'http://localhost:8000/v1',
        'STUDENT_API_KEY': 'unprinted-test-secret',
        'REFLECTION_MODEL': 'openai/nemotron', 'REFLECTION_API_BASE': 'https://reflection.example/v1',
    })
    assert resolved.student.model == 'openai/gemma'
    assert resolved.reflection.model == 'openai/nemotron'
    assert resolved.student.api_key_secret == 'STUDENT_API_KEY'
    assert 'unprinted-test-secret' not in resolved.model_dump_json()
    assert job.student.model == 'openai/gpt-4.1-mini'


def test_invalid_endpoint_error_hides_credential_values(job):
    data = job.model_dump()
    data['student']['api_base'] = 'https://user:private-value@example.test/v1'
    with pytest.raises(ValidationError) as error:
        load_job(data)
    assert 'private-value' not in str(error.value)


def test_reasoning_limits_are_not_silently_increased(job):
    from training_master.job import LMSpec
    with pytest.raises(ValueError, match='explicit max_tokens'):
        validate_lm(LMSpec(model='openai/gpt-5.4', api_key_secret='REFLECTION_API_KEY'))
    validate_lm(LMSpec(model='openai/gpt-5.4', api_key_secret='REFLECTION_API_KEY', max_tokens=16000, temperature=1))


class Example(dict):
    def with_inputs(self, *names):
        self.names = names
        return self

    def inputs(self):
        return {key: self[key] for key in self.names}


class Prediction(dict):
    def __getattr__(self, name):
        return self[name]


@pytest.fixture
def fake_dspy(monkeypatch):
    events = []
    class Program:
        def __init__(self, signature, **kwargs):
            self.signature = signature
            self.optimized = False
            self.lm = None
            self.kwargs = kwargs
        def set_lm(self, lm):
            self.lm = lm
        def deepcopy(self):
            return copy.deepcopy(self)
        def named_predictors(self):
            return [('predict', self)]
        def __call__(self, **kwargs):
            events.append(('predict', kwargs['question'], self.optimized))
            return Prediction(answer='ok' if self.optimized else 'wrong')
        def save(self, path, save_program=False):
            assert self.lm is None and not save_program
            Path(path).write_text(json.dumps({'signature': self.signature, 'optimized': self.optimized, 'lm': None}))
        def load(self, path, **kwargs):
            assert kwargs == {'allow_pickle': False, 'allow_unsafe_lm_state': False}
            self.optimized = json.loads(Path(path).read_text())['optimized']
    class GEPA:
        last_config = None
        def __init__(self, **kwargs):
            GEPA.last_config = kwargs
        def compile(self, program, *, trainset, valset):
            events.append(('compile', [e['question'] for e in trainset], [e['question'] for e in valset]))
            optimized = program.deepcopy()
            optimized.optimized = True
            optimized.detailed_results = types.SimpleNamespace(to_dict=lambda: {
                'candidates': [{'predict': 'optimized instructions'}], 'total_metric_calls': 12,
                'val_aggregate_scores': [0, 1], 'best_idx': 1,
            })
            return optimized
    lm = Mock(side_effect=lambda *args, **kwargs: types.SimpleNamespace(
        settings=kwargs, history=[{'usage': {'total_tokens': 10}, 'api_key': kwargs.get('api_key')}]))
    module = types.SimpleNamespace(Predict=Program, ChainOfThought=Program, ReAct=Program,
        GEPA=GEPA, LM=lm, Example=Example, Prediction=Prediction, context=lambda **kwargs: nullcontext())
    monkeypatch.setitem(sys.modules, 'dspy', module)
    monkeypatch.setattr(GEPAOptimizer, 'available', lambda self: True)
    monkeypatch.setenv('STUDENT_API_KEY', 'test-student-secret')
    monkeypatch.setenv('REFLECTION_API_KEY', 'test-reflection-secret')
    return module, events


@pytest.mark.parametrize('name', ['predict', 'chain_of_thought', 'react'])
def test_program_registry(job, fake_dspy, name):
    from training_master.job import ProgramSpec
    program = build_program(ProgramSpec(type=name, signature='question -> answer'))
    if name == 'react':
        assert program.kwargs == {'tools': [], 'max_iters': 3}
    with pytest.raises(ValueError, match='Unknown program tool'):
        build_program(ProgramSpec(type='react', tools=['unregistered']))


@pytest.mark.parametrize('name,expected', [('exact_match', 1), ('exact_match_with_feedback', 1), ('contains', 1)])
def test_metric_registry_and_feedback(job, fake_dspy, name, expected):
    from training_master.job import MetricSpec
    metric = resolve_metric(MetricSpec(type=name))
    value = metric({'answer': ' ok '}, Prediction(answer='OK'), None, 'predict', None)
    assert numeric_score(value) == expected
    if name == 'exact_match_with_feedback':
        wrong = metric({'answer': 'ok'}, Prediction(answer='wrong'))
        assert wrong.score == 0 and 'Expected' in wrong.feedback and 'wrong' in wrong.feedback


def test_lm_factory_uses_separate_named_keys_and_settings(job, fake_dspy, capsys):
    student, reflection = create_lm(job.student), create_lm(job.reflection)
    assert student.settings['api_key'] == 'test-student-secret'
    assert reflection.settings['api_key'] == 'test-reflection-secret'
    assert student.settings['max_tokens'] == 128 and reflection.settings['max_tokens'] == 768
    assert student.settings['num_retries'] == 0 and not student.settings['cache']
    assert capsys.readouterr().out == ''


def test_full_mocked_workflow_artifacts_and_heldout_order(job, fake_dspy, tmp_path):
    module, events = fake_dspy
    execution = prepare_execution(job, hardware=HardwareProfile(), output_dir=tmp_path)
    result = execution.execute()
    assert result['baseline_validation'] == result['baseline_test'] == 0
    assert result['optimized_validation'] == result['optimized_test'] == result['delta_test'] == 1
    assert result['status'] == 'COMPLETED' and not result['published']
    compile_index = next(i for i, e in enumerate(events) if e[0] == 'compile')
    assert all(not e[1].startswith('test-') for e in events[:compile_index] if e[0] == 'predict')
    assert all(not q.startswith('test-') for qs in events[compile_index][1:] for q in qs)
    statuses = [json.loads(line)['status'] for line in (execution.paths.root / 'optimization/logs/events.jsonl').read_text().splitlines()]
    assert statuses == ['PREPARING', 'BASELINE_EVAL', 'OPTIMIZING', 'TESTING', 'COMPLETED']
    for name in ['baseline/program.json', 'baseline/metrics.json', 'optimized/program.json', 'optimized/metrics.json', 'optimization/stats.json', 'optimization/candidates.json', 'optimization/usage.json', 'test/baseline.json', 'test/optimized.json', 'result.json']:
        assert (execution.paths.root / name).is_file()
    all_text = ''.join(p.read_text() for p in execution.paths.root.rglob('*') if p.is_file())
    assert 'test-student-secret' not in all_text and 'test-reflection-secret' not in all_text
    assert not list(execution.paths.root.rglob('*.pkl'))
    restored = load_program(job.program, execution.paths.root / 'optimized/program.json', dspy_module=module)
    assert restored.optimized
    with pytest.raises(ValueError, match='new run_id'):
        execution.execute()


def test_staged_notebook_calls_use_the_same_pipeline(job, fake_dspy, tmp_path):
    execution = prepare_execution(job, hardware=HardwareProfile(), output_dir=tmp_path)
    with pytest.raises(ValueError, match='baseline'):
        execution.optimize()
    assert execution.baseline()['baseline_validation'] == 0
    execution.optimize()
    assert execution.evaluate()['delta_test'] == 1
    with pytest.raises(ValueError):
        execution.optimize()
    with pytest.raises(ValueError):
        execution.evaluate()
    assert json.loads((execution.paths.root / 'status.json').read_text())['status'] == 'COMPLETED'


def test_failed_provider_records_safe_status(job, fake_dspy, tmp_path, monkeypatch):
    monkeypatch.setattr('training_master.optimizers.gepa.create_lm', Mock(side_effect=RuntimeError('sensitive-provider-body')))
    execution = prepare_execution(job, hardware=HardwareProfile(), output_dir=tmp_path)
    with pytest.raises(RuntimeError):
        execution.execute()
    status = (execution.paths.root / 'status.json').read_text()
    assert 'FAILED' in status and 'sensitive-provider-body' not in status


def test_publication_requires_explicit_job_configuration(job, fake_dspy, tmp_path):
    execution = prepare_execution(job, hardware=HardwareProfile(), output_dir=tmp_path)
    with pytest.raises(ValueError, match='artifacts.publish'):
        execution.execute(publish=True)
    assert fake_dspy[0].LM.call_count == 0


def test_optional_hf_layout_is_reused_without_upload(job, tmp_path):
    store = HFArtifactStore('test-owner/test-repo', api=Mock())
    assert store.prefix(job) == 'runs/prompt-unit'
    store.publish_run(job, RunPaths.for_job(job, tmp_path))
    assert store.api.upload_folder.call_args.kwargs['path_in_repo'] == 'runs/prompt-unit'


def test_bootstrap_installs_only_dspy(job):
    command = install_command(job)
    assert command[-1] == 'dspy==3.4.1'
    assert not any(x in str(command) for x in ['soup-cli', 'unsloth', 'nemo', 'torch'])


def test_dspy_examples_only_mark_signature_inputs(job, fake_dspy):
    examples = dspy_examples([{'question': 'q', 'answer': 'gold'}], job.program.signature)
    assert examples[0].inputs() == {'question': 'q'}


def test_gepa_notebook_structure_no_credentials_and_script_parity(tmp_path):
    import nbformat
    from update_all_notebooks import TRAINING_MASTER_NOTEBOOKS, convert_notebook_to_script
    name = 'Prompt_Optimization_GEPA'
    source = ROOT / 'nb' / f'{name}.ipynb'
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    markdown = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'markdown')
    code = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'code')
    for section in ['Runtime', 'Secrets', 'Bootstrap', 'JobSpec', 'Resolve', 'Baseline', 'Optimize', 'Evaluate', 'Artifacts']:
        assert section in markdown
    assert 'RUN_OPTIMIZATION = False' in code
    assert 'execution.baseline()' in code and 'execution.optimize()' in code and 'execution.evaluate()' in code
    assert 'STUDENT_API_KEY=' not in code and 'REFLECTION_API_KEY=' not in code
    assert not any(cell.outputs or cell.execution_count is not None for cell in notebook.cells if cell.cell_type == 'code')
    assert source.name in TRAINING_MASTER_NOTEBOOKS
    target = tmp_path / f'{name}.py'
    convert_notebook_to_script(str(source), str(target))
    assert target.read_bytes() == (ROOT / 'python_scripts' / target.name).read_bytes()


def test_keyless_endpoint_uses_explicit_empty_key(job, fake_dspy, monkeypatch):
    from training_master.job import LMSpec
    monkeypatch.setenv('OPENAI_API_KEY', 'unrelated-global-key')
    lm = create_lm(LMSpec(model='openai/gemma', api_base='http://localhost:8000/v1'))
    assert lm.settings['api_key'] == ''


def test_publication_failure_is_recorded_without_false_success(raw, fake_dspy, tmp_path, monkeypatch):
    raw['artifacts'] = {'hf_repo': 'test-owner/test-repo', 'publish': True}
    monkeypatch.setenv('HF_TOKEN', 'test-hf-secret')
    store = Mock()
    store.publish_run.side_effect = RuntimeError('private-provider-body')
    monkeypatch.setattr('training_master.prompt_runner.HFArtifactStore', lambda *a, **k: store)
    execution = prepare_execution(load_job(raw), hardware=HardwareProfile(), output_dir=tmp_path)
    with pytest.raises(RuntimeError):
        execution.execute()
    result = json.loads((execution.paths.root / 'result.json').read_text())
    status = json.loads((execution.paths.root / 'status.json').read_text())
    assert result['status'] == status['status'] == 'FAILED' and not result['published']
    assert 'private-provider-body' not in json.dumps(status)


def test_stats_disabled_has_explicit_unavailable_record(job, tmp_path):
    paths = RunPaths.for_job(job, tmp_path)
    GEPAOptimizer()._save_stats(types.SimpleNamespace(), paths)
    assert json.loads((paths.root / 'optimization/stats.json').read_text()) == {'available': False}


def test_cli_reports_prompt_budget_before_any_execution(job, monkeypatch, capsys):
    from training_master.runner import main
    order = []
    fake = types.SimpleNamespace(job=job,
        dry_run=lambda: (order.append('plan') or {'max_metric_calls': 16}),
        execute=lambda **kwargs: (order.append('execute') or {'status': 'COMPLETED'}))
    monkeypatch.setattr('training_master.runner.prepare_execution', lambda *a, **k: fake)
    monkeypatch.setattr(sys, 'argv', ['training-master', 'job.yaml', '--execute'])
    main()
    assert order == ['plan', 'execute']
    output = capsys.readouterr()
    assert 'max_metric_calls' in output.err and '16' in output.err
