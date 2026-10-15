"""Real CPU child processes and source-independent training path contracts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from training_master.artifacts import RunPaths
from training_master.engines import SoupEngine
from training_master.hardware import HardwareProfile
from training_master.job import job_digest, load_job
from training_master.paths import REPO_ROOT, resolve_runtime_path
from training_master.platform import RuntimePlatform
from training_master.process import EngineProcessError, run_process
from training_master.runner import prepare_execution


def command(tmp_path, script, **kwargs):
    return run_process([sys.executable, '-u', '-c', script], cwd=tmp_path,
                       log_path=tmp_path / 'logs/process.log', record_path=tmp_path / 'commands/process.json', **kwargs)


def test_success_unicode_logs_live_and_ordered(tmp_path, capsys):
    result = command(tmp_path, 'import sys; print("hello 🌍"); print("stderr café", file=sys.stderr)')
    assert result.exit_code == 0 and result.duration >= 0
    assert result.stderr_tail == ''
    assert result.stdout_tail.splitlines() == ['hello 🌍', 'stderr café']
    assert 'café' in capsys.readouterr().out
    record = json.loads((tmp_path / 'commands/process.json').read_text())
    assert record['cwd'] == str(tmp_path) and record['exit_code'] == 0
    assert record['started_at'] <= record['finished_at'] and 'env' not in record


def test_failure_final_line_bounded_tail_complete_log(tmp_path):
    with pytest.raises(EngineProcessError) as caught:
        command(tmp_path, 'import sys; [print(f"line {i}") for i in range(1000)]; print("FINAL ERROR", file=sys.stderr); sys.exit(1)', tail_lines=80, stream=False)
    error = caught.value
    assert error.exit_code == 1 and error.__cause__.returncode == 1
    assert len(error.stdout_tail.splitlines()) == 80
    assert 'FINAL ERROR' in str(error) and 'line 0\n' not in str(error)
    assert 'line 0\n' in Path(error.log_path).read_text()
    assert 'FINAL ERROR' in Path(error.log_path).read_text()


def test_huge_single_line_is_bounded(tmp_path):
    with pytest.raises(EngineProcessError) as caught:
        command(tmp_path, 'print("x" * 100000); raise SystemExit(1)', tail_chars=100, stream=False)
    assert len(caught.value.stdout_tail) <= 100
    assert Path(caught.value.log_path).stat().st_size > 100000


def test_secret_redacted_in_log_tail_and_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('STUDENT_API_KEY', 'private-fixture-key')
    result = command(tmp_path, 'import os; print(os.environ["STUDENT_API_KEY"])')
    assert 'private-fixture-key' not in result.stdout_tail
    assert 'private-fixture-key' not in Path(result.log_path).read_text()
    assert 'private-fixture-key' not in capsys.readouterr().out


def test_interrupt_terminates_and_records_child(tmp_path):
    def interrupt():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        command(tmp_path, 'import time; print("ready", flush=True); time.sleep(30)', on_poll=interrupt, stream=False)
    record = json.loads((tmp_path / 'commands/process.json').read_text())
    assert record['interrupted'] and record['exit_code'] is not None


def test_missing_executable_records_launch_failure(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_process(['/missing-training-master-executable'], cwd=tmp_path, log_path=tmp_path / 'out.log', record_path=tmp_path / 'command.json')
    assert json.loads((tmp_path / 'command.json').read_text())['error_type'] == 'FileNotFoundError'


@pytest.mark.parametrize('root_name', ['content', 'kaggle/working'])
def test_paths_are_independent_of_notebook_cwd(tmp_path, monkeypatch, root_name):
    repo = tmp_path / root_name / 'training-master-source'
    (repo / 'examples/data').mkdir(parents=True)
    data = repo / 'examples/data/train.jsonl'
    data.write_text('{"text":"tiny"}\n')
    runtime_root = repo.parent
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir(exist_ok=True)
    monkeypatch.chdir(elsewhere)
    job = load_job({'run_id':'path-test','model':{'id':'Qwen/Qwen3-0.6B'},'dataset':{'source':'examples/data/train.jsonl'}})
    digest = job_digest(job)
    execution = prepare_execution(job, repo_root=repo, platform=RuntimePlatform('colab',runtime_root,runtime_root / 'runs','environment',True), output_dir="runs", planning_only=True)
    config = yaml.safe_load((execution.paths.root / 'soup.yaml').read_text())
    assert Path(config['data']['train']).is_absolute()
    assert Path(config['output']).is_absolute()
    assert execution.resolved_job.dataset.source == str(data)
    assert execution.resolved_job.model.id == 'Qwen/Qwen3-0.6B'
    assert job_digest(execution.job) == digest
    manifest = execution.diagnostics()['resolved']
    assert manifest['repo_root'] == str(repo) and manifest['engine_workdir'] == str(execution.paths.root)
    assert manifest['runtime_root'] == str(runtime_root) and manifest['dataset_train'] == str(data)


@pytest.mark.parametrize('value',['Qwen/Qwen3-0.6B','Leon4gr45/my-dataset','hf://owner/repo/file','https://example.org/data'])
def test_remote_identifiers_preserved(tmp_path, value):
    assert resolve_runtime_path(value, repo_root=tmp_path) == value


def test_absolute_and_missing_local_paths(tmp_path):
    local = tmp_path / 'data.jsonl'
    local.touch()
    assert resolve_runtime_path(str(local), repo_root=tmp_path) == str(local)
    with pytest.raises(FileNotFoundError):
        resolve_runtime_path('missing.jsonl', repo_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_runtime_path('./models/missing', repo_root=tmp_path)


class FailingSoup(SoupEngine):
    def available(self):
        return True

    def dry_run(self, *args, **kwargs):
        return {'cli_preflight':True}

    def run(self, job, hardware, paths, **kwargs):
        return command(paths.root, 'print("useful engine error"); raise SystemExit(1)', stream=False)


@pytest.mark.parametrize('interrupt', [False,True])
def test_execution_failure_state_and_diagnostics(tmp_path, interrupt):
    engine = FailingSoup()
    if interrupt:
        engine.run = Mock(side_effect=KeyboardInterrupt())
    job = load_job({'run_id':'failed-test','model':{'id':'public/model'},'dataset':{'source':'public/data'}})
    execution = prepare_execution(job, output_dir=tmp_path, engines={'soup':engine})
    with pytest.raises(KeyboardInterrupt if interrupt else EngineProcessError):
        execution.execute()
    diagnostics = execution.diagnostics()
    assert diagnostics['status']['state'] == 'FAILED'
    assert diagnostics['status']['stage'] == 'TRAINING'
    assert diagnostics['resolved'] and diagnostics['generated_config'] and diagnostics['environment']
    if not interrupt:
        assert diagnostics['status']['exit_code'] == 1
        assert 'useful engine error' in diagnostics['recent_log_tail']


def test_doctor_failure_prevents_training_and_records_preflight(tmp_path, monkeypatch):
    engine = SoupEngine()
    monkeypatch.setattr(engine,'available', lambda:True)
    def fail(args, paths, **kwargs):
        return command(paths.root, 'print("doctor says broken dependency"); raise SystemExit(2)', stream=False)
    monkeypatch.setattr(engine,'_command',fail)
    run = Mock()
    monkeypatch.setattr(engine,'run',run)
    job = load_job(REPO_ROOT / 'examples/jobs/tiny-soup-sft.yaml')
    gpu = HardwareProfile(gpu_count=1,gpu_vram_gb=[15],cuda_available=True)
    execution = prepare_execution(job, hardware=gpu, output_dir=tmp_path, engines={'soup':engine})
    with pytest.raises(EngineProcessError,match='broken dependency'):
        execution.execute()
    run.assert_not_called()
    assert execution.diagnostics()['status']['stage'] == 'PREFLIGHT'


def test_missing_local_dataset_persists_preparation_failure(tmp_path):
    job = load_job({'run_id':'missing-test','model':{'id':'public/model'},'dataset':{'source':'missing.jsonl'}})
    with pytest.raises(FileNotFoundError):
        prepare_execution(job, repo_root=tmp_path, output_dir=tmp_path, planning_only=True)
    status = json.loads((tmp_path / 'missing-test/status.json').read_text())
    assert status['state'] == 'FAILED' and status['stage'] == 'PREPARING'


def test_soup_retries_preserve_command_history(tmp_path, monkeypatch):
    engine = SoupEngine()
    paths = RunPaths(tmp_path)
    run = Mock(return_value=None)
    def save_record(argv, **kwargs):
        kwargs['record_path'].parent.mkdir(parents=True,exist_ok=True)
        kwargs['record_path'].write_text('{}')
        return run(argv, **kwargs)
    monkeypatch.setattr('training_master.engines.soup.run_process',save_record)
    engine._command(['doctor'], paths, name='soup-doctor')
    engine._command(['doctor'], paths, name='soup-doctor')
    assert len(list((tmp_path / 'commands').glob('soup-doctor*.json'))) == 2
    assert run.call_count == 2


def test_invalid_utf8_is_replaced_without_losing_final_error(tmp_path):
    with pytest.raises(EngineProcessError) as caught:
        command(tmp_path, 'import os; os.write(1, b"bad: \\xff\\nfinal error\\n"); raise SystemExit(1)', stream=False)
    assert 'final error' in caught.value.stdout_tail
    assert '\ufffd' in Path(caught.value.log_path).read_text()


def test_kernel_script_directory_can_differ_from_python_bin(tmp_path, monkeypatch, capsys):
    from training_master.bootstrap import verify_engine_environment
    scripts = tmp_path / 'kernel-scripts'
    scripts.mkdir()
    soup = scripts / 'soup'
    soup.touch()
    monkeypatch.setattr('training_master.bootstrap.sysconfig.get_path',lambda name:str(scripts))
    monkeypatch.setattr('training_master.engines.soup.shutil.which',lambda name:str(soup))
    monkeypatch.setattr('training_master.bootstrap.importlib.metadata.version',lambda name:'0.76.0')
    job = load_job({'run_id':'environment-test','engine':{'type':'soup'},'model':{'id':'public/model'},'dataset':{'source':'public/data'}})
    verify_engine_environment(job)
    assert SoupEngine().available()
    assert 'Soup version: 0.76.0' in capsys.readouterr().out
