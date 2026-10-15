"""Safe, lightweight provenance and after-failure notebook diagnostics."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from .process import redact


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_manifest(repo_root, hardware):
    try:
        sha = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        sha = None
    return {"python": platform.python_version(), "python_executable": sys.executable,
            "platform": platform.platform(), "runner_commit": sha,
            "packages": {name: package_version(name) for name in ("soup-cli", "torch", "transformers", "datasets", "accelerate", "bitsandbytes", "unsloth")},
            "cuda_version": hardware.cuda_version, "gpu": hardware.gpu_names}


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_diagnostics(root: Path, *, tail_lines=100):
    commands = sorted((root / "commands").glob("*.json"), key=lambda p: p.stat().st_mtime_ns)
    latest = read_json(commands[-1]) if commands else None
    tail = ""
    if latest:
        log_path = Path(latest["log_path"])
        # Read a bounded suffix, not an entire multi-GB training log.
        if log_path.is_file() and log_path.resolve().is_relative_to(root.resolve()):
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 24000))
                tail = "\n".join(stream.read().decode("utf-8", errors="replace").splitlines()[-tail_lines:])
    config = root / "soup.yaml"
    return {"status": read_json(root / "status.json"), "resolved": read_json(root / "resolved.json"),
            "environment": read_json(root / "environment.json"),
            "generated_config": redact(config.read_text(encoding="utf-8")) if config.exists() else None,
            "latest_command": latest, "recent_log_tail": redact(tail),
            "checkpoints": [str(p) for p in root.glob("checkpoints/**/checkpoint-*") if p.is_dir()],
            "artifact_state": read_json(root / "final" / "training_master_final.json")}


def show_run_diagnostics(run_root: str | Path) -> None:
    """Inspect a failed preparation even when no Execution object was returned."""
    print(redact(json.dumps(run_diagnostics(Path(run_root).resolve()), indent=2)))
