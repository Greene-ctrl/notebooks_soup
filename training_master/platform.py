"""Execution paths and secret providers, independent of training behavior."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class RuntimePlatform:
    name: str
    working_dir: Path
    output_dir: Path
    secrets_provider: str
    persistent_notebook: bool


def detect_platform(environ: Mapping[str, str] | None = None) -> RuntimePlatform:
    env = os.environ if environ is None else environ
    if env.get("KAGGLE_KERNEL_RUN_TYPE") or env.get("KAGGLE_URL_BASE"):
        root = Path("/kaggle/working")
        return RuntimePlatform("kaggle", root, root / "runs", "kaggle", True)
    if env.get("COLAB_RELEASE_TAG") or env.get("COLAB_BACKEND_VERSION") or env.get("COLAB_GPU"):
        root = Path("/content")
        return RuntimePlatform("colab", root, root / "runs", "colab", True)
    root = Path.cwd()
    return RuntimePlatform("local", root, root / "runs", "environment", False)
