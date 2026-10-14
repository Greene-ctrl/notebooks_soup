"""Explicit, selected-engine installation; never install frameworks on import."""
from __future__ import annotations

import subprocess
import sys
import importlib.metadata
from .job import JobSpec, PromptJobSpec, HFDatasetSpec

SOUP_VERSION = "0.76.0"
SOUP_REF = "d5cb1a7e87a0042de27b83947f6948c3ca9456b7"
NEMOTRON_REF = "ca8c409f08a9a5a5648d427383adea741b61966a"
UNSLOTH_VERSION = "2026.10.5"


def _torch_constraints(*, unsloth: bool) -> list[str]:
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        version = "2.10.0"
    major_minor = tuple(int(part) for part in version.split(".")[:2])
    minimum = (2, 8) if unsloth else (2, 6)
    if major_minor < minimum:
        raise RuntimeError(f"Selected engine requires a supported torch >= {minimum[0]}.{minimum[1]} runtime; provision its matching CUDA wheel first")
    constraints = [f"torch=={version}"]
    if unsloth:
        # torchao has no torch dependency metadata, but its imports do depend on
        # torch APIs. Match the compatibility window used by upstream notebooks.
        constraints.append("torchao==0.16.0" if major_minor < (2, 11) else "torchao==0.18.0")
    return constraints


def selected_bootstrap_engine(job: JobSpec) -> str:
    from .engines import SoupEngine, UnslothEngine
    if job.engine.type != "auto":
        return job.engine.type
    # An already provisioned NVIDIA environment can be considered by the resolver.
    # Do not install NVIDIA to satisfy auto; Soup remains the generic default.
    if not SoupEngine().unsupported_fields(job):
        return "soup"
    if not UnslothEngine().unsupported_fields(job):
        return "unsloth"
    raise ValueError("No executable engine supports all requested JobSpec fields")


def install_command(job: JobSpec, *, training: bool = False) -> list[str]:
    if isinstance(job, PromptJobSpec):
        if not (3, 10) <= sys.version_info[:2] < (3, 15):
            raise RuntimeError("DSPy GEPA 3.4.1 requires Python 3.10–3.14")
        requirements = ["dspy==3.4.1"]
        if any(isinstance(getattr(job.dataset, split), HFDatasetSpec) for split in ("train", "validation", "test")):
            requirements.append("datasets>=2.14")
        return [sys.executable, "-m", "pip", "install", *requirements]
    engine = selected_bootstrap_engine(job)
    if engine == "nemo_automodel":
        raise ValueError("NeMo AutoModel is opt-in: use an NVIDIA-supported environment/recipe; automatic installation is not provided")
    if engine == "soup":
        from .engines import SoupEngine
        errors = SoupEngine().unsupported_fields(job)
        if errors:
            raise ValueError("; ".join(errors))
        if not (3, 10) <= sys.version_info[:2] < (3, 13):
            raise RuntimeError("Soup 0.76.0 requires Python 3.10–3.12; use a Python 3.12 runtime/venv")
        extras = "[train,fast]" if training and job.engine.backend == "unsloth" else "[train]" if training else ""
        requirements = [f"soup-cli{extras} @ git+https://github.com/MakazhanAlpamys/Soup.git@{SOUP_REF}"]
        if training:
            requirements += _torch_constraints(unsloth=job.engine.backend == "unsloth")
            if job.engine.backend == "unsloth":
                requirements += [f"unsloth=={UNSLOTH_VERSION}", f"unsloth-zoo=={UNSLOTH_VERSION}"]
    elif engine == "unsloth":
        from .engines import UnslothEngine
        errors = UnslothEngine().unsupported_fields(job)
        if errors:
            raise ValueError("; ".join(errors))
        if not training:
            return []
        requirements = [f"unsloth=={UNSLOTH_VERSION}", f"unsloth-zoo=={UNSLOTH_VERSION}", "trl==0.22.2", "transformers==4.57.6", "tokenizers>=0.22.0,<=0.23.0", "datasets", "peft", "accelerate"]
        # Keep a hosted runtime's working torch/CUDA pair while pip finds a
        # matching xformers. Never resolve unconstrained to a CUDA 13 wheel on T4.
        requirements += _torch_constraints(unsloth=True)
    else:
        raise ValueError(f"No installer for {engine}")
    return [sys.executable, "-m", "pip", "install", *requirements]


def bootstrap(job: JobSpec, *, training: bool = False) -> None:
    # A notebook may have imported torch for its hardware probe before pip
    # replaced that wheel. Refuse to train with stale modules in that kernel.
    loaded = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "trl", "numpy") if name in sys.modules}
    command = install_command(job, training=training)
    if command:
        subprocess.run(command, check=True)
    if any(importlib.metadata.version(name) != version for name, version in loaded.items()):
        raise RuntimeError("Training dependencies changed already-imported frameworks; restart the notebook kernel and rerun the cells")
