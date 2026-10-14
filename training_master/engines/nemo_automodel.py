"""Opt-in NeMo AutoModel boundary, without untested recipe translation."""
from __future__ import annotations

from .base import TrainingEngine, UnsupportedJob, installed


class NemoAutoModelEngine(TrainingEngine):
    name = "nemo_automodel"

    def available(self) -> bool:
        return installed("nemo_automodel")

    def candidate(self, job, hardware) -> bool:
        # A conservative capability hint, NOT a model fit estimate. NVIDIA's
        # published Nemotron recipes need substantially more than this floor.
        return (
            "nemotron" in job.model.id.lower() and job.task.type == "sft"
            and hardware.cuda_available and hardware.bf16_supported
            and max(hardware.gpu_vram_gb, default=0) >= max(24, job.requirements.min_vram_gb)
            and job.training.quantization == "none" and not job.training.stream_layers
        )

    def unsupported_fields(self, job) -> list[str]:
        return ["NeMo AutoModel recipe translation/execution is not implemented; provision and validate a model-specific NVIDIA recipe separately"]

    def supports(self, job, hardware) -> bool:
        # Keep auto from choosing a detection-only adapter as an executable engine.
        return False

    def dry_run(self, job, hardware, paths=None) -> dict:
        return {"engine": self.name, "installed": self.available(), "hardware_candidate": self.candidate(job, hardware), "executable": False, "limitations": self.unsupported_fields(job)}

    def run(self, job, hardware, paths, *, resume=None, on_checkpoint=None) -> dict:
        raise UnsupportedJob(self.unsupported_fields(job)[0])
