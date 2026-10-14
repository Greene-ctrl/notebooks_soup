"""Policy selects engines without changing tasks or dropping job settings."""
from .base import TrainingEngine, UnsupportedJob, hardware_errors
from .nemo_automodel import NemoAutoModelEngine
from .soup import SoupEngine
from .unsloth import UnslothEngine


def resolve_engine(job, hardware, *, engines=None, allow_unavailable: bool = False) -> TrainingEngine:
    registry = engines if engines is not None else {
        "soup": SoupEngine(), "unsloth": UnslothEngine(), "nemo_automodel": NemoAutoModelEngine(),
    }
    explicit = job.engine.type != "auto"
    if explicit:
        choices = [job.engine.type]
    else:
        choices = ["soup", "unsloth"]
        if "nemotron" in job.model.id.lower():
            choices.insert(0, "nemo_automodel")
    reasons = []
    for name in choices:
        engine = registry.get(name)
        if engine is None:
            reasons.append(f"{name}: no adapter registered")
            continue
        if not engine.supports(job, hardware):
            details = engine.unsupported_fields(job) + hardware_errors(job, hardware)
            reasons.append(f"{name}: " + "; ".join(details or ["unsupported hardware/configuration"]))
            continue
        if not allow_unavailable and not engine.available():
            reasons.append(f"{name}: selected runtime dependencies are missing or incompatible")
            continue
        return engine
    raise UnsupportedJob("Cannot resolve requested engine/task: " + " | ".join(reasons))


__all__ = ["TrainingEngine", "SoupEngine", "UnslothEngine", "NemoAutoModelEngine", "resolve_engine", "UnsupportedJob"]
