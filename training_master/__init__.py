"""Portable execution helpers; imports never install or load training frameworks."""
from .job import JobSpec, PromptJobSpec, WorkflowJobSpec, load_job

__all__ = ["JobSpec", "PromptJobSpec", "WorkflowJobSpec", "load_job"]
__version__ = "0.1.0"
