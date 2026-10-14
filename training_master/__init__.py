"""Portable execution helpers; imports never install or load training frameworks."""
from .job import JobSpec, load_job

__all__ = ["JobSpec", "load_job"]
__version__ = "0.1.0"
