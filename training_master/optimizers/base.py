"""Prompt optimizer contract, deliberately separate from TrainingEngine."""
from abc import ABC, abstractmethod


class OptimizerEngine(ABC):
    name: str

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def supports(self, job) -> bool: ...

    @abstractmethod
    def prepare(self, job, **kwargs): ...

    @abstractmethod
    def dry_run(self, job, **kwargs) -> dict: ...

    @abstractmethod
    def run(self, job, **kwargs) -> dict: ...
