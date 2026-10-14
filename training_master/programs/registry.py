"""Small explicit DSPy registry; no dynamic Python imports or executable signatures."""
from __future__ import annotations

import re
from typing import Callable


def signature_fields(signature: str) -> tuple[list[str], str]:
    sides = signature.split("->")
    if len(sides) != 2:
        raise ValueError("program.signature must be 'input[, input] -> output'")
    inputs = [x.strip() for x in sides[0].split(",")]
    outputs = [x.strip() for x in sides[1].split(",")]
    fields = inputs + outputs
    if len(outputs) != 1 or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in fields) or len(set(fields)) != len(fields):
        raise ValueError("Initial programs require distinct untyped text inputs and one text output")
    return inputs, outputs[0]


# Deliberately empty. A future caller may explicitly register trusted tools here;
# JobSpecs can name them, but cannot import arbitrary functions or code.
TOOL_REGISTRY: dict[str, Callable] = {}


def build_program(spec, *, dspy_module=None):
    if dspy_module is None:
        import dspy as dspy_module
    signature_fields(spec.signature)
    if spec.type == "predict":
        return dspy_module.Predict(spec.signature)
    if spec.type == "chain_of_thought":
        return dspy_module.ChainOfThought(spec.signature)
    if spec.type == "react":
        unknown = set(spec.tools) - TOOL_REGISTRY.keys()
        if unknown:
            raise ValueError("Unknown program tool names: " + ", ".join(sorted(unknown)))
        return dspy_module.ReAct(spec.signature, tools=[TOOL_REGISTRY[name] for name in spec.tools], max_iters=spec.max_iters)
    raise ValueError(f"Unknown program type: {spec.type}")


def save_program(program, path) -> None:
    """Save portable JSON state with no bound LM, credentials, or pickle."""
    portable = program.deepcopy()
    portable.set_lm(None)
    path.parent.mkdir(parents=True, exist_ok=True)
    portable.save(str(path), save_program=False)


def load_program(spec, path, *, lm=None, dspy_module=None):
    program = build_program(spec, dspy_module=dspy_module)
    program.load(str(path), allow_pickle=False, allow_unsafe_lm_state=False)
    if lm is not None:
        program.set_lm(lm)
    return program
