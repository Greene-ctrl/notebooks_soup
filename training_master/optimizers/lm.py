"""Independent student/reflection LM settings and secret lookup."""
from __future__ import annotations

import os
import re

from ..job import LMSpec, PromptJobSpec
from ..secrets import get_secret


def apply_lm_environment(job: PromptJobSpec, environ=None) -> PromptJobSpec:
    env = os.environ if environ is None else environ
    data = job.model_dump(mode="json")
    for role in ("student", "reflection"):
        prefix = role.upper()
        for setting in ("model", "api_base"):
            name = f"{prefix}_{setting.upper()}"
            if name in env:
                data[role][setting] = env[name] or None
        # Credential values NEVER enter the JobSpec or generated config.
        if env.get(f"{prefix}_API_KEY"):
            data[role]["api_key_secret"] = f"{prefix}_API_KEY"
    return PromptJobSpec.model_validate(data)


def validate_lm(spec: LMSpec) -> None:
    # Match the installed DSPy 3.4.1 constructor's reasoning-model restrictions.
    family = spec.model.split("/")[-1].lower()
    reasoning = re.match(r"^(?:o[1345](?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?|gpt-5(?:\.\d+)*(?!-chat)(?:-.*)?)$", family)
    if reasoning and (spec.temperature not in (None, 1) or spec.max_tokens < 16000):
        raise ValueError("DSPy reasoning LMs require temperature=None/1 and an explicit max_tokens>=16000; select a cheaper reflection model for the small smoke")


def create_lm(spec: LMSpec, *, platform=None):
    import dspy
    validate_lm(spec)
    key = get_secret(spec.api_key_secret, platform, required=True) if spec.api_key_secret else None
    if key is None and spec.api_base is None:
        raise ValueError("Cloud LM requires api_key_secret; keyless endpoints must set api_base explicitly")
    # cache=False avoids credential-bearing request caches and makes observed
    # usage easier to interpret. No global dspy.configure / provider mutation.
    return dspy.LM(
        spec.model, api_base=spec.api_base, api_key=key if key is not None else "",
        temperature=spec.temperature, max_tokens=spec.max_tokens,
        num_retries=0, cache=False,
    )


def lm_usage(lm) -> dict:
    entries = list(getattr(lm, "history", []))
    totals = {}
    for entry in entries:
        usage = entry.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
    return {"observed_history_entries": len(entries), "tokens": totals, "note": "Observed LM history only; SDK history can be truncated. No dollar estimate."}
