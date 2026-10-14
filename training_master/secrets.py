"""Runtime-managed secrets; do not log tokens or serialize them into jobs."""
from __future__ import annotations

import os
from .platform import RuntimePlatform, detect_platform

SECRET_NAMES = ("HF_TOKEN", "WANDB_API_KEY", "TRAINING_MASTER_TOKEN")


def get_secret(name: str, platform: RuntimePlatform | None = None, *, required: bool = False) -> str | None:
    runtime = platform or detect_platform()
    value = None
    try:
        if runtime.secrets_provider == "colab":
            from google.colab import userdata
            value = userdata.get(name)
        elif runtime.secrets_provider == "kaggle":
            from kaggle_secrets import UserSecretsClient
            value = UserSecretsClient().get_secret(name)
    except Exception:
        # Provider denial or a missing binding can fall back to an injected env var.
        value = None
    value = value or os.environ.get(name) or None
    if required and value is None:
        raise RuntimeError(f"Missing {name}; add it to runtime secrets or environment settings")
    return value


def restore_secrets(platform: RuntimePlatform | None = None) -> dict[str, bool]:
    presence = {}
    for name in SECRET_NAMES:
        value = get_secret(name, platform)
        if value:
            os.environ[name] = value
        presence[name] = bool(value)
    return presence
