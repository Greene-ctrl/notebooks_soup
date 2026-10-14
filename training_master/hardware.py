"""Best-effort CPU/GPU probe with JSON-compatible results and no mandatory torch."""
from __future__ import annotations

import csv
import importlib
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .platform import RuntimePlatform, detect_platform


@dataclass(frozen=True)
class HardwareProfile:
    platform: str = "local"
    gpu_count: int = 0
    gpu_names: list[str] = field(default_factory=list)
    gpu_vram_gb: list[float] = field(default_factory=list)
    cuda_available: bool = False
    torch_version: str | None = None
    cuda_version: str | None = None
    bf16_supported: bool = False
    disk_free_gb: float = 0
    ram_gb: float = 0
    architecture: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def probe_hardware(platform: RuntimePlatform | None = None) -> HardwareProfile:
    import platform as host_platform
    runtime = platform or detect_platform()
    info = HardwareProfile(platform=runtime.name, architecture=host_platform.machine()).to_dict()
    disk = runtime.working_dir if runtime.working_dir.exists() else Path.cwd()
    info["disk_free_gb"] = round(shutil.disk_usage(disk).free / 1024**3, 2)
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 2)
    except (ValueError, OSError, AttributeError):
        info["warnings"].append("RAM size unavailable")
    try:
        torch = importlib.import_module("torch")
        info["torch_version"] = str(torch.__version__)
        info["cuda_version"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(info["gpu_count"])]
            info["gpu_vram_gb"] = [round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2) for i in range(info["gpu_count"])]
            major = getattr(torch.cuda.get_device_properties(0), "major", None)
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported()) and (major is None or major >= 8)
    except Exception:
        info.update(gpu_count=0, gpu_names=[], gpu_vram_gb=[], cuda_available=False, bf16_supported=False)
        info["warnings"].append("PyTorch unavailable or its CUDA probe failed")
    if not info["gpu_count"]:
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
            rows = list(csv.reader(result.stdout.splitlines()))
            info["gpu_names"] = [r[0].strip() for r in rows]
            info["gpu_vram_gb"] = [round(float(r[1]) / 1024, 2) for r in rows]
            info["gpu_count"] = len(rows)
            # Driver evidence is not proof the installed torch can use CUDA.
            if rows:
                info["warnings"].append("Driver sees GPUs; verify CUDA with PyTorch before training")
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
    return HardwareProfile(**info)
