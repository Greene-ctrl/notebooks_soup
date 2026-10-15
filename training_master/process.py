"""Live, ordered subprocess logs with bounded errors and durable command records."""
from __future__ import annotations

import os
import queue
import re
import shlex
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import write_json


def redact(text: str) -> str:
    # Provider secrets restored into the environment are never written to logs.
    for name, value in os.environ.items():
        if value and len(value) >= 4 and any(part in name.upper() for part in ("TOKEN", "API_KEY", "SECRET", "PASSWORD")):
            text = text.replace(value, "[REDACTED]")
    text = re.sub(r"\b(?:hf_|sk-|wandb_)[A-Za-z0-9_-]{12,}", "[REDACTED]", text)
    text = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", text)
    text = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[REDACTED]", text)
    text = re.sub(r"(?i)(authorization:\s*bearer\s+)\S+", r"\1[REDACTED]", text)
    return text


@dataclass(frozen=True)
class ProcessResult:
    command: list[str]
    exit_code: int | None
    stdout_tail: str
    stderr_tail: str  # Empty: stderr is merged into stdout to retain ordering.
    log_path: str
    duration: float


class EngineProcessError(RuntimeError):
    def __init__(self, result: ProcessResult):
        self.__dict__.update(result.__dict__)
        super().__init__(f"Engine execution failed.\n\nCommand:\n{shlex.join(result.command)}\n\nExit code: {result.exit_code}\nLog: {result.log_path}\n\nLast output:\n{'-' * 50}\n{result.stdout_tail}\n{'-' * 50}")


def run_process(argv: list[str], *, cwd: Path, log_path: Path, record_path: Path,
                tail_lines: int = 100, tail_chars: int = 24000, on_poll=None, stream: bool = True) -> ProcessResult:
    """Run with explicit cwd, combined UTF-8 output, no environment dump.

    Full sanitized logs stream live; exceptions contain only the bounded tail.
    Poll callbacks run even during quiet commands. Cancellation terminates then
    kills the child, joins the reader and records its final exit code.
    """
    if tail_lines < 1 or tail_chars < 1:
        raise ValueError("Tail limits must be positive")
    cwd = cwd.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    tail = deque(maxlen=tail_lines)
    safe_argv = [redact(a) for a in argv]
    records = {"argv": safe_argv, "cwd": str(cwd), "started_at": started_at, "log_path": str(log_path), "stderr": "merged"}
    write_json(record_path, {**records, "finished_at": None, "exit_code": None})
    process = None
    reader = None
    pending = queue.Queue()
    interrupted = False
    error_type = None
    failure = None
    last_poll = 0.0
    code = None

    def read_output():
        try:
            for line in process.stdout:
                pending.put(line)
        finally:
            pending.put(None)

    def emit(line, log):
        line = redact(line)
        log.write(line)
        log.flush()
        for part in line.splitlines():
            tail.append(part[-tail_chars:])
        # Bound total stored text even when individual lines are enormous.
        while len(tail) > 1 and sum(len(s) + 1 for s in tail) > tail_chars:
            tail.popleft()
        if stream:
            print(line, end="", flush=True)

    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1, start_new_session=os.name == "posix")
            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            done = False
            while not done:
                try:
                    line = pending.get(timeout=0.2)
                    if line is None:
                        done = True
                    else:
                        emit(line, log)
                except queue.Empty:
                    pass
                if on_poll and time.monotonic() - last_poll >= 0.2:
                    on_poll()
                    last_poll = time.monotonic()
            code = process.wait()
        except BaseException as error:
            failure = error
            interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
            error_type = type(error).__name__
            raise
        finally:
            if process is not None:
                if process.poll() is None:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                        process.wait()
                code = process.returncode
                if reader:
                    reader.join(timeout=5)
                while not pending.empty():
                    line = pending.get_nowait()
                    if line is not None:
                        emit(line, log)
                process.stdout.close()
            duration = time.monotonic() - start
            records.update(finished_at=datetime.now(timezone.utc).isoformat(), duration_seconds=duration,
                           exit_code=code, interrupted=interrupted, error_type=error_type)
            write_json(record_path, records)
            if failure is not None:
                failure.engine_process_result = ProcessResult(safe_argv, code, "\n".join(tail)[-tail_chars:], "", str(log_path), duration)
    result = ProcessResult(safe_argv, code, "\n".join(tail)[-tail_chars:], "", str(log_path), duration)
    if code:
        raise EngineProcessError(result) from subprocess.CalledProcessError(code, safe_argv)
    return result
