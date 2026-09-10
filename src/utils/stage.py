from __future__ import annotations

import json
import logging
import platform
import socket
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .config import PROJECT_ROOT
from .hashing import sha256_file, sha256_paths


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def source_files() -> list[Path]:
    result: list[Path] = []
    for root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", PROJECT_ROOT / "configs"):
        if root.exists():
            result.extend(p for p in root.rglob("*") if p.is_file())
    result.extend(p for p in (PROJECT_ROOT / "README.md", PROJECT_ROOT / "environment.yml") if p.exists())
    return result


class StageBlocked(RuntimeError):
    pass


class StageContext:
    def __init__(self, number: int, name: str, force: bool = False):
        self.number = number
        self.name = name
        self.force = force
        self.state_dir = PROJECT_ROOT / "state"
        self.log_dir = PROJECT_ROOT / "logs"
        self.done_path = self.state_dir / f"stage_{number:02d}_{name}.done"
        self.config_paths = list((PROJECT_ROOT / "configs").glob("*.yaml"))
        self.source_hash = sha256_paths(source_files())
        self.config_hash = sha256_paths(self.config_paths)
        self.started = utc_now()
        self.logger = logging.getLogger(f"stage_{number:02d}_{name}")

    def valid_done(self) -> bool:
        if self.force or not self.done_path.exists():
            return False
        try:
            state = json.loads(self.done_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return state.get("source_hash") == self.source_hash and state.get("config_hash") == self.config_hash

    def mark_done(self, outputs: list[Path], inputs: list[Path], warnings: list[str] | None = None) -> None:
        missing = [str(p) for p in outputs if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Cannot mark stage done; outputs missing: {missing}")
        output_hash = sha256_paths(outputs)
        input_hash = sha256_paths(p for p in inputs if p.is_file())
        payload = {
            "stage": self.number,
            "name": self.name,
            "status": "done",
            "start_time_utc": self.started,
            "end_time_utc": utc_now(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "git_commit": git_commit(),
            "config_hash": self.config_hash,
            "source_hash": self.source_hash,
            "input_hash": input_hash,
            "inputs": [str(p) for p in inputs],
            "output_hash": output_hash,
            "outputs": [str(p) for p in outputs],
            "warnings": warnings or [],
        }
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.done_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@contextmanager
def stage_context(number: int, name: str, force: bool = False) -> Iterator[StageContext | None]:
    ctx = StageContext(number, name, force)
    ctx.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = ctx.log_dir / f"stage_{number:02d}_{name}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    ctx.logger.handlers.clear()
    ctx.logger.addHandler(handler)
    ctx.logger.setLevel(logging.INFO)
    if ctx.valid_done():
        print(f"Stage {number:02d} {name}: SKIP (matching .done state)")
        yield None
        return
    ctx.logger.info("start host=%s python=%s git=%s", socket.gethostname(), sys.version, git_commit())
    try:
        yield ctx
    except Exception:
        ctx.logger.exception("stage failed")
        raise
    finally:
        handler.close()
        ctx.logger.removeHandler(handler)

