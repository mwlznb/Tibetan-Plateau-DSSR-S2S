from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(name: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "configs" / name
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a mapping")
    return value


def resolved_paths() -> dict[str, Any]:
    cfg = load_yaml("paths.yaml")
    cfg["code_root"] = PROJECT_ROOT
    cfg["work_root"] = (PROJECT_ROOT / cfg.get("work_root", "outputs")).resolve()
    cfg["env_python"] = None
    return cfg
