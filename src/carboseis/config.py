"""Scenario files (TOML) shared by the Julia driver and CarboSeis."""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path

_PATH_KEYS = {"output", "sea_level_file"}


def _resolve(d, root: Path):
    for k, v in d.items():
        if isinstance(v, dict):
            _resolve(v, root)
        elif k in _PATH_KEYS and isinstance(v, str) and not Path(v).is_absolute():
            d[k] = str(root / v)


def set_dotted(d, key: str, value):
    parts = key.split(".")
    for p in parts[:-1]:
        d = d[int(p)] if isinstance(d, list) else d[p]
    last = parts[-1]
    if isinstance(d, list):
        d[int(last)] = value
    else:
        d[last] = value


def get_dotted(d: dict, key: str):
    for p in key.split("."):
        d = d[int(p)] if isinstance(d, list) else d[p]
    return d


def load_scenario(path: str | Path, overrides: dict | None = None) -> dict:
    """Load a scenario; paths are resolved relative to the repository root
    (the parent of the ``scenarios`` folder).  ``overrides`` maps dotted keys,
    e.g. ``{"diagenesis.meteoric": False}``, to new values."""
    path = Path(path).resolve()
    with open(path, "rb") as f:
        sc = tomllib.load(f)
    sc = copy.deepcopy(sc)
    root = path.parent.parent if path.parent.name == "scenarios" else path.parent
    _resolve(sc, root)
    sc["_root"] = str(root)
    for k, v in (overrides or {}).items():
        set_dotted(sc, k, v)
    return sc
