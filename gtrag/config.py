"""Config loading: configs/default.yaml -> dataset YAML -> command-line overrides."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO_ROOT / "configs" / "default.yaml"


def resolve_repo_path(path: str | Path) -> Path:
    """Relative paths (e.g. the prompt directory) are taken relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


class Config(dict):
    """dict with attribute access: cfg.chunking.chunk_size"""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    return obj


def _unwrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _unwrap(v) for k, v in obj.items()}
    return obj


def _deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_by_path(cfg: dict, dotted: str, raw_value: str) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"Unknown config section: {dotted}")
        node = node[k]
    if keys[-1] not in node:  # catches typos instead of silently adding keys
        raise KeyError(f"Unknown config key: {dotted}")
    node[keys[-1]] = yaml.safe_load(raw_value)  # parses 3 -> int, 0.5 -> float, true -> bool


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> Config:
    with open(DEFAULT_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    for item in overrides:
        key, _, value = item.partition("=")
        _set_by_path(cfg, key.strip(), value)
    return _wrap(cfg)


def save_config(cfg: Config, out_dir: str | Path, name: str = "resolved_config.yaml") -> Path:
    """Write the fully resolved config next to the results for reproducibility."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / name
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(_unwrap(cfg), f, sort_keys=False)
    return target


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="dataset YAML, e.g. configs/hotpotqa.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="override, e.g. chunking.chunk_size=1000")
    return p
