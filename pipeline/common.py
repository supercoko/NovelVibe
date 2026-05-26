"""共享工具：配置加载（带 mtime 热重载）、路径、hash 计算。"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
_CFG_PATH = ROOT / "config.yaml"

# 热加载缓存：避免每次调用都解析 YAML
_CACHE: dict[str, Any] = {"mtime": 0.0, "data": None}


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """读取配置；不缓存，每次都读盘。仅供首次或一次性场景。"""
    cfg_path = Path(path) if path else _CFG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_cfg() -> dict[str, Any]:
    """按 mtime 缓存读取。文件改动后下一次调用自动重读。"""
    try:
        mtime = _CFG_PATH.stat().st_mtime
    except FileNotFoundError:
        if _CACHE["data"] is not None:
            return _CACHE["data"]
        raise
    if mtime != _CACHE["mtime"] or _CACHE["data"] is None:
        _CACHE["data"] = load_config(_CFG_PATH)
        _CACHE["mtime"] = mtime
    return _CACHE["data"]


def reload_cfg() -> dict[str, Any]:
    """强制重读配置，无视 mtime 缓存。"""
    _CACHE["mtime"] = 0.0
    _CACHE["data"] = None
    return get_cfg()


def cfg_mtime() -> float:
    """返回当前已缓存配置的 mtime。"""
    return _CACHE["mtime"]


def ensure_dirs(cfg: dict[str, Any] | None = None) -> None:
    cfg = cfg or get_cfg()
    for key in ("cache_dir", "output_dir", "script_dir"):
        Path(ROOT / cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    # 小说持久化目录
    (ROOT / "books").mkdir(parents=True, exist_ok=True)


def text_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


def file_hash(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
