"""IndexTTS2 封装：单例加载 + (text, ref_audio) hash 缓存。

第一次 import 会很慢（加载模型到 GPU），后续 infer 复用。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .common import ROOT, text_hash, file_hash


_TTS_INSTANCE = None  # 全局单例


def _ensure_import_path(repo_dir: str) -> None:
    rp = str(Path(repo_dir).resolve())
    if rp not in sys.path:
        sys.path.insert(0, rp)


def get_tts(cfg: dict[str, Any]):
    """惰性加载 IndexTTS2 单例。"""
    global _TTS_INSTANCE
    if _TTS_INSTANCE is not None:
        return _TTS_INSTANCE

    it_cfg = cfg["indextts"]
    _ensure_import_path(it_cfg["repo_dir"])
    # IndexTTS2 内部用相对路径 ./checkpoints/hf_cache，切到 repo_dir
    cwd = os.getcwd()
    os.chdir(it_cfg["repo_dir"])
    try:
        from indextts.infer_v2 import IndexTTS2  # type: ignore
        _TTS_INSTANCE = IndexTTS2(
            cfg_path=it_cfg["cfg_path"],
            model_dir=it_cfg["model_dir"],
            use_fp16=it_cfg.get("use_fp16", True),
            device=it_cfg.get("device"),
            use_cuda_kernel=it_cfg.get("use_cuda_kernel"),
        )
    finally:
        os.chdir(cwd)
    return _TTS_INSTANCE


class TTSEngine:
    """对外接口：synth(text, ref_audio) -> wav 路径，自动缓存。"""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.cache_dir = Path(ROOT / cfg["paths"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ref_hash_cache: dict[str, str] = {}

    def _ref_hash(self, ref_audio: str) -> str:
        if ref_audio not in self._ref_hash_cache:
            self._ref_hash_cache[ref_audio] = file_hash(ref_audio)
        return self._ref_hash_cache[ref_audio]

    def synth(self, text: str, ref_audio: str) -> Path:
        ref_audio = str(Path(ref_audio).resolve())
        key = text_hash(text, self._ref_hash(ref_audio))
        out_path = self.cache_dir / f"{key}.wav"
        if out_path.exists():
            return out_path
        tts = get_tts(self.cfg)
        tts.infer(
            spk_audio_prompt=ref_audio,
            text=text,
            output_path=str(out_path),
            verbose=False,
        )
        if not out_path.exists():
            raise RuntimeError(f"IndexTTS2 未生成音频：text={text[:30]!r}")
        return out_path
