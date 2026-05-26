"""IndexTTS2 封装：单例加载 + (text, ref_audio, emotion) hash 缓存 + 显式卸载。

第一次 synth 会触发模型加载到 GPU；调用 unload_tts() 可释放显存/内存。
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Any

from .common import ROOT, text_hash, file_hash


# IndexTTS2 内部向量顺序：高兴/愤怒/悲伤/恐惧/反感/低落/惊讶/自然
EMO_ORDER = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]

# 情绪标签 → 8 维向量。强度 0.7 既明显又留 calm 基底，避免破坏性发音。
# IndexTTS2 内部要求 sum <= 0.8（normalize_emo_vec 会收敛）。
_BASE_VEC = [0.0] * 8
EMOTION_VECTORS: dict[str, list[float]] = {}
for i, label in enumerate(EMO_ORDER):
    v = list(_BASE_VEC)
    if label == "calm":
        v[7] = 0.6        # 中性时给 calm 一些权重
    else:
        v[i] = 0.7
        v[7] = 0.1        # 其他情绪叠 calm 基底，发音稳一点
    EMOTION_VECTORS[label] = v


def emotion_to_vector(emotion: str | None) -> list[float] | None:
    """未知 / 空 / 'calm' 返回 None（让 IndexTTS2 走默认参考音色，最快路径）。"""
    if not emotion:
        return None
    e = emotion.strip().lower()
    if e == "calm" or e not in EMOTION_VECTORS:
        return None
    return EMOTION_VECTORS[e]


_TTS_INSTANCE = None  # 全局单例


def _ensure_import_path(repo_dir: str) -> None:
    rp = str(Path(repo_dir).resolve())
    if rp not in sys.path:
        sys.path.insert(0, rp)


def is_loaded() -> bool:
    return _TTS_INSTANCE is not None


def get_tts(cfg: dict[str, Any]):
    """惰性加载 IndexTTS2 单例。"""
    global _TTS_INSTANCE
    if _TTS_INSTANCE is not None:
        return _TTS_INSTANCE

    it_cfg = cfg["indextts"]
    _ensure_import_path(it_cfg["repo_dir"])
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


def unload_tts() -> dict[str, Any]:
    """卸载 IndexTTS2 单例，释放 GPU 显存和大部分系统内存。

    返回卸载前后的 GPU 显存占用（如可读到），供 UI 显示。
    """
    global _TTS_INSTANCE
    info: dict[str, Any] = {"was_loaded": _TTS_INSTANCE is not None}
    if _TTS_INSTANCE is None:
        info["msg"] = "模型本就未加载"
        return info

    # 尝试拿一份卸载前显存
    before_mb = None
    try:
        import torch
        if torch.cuda.is_available():
            before_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    except Exception:
        pass

    # 把 IndexTTS2 内部缓存清掉，再释放各模块到 CPU/None
    inst = _TTS_INSTANCE
    try:
        # 各子模块尝试 .to('cpu') + del；属性名按 infer_v2.py 实际命名
        for attr in ("gpt", "semantic_model", "semantic_codec", "s2mel", "bigvgan",
                     "campplus", "qwen_emo", "extract_features"):
            obj = getattr(inst, attr, None)
            if obj is None:
                continue
            try:
                if hasattr(obj, "to"):
                    obj.to("cpu")
            except Exception:
                pass
            try:
                setattr(inst, attr, None)
            except Exception:
                pass
        # IndexTTS2 内部参考音色缓存
        for cache_attr in ("cache_spk_cond", "cache_s2mel_style", "cache_s2mel_prompt",
                           "cache_mel", "cache_spk_audio_prompt"):
            try:
                setattr(inst, cache_attr, None)
            except Exception:
                pass
    finally:
        _TTS_INSTANCE = None
        del inst
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    after_mb = None
    try:
        import torch
        if torch.cuda.is_available():
            after_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    except Exception:
        pass

    info["before_mb"] = before_mb
    info["after_mb"] = after_mb
    info["msg"] = "已卸载"
    return info


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

    def synth(self, text: str, ref_audio: str, emotion: str | None = None) -> Path:
        ref_audio = str(Path(ref_audio).resolve())
        emo_tag = (emotion or "calm").strip().lower()
        if emo_tag not in EMOTION_VECTORS:
            emo_tag = "calm"
        # 缓存 key 包含情绪：换情绪自动生成新音频
        key = text_hash(text, self._ref_hash(ref_audio), emo_tag)
        out_path = self.cache_dir / f"{key}.wav"
        if out_path.exists():
            return out_path
        tts = get_tts(self.cfg)
        emo_vector = emotion_to_vector(emo_tag)
        kwargs: dict[str, Any] = {
            "spk_audio_prompt": ref_audio,
            "text": text,
            "output_path": str(out_path),
            "verbose": False,
        }
        if emo_vector is not None:
            kwargs["emo_vector"] = emo_vector
        tts.infer(**kwargs)
        if not out_path.exists():
            raise RuntimeError(f"IndexTTS2 未生成音频：text={text[:30]!r}")
        return out_path
