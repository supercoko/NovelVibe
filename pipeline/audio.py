"""音频拼接：插入段落停顿、对话切换停顿，导出 mp3。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from pydub import AudioSegment

from .common import ROOT
from .splitter import Segment
from .tts import TTSEngine
from .voice_mapper import VoiceMapper


def _pause_between(prev: Segment | None, cur: Segment, cfg_audio: dict[str, Any]) -> int:
    """根据前后段类型，决定中间插入的静音毫秒。"""
    if prev is None:
        return 0
    # 对话切换：两条 dialogue 且说话人不同
    if prev.type == "dialogue" and cur.type == "dialogue" and prev.speaker != cur.speaker:
        return cfg_audio.get("pause_dialogue_switch", 600)
    # 旁白 <-> 对话切换
    if prev.type != cur.type:
        return cfg_audio.get("pause_dialogue_switch", 600)
    return cfg_audio.get("pause_paragraph", 500)


def synthesize_chapter(
    segments: list[Segment],
    tts: TTSEngine,
    voice: VoiceMapper,
    cfg: dict[str, Any],
    progress_cb=None,
) -> AudioSegment:
    """逐段合成并拼接为完整章节 AudioSegment。"""
    audio_cfg = cfg.get("audio", {}) or {}
    combined = AudioSegment.silent(duration=0)
    prev: Segment | None = None
    total = len(segments)
    for i, seg in enumerate(segments):
        ref = voice.get(seg.speaker)
        wav_path = tts.synth(seg.text, ref, getattr(seg, "emotion", "calm"))
        clip = AudioSegment.from_file(str(wav_path))
        pause = _pause_between(prev, seg, audio_cfg)
        if pause > 0:
            combined += AudioSegment.silent(duration=pause)
        combined += clip
        prev = seg
        if progress_cb:
            progress_cb(i + 1, total, seg)
    return combined


def export(audio: AudioSegment, out_path: str | Path, cfg: dict[str, Any]) -> Path:
    audio_cfg = cfg.get("audio", {}) or {}
    fmt = audio_cfg.get("output_format", "mp3")
    bitrate = audio_cfg.get("bitrate", "128k")
    out = Path(out_path).with_suffix(f".{fmt}")
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"format": fmt}
    if fmt == "mp3":
        kwargs["bitrate"] = bitrate
    audio.export(str(out), **kwargs)
    return out
