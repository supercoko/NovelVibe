"""LLM 拆分器：把小说章节文本切成 (type, speaker, text) 段。

走 LMStudio 的 OpenAI 兼容接口。维护跨 chunk 的角色字典，便于
同一角色在不同片段保持一致。

提供三种 API：
- chunk_text(text)            : 仅切 chunk，便于 UI 预览总数
- Splitter.split(text)        : 一次性同步切完整章
- Splitter.split_stream(...)  : 生成器，每 chunk 完成 yield 一次，受 controller 控制
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterator

from openai import OpenAI

from .common import ROOT


PROMPT_PATH = ROOT / "prompts" / "splitter.txt"


def list_providers(cfg: dict[str, Any]) -> list[str]:
    """返回所有可选 provider 名称。"""
    llm = cfg.get("llm")
    if isinstance(llm, dict) and isinstance(llm.get("providers"), dict):
        return sorted(llm["providers"].keys())
    if "lmstudio" in cfg:  # 旧 schema 兼容
        return ["lmstudio"]
    return []


def resolve_provider(cfg: dict[str, Any], name: str | None = None) -> dict[str, Any]:
    """根据 cfg + 可选 provider 名称，组装一份完整的连接参数。

    兼容旧 schema (顶层 `lmstudio:`)；新 schema 用 `llm.providers.<name>`。
    所有字符串字段中的 ${ENV_VAR} 都会自动展开。
    """
    import os

    if "llm" in cfg and isinstance(cfg["llm"], dict):
        llm = cfg["llm"]
        providers = llm.get("providers") or {}
        active = name or llm.get("active")
        if active not in providers:
            raise KeyError(f"未知 provider '{active}'，可选: {list(providers)}")
        prov = dict(providers[active])
        prov["name"] = active
        prov.setdefault("temperature", llm.get("temperature", 0.2))
        prov.setdefault("max_chunk_chars", llm.get("max_chunk_chars", 1500))
        prov.setdefault("request_timeout", llm.get("request_timeout", 180))
    elif "lmstudio" in cfg:  # 旧 schema 兼容
        lm = cfg["lmstudio"]
        prov = {
            "name": "lmstudio",
            "base_url": lm["base_url"],
            "api_key": lm.get("api_key", "lm-studio"),
            "model": lm["model"],
            "temperature": lm.get("temperature", 0.2),
            "max_chunk_chars": lm.get("max_chunk_chars", 1500),
            "request_timeout": lm.get("request_timeout", 180),
        }
    else:
        raise KeyError("config.yaml 缺少 llm 或 lmstudio 配置")

    # 展开所有字符串值里的 ${ENV_VAR}
    for k, v in list(prov.items()):
        if isinstance(v, str) and "${" in v:
            expanded = os.path.expandvars(v)
            if expanded == v:  # 没展开成功，多半是 env 没设
                print(f"[provider] 警告：{k}={v!r} 中的环境变量未定义")
            prov[k] = expanded
    return prov


# 8 种情绪标签（与 IndexTTS2 内部一致）
EMOTIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]


# LMStudio / OpenAI 结构化输出 schema
SPLIT_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["narration", "dialogue"]},
                    "speaker": {"type": "string"},
                    "text": {"type": "string"},
                    "emotion": {"type": "string", "enum": EMOTIONS},
                },
                "required": ["type", "speaker", "text", "emotion"],
            },
        },
        "characters": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "note": {"type": "string"},
                },
            },
        },
    },
    "required": ["segments", "characters"],
}


@dataclass
class Segment:
    type: str
    speaker: str
    text: str
    emotion: str = "calm"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class SplitResult:
    segments: list[Segment]
    characters: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ChunkProgress:
    """生成器每 chunk 完成时 yield 的一次进度。"""
    index: int                       # 当前 chunk 下标 (0-based)
    total: int                       # chunk 总数
    new_segments: list[Segment]      # 这一 chunk 新出的段
    all_segments: list[Segment]      # 累计所有段
    characters: dict[str, dict[str, Any]]
    stopped: bool = False            # 是否被用户停止


class SplitController:
    """跨线程暂停 / 停止开关。模块级单例。"""

    def __init__(self) -> None:
        self._pause = threading.Event()
        self._pause.set()  # set = 跑；clear = 暂停
        self._stop = threading.Event()

    def reset(self) -> None:
        self._pause.set()
        self._stop.clear()

    def pause(self) -> None:
        self._pause.clear()

    def resume(self) -> None:
        self._pause.set()

    def stop(self) -> None:
        self._stop.set()
        self._pause.set()  # 唤醒被 pause 卡住的 wait

    @property
    def is_paused(self) -> bool:
        return not self._pause.is_set()

    @property
    def is_stopped(self) -> bool:
        return self._stop.is_set()

    def wait_if_paused(self) -> None:
        """阻塞直到被 resume 或 stop。"""
        self._pause.wait()


# 模块级单例：一次只允许一个 split job 在跑（UI 也只暴露一个按钮）
CONTROLLER = SplitController()

# TTS 合成专用控制器（与 LLM 拆分独立）
TTS_CONTROLLER = SplitController()


def chunk_text(text: str, max_chars: int) -> list[str]:
    """按段落聚合，单段超过上限则按句号再切。"""
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    cur = 0
    for p in paragraphs:
        if cur + len(p) > max_chars and buf:
            chunks.append("\n".join(buf))
            buf, cur = [], 0
        if len(p) > max_chars:
            sentences = re.split(r"(?<=[。！？!?])", p)
            for s in sentences:
                if not s.strip():
                    continue
                if cur + len(s) > max_chars and buf:
                    chunks.append("\n".join(buf))
                    buf, cur = [], 0
                buf.append(s)
                cur += len(s)
        else:
            buf.append(p)
            cur += len(p)
    if buf:
        chunks.append("\n".join(buf))
    return chunks


# 旧名兼容
_chunk_text = chunk_text


def _extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).rsplit("```", 1)[0]
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e != -1 and e > s:
        raw = raw[s:e + 1]
    return json.loads(raw)


class Splitter:
    def __init__(self, cfg: dict[str, Any], provider_name: str | None = None):
        provider = resolve_provider(cfg, provider_name)
        self.client = OpenAI(
            base_url=provider["base_url"],
            api_key=provider["api_key"] or "sk-placeholder",
            timeout=provider["request_timeout"],
        )
        self.model = provider["model"]
        self.temperature = provider["temperature"]
        self.max_chunk_chars = provider["max_chunk_chars"]
        self.provider_name = provider["name"]
        self.prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    def _call_llm(self, chunk: str, known_chars: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            self.prompt_template
            .replace("{known_characters_json}", json.dumps(known_chars, ensure_ascii=False))
            .replace("{chunk}", chunk)
        )
        response_formats = [
            {"type": "json_schema",
             "json_schema": {"name": "novel_split", "schema": SPLIT_SCHEMA}},
            {"type": "text"},
        ]
        last_err: Exception | None = None
        for response_format in response_formats:
            for attempt in range(2):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        response_format=response_format,
                        messages=[
                            {"role": "system", "content": "你严格按用户指定的 JSON schema 输出，不输出任何额外字符。"},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    content = resp.choices[0].message.content or ""
                    return _extract_json(content)
                except Exception as e:
                    last_err = e
                    if "response_format" in str(e) or "400" in str(e):
                        break
        print(f"[splitter] LLM 调用失败，整段降级为旁白：{last_err}")
        return {
            "segments": [{"type": "narration", "speaker": "narrator", "text": chunk}],
            "characters": {},
        }

    @staticmethod
    def _merge_characters(global_chars: dict[str, dict[str, Any]],
                          new_chars: dict[str, dict[str, Any]]) -> None:
        for name, info in new_chars.items():
            if name not in global_chars:
                global_chars[name] = {"aliases": [], "note": ""}
            cur = global_chars[name]
            cur.setdefault("aliases", [])
            cur.setdefault("note", "")
            for alias in info.get("aliases", []) or []:
                if alias and alias not in cur["aliases"]:
                    cur["aliases"].append(alias)
            if info.get("note") and not cur.get("note"):
                cur["note"] = info["note"]

    @staticmethod
    def _normalize_segments(raw_segments: list[dict[str, Any]]) -> list[Segment]:
        out: list[Segment] = []
        for seg in raw_segments or []:
            t = seg.get("type", "narration")
            if t not in ("narration", "dialogue"):
                t = "narration"
            speaker = (seg.get("speaker") or "").strip() or "narrator"
            if t == "narration":
                speaker = "narrator"
            text_v = (seg.get("text") or "").strip()
            emotion = (seg.get("emotion") or "calm").strip().lower()
            if emotion not in EMOTIONS:
                emotion = "calm"
            if text_v:
                out.append(Segment(type=t, speaker=speaker, text=text_v, emotion=emotion))
        return out

    def split(self, text: str) -> SplitResult:
        """一次性同步拆分（保留旧 API）。"""
        result = SplitResult(segments=[], characters={})
        for prog in self.split_stream(text):
            result.segments = prog.all_segments
            result.characters = prog.characters
        return result

    def split_stream(
        self,
        text: str,
        *,
        start_chunk: int = 0,
        existing_segments: list[Segment] | None = None,
        existing_characters: dict[str, dict[str, Any]] | None = None,
        controller: SplitController | None = None,
    ) -> Iterator[ChunkProgress]:
        """生成器：每 chunk 完成 yield 一次 ChunkProgress。

        - start_chunk:    跳过前 N 个 chunk（续跑场景）
        - existing_*:     已有的累计段 / 角色字典，作为续跑起点
        - controller:     暂停/停止开关；默认用模块单例 CONTROLLER
        """
        ctrl = controller or CONTROLLER
        chunks = chunk_text(text, self.max_chunk_chars)
        all_segments: list[Segment] = list(existing_segments or [])
        characters: dict[str, dict[str, Any]] = dict(existing_characters or {})

        for i, chunk in enumerate(chunks):
            if i < start_chunk:
                continue
            # 每个 chunk 开始前检查暂停 / 停止
            ctrl.wait_if_paused()
            if ctrl.is_stopped:
                yield ChunkProgress(
                    index=i - 1, total=len(chunks),
                    new_segments=[], all_segments=all_segments,
                    characters=characters, stopped=True,
                )
                return

            print(f"[splitter] chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
            data = self._call_llm(chunk, characters)
            new_segs = self._normalize_segments(data.get("segments", []) or [])
            all_segments.extend(new_segs)
            self._merge_characters(characters, data.get("characters", {}) or {})

            yield ChunkProgress(
                index=i, total=len(chunks),
                new_segments=new_segs, all_segments=all_segments,
                characters=characters, stopped=False,
            )


# ---------------------------------------------------------------------------
# 持久化
# ---------------------------------------------------------------------------

def save_script(result: SplitResult, path: str | Path,
                progress: dict[str, int] | None = None) -> None:
    data: dict[str, Any] = {
        "characters": result.characters,
        "segments": [s.to_dict() for s in result.segments],
    }
    if progress is not None:
        data["progress"] = progress
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_script(path: str | Path) -> SplitResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    segs = []
    for s in data.get("segments", []):
        emotion = (s.get("emotion") or "calm").strip().lower()
        if emotion not in EMOTIONS:
            emotion = "calm"
        segs.append(Segment(
            type=s.get("type", "narration"),
            speaker=s.get("speaker", "narrator"),
            text=s.get("text", ""),
            emotion=emotion,
        ))
    return SplitResult(segments=segs, characters=data.get("characters", {}))


def load_progress(path: str | Path) -> dict[str, int]:
    """读取脚本里的 progress 字段（done / total），不存在返回空。"""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("progress", {}) or {}
    except Exception:
        return {}
