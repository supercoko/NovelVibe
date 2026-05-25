"""LLM 拆分器：把小说章节文本切成 (type, speaker, text) 段。

走 LMStudio 的 OpenAI 兼容接口。维护跨 chunk 的角色字典，便于
同一角色在不同片段保持一致。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from .common import ROOT


PROMPT_PATH = ROOT / "prompts" / "splitter.txt"


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
                },
                "required": ["type", "speaker", "text"],
            },
        },
        "characters": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "gender": {"type": "string"},
                    "note": {"type": "string"},
                },
            },
        },
    },
    "required": ["segments", "characters"],
}


@dataclass
class Segment:
    type: str           # "narration" | "dialogue"
    speaker: str        # "narrator" 或 角色名
    text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class SplitResult:
    segments: list[Segment]
    characters: dict[str, dict[str, Any]] = field(default_factory=dict)


def _chunk_text(text: str, max_chars: int) -> list[str]:
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


def _extract_json(raw: str) -> dict[str, Any]:
    """LLM 偶尔会带 ```json 包裹或前后多话，做容错。"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).rsplit("```", 1)[0]
    # 兜底：取首个 { 到末尾 } 之间
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e != -1 and e > s:
        raw = raw[s:e + 1]
    return json.loads(raw)


class Splitter:
    def __init__(self, cfg: dict[str, Any]):
        lm = cfg["lmstudio"]
        self.client = OpenAI(base_url=lm["base_url"], api_key=lm.get("api_key", "lm-studio"),
                             timeout=lm.get("request_timeout", 180))
        self.model = lm["model"]
        self.temperature = lm.get("temperature", 0.2)
        self.max_chunk_chars = lm.get("max_chunk_chars", 1500)
        self.prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    def _call_llm(self, chunk: str, known_chars: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            self.prompt_template
            .replace("{known_characters_json}", json.dumps(known_chars, ensure_ascii=False))
            .replace("{chunk}", chunk)
        )
        # LMStudio 支持 json_schema；不支持时退回 text 模式并由 _extract_json 兜底
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
                    msg = str(e)
                    # 400: 当前后端不支持该 response_format，立刻换下一种
                    if "response_format" in msg or "400" in msg:
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
                global_chars[name] = {"aliases": [], "gender": "", "note": ""}
            cur = global_chars[name]
            for alias in info.get("aliases", []) or []:
                if alias and alias not in cur["aliases"]:
                    cur["aliases"].append(alias)
            for key in ("gender", "note"):
                if info.get(key) and not cur.get(key):
                    cur[key] = info[key]

    def split(self, text: str) -> SplitResult:
        chunks = _chunk_text(text, self.max_chunk_chars)
        all_segments: list[Segment] = []
        characters: dict[str, dict[str, Any]] = {}
        for i, chunk in enumerate(chunks):
            print(f"[splitter] chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
            data = self._call_llm(chunk, characters)
            for seg in data.get("segments", []) or []:
                t = seg.get("type", "narration")
                if t not in ("narration", "dialogue"):
                    t = "narration"
                speaker = (seg.get("speaker") or "").strip() or "narrator"
                if t == "narration":
                    speaker = "narrator"
                text_v = (seg.get("text") or "").strip()
                if text_v:
                    all_segments.append(Segment(type=t, speaker=speaker, text=text_v))
            self._merge_characters(characters, data.get("characters", {}) or {})
        return SplitResult(segments=all_segments, characters=characters)


def save_script(result: SplitResult, path: str | Path) -> None:
    data = {
        "characters": result.characters,
        "segments": [s.to_dict() for s in result.segments],
    }
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_script(path: str | Path) -> SplitResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    segs = [Segment(**s) for s in data.get("segments", [])]
    return SplitResult(segments=segs, characters=data.get("characters", {}))
