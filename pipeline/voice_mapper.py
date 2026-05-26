"""角色 -> 参考音色映射。

narrator 必填。"旁白" 视为 narrator 的别名。
其他角色未指定时回退到 narrator，并在 UI 提示用户绑定。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import ROOT


NARRATOR_ALIASES = {"narrator", "旁白", "旁 白"}


class VoiceMapper:
    def __init__(self, cfg: dict[str, Any], overrides: dict[str, str] | None = None):
        base = dict(cfg.get("voices", {}) or {})
        if overrides:
            base.update({k: v for k, v in overrides.items() if v})
        resolved: dict[str, str] = {}
        for name, path in base.items():
            if not path:
                continue
            p = Path(path)
            if not p.is_absolute():
                p = ROOT / p
            resolved[name] = str(p.resolve())
        # 把"旁白"等别名同步到 narrator（任一存在即可）
        if "narrator" not in resolved:
            for alias in NARRATOR_ALIASES:
                if alias in resolved:
                    resolved["narrator"] = resolved[alias]
                    break
        if "narrator" not in resolved or not Path(resolved["narrator"]).exists():
            raise FileNotFoundError(
                "缺少 narrator 参考音色，请在 config.yaml 或 UI 中绑定。"
            )
        self.mapping = resolved
        self._warned: set[str] = set()

    def get(self, speaker: str) -> str:
        speaker = (speaker or "").strip()
        if speaker in NARRATOR_ALIASES:
            speaker = "narrator"
        if speaker in self.mapping and Path(self.mapping[speaker]).exists():
            return self.mapping[speaker]
        if speaker not in self._warned:
            print(f"[voice] 角色 {speaker!r} 未绑定音色，回退到 narrator")
            self._warned.add(speaker)
        return self.mapping["narrator"]

    def unmapped(self, speakers: list[str]) -> list[str]:
        return [s for s in speakers
                if s not in NARRATOR_ALIASES
                and (s not in self.mapping or not Path(self.mapping[s]).exists())]
