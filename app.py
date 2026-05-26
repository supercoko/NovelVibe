"""Gradio UI：3 个 tab —— 导入 / 拆分 / 生成。

防丢失策略：
- 上传小说立即落盘 books/<safe_name>.json，刷新页面可从下拉框恢复
- LLM 拆分结果立即落盘 scripts/<idx>_<title>.json
- 用户编辑表格 → 自动保存回 JSON
- 角色音色映射保存到同名 .voices.json
- TTS 单段失败不中断，已生成 wav 由 cache/ 缓存
- config.yaml 改动按 mtime 自动热加载（IndexTTS2 段除外，模型已加载）
"""
from __future__ import annotations

import json
import re
import time
import traceback
from pathlib import Path

import gradio as gr
import pandas as pd

from pipeline.common import get_cfg, reload_cfg, ensure_dirs, ROOT
from pipeline.loader import load_book
from pipeline.splitter import (
    Splitter, Segment, save_script, load_script, load_progress,
    chunk_text, CONTROLLER, TTS_CONTROLLER, list_providers, resolve_provider,
)
from pipeline.voice_mapper import VoiceMapper
from pipeline.tts import TTSEngine, unload_tts, is_loaded
from pipeline.audio import export


ensure_dirs()
BOOKS_DIR = ROOT / "books"


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _safe_name(s: str) -> str:
    return re.sub(r"[^\w一-龥\-]+", "_", s)[:60].strip("_") or "chapter"


def _script_dir() -> Path:
    return ROOT / get_cfg()["paths"]["script_dir"]


def _voices_path(script_path: str | Path) -> Path:
    return Path(script_path).with_suffix(".voices.json")


def _list_scripts() -> list[str]:
    return sorted(str(p) for p in _script_dir().glob("*.json")
                  if not p.name.endswith(".voices.json"))


def _list_books() -> list[str]:
    return sorted(str(p) for p in BOOKS_DIR.glob("*.json"))


def _book_path(name: str) -> Path:
    return BOOKS_DIR / f"{_safe_name(name)}.json"


def _save_book(name: str, chapters: list[dict]) -> Path:
    p = _book_path(name)
    p.write_text(
        json.dumps({"name": name, "chapters": chapters}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p


def _load_book_file(path: str) -> tuple[str, list[dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("name", Path(path).stem), data.get("chapters", [])


def _df_to_segments(df: pd.DataFrame) -> list[Segment]:
    segs: list[Segment] = []
    if df is None or len(df) == 0:
        return segs
    for _, r in df.iterrows():
        text = str(r.get("text", "")).strip()
        if not text:
            continue
        t = str(r.get("type", "narration")).strip() or "narration"
        speaker = str(r.get("speaker", "narrator")).strip() or "narrator"
        if t == "narration":
            speaker = "narrator"
        emo_raw = r.get("emotion", "")
        if pd.isna(emo_raw):
            emo_raw = ""
        emotion = str(emo_raw).strip().lower() or "calm"
        segs.append(Segment(type=t, speaker=speaker, text=text, emotion=emotion))
    return segs


def _save_segments_df(df: pd.DataFrame, script_path: str) -> str:
    if not script_path:
        return "尚未拆分，无路径可保存"
    segs = _df_to_segments(df)
    chars: dict = {}
    p = Path(script_path)
    if p.exists():
        try:
            chars = json.loads(p.read_text(encoding="utf-8")).get("characters", {})
        except Exception:
            pass
    p.write_text(
        json.dumps({"characters": chars, "segments": [s.to_dict() for s in segs]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return f"已保存 {len(segs)} 段到 {p.name}"


def _save_voices_df(df: pd.DataFrame, script_path: str) -> str:
    if not script_path:
        return ""
    mapping: dict[str, str] = {}
    if df is not None and len(df) > 0:
        for _, r in df.iterrows():
            spk = str(r.get("speaker", "")).strip()
            ref = str(r.get("ref_audio", "")).strip()
            if spk and ref:
                mapping[spk] = ref
    _voices_path(script_path).write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return f"已保存 {len(mapping)} 个音色映射"


def _load_voices_df(script_path: str) -> pd.DataFrame:
    p = _voices_path(script_path)
    if not p.exists():
        return pd.DataFrame(columns=["speaker", "ref_audio", "aliases"])
    mapping = json.loads(p.read_text(encoding="utf-8"))
    # 把 script JSON 里的 characters 拿出来做 aliases 列
    chars: dict = {}
    sp = Path(script_path)
    if sp.exists():
        try:
            chars = json.loads(sp.read_text(encoding="utf-8")).get("characters", {}) or {}
        except Exception:
            chars = {}
    rows = []
    for k, v in mapping.items():
        aliases = ", ".join(chars.get(k, {}).get("aliases", []) or [])
        rows.append({"speaker": k, "ref_audio": v, "aliases": aliases})
    return pd.DataFrame(rows, columns=["speaker", "ref_audio", "aliases"])


def _chapters_to_ui(chapters: list[dict]):
    rows = [[c["index"], c["title"], len(c["text"])] for c in chapters]
    choices = [f"{c['index']:03d}. {c['title']}" for c in chapters]
    return rows, choices


# ---------------------------------------------------------------------------
# Tab1: 导入
# ---------------------------------------------------------------------------

def import_book(file):
    if file is None:
        return ([], gr.update(choices=[], value=None), pd.DataFrame(),
                gr.update(), "请先上传 txt 或 epub")
    path = file.name if hasattr(file, "name") else file
    name = Path(path).stem
    chapters = load_book(path)
    state = [{"index": c.index, "title": c.title, "text": c.text} for c in chapters]
    rows, choices = _chapters_to_ui(state)
    saved = _save_book(name, state)
    msg = f"共解析 {len(chapters)} 章，已存为 {saved.name}。"
    return (state,
            gr.update(choices=choices, value=choices[0] if choices else None),
            rows,
            gr.update(choices=_list_books(), value=str(saved)),
            msg)


def load_book_from_disk(book_path: str):
    """从 books/*.json 恢复一本已解析的小说。"""
    if not book_path or not Path(book_path).exists():
        return ([], gr.update(choices=[], value=None), pd.DataFrame(),
                "未选择已存书目")
    _, chapters = _load_book_file(book_path)
    rows, choices = _chapters_to_ui(chapters)
    msg = f"已从磁盘恢复 {len(chapters)} 章（{Path(book_path).name}）。"
    return (chapters,
            gr.update(choices=choices, value=choices[0] if choices else None),
            rows,
            msg)


# ---------------------------------------------------------------------------
# Tab2: LLM 拆分
# ---------------------------------------------------------------------------

def run_split(chapters_state, chapter_choice, start_chunk_1based, provider_name,
              progress=gr.Progress()):
    """流式生成器：每 chunk 完成 yield 一次更新。

    Returns 顺序: script_path, seg_df, char_json, split_msg, existing_scripts,
                  start_chunk_field
    """
    empty_outputs = ("", pd.DataFrame(), "{}", "请先在 Tab1 导入并选择章节",
                     gr.update(), gr.update())
    if not chapters_state or not chapter_choice:
        yield empty_outputs
        return
    idx = int(chapter_choice.split(".")[0])
    chapter = next((c for c in chapters_state if c["index"] == idx), None)
    if chapter is None:
        yield ("", pd.DataFrame(), "{}", "找不到所选章节", gr.update(), gr.update())
        return

    # 重置控制器（清除上一次的 pause/stop 标志）
    CONTROLLER.reset()

    cfg = get_cfg()
    splitter = Splitter(cfg, provider_name=provider_name or None)
    script_path = _script_dir() / f"{idx:03d}_{_safe_name(chapter['title'])}.json"

    # 续跑：如果脚本已存在，复用累计段 + characters
    existing_segments: list[Segment] = []
    existing_chars: dict = {}
    if script_path.exists():
        try:
            prev = load_script(script_path)
            existing_segments = list(prev.segments)
            existing_chars = dict(prev.characters)
        except Exception:
            pass

    start_chunk = max(0, int(start_chunk_1based) - 1)
    # 用户选择从某 chunk 重做：截断累计段（按这之前的 chunk 留下的段）
    # 这里粗略策略：start_chunk == 0 时彻底重来，否则信任已有段（用户自己选好起点）
    if start_chunk == 0:
        existing_segments = []
        existing_chars = {}

    total_chunks = len(chunk_text(chapter["text"], splitter.max_chunk_chars))
    if start_chunk >= total_chunks:
        yield (str(script_path), pd.DataFrame([s.to_dict() for s in existing_segments]),
               json.dumps(existing_chars, ensure_ascii=False, indent=2),
               f"起始 chunk {start_chunk + 1} 超出总数 {total_chunks}",
               gr.update(), gr.update(value=1))
        return

    msg = f"开始拆分，共 {total_chunks} chunk，从第 {start_chunk + 1} 块起..."
    yield (str(script_path), pd.DataFrame([s.to_dict() for s in existing_segments]),
           json.dumps(existing_chars, ensure_ascii=False, indent=2),
           msg, gr.update(), gr.update(value=start_chunk + 1))

    for prog in splitter.split_stream(
        chapter["text"],
        start_chunk=start_chunk,
        existing_segments=existing_segments,
        existing_characters=existing_chars,
    ):
        # 每 chunk 完成立即落盘
        from pipeline.splitter import SplitResult
        save_script(
            SplitResult(segments=prog.all_segments, characters=prog.characters),
            script_path,
            progress={"chunk_done": prog.index + 1, "chunk_total": prog.total,
                      "stopped": prog.stopped},
        )

        df = pd.DataFrame([s.to_dict() for s in prog.all_segments])
        chars_json = json.dumps(prog.characters, ensure_ascii=False, indent=2)
        if prog.stopped:
            status = f"⏹ 已停止 @ chunk {prog.index + 1}/{prog.total}，累计 {len(prog.all_segments)} 段"
        elif prog.index + 1 == prog.total:
            status = f"✅ 全部完成，{prog.total} chunk，{len(prog.all_segments)} 段"
        else:
            status = f"chunk {prog.index + 1}/{prog.total} 完成，累计 {len(prog.all_segments)} 段"
        yield (str(script_path), df, chars_json, status,
               gr.update(choices=_list_scripts(), value=str(script_path)),
               gr.update(value=prog.index + 2))  # 下次默认从下一个 chunk


def on_pause_split():
    CONTROLLER.pause()
    return "⏸ 已请求暂停，当前 chunk 跑完后停下"


def on_resume_split():
    CONTROLLER.resume()
    return "▶ 已恢复"


def on_stop_split():
    CONTROLLER.stop()
    return "⏹ 已请求停止"


def refresh_start_chunk(script_path: str):
    """根据脚本里的 progress 字段，给起始 chunk 建议下一次接续点。"""
    if not script_path or not Path(script_path).exists():
        return gr.update(value=1, maximum=999, info="未选脚本：默认从第 1 块开始")
    p = load_progress(script_path)
    done = int(p.get("chunk_done", 0))
    total = int(p.get("chunk_total", 0))
    if total == 0:
        return gr.update(value=1, maximum=999, info="脚本无进度元数据")
    nxt = min(done + 1, total)
    return gr.update(value=nxt, maximum=total,
                     info=f"已完成 {done}/{total}，下次默认从第 {nxt} 块开始")


def load_existing_script(script_path: str):
    if not script_path or not Path(script_path).exists():
        return ("", pd.DataFrame(), "{}", "脚本不存在",
                pd.DataFrame(columns=["speaker", "ref_audio", "aliases"]))
    result = load_script(script_path)
    df = pd.DataFrame([s.to_dict() for s in result.segments])
    chars_json = json.dumps(result.characters, ensure_ascii=False, indent=2)
    voices_df = _load_voices_df(script_path)
    msg = f"已加载 {len(result.segments)} 段，{len(voices_df)} 个音色映射。"
    return script_path, df, chars_json, msg, voices_df


# ---------------------------------------------------------------------------
# Tab3: 音色映射 + 生成
# ---------------------------------------------------------------------------

def build_voice_table(segments_df: pd.DataFrame, script_path: str):
    """从脚本提取角色列表：合并 segments 中出现的 speaker、characters 字典里
    的主名 + 别名，过滤掉 NaN/空，规一化"旁白"="narrator"。
    """
    NAR_ALIASES = {"narrator", "旁白", "旁 白"}

    # 1. 加载 script 里的 characters 字典（含 aliases）
    chars_dict: dict[str, dict] = {}
    if script_path and Path(script_path).exists():
        try:
            chars_dict = json.loads(Path(script_path).read_text(encoding="utf-8")).get(
                "characters", {}) or {}
        except Exception:
            chars_dict = {}

    # alias -> 主名 映射；主名→自身也加进去
    alias_to_main: dict[str, str] = {}
    for main, info in chars_dict.items():
        m = main.strip()
        if not m:
            continue
        alias_to_main[m] = m
        for a in (info.get("aliases") or []):
            if a and a.strip():
                alias_to_main[a.strip()] = m

    # 2. 收集 segments 里出现的所有 speaker
    seg_speakers: set[str] = set()
    has_narration = False
    if segments_df is not None and len(segments_df) > 0:
        for _, r in segments_df.iterrows():
            t = str(r.get("type", "")).strip()
            raw = r.get("speaker", "")
            # 过滤 NaN/None/"nan"/"None"/空
            if pd.isna(raw):
                continue
            s = str(raw).strip()
            if not s or s.lower() in {"nan", "none", "null"}:
                continue
            if t == "narration" or s in NAR_ALIASES:
                has_narration = True
                continue
            # alias 归一到主名
            s = alias_to_main.get(s, s)
            seg_speakers.add(s)

    # 3. 主名集合 = characters 字典里的所有主名 ∪ segments 里的发言人
    all_speakers = set(chars_dict.keys()) | seg_speakers

    # 4. 必加 narrator（任何脚本都有旁白）
    if has_narration or not all_speakers:
        all_speakers.add("narrator")

    # 5. 读取已有 voice 映射
    existing: dict[str, str] = {}
    if script_path:
        p = _voices_path(script_path)
        if p.exists():
            try:
                existing = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
    cfg_voices = get_cfg().get("voices", {}) or {}

    # 6. 排序：narrator 永远在第一行
    rows = []
    sorted_others = sorted(s for s in all_speakers if s != "narrator")
    final_order = (["narrator"] if "narrator" in all_speakers else []) + sorted_others
    for s in final_order:
        ref = existing.get(s) or cfg_voices.get(s, "")
        aliases = ", ".join(chars_dict.get(s, {}).get("aliases", []) or [])
        rows.append({"speaker": s, "ref_audio": ref, "aliases": aliases})
    return pd.DataFrame(rows, columns=["speaker", "ref_audio", "aliases"])


def generate_audio(segments_df: pd.DataFrame, voices_df: pd.DataFrame,
                   chapter_title: str, script_path: str, start_seg_1based,
                   progress=gr.Progress()):
    """流式合成：每段 yield 一次状态。

    输出顺序: final_audio_path, gen_msg, current_audio_preview, start_seg_field
    """
    if segments_df is None or len(segments_df) == 0:
        yield None, "没有可合成的段落", None, gr.update()
        return

    # 重置控制器
    TTS_CONTROLLER.reset()

    if script_path:
        _save_segments_df(segments_df, script_path)
        _save_voices_df(voices_df, script_path)

    cfg = get_cfg()
    overrides: dict[str, str] = {}
    if voices_df is not None and len(voices_df) > 0:
        for _, row in voices_df.iterrows():
            spk = str(row["speaker"]).strip()
            ref = str(row["ref_audio"]).strip() if row.get("ref_audio") else ""
            if spk and ref:
                overrides[spk] = ref

    try:
        voice = VoiceMapper(cfg, overrides=overrides)
    except FileNotFoundError as e:
        yield None, str(e), None, gr.update()
        return

    yield None, "🔄 正在加载 IndexTTS2（首次较慢）...", None, gr.update()
    tts = TTSEngine(cfg)

    segments = _df_to_segments(segments_df)
    total = len(segments)
    start_seg = max(0, int(start_seg_1based) - 1) if start_seg_1based else 0
    if start_seg >= total:
        yield (None, f"起始段 {start_seg + 1} 超出总数 {total}",
               None, gr.update(value=1))
        return

    failed: list[tuple[int, str]] = []
    cached_count = 0

    from pydub import AudioSegment
    audio_cfg = cfg.get("audio", {}) or {}
    combined = AudioSegment.silent(duration=0)
    prev: Segment | None = None

    yield (None,
           f"▶ 开始合成 {total} 段，从第 {start_seg + 1} 段起...",
           None, gr.update(value=start_seg + 1))

    for i, seg in enumerate(segments):
        if i < start_seg:
            continue

        # 暂停 / 停止检查
        TTS_CONTROLLER.wait_if_paused()
        if TTS_CONTROLLER.is_stopped:
            msg = (f"⏹ 已停止 @ {i + 1}/{total}，已合成 {len(combined)/1000:.1f}s"
                   f"（缓存命中 {cached_count} 段，失败 {len(failed)} 段）")
            yield (None, msg, None, gr.update(value=i + 1))
            # 即使停止也把已合成部分导出
            if len(combined) > 0:
                out_path = ROOT / cfg["paths"]["output_dir"] / f"{_safe_name(chapter_title or 'chapter')}_partial"
                final = export(combined, out_path, cfg)
                yield str(final), msg + f"\n💾 部分音频已导出: {final.name}", None, gr.update(value=i + 1)
            return

        # 检测缓存命中
        ref = voice.get(seg.speaker)
        cache_key_path = None
        try:
            from pipeline.common import text_hash
            from pipeline.tts import emotion_to_vector
            ref_resolved = str(Path(ref).resolve())
            ref_hash = tts._ref_hash(ref_resolved)
            key = text_hash(seg.text, ref_hash, (seg.emotion or "calm").lower())
            cache_key_path = tts.cache_dir / f"{key}.wav"
        except Exception:
            pass
        is_cached = cache_key_path is not None and cache_key_path.exists()

        # 状态：开始合成本段
        if not is_cached:
            status_now = (f"🎙 ({i + 1}/{total}) 合成中: {seg.speaker}"
                          f"({seg.emotion}) - {seg.text[:24]}...")
        else:
            status_now = (f"⚡ ({i + 1}/{total}) 缓存命中: {seg.speaker}"
                          f"({seg.emotion}) - {seg.text[:24]}")
        yield (None, status_now, None, gr.update(value=i + 1))

        try:
            wav_path = tts.synth(seg.text, ref, seg.emotion)
            clip = AudioSegment.from_file(str(wav_path))
            if is_cached:
                cached_count += 1
        except Exception as e:
            failed.append((i, f"{seg.speaker}({seg.emotion}): {seg.text[:20]} -> {e}"))
            traceback.print_exc()
            # 失败也推一次状态
            yield (None,
                   f"⚠ ({i + 1}/{total}) 失败: {seg.speaker} - {str(e)[:40]}",
                   None, gr.update(value=i + 1))
            continue

        # 拼接停顿
        if prev is not None:
            if prev.type == "dialogue" and seg.type == "dialogue" and prev.speaker != seg.speaker:
                gap = audio_cfg.get("pause_dialogue_switch", 600)
            elif prev.type != seg.type:
                gap = audio_cfg.get("pause_dialogue_switch", 600)
            else:
                gap = audio_cfg.get("pause_paragraph", 500)
            combined += AudioSegment.silent(duration=gap)
        combined += clip
        prev = seg

        # progress bar 更新
        progress((i + 1) / total,
                 desc=f"({i + 1}/{total}) {seg.speaker}: {seg.text[:18]}")

        # 已合成段数 + 速率
        done_count = i + 1 - start_seg
        status = (f"✓ ({i + 1}/{total}) {seg.speaker}({seg.emotion}): {seg.text[:24]}  "
                  f"|  已生成 {len(combined)/1000:.1f}s 音频 "
                  f"|  缓存命中 {cached_count}/{done_count} 段")
        if failed:
            status += f"  |  失败 {len(failed)} 段"

        # 让用户能边合成边听最近一段
        yield (None, status, str(wav_path), gr.update(value=i + 2))

    # 全部完成
    out_path = ROOT / cfg["paths"]["output_dir"] / f"{_safe_name(chapter_title or 'chapter')}"
    final = export(combined, out_path, cfg)
    msg = f"✅ 完成：{final.name}（{combined.duration_seconds:.1f}s, 共 {total} 段）"
    if failed:
        msg += f"  ⚠ 跳过 {len(failed)} 段失败："
        for idx, info in failed[:5]:
            msg += f"\n  - #{idx} {info}"
        if len(failed) > 5:
            msg += f"\n  ...另有 {len(failed) - 5} 段"
    yield str(final), msg, None, gr.update(value=1)


def on_pause_tts():
    TTS_CONTROLLER.pause()
    return "⏸ 已请求暂停，当前段合成完后停下"


def on_resume_tts():
    TTS_CONTROLLER.resume()
    return "▶ 已恢复"


def on_stop_tts():
    TTS_CONTROLLER.stop()
    return "⏹ 已请求停止"


def preview_voice(speaker: str, ref_audio: str, emotion: str = "calm"):
    if not ref_audio or not Path(ref_audio).exists():
        return None, "参考音频不存在"
    tts = TTSEngine(get_cfg())
    sample = f"你好，我是{speaker}，很高兴为你朗读。"
    try:
        wav = tts.synth(sample, ref_audio, emotion)
        return str(wav), f"试听已生成（emotion={emotion}）"
    except Exception as e:
        return None, f"试听失败：{e}"


# ---------- IndexTTS2 模型卸载 ----------

def _gpu_status() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "（无 CUDA）"
        a = torch.cuda.memory_allocated() / (1024 ** 2)
        r = torch.cuda.memory_reserved() / (1024 ** 2)
        return f"显存 allocated={a:.0f} MB / reserved={r:.0f} MB"
    except Exception as e:
        return f"（{e}）"


def tts_status() -> str:
    state = "🟢 已加载" if is_loaded() else "⚪ 未加载"
    return f"IndexTTS2 状态：{state}  |  {_gpu_status()}"


def on_unload_tts():
    info = unload_tts()
    if not info.get("was_loaded"):
        return f"⚪ {info.get('msg', '未加载')}  |  {_gpu_status()}"
    before = info.get("before_mb")
    after = info.get("after_mb")
    delta = ""
    if before is not None and after is not None:
        delta = f"（释放 {before - after:.0f} MB）"
    return f"✅ 已卸载 {delta}  |  {_gpu_status()}"


# ---------- 配置热加载 ----------

def on_reload_cfg():
    cfg = reload_cfg()
    msg = f"配置已重载 @ {time.strftime('%H:%M:%S')}。LMStudio / pause / voices 立即生效；IndexTTS2 段需重启 app。"
    return msg, json.dumps(cfg, ensure_ascii=False, indent=2)


# ---------- LLM provider 切换 ----------

def _provider_info(name: str) -> str:
    try:
        prov = resolve_provider(get_cfg(), name)
    except Exception as e:
        return f"⚠ {e}"
    masked = (prov['api_key'] or '')[:6] + "..." if prov.get('api_key') else "(空)"
    return (f"**provider**: `{prov['name']}`  |  **base_url**: `{prov['base_url']}`  "
            f"|  **model**: `{prov['model']}`  |  **api_key**: `{masked}`")


def on_pick_provider(name: str):
    """切换后只在内存生效；点'保存激活'写回 config.yaml。"""
    return _provider_info(name)


def on_save_active_provider(name: str):
    """把 llm.active 写回 config.yaml。其余字段保持原样（按 yaml.safe_dump 重写）。"""
    import yaml
    cfg_path = ROOT / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if "llm" not in cfg or "providers" not in cfg["llm"]:
        return "⚠ config.yaml 不是新 schema，无法写回。请手动编辑。"
    if name not in cfg["llm"]["providers"]:
        return f"⚠ 未知 provider {name!r}"
    cfg["llm"]["active"] = name
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
    reload_cfg()
    return f"✅ 已保存 active = {name}，下一次拆分立即生效"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="小说朗读工具", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 小说朗读工具\nLMStudio 拆分 + IndexTTS2 合成（编辑自动落盘，刷新/闪退可恢复）")

    with gr.Accordion("⚙ 全局配置（config.yaml 自动热加载）", open=False):
        with gr.Row():
            btn_reload_cfg = gr.Button("立即重载 config.yaml")
            cfg_msg = gr.Markdown()
        cfg_preview = gr.Code(
            label="当前配置",
            language="json",
            value=json.dumps(get_cfg(), ensure_ascii=False, indent=2),
        )

    chapters_state = gr.State([])
    script_path_state = gr.State("")

    with gr.Tabs():
        # ---------- Tab 1 ----------
        with gr.Tab("1. 导入"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_in = gr.File(label="上传 txt 或 epub", file_types=[".txt", ".epub"])
                    btn_import = gr.Button("解析并保存", variant="primary")
                    gr.Markdown("---")
                    existing_books = gr.Dropdown(
                        label="或恢复已导入的书（刷新不丢）",
                        choices=_list_books(),
                        interactive=True,
                    )
                    btn_load_book = gr.Button("加载该书")
                    import_msg = gr.Markdown()
                with gr.Column(scale=2):
                    chapters_table = gr.Dataframe(
                        headers=["#", "标题", "字数"],
                        label="章节列表（只读）",
                        interactive=False,
                    )
                    chapter_choice = gr.Dropdown(label="选择要处理的章节", choices=[])

        # ---------- Tab 2 ----------
        with gr.Tab("2. LLM 拆分"):
            with gr.Row():
                provider_dd = gr.Dropdown(
                    label="LLM provider",
                    choices=list_providers(get_cfg()),
                    value=get_cfg().get("llm", {}).get("active") or "lmstudio",
                    interactive=True,
                )
                btn_save_provider = gr.Button("保存为默认")
                provider_msg = gr.Markdown(_provider_info(
                    get_cfg().get("llm", {}).get("active") or "lmstudio"))
            with gr.Row():
                btn_split = gr.Button("▶ 开始 / 续跑", variant="primary")
                btn_pause = gr.Button("⏸ 暂停")
                btn_resume = gr.Button("继续")
                btn_stop = gr.Button("⏹ 停止", variant="stop")
                start_chunk_in = gr.Number(
                    label="起始 chunk (1-based)",
                    value=1, minimum=1, precision=0,
                    info="从指定块开始拆分；0/1 表示从头",
                )
            with gr.Row():
                existing_scripts = gr.Dropdown(
                    label="或加载已有脚本（防闪退恢复）",
                    choices=_list_scripts(),
                    interactive=True,
                )
                btn_load_existing = gr.Button("加载")
                btn_save_segs = gr.Button("手动保存当前编辑")
                btn_refresh_start = gr.Button("用脚本进度刷新起始 chunk")
                split_msg = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=2):
                    seg_df = gr.Dataframe(
                        headers=["type", "speaker", "emotion", "text"],
                        label="脚本（流式追加，每 chunk 完成自动落盘；emotion: happy/angry/sad/afraid/disgusted/melancholic/surprised/calm）",
                        interactive=True,
                        wrap=True,
                    )
                with gr.Column(scale=1):
                    char_json = gr.Code(label="识别出的角色", language="json")
                    script_file = gr.Textbox(label="当前脚本路径", interactive=False)

        # ---------- Tab 3 ----------
        with gr.Tab("3. 生成"):
            with gr.Row():
                btn_build_voices = gr.Button("从脚本提取角色")
                btn_generate = gr.Button("▶ 开始 / 续跑合成", variant="primary")
                btn_pause_tts = gr.Button("⏸ 暂停")
                btn_resume_tts = gr.Button("继续")
                btn_stop_tts = gr.Button("⏹ 停止", variant="stop")
            with gr.Row():
                start_seg_in = gr.Number(
                    label="起始段 (1-based)",
                    value=1, minimum=1, precision=0,
                    info="从指定段开始合成；缓存命中段会秒过",
                )
                btn_unload = gr.Button("🧹 卸载 IndexTTS2 释放显存")
                btn_refresh_status = gr.Button("刷新状态")
            tts_status_md = gr.Markdown(tts_status())
            gen_msg = gr.Markdown()
            voices_df = gr.Dataframe(
                headers=["speaker", "ref_audio", "aliases"],
                label="角色 → 参考音色（编辑自动落盘到 .voices.json；aliases 仅供参考）",
                interactive=True,
                wrap=True,
            )
            with gr.Row():
                test_speaker = gr.Textbox(label="试听角色名", value="narrator")
                test_ref = gr.Textbox(label="试听参考音色路径", value=str(ROOT / "voices/narrator.wav"))
                test_emotion = gr.Dropdown(
                    label="情绪",
                    choices=["calm", "happy", "angry", "sad", "afraid",
                             "disgusted", "melancholic", "surprised"],
                    value="calm",
                )
                btn_preview = gr.Button("试听一句")
            preview_audio = gr.Audio(label="试听结果", interactive=False)
            preview_msg = gr.Markdown()
            current_seg_audio = gr.Audio(label="当前段（合成时实时刷新）", interactive=False)
            final_audio = gr.Audio(label="最终成品", interactive=False)

    # ----- 事件 -----
    btn_reload_cfg.click(on_reload_cfg, outputs=[cfg_msg, cfg_preview])

    provider_dd.change(on_pick_provider, inputs=[provider_dd], outputs=[provider_msg])
    btn_save_provider.click(on_save_active_provider, inputs=[provider_dd],
                             outputs=[provider_msg])

    btn_import.click(
        import_book,
        inputs=[file_in],
        outputs=[chapters_state, chapter_choice, chapters_table, existing_books, import_msg],
    )

    btn_load_book.click(
        load_book_from_disk,
        inputs=[existing_books],
        outputs=[chapters_state, chapter_choice, chapters_table, import_msg],
    )

    btn_split.click(
        run_split,
        inputs=[chapters_state, chapter_choice, start_chunk_in, provider_dd],
        outputs=[script_file, seg_df, char_json, split_msg, existing_scripts, start_chunk_in],
    ).then(
        lambda p: p, inputs=[script_file], outputs=[script_path_state],
    )

    btn_pause.click(on_pause_split, outputs=[split_msg])
    btn_resume.click(on_resume_split, outputs=[split_msg])
    btn_stop.click(on_stop_split, outputs=[split_msg])

    btn_refresh_start.click(refresh_start_chunk,
                             inputs=[script_path_state],
                             outputs=[start_chunk_in])

    btn_load_existing.click(
        load_existing_script,
        inputs=[existing_scripts],
        outputs=[script_file, seg_df, char_json, split_msg, voices_df],
    ).then(
        lambda p: p, inputs=[script_file], outputs=[script_path_state],
    ).then(
        refresh_start_chunk, inputs=[script_path_state], outputs=[start_chunk_in],
    )

    seg_df.change(_save_segments_df, inputs=[seg_df, script_path_state], outputs=[split_msg])
    voices_df.change(_save_voices_df, inputs=[voices_df, script_path_state], outputs=[gen_msg])
    btn_save_segs.click(_save_segments_df, inputs=[seg_df, script_path_state], outputs=[split_msg])

    btn_build_voices.click(
        build_voice_table,
        inputs=[seg_df, script_path_state],
        outputs=[voices_df],
    )

    btn_preview.click(
        preview_voice,
        inputs=[test_speaker, test_ref, test_emotion],
        outputs=[preview_audio, preview_msg],
    )

    btn_generate.click(
        generate_audio,
        inputs=[seg_df, voices_df, chapter_choice, script_path_state, start_seg_in],
        outputs=[final_audio, gen_msg, current_seg_audio, start_seg_in],
    ).then(tts_status, outputs=[tts_status_md])

    btn_pause_tts.click(on_pause_tts, outputs=[gen_msg])
    btn_resume_tts.click(on_resume_tts, outputs=[gen_msg])
    btn_stop_tts.click(on_stop_tts, outputs=[gen_msg])

    btn_unload.click(on_unload_tts, outputs=[tts_status_md])
    btn_refresh_status.click(tts_status, outputs=[tts_status_md])


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
