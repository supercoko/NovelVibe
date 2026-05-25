"""Gradio UI：3 个 tab —— 导入 / 拆分 / 生成。

防丢失策略：
- LLM 拆分结果立即落盘 scripts/<idx>_<title>.json
- 用户编辑表格 → 自动保存回 JSON
- 角色音色映射保存到同名 .voices.json
- TTS 单段失败不中断，已生成 wav 由 cache/ 缓存
- 重启后从 scripts/ 下拉框直接恢复
"""
from __future__ import annotations

import json
import re
import traceback
from pathlib import Path

import gradio as gr
import pandas as pd

from pipeline.common import load_config, ensure_dirs, ROOT
from pipeline.loader import load_book
from pipeline.splitter import Splitter, Segment, save_script, load_script
from pipeline.voice_mapper import VoiceMapper
from pipeline.tts import TTSEngine
from pipeline.audio import synthesize_chapter, export


CFG = load_config()
ensure_dirs(CFG)
SCRIPT_DIR = ROOT / CFG["paths"]["script_dir"]


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _safe_name(s: str) -> str:
    return re.sub(r"[^\w一-龥\-]+", "_", s)[:60].strip("_") or "chapter"


def _voices_path(script_path: str | Path) -> Path:
    return Path(script_path).with_suffix(".voices.json")


def _list_scripts() -> list[str]:
    return sorted(str(p) for p in SCRIPT_DIR.glob("*.json")
                  if not p.name.endswith(".voices.json"))


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
        segs.append(Segment(type=t, speaker=speaker, text=text))
    return segs


def _save_segments_df(df: pd.DataFrame, script_path: str) -> str:
    if not script_path:
        return "尚未拆分，无路径可保存"
    segs = _df_to_segments(df)
    # 保留 characters 字段（重新加载已有 JSON 合并）
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
        return pd.DataFrame(columns=["speaker", "ref_audio"])
    mapping = json.loads(p.read_text(encoding="utf-8"))
    return pd.DataFrame([{"speaker": k, "ref_audio": v} for k, v in mapping.items()])


# ---------------------------------------------------------------------------
# Tab1: 导入
# ---------------------------------------------------------------------------

def import_book(file):
    if file is None:
        return [], gr.update(choices=[], value=None), pd.DataFrame(), "请先上传 txt 或 epub"
    path = file.name if hasattr(file, "name") else file
    chapters = load_book(path)
    rows = [[c.index, c.title, len(c.text)] for c in chapters]
    choices = [f"{c.index:03d}. {c.title}" for c in chapters]
    state = [{"index": c.index, "title": c.title, "text": c.text} for c in chapters]
    msg = f"共解析 {len(chapters)} 章。"
    return state, gr.update(choices=choices, value=choices[0] if choices else None), rows, msg


# ---------------------------------------------------------------------------
# Tab2: LLM 拆分
# ---------------------------------------------------------------------------

def run_split(chapters_state, chapter_choice, progress=gr.Progress()):
    if not chapters_state or not chapter_choice:
        return "", pd.DataFrame(), "{}", "请先在 Tab1 导入并选择章节", gr.update()
    idx = int(chapter_choice.split(".")[0])
    chapter = next((c for c in chapters_state if c["index"] == idx), None)
    if chapter is None:
        return "", pd.DataFrame(), "{}", "找不到所选章节", gr.update()

    progress(0.05, desc="连接 LMStudio...")
    splitter = Splitter(CFG)
    progress(0.1, desc="LLM 切分中...")
    result = splitter.split(chapter["text"])

    script_path = SCRIPT_DIR / f"{idx:03d}_{_safe_name(chapter['title'])}.json"
    save_script(result, script_path)

    df = pd.DataFrame([s.to_dict() for s in result.segments])
    chars_json = json.dumps(result.characters, ensure_ascii=False, indent=2)
    progress(1.0, desc="完成")
    msg = f"拆分完成，{len(result.segments)} 段，{len(result.characters)} 个角色。已保存 {script_path.name}"
    return str(script_path), df, chars_json, msg, gr.update(choices=_list_scripts(), value=str(script_path))


def load_existing_script(script_path: str):
    if not script_path or not Path(script_path).exists():
        return "", pd.DataFrame(), "{}", "脚本不存在", pd.DataFrame(columns=["speaker", "ref_audio"])
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
    """从脚本提取角色列表，合并已有音色映射。"""
    if segments_df is None or len(segments_df) == 0:
        return pd.DataFrame(columns=["speaker", "ref_audio"])
    speakers = sorted(set(str(s).strip() for s in segments_df["speaker"].tolist() if str(s).strip()))
    existing: dict[str, str] = {}
    if script_path:
        p = _voices_path(script_path)
        if p.exists():
            try:
                existing = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
    cfg_voices = CFG.get("voices", {}) or {}
    rows = []
    for s in speakers:
        ref = existing.get(s) or cfg_voices.get(s, "")
        rows.append({"speaker": s, "ref_audio": ref})
    return pd.DataFrame(rows)


def generate_audio(segments_df: pd.DataFrame, voices_df: pd.DataFrame,
                   chapter_title: str, script_path: str,
                   progress=gr.Progress()):
    if segments_df is None or len(segments_df) == 0:
        return None, "没有可合成的段落"

    # 任何变更都先落盘
    if script_path:
        _save_segments_df(segments_df, script_path)
        _save_voices_df(voices_df, script_path)

    overrides: dict[str, str] = {}
    if voices_df is not None and len(voices_df) > 0:
        for _, row in voices_df.iterrows():
            spk = str(row["speaker"]).strip()
            ref = str(row["ref_audio"]).strip() if row.get("ref_audio") else ""
            if spk and ref:
                overrides[spk] = ref

    try:
        voice = VoiceMapper(CFG, overrides=overrides)
    except FileNotFoundError as e:
        return None, str(e)

    progress(0.02, desc="加载 IndexTTS2 (首次较慢)...")
    tts = TTSEngine(CFG)

    segments = _df_to_segments(segments_df)
    failed: list[tuple[int, str]] = []

    # 容错：单段失败收集起来，不阻断整章
    from pydub import AudioSegment
    audio_cfg = CFG.get("audio", {}) or {}
    combined = AudioSegment.silent(duration=0)
    prev: Segment | None = None
    for i, seg in enumerate(segments):
        try:
            ref = voice.get(seg.speaker)
            wav_path = tts.synth(seg.text, ref)
            clip = AudioSegment.from_file(str(wav_path))
        except Exception as e:
            failed.append((i, f"{seg.speaker}: {seg.text[:20]} -> {e}"))
            traceback.print_exc()
            continue
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
        progress((i + 1) / max(len(segments), 1),
                 desc=f"({i + 1}/{len(segments)}) {seg.speaker}: {seg.text[:18]}")

    out_path = ROOT / CFG["paths"]["output_dir"] / f"{_safe_name(chapter_title or 'chapter')}"
    final = export(combined, out_path, CFG)
    msg = f"完成：{final.name}（{combined.duration_seconds:.1f}s）"
    if failed:
        msg += f"  ⚠ 跳过 {len(failed)} 段失败："
        for idx, info in failed[:5]:
            msg += f"\n  - #{idx} {info}"
        if len(failed) > 5:
            msg += f"\n  ...另有 {len(failed) - 5} 段"
    return str(final), msg


def preview_voice(speaker: str, ref_audio: str):
    if not ref_audio or not Path(ref_audio).exists():
        return None, "参考音频不存在"
    tts = TTSEngine(CFG)
    sample = f"你好，我是{speaker}，很高兴为你朗读。"
    try:
        wav = tts.synth(sample, ref_audio)
        return str(wav), "试听已生成"
    except Exception as e:
        return None, f"试听失败：{e}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="小说朗读工具", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 小说朗读工具\nLMStudio 拆分 + IndexTTS2 合成（编辑自动落盘，闪退可恢复）")

    chapters_state = gr.State([])
    script_path_state = gr.State("")

    with gr.Tabs():
        # ---------- Tab 1 ----------
        with gr.Tab("1. 导入"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_in = gr.File(label="上传 txt 或 epub", file_types=[".txt", ".epub"])
                    btn_import = gr.Button("解析章节", variant="primary")
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
                btn_split = gr.Button("调用 LMStudio 拆分", variant="primary")
                existing_scripts = gr.Dropdown(
                    label="或加载已有脚本（防闪退恢复）",
                    choices=_list_scripts(),
                    interactive=True,
                )
                btn_load_existing = gr.Button("加载")
                btn_save_segs = gr.Button("手动保存当前编辑")
                split_msg = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=2):
                    seg_df = gr.Dataframe(
                        headers=["type", "speaker", "text"],
                        label="脚本（任何编辑自动落盘）",
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
                btn_generate = gr.Button("开始合成", variant="primary")
                gen_msg = gr.Markdown()
            voices_df = gr.Dataframe(
                headers=["speaker", "ref_audio"],
                label="角色 → 参考音色（编辑自动落盘到 .voices.json）",
                interactive=True,
            )
            with gr.Row():
                test_speaker = gr.Textbox(label="试听角色名", value="narrator")
                test_ref = gr.Textbox(label="试听参考音色路径", value=str(ROOT / "voices/narrator.wav"))
                btn_preview = gr.Button("试听一句")
            preview_audio = gr.Audio(label="试听结果", interactive=False)
            preview_msg = gr.Markdown()
            final_audio = gr.Audio(label="最终成品", interactive=False)

    # ----- 事件 -----
    btn_import.click(
        import_book,
        inputs=[file_in],
        outputs=[chapters_state, chapter_choice, chapters_table, import_msg],
    )

    btn_split.click(
        run_split,
        inputs=[chapters_state, chapter_choice],
        outputs=[script_file, seg_df, char_json, split_msg, existing_scripts],
    ).then(
        lambda p: p, inputs=[script_file], outputs=[script_path_state],
    )

    btn_load_existing.click(
        load_existing_script,
        inputs=[existing_scripts],
        outputs=[script_file, seg_df, char_json, split_msg, voices_df],
    ).then(
        lambda p: p, inputs=[script_file], outputs=[script_path_state],
    )

    # 编辑表格自动落盘
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
        inputs=[test_speaker, test_ref],
        outputs=[preview_audio, preview_msg],
    )

    btn_generate.click(
        generate_audio,
        inputs=[seg_df, voices_df, chapter_choice, script_path_state],
        outputs=[final_audio, gen_msg],
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
