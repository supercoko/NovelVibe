# 小说朗读工具

LMStudio + IndexTTS2 端到端：上传 txt/epub → LLM 拆分旁白与对话 → 按角色绑定音色 → 合成 mp3。

## 一、依赖

```bash
# 1. Python 依赖
pip install -r requirements.txt

# 2. IndexTTS2 本体（你本地已安装在 D:/AI/index-tts）
pip install -e D:/AI/index-tts

# 3. ffmpeg：pydub 导出 mp3 需要
#    Windows 装包后把 ffmpeg.exe 加进 PATH
```

## 二、准备

1. **启动 LMStudio**，加载模型（当前 config.yaml 里写的是 `qwen/qwen3.5-9b`），开启 Local Server（默认 `http://localhost:1234`）。
2. **检查 `voices/narrator.wav`**：项目目录里已经把 `xiaoyan.wav` 复制为 narrator 默认音色。想换可直接覆盖。
3. **核对 `config.yaml`** 中的路径与模型 ID。

## 三、运行

```bash
python app.py
```

浏览器自动打开 `http://127.0.0.1:7860`。

### 三个 Tab 的工作流

1. **导入**：上传小说文件，自动切章。
2. **LLM 拆分**：选一章 → 点拆分 → 中间表格可手动修正说话人。
3. **生成**：点"从脚本提取角色" → 给每个角色填一段 wav 路径作参考音色（5–10 秒最佳） → 点"开始合成"。

## 四、目录结构

```
小说朗读工具/
├── app.py                  # Gradio 入口
├── config.yaml             # LMStudio / IndexTTS2 / 停顿 配置
├── requirements.txt
├── pipeline/
│   ├── common.py           # 配置 + hash
│   ├── loader.py           # txt + epub 章节加载
│   ├── splitter.py         # LMStudio LLM 拆分
│   ├── voice_mapper.py     # 角色 → 音色
│   ├── tts.py              # IndexTTS2 单例 + 缓存
│   └── audio.py            # 拼接 + 停顿 + 导出
├── prompts/splitter.txt    # 切分 prompt
├── voices/                 # 参考音色 wav
├── cache/                  # TTS 片段缓存（按文本+音色 hash）
├── scripts/                # LLM 拆分结果 JSON
└── output/                 # 成品 mp3
```

## 五、缓存与断点续传

- 同一段 `(text, speaker_ref_audio)` 二次生成会命中 `cache/` 里的 wav，几乎瞬时。
- LLM 拆分结果保存在 `scripts/<章号>_<标题>.json`，可手工编辑后跳过 Tab2 直接生成。

## 六、已知边界（MVP 之外）

- 仅按"旁白 / 对话 / 说话人"三维拆分，未接情感与内心独白。
- 角色字典只在单次拆分里维护，不跨章合并 —— 长篇建议每章导出 JSON 后手动 merge。
- 多音字与英文/数字读法未做规范化，依赖 IndexTTS2 自身处理。
