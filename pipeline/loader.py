"""章节加载器：txt 用正则切章节，epub 走 spine。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chapter:
    index: int
    title: str
    text: str


# 兼容 “第X章 / 第X回 / 卷X 第X章 / Chapter X”
_CHAPTER_RE = re.compile(
    r"^\s*("
    r"第[一二三四五六七八九十百千零〇两\d]+[章回节卷篇]"
    r"|卷[一二三四五六七八九十百千零〇两\d]+\s+第[一二三四五六七八九十百千零〇两\d]+[章回节]"
    r"|Chapter\s+\d+"
    r"|CHAPTER\s+\d+"
    r")\s*[:：·\-—\s]*(.*?)\s*$",
    re.MULTILINE,
)


def load_txt(path: str | Path) -> list[Chapter]:
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_CHAPTER_RE.finditer(raw))
    if not matches:
        return [Chapter(index=1, title=Path(path).stem, text=raw.strip())]
    chapters: list[Chapter] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        head = m.group(1).strip()
        sub = m.group(2).strip()
        title = f"{head} {sub}".strip() if sub else head
        body = raw[m.end():end].strip()
        if not body:
            continue
        chapters.append(Chapter(index=i + 1, title=title, text=body))
    return chapters


def load_epub(path: str | Path) -> list[Chapter]:
    from ebooklib import epub, ITEM_DOCUMENT
    from bs4 import BeautifulSoup

    book = epub.read_epub(str(path))
    chapters: list[Chapter] = []
    idx = 0
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "lxml")
        # 优先用 <h1>/<h2>/<h3> 作为标题
        title_tag = soup.find(["h1", "h2", "h3"])
        title = title_tag.get_text(strip=True) if title_tag else item.get_name()
        text = soup.get_text("\n", strip=True)
        if not text.strip():
            continue
        idx += 1
        chapters.append(Chapter(index=idx, title=title or f"Chapter {idx}", text=text))
    return chapters


def load_book(path: str | Path) -> list[Chapter]:
    p = Path(path)
    if p.suffix.lower() == ".epub":
        return load_epub(p)
    return load_txt(p)
