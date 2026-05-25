import requests
from bs4 import BeautifulSoup
import re
import os

def get_chapter_urls(catalog_url, start_chapter=1544):
    """
    从目录页获取指定起始章节之后的所有章节信息。
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(catalog_url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"获取目录页失败: {e}")
        return []

    # 找到目录区域
    chapter_divs = soup.find_all('div', class_='module-row-info')
    chapter_data = []
    base_url = 'https://www.bilixs.com'

    for div in chapter_divs:
        link_tag = div.find('a')
        if not link_tag:
            continue

        title_text = link_tag.get('title', '') or link_tag.get_text(strip=True)
        if not title_text:
            continue

        # 提取章节号
        chapter_match = re.search(r'第(\d+)章', title_text)
        if not chapter_match:
            continue

        chapter_num = int(chapter_match.group(1))
        if chapter_num < start_chapter:
            continue

        chapter_url = link_tag.get('href', '')
        if chapter_url.startswith('/'):
            chapter_url = base_url + chapter_url
        elif not chapter_url.startswith('http'):
            continue

        chapter_data.append({
            'title': title_text,
            'url': chapter_url,
            'num': chapter_num
        })

    # 按章节号升序排列
    chapter_data.sort(key=lambda x: x['num'])
    return chapter_data

def download_chapter(chapter_info):
    """
    下载并提取单个章节的正文内容。
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(chapter_info['url'], headers=headers, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"下载失败 {chapter_info['title']}: {e}")
        return ""

    # 尝试多种方式定位正文内容
    content = ""
    # 方式1: 查找包含id "content" 的div
    content_div = soup.find('div', id='content')
    if content_div:
        content = content_div.get_text(separator='\n').strip()

    # 方式2: 如果未找到，尝试查找 "chapter-content" 类
    if not content:
        content_div = soup.find('div', class_='chapter-content')
        if content_div:
            content = content_div.get_text(separator='\n').strip()

    # 方式3: 如果还未找到，尝试通过识别章节标题来定位内容
    if not content:
        # 尝试寻找页面中的章节标题
        page_title = soup.find('h1')
        if page_title:
            # 获取标题后的所有文本内容
            content_parts = []
            for element in page_title.find_all_next(text=True):
                if element.parent.name not in ['script', 'style', 'a', 'link', 'meta', 'noscript']:
                    content_parts.append(element.strip())
            content = '\n'.join(content_parts)

    # 如果仍然没有内容，则返回空
    if not content:
        print(f"警告: 未能提取 {chapter_info['title']} 的正文内容")
        return ""

    # 清理内容中的广告和js代码
    content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL)
    content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL)
    content = re.sub(r'<.*?>', '', content)  # 移除所有HTML标签
    content = re.sub(r'&nbsp;', ' ', content)
    content = re.sub(r'\s+', ' ', content).strip()

    # 格式化输出
    formatted_content = f"{chapter_info['title']}\n{'=' * 50}\n{content}\n\n{'=' * 50}\n\n"
    return formatted_content

def main():
    catalog_url = 'https://www.bilixs.com/novel/doupocangqiong/catalog'
    start_chapter = 1544
    output_filename = f'斗破苍穹_{start_chapter}章之后.txt'

    print(f"开始获取《斗破苍穹》第{start_chapter}章之后的章节信息...")
    chapters = get_chapter_urls(catalog_url, start_chapter)

    if not chapters:
        print("未获取到任何章节信息，请检查网络或网站结构是否变化。")
        return

    print(f"共获取到 {len(chapters)} 个章节，开始下载...")

    # 如果文件存在，先删除以重新下载
    if os.path.exists(output_filename):
        os.remove(output_filename)

    downloaded_count = 0
    with open(output_filename, 'a', encoding='utf-8') as f:
        for i, chapter in enumerate(chapters, 1):
            print(f"正在下载 ({i}/{len(chapters)}): {chapter['title']}")
            content = download_chapter(chapter)
            if content:
                f.write(content)
                downloaded_count += 1
                # 每下载10章输出一次进度
                if i % 10 == 0:
                    print(f"  进度: {i}/{len(chapters)}")
            else:
                print(f"  失败: {chapter['title']}")

    print(f"\n下载完成！共成功下载 {downloaded_count}/{len(chapters)} 章。")
    print(f"内容已保存至: {output_filename}")

if __name__ == "__main__":
    main()