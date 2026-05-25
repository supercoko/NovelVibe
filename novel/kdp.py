#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
斗破苍穹小说下载器 - bilixs.com
下载第1544章及之后的所有章节
"""

import requests
import re
import os
import time
import json
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# ============ 配置 ============
BASE_URL = "https://www.bilixs.com"
NOVEL_PATH = "/novel/doupocangqiong"
CATALOG_URL = f"{BASE_URL}{NOVEL_PATH}/catalog"
START_CHAPTER = 1544  # 从第1544章开始下载
OUTPUT_DIR = "斗破苍穹_1544章后"
MAX_RETRIES = 3
DELAY = 1  # 请求间隔（秒）

# 请求头，模拟浏览器
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer': 'https://www.bilixs.com/',
    'Connection': 'keep-alive',
}

# ============ 核心类 ============

class NovelDownloader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.output_dir = OUTPUT_DIR
        self.chapters = []  # 章节列表
        self.failed_chapters = []  # 下载失败的章节
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def get_catalog(self):
        """获取目录列表"""
        print(f"正在获取目录: {CATALOG_URL}")
        
        try:
            response = self.session.get(CATALOG_URL, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 尝试多种可能的目录结构
            chapters = []
            
            # 方法1: 查找所有包含章节链接的a标签
            for a in soup.find_all('a'):
                href = a.get('href', '')
                text = a.get_text(strip=True)
                
                # 匹配章节链接模式
                if href and ('chapter' in href or re.search(r'/\d+\.html', href)):
                    # 提取章节号
                    match = re.search(r'第?(\d+)[章话回]', text)
                    if match:
                        chapter_num = int(match.group(1))
                        if chapter_num >= START_CHAPTER:
                            full_url = urljoin(BASE_URL, href)
                            chapters.append({
                                'num': chapter_num,
                                'title': text,
                                'url': full_url
                            })
            
            # 方法2: 如果上面没找到，尝试直接构造URL
            if not chapters:
                print("目录页未找到章节链接，尝试直接构造URL...")
                # 斗破苍穹通常有1647章左右
                for i in range(START_CHAPTER, 1700):
                    chapters.append({
                        'num': i,
                        'title': f'第{i}章',
                        'url': f"{BASE_URL}{NOVEL_PATH}/{i}.html"
                    })
            
            # 去重并排序
            seen = set()
            unique_chapters = []
            for ch in sorted(chapters, key=lambda x: x['num']):
                if ch['num'] not in seen:
                    seen.add(ch['num'])
                    unique_chapters.append(ch)
            
            self.chapters = unique_chapters
            print(f"找到 {len(self.chapters)} 个章节（从第{START_CHAPTER}章开始）")
            return True
            
        except Exception as e:
            print(f"获取目录失败: {e}")
            return False
    
    def fetch_chapter(self, chapter_info, retry=0):
        """获取单个章节内容"""
        url = chapter_info['url']
        num = chapter_info['num']
        
        try:
            print(f"正在下载第{num}章: {chapter_info['title']}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 尝试多种可能的内容选择器
            content = None
            
            # 方法1: 常见的正文容器
            selectors = [
                '.chapter-content',
                '#chapter-content',
                '.content',
                '#content',
                '.read-content',
                '#read-content',
                'article',
                '.text',
                '#txt',
                '.noveltext',
                '#noveltext',
            ]
            
            for selector in selectors:
                elem = soup.select_one(selector)
                if elem:
                    content = elem.get_text(separator='\n', strip=True)
                    if len(content) > 100:  # 确保内容足够长
                        break
            
            # 方法2: 如果没有找到，尝试查找最长的div或p段落
            if not content or len(content) < 100:
                # 查找所有段落
                paragraphs = soup.find_all('p')
                texts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20]
                if texts:
                    content = '\n\n'.join(texts)
            
            # 方法3: 尝试查找script中的数据（某些网站用JS加载）
            if not content or len(content) < 100:
                scripts = soup.find_all('script')
                for script in scripts:
                    script_text = script.string if script.string else ''
                    # 查找可能包含章节内容的JSON数据
                    if 'content' in script_text or 'chapter' in script_text:
                        # 尝试提取
                        matches = re.findall(r'["\']content["\']\s*:\s*["\'](.+?)["\']', script_text)
                        if matches:
                            content = matches[0].replace('\\n', '\n').replace('\\u', '\\\\u')
                            break
            
            if content and len(content) > 50:
                return {
                    'num': num,
                    'title': chapter_info['title'],
                    'content': content,
                    'url': url
                }
            else:
                print(f"  ⚠️ 第{num}章内容为空或太短，可能页面结构不同")
                return None
                
        except requests.exceptions.RequestException as e:
            if retry < MAX_RETRIES:
                print(f"  请求失败，{retry+1}/{MAX_RETRIES} 重试...")
                time.sleep(2)
                return self.fetch_chapter(chapter_info, retry + 1)
            else:
                print(f"  ❌ 第{num}章下载失败: {e}")
                self.failed_chapters.append(chapter_info)
                return None
    
    def save_chapter(self, chapter_data):
        """保存单个章节到文件"""
        if not chapter_data:
            return False
        
        num = chapter_data['num']
        title = chapter_data['title']
        content = chapter_data['content']
        
        # 清理文件名
        safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)
        filename = f"{self.output_dir}/第{num}章_{safe_title}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"{title}\n")
                f.write("=" * 50 + "\n\n")
                f.write(content)
                f.write("\n\n")
            
            print(f"  ✅ 已保存: {filename}")
            return True
        except Exception as e:
            print(f"  ❌ 保存失败: {e}")
            return False
    
    def save_to_single_file(self):
        """将所有章节合并为一个文件"""
        merged_file = f"{self.output_dir}/斗破苍穹_第{START_CHAPTER}章起_合集.txt"
        
        try:
            # 获取所有已下载的章节文件并排序
            chapter_files = []
            for f in os.listdir(self.output_dir):
                if f.startswith('第') and f.endswith('.txt') and f != os.path.basename(merged_file):
                    match = re.search(r'第(\d+)章', f)
                    if match:
                        chapter_files.append((int(match.group(1)), f))
            
            chapter_files.sort(key=lambda x: x[0])
            
            with open(merged_file, 'w', encoding='utf-8') as out:
                out.write("《斗破苍穹》\n")
                out.write(f"作者：天蚕土豆\n")
                out.write(f"从第{START_CHAPTER}章开始\n")
                out.write("=" * 50 + "\n\n")
                
                for _, filename in chapter_files:
                    filepath = os.path.join(self.output_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as ch_file:
                        out.write(ch_file.read())
                        out.write("\n\n")
            
            print(f"\n📚 合集已保存: {merged_file}")
            return True
        except Exception as e:
            print(f"合并文件失败: {e}")
            return False
    
    def download_all(self):
        """下载所有章节"""
        print("=" * 60)
        print("斗破苍穹小说下载器")
        print(f"目标网站: {BASE_URL}")
        print(f"起始章节: 第{START_CHAPTER}章")
        print("=" * 60)
        
        # 获取目录
        if not self.get_catalog():
            print("无法获取目录，尝试直接构造URL下载...")
            # 直接构造URL范围
            for i in range(START_CHAPTER, 1650):
                self.chapters.append({
                    'num': i,
                    'title': f'第{i}章',
                    'url': f"{BASE_URL}{NOVEL_PATH}/{i}.html"
                })
        
        # 下载章节
        success_count = 0
        for i, chapter in enumerate(self.chapters):
            print(f"\n[{i+1}/{len(self.chapters)}] ", end="")
            result = self.fetch_chapter(chapter)
            if result:
                self.save_chapter(result)
                success_count += 1
            
            # 间隔请求，避免被封
            time.sleep(DELAY)
        
        # 输出统计
        print("\n" + "=" * 60)
        print("下载完成!")
        print(f"成功: {success_count} 章")
        print(f"失败: {len(self.failed_chapters)} 章")
        
        if self.failed_chapters:
            print("\n失败的章节:")
            for ch in self.failed_chapters:
                print(f"  - 第{ch['num']}章: {ch['url']}")
        
        # 合并文件
        self.save_to_single_file()
        
        # 保存失败列表以便重试
        if self.failed_chapters:
            with open(f"{self.output_dir}/failed_chapters.json", 'w', encoding='utf-8') as f:
                json.dump(self.failed_chapters, f, ensure_ascii=False, indent=2)
            print(f"\n失败列表已保存到: failed_chapters.json")
        
        print("=" * 60)


# ============ 运行 ============

if __name__ == "__main__":
    # 检查依赖
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("请先安装依赖库:")
        print("  pip install requests beautifulsoup4")
        exit(1)
    
    downloader = NovelDownloader()
    downloader.download_all()