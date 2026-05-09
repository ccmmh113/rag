#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Dict, List, Optional, Tuple, Union

import PyPDF2
import markdown
import html2text
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
from RAG.chunking import ChunkingConfig, MarkdownAwareChunker
from RAG.chunking import count_tokens, token_overlap, trim_to_tokens

enc = None


class ReadFiles:
    """
    class to read files
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self.file_list = self.get_files()

    def get_files(self):
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            for filename in filenames:
                if filename.endswith((".md", ".txt", ".pdf")):
                    full = os.path.normpath(os.path.join(filepath, filename))
                    if full.startswith(".\\") or full.startswith("./"):
                        full = full[2:]
                    file_list.append(full)
        return file_list

    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        """Return chunk texts for backward compatibility."""
        return [doc.text for doc in self.get_documents(max_token_len, cover_content)]

    def get_documents(self, max_token_len: int = 600, cover_content: int = 150):
        """Return chunk Documents with metadata."""
        docs = []
        chunker = MarkdownAwareChunker(
            ChunkingConfig(max_tokens=max_token_len, overlap_tokens=cover_content)
        )
        # 读取文件内容
        for file in self.file_list:
            if file.endswith(".pdf"):
                for page_num, page_text in self.read_pdf_pages(file):
                    docs.extend(
                        chunker.chunk(
                            page_text,
                            source=file,
                            page=page_num,
                        )
                    )
            else:
                content = self.read_file_content(file)
                docs.extend(chunker.chunk(content, source=file))
        return docs

    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        chunk_text = []

        curr_len = 0
        curr_chunk = ''

        token_len = max_token_len - cover_content
        lines = text.splitlines()  # 假设以换行符分割文本为行

        for line in lines:
            line = line.replace(' ', '')
            line_len = count_tokens(line)
            if line_len > max_token_len:
                # 如果单行长度就超过限制，则将其分割成多个块
                num_chunks = (line_len + token_len - 1) // token_len
                for i in range(num_chunks):
                    curr_chunk = token_overlap(curr_chunk, cover_content) + trim_to_tokens(line, token_len)
                    chunk_text.append(curr_chunk)
                # 处理最后一个块
                curr_chunk = token_overlap(curr_chunk, cover_content) + trim_to_tokens(line, token_len)
                chunk_text.append(curr_chunk)
                
            if curr_len + line_len <= token_len:
                curr_chunk += line
                curr_chunk += '\n'
                curr_len += line_len
                curr_len += 1
            else:
                chunk_text.append(curr_chunk)
                curr_chunk = token_overlap(curr_chunk, cover_content)+line
                curr_len = line_len + cover_content

        if curr_chunk:
            chunk_text.append(curr_chunk)

        return chunk_text

    @classmethod
    def read_file_content(cls, file_path: str):
        # 根据文件扩展名选择读取方法
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        else:
            raise ValueError("Unsupported file type")

    @classmethod
    def read_pdf(cls, file_path: str):
        # 读取PDF文件
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            return text

    @classmethod
    def read_pdf_pages(cls, file_path: str):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                yield page_num + 1, reader.pages[page_num].extract_text() or ""

    @classmethod
    def read_markdown(cls, file_path: str):
        # Keep markdown structure so heading-aware and code-aware chunking can work.
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    @classmethod
    def read_text(cls, file_path: str):
        # 读取文本文件
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()


class Documents:
    """
        获取已分好类的json格式文档
    """
    def __init__(self, path: str = '') -> None:
        self.path = path
    
    def get_content(self):
        with open(self.path, mode='r', encoding='utf-8') as f:
            content = json.load(f)
        return content
