"""
文档问答工具函数
实现 DOCUMENT_QA 任务所需的工具函数
"""

import sys
import os
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd

# 添加路径以导入相关模块
sys.path.append('/home/icesnls/workplace/zgx/work/Docs_Agent')
sys.path.append('/home/icesnls/workplace/zgx/work/Docs_Agent/src/llm_generate')

from llama_index.llms.zhipuai import ZhipuAI


@dataclass
class DocumentChunk:
    """文档块"""
    content: str
    chunk_id: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """搜索结果"""
    chunk: DocumentChunk
    score: float
    relevance_reason: str


class DocumentProcessor:
    """文档处理器"""

    def __init__(self, llm: Optional[ZhipuAI] = None):
        self.llm = llm or ZhipuAI(model="glm-4-flash", api_key=os.environ.get('ZHIPUAI_API_KEY', 'bdefa5d91470437380967660cf6592f0.UN5HkGKi6hYu8kVB'))

    def parse_document(self, content: Any) -> Dict[str, Any]:
        """
        解析文档内容

        Args:
            content: 文档内容（字符串、字典等）

        Returns:
            解析后的文档结构
        """
        if isinstance(content, str):
            text_content = content
        elif isinstance(content, dict):
            # 如果是字典，尝试提取文本内容
            text_content = content.get('content', str(content))
        else:
            text_content = str(content)

        # 清理和规范化文本
        cleaned_text = self._clean_text(text_content)

        # 提取基本元数据
        metadata = {
            'total_chars': len(cleaned_text),
            'total_words': len(cleaned_text.split()),
            'total_lines': len(cleaned_text.split('\n')),
        }

        # 尝试提取标题和段落
        paragraphs = self._extract_paragraphs(cleaned_text)
        headings = self._extract_headings(cleaned_text)

        return {
            'original_content': content,
            'cleaned_text': cleaned_text,
            'paragraphs': paragraphs,
            'headings': headings,
            'metadata': metadata
        }

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[DocumentChunk]:
        """
        将文档分割为文本块

        Args:
            text: 文档文本
            chunk_size: 块大小（字符数）
            overlap: 重叠字符数

        Returns:
            文档块列表
        """
        if not text:
            return []

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            # 计算块的结束位置
            end = start + chunk_size

            # 如果不是最后一块，确保在句子边界结束
            if end < len(text):
                # 寻找句子结束符
                sentence_endings = ['。', '！', '？', '.', '!', '?', '\n\n']
                for ending in sentence_endings:
                    last_ending = text.rfind(ending, start, end)
                    if last_ending != -1 and last_ending > start + chunk_size // 2:
                        end = last_ending + len(ending)
                        break

            # 提取块内容
            chunk_content = text[start:end].strip()
            if chunk_content:  # 只添加非空块
                chunk = DocumentChunk(
                    content=chunk_content,
                    chunk_id=f"chunk_{chunk_id:03d}",
                    start_pos=start,
                    end_pos=end,
                    metadata={
                        'chunk_size': len(chunk_content),
                        'word_count': len(chunk_content.split()),
                        'has_overlap': overlap > 0
                    }
                )
                chunks.append(chunk)
                chunk_id += 1

            # 计算下一个块的起始位置（考虑重叠）
            start = end - overlap if overlap > 0 else end

            # 防止无限循环
            if start >= len(text) or len(chunk_content) < chunk_size // 4:
                break

        return chunks

    def semantic_search(self, query: str, chunks: List[DocumentChunk], top_k: int = 3) -> List[SearchResult]:
        """
        语义搜索相关文本片段

        Args:
            query: 查询字符串
            chunks: 文档块列表
            top_k: 返回最相关的块数量

        Returns:
            搜索结果列表
        """
        if not chunks:
            return []

        results = []

        for chunk in chunks:
            # 简单的相关性计算（可以根据需要改进）
            score = self._calculate_relevance_score(query, chunk.content)

            if score > 0:  # 只保留有一定相关性的结果
                result = SearchResult(
                    chunk=chunk,
                    score=score,
                    relevance_reason=self._generate_relevance_reason(query, chunk.content, score)
                )
                results.append(result)

        # 按相关性分数排序
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    def generate_answer(self, query: str, context_chunks: List[DocumentChunk]) -> str:
        """
        基于上下文生成答案

        Args:
            query: 用户查询
            context_chunks: 相关上下文块

        Returns:
            生成的答案
        """
        if not context_chunks:
            return "抱歉，在提供的文档中没有找到相关信息。"

        # 合并上下文内容
        context_text = "\n\n".join([chunk.content for chunk in context_chunks])

        # 构建提示词
        prompt = f"""你是一个专业的文档问答助手。请基于提供的上下文信息回答用户的问题。

用户问题：{query}

上下文信息：
{context_text}

请提供准确、简洁的答案。如果上下文中没有足够的信息来回答问题，请说明这一点。

答案："""

        try:
            response = self.llm.complete(prompt)
            answer = response.text.strip()

            # 如果答案太短或不相关，可能需要重新生成
            if len(answer) < 10 or "抱歉" in answer:
                # 尝试使用更简单的提示
                simple_prompt = f"根据以下内容回答：{query}\n\n内容：{context_text[:500]}..."
                response = self.llm.complete(simple_prompt)
                answer = response.text.strip()

            return answer

        except Exception as e:
            print(f"生成答案时出错: {e}")
            # 返回基于上下文的简单答案
            return f"基于文档内容，关于'{query}'的信息包含在提供的文本中。请查看相关段落获取详细信息。"

    def _clean_text(self, text: str) -> str:
        """清理和规范化文本"""
        if not text:
            return ""

        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除多余的换行符
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        return text.strip()

    def _extract_paragraphs(self, text: str) -> List[str]:
        """提取段落"""
        paragraphs = []
        for para in text.split('\n\n'):
            para = para.strip()
            if para and len(para) > 10:  # 只保留有意义的段落
                paragraphs.append(para)
        return paragraphs

    def _extract_headings(self, text: str) -> List[str]:
        """提取标题"""
        headings = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            # 检查是否是标题（以#开头或全大写或特定格式）
            if (line.startswith('#') or
                (len(line) < 50 and line.isupper()) or
                (line and not line[0].islower() and len(line.split()) <= 10)):
                headings.append(line)

        return headings

    def _calculate_relevance_score(self, query: str, content: str) -> float:
        """计算相关性分数"""
        if not query or not content:
            return 0.0

        query_words = set(self._tokenize_text(query.lower()))
        content_words = set(self._tokenize_text(content.lower()))

        # 计算词语重叠度
        if not query_words:
            return 0.0

        overlap = len(query_words.intersection(content_words))
        score = overlap / len(query_words)

        # 额外奖励：如果包含完整查询词组
        if query.lower() in content.lower():
            score += 0.5

        return min(score, 1.0)  # 确保分数不超过1.0

    def _tokenize_text(self, text: str) -> List[str]:
        """简单的文本分词"""
        # 移除标点符号
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        # 分词（简单按空格和中文字符分割）
        words = re.findall(r'[\w\u4e00-\u9fff]+', text)
        return [word for word in words if len(word) > 1]  # 过滤单字符词

    def _generate_relevance_reason(self, query: str, content: str, score: float) -> str:
        """生成相关性理由"""
        if score >= 0.8:
            return "高度相关：包含查询中的大部分关键词"
        elif score >= 0.5:
            return "中等相关：包含部分查询关键词"
        elif score >= 0.2:
            return "低度相关：包含少数查询关键词"
        else:
            return "基本相关：包含少量相关信息"


# 全局实例
document_processor = DocumentProcessor()


# 工具函数接口
def parse_document(content: Any) -> Dict[str, Any]:
    """解析文档内容"""
    return document_processor.parse_document(content)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """将文档分割为文本块"""
    chunks = document_processor.chunk_text(text, chunk_size, overlap)
    # 转换为字典格式以便序列化
    return [
        {
            'content': chunk.content,
            'chunk_id': chunk.chunk_id,
            'start_pos': chunk.start_pos,
            'end_pos': chunk.end_pos,
            'metadata': chunk.metadata
        }
        for chunk in chunks
    ]


def semantic_search(query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """搜索相关文本片段"""
    # 将字典转换回 DocumentChunk 对象
    document_chunks = [
        DocumentChunk(
            content=chunk['content'],
            chunk_id=chunk['chunk_id'],
            start_pos=chunk['start_pos'],
            end_pos=chunk['end_pos'],
            metadata=chunk['metadata']
        )
        for chunk in chunks
    ]

    results = document_processor.semantic_search(query, document_chunks, top_k)

    # 转换为字典格式
    return [
        {
            'chunk': {
                'content': result.chunk.content,
                'chunk_id': result.chunk.chunk_id,
                'start_pos': result.chunk.start_pos,
                'end_pos': result.chunk.end_pos,
                'metadata': result.chunk.metadata
            },
            'score': result.score,
            'relevance_reason': result.relevance_reason
        }
        for result in results
    ]


def generate_answer(query: str, context: List[Dict[str, Any]]) -> str:
    """基于上下文生成答案"""
    # 将字典转换回 DocumentChunk 对象
    chunks = [
        DocumentChunk(
            content=item['chunk']['content'],
            chunk_id=item['chunk']['chunk_id'],
            start_pos=item['chunk']['start_pos'],
            end_pos=item['chunk']['end_pos'],
            metadata=item['chunk']['metadata']
        )
        for item in context
    ]

    return document_processor.generate_answer(query, chunks)
