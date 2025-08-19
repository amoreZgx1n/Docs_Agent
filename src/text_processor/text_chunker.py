"""
文本分块器模块
负责将长文本分割成适合LLM处理的小块
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from ..utils.logger import LoggerMixin
from ..utils.config import get_config

try:
    from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
    from llama_index.core.schema import TextNode, Document
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    print("警告: LlamaIndex未安装，将使用基础文本分块器")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("警告: LangChain未安装，将使用基础文本分块器")


class TextChunker(LoggerMixin):
    """文本分块器类"""
    
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.chunk_size = self.config.text_processing.chunk_size
        self.chunk_overlap = self.config.text_processing.chunk_overlap
        self.separator = self.config.text_processing.separator
        
        # 初始化分块器
        self._init_splitters()
    
    def _init_splitters(self):
        """初始化各种分块器"""
        self.splitters = {}
        
        # LlamaIndex分块器
        if LLAMAINDEX_AVAILABLE:
            try:
                # 句子分割器
                self.splitters['llama_sentence'] = SentenceSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                
                # Token分割器
                self.splitters['llama_token'] = TokenTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                
                self.log_info("LlamaIndex分块器初始化成功")
            except Exception as e:
                self.log_warning(f"LlamaIndex分块器初始化失败: {e}")
        
        # LangChain分块器
        if LANGCHAIN_AVAILABLE:
            try:
                # 递归字符分割器
                self.splitters['langchain_recursive'] = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
                )
                
                # Markdown分割器
                self.splitters['langchain_markdown'] = MarkdownHeaderTextSplitter(
                    headers_to_split_on=[
                        ("#", "标题1"),
                        ("##", "标题2"),
                        ("###", "标题3"),
                        ("####", "标题4"),
                        ("#####", "标题5"),
                        ("######", "标题6"),
                    ]
                )
                
                self.log_info("LangChain分块器初始化成功")
            except Exception as e:
                self.log_warning(f"LangChain分块器初始化失败: {e}")
    
    def chunk_text(self, text: str, method: str = "auto") -> List[str]:
        """
        将文本分割成块
        
        Args:
            text: 原始文本
            method: 分块方法 ("auto", "llama_sentence", "llama_token", "langchain_recursive", "langchain_markdown", "basic")
        
        Returns:
            文本块列表
        """
        if not text.strip():
            self.log_warning("输入文本为空")
            return []
        
        self.log_info(f"开始文本分块，方法: {method}，文本长度: {len(text)}")
        
        try:
            if method == "auto":
                return self._auto_chunk_text(text)
            elif method == "llama_sentence" and 'llama_sentence' in self.splitters:
                return self._chunk_with_llama_sentence(text)
            elif method == "llama_token" and 'llama_token' in self.splitters:
                return self._chunk_with_llama_token(text)
            elif method == "langchain_recursive" and 'langchain_recursive' in self.splitters:
                return self._chunk_with_langchain_recursive(text)
            elif method == "langchain_markdown" and 'langchain_markdown' in self.splitters:
                return self._chunk_with_langchain_markdown(text)
            else:
                return self._chunk_with_basic(text)
                
        except Exception as e:
            self.log_error(f"文本分块失败: {e}")
            # 回退到基础分块
            return self._chunk_with_basic(text)
    
    def _auto_chunk_text(self, text: str) -> List[str]:
        """自动选择最佳分块方法"""
        # 检测文本类型
        if self._is_markdown(text):
            if 'langchain_markdown' in self.splitters:
                return self._chunk_with_langchain_markdown(text)
            elif 'llama_sentence' in self.splitters:
                return self._chunk_with_llama_sentence(text)
            else:
                return self._chunk_with_basic(text)
        else:
            if 'llama_sentence' in self.splitters:
                return self._chunk_with_llama_sentence(text)
            elif 'langchain_recursive' in self.splitters:
                return self._chunk_with_langchain_recursive(text)
            else:
                return self._chunk_with_basic(text)
    
    def _is_markdown(self, text: str) -> bool:
        """检测是否为Markdown文本"""
        markdown_patterns = [
            r'^#\s+',  # 标题
            r'\*\*.*?\*\*',  # 粗体
            r'\*.*?\*',  # 斜体
            r'\[.*?\]\(.*?\)',  # 链接
            r'```.*?```',  # 代码块
            r'^\s*[-*+]\s+',  # 列表
            r'^\s*\d+\.\s+',  # 有序列表
        ]
        
        for pattern in markdown_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False
    
    def _chunk_with_llama_sentence(self, text: str) -> List[str]:
        """使用LlamaIndex句子分割器"""
        try:
            splitter = self.splitters['llama_sentence']
            # 使用正确的API
            nodes = splitter.split_text(text)
            # LlamaIndex返回字符串列表，不是Node对象
            chunks = [node for node in nodes if node.strip()]
            
            self.log_info(f"LlamaIndex句子分割完成，生成 {len(chunks)} 个块")
            return chunks
            
        except Exception as e:
            self.log_error(f"LlamaIndex句子分割失败: {e}")
            return self._chunk_with_basic(text)
    
    def _chunk_with_llama_token(self, text: str) -> List[str]:
        """使用LlamaIndex Token分割器"""
        try:
            splitter = self.splitters['llama_token']
            # 使用正确的API
            nodes = splitter.split_text(text)
            # LlamaIndex返回字符串列表，不是Node对象
            chunks = [node for node in nodes if node.strip()]
            
            self.log_info(f"LlamaIndex Token分割完成，生成 {len(chunks)} 个块")
            return chunks
            
        except Exception as e:
            self.log_error(f"LlamaIndex Token分割失败: {e}")
            return self._chunk_with_basic(text)
    
    def _chunk_with_langchain_recursive(self, text: str) -> List[str]:
        """使用LangChain递归分割器"""
        try:
            splitter = self.splitters['langchain_recursive']
            chunks = splitter.split_text(text)
            
            self.log_info(f"LangChain递归分割完成，生成 {len(chunks)} 个块")
            return chunks
            
        except Exception as e:
            self.log_error(f"LangChain递归分割失败: {e}")
            return self._chunk_with_basic(text)
    
    def _chunk_with_langchain_markdown(self, text: str) -> List[str]:
        """使用LangChain Markdown分割器"""
        try:
            splitter = self.splitters['langchain_markdown']
            # LangChain Markdown分割器返回Document对象，需要提取文本
            documents = splitter.split_text(text)
            chunks = []
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    chunks.append(doc.page_content)
                elif hasattr(doc, 'text'):
                    chunks.append(doc.text)
                else:
                    chunks.append(str(doc))
            
            self.log_info(f"LangChain Markdown分割完成，生成 {len(chunks)} 个块")
            return chunks
            
        except Exception as e:
            self.log_error(f"LangChain Markdown分割失败: {e}")
            return self._chunk_with_basic(text)
    
    def _chunk_with_basic(self, text: str) -> List[str]:
        """基础文本分块方法"""
        chunks = []
        
        # 按段落分割
        paragraphs = text.split(self.separator)
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 如果当前块加上新段落不超过限制
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                # 保存当前块
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # 开始新块
                if len(paragraph) <= self.chunk_size:
                    current_chunk = paragraph + "\n\n"
                else:
                    # 段落太长，需要进一步分割
                    sub_chunks = self._split_long_paragraph(paragraph)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
        
        # 添加最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        self.log_info(f"基础分块完成，生成 {len(chunks)} 个块")
        return chunks
    
    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """分割过长的段落"""
        chunks = []
        
        # 按句子分割
        sentences = re.split(r'[。！？.!?]', paragraph)
        
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + "。"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_markdown(self, markdown_text: str) -> List[Dict[str, Any]]:
        """
        专门处理Markdown文本的分块
        
        Args:
            markdown_text: Markdown文本
        
        Returns:
            包含元数据的文本块列表
        """
        self.log_info("开始Markdown文本分块")
        
        try:
            # 使用Markdown分块器
            if 'langchain_markdown' in self.splitters:
                chunks = self._chunk_with_langchain_markdown(markdown_text)
            else:
                chunks = self._chunk_with_basic(markdown_text)
            
            # 确保chunks是字符串列表
            if not chunks:
                return []
            
            # 处理可能的Document对象
            processed_chunks = []
            for chunk in chunks:
                if hasattr(chunk, 'page_content'):
                    processed_chunks.append(chunk.page_content)
                elif hasattr(chunk, 'text'):
                    processed_chunks.append(chunk.text)
                else:
                    processed_chunks.append(str(chunk))
            
            # 为每个块添加元数据
            chunk_metadata = []
            for i, chunk in enumerate(processed_chunks):
                metadata = {
                    'chunk_id': i,
                    'chunk_size': len(chunk),
                    'chunk_type': 'markdown',
                    'has_headers': bool(re.search(r'^#{1,6}\s+', chunk, re.MULTILINE)),
                    'has_code': bool(re.search(r'```', chunk)),
                    'has_links': bool(re.search(r'\[.*?\]\(.*?\)', chunk)),
                    'has_lists': bool(re.search(r'^\s*[-*+]\s+', chunk, re.MULTILINE))
                }
                chunk_metadata.append({
                    'text': chunk,
                    'metadata': metadata
                })
            
            self.log_info(f"Markdown分块完成，生成 {len(chunk_metadata)} 个块")
            return chunk_metadata
            
        except Exception as e:
            self.log_error(f"Markdown分块失败: {e}")
            return []
    
    def get_chunk_info(self, chunks: List[str]) -> Dict[str, Any]:
        """
        获取分块信息统计
        
        Args:
            chunks: 文本块列表
        
        Returns:
            分块信息统计
        """
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk) for chunk in chunks]
        
        info = {
            'total_chunks': len(chunks),
            'total_characters': sum(chunk_sizes),
            'average_chunk_size': sum(chunk_sizes) / len(chunks),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'chunk_size_distribution': {
                'small': len([s for s in chunk_sizes if s < self.chunk_size // 2]),
                'medium': len([s for s in chunk_sizes if self.chunk_size // 2 <= s <= self.chunk_size]),
                'large': len([s for s in chunk_sizes if s > self.chunk_size])
            }
        }
        
        return info 