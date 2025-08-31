"""
增强版PDF处理器模块
使用MarkItDown库进行更精确的PDF到Markdown转换
"""

import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import Dict, List, Optional
from ..utils.logger import LoggerMixin
from ..utils.config import get_config
from ..utils.file_utils import get_file_utils

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False


class EnhancedPDFProcessor(LoggerMixin):
    """增强版PDF处理器类"""
    
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.file_utils = get_file_utils()
        self.markitdown = None
        
        if MARKITDOWN_AVAILABLE:
            try:
                self.markitdown = MarkItDown()
                self.log_info("MarkItDown库初始化成功")
            except Exception as e:
                self.log_warning(f"MarkItDown初始化失败: {e}")
    
    def convert_to_markdown(self, pdf_path: str, use_markitdown: bool = True) -> str:
        """将PDF转换为Markdown格式"""
        pdf_path = Path(pdf_path)
        self.log_info(f"开始处理PDF文件: {pdf_path}")
        
        try:
            if not self._validate_pdf_file(pdf_path):
                raise ValueError(f"PDF文件验证失败: {pdf_path}")
            
            if use_markitdown and self.markitdown and MARKITDOWN_AVAILABLE:
                return self._convert_with_markitdown(pdf_path)
            else:
                return self._convert_with_pymupdf(pdf_path)
                
        except Exception as e:
            self.log_error(f"PDF转换失败: {e}")
            raise
    
    def _convert_with_markitdown(self, pdf_path: Path) -> str:
        """使用MarkItDown库转换PDF"""
        self.log_info("使用MarkItDown库进行PDF转换")
        
        try:
            markdown_content = self.markitdown.convert(str(pdf_path))
            processed_content = self._post_process_markdown(markdown_content)
            self.log_info(f"MarkItDown转换完成，生成 {len(processed_content)} 字符的Markdown内容")
            return processed_content
            
        except Exception as e:
            self.log_error(f"MarkItDown转换失败: {e}")
            self.log_info("回退到PyMuPDF转换")
            return self._convert_with_pymupdf(pdf_path)
    
    def _convert_with_pymupdf(self, pdf_path: Path) -> str:
        """使用PyMuPDF转换PDF"""
        self.log_info("使用PyMuPDF进行PDF转换")
        
        try:
            doc = fitz.open(str(pdf_path))
            self.log_info(f"PDF文档已打开，共 {len(doc)} 页")
            
            markdown_content = self._extract_text_to_markdown(doc)
            doc.close()
            
            self.log_info(f"PyMuPDF转换完成，生成 {len(markdown_content)} 字符的Markdown内容")
            return markdown_content
            
        except Exception as e:
            self.log_error(f"PyMuPDF转换失败: {e}")
            raise
    
    def _post_process_markdown(self, markdown_content: str) -> str:
        """后处理Markdown内容"""
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', markdown_content)
        content = re.sub(r'^(\s*)(#{1,6})\s*', r'\1\2 ', content, flags=re.MULTILINE)
        content = re.sub(r'^(\s*)([-*+])\s*', r'\1\2 ', content, flags=re.MULTILINE)
        content = re.sub(r'```\s*\n', '```\n', content)
        return content.strip()
    
    def _validate_pdf_file(self, pdf_path: Path) -> bool:
        """验证PDF文件"""
        if not pdf_path.exists():
            self.log_error(f"PDF文件不存在: {pdf_path}")
            return False
        
        if not pdf_path.suffix.lower() == '.pdf':
            self.log_error(f"文件不是PDF格式: {pdf_path}")
            return False
        
        if not self.file_utils.validate_file_size(pdf_path):
            self.log_error(f"PDF文件大小超出限制: {pdf_path}")
            return False
        
        return True
    
    def _extract_text_to_markdown(self, doc: fitz.Document) -> str:
        """提取PDF文本并转换为Markdown格式"""
        markdown_parts = []
        
        title = self._extract_title(doc)
        if title:
            markdown_parts.append(f"# {title}\n")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            self.log_debug(f"处理第 {page_num + 1} 页")
            
            page_markdown = self._process_page(page, page_num + 1)
            if page_markdown.strip():
                markdown_parts.append(page_markdown)
        
        return "\n\n".join(markdown_parts)
    
    def _extract_title(self, doc: fitz.Document) -> Optional[str]:
        """提取文档标题"""
        try:
            metadata = doc.metadata
            if metadata.get('title'):
                return metadata['title']
            
            if len(doc) > 0:
                first_page = doc[0]
                text = first_page.get_text()
                lines = text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if line and len(line) < 100:
                        return line
            
            return None
            
        except Exception as e:
            self.log_warning(f"提取标题失败: {e}")
            return None
    
    def _process_page(self, page: fitz.Page, page_num: int) -> str:
        """处理单个页面"""
        page_parts = []
        page_parts.append(f"## 第 {page_num} 页\n")
        
        text_blocks = self._extract_text_blocks(page)
        
        for block in text_blocks:
            block_markdown = self._process_text_block(block)
            if block_markdown.strip():
                page_parts.append(block_markdown)
        
        return "\n\n".join(page_parts)
    
    def _extract_text_blocks(self, page: fitz.Page) -> List[Dict]:
        """提取页面中的文本块"""
        blocks = []
        text_dict = page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # 文本块
                block_info = {
                    "bbox": block.get("bbox"),
                    "lines": []
                }
                
                for line in block.get("lines", []):
                    line_info = {
                        "bbox": line.get("bbox"),
                        "spans": []
                    }
                    
                    for span in line.get("spans", []):
                        span_info = {
                            "text": span.get("text", ""),
                            "font": span.get("font", ""),
                            "size": span.get("size", 0),
                            "flags": span.get("flags", 0),
                            "bbox": span.get("bbox")
                        }
                        line_info["spans"].append(span_info)
                    
                    block_info["lines"].append(line_info)
                
                blocks.append(block_info)
        
        return blocks
    
    def _process_text_block(self, block: Dict) -> str:
        """处理文本块"""
        lines_markdown = []
        
        for line in block.get("lines", []):
            line_text = ""
            
            for span in line.get("spans", []):
                text = span.get("text", "")
                font_size = span.get("size", 0)
                flags = span.get("flags", 0)
                
                formatted_text = self._apply_text_formatting(text, font_size, flags)
                line_text += formatted_text
            
            if line_text.strip():
                lines_markdown.append(line_text)
        
        return "\n".join(lines_markdown)
    
    def _apply_text_formatting(self, text: str, font_size: float, flags: int) -> str:
        """应用文本格式"""
        if not text.strip():
            return text
        
        if font_size >= 16:
            return f"### {text}"
        elif font_size >= 14:
            return f"#### {text}"
        elif font_size >= 12:
            return f"##### {text}"
        
        if flags & 2**4:  # 粗体
            text = f"**{text}**"
        if flags & 2**1:  # 斜体
            text = f"*{text}*"
        if flags & 2**2:  # 下划线
            text = f"<u>{text}</u>"
        
        return text
    
    def get_document_info(self, pdf_path: str) -> Dict:
        """获取PDF文档信息"""
        pdf_path = Path(pdf_path)
        self.log_info(f"获取PDF文档信息: {pdf_path}")
        
        try:
            doc = fitz.open(str(pdf_path))
            
            info = {
                "file_path": str(pdf_path),
                "file_name": pdf_path.name,
                "page_count": len(doc),
                "metadata": doc.metadata,
                "file_size": pdf_path.stat().st_size,
                "file_size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2),
                "markitdown_available": MARKITDOWN_AVAILABLE
            }
            
            doc.close()
            return info
            
        except Exception as e:
            self.log_error(f"获取文档信息失败: {e}")
            return {} 