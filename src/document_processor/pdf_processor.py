"""
PDF处理器模块
负责PDF文档的处理和转换
"""

import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ..utils.logger import LoggerMixin
from ..utils.config import get_config
from ..utils.file_utils import get_file_utils


class PDFProcessor(LoggerMixin):
    """PDF处理器类"""
    
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.file_utils = get_file_utils()
    
    def convert_to_markdown(self, pdf_path: str) -> str:
        """
        将PDF转换为Markdown格式
        
        Args:
            pdf_path: PDF文件路径
        
        Returns:
            Markdown格式的文本内容
        """
        pdf_path = Path(pdf_path)
        self.log_info(f"开始处理PDF文件: {pdf_path}")
        
        try:
            # 验证文件
            if not self._validate_pdf_file(pdf_path):
                raise ValueError(f"PDF文件验证失败: {pdf_path}")
            
            # 打开PDF文档
            doc = fitz.open(str(pdf_path))
            self.log_info(f"PDF文档已打开，共 {len(doc)} 页")
            
            # 提取文本内容
            markdown_content = self._extract_text_to_markdown(doc)
            
            # 关闭文档
            doc.close()
            
            self.log_info(f"PDF转换完成，生成 {len(markdown_content)} 字符的Markdown内容")
            return markdown_content
            
        except Exception as e:
            self.log_error(f"PDF转换失败: {e}")
            raise
    
    def _validate_pdf_file(self, pdf_path: Path) -> bool:
        """
        验证PDF文件
        
        Args:
            pdf_path: PDF文件路径
        
        Returns:
            是否有效
        """
        if not pdf_path.exists():
            self.log_error(f"PDF文件不存在: {pdf_path}")
            return False
        
        if not pdf_path.suffix.lower() == '.pdf':
            self.log_error(f"文件不是PDF格式: {pdf_path}")
            return False
        
        # 检查文件大小
        if not self.file_utils.validate_file_size(pdf_path):
            self.log_error(f"PDF文件大小超出限制: {pdf_path}")
            return False
        
        return True
    
    def _extract_text_to_markdown(self, doc: fitz.Document) -> str:
        """
        提取PDF文本并转换为Markdown格式
        
        Args:
            doc: PyMuPDF文档对象
        
        Returns:
            Markdown格式的文本
        """
        markdown_parts = []
        
        # 添加文档标题
        title = self._extract_title(doc)
        if title:
            markdown_parts.append(f"# {title}\n")
        
        # 处理每一页
        for page_num in range(len(doc)):
            page = doc[page_num]
            self.log_debug(f"处理第 {page_num + 1} 页")
            
            # 提取页面文本
            page_markdown = self._process_page(page, page_num + 1)
            if page_markdown.strip():
                markdown_parts.append(page_markdown)
        
        return "\n\n".join(markdown_parts)
    
    def _extract_title(self, doc: fitz.Document) -> Optional[str]:
        """
        提取文档标题
        
        Args:
            doc: PyMuPDF文档对象
        
        Returns:
            文档标题
        """
        try:
            # 尝试从元数据获取标题
            metadata = doc.metadata
            if metadata.get('title'):
                return metadata['title']
            
            # 从第一页提取可能的标题
            if len(doc) > 0:
                first_page = doc[0]
                text = first_page.get_text()
                lines = text.split('\n')
                
                # 查找可能的标题（第一行非空文本）
                for line in lines:
                    line = line.strip()
                    if line and len(line) < 100:  # 标题通常不会太长
                        return line
            
            return None
            
        except Exception as e:
            self.log_warning(f"提取标题失败: {e}")
            return None
    
    def _process_page(self, page: fitz.Page, page_num: int) -> str:
        """
        处理单个页面
        
        Args:
            page: 页面对象
            page_num: 页码
        
        Returns:
            页面的Markdown内容
        """
        page_parts = []
        
        # 添加页码标记
        page_parts.append(f"## 第 {page_num} 页\n")
        
        # 提取文本块
        text_blocks = self._extract_text_blocks(page)
        
        # 处理文本块
        for block in text_blocks:
            block_markdown = self._process_text_block(block)
            if block_markdown.strip():
                page_parts.append(block_markdown)
        
        return "\n\n".join(page_parts)
    
    def _extract_text_blocks(self, page: fitz.Page) -> List[Dict]:
        """
        提取页面中的文本块
        
        Args:
            page: 页面对象
        
        Returns:
            文本块列表
        """
        blocks = []
        
        # 获取文本字典
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
        """
        处理文本块
        
        Args:
            block: 文本块信息
        
        Returns:
            Markdown格式的文本
        """
        lines_markdown = []
        
        for line in block.get("lines", []):
            line_text = ""
            line_formatting = []
            
            for span in line.get("spans", []):
                text = span.get("text", "")
                font_size = span.get("size", 0)
                flags = span.get("flags", 0)
                
                # 应用格式
                formatted_text = self._apply_text_formatting(text, font_size, flags)
                line_text += formatted_text
            
            if line_text.strip():
                lines_markdown.append(line_text)
        
        return "\n".join(lines_markdown)
    
    def _apply_text_formatting(self, text: str, font_size: float, flags: int) -> str:
        """
        应用文本格式
        
        Args:
            text: 原始文本
            font_size: 字体大小
            flags: 格式标志
        
        Returns:
            格式化后的文本
        """
        if not text.strip():
            return text
        
        # 根据字体大小判断标题级别
        if font_size >= 16:
            return f"### {text}"
        elif font_size >= 14:
            return f"#### {text}"
        elif font_size >= 12:
            return f"##### {text}"
        
        # 根据格式标志应用样式
        if flags & 2**4:  # 粗体
            text = f"**{text}**"
        if flags & 2**1:  # 斜体
            text = f"*{text}*"
        if flags & 2**2:  # 下划线
            text = f"<u>{text}</u>"
        
        return text
    
    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """
        提取PDF中的表格
        
        Args:
            pdf_path: PDF文件路径
        
        Returns:
            表格数据列表
        """
        pdf_path = Path(pdf_path)
        self.log_info(f"开始提取PDF表格: {pdf_path}")
        
        try:
            doc = fitz.open(str(pdf_path))
            tables = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_tables = self._extract_page_tables(page, page_num + 1)
                tables.extend(page_tables)
            
            doc.close()
            self.log_info(f"提取到 {len(tables)} 个表格")
            return tables
            
        except Exception as e:
            self.log_error(f"表格提取失败: {e}")
            return []
    
    def _extract_page_tables(self, page: fitz.Page, page_num: int) -> List[Dict]:
        """
        提取页面中的表格
        
        Args:
            page: 页面对象
            page_num: 页码
        
        Returns:
            表格数据列表
        """
        tables = []
        
        # 使用PyMuPDF的表格检测功能
        try:
            table_list = page.find_tables()
            
            for i, table in enumerate(table_list):
                table_data = {
                    "page": page_num,
                    "table_index": i,
                    "rows": []
                }
                
                # 提取表格数据
                for row in table.extract():
                    table_data["rows"].append(row)
                
                tables.append(table_data)
                
        except Exception as e:
            self.log_warning(f"页面 {page_num} 表格提取失败: {e}")
        
        return tables
    
    def get_document_info(self, pdf_path: str) -> Dict:
        """
        获取PDF文档信息
        
        Args:
            pdf_path: PDF文件路径
        
        Returns:
            文档信息字典
        """
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
                "file_size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2)
            }
            
            doc.close()
            return info
            
        except Exception as e:
            self.log_error(f"获取文档信息失败: {e}")
            return {} 