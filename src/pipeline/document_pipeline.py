"""
文档处理主流水线
整合PDF处理、表格处理、文本分块、向量化和存储功能
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from src.utils.config import get_config
from src.utils.logger import LoggerMixin
from src.utils.file_utils import get_file_utils
from src.document_processor.pdf_processor_enhanced import EnhancedPDFProcessor
from src.document_processor.table_processor import TableProcessor
from src.text_processor.text_chunker import TextChunker
from src.text_processor.text_embedder import TextEmbedder
from src.storage.vector_store import ChromaVectorStore


@dataclass
class ProcessingResult:
    """处理结果数据类"""
    file_path: str
    file_type: str
    success: bool
    chunks_count: int = 0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class DocumentPipeline(LoggerMixin):
    """
    文档处理主流水线
    整合所有模块，提供端到端的文档处理功能
    """
    
    def __init__(self, config=None):
        """初始化主流水线"""
        super().__init__()
        self.config = config or get_config()
        
        # 初始化各个模块
        self.pdf_processor = EnhancedPDFProcessor()
        self.table_processor = TableProcessor()
        self.text_chunker = TextChunker()
        self.text_embedder = TextEmbedder()
        self.vector_store = ChromaVectorStore()
        self.file_utils = get_file_utils()
        
        self.logger.info("文档处理主流水线初始化完成")
    
    def process_document(self, file_path: Union[str, Path]) -> ProcessingResult:
        """
        处理单个文档
        
        Args:
            file_path: 文档路径
            
        Returns:
            ProcessingResult: 处理结果
        """
        file_path = Path(file_path)
        start_time = time.time()
        
        try:
            self.logger.info(f"开始处理文档: {file_path}")
            
            # 获取文件信息
            file_info = self.file_utils.get_file_info(file_path)
            file_hash = self.file_utils.get_file_hash(file_path)
            
            # 基础元数据
            base_metadata = {
                'source': str(file_path),
                'file_name': file_path.name,
                'file_type': file_path.suffix.lower(),
                'file_size': file_info['size'],
                'file_hash': file_hash,
                'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'pipeline_version': '1.0'
            }
            
            # 根据文件类型选择处理方式
            if file_path.suffix.lower() in ['.pdf']:
                return self._process_pdf(file_path, base_metadata, start_time)
            elif file_path.suffix.lower() in ['.csv', '.xlsx', '.xls']:
                return self._process_table(file_path, base_metadata, start_time)
            else:
                error_msg = f"不支持的文件类型: {file_path.suffix}"
                self.logger.error(error_msg)
                return ProcessingResult(
                    file_path=str(file_path),
                    file_type=file_path.suffix.lower(),
                    success=False,
                    error_message=error_msg,
                    processing_time=time.time() - start_time
                )
                
        except Exception as e:
            error_msg = f"处理文档失败: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return ProcessingResult(
                file_path=str(file_path),
                file_type=file_path.suffix.lower() if file_path.exists() else 'unknown',
                success=False,
                error_message=error_msg,
                processing_time=time.time() - start_time
            )
    
    def _process_pdf(self, file_path: Path, base_metadata: Dict, start_time: float) -> ProcessingResult:
        """处理PDF文档"""
        try:
            # 1. PDF转Markdown
            self.logger.info(f"开始PDF转Markdown: {file_path}")
            markdown_content = self.pdf_processor.convert_to_markdown(str(file_path))
            
            if not markdown_content.strip():
                raise ValueError("PDF转换结果为空")
            
            # 2. 文本分块
            self.logger.info("开始文本分块")
            chunks = self.text_chunker.chunk_text(
                markdown_content,
                method='basic'
            )
            
            # 3. 向量化和存储
            self.logger.info(f"开始向量化 {len(chunks)} 个文本块")
            documents = []
            
            for i, chunk in enumerate(chunks):
                # 为每个块添加元数据
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_id': i + 1,
                    'chunk_index': i,
                    'chunk_size': len(chunk),
                    'content_type': 'markdown',
                    'processing_stage': 'pdf_to_markdown'
                })
                
                # 向量化
                embedding = self.text_embedder.embed_text(chunk)
                
                documents.append({
                    'text': chunk,
                    'embedding': embedding,
                    'metadata': chunk_metadata
                })
            
            # 4. 存储到向量数据库
            self.logger.info(f"存储 {len(documents)} 个文档到向量数据库")
            doc_ids = self.vector_store.add_documents(documents)
            
            processing_time = time.time() - start_time
            self.logger.info(f"PDF处理完成，耗时: {processing_time:.2f}秒")
            
            return ProcessingResult(
                file_path=str(file_path),
                file_type='pdf',
                success=True,
                chunks_count=len(chunks),
                processing_time=processing_time,
                metadata=base_metadata
            )
            
        except Exception as e:
            raise Exception(f"PDF处理失败: {str(e)}")
    
    def _process_table(self, file_path: Path, base_metadata: Dict, start_time: float) -> ProcessingResult:
        """处理表格文档"""
        try:
            # 1. 加载表格数据
            self.logger.info(f"开始加载表格: {file_path}")
            df = self.table_processor.load_table(str(file_path))
            
            if df.empty:
                raise ValueError("表格数据为空")
            
            # 2. 将DataFrame转换为文本描述
            self.logger.info("将表格转换为文本描述")
            table_info = {
                'rows': len(df),
                'columns': len(df.columns),
                'columns_list': df.columns.tolist(),
                'dtypes_summary': str(df.dtypes.to_dict())
            }
            
            # 创建表格的文本描述
            table_text = f"""表格文件: {file_path.name}
行数: {table_info['rows']}
列数: {table_info['columns']}
列名: {', '.join(table_info['columns_list'][:5])}{'...' if len(table_info['columns_list']) > 5 else ''}

数据预览（前3行）:
{df.head(3).to_string(max_cols=5, max_colwidth=20)}"""
            
            # 3. 文本分块（表格数据通常不需要复杂分块）
            chunks = self.text_chunker.chunk_text(
                table_text,
                method='basic'
            )
            
            # 4. 向量化和存储
            self.logger.info(f"开始向量化 {len(chunks)} 个文本块")
            documents = []
            
            for i, chunk in enumerate(chunks):
                # 为每个块添加元数据
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_id': i + 1,
                    'chunk_index': i,
                    'chunk_size': len(chunk),
                    'content_type': 'table_description',
                    'processing_stage': 'table_to_text',
                    'table_rows': table_info['rows'],
                    'table_columns': table_info['columns'],
                    'table_columns_list': ', '.join(table_info['columns_list'])
                })
                
                # 向量化
                embedding = self.text_embedder.embed_text(chunk)
                
                documents.append({
                    'text': chunk,
                    'embedding': embedding,
                    'metadata': chunk_metadata
                })
            
            # 5. 存储到向量数据库
            self.logger.info(f"存储 {len(documents)} 个文档到向量数据库")
            doc_ids = self.vector_store.add_documents(documents)
            
            processing_time = time.time() - start_time
            self.logger.info(f"表格处理完成，耗时: {processing_time:.2f}秒")
            
            return ProcessingResult(
                file_path=str(file_path),
                file_type=file_path.suffix.lower(),
                success=True,
                chunks_count=len(chunks),
                processing_time=processing_time,
                metadata=base_metadata
            )
            
        except Exception as e:
            raise Exception(f"表格处理失败: {str(e)}")
    
    def process_directory(self, directory_path: Union[str, Path]) -> List[ProcessingResult]:
        """
        批量处理目录中的文档
        
        Args:
            directory_path: 目录路径
            
        Returns:
            List[ProcessingResult]: 处理结果列表
        """
        directory_path = Path(directory_path)
        results = []
        
        if not directory_path.exists():
            raise ValueError(f"目录不存在: {directory_path}")
        
        # 支持的文件类型
        supported_extensions = {'.pdf', '.csv', '.xlsx', '.xls'}
        
        # 查找所有支持的文件
        files = []
        for ext in supported_extensions:
            files.extend(directory_path.glob(f"*{ext}"))
        
        self.logger.info(f"找到 {len(files)} 个支持的文件")
        
        # 处理每个文件
        for file_path in files:
            try:
                result = self.process_document(file_path)
                results.append(result)
                
                if result.success:
                    self.logger.info(f"✓ {file_path.name} 处理成功")
                else:
                    self.logger.error(f"✗ {file_path.name} 处理失败: {result.error_message}")
                    
            except Exception as e:
                self.logger.error(f"处理文件 {file_path} 时发生异常: {str(e)}")
                results.append(ProcessingResult(
                    file_path=str(file_path),
                    file_type=file_path.suffix.lower(),
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    def search_documents(self, query: str, n_results: int = 5, 
                        filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        搜索文档
        
        Args:
            query: 搜索查询
            n_results: 返回结果数量
            filter_metadata: 元数据过滤条件
            
        Returns:
            List[Dict]: 搜索结果
        """
        return self.vector_store.search(query, n_results, filter_metadata)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.vector_store.get_statistics()
    
    def clear_database(self):
        """清空向量数据库"""
        self.vector_store.reset_database()
        self.logger.info("向量数据库已清空") 