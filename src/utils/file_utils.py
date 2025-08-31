"""
文件工具模块
提供文件操作相关的工具函数
"""

import os
import hashlib
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd
from .logger import LoggerMixin
from .config import get_config


class FileUtils(LoggerMixin):
    """文件工具类"""
    
    def __init__(self):
        super().__init__()
        self.config = get_config()
    
    def get_file_hash(self, file_path: Union[str, Path]) -> str:
        """
        计算文件的MD5哈希值
        
        Args:
            file_path: 文件路径
        
        Returns:
            文件的MD5哈希值
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        获取文件信息
        
        Args:
            file_path: 文件路径
        
        Returns:
            文件信息字典
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        stat = file_path.stat()
        return {
            "name": file_path.name,
            "path": str(file_path),
            "size": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "extension": file_path.suffix.lower(),
            "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "hash": self.get_file_hash(file_path)
        }
    
    def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """
        检查文件格式是否支持
        
        Args:
            file_path: 文件路径
        
        Returns:
            是否支持该格式
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        # 检查PDF格式
        if extension in self.config.document_processing.pdf_supported_formats:
            return True
        
        # 检查表格格式
        if extension in self.config.document_processing.table_supported_formats:
            return True
        
        return False
    
    def validate_file_size(self, file_path: Union[str, Path]) -> bool:
        """
        验证文件大小是否符合要求
        
        Args:
            file_path: 文件路径
        
        Returns:
            文件大小是否符合要求
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return False
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        extension = file_path.suffix.lower()
        
        if extension in self.config.document_processing.pdf_supported_formats:
            return file_size_mb <= self.config.document_processing.pdf_max_file_size_mb
        
        if extension in self.config.document_processing.table_supported_formats:
            return file_size_mb <= self.config.document_processing.table_max_file_size_mb
        
        return False
    
    def copy_file_to_documents(self, source_path: Union[str, Path]) -> Path:
        """
        将文件复制到文档目录
        
        Args:
            source_path: 源文件路径
        
        Returns:
            目标文件路径
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"源文件不存在: {source_path}")
        
        # 确保文档目录存在
        docs_dir = Path(self.config.paths.documents)
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成目标文件名（避免重名）
        target_name = source_path.name
        target_path = docs_dir / target_name
        
        # 如果文件已存在，添加时间戳
        if target_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_without_ext = source_path.stem
            extension = source_path.suffix
            target_name = f"{name_without_ext}_{timestamp}{extension}"
            target_path = docs_dir / target_name
        
        # 复制文件
        shutil.copy2(source_path, target_path)
        self.log_info(f"文件已复制到: {target_path}")
        
        return target_path
    
    def save_processed_document(self, content: str, original_file: Union[str, Path], 
                               format_type: str = "markdown") -> Path:
        """
        保存处理后的文档
        
        Args:
            content: 文档内容
            original_file: 原始文件路径
            format_type: 格式类型
        
        Returns:
            保存的文件路径
        """
        original_file = Path(original_file)
        processed_dir = Path(self.config.paths.processed)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成处理后的文件名
        name_without_ext = original_file.stem
        if format_type == "markdown":
            extension = ".md"
        elif format_type == "json":
            extension = ".json"
        else:
            extension = ".txt"
        
        processed_file = processed_dir / f"{name_without_ext}_processed{extension}"
        
        # 保存文件
        with open(processed_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.log_info(f"处理后的文档已保存: {processed_file}")
        return processed_file
    
    def save_metadata(self, metadata: Dict[str, Any], file_id: str) -> Path:
        """
        保存元数据
        
        Args:
            metadata: 元数据字典
            file_id: 文件ID
        
        Returns:
            元数据文件路径
        """
        metadata_dir = Path(self.config.paths.metadata)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = metadata_dir / f"{file_id}_metadata.json"
        
        # 添加时间戳
        metadata["saved_time"] = datetime.now().isoformat()
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        self.log_info(f"元数据已保存: {metadata_file}")
        return metadata_file
    
    def load_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        加载元数据
        
        Args:
            file_id: 文件ID
        
        Returns:
            元数据字典，如果不存在则返回None
        """
        metadata_dir = Path(self.config.paths.metadata)
        metadata_file = metadata_dir / f"{file_id}_metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.log_error(f"加载元数据失败: {e}")
            return None
    
    def list_processed_files(self) -> List[Dict[str, Any]]:
        """
        列出所有已处理的文件
        
        Returns:
            已处理文件信息列表
        """
        processed_dir = Path(self.config.paths.processed)
        if not processed_dir.exists():
            return []
        
        files_info = []
        for file_path in processed_dir.glob("*_processed.*"):
            try:
                file_info = self.get_file_info(file_path)
                files_info.append(file_info)
            except Exception as e:
                self.log_error(f"获取文件信息失败 {file_path}: {e}")
        
        return files_info
    
    def cleanup_old_files(self, days: int = 30) -> int:
        """
        清理旧文件
        
        Args:
            days: 保留天数
        
        Returns:
            清理的文件数量
        """
        import time
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        cleaned_count = 0
        
        # 清理处理后的文件
        processed_dir = Path(self.config.paths.processed)
        if processed_dir.exists():
            for file_path in processed_dir.glob("*_processed.*"):
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        self.log_info(f"已删除旧文件: {file_path}")
                    except Exception as e:
                        self.log_error(f"删除文件失败 {file_path}: {e}")
        
        # 清理元数据文件
        metadata_dir = Path(self.config.paths.metadata)
        if metadata_dir.exists():
            for file_path in metadata_dir.glob("*_metadata.json"):
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        self.log_info(f"已删除旧元数据: {file_path}")
                    except Exception as e:
                        self.log_error(f"删除元数据失败 {file_path}: {e}")
        
        return cleaned_count
    
    def get_storage_usage(self) -> Dict[str, float]:
        """
        获取存储使用情况
        
        Returns:
            各目录的存储使用情况（MB）
        """
        usage = {}
        
        for dir_name, dir_path in [
            ("documents", self.config.paths.documents),
            ("processed", self.config.paths.processed),
            ("vectors", self.config.paths.vectors),
            ("metadata", self.config.paths.metadata),
            ("logs", self.config.paths.logs)
        ]:
            dir_path = Path(dir_path)
            if dir_path.exists():
                total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                usage[dir_name] = round(total_size / (1024 * 1024), 2)
            else:
                usage[dir_name] = 0.0
        
        return usage


# 全局文件工具实例
_file_utils_instance: Optional[FileUtils] = None


def get_file_utils() -> FileUtils:
    """获取全局文件工具实例"""
    global _file_utils_instance
    if _file_utils_instance is None:
        _file_utils_instance = FileUtils()
    return _file_utils_instance 