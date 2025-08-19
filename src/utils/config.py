"""
配置管理模块
负责加载和管理项目配置
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# 尝试加载.env文件
def load_env_file():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


@dataclass
class DocumentProcessingConfig:
    """文档处理配置"""
    pdf_max_file_size_mb: int = 10
    pdf_supported_formats: list = None
    pdf_preserve_formatting: bool = True
    pdf_extract_images: bool = False
    table_supported_formats: list = None
    table_max_file_size_mb: int = 10
    table_encoding: str = "utf-8"
    table_engine: str = "openpyxl"

    def __post_init__(self):
        if self.pdf_supported_formats is None:
            self.pdf_supported_formats = [".pdf"]
        if self.table_supported_formats is None:
            self.table_supported_formats = [".csv", ".xlsx", ".xls"]


@dataclass
class EmbeddingAPIConfig:
    """向量化API配置"""
    url: str = "https://api.siliconflow.cn/v1/embeddings"
    model: str = "netease-youdao/bce-embedding-base_v1"
    max_tokens: int = 512
    timeout: int = 30
    retries: int = 3
    batch_size: int = 10


@dataclass
class EmbeddingCacheConfig:
    """向量化缓存配置"""
    enabled: bool = True
    path: str = "./data/embedding_cache"
    expire_days: int = 30
    format: str = "pickle"


@dataclass
class EmbeddingMockConfig:
    """向量化Mock配置"""
    enabled: bool = False
    dimension: int = 768


@dataclass
class TextProcessingConfig:
    """文本处理配置"""
    chunk_size: int = 300
    chunk_overlap: int = 30
    separator: str = "\n\n"
    chunk_overlap_ratio: float = 0.1
    embedding_model_name: str = "moka-ai/m3e-base"
    embedding_max_length: int = 512
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"
    embedding_normalize: bool = True
    embedding_api: EmbeddingAPIConfig = None
    embedding_cache: EmbeddingCacheConfig = None
    embedding_mock: EmbeddingMockConfig = None

    def __post_init__(self):
        if self.embedding_api is None:
            self.embedding_api = EmbeddingAPIConfig()
        if self.embedding_cache is None:
            self.embedding_cache = EmbeddingCacheConfig()
        if self.embedding_mock is None:
            self.embedding_mock = EmbeddingMockConfig()


@dataclass
class StorageConfig:
    """存储配置"""
    vector_db_type: str = "chroma"
    vector_db_persist_directory: str = "./chroma_db"
    vector_db_collection_name: str = "document_chunks"
    vector_db_distance_metric: str = "cosine"
    vector_db_embedding_function: str = "default"
    vector_db_metadata_hnsw_space: str = "cosine"
    vector_db_allow_reset: bool = True
    vector_db_anonymized_telemetry: bool = False
    metadata_storage_path: str = "./data/metadata"
    metadata_format: str = "json"
    processed_docs_storage_path: str = "./data/processed"
    processed_docs_format: str = "markdown"


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "./logs/agent.log"
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class PerformanceConfig:
    """性能配置"""
    max_concurrent_tasks: int = 3
    timeout_seconds: int = 300
    max_memory_usage_mb: int = 6000
    cleanup_threshold: float = 0.8


@dataclass
class ErrorHandlingConfig:
    """错误处理配置"""
    max_retries: int = 3
    retry_delay_seconds: int = 5
    log_errors: bool = True
    save_failed_docs: bool = True


@dataclass
class PathsConfig:
    """路径配置"""
    documents: str = "./data/documents"
    processed: str = "./data/processed"
    vectors: str = "./data/vectors"
    metadata: str = "./data/metadata"
    logs: str = "./logs"


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为项目根目录下的config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config_data: Dict[str, Any] = {}
        self._load_config()
        self._setup_config_objects()
    
    def _load_config(self):
        """加载配置文件"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f)
                
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            # 使用默认配置
            self._config_data = {}
    
    def _setup_config_objects(self):
        """设置配置对象"""
        # 文档处理配置
        doc_config = self._config_data.get('document_processing', {})
        pdf_config = doc_config.get('pdf', {})
        table_config = doc_config.get('table', {})
        
        self.document_processing = DocumentProcessingConfig(
            pdf_max_file_size_mb=pdf_config.get('max_file_size_mb', 10),
            pdf_supported_formats=pdf_config.get('supported_formats', [".pdf"]),
            pdf_preserve_formatting=pdf_config.get('markdown_conversion', {}).get('preserve_formatting', True),
            pdf_extract_images=pdf_config.get('markdown_conversion', {}).get('extract_images', False),
            table_supported_formats=table_config.get('supported_formats', [".csv", ".xlsx", ".xls"]),
            table_max_file_size_mb=table_config.get('max_file_size_mb', 10),
            table_encoding=table_config.get('encoding', "utf-8"),
            table_engine=table_config.get('pandas_options', {}).get('engine', "openpyxl")
        )
        
        # 文本处理配置
        text_config = self._config_data.get('text_processing', {})
        chunk_config = text_config.get('chunking', {})
        embed_config = text_config.get('embedding', {})
        api_config = embed_config.get('api', {})
        cache_config = embed_config.get('cache', {})
        mock_config = embed_config.get('mock', {})
        
        self.text_processing = TextProcessingConfig(
            chunk_size=chunk_config.get('chunk_size', 512),
            chunk_overlap=chunk_config.get('chunk_overlap', 50),
            separator=chunk_config.get('separator', "\n\n"),
            chunk_overlap_ratio=chunk_config.get('chunk_overlap_ratio', 0.1),
            embedding_model_name=embed_config.get('model_name', "moka-ai/m3e-base"),
            embedding_max_length=embed_config.get('max_length', 512),
            embedding_batch_size=embed_config.get('embed_batch_size', 32),
            embedding_device=embed_config.get('device', "cpu"),
            embedding_normalize=embed_config.get('normalize', True),
            embedding_api=EmbeddingAPIConfig(
                url=api_config.get('url', "https://api.siliconflow.cn/v1/embeddings"),
                model=api_config.get('model', "netease-youdao/bce-embedding-base_v1"),
                max_tokens=api_config.get('max_tokens', 512),
                timeout=api_config.get('timeout', 30),
                retries=api_config.get('retries', 3),
                batch_size=api_config.get('batch_size', 10)
            ),
            embedding_cache=EmbeddingCacheConfig(
                enabled=cache_config.get('enabled', True),
                path=cache_config.get('path', "./data/embedding_cache"),
                expire_days=cache_config.get('expire_days', 30),
                format=cache_config.get('format', "pickle")
            ),
            embedding_mock=EmbeddingMockConfig(
                enabled=mock_config.get('enabled', False),
                dimension=mock_config.get('dimension', 768)
            )
        )
        
        # 存储配置
        storage_config = self._config_data.get('storage', {})
        vector_config = storage_config.get('vector_db', {})
        metadata_config = storage_config.get('metadata', {})
        processed_config = storage_config.get('processed_docs', {})
        
        self.storage = StorageConfig(
            vector_db_type=vector_config.get('type', "chroma"),
            vector_db_persist_directory=vector_config.get('persist_directory', "./chroma_db"),
            vector_db_collection_name=vector_config.get('collection_name', "document_chunks"),
            vector_db_distance_metric=vector_config.get('distance_metric', "cosine"),
            vector_db_embedding_function=vector_config.get('embedding_function', "default"),
            vector_db_metadata_hnsw_space=vector_config.get('metadata_hnsw_space', "cosine"),
            vector_db_allow_reset=vector_config.get('allow_reset', True),
            vector_db_anonymized_telemetry=vector_config.get('anonymized_telemetry', False),
            metadata_storage_path=metadata_config.get('storage_path', "./data/metadata"),
            metadata_format=metadata_config.get('format', "json"),
            processed_docs_storage_path=processed_config.get('storage_path', "./data/processed"),
            processed_docs_format=processed_config.get('format', "markdown")
        )
        
        # 日志配置
        logging_config = self._config_data.get('logging', {})
        self.logging = LoggingConfig(
            level=logging_config.get('level', "INFO"),
            format=logging_config.get('format', "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file_path=logging_config.get('file_path', "./logs/agent.log"),
            max_file_size_mb=logging_config.get('max_file_size_mb', 10),
            backup_count=logging_config.get('backup_count', 5)
        )
        
        # 性能配置
        perf_config = self._config_data.get('performance', {})
        async_config = perf_config.get('async', {})
        memory_config = perf_config.get('memory', {})
        
        self.performance = PerformanceConfig(
            max_concurrent_tasks=async_config.get('max_concurrent_tasks', 3),
            timeout_seconds=async_config.get('timeout_seconds', 300),
            max_memory_usage_mb=memory_config.get('max_memory_usage_mb', 6000),
            cleanup_threshold=memory_config.get('cleanup_threshold', 0.8)
        )
        
        # 错误处理配置
        error_config = self._config_data.get('error_handling', {})
        self.error_handling = ErrorHandlingConfig(
            max_retries=error_config.get('max_retries', 3),
            retry_delay_seconds=error_config.get('retry_delay_seconds', 5),
            log_errors=error_config.get('log_errors', True),
            save_failed_docs=error_config.get('save_failed_docs', True)
        )
        
        # 路径配置
        paths_config = self._config_data.get('paths', {})
        self.paths = PathsConfig(
            documents=paths_config.get('documents', "./data/documents"),
            processed=paths_config.get('processed', "./data/processed"),
            vectors=paths_config.get('vectors', "./data/vectors"),
            metadata=paths_config.get('metadata', "./data/metadata"),
            logs=paths_config.get('logs', "./logs")
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self._config_data.get(key, default)
    
    def reload(self):
        """重新加载配置"""
        self._load_config()
        self._setup_config_objects()
    
    def validate(self) -> bool:
        """验证配置的有效性"""
        try:
            # 检查必要的路径是否存在
            paths_to_check = [
                self.paths.documents,
                self.paths.processed,
                self.paths.vectors,
                self.paths.metadata,
                self.paths.logs
            ]
            
            for path in paths_to_check:
                Path(path).mkdir(parents=True, exist_ok=True)
            
            # 检查配置值的合理性
            if self.document_processing.pdf_max_file_size_mb <= 0:
                raise ValueError("PDF文件大小限制必须大于0")
            
            if self.text_processing.chunk_size <= 0:
                raise ValueError("文本块大小必须大于0")
            
            if self.text_processing.chunk_overlap >= self.text_processing.chunk_size:
                raise ValueError("文本块重叠度不能大于等于块大小")
            
            return True
            
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False


# 全局配置实例
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def set_config(config: Config):
    """设置全局配置实例"""
    global _config_instance
    _config_instance = config 