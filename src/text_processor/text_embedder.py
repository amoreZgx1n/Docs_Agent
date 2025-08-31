"""
文本向量化模块
负责将文本转换为向量表示，支持SiliconFlow API和缓存功能
"""

import os
import hashlib
import pickle
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import requests
import numpy as np

from ..utils.logger import LoggerMixin
from ..utils.config import get_config


class EmbeddingCache:
    """向量化缓存管理器"""
    
    def __init__(self, cache_path: str, expire_days: int = 30):
        self.cache_path = Path(cache_path)
        self.expire_days = expire_days
        self.cache_path.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cache_file(self, text: str) -> Path:
        """获取缓存文件路径"""
        cache_key = self._get_cache_key(text)
        return self.cache_path / f"{cache_key}.pkl"
    
    def get(self, text: str) -> Optional[List[float]]:
        """获取缓存的向量"""
        cache_file = self._get_cache_file(text)
        
        if not cache_file.exists():
            return None
        
        try:
            # 检查文件是否过期
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - file_time > timedelta(days=self.expire_days):
                cache_file.unlink()  # 删除过期文件
                return None
            
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data.get('embedding')
                
        except Exception as e:
            # 缓存文件损坏，删除它
            if cache_file.exists():
                cache_file.unlink()
            return None
    
    def set(self, text: str, embedding: List[float]):
        """设置缓存"""
        cache_file = self._get_cache_file(text)
        
        try:
            cached_data = {
                'embedding': embedding,
                'timestamp': datetime.now().isoformat(),
                'text_hash': self._get_cache_key(text)
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
                
        except Exception as e:
            # 缓存写入失败，忽略错误
            pass
    
    def clear(self):
        """清理所有缓存"""
        try:
            for cache_file in self.cache_path.glob("*.pkl"):
                cache_file.unlink()
        except Exception:
            pass
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        cache_files = list(self.cache_path.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_path': str(self.cache_path),
            'file_count': len(cache_files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'expire_days': self.expire_days
        }


class SiliconFlowEmbedder(LoggerMixin):
    """SiliconFlow API向量化器"""
    
    def __init__(self, api_key: str, config: Any):
        super().__init__()
        self.api_key = api_key
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def _make_request(self, texts: List[str]) -> Dict[str, Any]:
        """发送API请求"""
        payload = {
            "model": self.config.text_processing.embedding_api.model,
            "input": texts,
            "encoding_format": "float"
        }
        
        response = self.session.post(
            self.config.text_processing.embedding_api.url,
            json=payload,
            timeout=self.config.text_processing.embedding_api.timeout
        )
        
        if response.status_code != 200:
            error_msg = f"API请求失败: {response.status_code} - {response.text}"
            raise Exception(error_msg)
        
        return response.json()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量向量化文本"""
        if not texts:
            return []
        
        # 检查文本长度限制（使用字符长度作为近似，因为token数量计算复杂）
        for i, text in enumerate(texts):
            # 对于中文文本，一个字符大约对应1-2个token，我们使用保守的估计
            estimated_tokens = len(text) * 1.5  # 保守估计
            if estimated_tokens > self.config.text_processing.embedding_api.max_tokens:
                # 如果超过限制，截断文本
                max_chars = int(self.config.text_processing.embedding_api.max_tokens / 1.5)
                texts[i] = text[:max_chars]
                self.logger.warning(f"文本 {i} 超过长度限制，已截断到 {max_chars} 字符")
        
        # 分批处理
        batch_size = self.config.text_processing.embedding_api.batch_size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 重试机制
            for attempt in range(self.config.text_processing.embedding_api.retries):
                try:
                    response = self._make_request(batch_texts)
                    embeddings = [item['embedding'] for item in response['data']]
                    all_embeddings.extend(embeddings)
                    break
                    
                except Exception as e:
                    if attempt == self.config.text_processing.embedding_api.retries - 1:
                        raise e
                    time.sleep(2 ** attempt)  # 指数退避
        
        return all_embeddings


class MockEmbedder:
    """Mock向量化器，用于测试"""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """生成随机向量"""
        embeddings = []
        for text in texts:
            # 基于文本内容生成确定性但随机的向量
            seed = hash(text) % (2**32)
            np.random.seed(seed)
            embedding = np.random.normal(0, 1, self.dimension).tolist()
            embeddings.append(embedding)
        
        return embeddings


class TextEmbedder(LoggerMixin):
    """文本向量化器主类"""
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__()
        self.config = config or get_config()
        
        # 初始化缓存
        if self.config.text_processing.embedding_cache.enabled:
            self.cache = EmbeddingCache(
                self.config.text_processing.embedding_cache.path,
                self.config.text_processing.embedding_cache.expire_days
            )
        else:
            self.cache = None
        
        # 初始化向量化器
        if self.config.text_processing.embedding_mock.enabled:
            self.embedder = MockEmbedder(
                self.config.text_processing.embedding_mock.dimension
            )
            self.logger.info("使用Mock向量化器")
        else:
            # 获取API密钥
            api_key = self._get_api_key()
            if not api_key:
                raise ValueError("未找到SiliconFlow API密钥，请设置环境变量SILICONFLOW_API_KEY或在config.yaml中配置")
            
            self.embedder = SiliconFlowEmbedder(api_key, self.config)
            self.logger.info("使用SiliconFlow API向量化器")
    
    def _get_api_key(self) -> Optional[str]:
        """获取API密钥"""
        # 优先从环境变量获取
        api_key = os.getenv('SILICONFLOW_API_KEY')
        if api_key:
            return api_key
        
        # 从配置文件获取
        api_key = self.config.get('api_key')
        if api_key:
            return api_key
        
        return None
    
    def embed_text(self, text: str) -> List[float]:
        """向量化单个文本"""
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量向量化文本"""
        if not texts:
            return []
        
        self.logger.info(f"开始向量化 {len(texts)} 个文本")
        
        # 检查缓存
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        if self.cache:
            for i, text in enumerate(texts):
                cached_embedding = self.cache.get(text)
                if cached_embedding is not None:
                    cached_embeddings.append((i, cached_embedding))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # 向量化未缓存的文本
        new_embeddings = []
        if uncached_texts:
            try:
                new_embeddings = self.embedder.embed_texts(uncached_texts)
                
                # 缓存新的向量
                if self.cache:
                    for text, embedding in zip(uncached_texts, new_embeddings):
                        self.cache.set(text, embedding)
                        
            except Exception as e:
                self.logger.error(f"向量化失败: {e}")
                raise
        
        # 合并结果
        all_embeddings = [None] * len(texts)
        
        # 填充缓存的向量
        for i, embedding in cached_embeddings:
            all_embeddings[i] = embedding
        
        # 填充新的向量
        for i, embedding in zip(uncached_indices, new_embeddings):
            all_embeddings[i] = embedding
        
        self.logger.info(f"向量化完成，缓存命中: {len(cached_embeddings)}, 新生成: {len(new_embeddings)}")
        
        return all_embeddings
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """向量化文本块，返回包含向量的完整块信息"""
        if not chunks:
            return []
        
        # 提取文本
        texts = [chunk.get('text', '') for chunk in chunks]
        
        # 向量化
        embeddings = self.embed_texts(texts)
        
        # 合并结果
        result = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_copy = chunk.copy()
            chunk_copy['embedding'] = embedding
            result.append(chunk_copy)
        
        return result
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        if self.cache:
            return self.cache.get_cache_info()
        else:
            return {'enabled': False}
    
    def clear_cache(self):
        """清理缓存"""
        if self.cache:
            self.cache.clear()
            self.logger.info("缓存已清理")
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """验证向量是否有效"""
        if not embedding:
            return False
        
        # 检查是否为数值
        try:
            float_values = [float(x) for x in embedding]
            return len(float_values) > 0
        except (ValueError, TypeError):
            return False 