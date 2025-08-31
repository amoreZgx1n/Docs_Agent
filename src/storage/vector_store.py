"""
Chroma向量存储模块
负责管理embedding向量和支持RAG检索
"""

import os
import json
import shutil
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np

from ..utils.logger import LoggerMixin
from ..utils.config import get_config


class ChromaVectorStore(LoggerMixin):
    """Chroma向量存储管理器"""
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__()
        self.config = config or get_config()
        
        # 初始化Chroma客户端
        self._init_chroma_client()
        
        # 获取或创建集合
        self.collection = self._get_or_create_collection()
        
        # 缓存管理
        self._query_cache = {}
        self._cache_expire_time = timedelta(minutes=30)
        self._last_cache_cleanup = datetime.now()
        
        self.logger.info(f"Chroma向量存储初始化完成，路径: {self.config.storage.vector_db_persist_directory}")
    
    def _init_chroma_client(self):
        """初始化Chroma客户端"""
        try:
            # 确保存储目录存在
            persist_dir = Path(self.config.storage.vector_db_persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建Chroma客户端
            self.client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(
                    anonymized_telemetry=self.config.storage.vector_db_anonymized_telemetry,
                    allow_reset=self.config.storage.vector_db_allow_reset
                )
            )
            
            self.logger.info("Chroma客户端初始化成功")
            
        except Exception as e:
            self.logger.error(f"Chroma客户端初始化失败: {e}")
            raise
    
    def _get_or_create_collection(self):
        """获取或创建向量集合"""
        try:
            collection_name = self.config.storage.vector_db_collection_name
            
            # 尝试获取现有集合
            try:
                collection = self.client.get_collection(name=collection_name)
                self.logger.info(f"获取现有集合: {collection_name}")
            except:
                # 创建新集合（不使用默认embedding函数）
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={
                        "hnsw:space": self.config.storage.vector_db_metadata_hnsw_space,
                        "description": "Document chunks vector collection",
                        "created_at": datetime.now().isoformat()
                    }
                )
                self.logger.info(f"创建新集合: {collection_name}")
            
            return collection
            
        except Exception as e:
            self.logger.error(f"集合创建失败: {e}")
            raise
    
    def _get_collection_dimension(self) -> Optional[int]:
        """获取集合的向量维度"""
        try:
            # 尝试获取集合的维度信息
            collection_info = self.collection.get()
            if collection_info and 'embeddings' in collection_info and collection_info['embeddings']:
                return len(collection_info['embeddings'][0])
            return None
        except:
            return None
    
    def _validate_embedding(self, embedding: List[float]) -> bool:
        """验证向量格式"""
        if not embedding:
            return False
        
        try:
            # 检查是否为数值
            if not all(isinstance(x, (int, float)) for x in embedding):
                return False
            
            # 检查维度（支持多种维度）
            if len(embedding) not in [384, 768, 1024]:
                self.logger.warning(f"向量维度异常: {len(embedding)}, 支持: 384, 768, 1024")
            
            return True
            
        except Exception:
            return False
    
    def _generate_id(self, text: str, metadata: Dict[str, Any]) -> str:
        """生成唯一ID"""
        content = f"{text}_{json.dumps(metadata, sort_keys=True)}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _clean_cache(self):
        """清理过期缓存"""
        now = datetime.now()
        if now - self._last_cache_cleanup > timedelta(minutes=5):
            expired_keys = [
                key for key, (_, timestamp) in self._query_cache.items()
                if now - timestamp > self._cache_expire_time
            ]
            for key in expired_keys:
                del self._query_cache[key]
            self._last_cache_cleanup = now
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """批量添加文档"""
        if not documents:
            return []
        
        self.logger.info(f"开始添加 {len(documents)} 个文档")
        
        # 验证和准备数据
        valid_docs = []
        doc_ids = []
        
        # 获取集合的向量维度
        collection_dimension = self._get_collection_dimension()
        
        for doc in documents:
            try:
                # 验证必要字段
                if 'text' not in doc or 'embedding' not in doc:
                    self.logger.warning(f"文档缺少必要字段: {doc.keys()}")
                    continue
                
                # 验证向量
                if not self._validate_embedding(doc['embedding']):
                    self.logger.warning(f"文档向量无效: {doc.get('text', '')[:50]}...")
                    continue
                
                # 检查向量维度匹配
                embedding_dimension = len(doc['embedding'])
                if collection_dimension and embedding_dimension != collection_dimension:
                    self.logger.warning(f"向量维度不匹配: 文档{embedding_dimension}维, 集合{collection_dimension}维")
                    # 如果集合为空，重置集合以匹配新的维度
                    if collection_dimension == 0:
                        self.logger.info("集合为空，将使用文档的向量维度")
                        collection_dimension = embedding_dimension
                    else:
                        continue
                
                # 生成ID
                doc_id = self._generate_id(doc['text'], doc.get('metadata', {}))
                
                # 检查重复
                if self._is_duplicate(doc_id):
                    self.logger.info(f"跳过重复文档: {doc_id}")
                    continue
                
                valid_docs.append({
                    'id': doc_id,
                    'text': doc['text'],
                    'embedding': doc['embedding'],
                    'metadata': doc.get('metadata', {})
                })
                doc_ids.append(doc_id)
                
            except Exception as e:
                self.logger.error(f"处理文档时出错: {e}")
                continue
        
        if not valid_docs:
            self.logger.warning("没有有效的文档需要添加")
            return []
        
        try:
            # 分批处理，避免ChromaDB批处理大小限制
            batch_size = 1000  # ChromaDB推荐的批处理大小
            all_ids = []
            
            for i in range(0, len(valid_docs), batch_size):
                batch_end = min(i + batch_size, len(valid_docs))
                batch_docs = valid_docs[i:batch_end]
                
                self.logger.info(f"处理批次 {i//batch_size + 1}/{(len(valid_docs) + batch_size - 1)//batch_size}: {len(batch_docs)} 个文档")
                
                # 批量添加到Chroma
                self.collection.add(
                    ids=[doc['id'] for doc in batch_docs],
                    embeddings=[doc['embedding'] for doc in batch_docs],
                    documents=[doc['text'] for doc in batch_docs],
                    metadatas=[doc['metadata'] for doc in batch_docs]
                )
                
                all_ids.extend([doc['id'] for doc in batch_docs])
            
            self.logger.info(f"成功添加 {len(all_ids)} 个文档")
            return all_ids
            
        except Exception as e:
            self.logger.error(f"添加文档到Chroma失败: {e}")
            raise
    
    def _is_duplicate(self, doc_id: str) -> bool:
        """检查文档是否重复"""
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except:
            return False
    
    def search(self, 
               query: Union[str, List[float]], 
               n_results: int = 5,
               filter_metadata: Optional[Dict[str, Any]] = None,
               use_cache: bool = True) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        
        # 清理缓存
        self._clean_cache()
        
        # 生成缓存键
        cache_key = f"{hash(str(query))}_{n_results}_{hash(str(filter_metadata))}"
        
        if use_cache and cache_key in self._query_cache:
            self.logger.info("使用缓存查询结果")
            return self._query_cache[cache_key][0]
        
        try:
            # 执行搜索
            if isinstance(query, str):
                # 文本查询 - 需要先向量化
                try:
                    from ..text_processor.text_embedder import TextEmbedder
                    embedder = TextEmbedder()
                    query_embedding = embedder.embed_text(query)
                    
                    # 检查向量维度
                    collection_dimension = self._get_collection_dimension()
                    if collection_dimension and len(query_embedding) != collection_dimension:
                        self.logger.warning(f"查询向量维度不匹配: 查询{len(query_embedding)}维, 集合{collection_dimension}维")
                        return []
                    
                    result = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results,
                        where=filter_metadata
                    )
                except Exception as e:
                    self.logger.error(f"文本向量化失败: {e}")
                    return []
            else:
                # 向量查询
                # 检查向量维度
                collection_dimension = self._get_collection_dimension()
                if collection_dimension and len(query) != collection_dimension:
                    self.logger.warning(f"查询向量维度不匹配: 查询{len(query)}维, 集合{collection_dimension}维")
                    return []
                
                result = self.collection.query(
                    query_embeddings=[query],
                    n_results=n_results,
                    where=filter_metadata
                )
            
            # 格式化结果
            search_results = []
            if result['ids'] and result['ids'][0]:
                for i in range(len(result['ids'][0])):
                    search_results.append({
                        'id': result['ids'][0][i],
                        'text': result['documents'][0][i],
                        'metadata': result['metadatas'][0][i],
                        'distance': result['distances'][0][i] if 'distances' in result else None,
                        'score': 1 - result['distances'][0][i] if 'distances' in result else None
                    })
            
            # 缓存结果
            if use_cache:
                self._query_cache[cache_key] = (search_results, datetime.now())
            
            self.logger.info(f"搜索完成，返回 {len(search_results)} 个结果")
            return search_results
            
        except Exception as e:
            self.logger.error(f"搜索失败: {e}")
            raise
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """获取单个文档"""
        try:
            result = self.collection.get(ids=[doc_id])
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'text': result['documents'][0],
                    'metadata': result['metadatas'][0],
                    'embedding': result['embeddings'][0] if 'embeddings' in result else None
                }
            return None
        except Exception as e:
            self.logger.error(f"获取文档失败: {e}")
            return None
    
    def update_document(self, doc_id: str, 
                       text: Optional[str] = None,
                       embedding: Optional[List[float]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """更新文档"""
        try:
            # 获取现有文档
            existing = self.get_document(doc_id)
            if not existing:
                self.logger.warning(f"文档不存在: {doc_id}")
                return False
            
            # 准备更新数据
            update_data = {
                'text': text if text is not None else existing['text'],
                'embedding': embedding if embedding is not None else existing['embedding'],
                'metadata': metadata if metadata is not None else existing['metadata']
            }
            
            # 验证向量
            if embedding and not self._validate_embedding(embedding):
                self.logger.error(f"更新向量无效: {doc_id}")
                return False
            
            # 更新文档
            self.collection.update(
                ids=[doc_id],
                embeddings=[update_data['embedding']] if update_data['embedding'] else None,
                documents=[update_data['text']] if update_data['text'] else None,
                metadatas=[update_data['metadata']] if update_data['metadata'] else None
            )
            
            self.logger.info(f"文档更新成功: {doc_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"更新文档失败: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        try:
            self.collection.delete(ids=[doc_id])
            self.logger.info(f"文档删除成功: {doc_id}")
            return True
        except Exception as e:
            self.logger.error(f"删除文档失败: {e}")
            return False
    
    def delete_documents_by_filter(self, filter_metadata: Dict[str, Any]) -> int:
        """根据条件删除文档"""
        try:
            result = self.collection.delete(where=filter_metadata)
            count = result.get('count', 0)
            self.logger.info(f"根据条件删除 {count} 个文档")
            return count
        except Exception as e:
            self.logger.error(f"条件删除失败: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            # 获取集合信息
            count = self.collection.count()
            
            # 获取存储路径信息
            persist_dir = Path(self.config.storage.vector_db_persist_directory)
            total_size = sum(f.stat().st_size for f in persist_dir.rglob('*') if f.is_file())
            
            # 获取缓存信息
            cache_size = len(self._query_cache)
            
            return {
                'total_documents': count,
                'storage_path': str(persist_dir),
                'storage_size_bytes': total_size,
                'storage_size_mb': total_size / (1024 * 1024),
                'cache_size': cache_size,
                'collection_name': self.config.storage.vector_db_collection_name,
                'distance_metric': self.config.storage.vector_db_distance_metric
            }
            
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def clear_cache(self):
        """清理查询缓存"""
        self._query_cache.clear()
        self.logger.info("查询缓存已清理")
    
    def export_data(self, output_path: str, format: str = "json") -> bool:
        """导出数据"""
        try:
            # 获取所有文档
            result = self.collection.get()
            
            if format.lower() == "json":
                export_data = {
                    'metadata': {
                        'export_time': datetime.now().isoformat(),
                        'total_documents': len(result['ids']),
                        'collection_name': self.config.storage.vector_db_collection_name
                    },
                    'documents': []
                }
                
                for i in range(len(result['ids'])):
                    export_data['documents'].append({
                        'id': result['ids'][i],
                        'text': result['documents'][i],
                        'metadata': result['metadatas'][i],
                        'embedding': result['embeddings'][i] if 'embeddings' in result else None
                    })
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"数据导出成功: {output_path}")
                return True
                
            else:
                self.logger.error(f"不支持的导出格式: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"数据导出失败: {e}")
            return False
    
    def import_data(self, input_path: str, format: str = "json", 
                   skip_duplicates: bool = True) -> int:
        """导入数据"""
        try:
            if format.lower() == "json":
                with open(input_path, 'r', encoding='utf-8') as f:
                    import_data = json.load(f)
                
                documents = import_data.get('documents', [])
                imported_count = 0
                
                for doc in documents:
                    try:
                        # 检查重复
                        if skip_duplicates and self._is_duplicate(doc['id']):
                            continue
                        
                        # 添加文档
                        self.collection.add(
                            ids=[doc['id']],
                            embeddings=[doc['embedding']] if doc.get('embedding') else None,
                            documents=[doc['text']],
                            metadatas=[doc['metadata']]
                        )
                        imported_count += 1
                        
                    except Exception as e:
                        self.logger.warning(f"导入文档失败: {e}")
                        continue
                
                self.logger.info(f"数据导入成功: {imported_count} 个文档")
                return imported_count
                
            else:
                self.logger.error(f"不支持的导入格式: {format}")
                return 0
                
        except Exception as e:
            self.logger.error(f"数据导入失败: {e}")
            return 0
    
    def backup_database(self, backup_path: str) -> bool:
        """备份数据库"""
        try:
            source_path = Path(self.config.storage.vector_db_persist_directory)
            backup_path = Path(backup_path)
            
            # 创建备份目录
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 复制数据库文件
            if source_path.exists():
                shutil.copytree(source_path, backup_path, dirs_exist_ok=True)
                self.logger.info(f"数据库备份成功: {backup_path}")
                return True
            else:
                self.logger.warning("数据库目录不存在")
                return False
                
        except Exception as e:
            self.logger.error(f"数据库备份失败: {e}")
            return False
    
    def restore_database(self, backup_path: str) -> bool:
        """恢复数据库"""
        try:
            source_path = Path(self.config.storage.vector_db_persist_directory)
            backup_path = Path(backup_path)
            
            if not backup_path.exists():
                self.logger.error(f"备份文件不存在: {backup_path}")
                return False
            
            # 关闭当前连接
            self.client = None
            
            # 恢复数据库文件
            if source_path.exists():
                shutil.rmtree(source_path)
            shutil.copytree(backup_path, source_path)
            
            # 重新初始化
            self._init_chroma_client()
            self.collection = self._get_or_create_collection()
            
            self.logger.info(f"数据库恢复成功: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"数据库恢复失败: {e}")
            return False
    
    def reset_database(self) -> bool:
        """重置数据库"""
        try:
            # 删除集合
            self.client.delete_collection(self.config.storage.vector_db_collection_name)
            
            # 重新创建集合
            self.collection = self._get_or_create_collection()
            
            # 清理缓存
            self.clear_cache()
            
            self.logger.info("数据库重置成功")
            return True
            
        except Exception as e:
            self.logger.error(f"数据库重置失败: {e}")
            return False 