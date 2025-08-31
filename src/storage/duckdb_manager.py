"""
DuckDB数据库管理器
负责表格数据的存储、管理和SQL查询
"""

import duckdb
import pandas as pd
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import sqlite3

from ..utils.logger import LoggerMixin
from ..utils.config import get_config


class DuckDBManager(LoggerMixin):
    """DuckDB数据库管理器"""
    
    def __init__(self, db_path: Optional[str] = None):
        super().__init__()
        self.config = get_config()
        
        # 设置数据库路径
        if db_path is None:
            db_path = self.config.storage.duckdb.db_path
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库连接
        self.connection = None
        self._init_database()
        
        # 表元数据存储
        self.metadata_path = self.db_path.parent / "table_metadata.json"
        self.table_metadata = self._load_metadata()
    
    def _init_database(self):
        """初始化数据库连接"""
        try:
            self.connection = duckdb.connect(str(self.db_path))
            self.logger.info(f"DuckDB数据库初始化成功: {self.db_path}")
            
            # 创建系统表
            self._create_system_tables()
            
        except Exception as e:
            self.logger.error(f"DuckDB数据库初始化失败: {e}")
            raise
    
    def _create_system_tables(self):
        """创建系统表"""
        try:
            # 创建表元数据表
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS table_metadata (
                    table_name VARCHAR PRIMARY KEY,
                    source_file VARCHAR,
                    file_hash VARCHAR,
                    rows_count INTEGER,
                    columns_count INTEGER,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    schema_info TEXT,
                    sample_data TEXT
                )
            """)
            
            # 创建查询历史表
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY,
                    query_text TEXT,
                    table_name VARCHAR,
                    execution_time REAL,
                    result_rows INTEGER,
                    executed_at TIMESTAMP,
                    status VARCHAR
                )
            """)
            
            self.logger.info("系统表创建成功")
            
        except Exception as e:
            self.logger.error(f"创建系统表失败: {e}")
            raise
    
    def _load_metadata(self) -> Dict[str, Any]:
        """加载表元数据"""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.warning(f"加载元数据失败: {e}")
            return {}
    
    def _save_metadata(self):
        """保存表元数据"""
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.table_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存元数据失败: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """获取文件哈希值"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            self.logger.error(f"计算文件哈希失败: {e}")
            return ""
    
    def _generate_table_name(self, file_path: str) -> str:
        """生成表名"""
        file_name = Path(file_path).stem
        # 清理表名，只保留字母、数字和下划线
        table_name = ''.join(c for c in file_name if c.isalnum() or c == '_')
        # 确保表名不以数字开头
        if table_name and table_name[0].isdigit():
            table_name = f"table_{table_name}"
        return table_name.lower()
    
    def import_table(self, df: pd.DataFrame, file_path: str, table_name: Optional[str] = None) -> str:
        """
        导入表格数据到DuckDB
        
        Args:
            df: DataFrame数据
            file_path: 源文件路径
            table_name: 表名（可选，自动生成）
            
        Returns:
            str: 表名
        """
        try:
            if table_name is None:
                table_name = self._generate_table_name(file_path)
            
            # 检查表是否已存在
            if self.table_exists(table_name):
                self.logger.warning(f"表 {table_name} 已存在，将删除重建")
                self.drop_table(table_name)
            
            # 导入数据
            self.connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            
            # 获取文件哈希
            file_hash = self._get_file_hash(file_path)
            
            # 准备元数据
            schema_info = {
                'columns': df.columns.tolist(),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
                'shape': list(df.shape)
            }
            
            sample_data = df.head(5).to_dict('records')
            
            # 保存到系统表
            self.connection.execute("""
                INSERT OR REPLACE INTO table_metadata 
                (table_name, source_file, file_hash, rows_count, columns_count, 
                 created_at, updated_at, schema_info, sample_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                table_name,
                file_path,
                file_hash,
                len(df),
                len(df.columns),
                datetime.now(),
                datetime.now(),
                json.dumps(schema_info, ensure_ascii=False),
                json.dumps(sample_data, ensure_ascii=False, default=str)
            ))
            
            # 更新内存中的元数据
            self.table_metadata[table_name] = {
                'source_file': file_path,
                'file_hash': file_hash,
                'rows_count': len(df),
                'columns_count': len(df.columns),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'schema_info': schema_info,
                'sample_data': sample_data
            }
            
            self._save_metadata()
            
            self.logger.info(f"表格数据导入成功: {table_name} ({len(df)} 行, {len(df.columns)} 列)")
            return table_name
            
        except Exception as e:
            self.logger.error(f"导入表格数据失败: {e}")
            raise
    
    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        try:
            result = self.connection.execute(f"""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = '{table_name}'
            """).fetchone()
            return result[0] > 0
        except Exception as e:
            self.logger.error(f"检查表是否存在失败: {e}")
            return False
    
    def drop_table(self, table_name: str):
        """删除表"""
        try:
            if self.table_exists(table_name):
                self.connection.execute(f"DROP TABLE {table_name}")
                
                # 从系统表中删除
                self.connection.execute("DELETE FROM table_metadata WHERE table_name = ?", (table_name,))
                
                # 从内存中删除
                if table_name in self.table_metadata:
                    del self.table_metadata[table_name]
                    self._save_metadata()
                
                self.logger.info(f"表 {table_name} 删除成功")
        except Exception as e:
            self.logger.error(f"删除表失败: {e}")
            raise
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """获取表信息"""
        try:
            if not self.table_exists(table_name):
                return {}
            
            # 从系统表获取信息
            result = self.connection.execute("""
                SELECT * FROM table_metadata WHERE table_name = ?
            """, (table_name,)).fetchone()
            
            if result:
                return {
                    'table_name': result[0],
                    'source_file': result[1],
                    'file_hash': result[2],
                    'rows_count': result[3],
                    'columns_count': result[4],
                    'created_at': result[5],
                    'updated_at': result[6],
                    'schema_info': json.loads(result[7]) if result[7] else {},
                    'sample_data': json.loads(result[8]) if result[8] else []
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"获取表信息失败: {e}")
            return {}
    
    def list_tables(self) -> List[Dict[str, Any]]:
        """列出所有表"""
        try:
            result = self.connection.execute("""
                SELECT table_name, source_file, rows_count, columns_count, created_at
                FROM table_metadata
                ORDER BY created_at DESC
            """).fetchall()
            
            tables = []
            for row in result:
                tables.append({
                    'table_name': row[0],
                    'source_file': row[1],
                    'rows_count': row[2],
                    'columns_count': row[3],
                    'created_at': row[4]
                })
            
            return tables
            
        except Exception as e:
            self.logger.error(f"列出表失败: {e}")
            return []
    
    def execute_query(self, query: str, table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        执行SQL查询
        
        Args:
            query: SQL查询语句
            table_name: 相关表名（用于记录）
            
        Returns:
            Dict: 查询结果
        """
        start_time = datetime.now()
        
        try:
            # 执行查询
            result = self.connection.execute(query)
            
            # 获取结果
            if query.strip().upper().startswith('SELECT'):
                df_result = result.df()
                result_data = {
                    'success': True,
                    'data': df_result.to_dict('records'),
                    'columns': df_result.columns.tolist(),
                    'rows_count': len(df_result),
                    'execution_time': (datetime.now() - start_time).total_seconds()
                }
            else:
                result_data = {
                    'success': True,
                    'message': '查询执行成功',
                    'execution_time': (datetime.now() - start_time).total_seconds()
                }
            
            # 记录查询历史
            self._record_query_history(query, table_name, result_data['execution_time'], 
                                     result_data.get('rows_count', 0), 'SUCCESS')
            
            return result_data
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"SQL查询执行失败: {error_msg}")
            
            # 记录失败的查询
            self._record_query_history(query, table_name, 
                                     (datetime.now() - start_time).total_seconds(), 
                                     0, 'ERROR')
            
            return {
                'success': False,
                'error': error_msg,
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _record_query_history(self, query: str, table_name: Optional[str], 
                            execution_time: float, result_rows: int, status: str):
        """记录查询历史"""
        try:
            # 获取下一个ID
            result = self.connection.execute("SELECT MAX(id) FROM query_history").fetchone()
            next_id = 1 if result[0] is None else result[0] + 1
            
            self.connection.execute("""
                INSERT INTO query_history 
                (id, query_text, table_name, execution_time, result_rows, executed_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (next_id, query, table_name, execution_time, result_rows, datetime.now(), status))
        except Exception as e:
            self.logger.warning(f"记录查询历史失败: {e}")
    
    def get_query_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取查询历史"""
        try:
            result = self.connection.execute("""
                SELECT query_text, table_name, execution_time, result_rows, executed_at, status
                FROM query_history
                ORDER BY executed_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
            
            history = []
            for row in result:
                history.append({
                    'query_text': row[0],
                    'table_name': row[1],
                    'execution_time': row[2],
                    'result_rows': row[3],
                    'executed_at': row[4],
                    'status': row[5]
                })
            
            return history
            
        except Exception as e:
            self.logger.error(f"获取查询历史失败: {e}")
            return []
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """获取表结构"""
        try:
            if not self.table_exists(table_name):
                return {}
            
            # 获取列信息
            result = self.connection.execute(f"""
                DESCRIBE {table_name}
            """).fetchall()
            
            columns = []
            for row in result:
                columns.append({
                    'column_name': row[0],
                    'data_type': row[1],
                    'null': row[2],
                    'key': row[3],
                    'default': row[4],
                    'extra': row[5]
                })
            
            return {
                'table_name': table_name,
                'columns': columns,
                'total_columns': len(columns)
            }
            
        except Exception as e:
            self.logger.error(f"获取表结构失败: {e}")
            return {}
    
    def export_to_mysql_schema(self, table_name: str) -> str:
        """导出MySQL建表语句"""
        try:
            schema = self.get_table_schema(table_name)
            if not schema:
                return ""
            
            mysql_schema = f"CREATE TABLE `{table_name}` (\n"
            
            for i, column in enumerate(schema['columns']):
                mysql_schema += f"  `{column['column_name']}` {column['data_type']}"
                
                if column['null'] == 'NO':
                    mysql_schema += " NOT NULL"
                
                if column['default']:
                    mysql_schema += f" DEFAULT {column['default']}"
                
                if i < len(schema['columns']) - 1:
                    mysql_schema += ","
                
                mysql_schema += "\n"
            
            mysql_schema += ");"
            
            return mysql_schema
            
        except Exception as e:
            self.logger.error(f"导出MySQL建表语句失败: {e}")
            return ""
    
    def close(self):
        """关闭数据库连接"""
        try:
            if self.connection:
                self.connection.close()
                self.logger.info("DuckDB连接已关闭")
        except Exception as e:
            self.logger.error(f"关闭数据库连接失败: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 