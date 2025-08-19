"""
表格处理器模块
负责CSV、Excel等表格文件的处理
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
from datetime import datetime
from ..utils.logger import LoggerMixin
from ..utils.config import get_config
from ..utils.file_utils import get_file_utils


class TableProcessor(LoggerMixin):
    """表格处理器类"""
    
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.file_utils = get_file_utils()
        
        # 支持的格式
        self.supported_formats = self.config.document_processing.table_supported_formats
        self.max_file_size_mb = self.config.document_processing.table_max_file_size_mb
        self.default_encoding = self.config.document_processing.table_encoding
        self.excel_engine = self.config.document_processing.table_engine
    
    def load_table(self, file_path: str, **kwargs) -> pd.DataFrame:
        """加载表格文件"""
        file_path = Path(file_path)
        self.log_info(f"开始加载表格文件: {file_path}")
        
        try:
            if not self._validate_table_file(file_path):
                raise ValueError(f"表格文件验证失败: {file_path}")
            
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.csv':
                return self._load_csv(file_path, **kwargs)
            elif file_extension in ['.xlsx', '.xls']:
                return self._load_excel(file_path, **kwargs)
            else:
                raise ValueError(f"不支持的文件格式: {file_extension}")
                
        except Exception as e:
            self.log_error(f"表格加载失败: {e}")
            raise
    
    def _validate_table_file(self, file_path: Path) -> bool:
        """验证表格文件"""
        if not file_path.exists():
            self.log_error(f"表格文件不存在: {file_path}")
            return False
        
        if file_path.suffix.lower() not in self.supported_formats:
            self.log_error(f"不支持的文件格式: {file_path.suffix}")
            return False
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            self.log_error(f"文件大小超出限制: {file_size_mb:.2f}MB > {self.max_file_size_mb}MB")
            return False
        
        return True
    
    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """加载CSV文件"""
        self.log_info(f"加载CSV文件: {file_path}")
        
        default_params = {
            'encoding': self.default_encoding,
            'sep': ',',
            'header': 0,
            'na_values': ['', 'nan', 'NaN', 'NULL', 'null'],
            'keep_default_na': True,
            'low_memory': False
        }
        
        read_params = {**default_params, **kwargs}
        
        try:
            encodings_to_try = [self.default_encoding, 'utf-8', 'gbk', 'gb2312', 'latin1']
            
            for encoding in encodings_to_try:
                try:
                    read_params['encoding'] = encoding
                    df = pd.read_csv(file_path, **read_params)
                    self.log_info(f"CSV文件加载成功，使用编码: {encoding}")
                    return self._post_process_dataframe(df, file_path)
                except UnicodeDecodeError:
                    self.log_warning(f"编码 {encoding} 失败，尝试下一个")
                    continue
                except Exception as e:
                    self.log_warning(f"使用编码 {encoding} 加载失败: {e}")
                    continue
            
            raise ValueError("所有编码方式都失败了")
            
        except Exception as e:
            self.log_error(f"CSV文件加载失败: {e}")
            raise
    
    def _load_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """加载Excel文件"""
        self.log_info(f"加载Excel文件: {file_path}")
        
        default_params = {
            'engine': self.excel_engine,
            'header': 0,
            'na_values': ['', 'nan', 'NaN', 'NULL', 'null'],
            'keep_default_na': True
        }
        
        read_params = {**default_params, **kwargs}
        
        try:
            df = pd.read_excel(file_path, **read_params)
            self.log_info(f"Excel文件加载成功")
            return self._post_process_dataframe(df, file_path)
            
        except Exception as e:
            self.log_error(f"Excel文件加载失败: {e}")
            raise
    
    def _post_process_dataframe(self, df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
        """后处理DataFrame"""
        self.log_info(f"开始后处理DataFrame，原始形状: {df.shape}")
        
        # 清理列名
        df.columns = self._clean_column_names(df.columns)
        
        # 处理数据类型
        df = self._optimize_data_types(df)
        
        # 处理缺失值
        df = self._handle_missing_values(df)
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        self.log_info(f"DataFrame后处理完成，最终形状: {df.shape}")
        return df
    
    def _clean_column_names(self, columns: pd.Index) -> pd.Index:
        """清理列名"""
        cleaned_columns = []
        
        for col in columns:
            col_str = str(col).strip()
            col_str = col_str.replace('\n', ' ').replace('\r', ' ')
            col_str = ' '.join(col_str.split())
            
            if not col_str or col_str.isspace():
                col_str = f"Column_{len(cleaned_columns) + 1}"
            
            cleaned_columns.append(col_str)
        
        return pd.Index(cleaned_columns)
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化数据类型"""
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col], errors='raise')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except (ValueError, TypeError):
                    try:
                        pd.to_datetime(df[col], errors='raise')
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except (ValueError, TypeError):
                        pass
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            self.log_info(f"发现缺失值: {missing_counts.to_dict()}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "未知"
                df[col].fillna(mode_val, inplace=True)
        
        return df
    
    def get_table_info(self, file_path: str) -> Dict[str, Any]:
        """获取表格文件信息"""
        file_path = Path(file_path)
        self.log_info(f"获取表格文件信息: {file_path}")
        
        try:
            if not self._validate_table_file(file_path):
                return {}
            
            df = self.load_table(str(file_path))
            
            info = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                "file_format": file_path.suffix.lower(),
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                "sample_data": df.head(5).to_dict('records')
            }
            
            return info
            
        except Exception as e:
            self.log_error(f"获取表格信息失败: {e}")
            return {}
    
    def get_sheet_names(self, file_path: str) -> List[str]:
        """获取Excel文件中的工作表名称"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() not in ['.xlsx', '.xls']:
            return []
        
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            self.log_info(f"Excel文件工作表: {sheet_names}")
            return sheet_names
            
        except Exception as e:
            self.log_error(f"获取工作表名称失败: {e}")
            return []
    
    def load_specific_sheet(self, file_path: str, sheet_name: str, **kwargs) -> pd.DataFrame:
        """加载Excel文件的特定工作表"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() not in ['.xlsx', '.xls']:
            raise ValueError("文件不是Excel格式")
        
        try:
            sheet_names = self.get_sheet_names(str(file_path))
            if sheet_name not in sheet_names:
                raise ValueError(f"工作表 '{sheet_name}' 不存在。可用工作表: {sheet_names}")
            
            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                engine=self.excel_engine,
                **kwargs
            )
            
            self.log_info(f"成功加载工作表: {sheet_name}")
            return self._post_process_dataframe(df, file_path)
            
        except Exception as e:
            self.log_error(f"加载工作表失败: {e}")
            raise 