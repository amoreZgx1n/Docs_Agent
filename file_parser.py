"""
文件解析器 (File Parser)
负责解析不同类型的输入文件，为三种任务提供数据输入
"""

import sys
import os
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json

# 添加路径
sys.path.append('/home/icesnls/workplace/zgx/work/Docs_Agent')


@dataclass
class DocumentInput:
    """文档问答任务的输入"""
    query: str
    file_path: str
    content: str
    metadata: Dict[str, Any]


@dataclass
class SheetInput:
    """表格分析任务的输入"""
    query: str
    file_path: str
    dataframe: pd.DataFrame
    metadata: Dict[str, Any]


@dataclass
class SQLInput:
    """SQL分析任务的输入"""
    query: str
    db_config: Dict[str, Any]
    table_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None


class PDFParser:
    """PDF文件解析器"""

    def __init__(self):
        self.supported_formats = ['.pdf']
        try:
            # 尝试导入PDF解析库
            import PyPDF2
            self.pdf_lib = 'PyPDF2'
        except ImportError:
            try:
                import fitz  # PyMuPDF
                self.pdf_lib = 'PyMuPDF'
            except ImportError:
                print("⚠️ 未找到PDF解析库，请安装 PyPDF2 或 PyMuPDF")
                print("安装命令: pip install PyPDF2 或 pip install PyMuPDF")
                self.pdf_lib = None

    def can_parse(self, file_path: str) -> bool:
        """检查是否可以解析该文件"""
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_formats

    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        解析PDF文件

        Args:
            file_path: PDF文件路径

        Returns:
            包含解析结果的字典
        """
        if not self.pdf_lib:
            raise ImportError("未找到可用的PDF解析库")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        try:
            if self.pdf_lib == 'PyPDF2':
                return self._parse_with_pypdf2(file_path)
            elif self.pdf_lib == 'PyMuPDF':
                return self._parse_with_pymupdf(file_path)
            else:
                raise ImportError("不支持的PDF解析库")

        except Exception as e:
            print(f"PDF解析失败: {e}")
            # 返回基本的文件信息作为兜底
            return {
                'content': f"PDF文件解析失败: {str(e)}",
                'page_count': 0,
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'error': str(e)
            }

    def _parse_with_pypdf2(self, file_path: str) -> Dict[str, Any]:
        """使用PyPDF2解析PDF"""
        import PyPDF2

        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            # 提取文本内容
            content_parts = []
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text.strip():
                    content_parts.append(f"第{page_num + 1}页:\n{text.strip()}")

            full_content = "\n\n".join(content_parts)

            # 获取元数据
            metadata = {}
            if pdf_reader.metadata:
                metadata = {
                    'title': pdf_reader.metadata.get('/Title', ''),
                    'author': pdf_reader.metadata.get('/Author', ''),
                    'subject': pdf_reader.metadata.get('/Subject', ''),
                    'creator': pdf_reader.metadata.get('/Creator', ''),
                    'producer': pdf_reader.metadata.get('/Producer', ''),
                }

            return {
                'content': full_content,
                'page_count': len(pdf_reader.pages),
                'file_size': os.path.getsize(file_path),
                'metadata': metadata,
                'parser': 'PyPDF2'
            }

    def _parse_with_pymupdf(self, file_path: str) -> Dict[str, Any]:
        """使用PyMuPDF解析PDF"""
        import fitz

        doc = fitz.open(file_path)

        # 提取文本内容
        content_parts = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                content_parts.append(f"第{page_num + 1}页:\n{text.strip()}")

        full_content = "\n\n".join(content_parts)

        # 获取元数据
        metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'creator': doc.metadata.get('creator', ''),
            'producer': doc.metadata.get('producer', ''),
        }

        doc.close()

        return {
            'content': full_content,
            'page_count': doc.page_count,
            'file_size': os.path.getsize(file_path),
            'metadata': metadata,
            'parser': 'PyMuPDF'
        }


class CSVParser:
    """CSV文件解析器"""

    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls']
        self.encoding_options = ['utf-8', 'gbk', 'gb2312', 'latin1']

    def can_parse(self, file_path: str) -> bool:
        """检查是否可以解析该文件"""
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_formats

    def parse_csv(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        解析CSV/Excel文件

        Args:
            file_path: 文件路径
            **kwargs: pandas读取参数

        Returns:
            包含DataFrame和元数据的字典
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_ext = Path(file_path).suffix.lower()

        try:
            if file_ext == '.csv':
                df = self._parse_csv_file(file_path, **kwargs)
            elif file_ext in ['.xlsx', '.xls']:
                df = self._parse_excel_file(file_path, **kwargs)
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")

            # 生成元数据
            metadata = {
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'memory_usage': df.memory_usage(deep=True).sum(),
                'null_counts': df.isnull().sum().to_dict(),
                'parser': 'pandas'
            }

            # 数据质量检查
            quality_info = self._check_data_quality(df)
            metadata.update(quality_info)

            return {
                'dataframe': df,
                'metadata': metadata
            }

        except Exception as e:
            print(f"文件解析失败: {e}")
            # 返回空的DataFrame作为兜底
            empty_df = pd.DataFrame()
            return {
                'dataframe': empty_df,
                'metadata': {
                    'error': str(e),
                    'file_path': file_path,
                    'parser': 'pandas'
                }
            }

    def _parse_csv_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """解析CSV文件"""
        # 尝试不同的编码
        for encoding in self.encoding_options:
            try:
                df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                print(f"✅ CSV文件使用编码 {encoding} 成功解析")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"使用编码 {encoding} 解析失败: {e}")
                continue

        # 如果所有编码都失败，使用pandas默认行为
        return pd.read_csv(file_path, **kwargs)

    def _parse_excel_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """解析Excel文件"""
        return pd.read_excel(file_path, **kwargs)

    def _check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查数据质量"""
        quality = {}

        # 基本统计
        quality['total_rows'] = len(df)
        quality['total_columns'] = len(df.columns)
        quality['duplicate_rows'] = df.duplicated().sum()
        quality['total_nulls'] = df.isnull().sum().sum()

        # 列类型分布
        dtype_counts = df.dtypes.value_counts()
        quality['dtype_distribution'] = {str(dtype): count for dtype, count in dtype_counts.items()}

        # 数值列统计
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            quality['numeric_columns'] = list(numeric_cols)
            quality['numeric_stats'] = df[numeric_cols].describe().to_dict()

        # 文本列统计
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            quality['text_columns'] = list(text_cols)
            text_stats = {}
            for col in text_cols:
                text_stats[col] = {
                    'unique_values': df[col].nunique(),
                    'avg_length': df[col].astype(str).str.len().mean(),
                    'max_length': df[col].astype(str).str.len().max()
                }
            quality['text_stats'] = text_stats

        return quality


class DBConfigParser:
    """数据库配置解析器"""

    def __init__(self):
        self.required_fields = ['host', 'database', 'user']
        self.optional_fields = ['password', 'port', 'charset', 'table']

    def parse_db_config(self, config_input: Union[str, Dict[str, Any], Path]) -> Dict[str, Any]:
        """
        解析数据库配置

        Args:
            config_input: 配置文件路径或配置字典

        Returns:
            解析后的数据库配置
        """
        if isinstance(config_input, str):
            # 如果是字符串，尝试作为文件路径或JSON字符串解析
            if os.path.exists(config_input):
                return self._parse_config_file(config_input)
            else:
                return self._parse_config_json(config_input)
        elif isinstance(config_input, dict):
            return self._validate_config(config_input)
        elif isinstance(config_input, Path):
            return self._parse_config_file(str(config_input))
        else:
            raise ValueError(f"不支持的配置输入类型: {type(config_input)}")

    def _parse_config_file(self, file_path: str) -> Dict[str, Any]:
        """解析配置文件"""
        file_ext = Path(file_path).suffix.lower()

        try:
            if file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            elif file_ext == '.yaml' or file_ext == '.yml':
                import yaml
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {file_ext}")

            return self._validate_config(config)

        except Exception as e:
            raise ValueError(f"解析配置文件失败: {e}")

    def _parse_config_json(self, json_str: str) -> Dict[str, Any]:
        """解析JSON字符串配置"""
        try:
            config = json.loads(json_str)
            return self._validate_config(config)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析失败: {e}")

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置完整性"""
        validated_config = {}

        # 检查必需字段
        for field in self.required_fields:
            if field not in config:
                raise ValueError(f"缺少必需的配置字段: {field}")
            validated_config[field] = config[field]

        # 添加可选字段
        for field in self.optional_fields:
            if field in config:
                validated_config[field] = config[field]
            else:
                # 设置默认值
                if field == 'port':
                    validated_config[field] = 3306
                elif field == 'charset':
                    validated_config[field] = 'utf8mb4'
                elif field == 'password':
                    validated_config[field] = ''

        # 验证端口号
        if not isinstance(validated_config['port'], int) or validated_config['port'] <= 0:
            raise ValueError("端口号必须是正整数")

        # 生成连接字符串
        validated_config['connection_string'] = self._generate_connection_string(validated_config)

        return validated_config

    def _generate_connection_string(self, config: Dict[str, Any]) -> str:
        """生成数据库连接字符串"""
        password_part = f":{config['password']}@" if config.get('password') else "@"
        return f"mysql+pymysql://{config['user']}{password_part}{config['host']}:{config['port']}/{config['database']}"

    def get_table_info(self, config: Dict[str, Any], table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取表信息（如果可能的话）

        Args:
            config: 数据库配置
            table_name: 表名（如果已知）

        Returns:
            表信息字典
        """
        table_info = {
            'table_name': table_name or config.get('table'),
            'estimated_row_count': None,
            'columns': None,
            'primary_key': None,
            'indexes': None
        }

        # 这里可以添加实际的数据库连接来获取表信息
        # 暂时返回基本信息
        return table_info


class UnifiedFileParser:
    """统一文件解析器"""

    def __init__(self):
        self.pdf_parser = PDFParser()
        self.csv_parser = CSVParser()
        self.db_parser = DBConfigParser()

    def parse_document_input(self, query: str, file_path: str) -> DocumentInput:
        """
        解析文档问答任务的输入

        Args:
            query: 用户查询
            file_path: PDF文件路径

        Returns:
            DocumentInput 对象
        """
        print(f"📄 解析文档输入: {file_path}")

        if not self.pdf_parser.can_parse(file_path):
            raise ValueError(f"不支持的文档格式: {file_path}")

        parsed_result = self.pdf_parser.parse_pdf(file_path)

        return DocumentInput(
            query=query,
            file_path=file_path,
            content=parsed_result['content'],
            metadata={
                'parser': parsed_result.get('parser'),
                'page_count': parsed_result.get('page_count', 0),
                'file_size': parsed_result.get('file_size', 0),
                'pdf_metadata': parsed_result.get('metadata', {}),
                'parse_time': pd.Timestamp.now().isoformat()
            }
        )

    def parse_sheet_input(self, query: str, file_path: str, **kwargs) -> SheetInput:
        """
        解析表格分析任务的输入

        Args:
            query: 用户查询
            file_path: CSV/Excel文件路径
            **kwargs: 解析参数

        Returns:
            SheetInput 对象
        """
        print(f"📊 解析表格输入: {file_path}")

        if not self.csv_parser.can_parse(file_path):
            raise ValueError(f"不支持的表格格式: {file_path}")

        parsed_result = self.csv_parser.parse_csv(file_path, **kwargs)

        return SheetInput(
            query=query,
            file_path=file_path,
            dataframe=parsed_result['dataframe'],
            metadata={
                'parser': parsed_result['metadata'].get('parser'),
                'shape': parsed_result['metadata'].get('shape'),
                'columns': parsed_result['metadata'].get('columns'),
                'dtypes': parsed_result['metadata'].get('dtypes'),
                'quality_info': parsed_result['metadata'],
                'parse_time': pd.Timestamp.now().isoformat()
            }
        )

    def parse_sql_input(self, query: str, db_config_input: Union[str, Dict[str, Any]]) -> SQLInput:
        """
        解析SQL分析任务的输入

        Args:
            query: 用户查询
            db_config_input: 数据库配置（文件路径或字典）

        Returns:
            SQLInput 对象
        """
        print(f"🗄️ 解析数据库输入配置")

        db_config = self.db_parser.parse_db_config(db_config_input)
        table_info = self.db_parser.get_table_info(db_config)

        return SQLInput(
            query=query,
            db_config=db_config,
            table_info=table_info,
            metadata={
                'connection_string': db_config.get('connection_string'),
                'has_table_info': table_info.get('table_name') is not None,
                'config_source': 'file' if isinstance(db_config_input, str) and os.path.exists(db_config_input) else 'dict',
                'parse_time': pd.Timestamp.now().isoformat()
            }
        )

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """获取支持的文件格式"""
        return {
            'document': self.pdf_parser.supported_formats,
            'sheet': self.csv_parser.supported_formats,
            'database': ['.json', '.yaml', '.yml']  # 数据库配置支持的格式
        }


# 全局解析器实例
file_parser = UnifiedFileParser()


# 便捷函数
def parse_document_input(query: str, file_path: str) -> DocumentInput:
    """解析文档输入的便捷函数"""
    return file_parser.parse_document_input(query, file_path)


def parse_sheet_input(query: str, file_path: str, **kwargs) -> SheetInput:
    """解析表格输入的便捷函数"""
    return file_parser.parse_sheet_input(query, file_path, **kwargs)


def parse_sql_input(query: str, db_config_input: Union[str, Dict[str, Any]]) -> SQLInput:
    """解析SQL输入的便捷函数"""
    return file_parser.parse_sql_input(query, db_config_input)


if __name__ == "__main__":
    # 测试代码
    print("UnifiedFileParser 测试")
    print(f"支持的文件格式: {file_parser.get_supported_formats()}")

    # 示例用法
    print("\n示例用法:"    )
    print("1. 文档问答: parse_document_input(query, 'document.pdf')")
    print("2. 表格分析: parse_sheet_input(query, 'data.csv')")
    print("3. SQL分析: parse_sql_input(query, 'db_config.json') 或 parse_sql_input(query, {...})")
