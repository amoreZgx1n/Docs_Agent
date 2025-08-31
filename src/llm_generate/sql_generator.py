"""
SQL生成器模块
负责根据用户查询和表结构生成SQL语句
"""

import json
from typing import Dict, List, Optional, Any
from ..utils.logger import LoggerMixin
from ..llm_generate.llm import LLM


class SQLGenerator(LoggerMixin):
    """SQL生成器"""
    
    def __init__(self, llm_backend: str = "ollama"):
        super().__init__()
        self.llm = None
        self.use_llm = False
        
        # 暂时禁用LLM，直接使用规则引擎
        self.logger.info("使用规则引擎进行SQL生成")
    
    def generate_sql(self, user_query: str, table_info: Dict[str, Any], 
                    table_name: str, sample_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        根据用户查询生成SQL语句
        
        Args:
            user_query: 用户查询
            table_info: 表信息
            table_name: 表名
            sample_data: 样本数据（可选）
            
        Returns:
            Dict: 包含SQL语句和解释的结果
        """
        try:
            if self.use_llm and self.llm:
                # 使用LLM生成SQL
                prompt = self._build_sql_prompt(user_query, table_info, table_name, sample_data)
                response = self.llm.complete(prompt)
                result = self._parse_sql_response(response)
            else:
                # 使用规则引擎生成SQL
                result = self._rule_based_sql_generation(user_query, table_info, table_name)
            
            return {
                'success': True,
                'sql': result.get('sql', ''),
                'explanation': result.get('explanation', ''),
                'table_name': table_name,
                'user_query': user_query
            }
            
        except Exception as e:
            self.logger.error(f"SQL生成失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'user_query': user_query
            }
    
    def _build_sql_prompt(self, user_query: str, table_info: Dict[str, Any], 
                         table_name: str, sample_data: Optional[List[Dict]] = None) -> str:
        """构建SQL生成提示词"""
        
        # 获取表结构信息
        columns_info = []
        for column in table_info.get('columns', []):
            col_info = f"- {column['column_name']} ({column['data_type']})"
            if column.get('null') == 'NO':
                col_info += " NOT NULL"
            columns_info.append(col_info)
        
        columns_text = "\n".join(columns_info)
        
        # 构建样本数据文本
        sample_text = ""
        if sample_data:
            sample_text = "\n\n样本数据:\n"
            for i, row in enumerate(sample_data[:3]):  # 只显示前3行
                sample_text += f"行 {i+1}: {row}\n"
        
        prompt = f"""
你是一个专业的SQL生成器。请根据用户查询和表结构信息生成准确的SQL语句。

表名: {table_name}

表结构:
{columns_text}

{sample_text}

用户查询: {user_query}

请生成相应的SQL语句，并解释你的思路。

请按以下格式返回：
```sql
-- 生成的SQL语句
SELECT ...
```

解释：
- 查询目的：[说明查询的目的]
- 使用的方法：[说明使用的SQL技术]
- 注意事项：[如果有任何需要注意的地方]
"""
        
        return prompt
    
    def _rule_based_sql_generation(self, user_query: str, table_info: Dict[str, Any], table_name: str) -> Dict[str, str]:
        """基于规则的SQL生成"""
        try:
            query_lower = user_query.lower()
            columns = [col['column_name'].lower() for col in table_info.get('columns', [])]
            
            # 简单的规则匹配
            if '所有' in query_lower or '全部' in query_lower:
                sql = f"SELECT * FROM {table_name}"
                explanation = "查询所有数据"
            
            elif '总数' in query_lower or '数量' in query_lower:
                sql = f"SELECT COUNT(*) as total_count FROM {table_name}"
                explanation = "统计记录总数"
            
            elif '最高' in query_lower or '最大' in query_lower:
                # 查找数值列
                numeric_columns = []
                for col in table_info.get('columns', []):
                    col_type = col['data_type'].lower()
                    if any(num_type in col_type for num_type in ['int', 'decimal', 'float', 'double']):
                        numeric_columns.append(col['column_name'])
                
                if numeric_columns:
                    amount_col = numeric_columns[0]  # 使用第一个数值列
                    sql = f"SELECT * FROM {table_name} ORDER BY {amount_col} DESC LIMIT 5"
                    explanation = f"按{amount_col}降序排列，取前5条记录"
                else:
                    sql = f"SELECT * FROM {table_name} LIMIT 5"
                    explanation = "取前5条记录"
            
            elif '分类' in query_lower or '分组' in query_lower:
                # 查找可能的分类列
                category_columns = []
                for col in table_info.get('columns', []):
                    col_name = col['column_name'].lower()
                    if any(cat in col_name for cat in ['name', 'type', 'category', 'product', 'customer']):
                        category_columns.append(col['column_name'])
                
                if category_columns:
                    category_col = category_columns[0]
                    sql = f"SELECT {category_col}, COUNT(*) as count FROM {table_name} GROUP BY {category_col}"
                    explanation = f"按{category_col}分组统计数量"
                else:
                    # 如果没有找到分类列，尝试使用第一个非数值列
                    non_numeric_columns = []
                    for col in table_info.get('columns', []):
                        col_type = col['data_type'].lower()
                        if not any(num_type in col_type for num_type in ['int', 'decimal', 'float', 'double']):
                            non_numeric_columns.append(col['column_name'])
                    
                    if non_numeric_columns:
                        category_col = non_numeric_columns[0]
                        sql = f"SELECT {category_col}, COUNT(*) as count FROM {table_name} GROUP BY {category_col}"
                        explanation = f"按{category_col}分组统计数量"
                    else:
                        sql = f"SELECT COUNT(*) as total_count FROM {table_name}"
                        explanation = "统计总数"
            
            elif '2023' in query_lower or '年份' in query_lower:
                # 查找日期列
                date_columns = []
                for col in table_info.get('columns', []):
                    col_type = col['data_type'].lower()
                    if 'date' in col_type or 'time' in col_type:
                        date_columns.append(col['column_name'])
                
                if date_columns:
                    date_col = date_columns[0]
                    sql = f"SELECT * FROM {table_name} WHERE YEAR({date_col}) = 2023"
                    explanation = f"查询2023年的数据，使用{date_col}列"
                else:
                    sql = f"SELECT * FROM {table_name}"
                    explanation = "查询所有数据（未找到日期列）"
            
            else:
                # 默认查询
                sql = f"SELECT * FROM {table_name} LIMIT 10"
                explanation = "默认查询前10条记录"
            
            return {
                'sql': sql,
                'explanation': explanation
            }
            
        except Exception as e:
            self.logger.error(f"规则引擎SQL生成失败: {e}")
            return {
                'sql': f"SELECT * FROM {table_name} LIMIT 10",
                'explanation': f"生成失败，使用默认查询: {str(e)}"
            }
    
    def _parse_sql_response(self, response: str) -> Dict[str, str]:
        """解析LLM响应，提取SQL语句和解释"""
        try:
            # 尝试提取SQL代码块
            if '```sql' in response:
                sql_start = response.find('```sql') + 6
                sql_end = response.find('```', sql_start)
                sql = response[sql_start:sql_end].strip()
            elif '```' in response:
                # 如果没有sql标记，尝试提取第一个代码块
                code_start = response.find('```') + 3
                code_end = response.find('```', code_start)
                sql = response[code_start:code_end].strip()
            else:
                # 如果没有代码块，尝试提取SELECT语句
                lines = response.split('\n')
                sql_lines = []
                for line in lines:
                    if line.strip().upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')):
                        sql_lines.append(line.strip())
                sql = ' '.join(sql_lines)
            
            # 提取解释部分
            explanation = ""
            if '解释：' in response:
                explanation_start = response.find('解释：')
                explanation = response[explanation_start:].strip()
            elif '解释:' in response:
                explanation_start = response.find('解释:')
                explanation = response[explanation_start:].strip()
            
            return {
                'sql': sql,
                'explanation': explanation
            }
            
        except Exception as e:
            self.logger.warning(f"解析SQL响应失败: {e}")
            return {
                'sql': response,
                'explanation': '无法解析响应格式'
            }
    
    def validate_sql(self, sql: str, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证SQL语句的语法和逻辑
        
        Args:
            sql: SQL语句
            table_info: 表信息
            
        Returns:
            Dict: 验证结果
        """
        try:
            # 基本语法检查
            sql_upper = sql.upper()
            
            # 检查是否包含SELECT
            if not sql_upper.startswith('SELECT'):
                return {
                    'valid': False,
                    'error': 'SQL语句必须以SELECT开头'
                }
            
            # 检查表名是否存在
            table_name = table_info.get('table_name', '')
            if table_name and table_name.lower() not in sql.lower():
                return {
                    'valid': False,
                    'error': f'SQL语句中未找到表名: {table_name}'
                }
            
            # 检查列名是否存在
            valid_columns = [col['column_name'].lower() for col in table_info.get('columns', [])]
            sql_lower = sql.lower()
            
            # 简单的列名检查（可以改进）
            for col in valid_columns:
                if col in sql_lower and col != table_name.lower():
                    break
            else:
                # 如果没有找到任何列名，可能是使用了*
                if '*' not in sql_lower:
                    return {
                        'valid': False,
                        'error': 'SQL语句中未找到有效的列名'
                    }
            
            return {
                'valid': True,
                'message': 'SQL语句验证通过'
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'验证过程中发生错误: {str(e)}'
            }
    
    def optimize_sql(self, sql: str, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        优化SQL语句
        
        Args:
            sql: 原始SQL语句
            table_info: 表信息
            
        Returns:
            Dict: 优化结果
        """
        try:
            # 构建优化提示词
            prompt = f"""
你是一个SQL优化专家。请分析并优化以下SQL语句：

表结构:
{json.dumps(table_info, ensure_ascii=False, indent=2)}

原始SQL:
{sql}

请提供优化建议和优化后的SQL语句。考虑以下方面：
1. 性能优化
2. 索引使用
3. 查询效率
4. 可读性

请按以下格式返回：
```sql
-- 优化后的SQL
SELECT ...
```

优化说明：
- 性能改进：[说明性能方面的改进]
- 可读性改进：[说明可读性方面的改进]
- 建议：[其他建议]
"""
            
            response = self.llm.complete(prompt)
            
            # 解析优化结果
            result = self._parse_sql_response(response)
            
            return {
                'success': True,
                'original_sql': sql,
                'optimized_sql': result.get('sql', sql),
                'optimization_notes': result.get('explanation', ''),
                'table_info': table_info
            }
            
        except Exception as e:
            self.logger.error(f"SQL优化失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'original_sql': sql
            }
    
    def explain_sql(self, sql: str, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        解释SQL语句的执行计划
        
        Args:
            sql: SQL语句
            table_info: 表信息
            
        Returns:
            Dict: 解释结果
        """
        try:
            prompt = f"""
你是一个SQL执行计划分析专家。请分析以下SQL语句的执行计划：

表结构:
{json.dumps(table_info, ensure_ascii=False, indent=2)}

SQL语句:
{sql}

请分析：
1. 执行步骤
2. 可能的性能瓶颈
3. 建议的优化方向
4. 预期执行时间

请详细解释每个步骤的作用和影响。
"""
            
            response = self.llm.complete(prompt)
            
            return {
                'success': True,
                'sql': sql,
                'explanation': response,
                'table_info': table_info
            }
            
        except Exception as e:
            self.logger.error(f"SQL解释失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'sql': sql
            } 