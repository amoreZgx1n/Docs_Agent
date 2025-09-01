from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re
import json

from src.llm_generate.llm import LLM
from src.llm_generate.schema_manager_mysql import MySQLSchemaManager


@dataclass
class ParsedIntent:
    query_type: str
    metrics: List[str]
    dimensions: List[str]
    filters: List[Dict[str, Any]]
    order_by: List[Dict[str, str]]


class SQLGenerator:
    """
    高级/通用版：支持多表（需要 schema 摘要），分三步：
    1) 解析意图 -> JSON
    2) 表识别 -> 表名列表
    3) 生成 SQL -> 校验/修正
    适合未来扩展多表 JOIN 的复杂场景。
    """
    def __init__(self, llm: LLM, schema_manager: MySQLSchemaManager):
        self.llm = llm
        self.schema_manager = schema_manager

    async def generate_sql(
        self,
        user_intent: str,
        *,
        context: Optional[Dict] = None,
        tables: Optional[List[str]] = None,
        limit: int = 100,
    ) -> str:
        # 让 LLM 只“看见”必要的表（如未指定 tables，则让 schema_manager 返回所有表）
        schema_summary = (self.schema_manager.get_schema_summary(tables=tables)
                          if hasattr(self.schema_manager, "get_schema_summary")
                          else self.schema_manager.get_table_summary(tables[0]))

        # 1. 解析意图
        parsed_intent = await self._parse_intent(user_intent, context, schema_summary)

        # 2. 表识别（如已显式传入 tables，可跳过识别）
        relevant_tables = await self._identify_tables(parsed_intent, schema_summary) if not tables else tables

        # 3. 生成 SQL
        sql = await self._construct_sql(parsed_intent, relevant_tables, context, schema_summary, limit)

        # 4. 校验/优化
        return self._optimize_sql(sql, limit)

    async def _parse_intent(self, user_intent: str, context: Optional[Dict], schema_summary: Dict[str, Any]) -> ParsedIntent:
        prompt = f"""
你是一个 SQL 生成助手。
任务：将用户的自然语言分析需求转换为结构化查询意图。

用户需求:
{user_intent}

上下文:
{json.dumps(context or {}, ensure_ascii=False)}

数据库结构摘要(供参考，避免臆造表/列):
{json.dumps(schema_summary, ensure_ascii=False, indent=2)}

请以 JSON 返回，包含：
- query_type: str (aggregation/detail/statistics/trend)
- metrics: List[str]
- dimensions: List[str]
- filters: List[{{"column": str, "operator": str, "value": any}}]
- order_by: List[{{"column": str, "direction": "ASC" | "DESC"}}]
"""
        resp = await self.llm.complete(prompt)
        try:
            d = json.loads(resp)
        except Exception:
            raise ValueError(f"LLM 返回非 JSON: {resp}")
        return ParsedIntent(
            query_type=d.get("query_type", "aggregation"),
            metrics=d.get("metrics", []),
            dimensions=d.get("dimensions", []),
            filters=d.get("filters", []),
            order_by=d.get("order_by", []),
        )

    async def _identify_tables(self, parsed_intent: ParsedIntent, schema_summary: Dict[str, Any]) -> List[str]:
        prompt = f"""
根据以下意图与数据库结构，推断应使用哪些表。仅输出 JSON 数组（表名字符串列表）。

用户意图:
{json.dumps(parsed_intent.__dict__, ensure_ascii=False, indent=2)}

数据库结构摘要:
{json.dumps(schema_summary, ensure_ascii=False, indent=2)}
"""
        resp = await self.llm.complete(prompt)
        try:
            arr = json.loads(resp)
            return [t for t in arr if isinstance(t, str)]
        except Exception:
            # 兜底：若解析失败，回退为摘要中所有表
            return [t["name"] for t in schema_summary.get("tables", [])]

    async def _construct_sql(
        self,
        parsed_intent: ParsedIntent,
        tables: List[str],
        context: Optional[Dict],
        schema_summary: Dict[str, Any],
        limit: int
    ) -> str:
        prompt = f"""
        你是 MySQL 8.0 的 SQL 专家。请根据以下信息生成 **只读** 的 MySQL SQL（单条语句，可使用 CTE/窗口函数），严格遵守 MySQL 8.0 语法。

        【用户意图】
        {json.dumps(parsed_intent.__dict__, ensure_ascii=False, indent=2)}

        【待使用的表】
        {json.dumps(tables, ensure_ascii=False)}

        【上下文】
        {json.dumps(context or {}, ensure_ascii=False)}

        【数据库结构摘要】
        （只允许使用其中存在的表与列，严禁臆造）
        {json.dumps(schema_summary, ensure_ascii=False, indent=2)}

        【语法与风格要求】
        1. 仅输出 SQL，放在 ```sql 代码块中。
        2. **只读查询**：禁止 DDL/DML（INSERT/UPDATE/DELETE/CREATE/ALTER/DROP/MERGE/REPLACE）。
        3. 可以使用 CTE（WITH）、窗口函数（如 ROW_NUMBER/OVER）、JSON 函数、常见日期函数（DATE_FORMAT、STR_TO_DATE、TIMESTAMPDIFF 等）。
        4. 不要使用非 MySQL 方言函数/语法（例如：strftime/unnest/generate_series/ILIKE/:: 类型转换 等）。
        5. 不使用 SELECT *，显式列出所需字段；聚合列请使用有意义的别名。
        6. 如用户未指定限制，末尾必须包含 LIMIT {limit}。
        7. 字段或表名如有保留字冲突，使用反引号包裹（`like_this`）。

        请生成符合以上要求的 **MySQL 8.0** 查询。
        """
        resp = await self.llm.complete(prompt)
        return self._extract_sql(resp)

    def _optimize_sql(self, sql: str, limit: int) -> str:
        s = sql.strip().rstrip(";")
        if not re.match(r"(?is)^(with|select)\b", s):
            raise ValueError("生成的 SQL 不是只读查询（需要以 WITH/SELECT 开头）")
        if re.search(r"\b(insert|update|delete|drop|alter|create|merge)\b", s, re.I):
            raise ValueError("检测到危险语句（DDL/DML）")
        if "limit" not in s.lower():
            s = f"{s} LIMIT {limit}"
        return s

    def _extract_sql(self, text: str) -> str:
        m = re.search(r"```sql\s*(.*?)\s*```", text, re.S | re.I)
        return (m.group(1) if m else text).strip()


class SimpleSQLGenerator:
    """
    单表极速版：适合“上传文档→写入某张表”的场景。
    仅一次 LLM 调用：用户需求 + 单表结构 + 样例 -> SQL
    """
    def __init__(self, llm: LLM, schema_manager: MySQLSchemaManager, table: str):
        self.llm = llm
        self.schema_manager = schema_manager
        self.table = table

    async def generate_sql(self, user_intent: str, *, limit: int = 100) -> str:
        table_summary = self.schema_manager.get_table_summary(self.table)
        prompt = f"""
        你是 MySQL 8.0 的 SQL 专家。根据 **单表结构** 生成 **只读** 的 MySQL 查询语句（单条语句，可使用 CTE/窗口函数）。

        【用户需求】
        {user_intent}

        【目标表】
        {self.table}

        【表结构与样例】
        （仅可使用此表字段，严禁臆造列）
        {json.dumps(table_summary, ensure_ascii=False, indent=2)}

        【语法与风格要求】
        1. 仅输出 SQL，放在 ```sql 代码块中。
        2. **只读查询**：禁止 DDL/DML（INSERT/UPDATE/DELETE/CREATE/ALTER/DROP/MERGE/REPLACE）。
        3. 允许 CTE（WITH）、窗口函数（OVER）、常见日期与 JSON 函数。
        4. 禁止使用非 MySQL 方言（如 strftime/unnest/generate_series/ILIKE/:: 转换 等）。
        5. 不使用 SELECT *；显式列出列；聚合列加有意义的别名。
        6. 如用户未指定限制，SQL 末尾加上 LIMIT {limit}。
        7. 标识符遇到保留字时请使用反引号（`id`）。

        请生成符合以上要求的 **MySQL 8.0** 查询。
        """
        resp = self.llm.complete(prompt)
        sql = self._extract_sql(resp).strip().rstrip(";")
        if not re.match(r"(?is)^(with|select)\b", sql):
            raise ValueError("生成的 SQL 不是只读查询（需要以 WITH/SELECT 开头）")
        if "limit" not in sql.lower():
            sql = f"{sql} LIMIT {limit}"
        return sql

    def _extract_sql(self, text: str) -> str:
        m = re.search(r"```sql\s*(.*?)\s*```", text, re.S | re.I)
        return (m.group(1) if m else text).strip()


def make_sql_generator(
    llm: LLM,
    schema_manager: MySQLSchemaManager,
    *,
    table: Optional[str] = None
):
    """
    工厂函数：
      - 指定 table -> 返回 SimpleSQLGenerator（单表极速）
      - 未指定 table -> 返回 SQLGenerator（多表通用）
    """
    if table:
        return SimpleSQLGenerator(llm, schema_manager, table)
    return SQLGenerator(llm, schema_manager)
