# src/llm_generate/analyze.py
from __future__ import annotations
from typing import Optional, Dict
import asyncio
import pandas as pd

from sqlalchemy import create_engine

from src.llm_generate.llm import LLM
from src.llm_generate.code_generate import Code_Generator
from src.llm_generate.df_summary import DF_Summary
from src.llm_generate.code_executor import execute

# SQL 相关（MySQL）
from src.llm_generate.schema_manager_mysql import MySQLSchemaManager
from src.llm_generate.sql_generate import make_sql_generator   # SimpleSQLGenerator / SQLGenerator 工厂
from src.llm_generate.sql_executor import SQLExecutor
import asyncio

def _to_markdown_preview(df: pd.DataFrame, max_rows: int = 30) -> str:
    """把 DataFrame 渲染为 markdown（仅预览前 N 行），防止提示词过大"""
    try:
        df_preview = df.head(max_rows)
        return df_preview.to_markdown(index=False)
    except Exception:
        # 兜底：避免 to_markdown 依赖缺失导致异常
        return str(df.head(max_rows))


async def analyze_data(
    user_query: str,
    df: Optional[pd.DataFrame] = None,
    *,
    # —— SQL 选项（MySQL）——
    mysql_url: Optional[str] = None,        # e.g. "mysql+pymysql://user:pass@host:3306/db?charset=utf8mb4"
    table: Optional[str] = None,            # 目标表（上传文档落地表/主查询表）
    limit: int = 500,                       # SQL 默认 LIMIT
    source: str = "auto",                   # "df" | "mysql" | "auto"
    analyze_mode: str = "code"              # "code" | "passthrough"
) -> str:
    """
    统一数据分析入口：支持 DataFrame 与 MySQL SQL。
    步骤：
      1) 选择数据来源（df / mysql / auto）
      2) 对最终 DataFrame 做摘要 → 代码生成 → 执行 → 生成报告

    参数:
        user_query: 用户的数据分析问题
        df: 可选，若直接传入数据框
        mysql_url: MySQL 连接串（启用 SQL 时需要）
        table: MySQL 中的目标表名（上传文档写入的表 / 主查询表）
        limit: 生成 SQL 的默认 LIMIT
        source: 数据来源选择：'df' | 'mysql' | 'auto'
        analyze_mode:
            'code'        -> 用 Code_Generator 基于结果 df 再做二次分析/绘图
            'passthrough' -> 直接基于结果 df 生成解读报告，不再二次代码执行
    返回:
        str: markdown 格式的分析报告
    """
    # 1) 初始化模型
    llm = LLM(backend="ollama")

    # 2) 选择数据来源
    use_mysql = (source == "mysql") or (source == "auto" and (mysql_url is not None and table is not None))
    if use_mysql:
        # ---- SQL 路径：根据意图生成 SQL -> 执行 -> 得到 df_result ----
        if not mysql_url or not table:
            raise ValueError("使用 MySQL 模式时，请提供 mysql_url 和 table")

        # 2.1 连接与元信息（使用传入的 mysql_url）
        engine = create_engine(mysql_url)
        schema_mgr = MySQLSchemaManager(engine, sample_rows=2)

        # 2.2 选择单表极速版 SQL 生成器（如需多表可去掉 table 参数改为通用生成器）
        sql_gen = make_sql_generator(llm, schema_mgr, table=table)  # SimpleSQLGenerator

        # 2.3 生成 SQL（SimpleSQLGenerator 是 async 的，这里同步执行）
        sql = await sql_gen.generate_sql(user_intent=user_query, limit=limit)

        # 2.4 执行 SQL
        executor = SQLExecutor(engine, default_limit=limit)
        df_result, meta = executor.execute(sql)

        # 没数据直接给出提示
        if df_result is None or len(df_result) == 0:
            return f"**查询无结果。**\n\n**SQL**:\n```sql\n{sql}\n```\n"

        # 继续进入统一的下游流程
        working_df = df_result
        sql_block = f"\n\n**生成的 SQL**:\n```sql\n{sql}\n```\n"
    else:
        # ---- DataFrame 路径：直接使用传入的 df ----
        if df is None:
            raise ValueError("source='df' 或 'auto' 时，若不使用 MySQL，必须提供 df。")
        working_df = df
        sql_block = ""

    # 3) 对 working_df 做摘要 & 分析
    df_summary = DF_Summary(working_df).get_summary()

    if analyze_mode == "code":
        # 3.1 基于摘要生成分析代码
        code = Code_Generator(llm).generate_code(user_query, df_summary)
        # 3.2 执行生成的代码获得分析结果 DataFrame
        try:
            result_df = execute(working_df, code)
        except Exception as e:
            # 如果代码执行失败，退化为不执行代码，直接用 working_df 生成报告
            result_df = working_df
            code = f"# 代码执行失败，回退到原始数据。\n# 错误: {e}\n"
    else:
        # 直通模式，不进行二次代码生成/执行
        result_df = working_df
        code = None

    # 4) 生成最终报告
    df_md = _to_markdown_preview(result_df)
    code_block = f"\n\n**生成并执行的分析代码**:\n```python\n{code}\n```" if code else ""
    prompt = f"""
你是一个数据分析专家。请你根据用户问题，结合数据信息，进行充分分析，给出结论清晰、结构化的报告。
- 尽量用条目和小标题组织内容。
- 如有时间序列或分类对比，给出关键观察点与可能原因。
- 若存在数据质量或样本量限制，请指出。

# 用户问题
{user_query}

# 数据信息（预览）
{df_md}
"""
    analyze_result = llm.complete(prompt)

    # 5) 汇总输出（可选地包含 SQL 与代码）
    return f"{analyze_result}{sql_block}{code_block}"
