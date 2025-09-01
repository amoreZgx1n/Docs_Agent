from __future__ import annotations
from typing import Dict, Any, List, Optional
from sqlalchemy.engine import Engine
from sqlalchemy import text

class MySQLSchemaManager:
    """
    基于 information_schema 的 MySQL 表结构采集（按表名过滤最常用）
    - get_table_summary(table, schema=db_name)
    - get_schema_summary(tables=[...], schema=db_name)
    """
    def __init__(self, engine: Engine, sample_rows: int = 1, schema: Optional[str] = None):
        self.engine = engine
        # MySQL 里 schema 即 database 名
        self.schema = schema or engine.url.database
        self.sample_rows = int(sample_rows)

    def get_table_summary(self, table: str) -> Dict[str, Any]:
        with self.engine.connect() as conn:
            # 列信息
            cols = conn.execute(text("""
                SELECT column_name, column_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = :schema AND table_name = :table
                ORDER BY ordinal_position
            """), {"schema": self.schema, "table": table}).mappings().all()

            columns, pk_cols = [], []
            for r in cols:
                columns.append({
                    "name": r["COLUMN_NAME"],
                    "type": r["COLUMN_TYPE"],
                    "nullable": r["IS_NULLABLE"] == "YES",
                    "default": r["COLUMN_DEFAULT"]
                })

            # 主键
            pk = conn.execute(text("""
                SELECT column_name
                FROM information_schema.key_column_usage
                WHERE constraint_schema = :schema
                  AND table_name = :table
                  AND constraint_name = 'PRIMARY'
                ORDER BY ordinal_position
            """), {"schema": self.schema, "table": table}).fetchall()
            pk_cols = [x[0] for x in pk]

            # 样例行
            sample = []
            if self.sample_rows > 0:
                try:
                    rows = conn.execute(text(f"SELECT * FROM `{self.schema}`.`{table}` LIMIT :lim"),
                                        {"lim": self.sample_rows}).fetchall()
                    sample = [dict(r._mapping) for r in rows]
                except Exception:
                    pass

        return {
            "database": str(self.schema),
            "dialect": "mysql",
            "schema": str(self.schema),
            "tables": [{
                "name": table,
                "columns": columns,
                "primary_key": pk_cols,
                "foreign_keys": [],  # 可选：需要时再补
                "sample_rows": sample,
                "row_count": None
            }]
        }

    def get_schema_summary(self, tables: Optional[List[str]] = None) -> Dict[str, Any]:
        if not tables:
            with self.engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = :schema
                    ORDER BY table_name
                """), {"schema": self.schema}).fetchall()
                tables = [r[0] for r in rows]

        return {
            "database": str(self.schema),
            "dialect": "mysql",
            "schema": str(self.schema),
            "tables": [self.get_table_summary(t)["tables"][0] for t in tables]
        }
