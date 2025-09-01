from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import re
import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy import text

class SQLExecutor:
    """
    MySQL 只读执行器
    - 仅允许 WITH/SELECT
    - 自动补 LIMIT（可关闭 ensure_limit）
    - 返回 (DataFrame, meta)
    """
    def __init__(self, engine: Engine, default_limit: int = 1000):
        self.engine = engine
        self.default_limit = int(default_limit)

    def execute(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        ensure_limit: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        q = sql.strip().rstrip(";")
        if not re.match(r"(?is)^(with|select)\b", q):
            raise ValueError("只允许执行 WITH/SELECT 查询")
        if ensure_limit and "limit" not in q.lower():
            q = f"{q} LIMIT {self.default_limit}"

        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(q), conn, params=params or {})

        return df, {
            "rows": len(df),
            "columns": list(df.columns),
            "limit_applied": ensure_limit and "limit" in q.lower(),
        }
