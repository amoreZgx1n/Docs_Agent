# data_analyze.py — orchestrator
import os
from typing import Any, Dict, List
from llm import LLM
from logger import logger
import pandas as pd
from mcp_chart import mcp_call_and_save,slugify
from chart_schema import CHART_SCHEMAS

TOOLS = list(CHART_SCHEMAS.keys())
FALLBACK = {k: v["required_keys"] for k, v in CHART_SCHEMAS.items()}

def _pack_schemas_for_prompt() -> str:
    # 压缩成简短上下文，尽量省 tokens
    lines = []
    for name, info in CHART_SCHEMAS.items():
        keys = ",".join(info["required_keys"])
        hint = info.get("hint", "")
        lines.append(f"- {name}: required_keys=[{keys}] | {hint}")
    return "\n".join(lines)

def choose_tool(user_query: str, cols: List[str], sample_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    llm = LLM(backend="siliconflow")
    schema_list_text = _pack_schemas_for_prompt()
    # 只给前5行样本，避免变长
    sample_rows = sample_rows[:5]

    prompt = f"""
    你是图表选择器。根据用户意图、可用列与样本数据，在下列工具及其schema中选择最合适的一种，并给出字段映射。

    可选工具与简要schema：
    {schema_list_text}

    用户意图：{user_query}
    可用列：{cols}
    数据样本(最多5行)：{sample_rows}

    要求：
    1) "tool" 必须从可选工具列表中选择。
    2) "required_keys" 必须与所选工具对应（见上文）。
    3) "field_mapping" 把 required_keys 映射到你的列名（如 {{"time":"month","value":"sales"}}）。
    4) "img_name" 结合用户意图和生成的图表，给出简短图表描述。
    5) "reason" 一句话说明选择该工具的理由。
    5) 只输出 JSON，格式如下：
    {{
    "tool": "<tool_from_list>",
    "required_keys": ["...", "..."],
    "field_mapping": {{"<required_key>": "<your_column_name>"}},
    "img_name": "<optional>",
    "reason": "..."
    }}
    """
    schema = {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": TOOLS},
            "required_keys": {"type": "array", "items": {"type": "string"}},
            "field_mapping": {"type": "object"},
            "img_name": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["tool", "required_keys", "field_mapping"],
        "additionalProperties": False,
    }

    out = llm.json(prompt, json_schema=schema) or {}
    tool = out.get("tool")
    # 兜底纠正 required_keys（防止模型瞎填）
    if tool in FALLBACK:
        out["required_keys"] = FALLBACK[tool]

    # 确保每个必需键都有映射键位（值留空，后面兜底）
    req = out.get("required_keys", [])
    mapping = out.get("field_mapping", {}) or {}
    for k in req:
        mapping.setdefault(k, "")
    out["field_mapping"] = mapping
    return out

def build_chart_params(
    df: pd.DataFrame,
    required_keys: List[str],
    field_mapping: Dict[str, str],
    title: str = ""
) -> Dict[str, Any]:
    """
    极简方式：根据字段映射生成新DataFrame，列名直接对应required_keys
    
    Args:
        df: 原始数据框
        required_keys: 图表工具需要的字段列表
        field_mapping: 字段映射 {required_key: source_column}
        title: 图表标题
        
    Returns:
        MCP图表服务所需的参数
    """
    # 1) 构建新DataFrame，列名直接使用required_keys
    new_data = {}
    for required_key in required_keys:
        source_col = field_mapping.get(required_key, "")
        if source_col and source_col in df.columns:
            new_data[required_key] = df[source_col]
        else:
            # 缺失字段填充None
            new_data[required_key] = None
    
    new_df = pd.DataFrame(new_data)
    
    # 2) 类型转换兜底
    num_like_fields = {"value", "value1", "value2", "x", "y", "lng", "lat", "size"}
    
    for col in required_keys:
        if col == "time":
            # 时间字段转换为字符串格式
            new_df[col] = pd.to_datetime(new_df[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        elif col in num_like_fields:
            # 数值字段转换为float
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
        else:
            # 其他字段转换为字符串
            new_df[col] = new_df[col].astype(str)
    
    # 3) 清理空值
    new_df = new_df.dropna()
    
    return {
        "data": new_df.to_dict(orient="records"),
        "title": title,
        "description": title,
    }

def data_analyze(user_query: str, df: pd.DataFrame) -> dict:
    choice = choose_tool(user_query, list(df.columns), df.head(5).to_dict(orient="records"))
    tool = choice["tool"]
    mapping = choice["field_mapping"]              # 直接用 LLM 给的映射
    required_keys = choice["required_keys"]        # 直接用 LLM 给的 required_keys
    img_name = (choice.get("img_name") or user_query).strip()
    img_slug = slugify(img_name)

    fields = {"required_keys": required_keys, "field_mapping": mapping}
    params = build_chart_params(df, required_keys, mapping, img_slug)

    fpath, meta = mcp_call_and_save(params=params, chart_type=tool, img_name=img_slug)

    md = (
        f"# 分析报告\n\n**问题**：{user_query}\n\n![]({fpath})\n\n"
        f"> 工具：{tool}\n\n（图表由 MCP 生成。下方为简要分析：）\n\n"
    )
    analysis = LLM().generate(
        f"请结合数据，针对用户问题给出充分分析（文字分析即可），输出 Markdown 段落。问题：{user_query}；数据：{df}"
    )
    return {"markdown": md + analysis, "meta": {"tool": tool, "mapping": mapping, **meta}}

