
from typing import Dict, Any, List
from src.llm_generate.llm import LLM


DOC_QA_TOOL_SCHEMA = {
    "name": "doc_qa",
    "description": "基于提供的上下文片段进行文档问答，返回 Markdown 答案（含引用）。",
    "parameters": {
        "type": "object",
        "properties": {
            "user_query": {"type": "string"},
            "snippets": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "text": {"type": "string"},
                    "source": {"type": "string"}
                }, "required": ["text"]}
            }
        },
        "required": ["user_query", "snippets"]
    }
}


def doc_qa(user_query: str, snippets: List[Dict[str, str]]) -> Dict[str, Any]:
    llm = LLM()
    context = "\n\n".join(
        [f"[片段{i+1}] 来源：{s.get('source','-')}\n{s['text']}" for i, s in enumerate(snippets)]
    )
    prompt = f"""
    你是严谨的文档助理。仅基于给定片段回答问题，使用 Markdown。引用格式：在答案末尾列出使用到的片段编号及来源。

    用户问题：{user_query}

    可用上下文片段：
    {context}
    """
    answer = llm.generate(prompt)
    used = ", ".join([str(i+1) for i in range(len(snippets))])
    ref = "\n\n---\n**引用：** " + "; ".join([
        f"[{i+1}] {s.get('source','-')}" for i, s in enumerate(snippets)
    ])
    return answer + ref

