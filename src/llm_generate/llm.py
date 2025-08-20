
from __future__ import annotations
import os
import time
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, ValidationError

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)
from llama_index.core.llms import ChatResponse, CompletionResponse
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import LLM as LIBaseLLM
from llama_index.llms.zhipuai import ZhipuAI
from llama_index.llms.ollama import Ollama 
import json
from pydantic import create_model


# ─────────────────────────────────────────────────────────────────────────────
# 小型类型与协议
# ─────────────────────────────────────────────────────────────────────────────

TokenCallback = Callable[[str], None]
ReasoningCallback = Callable[[str], None]


class ReasoningConfig(BaseModel):
    enabled: bool = False
    # 对支持 reasoning 的模型（如 OpenAI o3/o4-mini 等）生效
    effort: str = "low"  # "low"|"medium"|"high"
    # 是否把模型返回的 reasoning 以“摘要”形式通过回调暴露
    expose_summary: bool = True


class InferenceOptions(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: Optional[int] = 2048
    # 额外后端参数（透传给 LlamaIndex LLM）
    extra: Dict[str, Any] = {}


# ─────────────────────────────────────────────────────────────────────────────
# 后端工厂：根据环境变量与参数创建 LlamaIndex LLM 实例
# ─────────────────────────────────────────────────────────────────────────────

class _LLMFactory:
    @staticmethod
    def build(
        backend: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        reasoning: Optional[ReasoningConfig] = None,
        options: Optional[InferenceOptions] = None,
    ) -> LIBaseLLM:
        backend = (backend or os.getenv("LLM_BACKEND") or "ollama").lower()
        options = options or InferenceOptions()
        reasoning = reasoning or ReasoningConfig()

        if backend == "ollama":
            base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
            # Ollama 的额外参数透传到 "request_timeout"/"additional_kwargs" 等
            return Ollama(
                model=model,
                base_url=base_url,
                request_timeout=600,
                temperature=options.temperature,
                max_tokens=options.max_tokens,
                thinking=True,
                additional_kwargs={
                    "top_p": options.top_p,
                    **(options.extra or {}),
                },
            )
        elif backend == "zhipu":
            # 独立的环境变量，避免误用 OPENAI_API_KEY
            api_key = api_key or os.getenv("ZHIPU_API_KEY")
            if not api_key:
                raise ValueError("ZHIPU_API_KEY 未设置")
            base_url = base_url or os.getenv("ZHIPU_BASE_URL")
            model = model or os.getenv("ZHIPU_MODEL")

            return ZhipuAI(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=(options or InferenceOptions()).temperature,
                max_tokens=options.max_tokens,
                additional_kwargs={**((options or InferenceOptions()).extra or {})},
            )


# ─────────────────────────────────────────────────────────────────────────────
# 核心 LLM 门面：统一流式/非流式、思考开关、结构化输出
# ─────────────────────────────────────────────────────────────────────────────

class LLM:
    """
    LLM 类是对各种大语言模型后端（如 Ollama、ZhipuAI 等）的统一封装。
    提供了文本补全、对话、流式输出、结构化 JSON 输出等功能。
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        reasoning: Optional[ReasoningConfig] = None,
        options: Optional[InferenceOptions] = None,
    ) -> None:
        """
        初始化 LLM 实例。

        :param backend: 后端类型，如 "ollama" 或 "zhipu"。如果未提供，将从环境变量 LLM_BACKEND 获取，默认为 "ollama"。
        :param model: 使用的模型名称。如果未提供，将根据后端从环境变量获取默认值。
        :param base_url: 后端服务的基础 URL。如果未提供，将根据后端从环境变量获取默认值。
        :param api_key: API 密钥。如果未提供，将根据后端从环境变量获取默认值。
        :param reasoning: 推理配置对象，控制是否启用推理模式及其相关参数。
        :param options: 推理选项对象，包含 temperature、top_p、max_tokens 等参数。
        """
        self.reasoning = reasoning or ReasoningConfig()
        self.options = options or InferenceOptions()
        self._llm = _LLMFactory.build(
            backend=backend,
            model=model,
            base_url=base_url,
            api_key=api_key,
            reasoning=self.reasoning,
            options=self.options,
        )

    # ───────────── 基础：补全/聊天（非流式） ─────────────

    def complete(self, prompt: str) -> str:
        """
        使用模型完成给定的提示（prompt），返回完整的文本结果。

        :param prompt: 输入的文本提示。
        :return: 模型生成的完整文本。
        """
        resp: CompletionResponse = self._llm.complete(prompt)
        # 1) 常规字段
        text = (
            getattr(resp, "text", None)
            or getattr(resp, "output_text", None)
        )
        if text:
            return text

        # 2) 兜底：从 raw 中取 content 或 reasoning_content
        raw = getattr(resp, "raw", None)
        if raw and getattr(raw, "choices", None):
            try:
                msg = raw.choices[0].message
                return (getattr(msg, "content", "") or getattr(msg, "reasoning_content", "") or "").strip()
            except Exception:
                pass

        return ""

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        与模型进行多轮对话，返回最终回复。

        :param messages: 对话历史消息列表，每个元素是一个包含 'role' 和 'content' 键的字典。
        :return: 模型的回复文本。
        """
        li_msgs = [ChatMessage(role=MessageRole(m["role"]), content=m["content"]) for m in messages]
        resp: ChatResponse = self._llm.chat(li_msgs)
        return resp.message.content or ""

    # ───────────── 基础：补全/聊天（流式） ─────────────

    def stream_complete(
        self,
        prompt: str,
        on_token: Optional[TokenCallback] = None,
        on_reasoning: Optional[ReasoningCallback] = None,
    ) -> Iterator[str]:
        """
        流式地完成给定的提示（prompt），逐个返回生成的文本片段。

        :param prompt: 输入的文本提示。
        :param on_token: 可选回调函数，当有新的文本片段生成时调用。
        :param on_reasoning: 可选回调函数，当模型生成推理摘要时调用。
        :return: 生成文本片段的迭代器。
        """
        stream = self._llm.stream_complete(prompt)
        # 一边 yield token，一边回调
        for t in stream:
            text = t.delta or ""
            if text:
                if on_token:
                    on_token(text)
                yield text
        # 如果支持 reasoning 且需要暴露摘要（通过 extra 字段拿不到则忽略）
        if self.reasoning.enabled and self.reasoning.expose_summary and on_reasoning:
            summary = getattr(stream, "reasoning", None)
            if isinstance(summary, str) and summary.strip():
                on_reasoning(summary)

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        on_token: Optional[TokenCallback] = None,
        on_reasoning: Optional[ReasoningCallback] = None,
    ) -> Iterator[str]:
        """
        流式地与模型进行多轮对话，逐个返回生成的回复片段。

        :param messages: 对话历史消息列表。
        :param on_token: 可选回调函数，当有新的文本片段生成时调用。
        :param on_reasoning: 可选回调函数，当模型生成推理摘要时调用。
        :return: 生成回复片段的迭代器。
        """
        li_msgs = [ChatMessage(role=MessageRole(m["role"]), content=m["content"]) for m in messages]
        stream = self._llm.stream_chat(li_msgs)
        for ev in stream:
            piece = ev.delta or ""
            if piece:
                if on_token:
                    on_token(piece)
                yield piece
        if self.reasoning.enabled and self.reasoning.expose_summary and on_reasoning:
            summary = getattr(stream, "reasoning", None)
            if isinstance(summary, str) and summary.strip():
                on_reasoning(summary)

    def json(self, prompt: str, json_schema: Optional[Dict[str, Any]] = None, retries: int = 2) -> Dict[str, Any]:
        """
        让模型只输出符合指定 JSON Schema 的 JSON 对象。

        :param prompt: 输入的文本提示。
        :param json_schema: 可选的 JSON Schema，用于约束输出格式。
        :param retries: 失败重试次数。
        :return: 符合要求的 JSON 对象。
        """
        if json_schema:
            # 根据 JSON Schema 动态构建一个 Pydantic 模型
            fields = {}
            for key, prop in json_schema.get("properties", {}).items():
                typ = str if prop.get("type") == "string" else float if prop.get("type") == "number" else Any
                fields[key] = (typ, ...)
            DynamicModel = create_model("DynamicModel", **fields)  # noqa: F821
            parser = PydanticOutputParser(DynamicModel)
            schema_str = json.dumps(DynamicModel.model_json_schema(), ensure_ascii=False, indent=2)
        else:
            # 如果没提供 schema，就宽松解析为 dict[str, Any]
            DynamicModel = create_model("DynamicModel", __root__=(Dict[str, Any], ...))
            parser = PydanticOutputParser(DynamicModel)
            schema_str = "{}"

        tmpl = PromptTemplate(
            """
            你将根据用户指令，*只* 输出满足下述 JSON Schema 的 JSON 对象：

            {schema}

            用户指令：
            {inst}
            """.strip()
        )
        guarded = tmpl.format(schema=schema_str, inst=prompt)

        last_err = None
        for _ in range(max(1, retries + 1)):
            text = self.complete(guarded)
            try:
                return parser.parse(text).model_dump()
            except Exception as e:
                last_err = e
                time.sleep(0.2)
        return {}

    def structured(
        self,
        prompt: str,
        model: Type[BaseModel],
        retries: int = 2,
    ) -> BaseModel:
        """
        使用 Pydantic 模型约束结构化输出。

        :param prompt: 输入的文本提示。
        :param model: Pydantic 模型类，用于约束输出格式。
        :param retries: 失败重试次数。
        :return: 符合要求的 Pydantic 模型实例。
        :raises: ValidationError 或其他异常，如果多次尝试后仍失败。
        """
        parser = PydanticOutputParser(model)
        tmpl = PromptTemplate(
            """
            你将根据用户指令，*只* 输出满足下述 JSON Schema 的 JSON 对象：\n
            {schema}

            用户指令：\n{inst}
            """.strip()
        )
        schema_json = model.model_json_schema()
        schema_str = json.dumps(schema_json, ensure_ascii=False, indent=2)

        guarded = tmpl.format(schema=schema_str, inst=prompt)

        last_err: Optional[Exception] = None
        for _ in range(max(1, retries + 1)):
            text = self.complete(guarded)
            try:
                return parser.parse(text)
            except (ValidationError, Exception) as e:
                last_err = e
                time.sleep(0.2)
        # 如果还失败，抛出最后错误
        if last_err:
            raise last_err
        raise RuntimeError("结构化解析失败")

    # ───────────── 便捷包装：兼容你现有接口 ─────────────

    def generate(self, prompt: str) -> str:
        """
        generate 方法是 complete 方法的别名，用于文本补全。

        :param prompt: 输入的文本提示。
        :return: 模型生成的完整文本。
        """
        return self.complete(prompt)

    def stream_to_callback(
        self,
        prompt: str,
        on_delta: TokenCallback,
        on_reasoning: Optional[ReasoningCallback] = None,
    ) -> None:
        """
        流式生成文本，并将每个片段通过回调函数处理。

        :param prompt: 输入的文本提示。
        :param on_delta: 处理每个文本片段的回调函数。
        :param on_reasoning: 可选回调函数，处理模型的推理摘要。
        """
        for _ in self.stream_complete(prompt, on_token=on_delta, on_reasoning=on_reasoning):
            pass

# ─────────────────────────────────────────────────────────────────────────────
# 使用示例（可删除）
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pydantic import BaseModel

    class DemoSchema(BaseModel):
        title: str
        score: float

    llm = LLM(
        backend=os.getenv("LLM_BACKEND", "ollama"),
        reasoning=ReasoningConfig(enabled=bool(os.getenv("REASONING", "0") == "1")),
    )

    print("\n=== 非流式 ===")
    print(llm.generate("用10个字以内描述早晨的天空"))

    print("\n=== 流式 ===")
    for tok in llm.stream_complete("给我一句励志短句"):
        print(tok, end="", flush=True)
    print()

    print("\n=== JSON ===")
    print(llm.json("生成一个标题和评分", json_schema={
        "type": "object",
        "properties": {"title": {"type": "string"}, "score": {"type": "number"}},
        "required": ["title", "score"],
        "additionalProperties": False,
    }))

    print("\n=== Pydantic ===")
    print(llm.structured("生成标题与评分", DemoSchema))
