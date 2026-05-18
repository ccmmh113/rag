#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import copy
import os
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping, Sequence


Message = dict[str, str]


class PromptManager:
    """Build chat messages for RAG without mutating caller-owned history."""

    def __init__(
        self,
        system_prompt: str = "你是一个严谨的问答助手。请基于给定上下文回答问题。",
        answer_language: str = "中文",
    ) -> None:
        self.system_prompt = system_prompt
        self.answer_language = answer_language

    def build_rag_prompt(
        self,
        question: str,
        context: str,
        preferences: list[str] | None = None,
        session_context: str = "",
    ) -> str:
        parts: list[str] = []
        parts.append(
            f"请严格依据给定上下文回答用户的问题，总是使用{self.answer_language}回答。\n"
            "不要使用上下文之外的知识补全事实；如果上下文没有足够证据，请直接说明“数据库中没有这个内容，不知道”。\n"
            "如果可以回答，请在关键结论后标注引用，引用格式使用上下文头部中的 [source=... chunk=...]。\n\n"
        )

        if preferences:
            parts.append("用户背景:\n")
            for p in preferences:
                parts.append(f"  - {p}\n")
            parts.append("\n")

        if session_context:
            parts.append(f"会话上下文:\n{session_context}\n\n")

        parts.append(f"问题: {question}\n\n")
        parts.append("可参考的上下文:\n```\n")
        parts.append(f"{context}\n")
        parts.append("```\n\n")
        parts.append("回答要求:\n")
        parts.append("1. 只回答上下文能支持的内容。\n")
        parts.append("2. 关键事实后尽量附上 [source=... chunk=...] 引用。\n")
        parts.append("3. 上下文证据不足时，只回答“数据库中没有这个内容，不知道”。\n")
        parts.append("回答:")
        return "".join(parts)

    def build_messages(
        self,
        history: Sequence[Mapping[str, str]] | None,
        context: str,
        question: str,
        include_system_prompt: bool = True,
        preferences: list[str] | None = None,
        session_context: str = "",
    ) -> list[Message]:
        messages: list[Message] = copy.deepcopy(list(history or []))

        if include_system_prompt and not self._has_system_message(messages):
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        messages.append({
            "role": "user",
            "content": self.build_rag_prompt(question, context, preferences, session_context),
        })
        return messages

    @staticmethod
    def _has_system_message(messages: Sequence[Mapping[str, str]]) -> bool:
        return any(message.get("role") == "system" for message in messages)


class RAGEngine:
    """Compose RAG prompts and delegate API communication to an LLM instance."""

    def __init__(self, llm: "BaseModel", prompt_manager: PromptManager | None = None) -> None:
        self.llm = llm
        self.prompt_manager = prompt_manager or PromptManager()

    def answer(
        self,
        question: str,
        history: Sequence[Mapping[str, str]] | None = None,
        context: str = "",
        preferences: list[str] | None = None,
        session_context: str = "",
    ) -> str:
        messages = self.prompt_manager.build_messages(
            history=history,
            context=context,
            question=question,
            preferences=preferences,
            session_context=session_context,
        )
        return self.llm.chat(messages)


class BaseModel(ABC):
    """Base interface for chat model providers."""

    path: str
    model: str

    def __init__(self, path: str = "", model: str = "") -> None:
        self.path = path
        self.model = model
        self.cached_prompt_tokens: int = 0
        self.total_prompt_tokens: int = 0

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """Return the model response for OpenAI-style chat messages."""


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return

    load_dotenv(find_dotenv())


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _copy_messages(messages: list[dict]) -> list[dict]:
    return copy.deepcopy(messages)


def _extract_message_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


class OpenAIChat(BaseModel):
    """OpenAI-compatible chat provider."""

    client: Any
    temperature: float
    max_tokens: int | None

    def __init__(
        self,
        path: str = "",
        model: str = "gpt-5.4",
        temperature: float = 0.1,
        max_tokens: int | None = 512,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
    ) -> None:
        super().__init__(path=path, model=model)
        _load_dotenv_if_available()
        api_key = _required_env(api_key_env)
        base_url = os.getenv(base_url_env) or None

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Package 'openai' is required for OpenAIChat.") from exc

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict]) -> str:
        request_messages = _copy_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.total_prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                self.cached_prompt_tokens = getattr(details, "cached_tokens", 0) or 0
        return _extract_message_content(response.choices[0].message.content)


class SiliconflowChat(OpenAIChat):
    """SiliconFlow OpenAI-compatible chat provider."""

    def __init__(
        self,
        path: str = "",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.1,
        max_tokens: int | None = 512,
    ) -> None:
        super().__init__(
            path=path,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key_env="SILICONFLOW_API_KEY",
            base_url_env="SILICONFLOW_BASE_URL",
        )


class DashscopeChat(BaseModel):
    """DashScope chat provider."""

    temperature: float
    max_tokens: int | None

    def __init__(
        self,
        path: str = "",
        model: str = "qwen-max",
        temperature: float = 0.1,
        max_tokens: int | None = 2048,
    ) -> None:
        super().__init__(path=path, model=model)
        _load_dotenv_if_available()
        self.api_key = _required_env("DASHSCOPE_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict]) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Package 'dashscope' is required for DashscopeChat.") from exc


        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _copy_messages(messages),
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        response = client.chat.completions.create(**kwargs)
        if not getattr(response, "output", None):
            raise RuntimeError(f"DashScope chat request failed: {response}")
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.total_prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        return _extract_message_content(response.output.choices[0].message.content)


class ZhipuChat(BaseModel):
    """ZhipuAI chat provider."""

    client: Any
    temperature: float
    max_tokens: int | None

    def __init__(
        self,
        path: str = "",
        model: str = "glm-4",
        temperature: float = 0.1,
        max_tokens: int | None = 512,
    ) -> None:
        super().__init__(path=path, model=model)
        _load_dotenv_if_available()
        api_key = _required_env("ZHIPUAI_API_KEY")

        try:
            from zai import ZhipuAiClient
        except ImportError as exc:
            raise RuntimeError("Package 'zai' is required for ZhipuChat.") from exc

        self.client = ZhipuAiClient(api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict]) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _copy_messages(messages),
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.total_prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        return _extract_message_content(response.choices[0].message.content)


class LLMFactory:
    """Factory for creating chat providers by key."""

    _providers: ClassVar[dict[str, type[BaseModel]]] = {
        "openai": OpenAIChat,
        "dashscope": DashscopeChat,
        "zhipu": ZhipuChat,
        "siliconflow": SiliconflowChat,
    }

    @classmethod
    def register(cls, key: str, provider: type[BaseModel]) -> None:
        if not key:
            raise ValueError("LLM provider key must not be empty.")
        cls._providers[key.lower()] = provider

    @classmethod
    def create(cls, key: str, **kwargs: Any) -> BaseModel:
        provider = cls._providers.get(key.lower())
        if provider is None:
            available = ", ".join(sorted(cls._providers))
            raise ValueError(f"Unsupported LLM provider '{key}'. Available providers: {available}")
        return provider(**kwargs)

    @classmethod
    def available_providers(cls) -> Mapping[str, type[BaseModel]]:
        return dict(cls._providers)


def create_llm(key: str, **kwargs: Any) -> BaseModel:
    return LLMFactory.create(key, **kwargs)
