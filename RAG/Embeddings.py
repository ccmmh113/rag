#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os

# Suppress HF Hub unauthenticated warning when loading local models
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np


class BaseEmbeddings(ABC):
    """Base interface for all embedding providers."""

    path: str
    is_api: bool

    def __init__(self, path: str = "", is_api: bool = True) -> None:
        self.path = path
        self.is_api = is_api

    @abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        """Return the embedding vector for a single text input."""

    @classmethod
    def cosine_similarity(cls, vector1: Sequence[float], vector2: Sequence[float]) -> float:
        """Calculate cosine similarity and return 0.0 for zero vectors."""
        vec1 = np.asarray(vector1, dtype=np.float64)
        vec2 = np.asarray(vector2, dtype=np.float64)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))


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


class OpenAIEmbedding(BaseEmbeddings):
    """OpenAI-compatible embedding provider."""

    model: str
    client: Any

    def __init__(
        self,
        path: str = "",
        is_api: bool = True,
        model: str = "text-embedding-3-large",
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
    ) -> None:
        super().__init__(path=path, is_api=is_api)
        if not self.is_api:
            raise NotImplementedError("OpenAIEmbedding only supports API mode.")

        _load_dotenv_if_available()
        api_key = _required_env(api_key_env)
        base_url = os.getenv(base_url_env) or None

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Package 'openai' is required for OpenAIEmbedding.") from exc

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def get_embedding(self, text: str) -> list[float]:
        normalized_text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=[normalized_text], model=self.model)
        return list(response.data[0].embedding)


class JinaEmbedding(BaseEmbeddings):
    """Local Jina embedding provider."""

    _model: Any

    def __init__(
        self,
        path: str = "jinaai/jina-embeddings-v2-base-zh",
        is_api: bool = False,
    ) -> None:
        super().__init__(path=path, is_api=is_api)
        if self.is_api:
            raise NotImplementedError("JinaEmbedding only supports local model mode.")
        self._model = self._load_model()

    def get_embedding(self, text: str) -> list[float]:
        embedding = self._model.encode([text])[0]
        return embedding.tolist()

    def _load_model(self) -> Any:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("Packages 'torch' and 'transformers' are required for JinaEmbedding.") from exc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return AutoModel.from_pretrained(self.path, trust_remote_code=True).to(device)


class ZhipuEmbedding(BaseEmbeddings):
    """ZhipuAI embedding provider."""

    model: str
    client: Any

    def __init__(
        self,
        path: str = "",
        is_api: bool = True,
        model: str = "embedding-2",
    ) -> None:
        super().__init__(path=path, is_api=is_api)
        if not self.is_api:
            raise NotImplementedError("ZhipuEmbedding only supports API mode.")

        _load_dotenv_if_available()
        api_key = _required_env("ZHIPUAI_API_KEY")

        try:
            from zhipuai import ZhipuAI
        except ImportError as exc:
            raise RuntimeError("Package 'zhipuai' is required for ZhipuEmbedding.") from exc

        self.model = model
        self.client = ZhipuAI(api_key=api_key)

    def get_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)


class DashscopeEmbedding(BaseEmbeddings):
    """DashScope embedding provider."""

    model: str
    client: Any

    def __init__(
        self,
        path: str = "",
        is_api: bool = True,
        model: str = "text-embedding-v1",
    ) -> None:
        super().__init__(path=path, is_api=is_api)
        if not self.is_api:
            raise NotImplementedError("DashscopeEmbedding only supports API mode.")

        _load_dotenv_if_available()
        api_key = _required_env("DASHSCOPE_API_KEY")

        try:
            import dashscope
        except ImportError as exc:
            raise RuntimeError("Package 'dashscope' is required for DashscopeEmbedding.") from exc

        dashscope.api_key = api_key
        self.model = model
        self.client = dashscope.TextEmbedding

    def get_embedding(self, text: str) -> list[float]:
        response = self.client.call(model=self.model, input=text)
        if not getattr(response, "output", None):
            raise RuntimeError(f"DashScope embedding request failed: {response}")
        return list(response.output["embeddings"][0]["embedding"])


class BgeEmbedding(BaseEmbeddings):
    """Local BGE embedding provider."""

    _model: Any
    _tokenizer: Any

    def __init__(
        self,
        path: str = "BAAI/bge-base-zh-v1.5",
        is_api: bool = False,
    ) -> None:
        super().__init__(path=path, is_api=is_api)
        if self.is_api:
            raise NotImplementedError("BgeEmbedding only supports local model mode.")
        self._model, self._tokenizer = self._load_model()

    def get_embedding(self, text: str) -> list[float]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Package 'torch' is required for BgeEmbedding.") from exc

        encoded_input = self._tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        encoded_input = {key: value.to(self._model.device) for key, value in encoded_input.items()}
        with torch.no_grad():
            model_output = self._model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings[0].tolist()

    def _load_model(self) -> tuple[Any, Any]:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Packages 'torch' and 'transformers' are required for BgeEmbedding.") from exc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.path)
        model = AutoModel.from_pretrained(self.path).to(device)
        model.eval()
        return model, tokenizer


class BgeWithAPIEmbedding(OpenAIEmbedding):
    """OpenAI-compatible SiliconFlow BGE embedding provider."""

    def __init__(
        self,
        path: str = "",
        is_api: bool = True,
        model: str = "BAAI/bge-m3",
    ) -> None:
        super().__init__(
            path=path,
            is_api=is_api,
            model=model,
            api_key_env="SILICONFLOW_API_KEY",
            base_url_env="SILICONFLOW_BASE_URL",
        )


class EmbeddingFactory:
    """Factory for creating embedding providers by key."""

    _providers: ClassVar[dict[str, type[BaseEmbeddings]]] = {
        "openai": OpenAIEmbedding,
        "jina": JinaEmbedding,
        "zhipu": ZhipuEmbedding,
        "dashscope": DashscopeEmbedding,
        "bge": BgeEmbedding,
        "bge-api": BgeWithAPIEmbedding,
        "siliconflow-bge": BgeWithAPIEmbedding,
    }

    @classmethod
    def register(cls, key: str, provider: type[BaseEmbeddings]) -> None:
        if not key:
            raise ValueError("Embedding provider key must not be empty.")
        cls._providers[key.lower()] = provider

    @classmethod
    def create(cls, key: str, **kwargs: Any) -> BaseEmbeddings:
        provider = cls._providers.get(key.lower())
        if provider is None:
            available = ", ".join(sorted(cls._providers))
            raise ValueError(f"Unsupported embedding provider '{key}'. Available providers: {available}")
        return provider(**kwargs)

    @classmethod
    def available_providers(cls) -> Mapping[str, type[BaseEmbeddings]]:
        return dict(cls._providers)


def create_embedding(key: str, **kwargs: Any) -> BaseEmbeddings:
    return EmbeddingFactory.create(key, **kwargs)
