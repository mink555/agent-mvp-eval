"""LangChain LLM — OpenRouter 호환 ChatOpenAI 인스턴스."""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import get_settings


@lru_cache()
def get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.openrouter_model,
        api_key=s.openrouter_api_key,
        base_url=s.openrouter_base_url,
        temperature=0.2,
        max_tokens=2048,
    )
