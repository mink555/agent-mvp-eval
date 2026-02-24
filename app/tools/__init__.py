"""LangChain 도구 모음 — 모듈별 TOOLS 리스트에서 자동 수집.

새 도구 추가 시:
  1. 해당 모듈에 도구 함수 작성
  2. 모듈 하단 TOOLS 리스트에 추가
  (새 모듈이면 _TOOL_MODULES에 모듈 추가)
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.tools import BaseTool

from app.tools import (
    product, premium, coverage, underwriting,
    compliance, claims, customer_db, rag_tools,
)

_TOOL_MODULES = [
    product, premium, coverage, underwriting,
    compliance, claims, customer_db, rag_tools,
]


def _inject_when_not_to_use(tool: BaseTool) -> BaseTool:
    """ToolCard의 when_not_to_use를 LLM이 보는 tool description에 주입한다.

    LLM이 bind_tools()로 도구 목록을 받을 때 description 전체가 전달된다.
    혼동하기 쉬운 유사 도구 쌍(예: premium_estimate vs plan_options)을
    명시적으로 알려줌으로써 잘못된 도구 선택을 줄인다.

    when_not_to_use는 ChromaDB 임베딩 텍스트(to_embed_text)에는 포함되지 않는다.
    (타 도구 어휘가 포함돼 임베딩 벡터를 오염시키기 때문)
    """
    from app.tool_search.tool_cards import get_card

    card = get_card(tool.name)
    if not card or not card.when_not_to_use:
        return tool
    neg = "\n[사용 금지 상황]\n" + "\n".join(f"· {w}" for w in card.when_not_to_use)
    return tool.model_copy(update={"description": tool.description.rstrip() + neg})


@lru_cache()
def get_all_tools() -> tuple[BaseTool, ...]:
    """등록된 전체 도구 목록을 반환한다.

    각 tool 모듈의 TOOLS 리스트에서 자동 수집하며,
    ToolCard의 when_not_to_use를 description에 주입하여
    LLM이 유사 도구 간 혼동 없이 올바른 도구를 선택하도록 돕는다.
    """
    tools: list[BaseTool] = []
    for mod in _TOOL_MODULES:
        tools.extend(getattr(mod, "TOOLS", []))
    return tuple(_inject_when_not_to_use(t) for t in tools)
