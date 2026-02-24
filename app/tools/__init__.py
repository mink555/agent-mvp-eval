"""LangChain 도구 모음 — 동적 ToolRegistry로 런타임 도구 핫리로드 지원.

새 도구 추가 시:
  1. 해당 모듈에 도구 함수 작성
  2. 모듈 하단 TOOLS 리스트에 추가
  (새 모듈이면 _TOOL_MODULES에 모듈 추가)

런타임 등록:
  registry = get_tool_registry()
  registry.register(my_new_tool)          # 단일 도구 등록 + ChromaDB 자동 인덱싱
  registry.unregister("my_new_tool")      # 이름으로 해제
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from langchain_core.tools import BaseTool

from app.tools import (
    product, premium, coverage, underwriting,
    compliance, claims, customer_db, rag_tools,
)

logger = logging.getLogger("insurance.tools.registry")

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


class ToolRegistry:
    """스레드 안전 동적 도구 레지스트리.

    서버 시작 시 기존 모듈에서 도구를 일괄 로드하고,
    런타임에 register()/unregister()로 서버 재시작 없이 도구를 추가·제거한다.
    변경 시 등록된 콜백(on_change)을 호출하여 ChromaDB 재인덱싱 등을 트리거한다.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._lock = threading.Lock()
        self._version = 0
        self._on_change_callbacks: list[Callable[["ToolRegistry"], None]] = []

    # ── 조회 ──────────────────────────────────────────────────

    def get_all(self) -> tuple[BaseTool, ...]:
        with self._lock:
            return tuple(self._tools.values())

    def get_by_name(self, name: str) -> BaseTool | None:
        with self._lock:
            return self._tools.get(name)

    @property
    def version(self) -> int:
        return self._version

    def __len__(self) -> int:
        return len(self._tools)

    # ── 등록 / 해제 ──────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        """단일 도구 등록. when_not_to_use 자동 주입 + 변경 콜백 호출."""
        enriched = _inject_when_not_to_use(tool)
        with self._lock:
            self._tools[enriched.name] = enriched
            self._version += 1
        logger.info("Registered tool: %s (v=%d)", enriched.name, self._version)
        self._fire_on_change()

    def register_many(self, tools: list[BaseTool] | tuple[BaseTool, ...]) -> None:
        """다수 도구 일괄 등록. 콜백은 마지막에 한 번만 호출."""
        with self._lock:
            for t in tools:
                self._tools[t.name] = _inject_when_not_to_use(t)
            self._version += 1
        logger.info("Registered %d tools (v=%d)", len(tools), self._version)
        self._fire_on_change()

    def unregister(self, name: str) -> bool:
        """이름으로 도구 해제. 제거 성공 시 True."""
        with self._lock:
            removed = self._tools.pop(name, None)
            if removed:
                self._version += 1
        if removed:
            logger.info("Unregistered tool: %s (v=%d)", name, self._version)
            self._fire_on_change()
        return removed is not None

    # ── 변경 콜백 ────────────────────────────────────────────

    def on_change(self, callback: Callable[["ToolRegistry"], None]) -> None:
        """도구 목록 변경 시 호출될 콜백을 등록한다."""
        self._on_change_callbacks.append(callback)

    def _fire_on_change(self) -> None:
        for cb in self._on_change_callbacks:
            try:
                cb(self)
            except Exception:
                logger.exception("on_change callback failed")

    # ── 초기 로드 ────────────────────────────────────────────

    def load_from_modules(self) -> None:
        """_TOOL_MODULES에서 도구를 일괄 수집하여 등록. 서버 시작 시 1회 호출."""
        tools: list[BaseTool] = []
        for mod in _TOOL_MODULES:
            tools.extend(getattr(mod, "TOOLS", []))
        self.register_many(tools)


# ── 싱글톤 ────────────────────────────────────────────────────

_registry: ToolRegistry | None = None
_registry_lock = threading.Lock()


def get_tool_registry() -> ToolRegistry:
    """ToolRegistry 싱글톤을 반환한다."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ToolRegistry()
    return _registry


def get_all_tools() -> tuple[BaseTool, ...]:
    """하위 호환용 — 기존 코드가 get_all_tools()를 호출하는 곳에서 동작 유지."""
    return get_tool_registry().get_all()
