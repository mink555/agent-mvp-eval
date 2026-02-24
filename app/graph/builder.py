"""LangGraph 그래프 빌더 — ReAct + 가드레일.

  START
    │
    ▼
  input_guardrail ──(block)──→ END
    │(pass)
    ▼
  query_rewriter
    │
    ▼
  agent ◄──────┐
    │          │
    ▼          │
  tools_condition (prebuilt)
    │       │
    │(tools) │(end → output_guardrail)
    ▼       │
  tools ────┘
            │
            ▼
  output_guardrail ──(retry)──→ agent
            │(pass/block)
            ▼
           END

노드 5개, 조건부 엣지 3개, 무조건 엣지 3개.
"""

from __future__ import annotations

import logging

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from app.graph.state import AgentState
from app.graph.nodes import agent
from app.graph.guardrails import (
    input_guardrail, output_guardrail,
    route_after_input_guard, route_after_output_guard,
)
from app.graph.query_rewrite import query_rewriter
from app.tools import get_all_tools

logger = logging.getLogger("insurance.graph.builder")

RECURSION_LIMIT = 30

# ── Checkpointer 싱글톤 ────────────────────────────────────────────────────────
# ainvoke / astream_events 는 async이므로 AsyncSqliteSaver 가 필요하다.
# aiosqlite 미설치 시 MemorySaver로 폴백 (재시작 시 대화 초기화됨).

_checkpointer = None


async def init_checkpointer() -> None:
    """FastAPI lifespan에서 호출 — AsyncSqliteSaver 비동기 초기화.

    ainvoke / astream_events 등 async 그래프 실행에는 AsyncSqliteSaver 가 필요하다.
    aiosqlite 또는 langgraph-checkpoint-sqlite 미설치 시 MemorySaver로 폴백.
    """
    global _checkpointer
    if _checkpointer is not None:
        return

    try:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from app.config import get_settings

        path = get_settings().checkpoint_db_path
        conn = await aiosqlite.connect(path, check_same_thread=False)
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        _checkpointer = saver
        logger.info("Checkpointer: AsyncSqliteSaver (%s)", path)
    except (ImportError, Exception) as e:
        from langgraph.checkpoint.memory import MemorySaver

        logger.warning(
            "AsyncSqliteSaver 초기화 실패 (%s) → MemorySaver 사용 (재시작 시 대화 초기화됨). "
            "`pip install aiosqlite langgraph-checkpoint-sqlite` 설치를 권장합니다.",
            e,
        )
        _checkpointer = MemorySaver()


def get_checkpointer():
    """체크포인터 싱글톤 반환. init_checkpointer() 호출 후 사용해야 한다."""
    return _checkpointer


async def close_checkpointer() -> None:
    """앱 종료 시 체크포인터 연결 정리."""
    global _checkpointer
    if _checkpointer is None:
        return
    conn = getattr(_checkpointer, "conn", None)
    if conn is not None:
        try:
            await conn.close()
        except Exception:
            pass
    _checkpointer = None


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    tool_node = ToolNode(get_all_tools())

    graph = StateGraph(AgentState)

    # ── 노드 등록 (5개) ──
    graph.add_node("input_guardrail", input_guardrail)
    graph.add_node("query_rewriter", query_rewriter)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_node("output_guardrail", output_guardrail)

    # ── 엣지 배선 ──
    # START → input_guardrail → (block→END | pass→query_rewriter) → agent
    graph.add_edge(START, "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail", route_after_input_guard,
        {"pass": "query_rewriter", "block": END},
    )
    graph.add_edge("query_rewriter", "agent")
    graph.add_conditional_edges(
        "agent", tools_condition,
        {"tools": "tools", "__end__": "output_guardrail"},
    )
    graph.add_edge("tools", "agent")
    # output_guardrail → (retry→agent | pass/block→END)
    graph.add_conditional_edges("output_guardrail", route_after_output_guard)

    return graph


_graph = None


def get_graph():
    """컴파일된 LangGraph 그래프 싱글톤을 반환.

    init_checkpointer() 완료 후 최초 호출 시 컴파일한다.
    module-level 변수로 관리 — checkpointer가 async 초기화 후 결정되므로 lru_cache 불가.
    """
    global _graph
    if _graph is None:
        _graph = build_graph().compile(checkpointer=get_checkpointer())
    return _graph
