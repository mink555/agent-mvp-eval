"""LangGraph 노드 — ReAct Agent (동적 도구 레지스트리 연동).

Node:
  agent — LLM 호출, 다음 행동 결정 (tool_calls / 최종 답변)

Tool Node는 builder.py의 _dynamic_tool_node가 ToolRegistry에서 실시간 조회.
분기 조건은 langgraph.prebuilt.tools_condition을 사용.
"""

from __future__ import annotations

import logging
import time
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import get_settings
from app.graph.state import AgentState, extract_last_human_query
from app.llm import get_llm
from app.retry import llm_retry
from app.tool_search.embedder import get_tool_search
from app.tools import get_tool_registry
from app.tools.data import SYSTEM_PROMPTS

logger = logging.getLogger("insurance.graph.nodes")

SYSTEM_PROMPT = SystemMessage(content=SYSTEM_PROMPTS["answer"])


# ═══════════════════════════════════════════════════════════════════════════════
# Message Sanitization — 프로바이더 호환성 방어
# ═══════════════════════════════════════════════════════════════════════════════

def _rebuild_clean_ai_message(msg: AIMessage) -> AIMessage:
    """name이 없는 tool_call / invalid_tool_calls를 제거한 AIMessage를 재구성."""
    good = [tc for tc in (getattr(msg, "tool_calls", None) or []) if tc.get("name")]
    raw_kwargs = dict(getattr(msg, "additional_kwargs", {}))
    if "tool_calls" in raw_kwargs:
        raw_kwargs["tool_calls"] = [
            rc for rc in raw_kwargs["tool_calls"]
            if rc.get("function", {}).get("name")
        ]
    return AIMessage(
        content=msg.content or "",
        tool_calls=good,
        additional_kwargs=raw_kwargs,
    )


def _sanitize_history(messages: list) -> list:
    """LLM에 보내기 전 메시지 히스토리 정제.

    - name이 빈 tool_call을 AIMessage에서 제거
    - additional_kwargs의 raw tool_calls도 정리
    - orphan ToolMessage 제거
    """
    valid_tc_ids: set[str] = set()
    for msg in messages:
        for tc in getattr(msg, "tool_calls", []):
            if tc.get("name"):
                valid_tc_ids.add(tc.get("id"))

    cleaned: list = []
    for msg in messages:
        tc_id = getattr(msg, "tool_call_id", None)
        if tc_id is not None and tc_id not in valid_tc_ids:
            continue

        calls = getattr(msg, "tool_calls", None)
        invalid = getattr(msg, "invalid_tool_calls", None)
        needs_rebuild = (
            (calls and any(not tc.get("name") for tc in calls))
            or invalid
        )

        if needs_rebuild and calls is not None:
            msg = _rebuild_clean_ai_message(msg)
            if not msg.tool_calls and not msg.content:
                continue

        cleaned.append(msg)
    return cleaned


def _trim_history(messages: list) -> list:
    """max_conversation_turns 설정에 따라 오래된 메시지를 잘라낸다.

    HumanMessage 기준으로 실제 대화 턴을 계산하므로,
    ReAct 루프의 중간 ToolMessage가 많아도 정확한 턴 수를 유지한다.
    """
    max_turns = get_settings().max_conversation_turns
    human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]

    if len(human_indices) <= max_turns:
        return messages

    cutoff = human_indices[-max_turns]
    return messages[cutoff:]


def _sanitize_response(response: AIMessage) -> AIMessage:
    """LLM 응답에서 name 없는 tool_call을 즉시 제거."""
    calls = getattr(response, "tool_calls", None)
    invalid = getattr(response, "invalid_tool_calls", None)
    if not calls and not invalid:
        return response

    needs_fix = (
        (calls and any(not tc.get("name") for tc in calls))
        or invalid
    )
    if not needs_fix:
        return response

    return _rebuild_clean_ai_message(response)


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Node
# ═══════════════════════════════════════════════════════════════════════════════

@llm_retry
def _invoke_llm(llm, messages):
    """LLM 호출 + tenacity 자동 재시도 (RateLimit·Timeout·Connection)."""
    return llm.invoke(messages)


def agent(state: AgentState) -> dict:
    """Agent 노드 — LLM이 다음 행동을 자율 결정한다.

    ToolRegistry에서 최신 도구 목록을 가져온 뒤
    ChromaDB로 관련 도구를 필터링하고 LLM에 바인딩.
    """
    ts = time.time()
    llm = get_llm()
    registry = get_tool_registry()
    all_tools = registry.get_all()
    relevant_tools = _select_relevant_tools(state, all_tools)
    llm_with_tools = llm.bind_tools(relevant_tools)

    history = _trim_history(state["messages"])
    history = _sanitize_history(history)
    messages = [SYSTEM_PROMPT] + history

    try:
        response = _invoke_llm(llm_with_tools, messages)
    except Exception as e:
        if "tool_calls" in str(e) and "function.name" in str(e):
            logger.warning("Tool-call format error, retrying without tools: %s", e)
            response = _invoke_llm(llm, messages)
        else:
            raise

    response = _sanitize_response(response)

    return {
        "messages": [response],
        "trace": [{
            "node": "agent",
            "duration_ms": round((time.time() - ts) * 1000),
            "tools_bound": len(relevant_tools),
        }],
    }


def _select_relevant_tools(state: AgentState, all_tools: tuple) -> list:
    """ChromaDB Routing Index로 관련 도구만 필터링.

    query_rewriter가 재작성한 쿼리(rewritten_query)를 우선 사용한다.
    원본 쿼리가 단문/후속 질문일 때 ChromaDB 검색 정확도를 높인다.
    ToolRegistry에서 받은 최신 도구 목록을 기준으로 필터링한다.
    """
    try:
        query = state.get("rewritten_query") or extract_last_human_query(state["messages"])
        if not query:
            return list(all_tools)
        searcher = get_tool_search()
        candidates = searcher.search(query, top_k=get_settings().tool_search_top_k)

        if not candidates:
            return list(all_tools)

        names = {c.name for c in candidates}
        filtered = [t for t in all_tools if t.name in names]
        return filtered if filtered else list(all_tools)
    except Exception:
        logger.debug("ChromaDB tool search unavailable, using all tools")
        return list(all_tools)

