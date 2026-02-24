"""LangGraph State — 멀티턴 보험 챗봇 상태 스키마.

필수 필드:
  messages  — Human/AI/Tool 메시지 누적 (add_messages reducer)
  trace     — 노드별 실행 로그 (append reducer)
"""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState

from pydantic import Field

# ── Trace Reset Sentinel ───────────────────────────────────────────────────────
# ainvoke 시작 시 이전 턴 trace를 초기화하는 마커 객체.
# dict 타입이지만 객체 동일성(is)으로 비교하므로 사용자 데이터와 충돌 없음.
_TRACE_RESET: dict = {"__reset__": True}


def _append_trace(current: list[dict], update: list[dict]) -> list[dict]:
    """Trace reducer — 새 ainvoke 시작 시 초기화, 턴 내에서는 누적.

    update[0] is _TRACE_RESET 이면 이번 턴만 유지 (체크포인트 trace 폐기).
    """
    if update and update[0] is _TRACE_RESET:
        return list(update[1:])
    return current + update


class AgentState(MessagesState):
    """그래프 전체 상태. 모든 노드가 이 타입의 부분집합을 입출력."""

    trace: Annotated[list[dict], _append_trace]
    guardrail_action: Literal["pass", "block", "retry"] = Field(default="pass")
    rewritten_query: str = Field(default="")
    """query_rewriter가 단문/후속 질문을 재작성한 결과. 비어있으면 원본 사용."""
    guardrail_retry_count: int = Field(default=0)
    """output_guardrail 차단 후 재시도 횟수. 무한 루프 방지용."""
    conversation_started: bool = Field(default=False)
    """output_guardrail를 통과한 실제 AI 응답이 최소 1회 이상 전송된 적 있는지 여부.

    guardrail에서 거절된 응답만 있을 때 False를 유지함으로써
    도메인 체크 우회(1턴 차단 → 2턴 followup 판정) 취약점을 방지한다.
    build_graph_input()에서 리셋하지 않으므로 checkpointer를 통해 대화 전반에 걸쳐 유지된다.
    """


def build_graph_input(query: str) -> dict:
    """그래프 호출용 입력 dict를 구성한다.

    모든 진입점(FastAPI, MCP)이 이 함수를 사용하여
    AgentState 스키마에 맞는 입력을 생성한다.
    trace에 _TRACE_RESET을 주입하여 이전 턴 trace가 누적되지 않도록 한다.
    conversation_started는 의도적으로 리셋하지 않는다 (대화 전반 유지).
    """
    return {
        "messages": [HumanMessage(content=query)],
        "trace": [_TRACE_RESET],
        "guardrail_action": "pass",
        "rewritten_query": "",
        "guardrail_retry_count": 0,
    }


def extract_last_human_query(messages: list) -> str:
    """메시지 히스토리에서 가장 최근 HumanMessage의 content를 반환."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            return msg.content
    return ""


def extract_tools_used(messages: list) -> list[str]:
    """메시지 리스트에서 사용된 도구 이름을 중복 없이 순서 보존하여 반환."""
    return list(dict.fromkeys(
        msg.name for msg in messages
        if hasattr(msg, "name") and msg.name and getattr(msg, "type", "") == "tool"
    ))
