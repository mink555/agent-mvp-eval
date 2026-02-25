"""Query Rewriter 노드 — 단문·후속 질문을 대화 맥락 기반으로 재작성.

LangGraph 정석 패턴: 짧거나 문맥 의존적인 쿼리를 agent에 도달하기 전에
명확한 독립 질문으로 변환하여 ChromaDB 도구 라우팅 정확도를 높인다.

참고: LangChain OpenTutorial - LangGraph-Add-Query-Rewrite
      (https://langchain-opentutorial.gitbook.io/langchain-opentutorial/
       17-langgraph/02-structures/05-langgraph-add-query-rewrite)

재작성 조건 (둘 다 충족해야 함):
  1. 쿼리 길이 < 15자  (단문/후속 질문 신호)
  2. 이전 대화 히스토리 존재  (맥락이 있을 때만 재작성이 의미 있음)

조건 미충족 시 즉시 패스 → 추가 LLM 비용 없음.
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph.state import AgentState, extract_last_human_query
from app.llm import get_llm
from app.retry import llm_retry

logger = logging.getLogger("insurance.graph.query_rewrite")

_REWRITE_SYSTEM = (
    "당신은 보험 상담 챗봇의 질문 명확화 도우미입니다.\n"
    "사용자의 짧거나 문맥 의존적인 후속 질문을, 이전 대화를 참고하여 "
    "완전하고 독립적인 질문으로 재작성하세요.\n"
    "규칙:\n"
    "- 재작성된 질문 한 줄만 출력하세요.\n"
    "- 설명·따옴표·번호는 포함하지 마세요.\n"
    "- 원래 의도를 바꾸지 마세요.\n"
    "- 재작성이 불필요하면 원문 그대로 출력하세요.\n"
    "- 챗봇이 추가 정보(성별, 나이, 상품명 등)를 물었고 사용자가 단답으로 "
    "응답한 경우, 그 정보를 이전 요청에 합쳐서 완전한 질문으로 만드세요.\n"
    "  예: 챗봇이 '성별을 알려주세요' → 사용자 '아버지' → '70세 남성 기준 "
    "두 상품 합산 보험료를 알려줘'\n"
    "  예: 챗봇이 '어떤 상품인가요?' → 사용자 '종신보험' → '종신보험 상품 "
    "정보를 알려줘'"
)

_REWRITE_THRESHOLD = 15  # 이 글자 수 미만일 때만 재작성 시도


@llm_retry
def _invoke_rewriter(llm, prompt: list):
    """Query rewrite LLM 호출 — RateLimit/Timeout 자동 재시도."""
    return llm.invoke(prompt)


def query_rewriter(state: AgentState) -> dict:
    """단문·후속 질문을 대화 맥락 기반으로 재작성하는 LangGraph 노드.

    rewritten_query 필드에 결과를 저장한다.
    _select_relevant_tools는 이 필드를 ChromaDB 검색 쿼리로 우선 사용한다.
    원본 HumanMessage는 그대로 유지되어 LLM 히스토리에 남는다.
    """
    ts = time.time()
    query = extract_last_human_query(state["messages"])

    prior = [
        m for m in state["messages"][:-1]
        if isinstance(m, (HumanMessage, AIMessage))
    ]

    stripped = query.strip()
    if len(stripped) >= _REWRITE_THRESHOLD or not prior:
        return {
            "trace": [{
                "node": "query_rewriter", "action": "skip",
                "reason": "long query or no history",
                "duration_ms": round((time.time() - ts) * 1000),
            }],
        }

    _MEANINGFUL_SINGLE = {"네", "예", "응", "M", "F", "남", "여"}
    if len(stripped) <= 1 and stripped not in _MEANINGFUL_SINGLE:
        logger.info("Too short input (%r), treating as meaningless", stripped)
        return {
            "rewritten_query": stripped,
            "trace": [{
                "node": "query_rewriter", "action": "skip",
                "reason": f"too_short ({len(stripped)} chars)",
                "duration_ms": round((time.time() - ts) * 1000),
            }],
        }

    context_msgs = prior[-4:]  # 최근 2턴
    llm = get_llm()

    prompt = [
        SystemMessage(content=_REWRITE_SYSTEM),
        *context_msgs,
        HumanMessage(
            content=(
                f"위 대화를 참고하여 이 후속 질문을 완전한 독립 질문으로 재작성: 「{query}」"
            )
        ),
    ]

    try:
        response = _invoke_rewriter(llm, prompt)
        rewritten = response.content.strip().strip('"').strip("'").strip("「」")
        if rewritten and rewritten != query:
            logger.info("Query rewritten: %r → %r", query, rewritten)
            return {
                "rewritten_query": rewritten,
                "trace": [{
                    "node": "query_rewriter", "action": "rewrite",
                    "original": query, "rewritten": rewritten,
                    "duration_ms": round((time.time() - ts) * 1000),
                }],
            }
    except Exception as exc:
        logger.warning("Query rewrite failed: %s", exc)

    return {
        "trace": [{
            "node": "query_rewriter", "action": "skip",
            "reason": "rewrite not needed or failed",
            "duration_ms": round((time.time() - ts) * 1000),
        }],
    }
