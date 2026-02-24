"""RAG 문서 검색 도구 — 약관/상품요약서를 Tool 인터페이스로 제공."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from app.tools.data import _json


# ── Input Schemas ─────────────────────────────────────────────────────────────


class RagQueryInput(BaseModel):
    query: str = Field(
        ..., min_length=1, description="검색 질문 (예: 암 면책기간, 치아보험 보장개시일)"
    )
    product_code: str = Field(
        default="", description="상품 코드로 범위 한정 (빈 값이면 전체 검색)"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


_TERMS_SOURCES = {"별표", "표준약관"}
_CATALOG_KEYWORDS = {"목록", "뭐", "어떤", "전체", "종류", "리스트", "팔아", "상품"}
_CATALOG_BOOST = "판매 중인 보험상품 목록"
_COMPANY_INFO_SOURCE = "lina_info.txt"


def _is_terms_source(source: str) -> bool:
    return any(kw in source.lower() for kw in _TERMS_SOURCES)


def _is_catalog_query(query: str) -> bool:
    return sum(1 for kw in _CATALOG_KEYWORDS if kw in query) >= 2


# ── Tools ─────────────────────────────────────────────────────────────────────


@tool(args_schema=RagQueryInput)
def rag_terms_query_engine(query: str, product_code: str = "") -> str:
    """약관·규정 문서 전용 검색. 면책·예외·정의·고지의무 등 약관 원문이 필요할 때 사용.
    상품요약서·회사정보 검색은 rag_product_info_query_engine 사용."""
    from app.rag.retriever import get_rag_retriever

    retriever = get_rag_retriever()
    search_query = f"{product_code} {query}".strip() if product_code else query

    raw = retriever.retrieve(search_query, top_k=15)
    docs = [d for d in raw if _is_terms_source(d.get("metadata", {}).get("source", ""))]
    docs = docs[:5]

    return _json({
        "documents": [{"text": d["text"], "score": d["score"], "metadata": d["metadata"]} for d in docs],
        "total": len(docs),
        "query_used": search_query,
    })


@tool(args_schema=RagQueryInput)
def rag_product_info_query_engine(query: str, product_code: str = "") -> str:
    """상품요약서·회사정보 문서 전용 검색. 약관 원문 검색은 rag_terms_query_engine 사용."""
    from app.rag.retriever import get_rag_retriever

    retriever = get_rag_retriever()
    is_catalog = _is_catalog_query(query) and not product_code

    if product_code:
        search_query = f"{product_code} {query}".strip()
    elif is_catalog:
        search_query = f"{_CATALOG_BOOST} {query}"
    else:
        search_query = query

    where_filter: dict[str, Any] | None = None
    if product_code:
        where_filter = {"product_code": product_code}

    if is_catalog:
        docs = _catalog_two_pass_search(retriever, search_query)
    else:
        raw = retriever.retrieve(search_query, top_k=15, where=where_filter)
        docs = [d for d in raw if not _is_terms_source(d.get("metadata", {}).get("source", ""))]
        docs = docs[:5]

    return _json({
        "documents": [{"text": d["text"], "score": d["score"], "metadata": d["metadata"]} for d in docs],
        "total": len(docs),
        "query_used": search_query,
    })


def _catalog_two_pass_search(retriever: Any, search_query: str) -> list[dict[str, Any]]:
    priority = retriever.retrieve(search_query, top_k=5, where={"source": _COMPANY_INFO_SOURCE})
    general = retriever.retrieve(search_query, top_k=10)
    general = [d for d in general if not _is_terms_source(d.get("metadata", {}).get("source", ""))]
    seen_ids = {d["id"] for d in priority}
    for d in general:
        if d["id"] not in seen_ids:
            priority.append(d)
            seen_ids.add(d["id"])
    return priority[:8]


TOOLS = [
    rag_terms_query_engine, rag_product_info_query_engine,
]
