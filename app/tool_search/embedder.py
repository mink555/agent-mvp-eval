"""Tool Search — ChromaDB 멀티-벡터 임베딩 기반 tool top-k 후보 축소.

각 도구를 하나의 벡터로 인덱싱하는 대신 **목적문(purpose) + 사용 예시(when_to_use)
각각을 별도 문서**로 인덱싱한다. 검색 시 tool별 max score 로 집계하므로
단일 when_to_use 예시가 쿼리와 잘 맞기만 해도 해당 tool이 상위에 랭크된다.

Multi-vector 방식의 이점:
  - 여러 예시 문장을 평균 임베딩할 때 발생하는 벡터 희석(dilution) 제거
  - 쿼리와 정확히 일치하는 예시 문장이 있으면 즉시 high score
  - 새 예시 추가만으로 recall 개선 가능 (코드 변경 불필요)

임베딩 텍스트 우선순위:
  1. tool_cards.REGISTRY 에 ToolCard 가 있으면 purpose + when_to_use 개별 문서 생성
  2. 카드가 없으면 tool.description 단일 문서 (하위 호환)

새 tool 추가 시 app/tool_search/tool_cards.py 에 ToolCard 를 등록하세요.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import defaultdict
from functools import lru_cache

from langchain_core.tools import BaseTool

from app.config import get_settings, get_embedding_function, get_chromadb_client
from app.retry import db_retry
from app.tool_search.tool_cards import get_card, missing_cards

logger = logging.getLogger("insurance.tool_search")

# 인덱싱 방식(단일 벡터 vs 멀티 벡터)이 바뀌면 이 값을 올려서 강제 재인덱싱한다.
_INDEX_SCHEMA_VERSION = "mv1"


class ToolCandidate:
    __slots__ = ("name", "score", "description")

    def __init__(self, name: str, score: float, description: str = ""):
        self.name = name
        self.score = score
        self.description = description


def _tool_documents(tool: BaseTool) -> list[tuple[str, str]]:
    """도구 하나에 대한 (doc_id, text) 쌍 목록을 반환한다.

    ToolCard 가 있으면 purpose + when_to_use 각각을 별도 문서로 생성.
    카드가 없으면 tool.description 단일 문서.
    """
    card = get_card(tool.name)
    if not card:
        return [(f"tool_{tool.name}", tool.description)]

    docs: list[tuple[str, str]] = []
    # purpose 문서 (필수)
    docs.append((f"tool_{tool.name}", card.purpose))
    # when_to_use 각 예시를 별도 문서로
    for i, example in enumerate(card.when_to_use):
        docs.append((f"tool_{tool.name}__use_{i}", example))
    # 태그를 별도 문서로 (있을 때만)
    if card.tags:
        docs.append((f"tool_{tool.name}__tags", " ".join(card.tags)))
    return docs


def _compute_tools_hash(tools: list[BaseTool] | tuple[BaseTool, ...]) -> str:
    """도구 목록 + 카드 내용 + 인덱싱 스키마 버전의 변경 여부를 감지하는 해시.

    _INDEX_SCHEMA_VERSION 을 올리면 ToolCard 내용이 같아도 재인덱싱이 트리거된다.
    """
    content = f"schema:{_INDEX_SCHEMA_VERSION}|" + "|".join(
        f"{t.name}:{get_card(t.name).to_embed_text() if get_card(t.name) else t.description}"
        for t in sorted(tools, key=lambda x: x.name)
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class ToolEmbeddingSearch:
    def __init__(self) -> None:
        s = get_settings()
        self._ef = get_embedding_function()
        self._client = get_chromadb_client()
        self._collection = self._client.get_or_create_collection(
            name=s.chromadb_tool_collection,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )
        self._top_k = s.tool_search_top_k
        self._index_lock = threading.Lock()

    def index_tools(self, tools: list[BaseTool] | tuple[BaseTool, ...]) -> None:
        """LangChain 도구들을 ChromaDB에 멀티-벡터 임베딩. 변경 시에만 재인덱싱.

        Upsert-First 전략: 새 문서를 먼저 추가한 뒤 stale 문서만 제거한다.
        이전 Delete-All → Insert-All 방식에서 발생하던 제로벡터 구간(~3초)을
        완전히 제거하여 서비스 중 재인덱싱 시에도 검색 공백이 없다.
        """
        if not tools:
            logger.warning("No tools to index, skipping")
            return

        with self._index_lock:
            self._index_tools_impl(tools)

    def _index_tools_impl(self, tools: list[BaseTool] | tuple[BaseTool, ...]) -> None:
        new_hash = _compute_tools_hash(tools)

        try:
            ver = self._collection.get(ids=["__spec_version__"])
            if ver["metadatas"] and ver["metadatas"][0].get("spec_version") == new_hash:
                logger.info("Tool specs unchanged (v=%s), skip re-index", new_hash)
                return
        except Exception:
            pass

        no_card = missing_cards([t.name for t in tools])
        if no_card:
            logger.warning(
                "ToolCard 없는 도구 %d개 (description fallback): %s",
                len(no_card), ", ".join(no_card),
            )

        ids, docs, metas = [], [], []
        for t in tools:
            for doc_id, text in _tool_documents(t):
                ids.append(doc_id)
                docs.append(text)
                metas.append({
                    "tool_name": t.name,
                    "type": "tool",
                    "has_card": str(get_card(t.name) is not None),
                })

        new_id_set = set(ids)

        # 1) UPSERT FIRST — 새/수정 문서를 먼저 추가 (검색 공백 없음)
        self._collection.upsert(ids=ids, documents=docs, metadatas=metas)

        # 2) Stale 문서만 제거 (삭제된 도구 or when_to_use 개수 감소분)
        existing = self._collection.get(include=[])
        existing_tool_ids = {
            i for i in (existing["ids"] or []) if i.startswith("tool_")
        }
        stale_ids = existing_tool_ids - new_id_set
        if stale_ids:
            self._collection.delete(ids=list(stale_ids))
            logger.info("Cleaned %d stale tool documents", len(stale_ids))

        # 3) 버전 해시 갱신
        self._collection.upsert(
            ids=["__spec_version__"],
            documents=[f"version:{new_hash}"],
            metadatas=[{"type": "version", "spec_version": new_hash}],
        )
        logger.info(
            "Indexed %d tools → %d documents (v=%s)",
            len(tools), len(ids), new_hash,
        )

    def remove_tool(self, tool_name: str) -> None:
        """단일 도구의 벡터를 ChromaDB에서 제거. 런타임 도구 해제 시 호출."""
        existing = self._collection.get(include=[])
        to_delete = [
            i for i in (existing["ids"] or [])
            if i == f"tool_{tool_name}" or i.startswith(f"tool_{tool_name}__")
        ]
        if to_delete:
            self._collection.delete(ids=to_delete)
            logger.info("Removed %d documents for tool '%s'", len(to_delete), tool_name)

    @db_retry
    def search(self, query: str, top_k: int | None = None) -> list[ToolCandidate]:
        """쿼리와 가장 유사한 top-k tool 후보를 반환.

        멀티-벡터: 같은 tool의 여러 문서 중 max score 를 그 tool의 점수로 사용한다.

        e5 모델의 경우 ChromaDB EF 는 "passage: " 프리픽스로 문서를 인덱싱하므로,
        쿼리도 "query: " 프리픽스를 붙여 pre-embed 후 query_embeddings 로 전달한다.
        비대칭 임베딩 덕분에 tool 검색 정확도가 향상된다.
        """
        from app.config import get_raw_embedding_model, get_settings, is_e5_model

        k = top_k or self._top_k
        count = self._collection.count()
        if count == 0:
            return []

        settings = get_settings()
        if is_e5_model(settings.embedding_model):
            model = get_raw_embedding_model()
            q_vec = model.encode(
                [f"query: {query}"], normalize_embeddings=True
            )[0].tolist()
            query_kwargs: dict = {"query_embeddings": [q_vec]}
        else:
            query_kwargs = {"query_texts": [query]}

        # 각 tool 이 N개 문서를 가지므로 k * 5 개 후보를 fetch 후 집계
        fetch_n = min(k * 5, count)
        results = self._collection.query(
            **query_kwargs,
            n_results=fetch_n,
            where={"type": "tool"},
        )

        # tool_name 별 max score 집계
        best: dict[str, float] = defaultdict(float)
        best_desc: dict[str, str] = {}

        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0
                score = round(1.0 - distance, 4)
                tool_name = meta.get("tool_name", doc_id.split("__")[0].replace("tool_", ""))

                if score > best[tool_name]:
                    best[tool_name] = score
                    best_desc[tool_name] = (
                        results["documents"][0][i] if results["documents"] else ""
                    )

        candidates = [
            ToolCandidate(name=name, score=score, description=best_desc.get(name, ""))
            for name, score in best.items()
        ]
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:k]


@lru_cache()
def get_tool_search() -> ToolEmbeddingSearch:
    return ToolEmbeddingSearch()
