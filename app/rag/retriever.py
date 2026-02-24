"""RAG Retriever — ChromaDB 기반 문서 검색으로 응답 보강.

보험 상품요약서, 약관 PDF를 파싱하여 ChromaDB에 저장하고,
tool 결과가 불충분(partial/low_conf)할 때 추가 문서를 검색한다.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings, get_embedding_function, get_chromadb_client
from app.retry import db_retry
from app.rag.splitter import TextSplitter

logger = logging.getLogger("mcp_abtest.rag")


class RAGRetriever:
    def __init__(self) -> None:
        s = get_settings()
        self._ef = get_embedding_function()
        self._client = get_chromadb_client()
        self._collection = self._client.get_or_create_collection(
            name=s.chromadb_doc_collection,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )
        self._top_k = s.rag_top_k

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    def ingest_texts(
        self,
        doc_id_prefix: str,
        chunks: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """텍스트 청크들을 ChromaDB에 색인.

        동일 doc_id_prefix의 기존 청크를 모두 삭제한 뒤 새로 추가한다.
        PDF 교체·재색인 시 구버전 청크가 남지 않는다.

        모든 청크 메타데이터에 doc_id 필드를 자동 주입하여
        where 필터 기반 정확한 stale 삭제를 보장한다.
        """
        if not chunks:
            return 0

        # doc_id 주입 — source 필드명과 관계없이 항상 일치하는 삭제 키
        base_metas = metadatas or [{"source": doc_id_prefix}] * len(chunks)
        metas = [{**m, "doc_id": doc_id_prefix} for m in base_metas]

        # 동일 prefix의 기존 청크 삭제 (where 필터 — 전체 fetch 불필요)
        try:
            stale = self._collection.get(where={"doc_id": doc_id_prefix})
            if stale["ids"]:
                self._collection.delete(ids=stale["ids"])
                logger.info("Deleted %d stale chunks (doc_id=%s)", len(stale["ids"]), doc_id_prefix)
        except Exception as e:
            logger.warning("Stale chunk cleanup skipped: %s", e)

        ids = [f"{doc_id_prefix}_chunk_{i}" for i in range(len(chunks))]
        self._collection.add(ids=ids, documents=chunks, metadatas=metas)
        logger.info("Ingested %d chunks (doc_id=%s)", len(ids), doc_id_prefix)
        return len(ids)

    @db_retry
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """쿼리와 유사한 top-k 문서 청크를 검색.

        Args:
            where: ChromaDB where 필터 (예: {"source": {"$ne": "약관.pdf"}}).
        """
        k = top_k or self._top_k
        count = self._collection.count()
        if count == 0:
            return []

        from app.config import get_raw_embedding_model, get_settings, is_e5_model

        settings = get_settings()
        if is_e5_model(settings.embedding_model):
            model = get_raw_embedding_model()
            q_vec = model.encode(
                [f"query: {query}"], normalize_embeddings=True
            )[0].tolist()
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [q_vec],
                "n_results": min(k, count),
            }
        else:
            query_kwargs = {
                "query_texts": [query],
                "n_results": min(k, count),
            }
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        docs: list[dict[str, Any]] = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 1.0
                docs.append({
                    "id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "score": round(1.0 - distance, 4),
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })

        return docs


@lru_cache()
def get_rag_retriever() -> RAGRetriever:
    return RAGRetriever()


def ingest_text_file(
    txt_path: str | Path,
    splitter: TextSplitter | None = None,
    doc_version: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> int:
    """텍스트 파일(.txt)을 파싱하여 RAG 컬렉션에 색인.

    Args:
        txt_path: 텍스트 파일 경로.
        splitter: 텍스트 분할기. None이면 SentenceSplitter(기본값) 사용.
        doc_version: 문서 버전 문자열.
        extra_meta: 청크 메타데이터에 추가할 임의 필드.
    """
    path = Path(txt_path)
    if not path.exists():
        logger.warning("Text file not found: %s", path)
        return 0

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return 0

    if splitter is None:
        from app.rag.splitter import SentenceSplitter
        splitter = SentenceSplitter()

    chunks = splitter.split(text)
    if not chunks:
        return 0

    base: dict[str, Any] = {"source": path.name}
    if doc_version:
        base["doc_version"] = doc_version
    if extra_meta:
        base.update(extra_meta)

    metadatas = [{**base} for _ in chunks]

    retriever = get_rag_retriever()
    return retriever.ingest_texts(
        doc_id_prefix=path.stem,
        chunks=chunks,
        metadatas=metadatas,
    )


def ingest_pdf(
    pdf_path: str | Path,
    splitter: TextSplitter | None = None,
    doc_version: str | None = None,
    effective_date: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> int:
    """PDF 파일을 파싱하여 RAG 컬렉션에 색인.

    Args:
        pdf_path: PDF 파일 경로.
        splitter: 텍스트 분할기. None이면 SentenceSplitter(기본값) 사용.
        doc_version: 문서 버전 문자열 (예: "2024-03", "v2.1").
            메타데이터 필터링·감사 추적·컨텍스트 레이블에 사용된다.
        effective_date: 약관 적용 시작일 (예: "2024-01-01").
            상충하는 버전의 약관이 여러 개 있을 때 최신 여부를 판단하는 데 활용.
        extra_meta: 청크 메타데이터에 추가할 임의 필드 (예: product_code, category 등).
            메타데이터 필터링(where 절)을 위한 커스텀 필드 삽입에 사용.
    """
    import fitz  # pymupdf

    path = Path(pdf_path)
    if not path.exists():
        logger.warning("PDF not found: %s", path)
        return 0

    doc = fitz.open(str(path))
    if splitter is None:
        from app.rag.splitter import SentenceSplitter
        splitter = SentenceSplitter()

    page_texts: list[tuple[int, str]] = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text().strip()
        if text:
            page_texts.append((page_num + 1, text))
    doc.close()

    if not page_texts:
        return 0

    full_text = "\n\n".join(t for _, t in page_texts)
    chunks = splitter.split(full_text)
    if not chunks:
        return 0

    metadatas = _assign_page_metadata(
        chunks, full_text, page_texts,
        source=path.name,
        doc_version=doc_version,
        effective_date=effective_date,
        extra_meta=extra_meta,
    )

    retriever = get_rag_retriever()
    return retriever.ingest_texts(
        doc_id_prefix=path.stem,
        chunks=chunks,
        metadatas=metadatas,
    )


def _assign_page_metadata(
    chunks: list[str],
    full_text: str,
    page_texts: list[tuple[int, str]],
    source: str,
    doc_version: str | None = None,
    effective_date: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """각 청크가 속한 페이지 번호를 추정하여 메타데이터로 반환.

    doc_version·effective_date·extra_meta는 모든 청크에 공통 적용된다.
    extra_meta를 통해 product_code 같은 필터링용 필드를 자유롭게 추가할 수 있다.
    """
    page_offsets: list[tuple[int, int]] = []
    offset = 0
    for pn, t in page_texts:
        page_offsets.append((offset, pn))
        offset += len(t) + 2  # +2 for \n\n separator

    def _page_at(pos: int) -> int:
        for i in range(len(page_offsets) - 1, -1, -1):
            if pos >= page_offsets[i][0]:
                return page_offsets[i][1]
        return 1

    base: dict[str, Any] = {"source": source}
    if doc_version:
        base["doc_version"] = doc_version
    if effective_date:
        base["effective_date"] = effective_date
    if extra_meta:
        base.update(extra_meta)

    metadatas: list[dict[str, Any]] = []
    search_from = 0
    for chunk in chunks:
        needle = chunk[: min(60, len(chunk))]
        idx = full_text.find(needle, search_from)
        if idx < 0:
            idx = search_from
        metadatas.append({**base, "page": _page_at(idx)})
        search_from = idx + max(1, len(chunk) // 3)

    return metadatas
