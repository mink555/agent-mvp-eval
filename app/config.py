"""설정 — pydantic-settings v2 + 타입 안전 검증."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── OpenRouter LLM ────────────────────────────────────
    openrouter_api_key: str = ""
    openrouter_model: str = "qwen/qwen3-14b"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # ── Checkpointer ──────────────────────────────────────
    checkpoint_db_path: str = "./checkpoints.db"

    # ── ChromaDB ──────────────────────────────────────────
    chromadb_persist_dir: str = "./chroma_data"
    chromadb_tool_collection: str = "tool_embeddings"
    chromadb_doc_collection: str = "rag_documents"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # ── Search / RAG ──────────────────────────────────────
    tool_search_top_k: int = Field(default=5, ge=1, le=100)
    rag_top_k: int = Field(default=5, ge=1, le=50)
    max_conversation_turns: int = Field(default=20, ge=1, le=100)

    # ── FastAPI Server ────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8080, ge=1, le=65535)

    # ── MCP Server ────────────────────────────────────────
    mcp_server_name: str = "insurance-tools"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = Field(default=8000, ge=1, le=65535)
    mcp_transport: Literal["sse", "stdio", "streamable-http"] = "sse"

    # ── Logging ───────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("openrouter_api_key")
    @classmethod
    def _require_api_key(cls, v: str) -> str:
        """서버 시작 시 API key 미설정을 즉시 감지.

        빈 문자열이면 첫 LLM 호출 때가 아니라 Settings 로드 시점에 실패하도록 한다.
        .env 또는 환경변수 OPENROUTER_API_KEY 를 설정해야 한다.
        """
        if not v:
            import warnings
            warnings.warn(
                "OPENROUTER_API_KEY 가 설정되지 않았습니다. "
                "LLM 호출 시 AuthenticationError 가 발생합니다.",
                stacklevel=2,
            )
        return v


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def is_e5_model(model_name: str) -> bool:
    return "e5" in model_name.lower()


@lru_cache()
def get_raw_embedding_model():
    """SentenceTransformer 모델 객체 싱글톤.

    guardrails 등 query/passage 프리픽스를 직접 제어해야 하는 곳에서 사용한다.
    ChromaDB EF 래퍼를 거치지 않으므로 encode() 파라미터를 자유롭게 조정 가능.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


@lru_cache()
def get_embedding_function():
    """ChromaDB 컬렉션용 임베딩 함수 싱글톤.

    - e5 계열 모델(multilingual-e5-*): 문서 인덱싱 시 "passage: " 프리픽스 자동 부가.
      쿼리 임베딩은 embedder.py 의 search() 에서 "query: " 프리픽스로 별도 처리한다.
    - 그 외 모델: 기존 SentenceTransformerEmbeddingFunction 그대로 사용.
    """
    model_name = get_settings().embedding_model
    if is_e5_model(model_name):
        model = get_raw_embedding_model()

        class _E5PassageEF:
            """ChromaDB 전용 — 저장 시 'passage: ' 프리픽스 부가."""

            def name(self) -> str:  # ChromaDB EF 인터페이스 요구
                return f"e5-passage:{model_name}"

            def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
                import numpy as np

                prefixed = [f"passage: {t}" for t in input]
                vecs = model.encode(prefixed, normalize_embeddings=True)
                return np.array(vecs).tolist()

        return _E5PassageEF()

    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(model_name=model_name)


@lru_cache()
def get_chromadb_client():
    """ChromaDB 영속 클라이언트 싱글톤. embedder·retriever에서 공유.

    chromadb.Client(Settings(...)) 는 deprecated → PersistentClient() 사용.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    s = get_settings()
    return chromadb.PersistentClient(
        path=s.chromadb_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
