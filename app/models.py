"""API 요청/응답 모델 — Pydantic v2 검증 적용."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ..., min_length=1, max_length=5000, description="사용자 질문 (1~5000자)"
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="세션 식별자 (UUID)",
    )
    thread_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=r"^[\w\-:.]+$",
        description="대화 스레드 ID (영문·숫자·하이픈·콜론·밑줄)",
    )


class TraceEntry(BaseModel):
    """파이프라인 실행 추적 항목.

    extra="ignore"로 그래프 노드가 추가하는 임의 필드(disclaimer_appended 등)를
    스키마 변경 없이 수용한다.
    """

    model_config = ConfigDict(extra="ignore")

    node: str = Field(..., description="실행 노드 이름")
    duration_ms: float = Field(default=0, ge=0, description="노드 실행 시간(ms)")
    action: str | None = Field(default=None, description="가드레일 판정 (pass/block)")
    reason: str | None = Field(default=None, description="차단 사유")
    tools_bound: int | None = Field(default=None, ge=0, description="바인딩된 도구 수")
    disclaimer_appended: bool | None = Field(default=None, description="면책 문구 주입 여부")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="챗봇 응답 텍스트")
    session_id: str = Field(..., description="세션 식별자")
    thread_id: str = Field(..., description="대화 스레드 ID")
    tools_used: list[str] = Field(default_factory=list, description="사용된 도구 이름 목록")
    trace: list[TraceEntry] = Field(
        default_factory=list, description="파이프라인 실행 추적 로그"
    )
