"""텍스트 분할기 — 문장 경계를 존중하는 재사용 가능한 chunker.

사용법:
    from app.rag.splitter import SentenceSplitter, SplitterConfig

    splitter = SentenceSplitter()                          # 기본값
    splitter = SentenceSplitter(SplitterConfig(chunk_size=800))  # 커스텀
    splitter = SentenceSplitter(separators=[r"\\n\\n+", ...])    # 커스텀 분할자

    chunks = splitter.split(long_text)

확장:
    TextSplitter를 상속하여 새 전략을 구현할 수 있다.
    separators 리스트만 교체해도 도메인별 커스터마이징 가능.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SplitterConfig:
    """텍스트 분할 설정.

    chunk_size     : 청크 최대 글자 수 (overlap 포함 전 기준)
    chunk_overlap  : 이전 청크와 겹칠 문장 수준의 글자 예산
    min_chunk_size : 이보다 짧은 마지막 청크는 직전 청크에 병합
    """
    chunk_size: int = 500
    chunk_overlap: int = 100
    min_chunk_size: int = 50


class TextSplitter(ABC):
    """텍스트 분할 인터페이스 — 새 전략은 이 클래스를 상속."""

    def __init__(self, config: SplitterConfig | None = None):
        self.config = config or SplitterConfig()

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """텍스트를 청크 리스트로 분할."""
        ...


class SentenceSplitter(TextSplitter):
    """문장 경계를 존중하는 텍스트 분할기.

    분할 우선순위 (separators):
      1. 단락 경계 (\\n\\n)
      2. 약관 구조 (제N조/항/관 앞)
      3. 한국어 문장 종결 (다./요./까? 등 뒤)
      4. 영문 문장 종결 (./?/! 뒤)
      5. 줄바꿈

    높은 우선순위에서 chunk_size 이내로 분할되면 하위 레벨은 시도하지 않는다.
    모든 분할자가 실패하면 공백 기준으로 강제 분할한다.
    """

    KOREAN_LEGAL_SEPARATORS: list[str] = [
        r"\n\n+",
        r"(?=제\d{1,3}[조항관]\s)",
        r"(?<=[가-힣][.?!])\s+",
        r"(?<=[.?!])\s+",
        r"\n",
    ]

    def __init__(
        self,
        config: SplitterConfig | None = None,
        separators: list[str] | None = None,
    ):
        super().__init__(config)
        seps = separators if separators is not None else self.KOREAN_LEGAL_SEPARATORS
        self._compiled = [re.compile(s) for s in seps]

    def split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.config.chunk_size:
            return [text]

        segments = self._split_recursive(text, level=0)
        return self._merge_segments(segments)

    # ── 내부 구현 ────────────────────────────────────────────────────────────

    def _split_recursive(self, text: str, level: int) -> list[str]:
        """separators를 우선순위대로 시도하며 재귀 분할."""
        if len(text) <= self.config.chunk_size:
            return [text]
        if level >= len(self._compiled):
            return self._force_split(text)

        parts = self._compiled[level].split(text)
        parts = [p for p in parts if p.strip()]

        if len(parts) <= 1:
            return self._split_recursive(text, level + 1)

        result: list[str] = []
        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue
            if len(stripped) <= self.config.chunk_size:
                result.append(stripped)
            else:
                result.extend(self._split_recursive(stripped, level + 1))
        return result

    def _merge_segments(self, segments: list[str]) -> list[str]:
        """작은 세그먼트들을 chunk_size까지 병합 + 문장 단위 overlap."""
        if not segments:
            return []

        cfg = self.config
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for seg in segments:
            joiner = 1 if current else 0
            if current_len + joiner + len(seg) > cfg.chunk_size and current:
                chunks.append(" ".join(current))

                # overlap: 이전 청크 끝부분 문장을 예산 내에서 유지
                overlap_segs: list[str] = []
                budget = 0
                for s in reversed(current):
                    if budget + len(s) + 1 > cfg.chunk_overlap:
                        break
                    overlap_segs.insert(0, s)
                    budget += len(s) + 1

                current = overlap_segs
                current_len = sum(len(s) for s in current) + max(len(current) - 1, 0)

            current.append(seg)
            current_len += joiner + len(seg)

        if current:
            final = " ".join(current)
            if chunks and len(final) < cfg.min_chunk_size:
                chunks[-1] += " " + final
            else:
                chunks.append(final)

        return chunks

    def _force_split(self, text: str) -> list[str]:
        """모든 separator 소진 시 공백 기준으로 강제 분할."""
        cfg = self.config
        chunks: list[str] = []
        while len(text) > cfg.chunk_size:
            cut = text[: cfg.chunk_size].rfind(" ")
            if cut <= cfg.min_chunk_size:
                cut = cfg.chunk_size
            chunks.append(text[:cut].strip())
            text = text[cut:].strip()
        if text:
            chunks.append(text)
        return chunks
