"""ChromaDB 초기화 스크립트 — Tool 임베딩 색인 + PDF 문서 인제스트.

PDF 파일명 규칙: {상품코드}_{개정번호}_{유형}.pdf
  예) B00329010_0_S.pdf  → product_code=B00329010, revision=0
파일명이 규칙을 따르면 product_code·doc_version 메타데이터가 자동 설정된다.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.tool_search.embedder import get_tool_search
from app.tools import get_all_tools
from app.rag.retriever import ingest_pdf, ingest_text_file

# 파일명 패턴: B00329010_0_S.pdf
_PDF_NAME_RE = re.compile(r"^(B\d{6,9})_(\d+)_", re.IGNORECASE)


def _parse_pdf_meta(pdf: Path) -> dict:
    """파일명에서 product_code·doc_version을 자동 추출."""
    m = _PDF_NAME_RE.match(pdf.name)
    if not m:
        return {}
    return {
        "doc_version": m.group(2),
        "extra_meta": {"product_code": m.group(1)},
    }


def main():
    s = get_settings()
    print(f"ChromaDB persist dir: {s.chromadb_persist_dir}")

    print("\n[1/2] Tool 임베딩 색인 중...")
    searcher = get_tool_search()
    tools = get_all_tools()
    searcher.index_tools(tools)
    print(f"  {len(tools)}개 tool 색인 완료")

    print("\n[2/2] PDF 문서 인제스트 중...")
    pdf_dir = Path(__file__).resolve().parent.parent / "상품요약서_판매중_표준약관"
    if not pdf_dir.exists():
        print(f"  PDF 디렉토리 없음: {pdf_dir}")
        print("\n초기화 완료!")
        return

    total_chunks = 0

    pdf_files = list(pdf_dir.glob("*.pdf"))
    for pdf in pdf_files:
        meta = _parse_pdf_meta(pdf)
        print(f"  처리 중: {pdf.name}", end="")
        if meta.get("extra_meta"):
            print(f"  (product_code={meta['extra_meta']['product_code']}, rev={meta['doc_version']})", end="")
        print()
        n = ingest_pdf(pdf, **meta)
        total_chunks += n
        print(f"    → {n}개 청크 색인됨")

    txt_files = list(pdf_dir.glob("*.txt"))
    for txt in txt_files:
        print(f"  처리 중: {txt.name}")
        n = ingest_text_file(txt)
        total_chunks += n
        print(f"    → {n}개 청크 색인됨")

    print(f"  총 {total_chunks}개 청크 인제스트 완료")

    print("\n초기화 완료!")


if __name__ == "__main__":
    main()
