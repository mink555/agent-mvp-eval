---
title: Insurance Chatbot
emoji: 🏥
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Insurance Chatbot — LangGraph + MCP + ChromaDB

> 보험 도메인 54개 도구를 **정확하게, 빠르게, 안전하게** 호출하는 AI 챗봇

---

## 무엇을 하려 했는가

보험 상담에서 필요한 기능(상품 조회, 보험료 산출, 가입 심사, 보장 분석, 청구 안내, 컴플라이언스 검토)을 LLM 기반 챗봇으로 통합하되, **도구 54개를 LLM에 전부 넘기면 생기는 3가지 문제**를 해결하고자 했습니다.

| 문제 | 원인 | 영향 |
|------|------|------|
| 비용 증가 | 매번 54개 도구 스키마가 컨텍스트에 포함 | 토큰 낭비 |
| 정확도 하락 | 비슷한 이름의 도구 혼동 (premium_estimate vs plan_options) | 오호출 |
| 지연 증가 | 컨텍스트 길이 비례 응답 시간 상승 | UX 저하 |

---

## 무엇을 했는가

### 핵심: 2단계 도구 라우팅

54개 도구를 LLM에 전부 보여주지 않고, **ChromaDB로 먼저 5개로 줄이고 → LLM이 최종 선택**하는 구조를 만들었습니다.

```
사용자 질문
    │
    ▼
[Guardrail] ──(차단)──→ "보험 관련 질문만 가능합니다"
    │(통과)
    ▼
[ChromaDB 벡터 검색: 54개 → 5개]
    │
    ▼
[LLM이 최종 선택] → 도구 실행 → 출력 검증 → 답변
```

### 5노드 파이프라인

```
START → [input_guardrail] → [query_rewriter] → [agent ↔ tools] → [output_guardrail] → END
```

| # | 노드 | 역할 | 소요 시간 |
|---|------|------|-----------|
| 1 | input_guardrail | 정규식(L1) + 임베딩(L2)으로 이상 요청 차단 | <5ms |
| 2 | query_rewriter | 짧은 후속질문("그거 얼마야?")을 맥락 기반 재작성 | 0ms~1s |
| 3 | agent | ChromaDB로 관련 도구 필터링 → LLM 호출 | 1~5s |
| 4 | tools | 선택된 도구 실행 (ToolRegistry에서 동적 디스패치) | 10~100ms |
| 5 | output_guardrail | PII 노출·금칙어·빈 응답 검사 + 면책 문구 자동 추가 | <2ms |

### 도구 레벨 입력 가드

보험료 산출·가입 심사 등 나이/성별이 필요한 도구는 사용자가 직접 제공하지 않은 정보를 **추측하지 않고 되묻도록** 가드를 넣었습니다.

```
premium_estimate(age=None) → {"needs_user_input": true, "missing": ["나이", "성별"]}
→ LLM이 사용자에게 "나이와 성별을 알려주세요" 질문 생성
```

### 서빙: 두 가지 인터페이스

| 방식 | 설명 | 대상 |
|------|------|------|
| **FastAPI** (REST/SSE) | 웹 Chat UI + REST API | 일반 사용자 |
| **MCP Server** (SSE/stdio) | MCP 프로토콜로 도구·리소스·프롬프트 노출 | Claude Desktop, Cursor 등 AI 클라이언트 |

MCP Server는 `--inspect` 플래그로 **MCP Inspector UI**를 띄울 수 있습니다. 도구 54개, 리소스 17개, 프롬프트 8개를 브라우저에서 직접 테스트할 수 있습니다.

```bash
python run_mcp.py --inspect    # Inspector UI 자동 실행
```

---

## 결론: 측정 결과

`scripts/eval_tool_recall.py`로 79개 테스트 쿼리(tool-call 64개 + no-call 15개)를 평가했습니다.

| 지표 | k=3 | **k=5 (운영)** | k=10 |
|------|-----|----------------|------|
| **Recall@k** | 100% | **100%** | 100% |
| **Hit@1** | 98.4% | **98.4%** | 98.4% |
| **MRR** | 0.99 | **0.99** | 0.99 |

- k=3부터 Recall 100% — 64개 쿼리 전부 top-3 안에 정답 도구 포함
- k=5로 운영 — Recall 100% 유지하면서 LLM 컨텍스트 절반 이상 절감
- 유일한 미탐(1건)도 두 도구 모두 적절한 응답이 가능한 경계 사례

**한 줄 요약:** ChromaDB 멀티벡터 인덱싱으로 54개 도구를 5개로 줄여도 Recall@5 = 100%, 정확도를 유지하면서 비용과 지연을 동시에 줄였습니다.

---

## 부수 기능

### 상품공시실 PDF 기반 RAG

라이나생명 상품공시실에서 12개 판매 중 상품의 **상품요약서 PDF**(12개) + 표준약관(1개) + 회사 정보(1개)를 수집하여 ChromaDB에 인제스트했습니다.

```
PDF → PyMuPDF 텍스트 추출 → 500자 청크 + 100자 오버랩 → ChromaDB (~1,400 벡터)
```

약관 조항이나 면책/감액 규정처럼 도구 데이터에 없는 내용은 RAG가 보완합니다. 상품코드 메타데이터로 특정 상품 약관만 필터링 검색도 가능합니다.

### 쿼리 재작성 (Query Rewriter)

멀티턴 대화에서 "그거 얼마야?", "그건?"처럼 짧은 후속질문은 ChromaDB 검색 정확도가 떨어집니다. 15자 미만 후속질문을 감지하면 이전 대화 맥락을 참조해 구체적인 쿼리로 재작성합니다.

```
"그거 얼마야?" → "뉴스타트 암보험 월 보험료 알려줘"
```

### Agentic 시스템 프롬프트

시스템 프롬프트에 12개 상품 목록이 자동 반영됩니다. 새 상품을 `PRODUCTS`에 추가하면 프롬프트가 동적으로 업데이트되어 LLM이 즉시 인식합니다. 도구 체이닝 규칙("상품명만 알면 product_search → 해당 도구 순서로 호출")도 포함되어 있어 LLM이 자율적으로 도구를 연쇄 호출합니다.

### 런타임 도구 핫리로드

서버 재시작 없이 도구를 추가/제거할 수 있습니다. `ToolRegistry`가 도구 목록을 동적 관리하고, 변경 시 ChromaDB 재인덱싱을 자동 트리거합니다.

```bash
curl -X POST http://localhost:8080/api/tools/reload-module/product   # 모듈 재등록
curl -X DELETE http://localhost:8080/api/tools/product_search         # 도구 해제
```

### MCP Inspector

MCP Server의 도구·리소스·프롬프트를 브라우저에서 직접 테스트하는 디버깅 도구입니다. 도구 입력/출력, 리소스 조회, 프롬프트 렌더링을 시각적으로 확인할 수 있습니다.

### LLM 사고과정(think) 필터링

Qwen3 모델의 `<think>...</think>` 블록을 스트리밍 중 실시간 필터링하여 사용자에게는 최종 답변만 보여줍니다. 태그가 청크 경계에 걸리는 엣지 케이스도 처리합니다.

---

## 왜 이 기술을 선택했는가

### ChromaDB (벡터 DB)

| 기준 | FAISS | Milvus | **ChromaDB** |
|------|-------|--------|-------------|
| 메타데이터 필터링 | X | O | **O** |
| 영속성 | X | O | **O** |
| 실시간 upsert | 인덱스 rebuild | O | **O** |
| 인프라 | 없음 | Docker 3개 | **pip 1줄** |
| 적합 규모 | 수억 벡터 | 수억~수십억 | **수천~수만** |

벡터 ~1,800개(도구 370 + RAG 1,400) 규모에서 FAISS는 메타데이터를 직접 구현해야 하고, Milvus는 오버엔지니어링입니다. ChromaDB는 `pip install` 한 줄로 필요한 기능이 전부 됩니다.

### multilingual-e5-large (임베딩 모델)

| 선택 이유 | 설명 |
|-----------|------|
| 한국어 최상위 | MTEB 다국어 벤치마크 한국어 Retrieval 1위급 |
| 비대칭 검색 | "query: {질문}" / "passage: {도구설명}" 프리픽스로 정확도 향상 |
| 로컬 추론 | 외부 API 미의존, 매 요청 ~10ms |
| 실측 검증 | Recall@5 = 100%, MRR = 0.99 달성 |

### Multi-Vector 인덱싱

도구 하나를 단일 벡터로 임베딩하면 여러 사용 예시의 평균이 되어 벡터가 희석됩니다. 대신 purpose + when_to_use 각각을 별도 문서로 인덱싱하고, 검색 시 tool별 max score로 집계합니다.

```
도구 1개 → purpose 1개 + when_to_use N개 + tags 1개 = (N+2)개 문서
```

---

## Quick Start

```bash
# 1. 설치
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. 환경변수 (.env)
OPENROUTER_API_KEY=sk-or-v1-your-key
OPENROUTER_MODEL=qwen/qwen3-14b
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# 3. ChromaDB 초기화 (최초 1회)
python scripts/init_vectordb.py

# 4. 서버 실행
python run.py                         # FastAPI → http://localhost:8080
python run_mcp.py                     # MCP Server → http://127.0.0.1:8000
python run_mcp.py --inspect           # MCP Inspector UI

# 5. 도구 라우팅 평가 (선택)
python -m scripts.eval_tool_recall --compare
```

---

## 프로젝트 구조

```
mcp_abtest_v1/
├── app/
│   ├── main.py                 # FastAPI (REST/SSE + think 필터링)
│   ├── config.py               # Settings + 임베딩 모델 싱글톤
│   ├── graph/                  # LangGraph 5노드 파이프라인
│   │   ├── builder.py          #   그래프 빌드 + 동적 도구 디스패치
│   │   ├── nodes.py            #   agent 노드 (ChromaDB 라우팅 + LLM)
│   │   ├── guardrails.py       #   입력(L1+L2) / 출력 가드레일
│   │   └── query_rewrite.py    #   후속질문 재작성
│   ├── tools/                  # 54개 도구 (8개 모듈)
│   │   ├── __init__.py         #   ToolRegistry (핫리로드)
│   │   ├── data.py             #   12개 상품 시뮬레이션 데이터 + 시스템 프롬프트
│   │   └── product / premium / coverage / underwriting
│   │       compliance / claims / customer_db / rag_tools
│   ├── tool_search/            # ChromaDB 멀티벡터 라우팅
│   │   ├── embedder.py         #   임베딩 + top-k 검색
│   │   └── tool_cards.py       #   54개 ToolCard 메타데이터
│   ├── rag/                    # 약관 PDF RAG
│   │   ├── retriever.py        #   인제스트 + 검색
│   │   └── splitter.py         #   한국어 문장경계 청크 분할
│   └── mcp_server/             # MCP 프로토콜 서버
│       ├── server.py           #   도구 54 + 리소스 17 + 프롬프트 8
│       └── prompts.py / resources.py
├── scripts/
│   ├── init_vectordb.py        # ChromaDB 초기화
│   └── eval_tool_recall.py     # Recall@k / MRR 평가
├── templates/index.html        # 웹 Chat UI (SSE 스트리밍)
├── 상품요약서_판매중_표준약관/   # 상품공시실 PDF 원본 (RAG 소스)
└── Dockerfile                  # HF Spaces 배포용
```

### 도구 카탈로그 (54개)

| 모듈 | 도구 수 | 주요 기능 |
|------|---------|-----------|
| product | 10 | 상품 검색/조회/비교, 특약, FAQ |
| premium | 8 | 보험료 산출/비교, 플랜, 갱신 추정 |
| coverage | 9 | 보장 요약/상세, 급부 금액, ICD 매핑 |
| underwriting | 12 | 가입 심사, 녹아웃 룰, 직업 위험도 |
| compliance | 6 | 준법 멘트, 금칙어, 개인정보 마스킹 |
| claims | 4 | 청구 절차, 서류, 계약관리 |
| customer_db | 3 | 고객 검색, 계약 조회, 중복 가입 검사 |
| rag_tools | 2 | 약관 검색, 상품요약서 검색 |

---

## API 명세

| # | Method | Path | 설명 |
|---|--------|------|------|
| 1 | POST | /api/chat | 동기 응답 |
| 2 | POST | /api/chat/stream | SSE 스트리밍 (노드 진행 + 토큰) |
| 3 | GET | /api/health | 헬스체크 |
| 4 | GET | /api/tools | 도구 카탈로그 |
| 5 | POST | /api/tools/reload-module/{module} | 도구 핫리로드 |
| 6 | DELETE | /api/tools/{tool_name} | 도구 런타임 해제 |
| 7 | GET | / | 웹 Chat UI |

---

## 기술 스택

| 카테고리 | 기술 | 역할 |
|----------|------|------|
| LLM 오케스트레이션 | LangGraph | ReAct 그래프, 멀티턴, 조건부 분기 |
| LLM | OpenRouter (qwen/qwen3-14b) | 다중 모델 라우팅 |
| 벡터 DB | ChromaDB | 도구 라우팅 + RAG 검색 |
| 임베딩 | multilingual-e5-large (1024차원) | 한국어 비대칭 검색 |
| API 서버 | FastAPI | REST + SSE 스트리밍 |
| MCP 서버 | FastMCP | Claude Desktop/Cursor 연동 |
| 체크포인터 | langgraph-checkpoint-sqlite | 대화 상태 영구 저장 |
| PDF 파싱 | PyMuPDF | 약관/요약서 텍스트 추출 |
| 고객 DB | SQLite3 | 고객/계약 시뮬레이션 |
