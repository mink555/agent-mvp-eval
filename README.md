# MCP(AI TMR Assistant) — Tool Routing 고도화

> **Intelligent Tool Routing**으로 정확도를 개선하고, **Scalable Tool Architecture**로 운영 효율과 확장성을 동시에 확보함

---

## 목적

12개 보험 상품에 대해 상품 조회, 보험료 산출, 가입 심사, 보장 분석, 청구 안내, 컴플라이언스 검토를 수행하는 AI 챗봇을 구현함. 도구가 54개로 늘어나면서 발생하는 오호출·비용 증가·지연 문제를 해결하는 것이 핵심 과제였음.

| 문제 | 원인 | 영향 |
|------|------|------|
| 오호출 | 유사 도구 간 혼동 (premium_estimate vs plan_options) | 잘못된 답변 |
| 비용 증가 | 매 요청마다 54개 도구 스키마가 컨텍스트에 포함됨 | 토큰 낭비 |
| 지연 증가 | 컨텍스트 길이에 비례해 응답 시간 상승 | UX 저하 |

> 도구가 10개를 넘으면 정확도가 저하되기 시작하고, 37개 도구 기준 ~6,200 토큰이 소비됨 [(참고)](https://achan2013.medium.com/how-many-tools-functions-can-an-ai-agent-has-21e0a82b7847). RAG-MCP 패턴처럼 "모든 도구를 넘기지 말고, 필요한 것만 검색해서 넘기자"는 접근이 필요했음 [(참고)](https://writer.com/engineering/rag-mcp/).

---

## 전략 3축

### 1. Intelligent Tool Routing — 오호출을 줄이고 적확한 후보만 남김

54개 도구를 LLM에 전부 넘기지 않음. Guardrail → Tool Search → LLM 선택의 3단계로 필터링함.

```
사용자 질문
    │
    ▼
[Guardrail] ── 이상 요청 차단 ──→ 거절 응답 (No-Call)
    │(통과)
    ▼
[Tool Search] ── 54개 → Top-K 후보 추출 (ChromaDB 벡터 검색)
    │
    ▼
[LLM Tool Call] ── 후보 중 최종 선택 → Tool 실행
```

| 단계 | 실행 모듈 | 기능 | 측정 지표 |
|------|-----------|------|-----------|
| Guardrail | 정규식(L1) + 임베딩(L2) | 탈옥·비보험 질문을 <5ms에 차단 | No-Call Acc |
| Tool Search | ChromaDB 멀티벡터 검색 | 54개 중 관련 도구 Top-K 추출 | Tool Acc, Recall@k |
| LLM Selection | bind_tools() | 축소된 후보에서 실제 필요한 것만 호출 | — |

Guardrail이 먼저 작동하므로, 비보험 질문("오늘 날씨 어때?")이 Tool Search까지 도달하지 않아 불필요한 벡터 연산 + LLM 비용이 발생하지 않음.

### 2. Scalable Tool Architecture — 신규 Tool 추가가 운영 부담이 되지 않도록 자동화

```
[Scalable Tool Architecture]

새 Tool 추가
    │
    ▼
Tool Card 작성 (purpose + when_to_use + tags)
    │
    ▼
임베딩 생성 (multilingual-e5-large, 멀티벡터)
    │
    ▼
Vector Index에 자동 반영 → 즉시 검색 대상에 포함
```

두 가지 등록 방식을 지원함.

| 방식 | 절차 | 서버 재시작 |
|------|------|-------------|
| 정적 등록 | Tool 함수 작성 → ToolCard 등록 → 서버 재시작 | 필요 |
| 런타임 핫리로드 | Tool 함수 작성 → API 호출 (`POST /api/tools/reload-module/{module}`) | **불필요** |

ToolRegistry가 도구 목록을 동적 관리하고, 변경 시 ChromaDB 재인덱싱을 자동 트리거함. LangGraph 그래프 재컴파일 없이 다음 요청부터 즉시 반영됨.

### 3. Validation (Proof Layer) — 감이 아니라 숫자로 판단

`scripts/eval_tool_recall.py`로 Recall@k, MRR, Hit@1을 정량 측정함. 도구 추가·삭제·임베딩 변경 후 성능 변화를 비교하여 회귀(regression)를 방지함.

```bash
python -m scripts.eval_tool_recall --compare    # k=1,3,5,7,10 비교표
python -m scripts.eval_tool_recall --verbose     # 오판 사례 상세
```

---

## 결과

79개 테스트 쿼리(tool-call 64개 + no-call 15개) 평가 결과:

| 지표 | k=1 | k=3 | **k=5 (운영)** | k=7 | k=10 |
|------|-----|-----|----------------|-----|------|
| **Recall@k** | 98.4% | 100% | **100%** | 100% | 100% |
| **Hit@1** | 98.4% | 98.4% | **98.4%** | 98.4% | 98.4% |
| **MRR** | 0.9844 | 0.9922 | **0.9922** | 0.9922 | 0.9922 |

- k=1에서 이미 Hit@1 = 98.4%. 미탐 1건은 두 도구 모두 적절한 응답이 가능한 경계 사례
- k=3부터 Recall 100%. 64개 tool-call 쿼리 전부 Top-3 안에 정답 도구가 포함됨
- no-call 오판 4건 — 비보험 질문이지만 유사도 상위 도구가 0.86~0.88로 경계에 걸림. Guardrail(L1+L2)에서 사전 차단되므로 실제 운영에서는 Tool Search까지 도달하지 않음
- k=5로 운영 — Recall 100% 유지하면서 LLM 컨텍스트를 54개 → 5개(90% 절감)로 축소함

| 점수 분포 (k=10 기준) | min | avg | max |
|------------------------|-----|-----|-----|
| Tool-Call top-1 score | 0.867 | 0.922 | 0.947 |
| No-Call top-1 score | 0.831 | 0.853 | 0.877 |

**결론:** ChromaDB 멀티벡터 인덱싱으로 54개 → 5개 축소해도 Recall@5 = 100%, MRR = 0.99. 정확도를 유지하면서 비용과 지연을 동시에 줄임.

---

## 구현 상세

### 5노드 파이프라인 (LangGraph)

[LangGraph](https://langchain-ai.github.io/langgraph/)로 ReAct 패턴 기반 5노드 그래프를 구성함.

```
START → [input_guardrail] → [query_rewriter] → [agent ↔ tools] → [output_guardrail] → END
```

| 노드 | 역할 | 소요 시간 |
|------|------|-----------|
| input_guardrail | 정규식(L1) + 임베딩(L2)으로 이상 요청 차단 | <5ms |
| query_rewriter | 짧은 후속질문을 이전 맥락으로 재작성 | 0ms~1s |
| agent | ChromaDB로 Top-K 필터링 → LLM 호출 | 1~5s |
| tools | ToolRegistry에서 동적 디스패치로 도구 실행 | 10~100ms |
| output_guardrail | PII 노출·금칙어 검사 + 면책 문구 자동 추가 | <2ms |

### 도구 레벨 입력 가드

보험료 산출·가입 심사 등 나이/성별이 필수인 도구는, 사용자가 직접 제공하지 않은 정보를 추측하지 않음. 도구가 `needs_user_input`을 반환하면 LLM이 사용자에게 해당 정보를 질문함.

### 쿼리 재작성 (Query Rewriter)

"그거 얼마야?", "그건?" 같은 15자 미만 후속질문은 ChromaDB 검색 정확도가 떨어짐. 이전 대화 맥락을 참조해 구체적 쿼리로 재작성하여 Tool Search 정확도를 보완함. Query Transformation은 Advanced RAG의 핵심 기법 중 하나임 [(참고)](https://www.promptingguide.ai/research/rag).

### 상품공시실 PDF 기반 RAG

보험 상품공시실에서 12개 상품요약서 PDF + 표준약관 + 회사 정보를 수집함. PyMuPDF로 텍스트 추출 후 500자 청크로 분할하여 ChromaDB에 인제스트함(~1,400 벡터). 도구 데이터에 없는 약관 조항·면책 규정은 RAG가 보완함.

### Agentic 시스템 프롬프트

시스템 프롬프트에 12개 상품 목록이 PRODUCTS 딕셔너리에서 동적 반영됨. 새 상품 추가 시 프롬프트가 자동 업데이트되어 LLM이 즉시 인식함. 도구 체이닝 규칙("상품명만 알면 product_search → 해당 도구 순서로 호출")도 포함하여 LLM이 자율적으로 도구를 연쇄 호출함.

### LLM 사고과정 필터링

Qwen3 모델의 `<think>...</think>` 블록을 스트리밍 중 실시간 필터링함. 사용자에게는 최종 답변만 노출되고, 파이프라인 진행 상태(SSE 이벤트)로 체감 지연을 줄임.

### 서빙: 두 가지 인터페이스

| 방식 | 설명 | 대상 |
|------|------|------|
| FastAPI (REST/SSE) | 웹 Chat UI + REST API | 일반 사용자 |
| MCP Server (SSE/stdio) | 도구 54 + 리소스 17 + 프롬프트 8 노출 | Claude Desktop, Cursor 등 |

MCP Server는 `--inspect` 플래그로 **MCP Inspector UI**를 실행할 수 있음. 도구 입출력, 리소스 조회, 프롬프트 렌더링을 브라우저에서 직접 테스트 가능.

```bash
python run_mcp.py --inspect
```

---

## 기술 선택 근거

### ChromaDB

| 기준 | FAISS | Milvus | **ChromaDB** |
|------|-------|--------|-------------|
| 메타데이터 필터링 | X | O | **O** |
| 영속성 | X | O | **O** |
| 실시간 upsert | rebuild 필요 | O | **O** |
| 인프라 | 없음 | Docker 3개 | **pip 1줄** |

벡터 ~1,800개(도구 370 + RAG 1,400) 규모에서 Milvus는 오버엔지니어링이고, FAISS는 메타데이터 필터링을 직접 구현해야 함. ChromaDB는 `pip install` 한 줄로 필요한 기능이 전부 됨. 10M 벡터 미만 프로젝트에서 ChromaDB가 권장됨 [(참고)](https://www.firecrawl.dev/blog/best-vector-databases). 주요 벡터 DB 비교표는 [(DataCamp)](https://www.datacamp.com/blog/the-top-5-vector-databases) 참조.

### multilingual-e5-large

[Kor-IR 벤치마크](https://github.com/Atipico1/Kor-IR)(한국어 IR 전용)에서 오픈소스 모델 중 최상위 성능(NDCG@10 = 80.35, Avg = 81.03)을 기록함. Mr. TyDi 한국어 MRR@10 = 61.6으로 e5-base(55.8) 대비 +10% 향상 [(모델 카드)](https://huggingface.co/intfloat/multilingual-e5-large). 비대칭 검색 시 "query: " / "passage: " 프리픽스가 필수이며, 미적용 시 성능 저하가 발생함 [(E5 논문)](https://arxiv.org/abs/2402.05672). 로컬 추론(~10ms/쿼리)으로 외부 API 미의존.

### Multi-Vector 인덱싱

도구 하나를 단일 벡터로 임베딩하면 여러 사용 예시의 평균으로 벡터가 희석됨. purpose + when_to_use 각각을 별도 문서로 인덱싱하고, 검색 시 tool별 max score로 집계하여 희석 없이 정확한 매칭을 달성함. ColBERT 등 multi-vector 모델이 single-vector 대비 정확도가 높은 것과 동일한 원리임 [(참고)](https://www.pinecone.io/blog/cascading-retrieval-with-multi-vector-representations/).

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
python run_mcp.py                     # MCP Server
python run_mcp.py --inspect           # MCP Inspector UI

# 5. 도구 라우팅 평가
python -m scripts.eval_tool_recall --compare
```

---

## 프로젝트 구조

```
app/
├── main.py                 # FastAPI (REST/SSE + think 필터링)
├── config.py               # Settings + 임베딩 모델 싱글톤
├── graph/                  # LangGraph 5노드 파이프라인
│   ├── builder.py          #   그래프 빌드 + 동적 도구 디스패치
│   ├── nodes.py            #   agent 노드 (ChromaDB 라우팅 + LLM)
│   ├── guardrails.py       #   입력(L1+L2) / 출력 가드레일
│   └── query_rewrite.py    #   후속질문 재작성
├── tools/ (54개, 8모듈)    # product / premium / coverage / underwriting
│   ├── __init__.py         #   ToolRegistry (핫리로드)
│   └── data.py             #   12개 상품 데이터 + 시스템 프롬프트
├── tool_search/            # ChromaDB 멀티벡터 라우팅
│   ├── embedder.py         #   임베딩 + Top-K 검색
│   └── tool_cards.py       #   54개 ToolCard
├── rag/                    # 상품공시실 PDF RAG
│   ├── retriever.py        #   인제스트 + 검색
│   └── splitter.py         #   한국어 문장경계 청크 분할
└── mcp_server/             # MCP 프로토콜 서버 + Inspector

scripts/
├── init_vectordb.py        # ChromaDB 초기화
└── eval_tool_recall.py     # Recall@k / MRR 평가 (Proof Layer)
```

### 도구 카탈로그 (54개)

| 모듈 | 수 | 주요 기능 |
|------|----|-----------|
| product | 10 | 상품 검색/조회/비교, 특약, FAQ |
| premium | 8 | 보험료 산출/비교, 플랜, 갱신 추정 |
| coverage | 9 | 보장 요약/상세, 급부 금액 |
| underwriting | 12 | 가입 심사, 녹아웃 룰, 직업 위험도 |
| compliance | 6 | 준법 멘트, 금칙어, PII 마스킹 |
| claims | 4 | 청구 절차, 서류, 계약관리 |
| customer_db | 3 | 고객 검색, 계약 조회 |
| rag_tools | 2 | 약관/요약서 RAG 검색 |

---

## 기술 스택

| 카테고리 | 기술 | 역할 |
|----------|------|------|
| LLM 오케스트레이션 | [LangGraph](https://langchain-ai.github.io/langgraph/) | ReAct 그래프, 멀티턴, 조건부 분기 |
| LLM | [OpenRouter](https://openrouter.ai/) (qwen/qwen3-14b) | 다중 모델 라우팅 |
| 벡터 DB | [ChromaDB](https://www.trychroma.com/) | 도구 라우팅 + RAG 검색 |
| 임베딩 | [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) (1024차원) | 한국어 비대칭 검색 |
| API 서버 | [FastAPI](https://fastapi.tiangolo.com/) | REST + SSE 스트리밍 |
| MCP 서버 | [FastMCP](https://github.com/jlowin/fastmcp) | Claude Desktop/Cursor 연동 |
| 체크포인터 | langgraph-checkpoint-sqlite | 대화 상태 영구 저장 |
| PDF 파싱 | [PyMuPDF](https://pymupdf.readthedocs.io/) | 약관/요약서 텍스트 추출 |
| 고객 DB | SQLite3 | 고객/계약 시뮬레이션 |

---

## References

| 주제 | 출처 |
|------|------|
| 도구 수 증가 시 정확도 저하 | [How many tools can an AI Agent have?](https://achan2013.medium.com/how-many-tools-functions-can-an-ai-agent-has-21e0a82b7847) |
| RAG-MCP: 도구 검색 후 LLM에 전달 | [When too many tools become too much context — WRITER](https://writer.com/engineering/rag-mcp/) |
| 벡터 DB 비교 (ChromaDB, FAISS, Milvus 등) | [Best Vector Databases 2026 — Firecrawl](https://www.firecrawl.dev/blog/best-vector-databases) |
| 벡터 DB 기능 비교표 | [Top 7 Vector Databases — DataCamp](https://www.datacamp.com/blog/the-top-5-vector-databases) |
| 한국어 IR 벤치마크 (Kor-IR) | [Kor-IR: Korean Information Retrieval Benchmark](https://github.com/Atipico1/Kor-IR) |
| multilingual-e5-large 모델 카드 (Mr. TyDi) | [intfloat/multilingual-e5-large — Hugging Face](https://huggingface.co/intfloat/multilingual-e5-large) |
| E5 임베딩 논문 (비대칭 프리픽스) | [Multilingual E5 Text Embeddings — arXiv 2402.05672](https://arxiv.org/abs/2402.05672) |
| Multi-vector retrieval 원리 | [Cascading retrieval with multi-vector representations — Pinecone](https://www.pinecone.io/blog/cascading-retrieval-with-multi-vector-representations/) |
| Late Interaction 모델 개요 (ColBERT) | [Late Interaction Retrieval Models — Weaviate](https://weaviate.io/blog/late-interaction-overview) |
| Query Rewriting / Advanced RAG 기법 | [RAG for LLMs — Prompting Guide](https://www.promptingguide.ai/research/rag) |
| LangGraph 공식 문서 | [LangGraph Overview — LangChain](https://langchain-ai.github.io/langgraph/) |
