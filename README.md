---
title: Insurance Chatbot
emoji: 🏥
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# AI TMR Assistant — Intelligent Tool Routing

> 54개 도구를 운용하는 보험 AI 챗봇에서, **필요한 도구만 골라 전달하는 검색 기반 라우팅**으로 정확도·비용·지연을 동시에 개선함

---

## 1. 문제

12개 보험 상품 × 9개 기능(조회·산출·심사·보장·청구 등) = **54개 도구**.
도구가 많아지면서 세 가지 문제가 생김.

| 문제 | 원인 | 영향 |
|------|------|------|
| 오호출 | 유사 도구 혼동 (premium_estimate ↔ plan_options) | 잘못된 답변 |
| 비용 증가 | 매 요청마다 54개 스키마가 LLM 컨텍스트에 포함됨 | 토큰 낭비 |
| 지연 증가 | 컨텍스트 길이에 비례하여 응답 시간이 상승함 | UX 저하 |

도구 10개를 넘으면 정확도가 떨어지고, 37개 기준 ~6,200 토큰이 소비됨 [(참고)](https://achan2013.medium.com/how-many-tools-functions-can-an-ai-agent-has-21e0a82b7847).
"전부 넘기지 말고 필요한 것만 검색해서 넘기자" — 이것이 RAG-MCP 패턴의 핵심임 [(참고)](https://writer.com/engineering/rag-mcp/).

---

## 2. 해결 전략

### 2-1. 3단계 필터링으로 54개를 5개로 줄임

```
사용자 질문
    ▼
[Guardrail] ── 비보험 질문 차단 ──→ 거절 응답
    │ (통과)
    ▼
[Tool Search] ── ChromaDB 벡터 검색 → 54개 중 Top-5 추출
    ▼
[LLM Tool Call] ── 5개 후보에서 최종 선택 → 실행
```

| 단계 | 모듈 | 기능 | 속도 |
|------|------|------|------|
| Guardrail | 정규식(L1) + 임베딩(L2) | 탈옥·비보험 질문 사전 차단 | <5ms |
| Tool Search | ChromaDB 멀티벡터 | 54개 → Top-K 후보 추출 | ~10ms |
| LLM Selection | bind_tools() | 후보 중 최종 도구 선택·호출 | 1~5s |

Guardrail이 먼저 동작하므로 "오늘 날씨 어때?" 같은 질문은 벡터 검색이나 LLM 호출 없이 즉시 차단됨.

### 2-2. Tool Card로 검색 정확도를 높임

LLM 도구 description은 보통 한두 줄이라, 유사 도구끼리 벡터가 거의 같아져서 구분이 어려움.
이를 해결하기 위해 도구마다 **ToolCard**를 작성하여 임베딩 표면을 확장함.

```python
ToolCard(
    name="premium_estimate",
    purpose="나이·성별을 입력해 특정 상품의 예상 월 보험료를 산출한다.",
    when_to_use=("보험료 얼마야?", "40세 남성 보험료 계산해줘"),
    when_not_to_use=("납입 플랜이 궁금하다 → plan_options 사용",),
    tags=("보험료", "산출"),
)
```

| 필드 | 임베딩 포함 | 역할 |
|------|:-----------:|------|
| `purpose` | O | 도구의 핵심 기능을 한 문장으로 설명 |
| `when_to_use` | O | 실제 사용자 발화 예시 → 검색 표면 확장 |
| `tags` | O | 도메인 키워드 → 클러스터링 보조 |
| `when_not_to_use` | **X** | 혼동 가능한 도구 안내 → LLM 최종 선택 시에만 사용 |

`when_not_to_use`를 임베딩에서 제외하는 이유: 타 도구 이름("premium_estimate 사용")이 포함되면 벡터가 오염됨.
Tool-DE [(Lu et al., 2025)](https://arxiv.org/abs/2510.22670) ablation에서도 negative example 포함 시 성능이 저하됨.

각 필드는 **별도 문서**로 ChromaDB에 인덱싱하고, 검색 시 tool별 max score로 집계함.
단일 벡터로 합치면 여러 예시의 평균으로 희석되지만, 이 방식은 ColBERT 등 multi-vector 모델과 동일한 원리로 희석 없이 정확한 매칭이 가능함 [(Pinecone)](https://www.pinecone.io/blog/cascading-retrieval-with-multi-vector-representations/).

### 2-3. 코드 변경 없이 운영 중 튜닝

Admin Dashboard(`/admin/tools`)에서 ToolCard를 수정하면 즉시 챗봇에 반영됨.
배치 Recall 평가와 LLM 분석을 내장하여, 수정 전후 성능 차이를 바로 확인할 수 있음.

```
Admin UI에서 when_to_use 문장 수정
    ↓  [저장 & 반영] 클릭
① 메모리 REGISTRY 업데이트 + ChromaDB 재인덱싱
② data/toolcard_overrides.json 저장 (서버 재시작 후에도 유지)
③ 버전 이력 기록 → 문제 시 롤백 가능
```

---

## 3. 결과

79개 테스트 쿼리(tool-call 64개 + no-call 15개) 평가 결과:

| 지표 | k=1 | k=3 | **k=5 (운영)** | k=7 | k=10 |
|------|-----|-----|----------------|-----|------|
| **Recall@k** | 96.9% | 100% | **100%** | 100% | 100% |
| **Hit@1** | 96.9% | — | **96.9%** | — | 96.9% |
| **MRR** | 0.969 | 0.984 | **0.984** | — | 0.984 |
| **No-Call Acc** | — | — | **80.0%** | — | 80.0% |

- k=1 미탐 2건: 유사 도구 경계 사례 (coverage_detail ↔ benefit_amount, renewal_projection ↔ renewal_notice)
- k=3부터 Recall 100% — 64개 tool-call 쿼리가 전부 Top-3 안에 포함됨
- no-call 오판 3건: 유사도 0.86~0.88로 경계에 걸리지만, Guardrail에서 사전 차단되므로 실운영에서는 Tool Search에 도달하지 않음

**54개 → 5개로 90% 축소해도 Recall@5 = 100%, MRR = 0.98.
정확도를 유지하면서 비용과 지연을 동시에 줄임.**

---

## 4. 구현 상세

### 4-1. LangGraph 5노드 파이프라인

```
START → [input_guardrail] → [query_rewriter] → [agent ↔ tools] → [output_guardrail] → END
```

| 노드 | 역할 | 소요 시간 |
|------|------|-----------|
| input_guardrail | 정규식(L1) + 임베딩(L2)으로 이상 요청 차단 | <5ms |
| query_rewriter | "그거 얼마야?" 같은 후속질문을 이전 맥락으로 재작성 | 0~1s |
| agent | ChromaDB Top-K 검색 → LLM 도구 호출 | 1~5s |
| tools | ToolRegistry 동적 디스패치 → 도구 실행 | 10~100ms |
| output_guardrail | PII·금칙어 검사 + 면책 문구 자동 추가 | <2ms |

쿼리 재작성은 Advanced RAG 핵심 기법인 Query Transformation에 해당함 [(참고)](https://www.promptingguide.ai/research/rag).
짧은 후속질문의 벡터 검색 정확도를 보완하는 역할임.

### 4-2. 상품공시실 PDF RAG

보험 상품공시실에서 12개 상품요약서 + 표준약관 + 회사 정보를 수집함.
PyMuPDF로 텍스트 추출 → 500자 청크 → ChromaDB 인제스트(~1,400 벡터).
도구 데이터에 없는 약관 조항·면책 규정을 RAG가 보완함.

### 4-3. LLM 연동

| 항목 | 설명 |
|------|------|
| 시스템 프롬프트 | 12개 상품 목록이 PRODUCTS 딕셔너리에서 동적 반영됨. 도구 체이닝 규칙도 포함 |
| 사고과정 필터링 | Qwen3 `<think>` 블록을 스트리밍 중 실시간 필터링. 사용자에게 최종 답변만 노출 |
| 도구 레벨 가드 | 나이/성별 등 필수값 미제공 시 `needs_user_input` 반환 → LLM이 되묻는 구조 |

### 4-4. 서빙

| 방식 | 설명 | 대상 |
|------|------|------|
| FastAPI (REST/SSE) | 웹 Chat UI + REST API + Admin Dashboard | 일반 사용자·운영자 |
| MCP Server (SSE/stdio) | 도구 54 + 리소스 17 + 프롬프트 8 노출 | Claude Desktop, Cursor 등 |

```bash
python run.py                  # FastAPI → http://localhost:8080
python run_mcp.py              # MCP Server
python run_mcp.py --inspect    # MCP Inspector UI → http://localhost:5173
```

---

## 5. 운영

### 5-1. 도구 추가 체크리스트

새 도구를 추가할 때 아래 순서를 따름.

**① 도구 함수 작성** — `app/tools/` 아래 해당 모듈에 `@tool` 함수를 추가함. `tool.name`이 이후 모든 연동의 키가 됨.

**② ToolCard 등록** — `app/tool_search/tool_cards.py`의 `_CARDS` 리스트에 카드를 추가함.

| 필드 | 규칙 |
|------|------|
| `name` | tool.name과 정확히 일치해야 함 |
| `purpose` | 한 문장으로 도구 기능을 설명 |
| `when_to_use` | 실제 사용자 발화 패턴으로 작성 |
| `when_not_to_use` | 혼동 도구를 `→ tool_name 사용` 형식으로 명시 |
| `tags` | 도메인 키워드 (필터링용) |

> `when_to_use`가 다른 카드와 중복되면 임베딩이 충돌함. `validate_duplicate_when_to_use()`가 자동 검출하므로 평가 시 확인할 것.

**③ 혼동 쌍 관리** — 기능이 유사한 도구가 있으면 양방향으로 처리함.

```
1. 새 카드 when_not_to_use에 기존 유사 도구 언급
2. 기존 유사 도구 when_not_to_use에 새 도구 언급
3. CONFUSION_PAIRS 리스트에 (기존, 신규) 쌍 등록
```

**④ 검증**

```bash
python -m scripts.eval_tool_recall --compare   # Recall@k, MRR 확인
python -m scripts.eval_tool_recall --verbose    # 오판 사례 상세
```

| 연동 지점 | 자동/수동 | 비고 |
|-----------|:---------:|------|
| ChromaDB 임베딩 | 자동 | 서버 시작 시 해시 비교 → 변경분만 재인덱싱 |
| LLM tool description | 자동 | `when_not_to_use`가 description에 자동 주입 |
| 도구 함수 | **수동** | 카드만 있고 함수가 없으면 동작하지 않음 |
| `CONFUSION_PAIRS` | **수동** | 유사 도구 존재 시 반드시 등록 |

> ToolCard가 없는 도구는 `tool.description` 단일 문서로 fallback 됨. 동작은 하지만 검색 정확도가 낮음.

### 5-2. Admin Dashboard

CLI 대신 브라우저(`/admin/tools`)에서 도구 레지스트리를 관리하는 웹 UI임.
FastAPI 앱에 내장되어 별도 서버 없이 HF Spaces 등 배포 환경에서도 동작함.

| 기능 | 설명 |
|------|------|
| 대시보드 | 도구 수, ChromaDB 벡터 수, Registry 버전, 상태 모니터링 |
| ToolCard 편집 | purpose, when_to_use, when_not_to_use, tags 수정 → 즉시 반영 |
| 버전 이력 | 변경 이력 조회, Diff 비교, 특정 버전으로 롤백 |
| 퀵 테스트 | 실시간 쿼리 검색, 배치 Recall 평가, LLM 실패 분석 |
| 도구 해제 | 확인 모달 → DELETE → ChromaDB 벡터 즉시 삭제 |
| 모듈 핫리로드 | 8개 모듈 선택 → 코드 변경분 서버 재시작 없이 반영 |

퀵 테스트 탭에서 수정 전후 Recall@k를 비교하고, 실패 쿼리에 대해 LLM이 ToolCard 개선안을 제안하므로 **코드 변경 없이** 검색 정확도를 튜닝할 수 있음.

### 5-3. 런타임 API

```bash
curl http://localhost:8080/api/tools                              # 전체 도구 목록
curl -X DELETE http://localhost:8080/api/tools/premium_estimate    # 도구 해제
curl -X POST http://localhost:8080/api/tools/reload-module/premium # 모듈 핫리로드
```

ToolRegistry가 변경을 감지하고 ChromaDB 재인덱싱을 자동 트리거함.

---

## 6. 기술 선택 근거

### ChromaDB

| 기준 | FAISS | Milvus | **ChromaDB** |
|------|-------|--------|-------------|
| 메타데이터 필터링 | X | O | **O** |
| 영속성 | X | O | **O** |
| 실시간 upsert | rebuild 필요 | O | **O** |
| 인프라 | 없음 | Docker 3개 | **pip 1줄** |

~1,800 벡터 규모에서 Milvus는 오버엔지니어링, FAISS는 메타데이터 필터링 미지원.
10M 벡터 미만에서 ChromaDB 권장 [(Firecrawl)](https://www.firecrawl.dev/blog/best-vector-databases) [(DataCamp)](https://www.datacamp.com/blog/the-top-5-vector-databases).

### multilingual-e5-large

[Kor-IR 벤치마크](https://github.com/Atipico1/Kor-IR) 오픈소스 최상위(NDCG@10 = 80.35).
Mr. TyDi 한국어 MRR@10 = 61.6으로 e5-base(55.8) 대비 +10% [(모델 카드)](https://huggingface.co/intfloat/multilingual-e5-large).
비대칭 검색 시 `"query: "` / `"passage: "` 프리픽스 필수 [(논문)](https://arxiv.org/abs/2402.05672).
로컬 추론(~10ms/쿼리)으로 외부 API에 의존하지 않음.

### Tool Card 학술 근거

| 출처 | 핵심 기여 |
|------|-----------|
| [Tool-DE (Lu et al., 2025)](https://arxiv.org/abs/2510.22670) | 도구 문서 확장으로 NDCG@10 +6~7ppt, Recall@10 +10ppt |
| [Re-Invoke (Google, EMNLP 2024)](https://arxiv.org/abs/2408.01875) | 합성 쿼리 생성으로 nDCG@5 유의미 향상 |
| [ToolBench (ICLR 2024)](https://arxiv.org/abs/2307.16789) | 도구 수 증가 시 cross-reference가 정확도 저하를 방지 |
| [RAG-MCP (WRITER, 2025)](https://writer.com/engineering/rag-mcp/) | 메타데이터 기반 인덱싱 → 토큰 50%+ 절감 |

---

## 7. 알려진 한계 및 고도화 방향

| 한계 | 현상 | 방향 |
|------|------|------|
| when_to_use 오버핏 | product_search에 타 도구 영역 발화가 임베딩을 희석함 | 순수 상품 검색 발화만 유지 |
| cross-reference 누락 | renewal_projection ↔ renewal_notice 양방향 가이드 부재 | when_not_to_use 양방향 보완 |
| 수동 작성 한계 | 54개 × ~7개 = ~380개 when_to_use를 수동 관리 중 | Re-Invoke 방식 LLM 합성 쿼리 자동 생성 |
| 정적 no-call 임계값 | tool-call min(0.867)과 no-call max(0.877)이 겹침 | Reranker 2단계 도입 |

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
│   ├── tool_cards.py       #   54개 ToolCard
│   └── toolcard_store.py   #   ToolCard JSON 영속화 + 버전 이력
├── rag/                    # 상품공시실 PDF RAG
│   ├── retriever.py        #   인제스트 + 검색
│   └── splitter.py         #   한국어 문장경계 청크 분할
└── mcp_server/             # MCP 프로토콜 서버 + Inspector

templates/
├── index.html              # 챗봇 Chat UI (상품 카탈로그, 시나리오)
└── admin_tools.html        # Tool Admin Dashboard

scripts/
├── init_vectordb.py        # ChromaDB 초기화
└── eval_tool_recall.py     # Recall@k / MRR 평가
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
| 임베딩 | [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) (1024d) | 한국어 비대칭 검색 |
| API 서버 | [FastAPI](https://fastapi.tiangolo.com/) | REST + SSE 스트리밍 |
| MCP 서버 | [FastMCP](https://github.com/jlowin/fastmcp) | Claude Desktop/Cursor 연동 |
| PDF 파싱 | [PyMuPDF](https://pymupdf.readthedocs.io/) | 약관/요약서 텍스트 추출 |
| 고객 DB | SQLite3 | 고객/계약 시뮬레이션 |

---

## References

| 주제 | 출처 |
|------|------|
| 도구 수 증가 시 정확도 저하 | [How many tools can an AI Agent have?](https://achan2013.medium.com/how-many-tools-functions-can-an-ai-agent-has-21e0a82b7847) |
| RAG-MCP: 도구 검색 후 LLM에 전달 | [WRITER Engineering](https://writer.com/engineering/rag-mcp/) |
| Tool Document Expansion (Tool-DE) | [arXiv 2510.22670](https://arxiv.org/abs/2510.22670) |
| 합성 쿼리 기반 도구 검색 (Re-Invoke) | [EMNLP 2024](https://arxiv.org/abs/2408.01875) |
| 대규모 도구 벤치마크 (ToolBench) | [ICLR 2024](https://arxiv.org/abs/2307.16789) |
| 벡터 DB 비교 | [Firecrawl](https://www.firecrawl.dev/blog/best-vector-databases) · [DataCamp](https://www.datacamp.com/blog/the-top-5-vector-databases) |
| 한국어 IR 벤치마크 (Kor-IR) | [GitHub](https://github.com/Atipico1/Kor-IR) |
| multilingual-e5-large | [Hugging Face](https://huggingface.co/intfloat/multilingual-e5-large) · [arXiv 2402.05672](https://arxiv.org/abs/2402.05672) |
| Multi-vector retrieval | [Pinecone](https://www.pinecone.io/blog/cascading-retrieval-with-multi-vector-representations/) |
| Query Rewriting / Advanced RAG | [Prompting Guide](https://www.promptingguide.ai/research/rag) |
| LangGraph 공식 문서 | [LangChain](https://langchain-ai.github.io/langgraph/) |
