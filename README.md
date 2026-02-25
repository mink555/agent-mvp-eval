---
title: Insurance Chatbot
emoji: 🏥
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# AI TMR Assistant — Tool Routing 고도화

> **Intelligent Tool Routing**으로 정확도를 개선하고, **Scalable Tool Architecture**로 운영 효율과 확장성을 동시에 확보함

---

## 1. 무엇을 해결하려 했는가

12개 보험 상품 × 9개 기능(조회·산출·심사·보장·청구·컴플라이언스 등) = **54개 도구**를 운용하는 AI 챗봇.
도구가 많아지면서 세 가지 문제가 발생함.

| 문제 | 원인 | 영향 |
|------|------|------|
| 오호출 | 유사 도구 혼동 (premium_estimate ↔ plan_options) | 잘못된 답변 |
| 비용 증가 | 매 요청마다 54개 스키마가 LLM 컨텍스트에 포함 | 토큰 낭비 |
| 지연 증가 | 컨텍스트 길이에 비례한 응답 시간 상승 | UX 저하 |

> 도구 10개 초과 시 정확도가 저하되고, 37개 기준 ~6,200 토큰 소비 [(참고)](https://achan2013.medium.com/how-many-tools-functions-can-an-ai-agent-has-21e0a82b7847).
> "전부 넘기지 말고, 필요한 것만 검색해서 넘기자" — RAG-MCP 패턴 [(참고)](https://writer.com/engineering/rag-mcp/)

---

## 2. 어떻게 해결했는가

### 전략 A. Intelligent Tool Routing — 54개를 5개로 줄임

Guardrail → Tool Search → LLM 선택, 3단계 필터링으로 정확한 도구만 LLM에 전달함.

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
| Guardrail | 정규식(L1) + 임베딩(L2) | 탈옥·비보험 질문 차단 | <5ms |
| Tool Search | ChromaDB 멀티벡터 | 54개 → Top-K 후보 추출 | ~10ms |
| LLM Selection | bind_tools() | 후보 중 실제 필요한 도구만 호출 | 1~5s |

**핵심:** Guardrail이 먼저 동작하므로 "오늘 날씨 어때?" 같은 질문은 벡터 검색·LLM 호출 없이 즉시 차단됨.

### 전략 B. Scalable Tool Architecture — 도구 추가가 운영 부담이 되지 않도록

```
새 Tool 추가 → Tool Card 작성 → 임베딩 자동 생성 → 즉시 검색 대상에 포함
```

| 방식 | 절차 | 서버 재시작 |
|------|------|-------------|
| 정적 등록 | Tool 함수 + ToolCard → 서버 재시작 | 필요 |
| 런타임 핫리로드 | Tool 함수 + API 호출 (`POST /api/tools/reload-module/{module}`) | **불필요** |

ToolRegistry가 동적 관리하고, 변경 시 ChromaDB 재인덱싱을 자동 트리거함.

### 전략 C. Validation — 감이 아니라 숫자로 판단

`scripts/eval_tool_recall.py`로 Recall@k, MRR, Hit@1을 정량 측정함.

```bash
python -m scripts.eval_tool_recall --compare    # k=1,3,5,7,10 비교표
python -m scripts.eval_tool_recall --verbose     # 오판 사례 상세
```

---

## 3. 결과

79개 테스트 쿼리(tool-call 64개 + no-call 15개) 평가 결과:

| 지표 | k=1 | k=3 | **k=5 (운영)** | k=7 | k=10 |
|------|-----|-----|----------------|-----|------|
| **Recall@k** | 96.9% | 100% | **100%** | 100% | 100% |
| **Hit@1** | 96.9% | 96.9% | **96.9%** | 96.9% | 96.9% |
| **MRR** | 0.969 | 0.984 | **0.984** | 0.984 | 0.984 |
| **No-Call Acc** | 80.0% | 80.0% | **80.0%** | 80.0% | 80.0% |

| 점수 분포 (k=10) | min | avg | max |
|-------------------|-----|-----|-----|
| Tool-Call top-1 | 0.867 | 0.921 | 0.947 |
| No-Call top-1 | 0.831 | 0.853 | 0.877 |

- k=1 미탐 2건: 유사 도구 간 경계 사례 (coverage_detail ↔ benefit_amount, renewal_projection ↔ renewal_notice)
- k=3부터 Recall 100%. 64개 tool-call 쿼리 전부 Top-3 안에 정답 도구 포함
- no-call 오판 3건: 비보험 질문이지만 유사도 0.86~0.88로 경계에 걸림. Guardrail(L1+L2)에서 사전 차단되므로 실운영에서는 Tool Search 미도달

**결론:** 54개 → 5개로 90% 축소해도 **Recall@5 = 100%, MRR = 0.98**. 정확도를 유지하면서 비용과 지연을 동시에 줄임.

---

## 4. 구현 상세

### 4-1. 5노드 파이프라인 (LangGraph)

```
START → [input_guardrail] → [query_rewriter] → [agent ↔ tools] → [output_guardrail] → END
```

| 노드 | 역할 | 소요 시간 |
|------|------|-----------|
| input_guardrail | 정규식(L1) + 임베딩(L2)으로 이상 요청 차단 | <5ms |
| query_rewriter | 짧은 후속질문을 이전 맥락으로 재작성 | 0~1s |
| agent | ChromaDB Top-K 필터링 → LLM 호출 | 1~5s |
| tools | ToolRegistry 동적 디스패치 | 10~100ms |
| output_guardrail | PII·금칙어 검사 + 면책 문구 자동 추가 | <2ms |

### 4-2. 쿼리 재작성 (Query Rewriter)

"그거 얼마야?", "그건?" 같은 짧은 후속질문은 벡터 검색 정확도가 떨어짐. 이전 대화 맥락을 참조해 구체적 쿼리로 재작성하여 Tool Search 정확도를 보완함. Query Transformation은 Advanced RAG 핵심 기법 [(참고)](https://www.promptingguide.ai/research/rag).

### 4-3. 상품공시실 PDF 기반 RAG

보험 상품공시실에서 12개 상품요약서 PDF + 표준약관 + 회사 정보를 수집. PyMuPDF로 텍스트 추출 → 500자 청크 → ChromaDB 인제스트(~1,400 벡터). 도구 데이터에 없는 약관 조항·면책 규정을 RAG가 보완함.

### 4-4. Agentic 시스템 프롬프트

12개 상품 목록이 PRODUCTS 딕셔너리에서 동적 반영. 새 상품 추가 시 프롬프트가 자동 업데이트됨. 도구 체이닝 규칙("상품명만 알면 product_search → 해당 도구 순서")도 포함하여 LLM이 자율적으로 연쇄 호출함.

### 4-5. LLM 사고과정 필터링

Qwen3의 `<think>...</think>` 블록을 스트리밍 중 실시간 필터링. 사용자에게는 최종 답변만 노출, SSE 이벤트로 파이프라인 진행 상태를 표시하여 체감 지연을 줄임.

### 4-6. 도구 레벨 입력 가드

보험료 산출·가입 심사 등 나이/성별이 필수인 도구는, 사용자가 제공하지 않은 정보를 추측하지 않음. 도구가 `needs_user_input`을 반환하면 LLM이 해당 정보를 질문함.

### 4-7. 상품 카탈로그 UI

헤더의 "상품 목록" 버튼 또는 페이지 접속 시 자동으로 12개 상품 카탈로그 모달이 표시됨. 카테고리·갱신유형·간편심사 태그로 필터링하고, 상품 클릭 시 보장 내용 질문이 자동 세팅됨. 모바일 반응형 대응.

### 4-8. 서빙: 두 가지 인터페이스

| 방식 | 설명 | 대상 |
|------|------|------|
| FastAPI (REST/SSE) | 웹 Chat UI + REST API | 일반 사용자 |
| MCP Server (SSE/stdio) | 도구 54 + 리소스 17 + 프롬프트 8 노출 | Claude Desktop, Cursor 등 |

MCP Inspector UI로 도구 입출력, 리소스 조회, 프롬프트 렌더링을 브라우저에서 직접 테스트 가능.

```bash
python run_mcp.py --inspect
```

### 4-9. 도구 추가 체크리스트

새 도구를 추가할 때 아래 4단계를 순서대로 수행한다.

**① 도구 함수 작성** — `app/tools/` 아래 해당 모듈에 `@tool` 함수를 추가한다. 함수의 `tool.name`이 이후 모든 연동의 키가 된다.

**② ToolCard 등록** — `app/tool_search/tool_cards.py`의 `_CARDS` 리스트에 카드를 추가한다.

| 필드 | 규칙 | 예시 |
|------|------|------|
| `name` | tool.name과 **정확히** 일치 | `"premium_estimate"` |
| `purpose` | 한 문장으로 명확하게 | `"예상 월 보험료를 산출한다."` |
| `when_to_use` | **실제 사용자 발화** 패턴으로 작성 | `("보험료 얼마야?", "40세 남성 보험료")` |
| `when_not_to_use` | 혼동 도구명을 `→ tool_name 사용` 형식으로 명시 | `("납입 플랜 → plan_options 사용",)` |
| `tags` | 도메인 키워드 (필터링용) | `("보험료", "산출")` |

> `when_to_use`가 다른 도구 카드와 **중복되면 임베딩이 충돌**한다. `validate_duplicate_when_to_use()`가 자동 검출하므로 평가 스크립트를 반드시 실행할 것.

ToolCard 설계는 Tool Document Expansion [(Tool-DE, Lu et al. 2025)](https://arxiv.org/abs/2510.22670) 연구에 기반한다. purpose·when_to_use·tags로 임베딩 표면을 확장하고, when_not_to_use는 LLM description에만 주입하여 벡터 오염을 방지한다. Re-Invoke [(Google, EMNLP 2024)](https://arxiv.org/abs/2408.01875)의 합성 쿼리 전략과 동일 원리이며, ablation 결과 negative example을 임베딩에서 제외할 때 NDCG가 가장 높았다.

**③ 혼동 쌍 관리** — 기능이 유사한 도구가 있으면 양방향으로 처리한다.

```
1. 새 카드의 when_not_to_use에 기존 유사 도구 언급
2. 기존 유사 도구의 when_not_to_use에 새 도구 언급
3. CONFUSION_PAIRS 리스트에 (기존, 신규) 쌍 등록
```

`validate_confusion_pairs()`가 양방향 누락을 검출한다. 유사 도구 간 명시적 cross-reference는 ToolBench [(ICLR 2024)](https://arxiv.org/abs/2307.16789)에서 도구 수 증가 시 정확도 저하를 방지하는 핵심 전략으로 제시되었다.

**④ 검증**

```bash
python -m scripts.eval_tool_recall --compare   # Recall@k, MRR 확인
python -m scripts.eval_tool_recall --verbose    # 오판 사례 상세
```

**연동 자동/수동 요약:**

| 연동 지점 | 자동/수동 | 설명 |
|-----------|:---------:|------|
| ChromaDB 임베딩 | 자동 | 서버 재시작 시 해시 비교 → 변경 감지되면 재인덱싱 |
| LLM tool description | 자동 | `when_not_to_use`가 bind_tools() 시 description에 주입 |
| 평가 스크립트 | 자동 | 카드 정합성 검증이 평가 시 자동 실행 |
| 도구 함수 (`app/tools/`) | **수동** | 카드만 있고 실제 함수가 없으면 동작하지 않음 |
| `CONFUSION_PAIRS` | **수동** | 유사 도구가 있을 경우 반드시 등록 |

> ToolCard가 없는 도구는 `tool.description` 단일 문서로 fallback 되어 동작은 하지만 검색 정확도가 낮다. 서버 로그에 `"ToolCard 없는 도구 N개"` 경고가 출력된다.

### 4-10. 런타임 도구 관리 API

서버 재시작 없이 도구를 추가·제거·확인할 수 있는 REST API를 제공한다. ToolRegistry [(동적 레지스트리 패턴)](https://python.langchain.com/docs/how_to/tools_runtime/)가 변경을 감지하고 ChromaDB 재인덱싱을 자동 트리거한다.

```bash
# 전체 도구 목록 조회
curl http://localhost:8080/api/tools

# 특정 도구 런타임 해제 (ChromaDB 벡터도 자동 삭제)
curl -X DELETE http://localhost:8080/api/tools/premium_estimate

# 모듈 단위 핫리로드 (수정한 도구 코드를 서버 재시작 없이 반영)
curl -X POST http://localhost:8080/api/tools/reload-module/premium
```

| API | 메서드 | 기능 |
|-----|--------|------|
| `/api/tools` | GET | 전체 도구 목록 + 메타데이터 |
| `/api/tools/{tool_name}` | DELETE | 도구 해제 + ChromaDB 벡터 삭제 |
| `/api/tools/reload-module/{module}` | POST | 모듈 `importlib.reload()` → 도구 재등록 |

MCP Inspector에서도 도구 입출력을 브라우저에서 직접 테스트할 수 있다.

```bash
python run_mcp.py --inspect    # Inspector UI → http://localhost:5173
```

---

## 5. 기술 선택 근거

### ChromaDB

| 기준 | FAISS | Milvus | **ChromaDB** |
|------|-------|--------|-------------|
| 메타데이터 필터링 | X | O | **O** |
| 영속성 | X | O | **O** |
| 실시간 upsert | rebuild 필요 | O | **O** |
| 인프라 | 없음 | Docker 3개 | **pip 1줄** |

벡터 ~1,800개 규모에서 Milvus는 오버엔지니어링, FAISS는 메타데이터 필터링 미지원. 10M 벡터 미만 프로젝트에서 ChromaDB 권장 [(Firecrawl)](https://www.firecrawl.dev/blog/best-vector-databases) [(DataCamp)](https://www.datacamp.com/blog/the-top-5-vector-databases).

### multilingual-e5-large

[Kor-IR 벤치마크](https://github.com/Atipico1/Kor-IR)에서 오픈소스 최상위(NDCG@10 = 80.35). Mr. TyDi 한국어 MRR@10 = 61.6으로 e5-base(55.8) 대비 +10% [(모델 카드)](https://huggingface.co/intfloat/multilingual-e5-large). 비대칭 검색 시 "query: " / "passage: " 프리픽스 필수 [(E5 논문)](https://arxiv.org/abs/2402.05672). 로컬 추론(~10ms/쿼리)으로 외부 API 미의존.

### Multi-Vector 인덱싱

도구 하나를 단일 벡터로 임베딩하면 여러 사용 예시의 평균으로 벡터가 희석됨. purpose + when_to_use를 별도 문서로 인덱싱하고, 검색 시 tool별 max score로 집계하여 희석 없이 정확한 매칭을 달성. ColBERT 등 multi-vector 모델이 single-vector 대비 정확도가 높은 것과 동일 원리 [(Pinecone)](https://www.pinecone.io/blog/cascading-retrieval-with-multi-vector-representations/).

### Tool Card (Tool Document Expansion)

LLM 도구 description은 보통 한두 줄. 이 짧은 텍스트만 임베딩하면 유사 도구 간 벡터가 거의 같아져 검색 정확도가 떨어짐.

```python
ToolCard(
    name="premium_estimate",
    purpose="나이·성별을 입력해 특정 상품의 예상 월 보험료를 산출한다.",
    when_to_use=("보험료 얼마야?", "40세 남성 보험료 계산해줘"),
    when_not_to_use=("납입 플랜이 궁금하다 → plan_options 사용",),
    tags=("보험료", "산출"),
)
```

| ToolCard 필드 | 학술 대응 | 임베딩 포함 | 역할 |
|---------------|-----------|:-----------:|------|
| `purpose` | Tool-DE의 function_description | O | 도구 핵심 기능 |
| `when_to_use` | Re-Invoke의 synthetic queries | O | 검색 표면 확장 |
| `tags` | Tool-DE의 tags | O | 도메인 클러스터링 |
| `when_not_to_use` | Tool-DE의 limitations | X | LLM 최종 선택 시 혼동 방지 |

when_not_to_use를 임베딩에서 제외한 이유: 타 도구 어휘("premium_estimate 사용")가 포함되어 벡터가 오염됨. Tool-DE ablation에서도 negative example 포함 시 성능 저하 확인.

**학술 근거:**
- **Tool-DE** (Lu et al., 2025) — 도구 문서 확장으로 NDCG@10 +6~7ppt, Recall@10 +10ppt 개선 [(논문)](https://arxiv.org/abs/2510.22670)
- **Re-Invoke** (Google, EMNLP 2024) — 합성 쿼리 생성으로 nDCG@5 유의미 향상 [(논문)](https://arxiv.org/abs/2408.01875)
- **RAG-MCP** (WRITER, 2025) — 메타데이터 기반 인덱싱 → 토큰 50%+ 절감 [(블로그)](https://writer.com/engineering/rag-mcp/)

---

## 6. 알려진 한계 및 고도화 방향

| 한계 | 현상 | 고도화 방향 |
|------|------|-------------|
| product_search when_to_use 오버핏 | 타 도구 영역 발화 10개가 임베딩을 희석 | 순수 상품 검색 발화만 유지 |
| 유사 도구 cross-reference 누락 | renewal_projection ↔ renewal_notice 등 양방향 가이드 부재 | when_not_to_use 양방향 보완 |
| 수동 작성 한계 | 54개 × 7개 = ~380개 when_to_use 수동 관리 | Re-Invoke 방식 LLM 합성 쿼리 자동 생성 |
| 정적 no-call 임계값 | Tool-Call min(0.867)과 No-Call max(0.877)이 겹침 | Reranker 2단계 도입 (Tool-Rank) |

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
