# Insurance Chatbot — MCP + Intelligent Tool Routing

> LangGraph ReAct + MCP + ChromaDB 기반 **보험 도메인 AI 챗봇**

---

## TL;DR

| 항목 | 내용 |
|------|------|
| **무엇을 하나** | 라이나생명 12개 상품에 대해 상품 조회, 보험료 산출, 가입 심사, 보장 분석, 청구, 컴플라이언스까지 자연어로 응답 |
| **도구 수** | LangChain `@tool` 54개 |
| **핵심 구조** | 사용자 질문 → 가드레일 → ChromaDB로 54개→5개 축소 → LLM이 최종 선택 → 도구 실행 → 출력 검증 |
| **서빙** | FastAPI (REST/SSE, :8080) + MCP Server (SSE/stdio, :8000) |
| **Tool Routing 정확도** | Recall@5 = **100%**, Hit@1 = **98.4%**, MRR = **0.99** (64개 쿼리 기준) |
| **새 도구 추가** | ToolCard 1개 등록 → 코드 변경 없이 자동 인덱싱·검색 대상 포함 |
| **LLM** | OpenRouter (기본: qwen/qwen3-14b) |
| **임베딩** | `intfloat/multilingual-e5-large` (1024차원) |

---

## 목차

1. [프로젝트 개요 — 3가지 설계 전략](#1-프로젝트-개요--3가지-설계-전략)
2. [왜 이렇게 만들었나 — 설계 배경](#2-왜-이렇게-만들었나--설계-배경)
3. [전체 동작 흐름 — 질문에서 답변까지](#3-전체-동작-흐름--질문에서-답변까지)
4. [아키텍처 상세](#4-아키텍처-상세)
5. [핵심 모듈 설명](#5-핵심-모듈-설명)
6. [새 도구 등록 가이드](#6-새-도구-등록-가이드)
7. [Tool Routing 성능 실험](#7-tool-routing-성능-실험)
8. [프로젝트 구조](#8-프로젝트-구조)
9. [API 명세](#9-api-명세)
10. [Quick Start](#10-quick-start)
11. [운영 가이드 — 하드코딩 값 관리](#11-운영-가이드--하드코딩-값-관리)
12. [기술 스택](#12-기술-스택)

---

## 1. 프로젝트 개요 — 3가지 설계 전략

이 프로젝트는 **3가지 핵심 전략**을 중심으로 설계되었습니다.

### 1-1. Intelligent Tool Routing — 정확하게 호출하기

도구가 54개나 되면 LLM이 매번 전부 보고 고르기엔 혼란스럽습니다.  
그래서 질문과 관련 있는 도구만 추려서 LLM에게 넘기는 **2단계 선택 구조**를 만들었습니다.

```
사용자 질문 → [Guardrail] → [ChromaDB 검색: 54개 → ~5개] → [LLM이 최종 선택] → 도구 실행
```

| 단계 | 누가 | 무엇을 |
|------|------|--------|
| **1차** (코드) | ChromaDB 임베딩 검색 | 54개 중 관련 도구 top-k개 추출 |
| **2차** (AI) | LLM `bind_tools()` | 추출된 후보 중 실제 필요한 것만 호출 |

### 1-2. Scalable Tool Architecture — 쉽게 확장하기

새 도구를 추가할 때 운영 부담이 없도록 자동화했습니다.

```
① 도구 함수 작성  →  ② ToolCard 등록  →  ③ 서버 재시작
                                            ↓
                                   임베딩 자동 생성
                                   Vector Index 자동 갱신
                                   검색 대상에 즉시 포함
```

코드를 고칠 필요 없이 **ToolCard 하나만 등록하면** 기존 파이프라인에 녹아듭니다.  
(상세 절차는 [6. 새 도구 등록 가이드](#6-새-도구-등록-가이드) 참조)

### 1-3. Validation Layer — 측정하고 비교하기

감이 아니라 **숫자**로 판단합니다.

- `scripts/eval_tool_recall.py`로 Recall@k, MRR, Hit@1을 측정
- 도구 추가/삭제/임베딩 변경 후 성능 변화를 정량 비교
- 변경 전후 회귀(regression) 방지

### 1-4. 두 가지 서빙 방식

| 방식 | 설명 | 대상 |
|------|------|------|
| **FastAPI** (REST/SSE) | 웹 UI + REST API로 직접 채팅 | 일반 사용자, 프론트엔드 |
| **MCP Server** (SSE/stdio) | MCP 프로토콜로 도구·리소스·프롬프트 노출 | Claude Desktop, Cursor 등 AI 클라이언트 |

---

## 2. 왜 이렇게 만들었나 — 설계 배경

### 2-1. 문제: 도구가 많으면 LLM이 헷갈린다

54개 도구를 전부 LLM에게 넘기면 세 가지 문제가 생깁니다.

| 문제 | 설명 |
|------|------|
| **비용 증가** | 매번 54개 도구 스키마를 컨텍스트에 포함해야 함 |
| **정확도 하락** | 비슷한 이름의 도구를 혼동 (예: `premium_estimate` vs `plan_options`) |
| **지연 증가** | 토큰 수 증가로 응답 시간 상승 |

### 2-2. 해법: 2단계 필터링

1. **ChromaDB로 후보 축소** — 질문 임베딩과 도구 임베딩을 비교해서 관련 도구 top-k만 남김
2. **LLM이 최종 결정** — 축소된 메뉴판에서 실제 필요한 도구만 선택

이 구조 덕분에:
- LLM 컨텍스트가 가벼워져 → **비용과 지연 감소**
- 비관련 도구가 제거되어 → **오호출 감소**
- 새 도구를 추가해도 → 기존 도구 선택 정확도 **유지**

### 2-3. 왜 Multi-Vector 인덱싱인가

각 도구를 **단일 벡터**로 인덱싱하면, 여러 사용 예시의 평균이 되어 벡터가 희석됩니다.

대신 도구별로 `purpose`, `when_to_use` 각각을 **별도 문서**로 인덱싱하고, 검색 시 tool별 **max score**로 집계합니다.

```
도구 1개 → 문서 N개 (purpose 1개 + when_to_use 예시 여러 개 + tags 1개)
검색 시 → 같은 도구의 여러 문서 중 가장 높은 점수 = 그 도구의 최종 점수
```

하나의 `when_to_use` 예시가 쿼리와 잘 맞기만 하면 해당 도구가 상위에 올라옵니다.

---

## 3. 전체 동작 흐름 — 질문에서 답변까지

### 3-1. 파이프라인 5단계

```
사용자 질문
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ ① 입력 안전 검사 (input_guardrail)                    │
│    ├─ L1: 정규식 — 탈옥 시도 차단 (<1ms)              │
│    ├─ L2: 임베딩 — 보험 무관 질문 차단 (~3ms)         │
│    └─ 후속 질문이면 도메인 체크 생략                    │
│                                                       │
│    차단 → 거절 응답 → END                              │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ ② 질문 다듬기 (query_rewriter)                        │
│    "그거 얼마야?" → "암보험 월 보험료 얼마예요?"        │
│    (15자 미만 후속 질문일 때만, 아니면 0ms 패스)        │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ ③ AI 판단 + 도구 실행 (agent ↔ tools, 필요시 반복)    │
│    a. ChromaDB로 관련 도구 top-k 추출 (1차 선택)       │
│    b. LLM.bind_tools(관련 도구) (2차 선택)             │
│    c. LLM이 도구 호출 → 결과 수집 → 부족하면 반복      │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ ④ 출력 안전 검사 (output_guardrail)                   │
│    ├─ PII 노출 검사 (주민번호·전화번호 등)             │
│    ├─ 금칙어·과장표현 검사                              │
│    ├─ 빈 응답 검사                                     │
│    ├─ 차단 → LLM에게 재작성 요청 (최대 1회)            │
│    └─ 통과 → 면책 문구 자동 추가                       │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ ⑤ 최종 답변 반환                                      │
│    answer + tools_used + trace                        │
└─────────────────────────────────────────────────────┘
```

### 3-2. 실제 시나리오

```
사용자: "45세 남성인데 암보험 가입 가능한가요? 보험료는 얼마쯤 하나요?"

[① input_guardrail] → 통과 (보험 관련 질문, 235ms)
[② query_rewriter]  → 패스 (충분히 긴 질문)
[③ agent]           → product_search 호출 결정 → 암보험 2개 발견
   [tools]          → product_search 실행
   [agent]          → premium_estimate(B00115023, 45, M)
                      premium_estimate(B00355005, 45, M)
   [tools]          → 보험료 산출 실행
   [agent]          → 결과 종합하여 답변 생성
[④ output_guardrail] → 통과 + 면책 문구 자동 추가
[⑤ 반환]

응답: "현재 가입 가능한 암보험은 2개입니다.
       • 뉴스타트암보험(B00115023): 월 20,625원 (90일 면책)
       • 첫날부터 암보험(B00355005): 월 15,600원 (면책 없음)

       ※ 이 금액은 예시이며, 실제 보험료는 상품·보장내용·건강상태에
          따라 달라집니다. 정확한 보험료는 설계사 상담을 통해 확인해 주세요."
```

---

## 4. 아키텍처 상세

### 4-1. LangGraph 5노드 그래프

```
START → [input_guardrail] ──(차단)──→ END
              │(통과)
              ▼
        [query_rewriter]
              │
              ▼
          [agent] ◄──────── [tools]
              │                 ▲
              └─(도구 호출)────┘
              │(최종 답변)
              ▼
      [output_guardrail] ──(재시도)──→ [agent]
              │(통과/차단)
              ▼
             END
```

| # | 노드 | LLM | 소요 시간 | 역할 |
|---|------|-----|-----------|------|
| 1 | `input_guardrail` | X | <5ms | 정규식 + 임베딩 도메인 검증 |
| 2 | `query_rewriter` | 조건부 | 0ms~1s | 짧은 후속질문 맥락 재작성 |
| 3 | `agent` | O | 1~5s | ChromaDB 도구 라우팅 + LLM 호출 |
| 4 | `tools` | X | 10~100ms | 선택된 도구 실행 |
| 5 | `output_guardrail` | X | <2ms | PII·금칙어 검사 + 면책 문구 주입 |

### 4-2. 3계층 가드레일

| # | 레이어 | 방식 | 지연 | 담당 |
|---|--------|------|------|------|
| L1 | 정규식 | 10개 패턴 매칭 | <1ms | 탈옥·시스템 조작 시도 |
| L2 | 임베딩 | e5 + numpy 코사인 | ~3ms | 비보험 도메인 차단 |
| L3 | LLM | 시스템 프롬프트 | — | 모호한 케이스 최종 판단 |

**L2는 ChromaDB를 쓰지 않습니다.** 앱 시작 시 보험/비보험 예시 46개를 메모리(numpy 배열)에 올려두고, 매 요청마다 행렬 곱 한 번으로 끝냅니다.

```
[앱 시작 시 — 1회]
  보험 예시 26개 → e5 "passage:" → numpy (26, 1024) ← 메모리 상주
  비보험 예시 20개 → e5 "passage:" → numpy (20, 1024) ← 메모리 상주

[매 요청 — ~3ms]
  사용자 입력 → e5 "query:" → 벡터 1개 생성
  → max_in  = 보험 예시 26개 중 최대 유사도
  → max_out = 비보험 예시 20개 중 최대 유사도
  → max_in >= 0.87           → 즉시 통과 (확실한 보험 질문)
  → max_out - max_in >= 0.03 → 차단 (비보험이 확실히 우세)
  → 나머지                    → 통과 (LLM에 위임)
```

### 4-3. 전체 구성도

```
┌──────────────────────────────────────────────────────────────┐
│                        클라이언트                              │
│   웹 UI (Chat)  /  REST Client  /  MCP Client (Claude 등)    │
└──────┬─────────────────────┬─────────────────────────────────┘
       ▼                     ▼
┌───────────────┐    ┌────────────────────┐
│ FastAPI :8080 │    │ MCP Server :8000   │
│ REST + SSE    │    │ SSE / stdio        │
└──────┬────────┘    └─────────┬──────────┘
       └───────────┬───────────┘
                   ▼
┌──────────────────────────────────────────────────────────┐
│          LangGraph (ReAct + 가드레일, 5노드)              │
│                                                          │
│  ┌──────────────────────────────────┐                    │
│  │ ChromaDB Tool Routing Index     │  54개 → top-k 축소 │
│  └──────────────────────────────────┘                    │
└─────────────────────┬────────────────────────────────────┘
                      │
      ┌───────────────┼───────────────┐
      ▼               ▼               ▼
┌───────────┐  ┌────────────┐  ┌──────────────┐
│ 54개 도구  │  │ ChromaDB   │  │ SQLite DB    │
│ (LangChain)│  │ RAG 검색   │  │ 고객/계약     │
│ 상품·보험료│  │ 약관·요약서 │  │              │
│ 보장·심사 │  │            │  │              │
│ 준법·청구 │  │            │  │              │
└───────────┘  └────────────┘  └──────────────┘
```

### 4-4. 프레임워크 역할 분담

| # | 프레임워크 | 역할 |
|---|-----------|------|
| 1 | **LangGraph** | 5개 노드 연결, 조건 분기, 대화 상태 저장, 루프 제어 |
| 2 | **LangChain** | `@tool` 54개 도구 정의, `ChatOpenAI` LLM, `bind_tools()`, `ToolNode` |
| 3 | **FastAPI / FastMCP** | HTTP/MCP 수신, 결과 반환 |
| 4 | **프로젝트 고유** | 가드레일, ChromaDB 도구 라우팅, 쿼리 재작성, ToolCard, RAG |

> **한 줄 요약:** LangChain이 **부품**(도구·LLM)을 만들고, LangGraph가 **조립·실행**(그래프·상태·분기)하며, 프로젝트 고유 코드가 **도메인 최적화**(가드레일·라우팅·정제)를 담당합니다.

---

## 5. 핵심 모듈 설명

### 5-1. Tool Card — 도구의 신분증

각 도구에 대해 "언제 쓰고, 언제 쓰면 안 되는지"를 정의한 메타데이터입니다.

```python
ToolCard(
    name="premium_estimate",
    purpose="나이·성별을 입력해 특정 상품의 예상 월 보험료를 산출한다.",
    when_to_use=(
        "이 상품 보험료 얼마야?",
        "40세 남성 보험료 계산해줘",
        ...
    ),
    when_not_to_use=(
        "납입 플랜을 비교하고 싶다 → plan_options 사용",
        "예산 내 가입 가능 여부 → affordability_check 사용",
        ...
    ),
    tags=("보험료", "산출", "월납입액"),
)
```

| # | 필드 | 어디에 쓰이나 | 설명 |
|---|------|--------------|------|
| 1 | `purpose` | ChromaDB 임베딩 | 도구가 하는 일 한 문장 |
| 2 | `when_to_use` | ChromaDB 임베딩 | 이 도구를 써야 하는 사용자 발화 예시 |
| 3 | `when_not_to_use` | LLM description만 | 혼동하기 쉬운 유사 도구 안내 |
| 4 | `tags` | ChromaDB 임베딩 | 도메인 키워드 |

> `when_not_to_use`는 임베딩 텍스트에서 **의도적으로 제외**합니다. 다른 도구의 어휘가 섞이면 벡터가 오염되기 때문입니다. 대신 LLM이 `bind_tools()`로 받는 description에만 주입하여 유사 도구 간 혼동을 방지합니다.

### 5-2. ChromaDB 컬렉션 — 딱 2개

| # | 컬렉션 | 사용처 | 내용 |
|---|--------|--------|------|
| 1 | `tool_embeddings` | `tool_search/embedder.py` | 54개 도구 설명 벡터 (~370개 문서) |
| 2 | `rag_documents` | `rag/retriever.py` | PDF/TXT 청크 (~1,400개 문서) |

임베딩 모델: `intfloat/multilingual-e5-large` (1024차원)

```
저장할 때 → "passage: {텍스트}" → ChromaDB에 적재
검색할 때 → "query: {질문}"    → encode 후 ChromaDB에 전달
```

> 가드레일 도메인 체크(L2)는 ChromaDB를 **쓰지 않습니다.** 46개 소량 데이터에 디스크 I/O를 추가할 이유가 없어서, 메모리(numpy)만으로 ~3ms에 처리합니다.

### 5-3. 도구 카탈로그 (54개)

| # | 모듈 | 도구 수 | 주요 기능 |
|---|------|---------|-----------|
| 1 | `product.py` | 10 | 상품 검색/조회/비교, 특약, FAQ |
| 2 | `premium.py` | 8 | 보험료 산출/비교, 플랜 옵션, 갱신 추정 |
| 3 | `coverage.py` | 9 | 보장 요약/상세, 급부 금액, ICD 매핑 |
| 4 | `underwriting.py` | 12 | 가입 심사, 녹아웃 룰, 직업 위험도 |
| 5 | `compliance.py` | 6 | 준법 멘트, 금칙어 검사, 개인정보 마스킹 |
| 6 | `claims.py` | 4 | 청구 절차, 서류, 계약관리 |
| 7 | `customer_db.py` | 3 | 고객 검색, 계약 조회, 중복 가입 검사 |
| 8 | `rag_tools.py` | 2 | 약관 검색, 상품요약서 검색 |

---

## 6. 새 도구 등록 가이드

새 도구를 추가할 때 **3개 파일만 수정**하면 됩니다. 그래프 구조나 라우팅 로직은 건드리지 않습니다.

### 6-1. 전체 프로세스

```
Step 1. 도구 함수 작성         → app/tools/{모듈}.py
Step 2. ToolCard 등록          → app/tool_search/tool_cards.py
Step 3. 평가 케이스 추가 (권장) → scripts/eval_tool_recall.py
Step 4. 서버 재시작            → 자동 인덱싱
```

### 6-2. Step 1 — 도구 함수 작성

`app/tools/` 아래 적절한 모듈에 `@tool` 함수를 작성하고, 모듈 하단 `TOOLS` 리스트에 추가합니다.

```python
# app/tools/premium.py (예시)

from langchain_core.tools import tool
from app.tools.data import PRODUCTS, PREMIUM_TABLES, _json

@tool
def my_new_tool(product_code: str, age: int) -> str:
    """새 도구 설명 — LLM이 이 텍스트를 보고 도구 용도를 파악합니다."""
    # 구현...
    return _json(result)

# 모듈 하단
TOOLS = [
    ...,
    my_new_tool,       # ← 여기에 추가
]
```

> **새 모듈을 만든 경우**, `app/tools/__init__.py`의 `_TOOL_MODULES` 리스트에도 추가합니다.

### 6-3. Step 2 — ToolCard 등록

`app/tool_search/tool_cards.py`의 `_CARDS` 리스트에 ToolCard를 추가합니다.

```python
ToolCard(
    name="my_new_tool",                          # tool 함수 이름과 정확히 일치
    purpose="이 도구가 하는 일을 한 문장으로.",      # 임베딩 텍스트 핵심
    when_to_use=(
        "사용자가 이런 질문을 할 때 1",              # 실제 발화 패턴으로 작성
        "사용자가 이런 질문을 할 때 2",
        "사용자가 이런 질문을 할 때 3",
    ),
    when_not_to_use=(
        "이런 상황엔 → other_tool 사용",            # 혼동 쌍 명시
    ),
    tags=("키워드1", "키워드2"),
),
```

**작성 팁:**

| 항목 | 팁 |
|------|-----|
| `purpose` | 한 문장, 구체적으로. "보험료를 계산한다"보다 "나이·성별을 입력해 예상 월 보험료를 산출한다"가 낫습니다 |
| `when_to_use` | 실제 사용자 발화를 상상해서 5~10개 작성. 다양할수록 Recall이 올라갑니다 |
| `when_not_to_use` | 기존 도구 중 혼동 가능한 것을 지목. `"이런 상황 → {도구명} 사용"` 형식 |
| `tags` | 2~5개 도메인 키워드 |

### 6-4. Step 3 — 평가 케이스 추가 (권장)

`scripts/eval_tool_recall.py`의 `TEST_CASES`에 테스트 쿼리를 추가합니다.

```python
TEST_CASES: list[tuple[str, str]] = [
    ...,
    # ── my_new_tool ──
    ("예상 질문 1", "my_new_tool"),
    ("예상 질문 2", "my_new_tool"),
]
```

그런 다음 평가를 실행합니다.

```bash
python -m scripts.eval_tool_recall --verbose
```

Recall@k가 떨어졌다면 ToolCard의 `when_to_use`를 보강하거나, 기존 도구의 `when_not_to_use`에 새 도구를 명시하세요.

### 6-5. Step 4 — 서버 재시작

서버를 재시작하면 `lifespan` 이벤트에서 자동으로:
1. `get_all_tools()`가 새 도구를 수집
2. `index_tools()`가 ToolCard 변경을 감지하고 ChromaDB 재인덱싱
3. 새 도구가 검색 대상에 포함됨

```bash
python run.py
# INFO: Indexed 55 tools → 380 documents
```

### 6-6. 체크리스트

```
□ @tool 함수 작성 완료
□ 모듈 TOOLS 리스트에 추가
□ (새 모듈이면) __init__.py의 _TOOL_MODULES에 추가
□ tool_cards.py에 ToolCard 등록
□ name이 함수명과 정확히 일치하는지 확인
□ eval_tool_recall.py에 테스트 케이스 추가
□ python -m scripts.eval_tool_recall --verbose 실행
□ Recall@k 유지 확인
□ 서버 재시작 후 /api/tools에서 새 도구 확인
```

---

## 7. Tool Routing 성능 실험

`scripts/eval_tool_recall.py`로 79개 테스트 쿼리(tool-call 64개 + no-call 15개)에 대해 top-k별 성능을 측정했습니다.

### 7-1. 실험 환경

| 항목 | 값 |
|------|-----|
| 임베딩 모델 | `intfloat/multilingual-e5-large` (1024차원) |
| 인덱싱 방식 | Multi-vector (purpose + when_to_use 별도 문서) |
| 테스트 쿼리 | 79개 = tool-call 64개 (혼동 쌍 포함) + no-call 15개 |
| 도구 수 | 54개 → 인덱싱 문서 ~370개 |
| No-Call threshold | 0.86 (top-1 score가 이 미만이면 "도구 불필요"로 판정) |

### 7-2. k별 비교 결과

| 지표 | k=1 | k=3 | k=5 | k=7 | k=10 |
|------|-----|-----|-----|-----|------|
| **Tool Acc (Hit@1)** | 98.4% | 98.4% | 98.4% | 98.4% | 98.4% |
| **Recall@k** | 98.4% | **100%** | **100%** | **100%** | **100%** |
| **MRR** | 0.9844 | 0.9922 | 0.9922 | 0.9922 | 0.9922 |
| **No-Call Acc** | 73.3% | 73.3% | 73.3% | 73.3% | 73.3% |
| **Overall Acc** | 93.7% | 93.7% | 93.7% | 93.7% | 93.7% |

> **지표 설명**
> - **Tool Acc (Hit@1)**: 도구 호출 쿼리 중 top-1이 정답인 비율
> - **Recall@k**: 도구 호출 쿼리 중 top-k 안에 정답이 있는 비율
> - **No-Call Acc**: 도구 불필요 쿼리 중 top-1 score < threshold인 비율
> - **Overall Acc**: (Tool Acc 정답 + No-Call 정답) / 전체 쿼리

### 7-3. 점수 분포 분석

```
Tool-Call top-1 score : min=0.867  avg=0.922  max=0.947
No-Call   top-1 score : min=0.831  avg=0.853  max=0.877
분리 마진 (tool min - no-call max) = -0.010  ← 겹침 구간 존재
```

| threshold | No-Call Acc | 트레이드오프 |
|-----------|------------|-------------|
| 0.86 | **73.3%** (11/15) | 실용적 — tool-call 쿼리 영향 없음 |
| 0.88 | **100%** (15/15) | 이상적이지만 tool min(0.867)과 겹쳐 false no-call 위험 |

**핵심 인사이트:** No-Call 쿼리(일반 보험 지식, 대화형 발화)와 Tool-Call 쿼리의 임베딩 점수가 0.83~0.88 구간에서 **겹칩니다.** 이는 보험 도메인 특성상 자연스러운 현상입니다. "보험이란 무엇인가요?"도, "보험료 얼마야?"도 모두 보험 키워드를 포함하기 때문입니다.

→ **임베딩 score만으로 No-Call을 완벽히 분리하기 어렵기 때문에**, 실제 시스템에서는 **LLM이 2차로 "도구를 호출할 필요 없다"는 판단**을 내립니다. ChromaDB는 후보를 좁히는 역할만 하고, 최종 호출 여부는 LLM이 결정하는 2단계 구조가 이 한계를 보완합니다.

### 7-4. Tool-Call 세부 분석

- **k=3부터 Recall 100%** — 64개 쿼리 전부 top-3 안에 정답 도구 포함
- **Hit@1 = 98.4%** — 63/64 쿼리가 1순위 정답. 유일한 미탐은 "암 진단금이 얼마야?"에서 `coverage_detail` 대신 `benefit_amount_lookup`이 1위로 나온 것인데, 두 도구 모두 적절한 응답이 가능하므로 실질적 오류 아님
- **MRR 0.99** — 정답 도구가 거의 항상 1~2순위에 위치

### 7-5. No-Call 오판 사례 (threshold=0.86 기준)

| # | 쿼리 | top-1 도구 | score | 분석 |
|---|------|-----------|-------|------|
| 1 | "보험 가입 시 주의사항이 뭐야?" | underwriting_waiting_periods | 0.864 | 보험 가입과 관련, 도구 호출도 합리적 |
| 2 | "방금 말씀해주신 내용 요약해줘" | coverage_summary | 0.869 | "요약"이 coverage_summary와 매칭 |
| 3 | "보험 들 때 뭘 확인해야 할까?" | rag_terms_query_engine | 0.877 | 약관 검색이 도움될 수 있음 |
| 4 | "보험 설계사한테 뭘 물어봐야 해?" | product_search | 0.863 | 상품 검색도 유관 |

> 4건 모두 "도구를 써도 틀리지 않는" 경계 사례입니다. 실제 운영에서는 LLM이 컨텍스트를 보고 도구 호출 여부를 최종 결정하므로, 이 수준의 오판은 허용 범위입니다.

### 7-6. Top-K 권장값

| # | 상황 | 권장 k | 이유 |
|---|------|--------|------|
| 1 | 현재 (54개 도구) | **5** | Recall 100% + LLM 부담 최소화 |
| 2 | 도구 80~100개 확장 시 | 7~10 | 도구 밀도 증가 대비 |
| 3 | 극단적 비용 절약 | 3 | Recall 100% 유지 최소 k |

> 현재 설정은 `TOOL_SEARCH_TOP_K=10`이지만, 실험 결과 **k=5로 줄여도 성능 손실 없이 비용 절감** 가능합니다.

### 7-7. 실험 실행 방법

```bash
# 기본 (top-10, 단일 k)
python -m scripts.eval_tool_recall

# k=1,3,5,7,10 비교표
python -m scripts.eval_tool_recall --compare

# 커스텀 k + threshold + 상세
python -m scripts.eval_tool_recall --k 5 --threshold 0.88 --verbose

# 비교할 k 값 직접 지정
python -m scripts.eval_tool_recall --compare --ks 3 5 7 --threshold 0.88
```

> 도구를 추가/변경한 후 반드시 `--compare`로 실행하여 모든 지표가 유지되는지 확인하세요.

---

## 8. 프로젝트 구조

```
mcp_abtest_v1/
│
├── run.py                          # FastAPI 서버 진입점
├── run_mcp.py                      # MCP 서버 실행 + Inspector 연동
├── pyproject.toml                  # 프로젝트 설정 + 의존성
│
├── app/
│   ├── main.py                     # FastAPI 앱 (REST/SSE 엔드포인트)
│   ├── config.py                   # Settings + 임베딩 모델 싱글톤
│   ├── models.py                   # ChatRequest / ChatResponse (Pydantic v2)
│   ├── llm.py                      # OpenRouter ChatOpenAI 싱글톤
│   ├── retry.py                    # tenacity 재시도 (llm_retry / db_retry)
│   │
│   ├── graph/                      # ── LangGraph 계층 ──
│   │   ├── state.py                #   AgentState 정의 (trace·conversation_started)
│   │   ├── nodes.py                #   agent 노드 (ChromaDB 라우팅 + LLM)
│   │   ├── query_rewrite.py        #   쿼리 재작성 노드 (15자 미만)
│   │   ├── builder.py              #   5노드 그래프 + AsyncSqliteSaver
│   │   └── guardrails.py           #   입력(L1·L2) / 출력 가드레일
│   │
│   ├── tools/                      # ── 도구 계층 (54개) ──
│   │   ├── __init__.py             #   get_all_tools() + when_not_to_use 자동 주입
│   │   ├── data.py                 #   시뮬레이션 데이터 SSOT + 시스템 프롬프트
│   │   ├── db_setup.py             #   SQLite 초기화 + threading.Lock
│   │   ├── product.py              #   상품 조회/검색/비교 (10개)
│   │   ├── premium.py              #   보험료 산출/비교 (8개)
│   │   ├── coverage.py             #   보장 분석/급부 (9개)
│   │   ├── underwriting.py         #   가입 심사/고지 (12개)
│   │   ├── compliance.py           #   준법/금칙어 (6개)
│   │   ├── claims.py               #   청구/계약관리 (4개)
│   │   ├── customer_db.py          #   고객 DB 조회 (3개)
│   │   └── rag_tools.py            #   약관/요약서 RAG (2개)
│   │
│   ├── tool_search/                # ── 도구 라우팅 ──
│   │   ├── embedder.py             #   ChromaDB 멀티벡터 임베딩 + top-k 검색
│   │   └── tool_cards.py           #   54개 ToolCard SSOT
│   │
│   ├── rag/                        # ── RAG 계층 ──
│   │   ├── retriever.py            #   문서 검색 + PDF/TXT 인제스트
│   │   └── splitter.py             #   한국어 문장경계 텍스트 분할기
│   │
│   └── mcp_server/                 # ── MCP 서버 계층 ──
│       ├── server.py               #   FastMCP 서버 + 도구 동적 등록
│       ├── resources.py            #   19개 읽기전용 리소스
│       └── prompts.py              #   8개 프롬프트 템플릿
│
├── scripts/
│   ├── init_vectordb.py            # ChromaDB 초기화 (도구 임베딩 + PDF 인제스트)
│   ├── eval_tool_recall.py         # Tool Search Recall@k / MRR 평가
│   └── inspect.sh                  # MCP Inspector 실행 (shell)
│
├── templates/
│   └── index.html                  # 웹 Chat UI (SSE 스트리밍)
│
├── 상품요약서_판매중_표준약관/      # PDF + 회사정보 TXT (RAG 원본)
├── chroma_data/                    # ChromaDB 영구 저장소
├── customer.db                     # SQLite 고객/계약 DB (시뮬레이션)
└── checkpoints.db                  # LangGraph 대화 체크포인트
```

---

## 9. API 명세

### 9-1. 엔드포인트

| # | Method | Path | 설명 |
|---|--------|------|------|
| 1 | `POST` | `/api/chat` | 동기 응답 — 전체 답변 한 번에 |
| 2 | `POST` | `/api/chat/stream` | SSE 스트리밍 — 노드 진행 + 토큰 실시간 |
| 3 | `GET` | `/api/health` | 헬스체크 (도구 수, ChromaDB 상태) |
| 4 | `GET` | `/api/tools` | 도구 카탈로그 |
| 5 | `GET` | `/` | 웹 Chat UI |

### 9-2. 요청/응답 예시

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "치아보험 보험료 알려줘", "thread_id": "user-123"}'
```

```json
{
  "answer": "무배당 THE 건강한치아보험 V(B00197011)의 예상 월 보험료는...",
  "session_id": "...",
  "thread_id": "user-123",
  "tools_used": ["product_search", "premium_estimate"],
  "trace": [
    {"node": "input_guardrail", "action": "pass", "duration_ms": 4},
    {"node": "query_rewriter", "action": "skip"},
    {"node": "agent", "duration_ms": 1200, "tools_bound": 10},
    {"node": "output_guardrail", "action": "pass", "disclaimer_appended": true}
  ]
}
```

### 9-3. SSE 이벤트 프로토콜

| # | 이벤트 | 데이터 | 설명 |
|---|--------|--------|------|
| 1 | `node_start` | `{"node": "guard_input"}` | 노드 시작 |
| 2 | `tools_selected` | `{"tools": ["premium_estimate"]}` | LLM 도구 선택 |
| 3 | `tool_start` | `{"tool": "premium_estimate", "input": {...}}` | 도구 실행 시작 |
| 4 | `tool_end` | `{"tool": "premium_estimate", "duration_ms": 50}` | 도구 실행 완료 |
| 5 | `token` | `{"text": "보험료는"}` | 토큰 스트리밍 |
| 6 | `done` | `{"answer": "...", "tools_used": [...]}` | 완료 |
| 7 | `error` | `{"error": "..."}` | 오류 |

### 9-4. MCP Server

```bash
python run_mcp.py                       # SSE (기본)
python run_mcp.py --transport stdio      # stdio
python run_mcp.py --inspect              # Inspector UI 자동 실행
```

| 리소스 종류 | 수량 |
|------------|------|
| 도구 (Tools) | 54개 + `insurance_chat` 파이프라인 |
| 리소스 (Resources) | 19개 (읽기전용 보험 데이터) |
| 프롬프트 (Prompts) | 8개 (상담·심사·준법·청구 템플릿) |

---

## 10. Quick Start

### Step 1. 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Step 2. 환경변수

`.env` 파일을 프로젝트 루트에 생성:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=qwen/qwen3-14b
EMBEDDING_MODEL=intfloat/multilingual-e5-large
```

### Step 3. ChromaDB 초기화 (최초 1회)

```bash
python scripts/init_vectordb.py
```

도구 임베딩(54개)과 PDF 문서를 ChromaDB에 색인합니다.  
임베딩 모델이나 ToolCard를 변경했을 때 재실행합니다.

### Step 4. 서버 실행

```bash
# FastAPI (웹 UI + REST API)
python run.py                    # → http://localhost:8080

# MCP 서버 (별도 터미널)
python run_mcp.py                # → http://127.0.0.1:8000
```

### Step 5. Tool Search 평가 (선택)

```bash
python -m scripts.eval_tool_recall              # 기본 (top-10)
python -m scripts.eval_tool_recall --k 5 --verbose
```

---

## 11. 운영 가이드 — 하드코딩 값 관리

코드에 의도적으로 하드코딩된 값들입니다. 자동 최적화되지 않으므로, 환경 변경 시 사람이 직접 측정 후 조정해야 합니다.

### 11-1. 임베딩 모델 변경 시 (반드시 재조정)

| # | 값 | 위치 | 현재값 | 의미 |
|---|----|------|--------|------|
| 1 | `_DOMAIN_IN_THRESHOLD` | `guardrails.py` | `0.87` | 이 이상이면 보험 질문 즉시 통과 |
| 2 | `_DOMAIN_MARGIN_THRESHOLD` | `guardrails.py` | `0.03` | out-in 차이가 이 이상이면 차단 |

> 모델을 바꾸면 점수 분포가 완전히 달라집니다. `init_vectordb.py` 실행 후 점수를 확인하고 재조정하세요.

### 11-2. 도구/문서 수 변경 시 (재조정 권장)

| # | 값 | .env 키 | 현재값 | 의미 |
|---|----|---------|--------|------|
| 1 | `tool_search_top_k` | `TOOL_SEARCH_TOP_K` | `10` | ChromaDB에서 추출할 후보 수 |
| 2 | `rag_top_k` | `RAG_TOP_K` | `5` | RAG 검색 반환 문서 수 |
| 3 | `chunk_size` | — | `500`자 | PDF 분할 단위 |
| 4 | `chunk_overlap` | — | `100`자 | 청크 간 겹침 |

### 11-3. 시스템 안정성 (보통 건드리지 않음)

| # | 값 | 현재값 | 의미 |
|---|----|--------|------|
| 1 | `RECURSION_LIMIT` | `30` | agent↔tools 최대 반복 횟수 |
| 2 | `_MAX_OUTPUT_RETRIES` | `1` | 출력 가드레일 재시도 횟수 |
| 3 | `max_conversation_turns` | `20` | 히스토리 유지 턴 수 |
| 4 | `_REWRITE_THRESHOLD` | `15자` | 쿼리 재작성 트리거 길이 |

### 11-4. 변경 작업 흐름

```
임베딩 모델 변경
  → ① init_vectordb.py 재실행
  → ② 보험/비보험 점수 로그 확인
  → ③ 가드레일 임계값 재조정
  → ④ eval_tool_recall.py 실행

도구 추가 (±10개 이상)
  → ① tool_cards.py에 ToolCard 등록
  → ② eval_tool_recall.py 실행
  → ③ tool_search_top_k 재조정 (필요 시)

PDF 대량 추가
  → ① init_vectordb.py 재실행
  → ② RAG 품질 확인
  → ③ rag_top_k / chunk_size 검토
```

---

## 12. 기술 스택

| # | 카테고리 | 기술 | 역할 |
|---|----------|------|------|
| 1 | LLM 오케스트레이션 | LangGraph | ReAct 패턴 그래프 실행, 멀티턴 대화 |
| 2 | LLM 프로바이더 | OpenRouter (ChatOpenAI) | qwen/qwen3-14b |
| 3 | 벡터 DB | ChromaDB | 도구 임베딩 검색 + RAG 문서 검색 |
| 4 | 임베딩 | SentenceTransformers | `intfloat/multilingual-e5-large` (1024차원) |
| 5 | API 서버 | FastAPI | REST + SSE 스트리밍 |
| 6 | MCP 서버 | FastMCP | MCP 프로토콜 구현 |
| 7 | 체크포인터 | langgraph-checkpoint-sqlite | 대화 상태 영구 저장 (AsyncSqliteSaver) |
| 8 | PDF 파싱 | PyMuPDF (fitz) | 약관/요약서 텍스트 추출 |
| 9 | 고객 DB | SQLite3 | 고객/계약 시뮬레이션 |
| 10 | 재시도 | tenacity | LLM/DB exponential backoff |
| 11 | 검증 | Pydantic v2 | 요청/응답 모델, 설정 관리 |
