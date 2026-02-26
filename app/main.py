"""FastAPI — LangGraph 5역할 구조 기반 보험 챗봇 API.

State(상태) / Reducer(누적) / Node(행동) / Edge(분기) / IO Adapter(래퍼)

Endpoints:
  POST   /api/chat                        — 동기 응답
  POST   /api/chat/stream                 — SSE 스트리밍
  GET    /api/health                      — 헬스체크
  GET    /api/tools                       — 도구 카탈로그
  GET    /api/products                    — 상품 카탈로그
  GET    /api/admin/tools                 — Admin 도구 상세 (ToolCard 포함)
  DELETE /api/tools/{tool_name}           — 도구 런타임 해제
  POST   /api/tools/reload-module/{mod}   — 모듈 핫리로드
  GET    /admin/tools                     — Tool Admin UI
  POST   /api/admin/eval/search           — 단일 쿼리 검색 테스트
  POST   /api/admin/eval/bulk-search      — 멀티 쿼리 벌크 검색
  POST   /api/admin/eval/generate-queries — LLM 테스트 질문 생성
  POST   /api/admin/eval/batch/{name}     — ToolCard 배치 Recall 평가
  POST   /api/admin/eval/judge            — LLM-as-Judge 실패 분석
"""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.models import ChatRequest, ChatResponse
from app.graph.builder import get_graph, init_checkpointer, close_checkpointer, RECURSION_LIMIT
from app.graph.state import build_graph_input, extract_tools_used
from app.tools import get_tool_registry

logger = logging.getLogger("insurance.main")

_STAGE_MAP = {
    "input_guardrail": "guard_input",
    "tools": "execute",
    "output_guardrail": "guard_output",
}

_GRAPH_NODES = frozenset(_STAGE_MAP) | {"agent"}


def _node_to_stage(name: str, agent_call_count: int) -> str:
    """LangGraph 노드 이름 → 프론트엔드 파이프라인 스테이지 매핑."""
    if name == "agent":
        return "analyze" if agent_call_count <= 1 else "generate"
    return _STAGE_MAP.get(name, name)


def _on_registry_change(registry) -> None:
    """ToolRegistry 변경 콜백 — ChromaDB 도구 인덱스를 전체 재동기화."""
    try:
        from app.tool_search.embedder import get_tool_search
        searcher = get_tool_search()
        searcher.index_tools(registry.get_all())
        logger.info("ChromaDB re-indexed after registry change (v=%d)", registry.version)
    except Exception as e:
        logger.warning("ChromaDB re-index failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    logging.basicConfig(level=getattr(logging, s.log_level, logging.INFO))

    # 1. Checkpointer 비동기 초기화 (AsyncSqliteSaver) — graph 컴파일 전에 실행
    await init_checkpointer()

    # 2. ToolRegistry 초기화 — 모듈에서 도구 수집 + 변경 콜백 등록
    registry = get_tool_registry()
    registry.load_from_modules()
    registry.on_change(_on_registry_change)
    logger.info("ToolRegistry loaded %d tools", len(registry))

    # 3. 그래프 미리 컴파일 (checkpointer 준비 완료 후)
    get_graph()
    logger.info("LangGraph compiled")

    # 4. ToolCard JSON override 로딩
    try:
        from app.tool_search.tool_cards import apply_overrides
        n = apply_overrides()
        if n:
            logger.info("Applied %d ToolCard overrides from JSON", n)
    except Exception as e:
        logger.warning("ToolCard override loading skipped: %s", e)

    # 5. Tool embeddings 초기 인덱싱
    try:
        from app.tool_search.embedder import get_tool_search
        searcher = get_tool_search()
        searcher.index_tools(registry.get_all())
        logger.info("Tool embeddings indexed in ChromaDB")
    except Exception as e:
        logger.warning("ChromaDB tool indexing skipped: %s", e)

    yield

    from app.tools.db_setup import close_db
    close_db()
    await close_checkpointer()
    logger.info("Shutdown: resources released")


app = FastAPI(
    title="Insurance Chatbot",
    description="LangGraph 5역할 + MCP + RAG(ChromaDB) 보험 챗봇",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


def _build_config(thread_id: str) -> dict:
    """LangGraph 체크포인터용 config. thread_id가 대화 연속성의 키."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def _strip_think(text: str) -> str:
    """최종 응답에서 <think>...</think> 블록을 제거."""
    return _THINK_RE.sub("", text)


class _ThinkFilter:
    """스트리밍 중 <think>...</think> 블록을 실시간 필터링."""

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while self._buf:
            if self._inside:
                end = self._buf.find("</think>")
                if end >= 0:
                    self._buf = self._buf[end + 8:].lstrip("\n")
                    self._inside = False
                    continue
                for i in range(1, min(9, len(self._buf) + 1)):
                    if "</think>"[:i] == self._buf[-i:]:
                        self._buf = self._buf[-i:]
                        return ""
                self._buf = ""
                return ""
            start = self._buf.find("<think>")
            if start >= 0:
                out.append(self._buf[:start])
                self._buf = self._buf[start + 7:]
                self._inside = True
                continue
            for i in range(1, min(8, len(self._buf) + 1)):
                if "<think>"[:i] == self._buf[-i:]:
                    out.append(self._buf[:-i])
                    self._buf = self._buf[-i:]
                    return "".join(out)
            out.append(self._buf)
            self._buf = ""
        return "".join(out)


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    graph = get_graph()
    config = _build_config(req.thread_id)

    try:
        result = await graph.ainvoke(build_graph_input(req.query), config=config)
    except Exception:
        logger.exception("Pipeline execution failed")
        return ChatResponse(
            answer="처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            session_id=req.session_id,
            thread_id=req.thread_id,
        )

    last_msg = result["messages"][-1]

    return ChatResponse(
        answer=_strip_think(last_msg.content),
        session_id=req.session_id,
        thread_id=req.thread_id,
        tools_used=extract_tools_used(result["messages"]),
        trace=result.get("trace", []),
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 스트리밍 엔드포인트.

    이벤트 프로토콜:
      node_start      — {"node": stage_id}
      node_end        — {"node": stage_id, "duration_ms": float}
      tools_selected  — {"tools": [tool_name, ...]}
      tool_start      — {"tool": tool_name, "input": {...}}
      tool_end        — {"tool": tool_name, "duration_ms": float}
      token           — {"text": str}
      done            — {answer, tools_used, tool_used, ...}
      error           — {"error": str}
    """
    graph = get_graph()
    config = _build_config(req.thread_id)
    input_msg = build_graph_input(req.query)

    async def event_generator():
        agent_call_count = 0
        tools_used: list[str] = []
        node_start_times: dict[str, float] = {}
        tool_start_times: dict[str, float] = {}
        turn_trace: list[dict] = []
        think_filter = _ThinkFilter()

        try:
            async for event in graph.astream_events(
                input_msg,
                config=config,
                version="v2",
            ):
                kind = event["event"]
                name = event.get("name", "")

                if kind == "on_chain_start" and name in _GRAPH_NODES:
                    node_start_times[name] = time.perf_counter()
                    if name == "agent":
                        agent_call_count += 1
                    stage = _node_to_stage(name, agent_call_count)
                    yield _sse("node_start", {"node": stage})

                elif kind == "on_chain_end" and name in _GRAPH_NODES:
                    elapsed = (
                        time.perf_counter() - node_start_times.pop(name, time.perf_counter())
                    ) * 1000
                    stage = _node_to_stage(name, agent_call_count)

                    if name == "agent":
                        output = event.get("data", {}).get("output", {})
                        messages = output.get("messages", [])
                        for msg in messages:
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                new_tools = [tc["name"] for tc in msg.tool_calls]
                                tools_used.extend(new_tools)
                                yield _sse("tools_selected", {"tools": new_tools})

                    turn_trace.append({
                        "node": stage,
                        "duration_ms": round(elapsed, 1),
                    })

                    yield _sse("node_end", {
                        "node": stage,
                        "duration_ms": round(elapsed, 1),
                    })

                elif kind == "on_tool_start":
                    tool_start_times[name] = time.perf_counter()
                    tool_input = event.get("data", {}).get("input", {})
                    yield _sse("tool_start", {"tool": name, "input": tool_input})

                elif kind == "on_tool_end":
                    elapsed = (
                        time.perf_counter() - tool_start_times.pop(name, time.perf_counter())
                    ) * 1000
                    yield _sse("tool_end", {
                        "tool": name,
                        "duration_ms": round(elapsed, 1),
                    })

                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        visible = think_filter.feed(chunk.content)
                        if visible:
                            yield _sse("token", {"text": visible})

                elif kind == "on_chain_end" and name == "LangGraph":
                    output = event["data"].get("output", {})
                    messages = output.get("messages", [])
                    # tools_used 는 on_tool_end 이벤트에서 이미 채워짐.
                    # SSE 이벤트가 누락된 경우에만 messages 에서 보완.
                    all_tools = list(dict.fromkeys(
                        tools_used or extract_tools_used(messages)
                    ))

                    if messages:
                        answer = _strip_think(messages[-1].content)
                        yield _sse("done", {
                            "answer": answer,
                            "tools_used": all_tools,
                            "tool_used": all_tools[0] if all_tools else None,
                            "trace": turn_trace,
                        })

        except Exception as e:
            logger.exception("Streaming pipeline failed")
            yield _sse("error", {"error": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
async def health():
    registry = get_tool_registry()
    checks: dict = {"tools": len(registry), "registry_version": registry.version}

    try:
        from app.tool_search.embedder import get_tool_search
        checks["chromadb_tools"] = get_tool_search()._collection.count()
    except Exception:
        checks["chromadb_tools"] = "unavailable"

    try:
        from app.rag.retriever import get_rag_retriever
        checks["chromadb_docs"] = get_rag_retriever().doc_count
    except Exception:
        checks["chromadb_docs"] = "unavailable"

    all_ok = all(isinstance(v, (int, str)) and v != "unavailable" for v in checks.values())
    return {"status": "ok" if all_ok else "degraded", **checks}


@app.get("/api/products")
async def list_products():
    """판매 중인 상품 카탈로그 — 프론트엔드 상품 목록 표시용."""
    from app.tools.data import PRODUCTS
    return {
        "count": len(PRODUCTS),
        "products": [
            {
                "code": p["code"],
                "name": p["name"],
                "category": p["category"],
                "sales_status": p["sales_status"],
                "channels": p["channels"],
                "age_range": f"{p['min_age']}~{p['max_age']}세",
                "renewal_type": p["renewal_type"],
                "simplified": p.get("simplified_underwriting", False),
                "highlights": p["highlights"][:2],
            }
            for p in PRODUCTS.values()
        ],
    }


@app.get("/api/tools")
async def list_tools():
    """도구 카탈로그 — 프론트엔드가 동적으로 표시명을 가져감."""
    registry = get_tool_registry()
    tools = registry.get_all()
    return {
        "count": len(tools),
        "registry_version": registry.version,
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "short_name": t.name.replace("_", " ").title(),
            }
            for t in tools
        ],
    }


# ── 도구 핫리로드 API ─────────────────────────────────────────────────────────

@app.delete("/api/tools/{tool_name}")
async def unregister_tool(tool_name: str):
    """런타임에 도구를 해제한다. ChromaDB 벡터도 자동 삭제.

    1) remove_tool — ChromaDB에서 즉시 벡터 제거 (검색 즉시 반영)
    2) unregister — 레지스트리에서 해제 → 백그라운드 재인덱싱으로 해시 갱신
    """
    registry = get_tool_registry()
    if not registry.get_by_name(tool_name):
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    try:
        from app.tool_search.embedder import get_tool_search
        get_tool_search().remove_tool(tool_name)
    except Exception as e:
        logger.warning("ChromaDB removal failed for '%s': %s", tool_name, e)

    registry.unregister(tool_name)
    return {
        "status": "ok",
        "message": f"Tool '{tool_name}' unregistered",
        "tools_count": len(registry),
        "registry_version": registry.version,
    }


@app.post("/api/tools/reload-module/{module_name}")
async def reload_module_tools(module_name: str):
    """특정 도구 모듈의 도구들을 런타임에 재등록한다. 서버 재시작 없이 도구 복원/추가.

    register_many()로 일괄 등록하여 콜백(ChromaDB 재인덱싱)을 1회만 호출한다.
    기존 코드 변경도 반영되므로 진정한 '핫리로드'로 동작한다.
    """
    import importlib

    registry = get_tool_registry()
    try:
        mod = importlib.import_module(f"app.tools.{module_name}")
        mod = importlib.reload(mod)
        tools_in_mod = getattr(mod, "TOOLS", [])
        if not tools_in_mod:
            raise HTTPException(status_code=404, detail=f"No TOOLS in module 'app.tools.{module_name}'")

        registry.register_many(tools_in_mod)

        return {
            "status": "ok",
            "module": module_name,
            "registered": [t.name for t in tools_in_mod],
            "tools_count": len(registry),
            "registry_version": registry.version,
        }
    except ImportError:
        raise HTTPException(status_code=404, detail=f"Module 'app.tools.{module_name}' not found")


# ── Debug (개발 환경 전용) ────────────────────────────────────────────────────

@app.get("/api/debug/state/{thread_id}")
async def debug_state(thread_id: str):
    """LangGraph 체크포인터에서 특정 thread의 현재 상태를 조회.

    LangGraph가 기본 제공하는 get_state() / get_state_history() 를 활용한다.
    프로덕션에서는 이 엔드포인트를 비활성화하거나 인증을 추가해야 한다.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = await graph.aget_state(config)
        if state is None or not state.values:
            return {"thread_id": thread_id, "found": False}

        values = state.values
        messages = values.get("messages", [])
        return {
            "thread_id": thread_id,
            "found": True,
            "message_count": len(messages),
            "last_human": next(
                (m.content for m in reversed(messages)
                 if hasattr(m, "type") and m.type == "human"),
                None,
            ),
            "last_ai": next(
                (m.content[:200] for m in reversed(messages)
                 if hasattr(m, "type") and m.type == "ai" and m.content),
                None,
            ),
            "guardrail_action": values.get("guardrail_action"),
            "rewritten_query": values.get("rewritten_query") or None,
            "trace": values.get("trace", []),
            "next": list(state.next),
        }
    except Exception as e:
        logger.warning("debug_state failed for thread_id=%s: %s", thread_id, e)
        return {"thread_id": thread_id, "found": False, "error": str(e)}


# ── Admin API ─────────────────────────────────────────────────────────────────

@app.get("/api/admin/tools")
async def admin_list_tools():
    """Admin UI용 상세 도구 카탈로그 — ToolCard 메타데이터 포함."""
    from app.tool_search.tool_cards import get_card
    from app.tools import _TOOL_MODULES

    registry = get_tool_registry()
    tools = registry.get_all()

    tool_modules: dict[str, str] = {}
    for mod in _TOOL_MODULES:
        mod_name = mod.__name__.rsplit(".", 1)[-1]
        for t in getattr(mod, "TOOLS", []):
            tool_modules[t.name] = mod_name

    result = []
    for t in sorted(tools, key=lambda x: x.name):
        card = get_card(t.name)
        result.append({
            "name": t.name,
            "description": t.description[:100],
            "module": tool_modules.get(t.name, "unknown"),
            "has_card": card is not None,
            "purpose": card.purpose if card else None,
            "when_to_use": list(card.when_to_use) if card else [],
            "when_not_to_use": list(card.when_not_to_use) if card else [],
            "tags": list(card.tags) if card else [],
        })
    return {"count": len(result), "registry_version": registry.version, "tools": result}


# ── ToolCard CRUD API ─────────────────────────────────────────────────────────

def _get_store():
    from app.tool_search.toolcard_store import get_toolcard_store
    return get_toolcard_store(on_publish=_on_card_publish)


def _on_card_publish(card) -> None:
    """ToolCard publish 후 ChromaDB 재인덱싱 트리거."""
    try:
        from app.tool_search.embedder import get_tool_search
        registry = get_tool_registry()
        get_tool_search().index_tools(registry.get_all())
        logger.info("ChromaDB re-indexed after ToolCard publish (%s)", card.name)
    except Exception as e:
        logger.warning("ChromaDB re-index after card publish failed: %s", e)


@app.get("/api/admin/toolcards/{name}")
async def get_toolcard(name: str):
    """ToolCard 상세 — published/draft/code 원본 + 상태."""
    from app.tool_search.tool_cards import CODE_REGISTRY, REGISTRY

    store = _get_store()
    status = store.get_status(name)
    draft = store.get_draft(name)
    effective = REGISTRY.get(name)
    code_card = CODE_REGISTRY.get(name)

    return {
        **status,
        "effective": {
            "purpose": effective.purpose,
            "when_to_use": list(effective.when_to_use),
            "when_not_to_use": list(effective.when_not_to_use),
            "tags": list(effective.tags),
        } if effective else None,
        "code_original": {
            "purpose": code_card.purpose,
            "when_to_use": list(code_card.when_to_use),
            "when_not_to_use": list(code_card.when_not_to_use),
            "tags": list(code_card.tags),
        } if code_card else None,
        "draft": draft,
    }


@app.put("/api/admin/toolcards/{name}/draft")
async def save_toolcard_draft(name: str, request: Request):
    """ToolCard draft 저장. 챗봇에는 미반영."""
    body = await request.json()
    store = _get_store()
    draft = store.save_draft(name, body)
    return {"status": "draft_saved", "name": name, "draft": draft}


@app.post("/api/admin/toolcards/{name}/publish")
async def publish_toolcard(name: str, request: Request):
    """Draft → Published. 메모리 + ChromaDB 즉시 반영."""
    body = await request.json()
    note = body.get("note", "")
    store = _get_store()

    has_draft = store.get_draft(name)
    if not has_draft:
        data = body.get("data")
        if not data:
            raise HTTPException(400, "No draft exists and no data provided")
        store.save_draft(name, data)

    card = store.publish(name, note)
    status = store.get_status(name)
    return {"status": "published", "name": name, "version": status["version"]}


@app.post("/api/admin/toolcards/{name}/rollback")
async def rollback_toolcard(name: str, request: Request):
    """특정 버전으로 롤백."""
    body = await request.json()
    target_version = body.get("version")
    if not target_version:
        raise HTTPException(400, "version is required")
    store = _get_store()
    card = store.rollback(name, int(target_version))
    status = store.get_status(name)
    return {"status": "rolled_back", "name": name, "version": status["version"]}


@app.get("/api/admin/toolcards/{name}/history")
async def toolcard_history(name: str):
    """버전 이력 조회."""
    store = _get_store()
    history = store.get_history(name)
    return {"name": name, "history": history}


@app.delete("/api/admin/toolcards/{name}/override")
async def reset_toolcard(name: str):
    """Override 제거 → 코드 원본 카드로 복원."""
    store = _get_store()
    card = store.reset_to_code(name)
    return {"status": "reset_to_code", "name": name, "has_code_card": card is not None}


@app.post("/api/admin/toolcards/{name}/discard-draft")
async def discard_toolcard_draft(name: str):
    """Draft 폐기."""
    store = _get_store()
    store.discard_draft(name)
    return {"status": "draft_discarded", "name": name}


# ── Quick Eval API ────────────────────────────────────────────────────────────

@app.post("/api/admin/eval/search")
async def eval_search(request: Request):
    """단일 쿼리 → top-k 검색 결과 반환. 실시간 라우팅 확인용."""
    body = await request.json()
    query = body.get("query", "").strip()
    top_k = body.get("top_k", 5)
    if not query:
        raise HTTPException(400, "query is required")

    from app.tool_search.embedder import get_tool_search
    candidates = get_tool_search().search(query, top_k=top_k)
    return {
        "query": query,
        "results": [
            {"name": c.name, "score": c.score, "description": c.description[:80]}
            for c in candidates
        ],
    }


@app.post("/api/admin/eval/generate-queries")
async def eval_generate_queries(request: Request):
    """LLM이 특정 도구에 맞는 다양한 테스트 질문을 생성."""
    body = await request.json()
    tool_name = body.get("tool_name", "").strip()
    count = min(body.get("count", 8), 12)
    if not tool_name:
        raise HTTPException(400, "tool_name is required")

    from app.tool_search.tool_cards import REGISTRY
    from app.llm import get_llm

    card = REGISTRY.get(tool_name)
    if not card:
        raise HTTPException(404, f"ToolCard '{tool_name}' not found")

    existing = "\n".join(f"  - {q}" for q in card.when_to_use[:5])
    negative = "\n".join(f"  - {q}" for q in card.when_not_to_use[:3])

    prompt = f"""당신은 보험 챗봇의 Tool Routing 테스트 전문가입니다.

아래 도구에 대해 실제 계약자가 할 법한 다양한 테스트 질문을 {count}개 생성하세요.

## 도구: {tool_name}
- 목적: {card.purpose}
- 태그: {', '.join(card.tags)}
- 기존 when_to_use 예시:
{existing}
- when_not_to_use 예시:
{negative}

## 규칙
1. 기존 when_to_use와 겹치지 않는 새로운 표현을 사용하세요.
2. 구어체, 존댓말, 반말, 줄임말 등 다양한 말투를 섞으세요.
3. 쉬운 질문(명확히 이 도구)과 어려운 질문(다른 도구와 헷갈릴 수 있는)을 반반 섞으세요.
4. 반드시 이 도구가 정답인 질문만 만드세요.
5. 한 줄에 하나씩, 번호 없이, 질문만 출력하세요. 다른 설명은 하지 마세요."""

    llm = get_llm()
    try:
        result = await llm.ainvoke(prompt)
        text = result.content if hasattr(result, "content") else str(result)
        text = _strip_think(text)
        queries = [
            line.strip().lstrip("•-0123456789. ").strip('"').strip()
            for line in text.strip().split("\n")
            if line.strip() and len(line.strip()) > 3
        ][:count]
    except Exception as e:
        logger.warning("Query generation failed: %s", e)
        raise HTTPException(500, f"LLM 질문 생성 실패: {e}")

    return {"tool_name": tool_name, "queries": queries}


@app.post("/api/admin/eval/bulk-search")
async def eval_bulk_search(request: Request):
    """여러 쿼리를 한번에 검색 — As-Is/To-Be 비교 기반 데이터 수집용."""
    body = await request.json()
    queries = body.get("queries", [])
    tool_name = body.get("tool_name", "").strip()
    top_k = body.get("top_k", 5)

    if not queries or not tool_name:
        raise HTTPException(400, "queries and tool_name are required")

    from app.tool_search.embedder import get_tool_search
    searcher = get_tool_search()

    results = []
    for q in queries[:20]:
        hits = searcher.search(q, top_k=top_k)
        rank = next((i + 1 for i, c in enumerate(hits) if c.name == tool_name), None)
        score = next((c.score for c in hits if c.name == tool_name), 0)
        results.append({
            "query": q,
            "rank": rank,
            "score": round(score, 4) if score else 0,
            "top_hit": hits[0].name if hits else "",
            "top_score": round(hits[0].score, 4) if hits else 0,
        })

    return {"tool_name": tool_name, "results": results}


@app.post("/api/admin/eval/compare-analysis")
async def eval_compare_analysis(request: Request):
    """As-Is/To-Be 정량 비교 + ToolCard diff + LLM 정성 분석을 한번에 반환."""
    body = await request.json()
    tool_name = body.get("tool_name", "").strip()
    as_is = body.get("as_is", [])
    to_be = body.get("to_be", [])
    card_diff = body.get("card_diff", {})

    if not tool_name or not as_is or not to_be:
        raise HTTPException(400, "tool_name, as_is, to_be are required")

    n = len(as_is)
    as_r1 = sum(1 for r in as_is if r.get("rank") == 1)
    to_r1 = sum(1 for r in to_be if r.get("rank") == 1)
    as_in3 = sum(1 for r in as_is if r.get("rank") and r["rank"] <= 3)
    to_in3 = sum(1 for r in to_be if r.get("rank") and r["rank"] <= 3)

    improved = []
    regressed = []
    for a, t in zip(as_is, to_be):
        ar = a.get("rank") or 99
        tr = t.get("rank") or 99
        if tr < ar:
            improved.append(a.get("query", ""))
        elif tr > ar:
            regressed.append(a.get("query", ""))

    diff_desc_parts = []
    for field, changes in card_diff.items():
        added = changes.get("added", [])
        removed = changes.get("removed", [])
        if added:
            diff_desc_parts.append(f"[{field}] 추가: {added}")
        if removed:
            diff_desc_parts.append(f"[{field}] 삭제: {removed}")
    diff_summary = "\n".join(diff_desc_parts) if diff_desc_parts else "변경 없음"

    from app.llm import get_llm
    prompt = f"""당신은 Tool Routing 최적화 전문가입니다.

아래는 "{tool_name}" 도구의 ToolCard 수정 전후 비교 결과입니다. 간결하게 분석해주세요.

## ToolCard 변경 사항
{diff_summary}

## 정량 결과
- 1위 정확도: {round(as_r1/n*100)}% → {round(to_r1/n*100)}% ({"↑" if to_r1>as_r1 else "↓" if to_r1<as_r1 else "="})
- Top-3 포함: {round(as_in3/n*100)}% → {round(to_in3/n*100)}% ({"↑" if to_in3>as_in3 else "↓" if to_in3<as_in3 else "="})
- 개선 {len(improved)}건, 하락 {len(regressed)}건 / 전체 {n}건

## 개선된 쿼리
{chr(10).join(f'  - {q}' for q in improved[:5]) if improved else '  없음'}

## 하락한 쿼리
{chr(10).join(f'  - {q}' for q in regressed[:5]) if regressed else '  없음'}

## 요청
1. 이 변경이 전반적으로 긍정적인지 부정적인지 한 줄로 요약하세요.
2. 하락한 쿼리가 있다면 원인과 보완 방안을 제안하세요.
3. 추가로 개선할 수 있는 방향이 있다면 제안하세요.

한국어로 간결하게(5줄 이내) 답변하세요."""

    analysis = ""
    llm = get_llm()
    try:
        result = await llm.ainvoke(prompt)
        analysis = result.content if hasattr(result, "content") else str(result)
        analysis = _strip_think(analysis)
    except Exception as e:
        logger.warning("Compare analysis LLM failed: %s", e)
        analysis = f"LLM 분석 실패: {e}"

    return {
        "tool_name": tool_name,
        "quantitative": {
            "as_is_r1": round(as_r1 / n * 100) if n else 0,
            "to_be_r1": round(to_r1 / n * 100) if n else 0,
            "as_is_in3": round(as_in3 / n * 100) if n else 0,
            "to_be_in3": round(to_in3 / n * 100) if n else 0,
            "improved": len(improved),
            "regressed": len(regressed),
            "total": n,
        },
        "analysis": analysis,
    }


@app.post("/api/admin/eval/batch/{tool_name}")
async def eval_batch(tool_name: str):
    """ToolCard의 when_to_use 전체를 검색하여 Recall@1/3/5 산출."""
    from app.tool_search.tool_cards import REGISTRY
    from app.tool_search.embedder import get_tool_search

    card = REGISTRY.get(tool_name)
    if not card or not card.when_to_use:
        raise HTTPException(404, f"No ToolCard or when_to_use for '{tool_name}'")

    searcher = get_tool_search()
    details = []
    for query in card.when_to_use:
        hits = searcher.search(query, top_k=5)
        rank = next(
            (i + 1 for i, c in enumerate(hits) if c.name == tool_name), None
        )
        details.append({
            "query": query,
            "rank": rank,
            "pass_at_3": rank is not None and rank <= 3,
            "top_hits": [
                {"name": c.name, "score": c.score} for c in hits[:5]
            ],
        })

    n = len(details)
    return {
        "tool_name": tool_name,
        "total": n,
        "recall_at_1": round(sum(1 for d in details if d["rank"] == 1) / n, 4) if n else 0,
        "recall_at_3": round(sum(1 for d in details if d["pass_at_3"]) / n, 4) if n else 0,
        "recall_at_5": round(sum(1 for d in details if d["rank"] and d["rank"] <= 5) / n, 4) if n else 0,
        "details": details,
    }


@app.post("/api/admin/eval/judge")
async def eval_judge(request: Request):
    """LLM-as-Judge: 실패 케이스를 분석하고 ToolCard 개선안을 제안."""
    body = await request.json()
    tool_name = body.get("tool_name", "")
    failures = body.get("failures", [])

    if not tool_name or not failures:
        raise HTTPException(400, "tool_name and failures are required")

    from app.tool_search.tool_cards import REGISTRY
    from app.llm import get_llm

    card = REGISTRY.get(tool_name)
    card_info = ""
    if card:
        card_info = (
            f"purpose: {card.purpose}\n"
            f"when_to_use: {list(card.when_to_use)}\n"
            f"when_not_to_use: {list(card.when_not_to_use)}\n"
            f"tags: {list(card.tags)}"
        )

    failure_lines = []
    for f in failures[:10]:
        top_str = ", ".join(
            f"{h['name']}({h['score']})" for h in f.get("top_hits", [])[:3]
        )
        failure_lines.append(
            f"  쿼리: \"{f['query']}\"  → 상위결과: [{top_str}]  "
            f"(expected: {tool_name}, rank: {f.get('rank', 'N/A')})"
        )

    prompt = f"""당신은 Tool Routing 전문가입니다. 아래 도구의 ToolCard 정보와, 해당 도구로 라우팅되어야 했지만 실패한 쿼리들을 분석해주세요.

## 도구: {tool_name}
{card_info}

## 실패 케이스 ({len(failure_lines)}건)
{chr(10).join(failure_lines)}

## 요청사항
1. 각 실패 쿼리가 왜 다른 도구로 라우팅되었는지 원인을 분석하세요.
2. ToolCard를 어떻게 수정하면 이 쿼리들이 올바르게 라우팅될지 구체적으로 제안하세요.
   - 추가할 when_to_use 예시
   - 추가할 tags
   - 수정할 purpose
3. 주의: when_to_use에 이미 있는 쿼리와 너무 유사한 문장은 효과가 적습니다. 다양한 표현을 제안하세요.

한국어로 간결하게 답변하세요."""

    llm = get_llm()
    try:
        result = await llm.ainvoke(prompt)
        analysis = result.content if hasattr(result, "content") else str(result)
        analysis = _strip_think(analysis)
    except Exception as e:
        logger.warning("LLM Judge failed: %s", e)
        analysis = f"LLM 분석 실패: {e}"

    return {
        "tool_name": tool_name,
        "failure_count": len(failures),
        "analysis": analysis,
    }


# ── UI ────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/admin/tools", response_class=HTMLResponse)
async def admin_tools_page(request: Request):
    return templates.TemplateResponse("admin_tools.html", {"request": request})
