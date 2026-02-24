"""FastAPI — LangGraph 5역할 구조 기반 보험 챗봇 API.

State(상태) / Reducer(누적) / Node(행동) / Edge(분기) / IO Adapter(래퍼)

Endpoints:
  POST /api/chat        — 동기 응답
  POST /api/chat/stream — SSE 스트리밍 (노드 진행 + 토큰 스트리밍)
  GET  /api/health      — 헬스체크
  GET  /api/tools       — 도구 카탈로그
"""

from __future__ import annotations

import json
import logging
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

    # 4. Tool embeddings 초기 인덱싱
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
        answer=last_msg.content,
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
                        yield _sse("token", {"text": chunk.content})

                elif kind == "on_chain_end" and name == "LangGraph":
                    output = event["data"].get("output", {})
                    messages = output.get("messages", [])
                    # tools_used 는 on_tool_end 이벤트에서 이미 채워짐.
                    # SSE 이벤트가 누락된 경우에만 messages 에서 보완.
                    all_tools = list(dict.fromkeys(
                        tools_used or extract_tools_used(messages)
                    ))

                    if messages:
                        yield _sse("done", {
                            "answer": messages[-1].content,
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
                "short_name": t.description.split("—")[0].split("–")[0].strip()
                if "—" in t.description or "–" in t.description
                else t.description[:20],
            }
            for t in tools
        ],
    }


# ── 도구 핫리로드 API ─────────────────────────────────────────────────────────

@app.delete("/api/tools/{tool_name}")
async def unregister_tool(tool_name: str):
    """런타임에 도구를 해제한다. ChromaDB 벡터도 자동 삭제."""
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
    """특정 도구 모듈의 도구들을 런타임에 재등록한다. 서버 재시작 없이 도구 복원/추가."""
    import importlib

    registry = get_tool_registry()
    try:
        mod = importlib.import_module(f"app.tools.{module_name}")
        mod = importlib.reload(mod)
        tools_in_mod = getattr(mod, "TOOLS", [])
        if not tools_in_mod:
            raise HTTPException(status_code=404, detail=f"No TOOLS in module 'app.tools.{module_name}'")

        registered = []
        for t in tools_in_mod:
            if not registry.get_by_name(t.name):
                registry.register(t)
                registered.append(t.name)

        return {
            "status": "ok",
            "module": module_name,
            "registered": registered,
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


# ── UI ────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
