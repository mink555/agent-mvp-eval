"""MCP Server — FastMCP 기반 보험 도구 + LangGraph 파이프라인 통합.

LangChain 도구를 FastMCP에 동적 등록하고,
insurance_chat 도구로 전체 파이프라인도 노출한다.

IO Adapter 역할 (가이드 레벨 2):
  - tool 카탈로그 조회  → register_all_tools()
  - tool 호출           → _make_handler()
  - 결과 정규화         → JSON string 반환
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from app.config import get_settings
from app.graph.state import build_graph_input, extract_tools_used
from app.tools import get_all_tools
from app.tools.data import _json

logger = logging.getLogger("insurance.mcp_server")

Transport = Literal["stdio", "sse", "streamable-http"]

_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}

# ── 싱글톤 FastMCP 인스턴스 ───────────────────────────────────────────────────

_mcp: FastMCP | None = None


def get_mcp(**overrides: Any) -> FastMCP:
    """싱글톤 FastMCP 인스턴스를 반환.

    최초 호출 시 Settings 값으로 생성하며,
    overrides(host, port, …)로 개별 설정을 덮어쓸 수 있다.
    """
    global _mcp
    if _mcp is None:
        s = get_settings()
        _mcp = FastMCP(
            overrides.get("name", s.mcp_server_name),
            host=overrides.get("host", s.mcp_host),
            port=overrides.get("port", s.mcp_port),
        )
    return _mcp


# ── Tool 등록 헬퍼 ────────────────────────────────────────────────────────────

def _resolve_json_type(pinfo: dict) -> str:
    """JSON Schema 프로퍼티에서 Python 타입명을 추출.

    단순 {"type": "string"} 외에도 Optional 필드의
    {"anyOf": [{"type": "string"}, {"type": "null"}]} 형태를 처리한다.
    """
    if "type" in pinfo:
        return pinfo["type"]
    any_of = pinfo.get("anyOf", [])
    for item in any_of:
        t = item.get("type")
        if t and t != "null":
            return t
    return "string"


def _build_signature_from_tool(t) -> tuple[inspect.Signature, dict[str, type]]:
    schema = t.args_schema.model_json_schema() if t.args_schema else {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    annotations: dict[str, type] = {"return": str}

    required_params, optional_params = [], []
    for pname, pinfo in properties.items():
        json_type = _resolve_json_type(pinfo)
        py_type = _TYPE_MAP.get(json_type, str)
        annotations[pname] = py_type

        if pname in required:
            required_params.append(inspect.Parameter(
                pname, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=py_type,
            ))
        else:
            default = pinfo.get("default", "" if py_type is str else False if py_type is bool else 0 if py_type in (int, float) else [])
            optional_params.append(inspect.Parameter(
                pname, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default, annotation=py_type,
            ))

    return inspect.Signature(required_params + optional_params, return_annotation=str), annotations


def _make_handler(tool_obj):
    async def handler(**kwargs: Any) -> str:
        try:
            result = await asyncio.to_thread(tool_obj.invoke, kwargs)
            return result if isinstance(result, str) else _json(result)
        except Exception as e:
            return _json({"error": str(e)})
    return handler


def register_all_tools(mcp: FastMCP) -> int:
    tools = get_all_tools()
    for t in tools:
        handler = _make_handler(t)
        sig, annotations = _build_signature_from_tool(t)

        handler.__name__ = t.name
        handler.__doc__ = t.description
        handler.__signature__ = sig
        handler.__annotations__ = annotations

        mcp.add_tool(handler, name=t.name, description=t.description)

    logger.info("Registered %d tools to FastMCP server", len(tools))
    return len(tools)


# ── Pipeline Tool ─────────────────────────────────────────────────────────────

def _register_pipeline_tool(mcp: FastMCP) -> None:
    @mcp.tool(
        name="insurance_chat",
        description=(
            "보험 관련 자연어 질문에 답변합니다. "
            "내부적으로 agent ↔ tools ReAct 루프를 거칩니다."
        ),
    )
    async def insurance_chat(
        query: str,
        session_id: str = "mcp-default",
        thread_id: str = "mcp-default",
    ) -> str:
        from app.graph.builder import get_graph, RECURSION_LIMIT

        graph = get_graph()
        config = {
            "configurable": {"thread_id": f"{session_id}:{thread_id}"},
            "recursion_limit": RECURSION_LIMIT,
        }

        try:
            result = await graph.ainvoke(
                build_graph_input(query),
                config=config,
            )
        except Exception as e:
            logger.exception("Pipeline failed via MCP")
            return _json({"error": str(e)})

        last_msg = result["messages"][-1]

        return _json({
            "answer": last_msg.content,
            "tools_used": extract_tools_used(result["messages"]),
            "trace": result.get("trace", []),
        })


# ── 초기화 & 실행 ─────────────────────────────────────────────────────────────

_initialized = False


def init_mcp(**overrides: Any) -> FastMCP:
    """MCP 서버에 도구·리소스·프롬프트를 등록한다. 최초 1회만 실행."""
    global _initialized
    mcp = get_mcp(**overrides)

    if _initialized:
        return mcp
    _initialized = True

    register_all_tools(mcp)
    _register_pipeline_tool(mcp)

    from app.mcp_server.resources import register_all_resources
    from app.mcp_server.prompts import register_all_prompts

    rc = register_all_resources(mcp)
    logger.info("Registered %d resources to FastMCP server", rc)

    pc = register_all_prompts(mcp)
    logger.info("Registered %d prompts to FastMCP server", pc)

    return mcp


_TRANSPORT_RUNNERS = {
    "sse": "run_sse_async",
    "stdio": "run_stdio_async",
    "streamable-http": "run_streamable_http_async",
}


async def run_mcp_server(
    transport: Transport | None = None,
    **overrides: Any,
) -> None:
    """MCP 서버를 시작한다.

    Args:
        transport: "sse" | "stdio" | "streamable-http". None이면 Settings 값 사용.
        **overrides: get_mcp()에 전달할 host/port/name 오버라이드.
    """
    mcp = init_mcp(**overrides)
    s = get_settings()

    transport = transport or s.mcp_transport  # type: ignore[assignment]
    runner_name = _TRANSPORT_RUNNERS.get(transport, "run_sse_async")  # type: ignore[arg-type]

    logger.info(
        "Starting MCP server (transport=%s, host=%s, port=%s)...",
        transport, s.mcp_host, s.mcp_port,
    )

    runner = getattr(mcp, runner_name)
    await runner()
