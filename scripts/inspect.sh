#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# MCP Inspector 런처
#
# Usage:
#   ./scripts/inspect.sh                    # SSE (기본)
#   ./scripts/inspect.sh stdio              # stdio
#   ./scripts/inspect.sh sse 9000           # SSE + 커스텀 포트
#   ./scripts/inspect.sh streamable-http    # streamable-http
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TRANSPORT="${1:-sse}"
PORT="${2:-8000}"
HOST="${3:-127.0.0.1}"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3 || command -v python)"
fi

export DANGEROUSLY_OMIT_AUTH=true

if ! command -v npx &>/dev/null; then
    echo "❌  npx를 찾을 수 없습니다. Node.js(v18+)를 설치해주세요."
    exit 1
fi

case "$TRANSPORT" in
    stdio)
        echo "🔍  MCP Inspector (stdio) 시작..."
        exec npx @modelcontextprotocol/inspector "$PYTHON" "${PROJECT_ROOT}/run_mcp.py" --transport stdio
        ;;
    sse)
        echo "🚀  MCP 서버(SSE) 시작 → ${HOST}:${PORT}"
        "$PYTHON" "${PROJECT_ROOT}/run_mcp.py" --transport sse --host "$HOST" --port "$PORT" &
        SERVER_PID=$!
        trap 'echo "🛑  서버 종료(PID=$SERVER_PID)"; kill $SERVER_PID 2>/dev/null' EXIT INT TERM

        sleep 2
        SSE_URL="http://${HOST}:${PORT}/sse"
        echo "🔍  MCP Inspector → ${SSE_URL}"
        npx @modelcontextprotocol/inspector --cli --method sse "$SSE_URL"
        ;;
    streamable-http)
        echo "🚀  MCP 서버(streamable-http) 시작 → ${HOST}:${PORT}"
        "$PYTHON" "${PROJECT_ROOT}/run_mcp.py" --transport streamable-http --host "$HOST" --port "$PORT" &
        SERVER_PID=$!
        trap 'echo "🛑  서버 종료(PID=$SERVER_PID)"; kill $SERVER_PID 2>/dev/null' EXIT INT TERM

        sleep 2
        MCP_URL="http://${HOST}:${PORT}/mcp"
        echo "🔍  MCP Inspector → ${MCP_URL}"
        npx @modelcontextprotocol/inspector --cli --method streamableHttp "$MCP_URL"
        ;;
    *)
        echo "❌  지원하지 않는 transport: $TRANSPORT"
        echo "    사용 가능: stdio | sse | streamable-http"
        exit 1
        ;;
esac
