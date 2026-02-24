"""MCP 서버 단독 실행 + Inspector 연동.

Usage:
    python run_mcp.py                          # SSE (Settings 기본값)
    python run_mcp.py --transport stdio         # stdio
    python run_mcp.py --transport streamable-http
    python run_mcp.py --host 0.0.0.0 --port 9000
    python run_mcp.py --inspect                 # Inspector UI 자동 실행 (SSE)
    python run_mcp.py --inspect --transport stdio
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insurance MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  python run_mcp.py                            # SSE 모드\n"
            "  python run_mcp.py --transport stdio           # stdio 모드\n"
            "  python run_mcp.py --inspect                   # Inspector + SSE\n"
            "  python run_mcp.py --inspect --transport stdio  # Inspector + stdio\n"
        ),
    )
    parser.add_argument(
        "--transport", "-t",
        choices=["sse", "stdio", "streamable-http"],
        default=None,
        help="전송 방식 (기본: Settings.mcp_transport → sse)",
    )
    parser.add_argument("--host", default=None, help="바인드 호스트 (기본: Settings.mcp_host)")
    parser.add_argument("--port", "-p", type=int, default=None, help="포트 (기본: Settings.mcp_port)")
    parser.add_argument(
        "--inspect", action="store_true",
        help="MCP Inspector UI를 자동으로 실행",
    )
    parser.add_argument(
        "--stdio", action="store_true",
        help="--transport stdio 의 단축 옵션",
    )
    return parser.parse_args()


def _find_npx() -> str | None:
    return shutil.which("npx")


def _launch_inspector_sse(host: str, port: int, sse_path: str = "/sse") -> subprocess.Popen:
    """SSE 서버에 연결하는 Inspector 프로세스를 실행."""
    url = f"http://{host}:{port}{sse_path}"
    npx = _find_npx()
    if not npx:
        print("[ERROR] npx를 찾을 수 없습니다. Node.js(v18+)를 설치해주세요.", file=sys.stderr)
        sys.exit(1)

    print(f"[Inspector] SSE 서버 연결 → {url}")
    return subprocess.Popen(
        [npx, "@modelcontextprotocol/inspector", "--cli", "--method", "sse", url],
        env={**os.environ, "NODE_NO_WARNINGS": "1"},
    )


def _launch_inspector_stdio() -> None:
    """Inspector가 직접 이 프로세스를 stdio로 감싸서 실행."""
    npx = _find_npx()
    if not npx:
        print("[ERROR] npx를 찾을 수 없습니다. Node.js(v18+)를 설치해주세요.", file=sys.stderr)
        sys.exit(1)

    python = sys.executable
    cmd = [npx, "@modelcontextprotocol/inspector", python, "run_mcp.py", "--transport", "stdio"]
    print(f"[Inspector] stdio 모드 → {' '.join(cmd)}")
    os.execvp(npx, cmd)


def main() -> None:
    args = _parse_args()
    transport: str | None = args.transport
    if args.stdio:
        transport = "stdio"

    from app.config import get_settings
    s = get_settings()

    host = args.host or s.mcp_host
    port = args.port or s.mcp_port
    transport = transport or s.mcp_transport

    logging.basicConfig(
        level=getattr(logging, s.log_level, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ── Inspector 모드 ─────────────────────────────────────
    if args.inspect:
        if transport == "stdio":
            _launch_inspector_stdio()
            return

        async def _run_with_inspector():
            from app.mcp_server.server import run_mcp_server

            server_task = asyncio.create_task(
                run_mcp_server(transport=transport, host=host, port=port)  # type: ignore[arg-type]
            )

            await asyncio.sleep(1)
            inspector_proc = _launch_inspector_sse(host, port)
            try:
                await server_task
            finally:
                if inspector_proc and inspector_proc.poll() is None:
                    inspector_proc.terminate()

        try:
            asyncio.run(_run_with_inspector())
        except KeyboardInterrupt:
            print("\n[shutdown] 종료합니다.")
        return

    # ── 일반 서버 모드 ─────────────────────────────────────
    from app.mcp_server.server import run_mcp_server

    asyncio.run(run_mcp_server(transport=transport, host=host, port=port))  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
