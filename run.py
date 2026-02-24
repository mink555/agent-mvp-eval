"""앱 실행 진입점."""

import os

import uvicorn

from app.config import get_settings


def main():
    s = get_settings()
    is_dev = os.getenv("ENV", "production").lower() == "development"
    uvicorn.run(
        "app.main:app",
        host=s.api_host,
        port=s.api_port,
        reload=is_dev,
        log_level=s.log_level.lower(),
    )


if __name__ == "__main__":
    main()
