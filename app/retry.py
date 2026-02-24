"""재시도 전략 — tenacity 기반 재사용 가능한 retry 데코레이터.

사용법:
    from app.retry import llm_retry, db_retry

    @llm_retry
    def call_openai(llm, messages):
        return llm.invoke(messages)

    @db_retry
    def query_db(collection, query):
        return collection.query(query_texts=[query])
"""

from __future__ import annotations

import logging

import openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_LLM_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

_DB_RETRYABLE = (ConnectionError, TimeoutError, OSError)

llm_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(_LLM_RETRYABLE),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

db_retry = retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
    retry=retry_if_exception_type(_DB_RETRYABLE),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
