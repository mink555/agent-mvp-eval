"""SQLite3 가짜 고객/계약 DB 생성 및 시드 데이터.

운영 환경에서는 실제 DB/API로 대체.
테스트·개발 시 `init_db()`를 호출하면 인메모리 또는 파일 DB가 준비된다.
"""

from __future__ import annotations

import sqlite3
import os
import threading
from pathlib import Path

DB_PATH = os.getenv("CUSTOMER_DB_PATH", str(Path(__file__).resolve().parent.parent.parent / "customer.db"))

_conn: sqlite3.Connection | None = None


def get_connection() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
    return _conn


def init_db(seed: bool = True) -> sqlite3.Connection:
    """테이블 생성 + 시드 데이터 삽입. 이미 존재하면 스킵."""
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id   TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            age           INTEGER NOT NULL,
            gender        TEXT NOT NULL CHECK(gender IN ('M', 'F')),
            phone         TEXT
        );

        CREATE TABLE IF NOT EXISTS contracts (
            contract_id   TEXT PRIMARY KEY,
            customer_id   TEXT NOT NULL REFERENCES customers(customer_id),
            product_code  TEXT NOT NULL,
            product_name  TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'active'
                          CHECK(status IN ('active', 'lapsed', 'terminated', 'expired')),
            start_date    TEXT NOT NULL,
            end_date      TEXT,
            channel       TEXT
        );

        CREATE TABLE IF NOT EXISTS enrollment_rules (
            product_code      TEXT NOT NULL,
            rule_type         TEXT NOT NULL,
            rule_description  TEXT NOT NULL,
            rule_value        TEXT,
            PRIMARY KEY (product_code, rule_type)
        );
    """)

    if seed and not _has_data(cur):
        _seed_data(cur)

    conn.commit()
    return conn


def _has_data(cur: sqlite3.Cursor) -> bool:
    cur.execute("SELECT COUNT(*) FROM customers")
    return cur.fetchone()[0] > 0


def _seed_data(cur: sqlite3.Cursor) -> None:
    # ── 고객 ──
    customers = [
        ("C001", "김민수", 45, "M", "010-1234-5678"),
        ("C002", "이영희", 38, "F", "010-2345-6789"),
        ("C003", "박철수", 52, "M", "010-3456-7890"),
        ("C004", "최수진", 60, "F", "010-4567-8901"),
        ("C005", "정대영", 33, "M", "010-5678-9012"),
        ("C006", "한미라", 28, "F", "010-6789-0123"),
        ("C007", "오세훈", 47, "M", "010-7890-1234"),
        ("C008", "윤지은", 55, "F", "010-8901-2345"),
    ]
    cur.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers
    )

    # ── 계약 ──
    contracts = [
        # 김민수: 치아보험 + 암보험 보유
        ("CT-001", "C001", "B00197011", "무배당 THE 건강한치아보험 V", "active", "2023-01-15", "2033-01-15", "TM"),
        ("CT-002", "C001", "B00115023", "무배당 뉴스타트암보험(갱신형)", "active", "2023-06-01", "2033-06-01", "CM"),
        # 이영희: 치아보험만
        ("CT-003", "C002", "B00197011", "무배당 THE 건강한치아보험 V", "active", "2022-09-01", "2032-09-01", "TM"),
        # 박철수: 간편건강 + 간편정기
        ("CT-004", "C003", "B00172014", "무배당 THE 간편한건강보험(갱신형)", "active", "2024-01-01", "2034-01-01", "온라인"),
        ("CT-005", "C003", "B00155017", "무배당 THE 간편고지정기보험(갱신형)", "active", "2024-01-01", "2034-01-01", "온라인"),
        # 최수진: 종신보험 + 해지된 암보험
        ("CT-006", "C004", "B00312011", "무배당 THE 간편고지종신보험", "active", "2023-03-01", None, "CM"),
        ("CT-007", "C004", "B00115023", "무배당 뉴스타트암보험(갱신형)", "terminated", "2021-01-01", "2023-12-31", "TM"),
        # 정대영: 계약 없음 (신규)
        # 한미라: 치아보험 보유
        ("CT-008", "C006", "B00197011", "무배당 THE 건강한치아보험 V", "active", "2024-06-01", "2034-06-01", "TM"),
        # 오세훈: 암보험 보유
        ("CT-009", "C007", "B00115023", "무배당 뉴스타트암보험(갱신형)", "active", "2023-09-01", "2033-09-01", "CM"),
        # 윤지은: 간편건강 + 치아 + 종신
        ("CT-010", "C008", "B00172014", "무배당 THE 간편한건강보험(갱신형)", "active", "2022-06-01", "2032-06-01", "TM"),
        ("CT-011", "C008", "B00197011", "무배당 THE 건강한치아보험 V", "active", "2023-01-01", "2033-01-01", "CM"),
        ("CT-012", "C008", "B00312011", "무배당 THE 간편고지종신보험", "active", "2024-01-01", None, "온라인"),
    ]
    cur.executemany(
        "INSERT INTO contracts VALUES (?, ?, ?, ?, ?, ?, ?, ?)", contracts
    )

    # ── 가입 규칙 ──
    rules = [
        ("B00115023", "prerequisite",   "기존 당사 암보험 정상 유지 고객만 가입 가능", "requires_active:B00115023"),
        ("B00115023", "max_concurrent", "동일 상품 중복 가입 불가 (1인 1계약)", "1"),
        ("B00197011", "max_concurrent", "치아보험 동일 상품 중복 가입 불가 (1인 1계약)", "1"),
        ("B00197011", "same_category_limit", "치아보험 카테고리 내 동시 가입 최대 2건", "2"),
        ("B00172014", "max_concurrent", "동일 상품 중복 가입 불가", "1"),
        ("B00172014", "prerequisite",   "간편심사 대상자만 가입 가능 (일반심사 가입 가능 시 일반심사 상품 우선)", None),
        ("B00155017", "max_concurrent", "동일 상품 중복 가입 불가", "1"),
        ("B00312011", "max_concurrent", "동일 상품 중복 가입 불가", "1"),
        ("B00312011", "same_category_limit", "사망보험 카테고리 내 동시 가입 최대 2건", "2"),
    ]
    cur.executemany(
        "INSERT INTO enrollment_rules VALUES (?, ?, ?, ?)", rules
    )


_db_ready = False
_db_lock = threading.Lock()


def ensure_db_ready() -> None:
    """DB 초기화를 1회만 수행하는 가드. tool에서 호출.

    double-checked locking으로 asyncio.to_thread 환경에서의 race condition을 방지한다.
    """
    global _db_ready
    if _db_ready:
        return
    with _db_lock:
        if _db_ready:
            return
        init_db(seed=True)
        _db_ready = True


def close_db() -> None:
    global _conn, _db_ready
    with _db_lock:
        if _conn:
            _conn.close()
            _conn = None
        _db_ready = False
