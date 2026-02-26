"""계약자/계약 DB 조회 도구 — SQLite3 기반."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from app.tools.db_setup import get_connection, ensure_db_ready
from app.tools.data import PRODUCTS, _json


_STATUS_MAP = {
    "active": "유지",
    "terminated": "해지",
    "lapsed": "실효",
    "expired": "만기",
}

_GENDER_MAP = {"M": "남성", "F": "여성"}


# ── Input Schemas ─────────────────────────────────────────────────────────────


class ContractLookupInput(BaseModel):
    customer_id: str = Field(default="", description="계약자 ID (예: C001)")
    customer_name: str = Field(default="", description="계약자 이름 (부분 일치 검색)")


class DuplicateCheckInput(BaseModel):
    customer_id: str = Field(..., description="계약자 ID (예: C001)")
    product_code: str = Field(..., description="가입 확인할 상품 코드 (예: B00115023)")


class CustomerSearchInput(BaseModel):
    name: str = Field(default="", description="계약자 이름 (부분 일치)")
    age_min: int = Field(default=0, ge=0, le=120, description="최소 나이")
    age_max: int = Field(default=200, ge=0, le=200, description="최대 나이 (200이면 제한 없음)")
    gender: str = Field(default="", description="성별 필터 (M/F, 빈 값이면 전체)")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _rows_to_dicts(rows) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


def _safe_customer(d: dict[str, Any]) -> dict[str, Any]:
    """개인정보(phone 등)를 제외한 보험 용어 기반 계약자 정보만 반환."""
    return {
        "계약자ID": d.get("customer_id", ""),
        "이름": d.get("name", ""),
        "나이": d.get("age", ""),
        "성별": _GENDER_MAP.get(d.get("gender", ""), d.get("gender", "")),
    }


def _safe_contract(d: dict[str, Any]) -> dict[str, Any]:
    """계약 정보를 보험 용어 필드명으로 변환하여 반환."""
    status = d.get("status", "")
    result: dict[str, Any] = {
        "상품명": d.get("product_name", ""),
        "피보험자": d.get("insured_name", ""),
        "계약상태": _STATUS_MAP.get(status, status),
        "계약일": d.get("start_date", ""),
        "채널": d.get("channel", ""),
    }
    if d.get("end_date"):
        result["만료일"] = d["end_date"]
    if d.get("terminated_date"):
        result["해지일"] = d["terminated_date"]
    if d.get("renewal_date"):
        result["갱신일"] = d["renewal_date"]
    return result


# ── Tools ─────────────────────────────────────────────────────────────────────


@tool(args_schema=ContractLookupInput)
def customer_contract_lookup(customer_id: str = "", customer_name: str = "") -> str:
    """계약자 ID 또는 이름으로 기존 계약 목록을 조회합니다."""
    ensure_db_ready()
    conn = get_connection()
    if not customer_id and not customer_name:
        return _json({"error": "customer_id 또는 customer_name 중 하나를 입력해 주세요."})
    if customer_id:
        cust = conn.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
    else:
        cust = conn.execute("SELECT * FROM customers WHERE name LIKE ?", (f"%{customer_name}%",)).fetchone()
    if not cust:
        return _json({"error": f"계약자를 찾을 수 없습니다. (검색: {customer_id or customer_name})"})
    cust_dict = dict(cust)
    cid = cust_dict["customer_id"]
    raw_contracts = [dict(c) for c in conn.execute(
        "SELECT * FROM contracts WHERE customer_id = ? ORDER BY start_date DESC", (cid,)
    ).fetchall()]
    contract_list = [_safe_contract(c) for c in raw_contracts]
    active_count = sum(1 for c in raw_contracts if c["status"] == "active")
    return _json({
        "계약자": _safe_customer(cust_dict),
        "계약목록": contract_list,
        "총계약수": len(contract_list),
        "유지계약수": active_count,
    })


@tool(args_schema=DuplicateCheckInput)
def duplicate_enrollment_check(customer_id: str, product_code: str) -> str:
    """계약자의 기존 계약을 기반으로 특정 상품 추가 가입 가능 여부를 판단합니다."""
    ensure_db_ready()
    conn = get_connection()
    cust = conn.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
    if not cust:
        return _json({"error": f"계약자 '{customer_id}'을(를) 찾을 수 없습니다."})
    product = PRODUCTS.get(product_code)
    if not product:
        return _json({"error": f"상품 '{product_code}'을(를) 찾을 수 없습니다."})
    active_contracts = _rows_to_dicts(conn.execute("SELECT * FROM contracts WHERE customer_id = ? AND status = 'active'", (customer_id,)).fetchall())
    rules = _rows_to_dicts(conn.execute("SELECT * FROM enrollment_rules WHERE product_code = ?", (product_code,)).fetchall())
    blockers, warnings = [], []
    cust_age = dict(cust)["age"]
    min_age, max_age = product.get("min_age", 0), product.get("max_age", 999)
    if not (min_age <= cust_age <= max_age):
        blockers.append(f"나이 제한: {min_age}~{max_age}세 가입 가능, 계약자 나이 {cust_age}세")
    for rule in rules:
        rtype, rval = rule["rule_type"], rule["rule_value"]
        if rtype == "max_concurrent":
            same = [c for c in active_contracts if c["product_code"] == product_code]
            limit = int(rval) if rval else 1
            if len(same) >= limit:
                blockers.append(f"동일 상품 중복 가입 불가: 현재 {len(same)}건 (최대 {limit}건)")
        elif rtype == "same_category_limit":
            cat_key = product.get("category", "").split("/")[0]
            same_cat = [c for c in active_contracts if cat_key in PRODUCTS.get(c["product_code"], {}).get("category", "")]
            limit = int(rval) if rval else 1
            if len(same_cat) >= limit:
                blockers.append(f"동일 카테고리({cat_key}) 한도 초과: 현재 {len(same_cat)}건 (최대 {limit}건)")
            elif same_cat:
                warnings.append(f"동일 카테고리({cat_key}) {len(same_cat)}건 유지 중 (한도 {limit}건)")
        elif rtype == "prerequisite":
            if rval and rval.startswith("requires_active:"):
                req_code = rval.split(":")[1]
                has = any(c["product_code"] == req_code and c["status"] == "active" for c in active_contracts)
                if not has:
                    blockers.append(f"전제조건 미충족: {rule['rule_description']}")
    eligible = len(blockers) == 0
    return _json({
        "계약자": _safe_customer(dict(cust)),
        "상품": {"상품명": product["name"]},
        "가입가능": eligible,
        "제한사유": blockers,
        "참고": warnings,
        "요약": "가입 가능합니다." if eligible else f"가입 불가: {'; '.join(blockers)}",
    })


@tool(args_schema=CustomerSearchInput)
def customer_search(name: str = "", age_min: int = 0, age_max: int = 200, gender: str = "") -> str:
    """조건(이름, 나이, 성별)으로 계약자를 검색합니다."""
    ensure_db_ready()
    conn = get_connection()
    conditions, params = ["1=1"], []
    if name:
        conditions.append("name LIKE ?")
        params.append(f"%{name}%")
    if age_min:
        conditions.append("age >= ?")
        params.append(age_min)
    if age_max and age_max < 200:
        conditions.append("age <= ?")
        params.append(age_max)
    if gender:
        conditions.append("gender = ?")
        params.append(gender.upper())
    query = f"SELECT * FROM customers WHERE {' AND '.join(conditions)}"
    rows = conn.execute(query, params).fetchall()
    return _json({"계약자목록": [_safe_customer(dict(r)) for r in rows], "총인원": len(rows)})


TOOLS = [
    customer_contract_lookup, duplicate_enrollment_check, customer_search,
]
