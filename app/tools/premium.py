"""보험료/플랜/설계 도구."""

from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from app.tools.data import PRODUCTS, PREMIUM_TABLES, SURRENDER_VALUE_RULES, _json, _guard_user_info


# ── Input Schemas ─────────────────────────────────────────────────────────────


class PremiumEstimateInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00115023)")
    age: int | None = Field(default=None, description="피보험자 나이. 사용자가 언급하지 않았으면 null")
    gender: str | None = Field(default=None, description="성별 (M/F). 사용자가 언급하지 않았으면 null")


class PremiumCompareInput(BaseModel):
    codes: str = Field(
        ..., description="비교할 상품 코드 목록 (쉼표 또는 공백 구분, 예: B00115023, B00197011)"
    )
    age: int | None = Field(default=None, description="피보험자 나이. 사용자가 언급하지 않았으면 null")
    gender: str | None = Field(default=None, description="성별 (M/F). 사용자가 언급하지 않았으면 null")


class PlanOptionsInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00312011)")
    age: int | None = Field(default=None, description="피보험자 나이. 사용자가 언급하지 않았으면 null")
    gender: str | None = Field(default=None, description="성별 (M/F). 사용자가 언급하지 않았으면 null")


class AmountSuggestInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00155017)")
    income: int = Field(default=0, ge=0, description="월 소득 (만원 단위)")
    goal: str = Field(default="", description="보장 목적 (예: 사망보장, 암 대비)")


class RenewalProjectionInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (갱신형 상품, 예: B00115023)")
    age: int | None = Field(default=None, description="현재 나이. 사용자가 언급하지 않았으면 null")
    gender: str | None = Field(default=None, description="성별 (M 또는 F). 사용자가 언급하지 않았으면 null")
    horizon: int = Field(default=20, ge=1, le=50, description="추정 기간 (년)")


class AffordabilityInput(BaseModel):
    budget: int = Field(..., ge=1000, description="월 예산 (원 단위, 최소 1,000원)")
    product_code: str = Field(..., description="상품 코드 (예: B00172014)")
    age: int | None = Field(default=None, description="피보험자 나이. 사용자가 언급하지 않았으면 null")
    gender: str | None = Field(default=None, description="성별 (M/F). 사용자가 언급하지 않았으면 null")


class ProductCodeInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00197011)")


class SurrenderValueInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00312011)")
    plan_type: str = Field(default="", description="플랜 유형 (예: 1종, 2종)")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _calc_premium(product_code: str, age: int, gender: str, amount_factor: float = 1.0) -> int | None:
    table = PREMIUM_TABLES.get(product_code)
    if not table:
        return None
    base = table["base"]
    age_f = 1.0 + (age - 30) * table["age_factor"]
    gender_f = table.get(f"gender_{gender.lower()}", 1.0)
    return max(int(base * age_f * gender_f * amount_factor), 5000)


# ── Tools ─────────────────────────────────────────────────────────────────────


@tool(args_schema=PremiumEstimateInput)
def premium_estimate(product_code: str, age: int | None = None, gender: str | None = None) -> str:
    """보험 상품의 예상 월 보험료를 산출합니다. 나이와 성별이 필요합니다."""
    guard = _guard_user_info({"나이": age, "성별": gender})
    if guard:
        return guard
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    monthly = _calc_premium(product_code, age, gender)
    if monthly is None:
        return _json({"error": "보험료 테이블 없음"})
    return _json({
        "product_code": product_code, "name": p["name"],
        "age": age, "gender": gender,
        "estimated_monthly_premium": f"{monthly:,}원",
        "note": "이 금액은 예시이며, 실제 보험료는 상품·보장내용·건강상태에 따라 달라집니다.",
    })


@tool(args_schema=PremiumCompareInput)
def premium_compare(codes: str, age: int | None = None, gender: str | None = None) -> str:
    """여러 상품의 보험료를 비교합니다. 코드를 쉼표/공백으로 구분합니다."""
    guard = _guard_user_info({"나이": age, "성별": gender})
    if guard:
        return guard
    code_list = [c.strip() for c in codes.replace(",", " ").split() if c.strip()]
    comparison = []
    for code in code_list:
        p = PRODUCTS.get(code)
        if not p:
            continue
        monthly = _calc_premium(code, age, gender)
        comparison.append({
            "code": code, "name": p["name"],
            "monthly": f"{monthly:,}원" if monthly else "산출불가",
        })
    return _json({"age": age, "gender": gender, "comparison": comparison})


@tool(args_schema=PlanOptionsInput)
def plan_options(product_code: str, age: int | None = None, gender: str | None = None) -> str:
    """상품의 납입 기간·납입 방식 플랜 옵션(10년납·20년납·전기납 등)을 조회합니다.
    보험료 금액 산출은 premium_estimate 사용."""
    guard = _guard_user_info({"나이": age, "성별": gender})
    if guard:
        return guard
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    options = []
    term = p.get("term_years")
    if term:
        options.append({"payment_term": f"{term}년납(전기납)", "type": "전기납"})
    if p.get("renewal_type") == "비갱신형(종신)":
        for t in [10, 15, 20]:
            if age + t <= 80:
                monthly = _calc_premium(product_code, age, gender, t / 20)
                options.append({"payment_term": f"{t}년납", "estimated_monthly": f"{monthly:,}원" if monthly else "산출불가"})
    return _json({"product_code": product_code, "name": p["name"], "age": age, "gender": gender, "options": options, "plan_types": p.get("plan_types", [])})


@tool(args_schema=AmountSuggestInput)
def amount_suggest(product_code: str, income: int = 0, goal: str = "") -> str:
    """소득·목적 기반 적정 보험 가입금액(보험금)을 추천합니다.
    보험료(월납입액) 계산은 premium_estimate 사용."""
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    category = p.get("category", "")
    if "사망" in category:
        suggested = max(income * 36, 3000) if income else 3000
        unit, reason = "만원", "사망보장은 일반적으로 연소득의 3~5배 수준을 권장합니다."
    elif "암" in category or "건강" in category:
        suggested, unit, reason = 2000, "만원", "암/질병 진단비는 치료비 수준을 고려하여 설정합니다."
    elif "치아" in category:
        suggested, unit, reason = None, "정액", "치아보험은 치아 1개당 정액 지급 구조입니다."
    else:
        suggested, unit, reason = 1000, "만원", "보장 목표에 따라 적정 금액을 설정해 주세요."
    return _json({"product_code": product_code, "name": p["name"], "suggested_amount": suggested, "unit": unit, "reason": reason})


@tool(args_schema=RenewalProjectionInput)
def renewal_premium_projection(product_code: str, age: int | None = None, gender: str | None = None, horizon: int = 20) -> str:
    """갱신형 상품의 갱신 시점별 예상 보험료를 추정합니다."""
    guard = _guard_user_info({"나이": age, "성별": gender})
    if guard:
        return guard
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    if p.get("renewal_type") != "갱신형":
        return _json({"product_code": product_code, "message": "비갱신형 상품으로 보험료 변동이 없습니다."})
    term = p.get("term_years", 10)
    projections = []
    for i in range(0, horizon, term):
        future_age = age + i
        monthly = _calc_premium(product_code, future_age, gender)
        if monthly:
            projections.append({"age": future_age, "monthly": f"{monthly:,}원"})
    return _json({
        "product_code": product_code, "name": p["name"],
        "renewal_type": "갱신형", "term_years": term,
        "age": age, "projections": projections,
        "note": "예시 금액이며 실제 갱신보험료는 갱신 시점 위험률로 결정됩니다.",
    })


@tool(args_schema=AffordabilityInput)
def affordability_check(budget: int, product_code: str, age: int | None = None, gender: str | None = None) -> str:
    """월 예산 내에서 보험 가입이 가능한지 확인합니다."""
    guard = _guard_user_info({"나이": age, "성별": gender})
    if guard:
        return guard
    p = PRODUCTS.get(product_code)
    monthly = _calc_premium(product_code, age, gender)
    if monthly is None:
        return _json({"error": "보험료 산출 불가"})
    fits = monthly <= budget
    return _json({
        "product_code": product_code, "name": p["name"] if p else product_code,
        "budget": f"{budget:,}원", "estimated_premium": f"{monthly:,}원",
        "fits": fits,
        "message": "예산 내 가입 가능합니다." if fits else f"예상 보험료({monthly:,}원)가 예산({budget:,}원)을 초과합니다.",
    })


@tool(args_schema=ProductCodeInput)
def payment_cycle_options(product_code: str) -> str:
    """상품의 납입 주기 옵션(월납/연납 등)을 조회합니다."""
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    return _json({"product_code": product_code, "name": p["name"], "available_cycles": p.get("payment_cycles", ["월납"])})


@tool(args_schema=SurrenderValueInput)
def surrender_value_explain(product_code: str, plan_type: str = "") -> str:
    """해지(해약) 시 환급금 규정을 설명합니다. 해약·중도 해지·환급금 관련 질문에 사용."""
    rules = SURRENDER_VALUE_RULES.get(product_code, {})
    if not rules:
        rules = {"info": SURRENDER_VALUE_RULES["_default"]}
    if plan_type:
        matched = {k: v for k, v in rules.items() if plan_type in k}
        if matched:
            rules = matched
    p = PRODUCTS.get(product_code)
    return _json({"product_code": product_code, "name": p["name"] if p else product_code, "plan_type": plan_type or "전체", "surrender_value": rules})


TOOLS = [
    premium_estimate, premium_compare, plan_options, amount_suggest,
    renewal_premium_projection, affordability_check,
    payment_cycle_options, surrender_value_explain,
]
