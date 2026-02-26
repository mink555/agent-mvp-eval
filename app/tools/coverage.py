"""보장/급부/조건부 지급 판단 도구."""

from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from app.tools.data import (
    PRODUCTS, COVERAGES, BENEFIT_LIMITS,
    DIAGNOSIS_DEFINITIONS, ICD_MAPPINGS, RIDERS, WAITING_PERIODS,
    _json,
)


# ── Input Schemas ─────────────────────────────────────────────────────────────


class ProductCodeInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00115023)")


class CoverageDetailInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00115023)")
    coverage_type: str = Field(..., description="보장 유형 (예: 암, 사망, 치아, 뇌출혈)")


class BenefitAmountInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00115023)")
    benefit_name: str = Field(..., description="급부명 (예: 암진단비, 사망보험금, 크라운치료)")


class BenefitLimitInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00197011)")
    benefit_name: str = Field(default="", description="급부명 (빈 값이면 전체 한도 조회)")


class EventEligibilityInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00172014)")
    event_facts: str = Field(
        ..., min_length=1, description="사고/질병 사실관계 요약 (예: 뇌출혈 진단, 교통사고 입원)"
    )


class TermLookupInput(BaseModel):
    term: str = Field(..., min_length=1, description="의학/보험 용어 (예: 암, 뇌출혈, 치주질환)")


class IcdLookupInput(BaseModel):
    icd_code: str = Field(..., min_length=1, description="ICD/KCD 질병 코드 (예: C73, I21, K02)")


class MultiBenefitInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00197011)")
    treatments: str = Field(default="", description="치료 내역 (예: 충전+크라운 동시)")


class RiderBundleInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00172014)")
    goal: str = Field(default="", description="계약자 보장 목표 (예: 입원, 수술, 간병)")


# ── Tools ─────────────────────────────────────────────────────────────────────


@tool(args_schema=ProductCodeInput)
def coverage_summary(product_code: str) -> str:
    """상품의 전체 보장 내용을 한번에 요약합니다. 상품 코드(B로 시작) 필수.
    특정 보장 유형 1개만 조회할 때는 coverage_detail 사용."""
    cov = COVERAGES.get(product_code)
    if not cov:
        return _json({"error": f"상품 '{product_code}'의 보장 정보 없음"})
    p = PRODUCTS.get(product_code, {})
    return _json({"product_code": product_code, "name": p.get("name", ""), "coverage": cov})


@tool(args_schema=CoverageDetailInput)
def coverage_detail(product_code: str, coverage_type: str) -> str:
    """상품의 특정 보장 유형(암·사망·치아·입원 등) 1개를 상세 조회합니다.
    전체 보장 요약이 필요하면 coverage_summary 사용."""
    cov = COVERAGES.get(product_code)
    if not cov:
        return _json({"error": f"상품 '{product_code}'의 보장 정보 없음"})
    ct = coverage_type.lower()
    matched = {}
    for section_name, section_data in cov.items():
        if isinstance(section_data, dict):
            for k, v in section_data.items():
                if ct in k.lower():
                    matched[k] = v
        elif isinstance(section_data, str) and ct in section_name.lower():
            matched[section_name] = section_data
    periods = WAITING_PERIODS.get(product_code, {})
    return _json({"product_code": product_code, "coverage_type": coverage_type, "details": matched if matched else cov, "waiting_periods": periods})


@tool(args_schema=BenefitAmountInput)
def benefit_amount_lookup(product_code: str, benefit_name: str) -> str:
    """특정 급부(암진단비·사망보험금·크라운치료 등)의 보장 금액을 조회합니다.
    상품 코드와 급부명을 모두 알 때 사용."""
    cov = COVERAGES.get(product_code)
    if not cov:
        return _json({"error": f"상품 '{product_code}'의 보장 정보 없음"})
    bn = benefit_name.lower()
    found = {}
    for section_data in cov.values():
        if isinstance(section_data, dict):
            for k, v in section_data.items():
                if bn in k.lower() and isinstance(v, str):
                    found[k] = v
    if not found:
        return _json({"product_code": product_code, "benefit_name": benefit_name, "message": "해당 급부를 찾을 수 없습니다. 약관을 확인해 주세요."})
    return _json({"product_code": product_code, "benefit_name": benefit_name, "amounts": found})


@tool(args_schema=BenefitLimitInput)
def benefit_limit_rules(product_code: str, benefit_name: str = "") -> str:
    """급부별 보장 한도 규칙(연간 한도·횟수 제한·감액 조건 등)을 조회합니다."""
    limits = BENEFIT_LIMITS.get(product_code, {})
    bn = benefit_name.lower()
    matched = {k: v for k, v in limits.items() if bn in k.lower()} if bn else limits
    return _json({
        "product_code": product_code, "benefit_name": benefit_name,
        "limits": matched if matched else limits,
        "note": "실제 한도는 약관 기준이며, 여기 표시된 내용은 참고용입니다.",
    })


@tool(args_schema=EventEligibilityInput)
def event_eligibility_check(product_code: str, event_facts: str) -> str:
    """사고/질병 사실관계를 기반으로 보장 대상 여부를 사전 검토합니다."""
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    ef = event_facts.lower()
    assessment = {"likely_covered": True, "conditions": [], "caveats": []}
    periods = WAITING_PERIODS.get(product_code, {})
    if "면책" in periods.get("면책기간", ""):
        assessment["caveats"].append("면책기간 해당 여부를 확인해야 합니다.")
    if "감액" in periods.get("감액기간", ""):
        assessment["caveats"].append("감액기간 해당 여부를 확인해야 합니다.")
    if any(term in ef for term in ["자해", "고의", "음주운전", "범죄"]):
        assessment["likely_covered"] = False
        assessment["conditions"].append("고의/자해/범죄행위는 면책사유에 해당합니다.")
    assessment["disclaimer"] = "최종 보장 여부는 약관과 사실관계 확인 후 결정됩니다."
    return _json({"product_code": product_code, "event_facts": event_facts, "assessment": assessment})


@tool(args_schema=TermLookupInput)
def diagnosis_definition_lookup(term: str) -> str:
    """의학/보험 용어의 약관상 정의를 조회합니다. 예: 암, 뇌출혈, 치주질환."""
    t = term.lower()
    matched = {k: v for k, v in DIAGNOSIS_DEFINITIONS.items() if t in k.lower()}
    if not matched:
        return _json({"term": term, "message": "해당 용어의 정의를 찾을 수 없습니다. 약관을 참조해 주세요."})
    return _json({"term": term, "definitions": matched})


@tool(args_schema=IcdLookupInput)
def icd_mapping_lookup(icd_code: str) -> str:
    """ICD/KCD 질병 코드와 질병명 간의 매핑을 조회합니다."""
    ic = icd_code.upper()
    matched = {k: v for k, v in ICD_MAPPINGS.items() if ic in k.upper() or k.upper() in ic}
    if not matched:
        return _json({"icd_code": icd_code, "message": "해당 ICD 코드를 찾을 수 없습니다."})
    return _json({"icd_code": icd_code, "mappings": matched})


@tool(args_schema=MultiBenefitInput)
def multi_benefit_conflict_rule(product_code: str, treatments: str = "") -> str:
    """동일 사고에 대한 중복 급부/보장 충돌 규칙을 확인합니다."""
    if product_code == "B00197011":
        return _json({
            "product_code": product_code, "treatments": treatments,
            "rule": "동일 치아에 대해 충전치료와 크라운치료가 동시에 발생할 경우, 크라운치료만 인정됩니다.",
            "priority": "크라운 > 충전(동일 치아 기준)",
        })
    return _json({
        "product_code": product_code, "treatments": treatments,
        "rule": "동일 사고에 대한 중복 급부는 약관에서 정한 우선순위에 따릅니다.",
        "priority": "면책 > 감액 > 한도 > 중복지급불가 순서로 적용",
    })


@tool(args_schema=RiderBundleInput)
def rider_bundle_recommend(product_code: str, goal: str = "") -> str:
    """계약자 목표에 맞는 특약 조합을 추천합니다."""
    riders = RIDERS.get(product_code, [])
    if not riders:
        return _json({"error": f"상품 '{product_code}'의 특약 정보 없음"})
    gl = goal.lower()
    recommendations = []
    for r in riders:
        searchable = (r["name"] + r["desc"]).lower()
        if any(kw in searchable for kw in gl.split()) or not gl:
            recommendations.append(r)
    if not recommendations:
        recommendations = riders[:2]
    return _json({
        "product_code": product_code, "goal": goal,
        "recommendations": recommendations[:3],
        "note": "계약자의 보장 니즈와 예산에 따라 최종 결정해 주세요.",
    })


TOOLS = [
    coverage_summary, coverage_detail, benefit_amount_lookup,
    benefit_limit_rules, event_eligibility_check,
    diagnosis_definition_lookup, icd_mapping_lookup,
    multi_benefit_conflict_rule, rider_bundle_recommend,
]
