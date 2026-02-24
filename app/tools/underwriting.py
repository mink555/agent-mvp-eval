"""언더라이팅 도구."""

from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from app.tools.data import (
    PRODUCTS, KNOCKOUT_RULES, WAITING_PERIODS, EXCLUSIONS,
    HIGH_RISK_JOBS, PRODUCT_HISTORY_FLAGS, INSURANCE_AMOUNT_LIMITS,
    _json, _guard_user_info,
)


# ── Input Schemas ─────────────────────────────────────────────────────────────


class PrecheckInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00172014)")
    age: int | None = Field(default=None, description="피보험자 나이. 사용자가 언급하지 않았으면 null")
    gender: str = Field(default="", description="성별 (M: 남성, F: 여성, 빈 값 허용)")
    history_summary: str = Field(
        default="", description="병력/건강 이력 요약 (예: 고혈압 5년, 당뇨 투약 중)"
    )


class ProductCodeInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00115023)")


class DocsRequiredInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00312011)")
    case_type: str = Field(default="", description="심사 유형 (예: 추가심사, 일반)")


class LimitationsInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00155017)")
    age: int = Field(default=0, ge=0, le=120, description="피보험자 나이 (0~120)")


class RenewalEligibilityInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00115023)")
    current_age: int = Field(default=0, ge=0, le=120, description="현재 나이 (0~120)")


class RenewalPremiumNoticeInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00172014)")
    age: int = Field(default=0, ge=0, le=120, description="현재 나이 (0~120)")


class JobCheckInput(BaseModel):
    job_name: str = Field(..., min_length=1, description="직업명 (예: 배달라이더, 건설노동자)")


class DisclosureRiskInput(BaseModel):
    history_summary: str = Field(
        ..., min_length=1, description="병력 요약 (예: 고혈압 10년, 당뇨, 작년 입원)"
    )


# ── Tools ─────────────────────────────────────────────────────────────────────


@tool(args_schema=PrecheckInput)
def underwriting_precheck(product_code: str, age: int | None = None, gender: str = "", history_summary: str = "") -> str:
    """나이·병력 기반 인수 사전 적합성 검토. 특정 고객의 건강 이력으로 가입 가능 여부를 판단할 때 사용.
    상품 규정상 연령·채널 조건만 확인할 때는 eligibility_by_product_rule 사용."""
    guard = _guard_user_info({"나이": age})
    if guard:
        return guard
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"eligible": False, "reason": f"상품 '{product_code}' 없음"})
    knockout_issues, coverage_caveats = [], []
    if age < p.get("min_age", 0):
        knockout_issues.append(f"최소 가입 나이({p['min_age']}세) 미만")
    if age > p.get("max_age", 999):
        knockout_issues.append(f"최대 가입 나이({p['max_age']}세) 초과")
    if history_summary:
        hs = history_summary.lower()
        flag_groups = PRODUCT_HISTORY_FLAGS.get("_common", []) + PRODUCT_HISTORY_FLAGS.get(product_code, [])
        for flag in flag_groups:
            if any(kw in hs for kw in flag["keywords"]):
                (knockout_issues if flag["is_knockout"] else coverage_caveats).append(flag["note"])
        if p.get("simplified_underwriting"):
            for rule in KNOCKOUT_RULES.get("_simplified_common", []):
                if any(kw in hs for kw in ["암", "cancer", "입원", "수술"]):
                    knockout_issues.append(f"간편심사 고지사항 해당: {rule}")
                    break
    eligible = len(knockout_issues) == 0
    needs_expert_review = bool(coverage_caveats) or (bool(history_summary) and not knockout_issues and not coverage_caveats)
    return _json({
        "product_code": product_code, "product_name": p["name"], "age": age, "gender": gender,
        "eligible": eligible, "knockout_issues": knockout_issues,
        "coverage_caveats": coverage_caveats, "needs_expert_review": needs_expert_review,
        "simplified_underwriting": p.get("simplified_underwriting", False),
    })


@tool(args_schema=ProductCodeInput)
def underwriting_questions_generator(product_code: str) -> str:
    """상품별 인수심사 고지 질문 목록을 생성합니다."""
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    if p.get("simplified_underwriting"):
        questions = [
            "최근 3개월 이내에 의사로부터 입원·수술·추가검사를 권유받은 사실이 있습니까?",
            "최근 2년 이내에 입원하거나 수술을 받은 사실이 있습니까?",
            "최근 5년 이내에 암으로 진단받거나 치료를 받은 사실이 있습니까?",
        ]
        mode = "간편심사(3대 고지)"
    else:
        questions = [
            "최근 3개월 이내에 의사로부터 진찰·검사를 통해 질병 의심 소견을 받은 사실이 있습니까?",
            "최근 1년 이내에 의사의 진찰·검사·치료·투약을 받은 사실이 있습니까?",
            "최근 5년 이내에 입원·수술을 받은 사실이 있습니까?",
            "과거에 암·심장질환·뇌질환·간질환·정신질환으로 진단·치료를 받은 사실이 있습니까?",
        ]
        mode = "일반심사"
    return _json({"product_code": product_code, "mode": mode, "questions": questions})


@tool(args_schema=ProductCodeInput)
def underwriting_knockout_rules(product_code: str) -> str:
    """상품별 인수 거절(녹아웃) 조건을 조회합니다."""
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    rules = KNOCKOUT_RULES.get(product_code, [])
    if p.get("simplified_underwriting"):
        rules = KNOCKOUT_RULES.get("_simplified_common", []) + rules
    return _json({"product_code": product_code, "name": p["name"], "knockout_rules": rules})


@tool(args_schema=DocsRequiredInput)
def underwriting_docs_required(product_code: str, case_type: str = "") -> str:
    """가입 시 필요한 서류 목록을 확인합니다."""
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    base_docs = ["청약서", "신분증 사본"]
    additional = []
    if p.get("simplified_underwriting"):
        additional.append("간편심사 고지서(3대 고지)")
    else:
        additional.append("표준 건강고지서")
    if case_type and "추가" in case_type:
        additional.append("건강진단서(회사 지정 병원)")
    return _json({"product_code": product_code, "base_docs": base_docs, "additional_docs": additional})


@tool(args_schema=ProductCodeInput)
def underwriting_waiting_periods(product_code: str) -> str:
    """상품의 면책기간·감액기간·보장개시일(언제부터 보장되는지)을 조회합니다.
    보장이 안 되는 사유(면책 사유 목록)는 underwriting_exclusions 사용."""
    periods = WAITING_PERIODS.get(product_code)
    if not periods:
        return _json({"error": f"상품 '{product_code}'의 기간 정보 없음"})
    return _json({"product_code": product_code, "periods": periods})


@tool(args_schema=ProductCodeInput)
def underwriting_exclusions(product_code: str) -> str:
    """상품의 보장 제외(면책) 사유 목록을 조회합니다. 어떤 경우에 보험금이 안 나오는지 확인할 때 사용.
    면책기간(기간 조회)은 underwriting_waiting_periods 사용."""
    common = EXCLUSIONS.get("_common", [])
    specific = EXCLUSIONS.get(product_code, [])
    return _json({"product_code": product_code, "common_exclusions": common, "product_exclusions": specific})


@tool(args_schema=LimitationsInput)
def underwriting_limitations(product_code: str, age: int = 0) -> str:
    """상품별 최소/최대 가입금액 한도를 조회합니다."""
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    limit = {**INSURANCE_AMOUNT_LIMITS.get(product_code, {"min": None, "max": None, "unit": "정보 없음"})}
    if age and age > 60 and limit.get("max"):
        limit["max"] = int(limit["max"] * 0.7)
        limit["note"] = "60세 초과 시 최대 가입금액 축소 가능"
    return _json({"product_code": product_code, "limits": limit})


@tool(args_schema=ProductCodeInput)
def underwriting_reinstatement_rule(product_code: str) -> str:
    """실효 계약 부활(복원) 규정을 조회합니다."""
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    return _json({
        "product_code": product_code,
        "rule": {
            "부활가능기간": "실효일로부터 3년 이내",
            "필요사항": "연체보험료 + 이자 납입, 건강고지서 재작성",
            "면책재적용": "부활일부터 면책기간 재적용(암 90일 등)",
            "감액재적용": "부활일부터 감액기간 재적용",
        },
    })


@tool(args_schema=RenewalEligibilityInput)
def underwriting_renewal_eligibility(product_code: str, current_age: int = 0) -> str:
    """갱신 가능 여부 및 최대 갱신 나이를 확인합니다."""
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    max_age = p.get("max_renewal_age")
    if not max_age:
        return _json({"product_code": product_code, "renewable": False, "reason": "비갱신형 상품"})
    eligible = current_age < max_age
    return _json({
        "product_code": product_code, "current_age": current_age,
        "max_renewal_age": max_age, "eligible": eligible,
        "reason": "갱신 가능" if eligible else f"최대 갱신 나이({max_age}세) 초과",
        "note": "갱신 시 보험료는 갱신일 기준 위험률로 재산정됩니다.",
    })


@tool(args_schema=RenewalPremiumNoticeInput)
def underwriting_renewal_premium_notice(product_code: str, age: int = 0) -> str:
    """갱신 시 예상 보험료 인상 범위를 안내합니다."""
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    if p.get("renewal_type") != "갱신형":
        return _json({"product_code": product_code, "message": "비갱신형 상품으로 갱신 보험료 변동이 없습니다."})
    increase_pct = round(3 + (age - 30) * 0.5, 1) if age >= 30 else 3.0
    return _json({
        "product_code": product_code, "age": age,
        "estimated_increase_range": f"{increase_pct}% ~ {increase_pct + 5}%",
        "reason": "갱신 시 해당 나이의 위험률이 적용되어 보험료가 변동됩니다.",
    })


@tool(args_schema=JobCheckInput)
def underwriting_high_risk_job_check(job_name: str) -> str:
    """직업 위험도를 확인합니다. 가입 제한/할증 여부를 판단합니다."""
    jn = job_name.lower()
    for category, jobs in HIGH_RISK_JOBS.items():
        for j in jobs:
            if jn in j.lower() or j.lower() in jn:
                return _json({"job": job_name, "category": category, "restriction": category, "detail": f"'{job_name}'은(는) {category} 직군에 해당합니다."})
    return _json({"job": job_name, "category": "일반", "restriction": "없음", "detail": "특별한 가입 제한이 없는 직군입니다."})


@tool(args_schema=DisclosureRiskInput)
def underwriting_disclosure_risk_score(history_summary: str) -> str:
    """병력 요약을 기반으로 고지 위험 점수를 산출합니다."""
    hs = history_summary.lower()
    risk_score = 0
    warnings = []
    high_risk_terms = {"암": 5, "cancer": 5, "뇌졸중": 4, "심근경색": 4, "당뇨": 3, "고혈압": 2, "수술": 3, "입원": 2}
    for term, score in high_risk_terms.items():
        if term in hs:
            risk_score += score
            warnings.append(f"'{term}' 관련 이력 → 반드시 고지 필요")
    level = "낮음" if risk_score < 3 else ("중간" if risk_score < 6 else "높음")
    return _json({
        "history_summary": history_summary,
        "risk_score": risk_score, "risk_level": level, "warnings": warnings,
        "recommendation": "정확한 고지가 중요합니다. 고지의무 위반 시 보험금 지급 거절 사유가 될 수 있습니다.",
    })


TOOLS = [
    underwriting_precheck, underwriting_questions_generator,
    underwriting_knockout_rules, underwriting_docs_required,
    underwriting_waiting_periods, underwriting_exclusions,
    underwriting_limitations, underwriting_reinstatement_rule,
    underwriting_renewal_eligibility, underwriting_renewal_premium_notice,
    underwriting_high_risk_job_check, underwriting_disclosure_risk_score,
]
