"""청구/사후/유지관리 도구."""

from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from app.tools.data import CLAIM_GUIDES, CLAIM_FORMS, CONTRACT_ACTIONS, _json


# ── Input Schemas ─────────────────────────────────────────────────────────────


class ClaimGuideInput(BaseModel):
    claim_type: str = Field(
        ..., min_length=1,
        description="청구 유형 (예: 사망, 암진단, 입원, 수술, 치과, 뇌출혈, 심근경색)",
    )
    product_code: str = Field(default="", description="상품 코드 (빈 값이면 공통 안내)")


class ClaimFormsInput(BaseModel):
    product_code: str = Field(default="", description="상품 코드 (빈 값이면 전체)")
    claim_type: str = Field(default="", description="청구 유형 (예: 치과, 암)")


class ContractManageInput(BaseModel):
    action: str = Field(
        ..., min_length=1,
        description="계약 관리 행위 (조회, 갱신, 해지, 대출, 변경, 부활)",
    )
    contract_id: str = Field(default="", description="계약 ID (빈 값이면 일반 안내)")


class FollowupInput(BaseModel):
    case_summary: str = Field(
        ..., min_length=1,
        description="상담 사례 요약 (예: 신규 가입 상담, 해지 문의, 보험금 청구)",
    )


# ── Tools ─────────────────────────────────────────────────────────────────────


@tool(args_schema=ClaimGuideInput)
def claim_guide(claim_type: str, product_code: str = "") -> str:
    """보험금 청구 유형별 절차와 안내를 제공합니다. 예: 사망, 진단, 입원."""
    ct = claim_type.lower()
    result = {"공통": CLAIM_GUIDES["공통"]}
    for key, val in CLAIM_GUIDES.items():
        if key == "공통":
            continue
        if ct in key.lower():
            result[key] = val
    if len(result) == 1:
        result["note"] = f"'{claim_type}' 유형에 해당하는 세부 가이드를 찾지 못했습니다. 아래 공통 안내를 참고해 주세요."
    return _json({"claim_type": claim_type, "product_code": product_code or "전체", "guide": result})


@tool(args_schema=ClaimFormsInput)
def claim_required_forms(product_code: str = "", claim_type: str = "") -> str:
    """청구 유형별 필요 서류 및 양식을 조회합니다."""
    ct = claim_type.lower()
    matched = {}
    for form_name, fields in CLAIM_FORMS.items():
        if ct in form_name.lower():
            matched[form_name] = fields
    if not matched:
        matched = {"안내": [f"'{claim_type}' 유형의 양식 정보가 없습니다. 고객센터(1588-0058)에 문의해 주세요."]}
    return _json({"claim_type": claim_type, "product_code": product_code or "전체", "forms": matched})


@tool(args_schema=ContractManageInput)
def contract_manage(action: str, contract_id: str = "") -> str:
    """계약 관리(조회/갱신/해지/대출/변경/부활) 절차를 안내합니다."""
    act = action.lower()
    matched = {key: val for key, val in CONTRACT_ACTIONS.items() if act in key.lower()}
    if not matched:
        matched = CONTRACT_ACTIONS
    return _json({"action": action, "contract_id": contract_id or "미지정", "info": matched})


@tool(args_schema=FollowupInput)
def customer_followup_tasks(case_summary: str) -> str:
    """상담 사례 요약을 기반으로 후속 조치 목록을 생성합니다."""
    tasks = []
    cs = case_summary.lower()
    if any(kw in cs for kw in ["가입", "신규", "청약"]):
        tasks.extend([
            {"task": "고지사항 정확성 재확인", "priority": "높음"},
            {"task": "필수 설명사항 안내 완료 확인", "priority": "높음"},
            {"task": "청약서 서명/접수 처리", "priority": "높음"},
        ])
    if any(kw in cs for kw in ["청구", "보험금", "사고"]):
        tasks.extend([
            {"task": "필요 서류 안내 및 접수 확인", "priority": "높음"},
            {"task": "청구 진행 상태 안내", "priority": "중간"},
        ])
    if any(kw in cs for kw in ["갱신", "만기"]):
        tasks.extend([
            {"task": "갱신 보험료 안내", "priority": "높음"},
            {"task": "갱신/미갱신 의사 확인", "priority": "높음"},
        ])
    if any(kw in cs for kw in ["해지", "해약"]):
        tasks.extend([
            {"task": "해약환급금 안내", "priority": "높음"},
            {"task": "해지 철회 의사 재확인", "priority": "중간"},
            {"task": "유지 방안(감액완납/납입일시중지 등) 안내", "priority": "중간"},
        ])
    if any(kw in cs for kw in ["불만", "민원", "컴플레인"]):
        tasks.extend([
            {"task": "민원 내용 정리 및 상위 보고", "priority": "높음"},
            {"task": "고객 연락처/선호 연락 시간 확인", "priority": "중간"},
            {"task": "해결 방안 마련 후 콜백", "priority": "높음"},
        ])
    if not tasks:
        tasks = [
            {"task": "상담 내용 CRM 기록", "priority": "중간"},
            {"task": "추가 확인 필요 사항 정리", "priority": "중간"},
        ]
    return _json({"case_summary": case_summary, "followup_tasks": tasks, "total": len(tasks)})


TOOLS = [
    claim_guide, claim_required_forms, contract_manage,
    customer_followup_tasks,
]
