"""준법/컴플라이언스 도구."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from app.tools.data import (
    PRODUCTS, REQUIRED_DISCLOSURES, FORBIDDEN_PHRASES,
    COMPLIANCE_TEMPLATES, WAITING_PERIODS, PII_PATTERNS,
    _json,
)


# ── Input Schemas ─────────────────────────────────────────────────────────────


class ProductCodeInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00115023)")


class PhraseGeneratorInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00172014)")
    topic: str = Field(
        ..., min_length=1,
        description="준법 멘트 주제 (면책, 감액, 갱신, 해약, 간편심사)",
    )
    tone: str = Field(default="공식", description="멘트 톤 (공식, 친근)")


class TextInput(BaseModel):
    text: str = Field(..., min_length=1, description="검사 또는 마스킹할 텍스트")


class RecordingNoticeInput(BaseModel):
    channel: str = Field(default="", description="판매 채널 (TM, CM, 빈 값이면 범용)")


# ── Tools ─────────────────────────────────────────────────────────────────────


@tool(args_schema=ProductCodeInput)
def compliance_required_disclosure(product_code: str) -> str:
    """상품별 법적 필수 설명사항을 조회합니다."""
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    common = REQUIRED_DISCLOSURES.get("_common", [])
    specific = REQUIRED_DISCLOSURES.get(product_code, [])
    return _json({
        "product_code": product_code, "name": p["name"],
        "common_disclosures": common, "product_disclosures": specific,
        "total": len(common) + len(specific),
    })


@tool(args_schema=PhraseGeneratorInput)
def compliance_phrase_generator(product_code: str, topic: str, tone: str = "공식") -> str:
    """상품/주제별 준법 멘트 스크립트를 생성합니다. 예: 면책, 감액, 갱신, 해약."""
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    periods = WAITING_PERIODS.get(product_code, {})
    template_key = None
    params = {"product_name": p["name"]}
    topic_lower = topic.lower()
    if "면책" in topic_lower:
        template_key = "면책_안내"
        params["exemption_period"] = periods.get("면책기간", "약관에 정한")
    elif "감액" in topic_lower:
        template_key = "감액_안내"
        params["reduction_period"] = periods.get("감액기간", "약관에 정한 기간").split(",")[0]
        params["reduction_pct"] = "50%"
    elif "갱신" in topic_lower:
        template_key = "갱신_안내"
        params["renewal_type"] = p.get("renewal_type", "갱신형")
        params["max_renewal_age"] = str(p.get("max_renewal_age", "약관 참조"))
    elif "해약" in topic_lower or "환급" in topic_lower:
        template_key = "해약환급금_안내"
    elif "간편" in topic_lower:
        template_key = "간편심사_안내"
    if template_key and template_key in COMPLIANCE_TEMPLATES:
        try:
            script = COMPLIANCE_TEMPLATES[template_key].format(**params)
        except KeyError:
            script = COMPLIANCE_TEMPLATES[template_key]
    else:
        script = f"{p['name']}의 '{topic}'에 대한 안내 멘트를 준비 중입니다."
    return _json({"product_code": product_code, "topic": topic, "tone": tone, "script": script})


@tool(args_schema=TextInput)
def compliance_misleading_check(text: str) -> str:
    """판매 스크립트/멘트에서 금칙어·과장표현을 검사합니다."""
    issues = []
    tl = text.lower()
    for fp in FORBIDDEN_PHRASES:
        if fp["pattern"].lower() in tl:
            issues.append({"found": fp["pattern"], "reason": fp["reason"], "suggested_fix": fp["fix"]})
    return _json({"text": text, "is_ok": len(issues) == 0, "issues": issues, "total_issues": len(issues)})


@tool(args_schema=ProductCodeInput)
def comparison_disclaimer_generator(product_code: str) -> str:
    """상품 비교 시 필수 면책 문구를 생성합니다."""
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    disclaimers = []
    if p.get("simplified_underwriting"):
        disclaimers.append(f"본 상품({p['name']})은 간편(유병력자) 심사 상품으로, 일반심사 상품 대비 보험료가 높을 수 있습니다.")
        disclaimers.append("일반심사 상품에 가입 가능하신 경우, 일반심사 상품도 함께 비교해 보시기 바랍니다.")
    else:
        disclaimers.append("본 상품은 일반심사 상품입니다.")
    return _json({"product_code": product_code, "name": p["name"], "disclaimers": disclaimers})


@tool(args_schema=RecordingNoticeInput)
def recording_notice_script(channel: str = "") -> str:
    """녹취 고지 스크립트를 채널(TM/CM)별로 생성합니다."""
    ch = channel.upper()
    if ch == "TM":
        script = ("고객님, 안녕하세요. 라이나생명입니다. "
                  "본 통화는 보험업법 제95조의3에 따라 녹취되며, "
                  "통화 내용은 청약 내용의 정확성 확인 및 분쟁 해결을 위해 활용됩니다. "
                  "녹취에 동의하시면 상담을 진행하겠습니다.")
    elif ch == "CM":
        script = ("고객님, 안녕하세요. "
                  "본 상담은 보험업법에 따라 녹취되고 있음을 안내드립니다. "
                  "상담 내용은 계약 체결의 정확성 확인 목적으로만 사용됩니다.")
    else:
        script = COMPLIANCE_TEMPLATES.get("녹취_고지", "본 통화는 보험업법에 따라 녹취되며, 청약 내용의 정확성 확인을 위해 활용됩니다.")
    return _json({"channel": channel, "script": script})


@tool(args_schema=TextInput)
def privacy_masking(text: str) -> str:
    """텍스트에서 주민번호·전화번호·카드번호·이메일 등 개인정보를 마스킹합니다."""
    masked = text
    applied = []
    for pat, label in PII_PATTERNS:
        repl = f"[{label}]"
        new_text = re.sub(pat, repl, masked)
        if new_text != masked:
            applied.append(repl)
            masked = new_text
    return _json({"original_length": len(text), "masked_text": masked, "applied_masks": applied})


TOOLS = [
    compliance_required_disclosure, compliance_phrase_generator,
    compliance_misleading_check, comparison_disclaimer_generator,
    recording_notice_script, privacy_masking,
]
