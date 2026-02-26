"""LangGraph 가드레일 — 입력/출력 안전성 검증 + 규제 문구 주입 노드.

확장 방법:
  새 체크를 추가하려면 check 함수를 작성하고
  INPUT_CHECKS 또는 OUTPUT_CHECKS 리스트에 추가하면 된다.
  그래프 구조를 변경할 필요 없음.

구조:
  input_guardrail  — 사용자 입력 검증 (프롬프트 인젝션, 도메인 외 질문)
  output_guardrail — LLM 출력 검증 (PII패턴·금칙어·빈 응답) + 면책 문구 주입
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

from langchain_core.messages import AIMessage, SystemMessage

from app.graph.state import AgentState, extract_last_human_query, extract_tools_used
from app.tools.data import FORBIDDEN_PHRASES, PII_PATTERNS as _RAW_PII

logger = logging.getLogger("insurance.graph.guardrails")


# ═══════════════════════════════════════════════════════════════════════════════
# GuardrailResult — 모든 체크 함수의 공통 반환 타입
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class GuardrailResult:
    """체크 함수 결과. passed=False 이면 reason에 사유를 담는다."""
    passed: bool
    reason: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Input Checks — 규칙 기반, 빠름 (<1 ms)
# ═══════════════════════════════════════════════════════════════════════════════

_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)(ignore|disregard|forget)\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)"),
    re.compile(r"(?i)you\s+are\s+now\s+(a|an)\s+"),
    re.compile(r"(?i)(system\s*prompt|시스템\s*프롬프트|시스템\s*메시지)"),
    re.compile(r"(?i)(jailbreak|탈옥|DAN\s*mode)"),
    re.compile(r"(?i)(pretend|act\s+as\s+if)\s+you"),
    re.compile(r"(?i)역할을?\s*(바꿔|변경|무시)"),
    re.compile(r"(?i)(이전|위의?|앞의?)\s*(지시|명령|규칙|프롬프트)를?\s*(무시|잊어|버려)"),
    # 시스템/설정 조작 시도
    re.compile(r"(?i)(설정|지시|명령|규칙)을?\s*(무시|잊어|버려|바꿔|변경)"),
    re.compile(r"(?i)(권한|관리자\s*권한|모든\s*권한)을?\s*(줘|넘겨|부여|획득)"),
    re.compile(r"(?i)(모든\s*)?(제약|제한|규칙|필터)을?\s*(해제|무시|없애|풀어)"),
]

# ── 임베딩 기반 도메인 관련성 판단 ──────────────────────────────────────────
# 키워드 집합 방식의 한계:
#   - 신규 상품/용어 추가 시 매번 수동 갱신 필요
#   - "이거 알려줘", "계속해줘" 같은 후속 질문 오탐
#   - 보험과 무관한 문장도 "보험" 단어만 있으면 통과
#
# 임베딩 방식의 장점:
#   - 의미 기반 판단 → "치매 케어 비용 걱정돼" 같은 우회 표현도 탐지
#   - 신규 상품 추가 시 코드 변경 불필요 (예시 튜플 확장만으로 충분)
#   - get_raw_embedding_model() 싱글톤을 재사용하여 모델 중복 로드 없음
#   - 지연: 첫 요청 ~20ms (예시 임베딩 계산), 이후 ~3ms (쿼리 임베딩 1개 + 코사인)
#
# e5 모델 프리픽스 전략 (intfloat/multilingual-e5-*):
#   - 도메인 예시 (passage) : "passage: {text}" 로 인코딩
#   - 입력 쿼리 (query)    : "query: {text}" 로 인코딩
#   - 비대칭 임베딩 덕분에 도메인 내/외 분리가 paraphrase-MiniLM 대비 크게 향상됨
#     (측정값: "주식 추천해줘" in=0.837 vs out=0.898, gap=0.061)
#
# 참고: NeMo Guardrails의 intent classification도 동일 원리
#   (all-MiniLM-L6-v2 + KNN) — Rebedea et al., 2023 (arXiv:2310.10501)

_IN_DOMAIN_EXAMPLES: tuple[str, ...] = (
    # 상품 조회
    "암보험 뭐가 있어?",
    "치아보험 있어?",
    "우리 회사 판매 상품 알려줘",
    "라이나생명 어떤 보험 팔아?",
    "치매보험 상품 있어?",
    "실버치아보험 알려줘",
    # 보험료
    "보험료 얼마야?",
    "45세 여성 치아보험 보험료 알려줘",
    "월 납입액이 얼마야?",
    "50세 남성 종신보험 보험료 계산해줘",
    # 보장/심사
    "고혈압 있어도 가입 가능해?",
    "암 진단 받으면 보험금 얼마 받아?",
    "면책기간 뭐야?",
    "인수심사 기준 알려줘",
    "특약 어떤 거 있어?",
    "해약환급금 어떻게 계산해?",
    # 청구/계약
    "청구 방법 알려줘",
    "보험 해지하면 어떻게 돼?",
    "갱신형이랑 비갱신형 차이가 뭐야?",
    "보험 약관 어디서 봐?",
    "계약 부활 신청 방법",
    "보험금 청구 서류 뭐 필요해?",
    # 신규 상품 관련
    "치매간병보험 가입 조건",
    "첫날부터 암보험 보장 범위",
    "골라담는 간편건강보험 심사 기준",
    "채우는335 해약환급금 구조",
)

_OUT_DOMAIN_EXAMPLES: tuple[str, ...] = (
    # 완전히 무관한 도메인
    "오늘 날씨 어때?",
    "주식 살 만한 종목 추천해줘",
    "맛있는 식당 어디야?",
    "비트코인 시세 알려줘",
    "내일 미세먼지 농도는?",
    "영어 번역해줘",
    "스마트폰 어떤 거 살까?",
    "영화 추천해줘",
    "운동 방법 알려줘",
    "아파트 매매 시세",
    "자동차 구매 비용",
    "부동산 투자 방법",
    "대학원 입학 조건",
    "비자 신청 방법",
    "음식 레시피 알려줘",
    "여행 코스 추천",
    "세금 신고 방법",
    "은행 예금 금리 비교",
    "코로나 증상 뭐야?",
    "코딩 강의 추천해줘",
)

_DOMAIN_IN_THRESHOLD = 0.87
"""in-domain 예시와의 최대 코사인 유사도가 이 값 이상이면 무조건 통과.

intfloat/multilingual-e5-large + query/passage 프리픽스 기준 측정값:
  - 보험 쿼리 (통과 대상): in 0.876~0.921 → 임계값 초과, 즉시 통과
  - 비보험 쿼리 (차단 대상): in 0.777~0.853 → 임계값 미달, 마진 조건으로 넘어감
  값 근거: 보험 쿼리 최솟값(0.876)보다 낮고 비보험 쿼리 최댓값(0.853)보다 높은 0.87
"""

_DOMAIN_MARGIN_THRESHOLD = 0.03
"""out-domain 유사도가 in-domain 유사도를 이 값 이상 앞서면 차단.

e5-large 측정값 (query/passage 프리픽스 적용):
  - "주식 추천해줘"  : out(0.898) - in(0.853) = +0.045 → 차단
  - "음식 추천해줘"  : out(0.880) - in(0.840) = +0.039 → 차단
  - "비트코인 시세"  : out(0.873) - in(0.777) = +0.095 → 차단
  - "오늘 날씨"      : out(0.895) - in(0.788) = +0.107 → 차단
  - "라이나생명 상품": out(0.815) - in(0.817) = -0.002 → 통과 (모호, LLM 처리)
"""


@lru_cache(maxsize=1)
def _get_domain_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """도메인 예시 임베딩을 한 번만 계산해서 L2 정규화 후 캐싱.

    서버 시작 후 첫 번째 도메인 체크 시 ~30ms 소요 (이후 0ms).
    get_raw_embedding_model() 싱글톤을 직접 사용하여 모델 중복 로드 없음.
    e5 모델이면 "passage: " 프리픽스 자동 부가.
    """
    from app.config import get_raw_embedding_model, get_settings, is_e5_model

    model = get_raw_embedding_model()
    is_e5 = is_e5_model(get_settings().embedding_model)
    prefix = "passage: " if is_e5 else ""

    in_raw = model.encode(
        [prefix + t for t in _IN_DOMAIN_EXAMPLES], normalize_embeddings=True
    )
    out_raw = model.encode(
        [prefix + t for t in _OUT_DOMAIN_EXAMPLES], normalize_embeddings=True
    )
    return np.array(in_raw, dtype=np.float32), np.array(out_raw, dtype=np.float32)


def check_prompt_injection(text: str) -> GuardrailResult:
    """프롬프트 인젝션 패턴 탐지."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return GuardrailResult(
                passed=False,
                reason="죄송합니다. 해당 요청은 처리할 수 없습니다.",
            )
    return GuardrailResult(passed=True)


def check_domain_relevance(text: str) -> GuardrailResult:
    """임베딩 기반 보험 도메인 관련성 판별 (Layer 2: ~3~30ms).

    판정 기준 (e5-large query/passage 프리픽스 기준):
      1. 길이 < 5자  → 무조건 통과 (후속 단답 "네", "아니" 등)
      2. max_in >= _DOMAIN_IN_THRESHOLD(0.87) → 통과 (명확한 in-domain)
      3. max_out - max_in >= _DOMAIN_MARGIN_THRESHOLD(0.03) → 차단 (out이 확실히 우세)
      4. 그 외 모호한 경우 → 통과 (LLM 시스템 프롬프트가 도메인 외 응답 처리)

    임베딩 실패 시 안전하게 통과 처리 (서비스 중단 방지 우선).
    """
    if len(text.strip()) < 5:
        return GuardrailResult(passed=True)
    try:
        from app.config import get_raw_embedding_model, get_settings, is_e5_model

        model = get_raw_embedding_model()
        is_e5 = is_e5_model(get_settings().embedding_model)
        q_prefix = "query: " if is_e5 else ""

        q_vec = model.encode([q_prefix + text], normalize_embeddings=True)[0]
        q_vec = np.array(q_vec, dtype=np.float32)

        in_embs, out_embs = _get_domain_embeddings()
        max_in = float(np.max(in_embs @ q_vec))
        max_out = float(np.max(out_embs @ q_vec))

        logger.debug(
            "Domain check | in=%.3f out=%.3f gap=%.3f | %s",
            max_in, max_out, max_out - max_in, text[:40],
        )

        if max_in >= _DOMAIN_IN_THRESHOLD:
            return GuardrailResult(passed=True)

        if max_out - max_in >= _DOMAIN_MARGIN_THRESHOLD:
            return GuardrailResult(
                passed=False,
                reason="보험 관련 질문에만 답변할 수 있습니다. "
                       "보험 상품, 가입, 보장, 청구 등에 대해 질문해 주세요.",
            )

        return GuardrailResult(passed=True)
    except Exception:
        logger.warning("Domain embedding check failed, defaulting to pass")
        return GuardrailResult(passed=True)


INPUT_CHECKS = [check_prompt_injection, check_domain_relevance]
"""입력 체크 리스트. 새 체크 추가 시 여기에 함수를 append."""


# ═══════════════════════════════════════════════════════════════════════════════
# Output Checks — 규칙 기반, 빠름 (<1 ms)
# ═══════════════════════════════════════════════════════════════════════════════

_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(pat), label) for pat, label in _RAW_PII
]

_FORBIDDEN_OUTPUT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(fp["pattern"].replace(" ", r"\s*")), fp["reason"])
    for fp in FORBIDDEN_PHRASES
]

_SAFE_RESPONSE = (
    "죄송합니다. 응답을 생성하는 과정에서 문제가 발견되었습니다. "
    "다시 질문해 주시면 정확한 정보로 답변드리겠습니다."
)


def check_pii_leak(text: str) -> GuardrailResult:
    """응답에 개인정보가 노출되었는지 검사."""
    for pattern, pii_type in _PII_PATTERNS:
        if pattern.search(text):
            return GuardrailResult(passed=False, reason=f"응답에 {pii_type} 포함")
    return GuardrailResult(passed=True)


def check_forbidden_output(text: str) -> GuardrailResult:
    """금칙어/과장표현이 응답에 포함되었는지 검사."""
    for pattern, reason in _FORBIDDEN_OUTPUT_PATTERNS:
        match = pattern.search(text)
        if match:
            return GuardrailResult(
                passed=False,
                reason=f"부적절한 표현 감지: '{match.group()}' → {reason}",
            )
    return GuardrailResult(passed=True)


def check_empty_response(text: str) -> GuardrailResult:
    """빈 응답 또는 의미 없는 응답 검사."""
    if not text or not text.strip():
        return GuardrailResult(passed=False, reason="빈 응답")
    return GuardrailResult(passed=True)


_TOOL_NAME_PATTERN = re.compile(
    r'\b('
    r'product_search|product_get|product_compare|product_latest_version_check|'
    r'rider_list|rider_search|rider_get|eligibility_by_product_rule|'
    r'product_faq_lookup|sales_channel_availability|'
    r'premium_estimate|premium_compare|plan_options|amount_suggest|'
    r'renewal_premium_projection|affordability_check|payment_cycle_options|'
    r'surrender_value_explain|'
    r'coverage_summary|coverage_detail|benefit_amount_lookup|benefit_limit_rules|'
    r'event_eligibility_check|diagnosis_definition_lookup|icd_mapping_lookup|'
    r'multi_benefit_conflict_rule|rider_bundle_recommend|'
    r'underwriting_precheck|underwriting_questions_generator|'
    r'underwriting_knockout_rules|underwriting_docs_required|'
    r'underwriting_waiting_periods|underwriting_exclusions|'
    r'underwriting_limitations|underwriting_reinstatement_rule|'
    r'underwriting_renewal_eligibility|underwriting_renewal_premium_notice|'
    r'underwriting_high_risk_job_check|underwriting_disclosure_risk_score|'
    r'compliance_required_disclosure|compliance_phrase_generator|'
    r'compliance_misleading_check|comparison_disclaimer_generator|'
    r'recording_notice_script|privacy_masking|'
    r'claim_guide|claim_required_forms|contract_manage|customer_followup_tasks|'
    r'customer_contract_lookup|duplicate_enrollment_check|customer_search|'
    r'rag_terms_query_engine|rag_product_info_query_engine'
    r')\b'
)


_PRODUCT_CODE_PATTERN = re.compile(r'\(?\s*B\d{5,}\s*\)?')

def sanitize_tool_names(text: str) -> str:
    """응답에서 내부 도구명(snake_case 함수명)과 상품코드(B00...)를 제거한다."""
    text = _TOOL_NAME_PATTERN.sub('', text)
    text = _PRODUCT_CODE_PATTERN.sub('', text)
    return re.sub(r'  +', ' ', text).strip()


OUTPUT_CHECKS = [check_pii_leak, check_forbidden_output, check_empty_response]
"""출력 체크 리스트. 새 체크 추가 시 여기에 함수를 append."""


# ═══════════════════════════════════════════════════════════════════════════════
# 하드코딩 면책 문구 — 코드가 SSOT (LLM이 변조·누락 불가)
# 도구 패턴별로 적절한 면책 문구를 매핑
# ═══════════════════════════════════════════════════════════════════════════════

_DISCLAIMERS: list[tuple[set[str], str]] = [
    (
        {"premium_estimate", "premium_compare", "plan_options",
         "renewal_premium_projection", "affordability_check"},
        "이 금액은 예시이며, 실제 보험료는 상품·보장내용·건강상태에 따라 달라집니다. "
        "정확한 보험료는 설계사 상담 또는 공식 홈페이지를 통해 확인해 주세요.",
    ),
    (
        {"product_compare", "product_search", "product_get"},
        "상품 상세 내용은 약관을 기준으로 하며, "
        "가입 전 반드시 상품설명서와 약관을 확인하시기 바랍니다.",
    ),
    (
        {"coverage_summary", "coverage_detail", "benefit_amount_lookup",
         "benefit_limit_rules", "event_eligibility_check"},
        "보장 내용은 약관을 기준으로 하며, 여기 표시된 내용은 참고용입니다. "
        "실제 보장 범위와 지급 조건은 약관에서 정한 바에 따릅니다.",
    ),
]


def _select_disclaimer(tools_used: list[str]) -> str | None:
    """사용된 도구 목록에서 적절한 면책 문구를 선택."""
    used = set(tools_used)
    for trigger_tools, disclaimer in _DISCLAIMERS:
        if trigger_tools & used:
            return disclaimer
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Graph Nodes
# ═══════════════════════════════════════════════════════════════════════════════

def _has_prior_conversation(state: AgentState) -> bool:
    """출력 가드레일을 통과한 실제 AI 응답이 존재하는지 확인.

    guardrail 거절 메시지(AIMessage)는 제외한다.
    conversation_started=True는 output_guardrail pass 시에만 설정되므로,
    1턴 차단 → 2턴 followup 판정 우회 취약점을 방지한다.
    """
    return bool(state.get("conversation_started", False))


def input_guardrail(state: AgentState) -> dict:
    """입력 가드레일 노드 — 사용자 메시지를 검증한다.

    후속 질문 처리 원칙 (이전 AI 답변이 있는 경우):
      - 도메인 체크(check_domain_relevance) 건너뜀
        → 대화 맥락 안에서 이해되어야 하는 질문이므로 query_rewriter가 처리
      - 프롬프트 인젝션 체크(check_prompt_injection)는 항상 적용

    차단 시 거절 AIMessage를 messages에 추가하고
    guardrail_action='block'으로 설정하여 agent를 건너뛴다.
    """
    ts = time.time()
    query = extract_last_human_query(state["messages"])
    is_followup = _has_prior_conversation(state)

    for check_fn in INPUT_CHECKS:
        if is_followup and check_fn is check_domain_relevance:
            continue  # 후속 질문은 도메인 체크 생략 — query_rewriter가 맥락 처리
        result = check_fn(query)
        if not result.passed:
            logger.warning("Input blocked: %s (query=%.80s)", result.reason, query)
            return {
                "messages": [AIMessage(content=result.reason)],
                "guardrail_action": "block",
                "trace": [{
                    "node": "input_guardrail", "action": "block",
                    "reason": result.reason,
                    "is_followup": is_followup,
                    "duration_ms": round((time.time() - ts) * 1000),
                }],
            }

    return {
        "guardrail_action": "pass",
        "trace": [{
            "node": "input_guardrail", "action": "pass",
            "is_followup": is_followup,
            "duration_ms": round((time.time() - ts) * 1000),
        }],
    }


_MAX_OUTPUT_RETRIES = 1  # 재시도 최대 횟수 (무한 루프 방지)


def output_guardrail(state: AgentState) -> dict:
    """출력 가드레일 노드 — LLM 최종 응답을 검증하고 면책 문구를 주입한다.

    1) 안전성 체크 (PII·금칙어·빈 응답)
       - 첫 번째 차단: 차단 이유를 SystemMessage로 주입 후 agent에 재시도 요청
       - 두 번째 이상 차단: 안전 응답으로 교체 (무한 루프 방지)
    2) 통과 시: 도구 패턴에 맞는 면책 문구를 응답 끝에 주입
    """
    ts = time.time()
    last_msg = state["messages"][-1] if state["messages"] else None
    text = getattr(last_msg, "content", "") or ""
    retry_count = state.get("guardrail_retry_count", 0)

    for check_fn in OUTPUT_CHECKS:
        result = check_fn(text)
        if not result.passed:
            logger.warning(
                "Output blocked (retry=%d): %s | text_preview=%.120s",
                retry_count, result.reason, text,
            )
            if retry_count < _MAX_OUTPUT_RETRIES:
                # 재시도: LLM에게 차단 이유를 알려주고 재생성 요청
                hint = SystemMessage(
                    content=(
                        f"[출력 검증 실패] 직전 응답이 다음 이유로 차단되었습니다: {result.reason}\n"
                        "위반 표현을 사용하지 않고 같은 내용을 다시 답변해 주세요."
                    )
                )
                return {
                    "messages": [hint],
                    "guardrail_action": "retry",
                    "guardrail_retry_count": retry_count + 1,
                    "trace": [{
                        "node": "output_guardrail", "action": "retry",
                        "reason": result.reason,
                        "retry_count": retry_count + 1,
                        "duration_ms": round((time.time() - ts) * 1000),
                    }],
                }
            else:
                # 재시도 초과: 안전 응답으로 대체
                logger.error(
                    "Output blocked after max retries: %s | text=%.200s",
                    result.reason, text,
                )
                return {
                    "messages": [AIMessage(
                        content=_SAFE_RESPONSE,
                        id=getattr(last_msg, "id", None),
                    )],
                    "guardrail_action": "block",
                    "trace": [{
                        "node": "output_guardrail", "action": "block",
                        "reason": result.reason,
                        "retry_count": retry_count,
                        "duration_ms": round((time.time() - ts) * 1000),
                    }],
                }

    cleaned = sanitize_tool_names(text)
    tools_used = extract_tools_used(state["messages"])
    disclaimer = _select_disclaimer(tools_used)

    existing_disclaimers = cleaned.count('\n※ ')
    text_changed = cleaned != text

    if disclaimer and disclaimer not in cleaned and existing_disclaimers < 1:
        amended = f"{cleaned.rstrip()}\n\n※ {disclaimer}"
        return {
            "messages": [AIMessage(content=amended, id=last_msg.id)],
            "guardrail_action": "pass",
            "conversation_started": True,
            "trace": [{
                "node": "output_guardrail", "action": "pass",
                "disclaimer_appended": True,
                "tool_names_removed": text_changed,
                "duration_ms": round((time.time() - ts) * 1000),
            }],
        }

    if text_changed:
        return {
            "messages": [AIMessage(content=cleaned, id=last_msg.id)],
            "guardrail_action": "pass",
            "conversation_started": True,
            "trace": [{
                "node": "output_guardrail", "action": "pass",
                "tool_names_removed": True,
                "duration_ms": round((time.time() - ts) * 1000),
            }],
        }

    return {
        "guardrail_action": "pass",
        "conversation_started": True,
        "trace": [{
            "node": "output_guardrail", "action": "pass",
            "duration_ms": round((time.time() - ts) * 1000),
        }],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Routing — 조건부 엣지용
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_input_guard(state: AgentState) -> Literal["pass", "block"]:
    """입력 가드레일 후 분기: 차단 → END, 통과 → query_rewriter."""
    if state.get("guardrail_action") == "block":
        return "block"
    return "pass"


def route_after_output_guard(state: AgentState) -> Literal["agent", "__end__"]:
    """출력 가드레일 후 분기: 재시도 → agent, 그 외 → END."""
    if state.get("guardrail_action") == "retry":
        return "agent"
    return "__end__"
