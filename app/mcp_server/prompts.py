"""MCP Prompts — 보험 도메인 재사용 프롬프트 템플릿.

카테고리별로 정리된 프롬프트를 @mcp.prompt 데코레이터로 등록한다.
새 프롬프트 추가 시 해당 카테고리 섹션에 함수 하나만 추가하면 된다.

프롬프트 목록:
  [상품]   analyze_product        상품 종합 분석
  [상품]   compare_products       2개 상품 비교 분석
  [상담]   consultation_guide     계약자 맞춤 상담 가이드
  [상담]   needs_analysis         계약자 니즈 파악 질문 생성
  [심사]   underwriting_review    가입 심사 종합 검토
  [준법]   compliance_review      스크립트/멘트 준법 검토
  [준법]   sales_script           판매 스크립트 생성
  [청구]   claim_assistance       청구 절차 안내
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts.base import Message, UserMessage, AssistantMessage
from mcp.types import TextContent


def _user(text: str) -> UserMessage:
    return UserMessage(content=TextContent(type="text", text=text))


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(content=TextContent(type="text", text=text))


def register_all_prompts(mcp: FastMCP) -> int:
    """모든 프롬프트를 FastMCP에 등록. 등록된 수를 반환한다."""
    _count = 0

    def _counted_prompt(*args, **kwargs):
        nonlocal _count
        _count += 1
        return mcp.prompt(*args, **kwargs)

    # ═══════════════════ A. 상품 분석 ═══════════════════

    @_counted_prompt(
        name="analyze_product",
        description="상품 종합 분석 — 보장, 면책/감액, 보험료, 가입조건을 포함한 상세 분석",
    )
    def analyze_product(product_code: str) -> list[Message]:
        return [
            _user(
                f"보험 상품 '{product_code}'를 종합 분석해주세요.\n\n"
                "다음 항목을 순서대로 포함해주세요:\n"
                "1. 상품 개요 (카테고리, 판매상태, 가입연령, 채널)\n"
                "2. 보장 내용 (주계약 + 주요 특약)\n"
                "3. 면책/감액 기간\n"
                "4. 보험료 수준 (30세/40세/50세 남녀 기준)\n"
                "5. 경쟁 우위 및 주의사항\n\n"
                "활용 도구: product_get, coverage_summary, "
                "underwriting_waiting_periods, premium_estimate\n"
                f"활용 리소스: insurance://products/{product_code}, "
                f"insurance://products/{product_code}/coverage"
            ),
            _assistant(
                f"상품 '{product_code}' 종합 분석을 시작하겠습니다. "
                "먼저 상품 기본 정보를 조회합니다."
            ),
        ]

    @_counted_prompt(
        name="compare_products",
        description="2개 보험 상품 비교 분석 — 보장, 보험료, 가입조건 비교표 생성",
    )
    def compare_products(product_code_a: str, product_code_b: str) -> list[Message]:
        return [
            _user(
                f"'{product_code_a}'와 '{product_code_b}' 두 상품을 비교 분석해주세요.\n\n"
                "다음 항목을 비교표 형태로 정리해주세요:\n"
                "1. 상품 유형 및 카테고리\n"
                "2. 주요 보장 항목 (금액 포함)\n"
                "3. 면책/감액 기간 비교\n"
                "4. 보험료 비교 (동일 프로필 기준: 40세 남성)\n"
                "5. 가입 조건 차이\n"
                "6. 각 상품의 장단점 및 추천 대상\n\n"
                "활용 도구: product_compare, coverage_summary, "
                "premium_compare, underwriting_waiting_periods"
            ),
            _assistant(
                "두 상품의 비교 분석을 시작하겠습니다. "
                "먼저 각 상품의 기본 정보를 조회합니다."
            ),
        ]

    # ═══════════════════ B. 계약자 상담 ═══════════════════

    @_counted_prompt(
        name="consultation_guide",
        description="계약자 프로필 기반 맞춤 상담 가이드 — 추천 상품, 보험료, 고지사항 포함",
    )
    def consultation_guide(
        age: str, gender: str, interest: str = "종합보장"
    ) -> list[Message]:
        return [
            _user(
                f"다음 계약자에게 맞는 보험 상담 가이드를 작성해주세요.\n\n"
                f"계약자 정보:\n"
                f"- 나이: {age}세\n"
                f"- 성별: {gender}\n"
                f"- 관심 분야: {interest}\n\n"
                "포함 항목:\n"
                "1. 추천 상품 (2~3개, 이유 포함)\n"
                "2. 각 상품별 예상 보험료\n"
                "3. 가입 가능 여부 사전 점검\n"
                "4. 필수 고지사항 안내 포인트\n"
                "5. 상담 시 주의할 컴플라이언스 사항\n\n"
                "활용 도구: product_search, premium_estimate, "
                "underwriting_precheck, compliance_required_disclosure"
            ),
            _assistant(
                f"{age}세 {gender} 계약자의 '{interest}' 관심사에 맞는 "
                "상담 가이드를 준비하겠습니다."
            ),
        ]

    @_counted_prompt(
        name="needs_analysis",
        description="계약자 니즈 파악을 위한 질문 리스트 생성 — 초회 상담용",
    )
    def needs_analysis(customer_type: str = "신규") -> list[Message]:
        return [
            _user(
                f"'{customer_type}' 계약자를 위한 니즈 파악 질문 리스트를 생성해주세요.\n\n"
                "다음 영역별로 질문을 3~5개씩 만들어주세요:\n"
                "1. 기본 정보 (가족구성, 직업, 소득)\n"
                "2. 보장 니즈 (건강, 사망, 치아, 상해)\n"
                "3. 예산 및 납입 선호 (월 예산, 납입기간)\n"
                "4. 기존 보험 현황 (중복 확인)\n"
                "5. 우려사항 (건강이력, 직업 위험도)\n\n"
                "각 질문에 대해 답변에 따른 다음 액션도 함께 제시해주세요.\n\n"
                "활용 도구: underwriting_high_risk_job_check, "
                "underwriting_precheck"
            ),
        ]

    # ═══════════════════ C. 언더라이팅 ═══════════════════

    @_counted_prompt(
        name="underwriting_review",
        description="가입 심사 종합 검토 — 계약자 프로필 기반 가입 가능성 분석",
    )
    def underwriting_review(
        product_code: str, age: str, gender: str,
        health_history: str = "특이사항 없음",
    ) -> list[Message]:
        return [
            _user(
                f"다음 계약자의 '{product_code}' 가입 심사를 종합 검토해주세요.\n\n"
                f"계약자 프로필:\n"
                f"- 나이: {age}세 / 성별: {gender}\n"
                f"- 건강이력: {health_history}\n\n"
                "검토 항목:\n"
                "1. 가입 가능 여부 (녹아웃 룰 확인)\n"
                "2. 필수 고지 질문 목록\n"
                "3. 면책/감액 기간 안내\n"
                "4. 추가 필요 서류 확인\n"
                "5. 고지 누락 리스크 평가\n"
                "6. 종합 의견 및 권고사항\n\n"
                "활용 도구: underwriting_precheck, underwriting_knockout_rules, "
                "underwriting_questions_generator, underwriting_waiting_periods, "
                "underwriting_docs_required, underwriting_disclosure_risk_score"
            ),
            _assistant(
                f"'{product_code}' 상품에 대한 {age}세 {gender} 계약자의 "
                "가입 심사를 검토하겠습니다."
            ),
        ]

    # ═══════════════════ D. 컴플라이언스 ═══════════════════

    @_counted_prompt(
        name="compliance_review",
        description="스크립트/멘트 준법 검토 — 금칙어, 과장표현, 필수고지 누락 확인",
    )
    def compliance_review(script_text: str) -> list[Message]:
        return [
            _user(
                "다음 판매 스크립트를 준법 관점에서 검토해주세요.\n\n"
                f"검토 대상 텍스트:\n\"\"\"\n{script_text}\n\"\"\"\n\n"
                "검토 항목:\n"
                "1. 금칙어/과장표현 감지\n"
                "2. 필수 설명사항 누락 여부\n"
                "3. 오도 가능성 있는 표현\n"
                "4. 개인정보 노출 여부\n"
                "5. 수정 제안 (문제 있는 부분별)\n\n"
                "활용 도구: compliance_misleading_check, "
                "compliance_required_disclosure, privacy_masking\n"
                "활용 리소스: insurance://reference/forbidden-phrases"
            ),
            _assistant(
                "제출하신 스크립트의 준법 검토를 시작하겠습니다."
            ),
        ]

    @_counted_prompt(
        name="sales_script",
        description="판매 스크립트 생성 — 상품별 준법 멘트 포함 완성 스크립트",
    )
    def sales_script(
        product_code: str, channel: str = "TM",
    ) -> list[Message]:
        return [
            _user(
                f"'{product_code}' 상품의 {channel} 채널 판매 스크립트를 생성해주세요.\n\n"
                "스크립트 구성:\n"
                "1. 오프닝 (녹취 고지 포함)\n"
                "2. 상품 소개 (핵심 보장 요약)\n"
                "3. 보험료 안내\n"
                "4. 면책/감액 기간 설명 (필수)\n"
                "5. 갱신 관련 안내 (해당 시)\n"
                "6. 고지의무 안내\n"
                "7. 클로징 (청약 절차 안내)\n\n"
                "모든 멘트는 금칙어를 피하고, 필수 설명사항을 포함해야 합니다.\n\n"
                "활용 도구: compliance_phrase_generator, "
                "recording_notice_script, compliance_required_disclosure, "
                "coverage_summary, premium_estimate"
            ),
            _assistant(
                f"'{product_code}' 상품의 {channel} 채널 판매 스크립트를 "
                "생성하겠습니다. 먼저 필수 설명사항과 상품 정보를 확인합니다."
            ),
        ]

    # ═══════════════════ E. 청구/사후 ═══════════════════

    @_counted_prompt(
        name="claim_assistance",
        description="보험금 청구 절차 안내 — 유형별 단계, 필요 서류, 주의사항 포함",
    )
    def claim_assistance(
        claim_type: str, product_code: str = "",
    ) -> list[Message]:
        product_note = f" (상품: {product_code})" if product_code else ""
        return [
            _user(
                f"'{claim_type}' 유형의 보험금 청구 절차를 안내해주세요{product_note}.\n\n"
                "포함 항목:\n"
                "1. 청구 절차 (단계별)\n"
                "2. 필요 서류 목록\n"
                "3. 청구서 양식 필수 항목\n"
                "4. 처리 예상 기간\n"
                "5. 자주 실수하는 부분 및 주의사항\n\n"
                "활용 도구: claim_guide, claim_required_forms\n"
                "활용 리소스: insurance://claims/guides, insurance://claims/forms"
            ),
            _assistant(
                f"'{claim_type}' 청구 절차를 안내하겠습니다."
            ),
        ]

    return _count
