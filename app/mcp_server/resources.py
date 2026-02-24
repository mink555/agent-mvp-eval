"""MCP Resources — 보험 데이터를 읽기 전용 리소스로 노출."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.tools.data import (
    PRODUCTS, RIDERS, COVERAGES, PREMIUM_TABLES, WAITING_PERIODS,
    EXCLUSIONS, REQUIRED_DISCLOSURES, KNOCKOUT_RULES,
    DIAGNOSIS_DEFINITIONS, ICD_MAPPINGS, HIGH_RISK_JOBS, FORBIDDEN_PHRASES,
    CLAIM_GUIDES, CLAIM_FORMS, CONTRACT_ACTIONS,
    SYSTEM_PROMPTS,
    _json,
)


def register_all_resources(mcp: FastMCP) -> int:
    """모든 리소스를 FastMCP에 등록. 등록된 수를 반환한다."""
    _count = 0

    def _counted_resource(*args, **kwargs):
        nonlocal _count
        _count += 1
        return mcp.resource(*args, **kwargs)

    # ═══ 상품 ═══

    @_counted_resource("insurance://products", name="product_catalog",
                   description="전체 보험 상품 목록", mime_type="application/json")
    def list_products() -> str:
        return _json([{"product_code": c, "name": p.get("name"), "category": p.get("category"),
                       "sales_status": p.get("sales_status"), "channels": p.get("channels")}
                      for c, p in PRODUCTS.items()])

    @_counted_resource("insurance://products/{product_code}", name="product_detail",
                   description="상품 코드별 상세 정보", mime_type="application/json")
    def get_product(product_code: str) -> str:
        product = PRODUCTS.get(product_code)
        if not product:
            return _json({"error": f"상품 '{product_code}'을(를) 찾을 수 없습니다."})
        return _json({"product_code": product_code, **product})

    @_counted_resource("insurance://products/{product_code}/riders", name="product_riders",
                   description="상품별 특약 목록", mime_type="application/json")
    def get_riders(product_code: str) -> str:
        riders = RIDERS.get(product_code)
        if not riders:
            return _json({"error": f"'{product_code}' 특약 정보 없음"})
        return _json({"product_code": product_code, "riders": riders})

    @_counted_resource("insurance://products/{product_code}/coverage", name="product_coverage",
                   description="상품별 보장 내용", mime_type="application/json")
    def get_coverage(product_code: str) -> str:
        cov = COVERAGES.get(product_code)
        if not cov:
            return _json({"error": f"'{product_code}' 보장 정보 없음"})
        return _json({"product_code": product_code, "coverage": cov})

    @_counted_resource("insurance://products/{product_code}/premium-table", name="product_premium_table",
                   description="상품별 보험료 산출 테이블", mime_type="application/json")
    def get_premium_table(product_code: str) -> str:
        table = PREMIUM_TABLES.get(product_code)
        if not table:
            return _json({"error": f"'{product_code}' 보험료표 없음"})
        return _json({"product_code": product_code, "premium_table": table})

    @_counted_resource("insurance://products/{product_code}/waiting-periods", name="product_waiting_periods",
                   description="상품별 면책/감액 기간", mime_type="application/json")
    def get_waiting_periods(product_code: str) -> str:
        wp = WAITING_PERIODS.get(product_code)
        if not wp:
            return _json({"error": f"'{product_code}' 면책/감액 기간 없음"})
        return _json({"product_code": product_code, "waiting_periods": wp})

    @_counted_resource("insurance://products/{product_code}/exclusions", name="product_exclusions",
                   description="상품별 보장 제외 사유", mime_type="application/json")
    def get_exclusions(product_code: str) -> str:
        return _json({"product_code": product_code, "common_exclusions": EXCLUSIONS.get("_common", []),
                       "product_exclusions": EXCLUSIONS.get(product_code, [])})

    @_counted_resource("insurance://products/{product_code}/disclosures", name="product_disclosures",
                   description="상품별 필수 설명사항", mime_type="application/json")
    def get_disclosures(product_code: str) -> str:
        return _json({"product_code": product_code, "common_disclosures": REQUIRED_DISCLOSURES.get("_common", []),
                       "product_disclosures": REQUIRED_DISCLOSURES.get(product_code, [])})

    # ═══ 레퍼런스 ═══

    @_counted_resource("insurance://reference/diagnosis-definitions", name="diagnosis_definitions",
                   description="의학/보험 용어 정의", mime_type="application/json")
    def ref_diagnosis() -> str:
        return _json(DIAGNOSIS_DEFINITIONS)

    @_counted_resource("insurance://reference/icd-mappings", name="icd_mappings",
                   description="ICD 코드 매핑", mime_type="application/json")
    def ref_icd() -> str:
        return _json(ICD_MAPPINGS)

    @_counted_resource("insurance://reference/high-risk-jobs", name="high_risk_jobs",
                   description="위험직종 분류", mime_type="application/json")
    def ref_jobs() -> str:
        return _json(HIGH_RISK_JOBS)

    @_counted_resource("insurance://reference/knockout-rules", name="knockout_rules",
                   description="인수 거절 조건", mime_type="application/json")
    def ref_knockout() -> str:
        return _json(KNOCKOUT_RULES)

    @_counted_resource("insurance://reference/forbidden-phrases", name="forbidden_phrases",
                   description="금칙어 목록", mime_type="application/json")
    def ref_forbidden() -> str:
        return _json(FORBIDDEN_PHRASES)

    # ═══ 청구/계약 ═══

    @_counted_resource("insurance://claims/guides", name="claim_guides",
                   description="청구 가이드", mime_type="application/json")
    def claims_guides() -> str:
        return _json(CLAIM_GUIDES)

    @_counted_resource("insurance://claims/forms", name="claim_forms",
                   description="청구 양식", mime_type="application/json")
    def claims_forms() -> str:
        return _json(CLAIM_FORMS)

    @_counted_resource("insurance://contract/actions", name="contract_actions",
                   description="계약 관리 가이드", mime_type="application/json")
    def contract_actions() -> str:
        return _json(CONTRACT_ACTIONS)

    # ═══ 설정 ═══

    @_counted_resource("insurance://config/system-prompts", name="system_prompts",
                   description="시스템 프롬프트", mime_type="application/json")
    def config_prompts() -> str:
        return _json(SYSTEM_PROMPTS)

    # ═══ 도구 카탈로그 ═══

    @_counted_resource("insurance://tools/catalog", name="tool_catalog",
                   description="등록된 전체 도구 카탈로그", mime_type="application/json")
    def tool_catalog() -> str:
        from app.tools import get_all_tools
        tools = get_all_tools()
        return _json([{"name": t.name, "description": t.description} for t in tools])

    return _count
