"""상품/특약 탐색 도구."""

from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from app.tools.data import PRODUCTS, RIDERS, WAITING_PERIODS, _json


# ── Input Schemas ─────────────────────────────────────────────────────────────


class ProductSearchInput(BaseModel):
    keyword: str = Field(default="", description="상품명·키워드 검색어 (예: 암, 치아, 종신)")
    category: str = Field(default="", description="카테고리 필터 (예: 암/건강, 사망/종신)")


class ProductCodeInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00115023)")


class ProductCodesInput(BaseModel):
    codes: str = Field(
        ..., description="비교할 상품 코드 목록 (쉼표 또는 공백 구분, 예: B00115023, B00197011)"
    )


class RiderSearchInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00115023)")
    keyword: str = Field(..., description="특약 검색 키워드 (예: 암, 입원)")


class RiderCodeInput(BaseModel):
    rider_code: str = Field(..., description="특약 코드 또는 이름 (예: R-115-01, 암진단특약)")


class ProductFaqInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00115023)")
    topic: str = Field(default="", description="FAQ 주제 (갱신, 면책, 감액, 보장개시)")


class SalesChannelInput(BaseModel):
    product_code: str = Field(..., description="상품 코드 (예: B00115023)")
    channel: str = Field(default="", description="확인할 채널 (TM, CM, 온라인)")


# ── Tools ─────────────────────────────────────────────────────────────────────


@tool(args_schema=ProductSearchInput)
def product_search(keyword: str = "", category: str = "") -> str:
    """판매 중인 보험 상품을 키워드 또는 카테고리로 검색합니다. 우리 회사(라이나생명) 전체 상품 목록 조회, 당사 상품 확인, 어떤 상품 있는지 물어볼 때 사용합니다."""
    results = []
    kw = keyword.lower()
    for p in PRODUCTS.values():
        searchable = (
            p["name"] + p["category"] + " ".join(p["highlights"]) + p.get("insurer", "")
        ).lower()
        if kw and kw in searchable:
            results.append(p)
        elif category and category in p["category"]:
            results.append(p)
    if not results:
        results = list(PRODUCTS.values())
    return _json({"products": results, "total": len(results), "query": {"keyword": keyword, "category": category}})


@tool(args_schema=ProductCodeInput)
def product_get(product_code: str) -> str:
    """상품 코드(B로 시작)로 상품 상세 정보를 조회합니다."""
    p = PRODUCTS.get(product_code)
    if not p:
        for v in PRODUCTS.values():
            if product_code.lower() in v["name"].lower():
                p = v
                break
    if not p:
        return _json({"error": f"상품 코드 '{product_code}'를 찾을 수 없습니다.", "available": list(PRODUCTS.keys())})
    return _json({"product": p})


@tool(args_schema=ProductCodesInput)
def product_compare(codes: str) -> str:
    """2개 이상의 상품을 비교합니다. 쉼표 또는 공백으로 코드를 구분합니다."""
    code_list = [c.strip() for c in codes.replace(",", " ").split() if c.strip()]
    items = [PRODUCTS[c] for c in code_list if c in PRODUCTS]
    if len(items) < 2:
        return _json({"error": "비교를 위해 2개 이상의 유효한 상품 코드가 필요합니다.", "available": list(PRODUCTS.keys())})
    comparison = []
    for item in items:
        comparison.append({
            "code": item["code"], "name": item["name"], "category": item["category"],
            "renewal_type": item.get("renewal_type"), "term_years": item.get("term_years"),
            "age_range": f"{item.get('min_age', '?')}~{item.get('max_age', '?')}세",
            "plan_types": item.get("plan_types", []),
        })
    return _json({"comparison": comparison})


@tool(args_schema=ProductCodeInput)
def product_latest_version_check(product_code: str) -> str:
    """상품의 최신 요약서 버전인지 확인합니다."""
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    return _json({
        "product_code": product_code, "name": p["name"],
        "current_doc_date": p.get("doc_date"), "current_revision": p.get("revision"),
        "is_latest": True, "note": "현재 판매 중인 최신 요약서입니다.",
    })


@tool(args_schema=ProductCodeInput)
def rider_list(product_code: str) -> str:
    """상품의 특약(부가계약) 목록을 조회합니다. 의무부가/선택 구분 포함."""
    riders = RIDERS.get(product_code, [])
    if not riders:
        return _json({"error": f"상품 '{product_code}'의 특약 정보가 없습니다."})
    mandatory = [r for r in riders if r["type"] == "의무부가"]
    optional = [r for r in riders if r["type"] == "선택"]
    return _json({"product_code": product_code, "mandatory_riders": mandatory, "optional_riders": optional, "total": len(riders)})


@tool(args_schema=RiderSearchInput)
def rider_search(product_code: str, keyword: str) -> str:
    """특정 상품의 특약을 키워드로 검색합니다."""
    riders = RIDERS.get(product_code, [])
    kw = keyword.lower()
    matched = [r for r in riders if kw in (r["name"] + r["desc"]).lower()]
    return _json({"product_code": product_code, "keyword": keyword, "results": matched, "total": len(matched)})


@tool(args_schema=RiderCodeInput)
def rider_get(rider_code: str) -> str:
    """특약 코드 또는 이름으로 특약 상세 정보를 조회합니다."""
    rc = rider_code.lower()
    for product_code, riders in RIDERS.items():
        for r in riders:
            if rc in r["code"].lower() or rc in r["name"].lower():
                return _json({"rider": r, "product_code": product_code, "product_name": PRODUCTS[product_code]["name"]})
    return _json({"error": f"특약 '{rider_code}'를 찾을 수 없습니다."})


@tool(args_schema=ProductCodeInput)
def eligibility_by_product_rule(product_code: str) -> str:
    """상품 규정상 가입 자격 조건(연령 범위·판매 채널·간편심사 여부)을 확인합니다.
    특정 고객 병력 기반 인수 판단은 underwriting_precheck 사용."""
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    conditions = {
        "나이": f"{p.get('min_age', '?')}세 ~ {p.get('max_age', '?')}세",
        "판매채널": p.get("channels", []),
        "갱신유형": p.get("renewal_type"),
        "간편심사": p.get("simplified_underwriting", False),
    }
    special = []
    if product_code == "B00115023":
        special.append("기존 당사 암보험 정상 유지 고객만 가입 가능")
    if p.get("simplified_underwriting"):
        special.append("간편심사 대상(유병력자 등 일반심사 어려운 고객)")
    return _json({"product_code": product_code, "name": p["name"], "conditions": conditions, "special_requirements": special})


@tool(args_schema=ProductFaqInput)
def product_faq_lookup(product_code: str, topic: str = "") -> str:
    """상품 FAQ(갱신, 면책, 감액, 보장개시)를 조회합니다."""
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    faqs = {
        "갱신": f"이 상품은 {p.get('renewal_type', '정보 없음')}입니다. "
               + (f"최대 {p.get('max_renewal_age')}세까지 갱신 가능합니다." if p.get("max_renewal_age") else "갱신 없는 상품입니다."),
        "면책": WAITING_PERIODS.get(product_code, {}).get("면책기간", "면책기간 정보를 확인해 주세요."),
        "감액": WAITING_PERIODS.get(product_code, {}).get("감액기간", "감액기간 정보를 확인해 주세요."),
        "보장개시": WAITING_PERIODS.get(product_code, {}).get("보장개시일", "보장개시일 정보를 확인해 주세요."),
    }
    tp = topic.lower()
    matched = {k: v for k, v in faqs.items() if tp in k.lower()} if tp else faqs
    return _json({"product_code": product_code, "faq": matched or faqs})


@tool(args_schema=SalesChannelInput)
def sales_channel_availability(product_code: str, channel: str = "") -> str:
    """상품의 판매 채널별 가입 가능 여부를 확인합니다."""
    p = PRODUCTS.get(product_code)
    if not p:
        return _json({"error": f"상품 '{product_code}' 없음"})
    channels = p.get("channels", [])
    result = {"product_code": product_code, "name": p["name"], "sales_status": p.get("sales_status"), "channels": channels}
    if channel:
        result["requested_channel"] = channel
        result["available"] = channel.upper() in [c.upper() for c in channels]
    return _json(result)


TOOLS = [
    product_search, product_get, product_compare,
    product_latest_version_check, rider_list, rider_search, rider_get,
    eligibility_by_product_rule, product_faq_lookup, sales_channel_availability,
]
