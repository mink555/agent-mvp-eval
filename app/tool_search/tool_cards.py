"""Tool Card SSOT — ChromaDB 임베딩용 풍부한 도구 메타데이터.

Lu et al. (2025) 연구 결과에 따르면 purpose + when_to_use + when_not_to_use + tags
필드를 포함한 확장 임베딩 텍스트가 Recall@10 / NDCG@10 지표에서 가장 크게 기여합니다.
특히 when_not_to_use(negative_examples)가 유사 도구 간 혼동을 줄이는 데 핵심적입니다.

## 카드 추가 방법
1. ToolCard 인스턴스를 _CARDS 리스트에 추가
2. `name` 필드는 LangChain tool.name 과 정확히 일치해야 합니다
3. when_to_use / when_not_to_use 는 실제 사용자 발화 패턴으로 작성하세요

## 혼동 쌍 (Confusion Pairs) 설계 기준
다음 쌍들은 when_not_to_use 에서 서로를 명시적으로 지목합니다:
  - product_search  ↔ coverage_summary  (목록 vs 보장 내용)
  - coverage_summary  ↔ coverage_detail  (전체 요약 vs 특정 유형 상세)
  - underwriting_precheck  ↔ eligibility_by_product_rule  (병력 기반 vs 연령·채널 기반)
  - claim_guide  ↔ coverage_detail  (청구 절차 vs 보장 범위)
  - rag_terms_query_engine  ↔ rag_product_info_query_engine  (약관 vs 상품요약서)
  - premium_estimate  ↔ plan_options  (보험료 금액 vs 납입 방식/플랜)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolCard:
    """단일 도구의 임베딩용 풍부한 메타데이터."""

    name: str
    """LangChain tool.name 과 정확히 일치해야 함."""

    purpose: str
    """이 도구가 하는 일 — 한 문장으로 명확하게."""

    when_to_use: tuple[str, ...] = field(default_factory=tuple)
    """이 도구를 써야 하는 대표 사용자 발화 예시 (positive examples)."""

    when_not_to_use: tuple[str, ...] = field(default_factory=tuple)
    """이 도구를 쓰면 안 되는 발화 예시 + 올바른 도구 안내 (negative examples).
    유사 도구 간 혼동을 줄이는 핵심 필드."""

    tags: tuple[str, ...] = field(default_factory=tuple)
    """도메인 태그 — 빠른 필터링용."""

    def to_embed_text(self) -> str:
        """ChromaDB 임베딩용 텍스트를 생성한다.

        when_not_to_use 는 타 도구의 어휘를 포함하므로 임베딩에서 제외한다.
        (when_not_to_use 는 LLM 도구설명 docstring용으로만 활용)
        짧고 목적 중심의 텍스트가 paraphrase 모델에서 더 집중된 벡터를 만든다.
        """
        parts = [self.purpose]
        if self.when_to_use:
            parts.extend(self.when_to_use)
        if self.tags:
            parts.append(" ".join(self.tags))
        return "\n".join(parts)


# ──────────────────────────────────────────────────────────────
# REGISTRY  (이 리스트가 유일한 진실의 원천 — SSOT)
# ──────────────────────────────────────────────────────────────
_CARDS: list[ToolCard] = [

    # ── product.py ─────────────────────────────────────────────
    ToolCard(
        name="product_search",
        purpose="판매 중인 보험 상품 목록을 키워드·카테고리로 검색한다. 어떤 상품이 있는지, 목록·리스트가 필요할 때 사용한다. 보험료·보장 조회 전 상품코드를 모를 때 선행 호출한다.",
        when_to_use=(
            "우리 회사 상품 뭐 있어?",
            "라이나생명 판매 상품 목록 알려줘",
            "치아보험 있어?",
            "암보험 상품 뭐가 있어?",
            "건강보험 종류 알려줘",
            "어떤 상품 파는지 알고 싶어",
            "전체 상품 리스트 보여줘",
            "판매 중인 상품 전체 목록",
            "종신보험 상품 있어?",
            "간편심사 상품 목록",
            "치매 관련 상품 있어?",
            "어떤 보험 상품 파는지 알고 싶어",
            "상품 카탈로그 보여줘",
            "보험 상품 목록 조회",
            "치아보험 보험료 알려줘",
            "암보험 보험료 얼마야?",
            "건강보험 보험료 계산해줘",
            "치아보험 보장 내용 알려줘",
            "암보험 가입 조건 뭐야?",
            # 신규 상품
            "실버치아보험 있어?",
            "실버치아보험 보험료 알려줘",
            "치매보험 상품 뭐 있어?",
            "실속치매보험 보험료 얼마야?",
            "전에없던 치매간병보험 알려줘",
            "첫날부터 암보험 뭐야?",
            "골라담는 간편건강보험 있어?",
            "건강해지는 종신보험 뭐야?",
            "채우는335 상품 알려줘",
        ),
        when_not_to_use=(
            "특정 상품(코드 있음)의 보장 내용이 뭐야? → coverage_summary 사용",
            "이 상품 보험료가 얼마야? → premium_estimate 사용",
            "이 상품에 가입할 수 있어? → underwriting_precheck 사용",
            "이 상품 약관에서 면책 조건 찾아줘 → rag_terms_query_engine 사용",
        ),
        tags=("상품조회", "목록", "검색", "카탈로그", "리스트"),
    ),

    ToolCard(
        name="product_get",
        purpose="상품 코드(B로 시작)로 특정 상품의 기본 정보를 조회한다.",
        when_to_use=(
            "B00197011 상품 정보 보여줘",
            "이 상품 코드로 상세 정보 알려줘",
            "B00115023 어떤 상품이야?",
        ),
        when_not_to_use=(
            "상품 이름만 알고 코드가 없다 → product_search 로 먼저 검색",
            "보장 내용이 궁금하다 → coverage_summary 사용",
            "보험료가 궁금하다 → premium_estimate 사용",
        ),
        tags=("상품조회", "코드조회"),
    ),

    ToolCard(
        name="product_compare",
        purpose="여러 상품(코드 2개 이상)을 나란히 비교한다.",
        when_to_use=(
            "암보험이랑 건강보험 비교해줘",
            "B00115023 과 B00172014 차이가 뭐야?",
            "이 두 상품 어느 게 더 나아?",
        ),
        when_not_to_use=(
            "상품 하나만 상세히 알고 싶다 → product_get 또는 coverage_summary 사용",
        ),
        tags=("상품비교",),
    ),

    ToolCard(
        name="product_latest_version_check",
        purpose="상품 요약서가 최신 버전인지 확인한다.",
        when_to_use=(
            "이 상품 요약서 최신 버전 맞아?",
            "상품 버전 확인해줘",
        ),
        tags=("버전확인",),
    ),

    ToolCard(
        name="rider_list",
        purpose="특정 상품에 부가할 수 있는 특약 목록을 조회한다.",
        when_to_use=(
            "이 상품에 어떤 특약 있어?",
            "특약 목록 보여줘",
            "의무부가 특약이 뭐야?",
            "선택 특약 종류",
        ),
        when_not_to_use=(
            "특약 코드나 이름으로 상세 내용을 알고 싶다 → rider_get 사용",
            "고객 목표에 맞는 특약을 추천받고 싶다 → rider_bundle_recommend 사용",
        ),
        tags=("특약", "목록"),
    ),

    ToolCard(
        name="rider_search",
        purpose="특정 상품의 특약을 키워드로 검색한다.",
        when_to_use=(
            "암 관련 특약 있어?",
            "이 상품에서 수술 특약 찾아줘",
        ),
        tags=("특약", "검색"),
    ),

    ToolCard(
        name="rider_get",
        purpose="특약 코드 또는 이름으로 특약의 상세 내용을 조회한다.",
        when_to_use=(
            "R001 특약 상세 내용",
            "암진단특약 자세히 알려줘",
        ),
        tags=("특약", "상세조회"),
    ),

    ToolCard(
        name="eligibility_by_product_rule",
        purpose="상품 규정 기준(나이 범위·판매 채널·간편심사 여부)으로 가입 자격 조건을 확인한다.",
        when_to_use=(
            "이 상품 몇 살까지 가입 가능해?",
            "가입 가능 나이 범위",
            "어떤 채널에서 팔아?",
            "간편심사 상품이야?",
        ),
        when_not_to_use=(
            "특정 고객의 병력·나이로 실제 가입 가능 여부를 판단해야 한다 → underwriting_precheck 사용",
        ),
        tags=("가입조건", "자격확인"),
    ),

    ToolCard(
        name="product_faq_lookup",
        purpose="상품별 자주 묻는 질문(갱신, 면책, 감액, 보장개시)을 조회한다.",
        when_to_use=(
            "이 상품 FAQ",
            "갱신 관련 자주 묻는 질문",
            "면책기간 FAQ",
        ),
        tags=("FAQ",),
    ),

    ToolCard(
        name="sales_channel_availability",
        purpose="특정 상품의 판매 채널(TM·CM·대면 등) 가입 가능 여부를 확인한다.",
        when_to_use=(
            "이 상품 TM으로 가입돼?",
            "CM 채널에서 살 수 있어?",
            "어떤 채널에서 판매하는지 알려줘",
        ),
        tags=("판매채널",),
    ),

    # ── premium.py ─────────────────────────────────────────────
    ToolCard(
        name="premium_estimate",
        purpose="나이·성별을 입력해 특정 상품의 예상 월 보험료를 산출한다. 보험료 금액이 얼마인지 계산할 때 사용한다.",
        when_to_use=(
            "이 상품 보험료 얼마야?",
            "40세 남성 보험료 계산해줘",
            "월 납입액이 얼마나 돼?",
            "보험료 산출해줘",
            "보험료 계산해줘",
            "월 보험료가 얼마야?",
            "나이별 보험료 알려줘",
            # 신규 상품
            "65세 남성 실버치아보험 보험료",
            "실속치매보험 55세 여성 보험료",
            "치매간병보험 보험료 계산해줘",
            "첫날부터 암보험 보험료 얼마야?",
            "건강해지는종신보험 50세 보험료",
            "채우는335 보험료 산출해줘",
        ),
        when_not_to_use=(
            "여러 납입 플랜·방식을 비교하고 싶다 → plan_options 사용",
            "예산 내에서 가입 가능한지 확인하고 싶다 → affordability_check 사용",
            "갱신 후 미래 보험료를 알고 싶다 → renewal_premium_projection 사용",
            "적정 보험 가입 금액(보험금)을 추천받고 싶다 → amount_suggest 사용",
        ),
        tags=("보험료", "산출", "월납입액", "보험료계산"),
    ),

    ToolCard(
        name="premium_compare",
        purpose="여러 상품의 보험료를 동일 조건(나이·성별)으로 비교한다.",
        when_to_use=(
            "두 상품 보험료 비교해줘",
            "암보험이랑 건강보험 중 어느 게 더 싸?",
        ),
        tags=("보험료", "비교"),
    ),

    ToolCard(
        name="plan_options",
        purpose="상품의 납입 방식·납입 기간 플랜 옵션(10년납, 20년납, 전기납 등)을 조회한다. 어떤 납입 방식이 있는지 알고 싶을 때 사용한다.",
        when_to_use=(
            "납입 기간 옵션 뭐 있어?",
            "10년납 20년납 중 선택 가능해?",
            "납입 방식 알려줘",
            "플랜 종류 뭐가 있어?",
            "어떤 납입 플랜이 있어?",
            "납입 기간 몇 년짜리 있어?",
            "납입 방식 종류",
            "납입 기간 선택지",
            "납입 플랜 옵션 알려줘",
        ),
        when_not_to_use=(
            "실제 보험료 금액이 궁금하다 → premium_estimate 사용",
        ),
        tags=("납입플랜", "납입방식", "납입기간"),
    ),

    ToolCard(
        name="amount_suggest",
        purpose="소득·목적 기반으로 적정 보험 가입 금액(보험금 설정)을 제안한다. 보험료가 아니라 가입 금액(보험금 한도) 추천이 목적이다.",
        when_to_use=(
            "적정 가입금액 추천해줘",
            "얼마짜리 보험 들어야 해?",
            "소득 기준 보험금액 설정",
            "보험금 얼마로 설정하면 좋아?",
            "가입 금액 얼마로 해야 해?",
        ),
        when_not_to_use=(
            "보험료(월 납입액)를 계산하고 싶다 → premium_estimate 사용",
            "청구 방법이 궁금하다 → claim_guide 사용",
            "보장 내용이 궁금하다 → coverage_summary 또는 coverage_detail 사용",
            "상품 목록이 궁금하다 → product_search 사용",
            "납입 방식이 궁금하다 → plan_options 사용",
            "면책 사유가 궁금하다 → underwriting_exclusions 사용",
            "가입 가능 여부가 궁금하다 → underwriting_precheck 사용",
            "특약 추천이 필요하다 → rider_bundle_recommend 사용",
        ),
        tags=("가입금액", "보험금", "추천"),
    ),

    ToolCard(
        name="renewal_premium_projection",
        purpose="갱신형 상품의 향후 갱신 시점별 예상 보험료를 추정한다.",
        when_to_use=(
            "갱신하면 보험료 얼마나 올라?",
            "10년 후 보험료 예상",
            "갱신 보험료 변화 추이",
        ),
        when_not_to_use=(
            "현재 기준 보험료가 궁금하다 → premium_estimate 사용",
        ),
        tags=("갱신", "보험료", "예측"),
    ),

    ToolCard(
        name="affordability_check",
        purpose="월 예산 내에서 특정 상품 가입이 가능한지 확인한다.",
        when_to_use=(
            "한 달에 5만원 예산으로 이 보험 들 수 있어?",
            "예산 내 가입 가능 여부",
        ),
        tags=("예산", "가입가능"),
    ),

    ToolCard(
        name="payment_cycle_options",
        purpose="상품의 납입 주기 옵션(월납·연납 등)을 조회한다.",
        when_to_use=(
            "월납 말고 연납도 돼?",
            "납입 주기 선택 가능해?",
        ),
        tags=("납입주기",),
    ),

    ToolCard(
        name="surrender_value_explain",
        purpose="해약환급금 규정을 설명한다.",
        when_to_use=(
            "해약하면 돈 얼마 돌려받아?",
            "해약환급금이 얼마야?",
            "중도 해지 시 환급금",
        ),
        tags=("해약", "환급금"),
    ),

    # ── coverage.py ─────────────────────────────────────────────
    ToolCard(
        name="coverage_summary",
        purpose="특정 상품(코드 필요)의 전체 보장 내용을 한눈에 요약한다. 이 상품이 무엇을 보장하는지 전체 범위가 궁금할 때 사용한다.",
        when_to_use=(
            "이 상품 보장이 뭐야?",
            "B00197011 보장 내용 알려줘",
            "이 보험 뭘 보장해줘?",
            "보장 범위 전체 보여줘",
            "이 상품은 어떤 보장을 해줘?",
            "보장 내용 요약해줘",
            "어떤 질병을 보장하는지 알고 싶어",
            "상품 보장 내용 전체 알려줘",
            "이 보험 보장 범위 알려줘",
            "어떤 보장이 있어?",
            "보장되는 항목 알려줘",
        ),
        when_not_to_use=(
            "어떤 상품들이 있는지 목록이 궁금하다 → product_search 사용",
            "암이나 치아 등 특정 보장 유형만 상세히 알고 싶다 → coverage_detail 사용",
            "보험료가 궁금하다 → premium_estimate 사용",
            "상품 코드가 없고 카테고리로 찾고 싶다 → product_search 사용",
        ),
        tags=("보장내용", "보장범위", "요약"),
    ),

    ToolCard(
        name="coverage_detail",
        purpose="상품의 특정 보장 유형(암·사망·치아·입원 등)을 상세 조회한다. 특정 질병·사고 유형의 보장만 따로 보고 싶을 때 사용한다.",
        when_to_use=(
            "암 진단금이 얼마야?",
            "치아 보장이 구체적으로 어떻게 돼?",
            "사망보험금 상세 내용",
            "입원 특약 상세",
            "이 상품에서 입원 보장만 따로 보고 싶어",
            "암 보장 상세히 알려줘",
            "사망 관련 보장 내용만 알고 싶어",
            "치아 보장 상세 조회",
            "입원 보장 상세히 알려줘",
            "암에 대한 보장만 따로 알려줘",
            "치아 보장 어떻게 돼?",
        ),
        when_not_to_use=(
            "전체 보장 요약이 필요하다 → coverage_summary 사용",
            "보험료가 궁금하다 → premium_estimate 사용",
        ),
        tags=("보장내용", "상세조회", "특정보장"),
    ),

    ToolCard(
        name="benefit_amount_lookup",
        purpose="특정 급부(보장 항목)의 보장 금액을 조회한다.",
        when_to_use=(
            "암 진단금 얼마야?",
            "입원급부금 금액",
            "이 급부 얼마 받아?",
        ),
        tags=("급부금액",),
    ),

    ToolCard(
        name="benefit_limit_rules",
        purpose="급부별 보장 한도 규칙(연간 한도·횟수 제한 등)을 조회한다.",
        when_to_use=(
            "치아 보장 연간 몇 개까지야?",
            "한도 규칙 알려줘",
            "크라운 연간 한도",
        ),
        tags=("보장한도", "제한규칙"),
    ),

    ToolCard(
        name="event_eligibility_check",
        purpose="사고·질병 사실관계를 바탕으로 보장 대상 여부를 사전 검토한다.",
        when_to_use=(
            "이런 상황이면 보험금 받을 수 있어?",
            "교통사고 치료는 보장돼?",
            "3개월 전 입원 이력으로 보장 가능한지 확인해줘",
        ),
        when_not_to_use=(
            "가입 자격(인수 적합성)이 궁금하다 → underwriting_precheck 사용",
        ),
        tags=("보장가능여부", "사전검토"),
    ),

    ToolCard(
        name="diagnosis_definition_lookup",
        purpose="의학·보험 용어의 약관상 공식 정의를 조회한다.",
        when_to_use=(
            "약관에서 암의 정의가 뭐야?",
            "뇌출혈 약관 정의",
            "치주질환이 뭐야?",
        ),
        tags=("용어정의", "약관"),
    ),

    ToolCard(
        name="icd_mapping_lookup",
        purpose="ICD/KCD 질병 코드와 질병명 간의 매핑을 조회한다.",
        when_to_use=(
            "C50이 무슨 병이야?",
            "ICD 코드로 질병 찾아줘",
            "KCD 코드 매핑",
        ),
        tags=("ICD", "KCD", "질병코드"),
    ),

    ToolCard(
        name="multi_benefit_conflict_rule",
        purpose="동일 사고에서 여러 급부가 중복 청구될 때의 충돌 규칙을 확인한다.",
        when_to_use=(
            "같은 치아에 충전이랑 크라운 동시에 청구하면?",
            "중복 급부 규칙",
            "동일 사고 두 가지 보장 동시 가능해?",
        ),
        tags=("중복급부", "충돌규칙"),
    ),

    ToolCard(
        name="rider_bundle_recommend",
        purpose="고객의 목표(암 대비·치아 보호 등)에 맞는 특약 조합을 추천한다. 어떤 특약을 붙이면 좋은지 추천이 필요할 때 사용한다.",
        when_to_use=(
            "암 대비에 좋은 특약 추천해줘",
            "치아 관련 특약 조합",
            "어떤 특약 붙이는 게 좋아?",
            "고객 목표에 맞는 특약 추천해줘",
            "특약 추천해줘",
            "어떤 특약을 선택하면 좋아?",
            "특약 어떤 거 선택하면 좋아?",
            "특약 조합 추천",
            "어떤 특약이 좋을까?",
        ),
        when_not_to_use=(
            "특약 목록만 보고 싶다 → rider_list 사용",
        ),
        tags=("특약추천", "특약조합", "특약선택"),
    ),

    # ── underwriting.py ─────────────────────────────────────────
    ToolCard(
        name="underwriting_precheck",
        purpose="나이·성별·병력 요약을 기반으로 이 고객이 보험에 가입 가능한지 인수 적합성을 사전 판단한다. 특정 고객의 병력·건강 이력으로 가입 가능 여부를 확인할 때 사용한다.",
        when_to_use=(
            "당뇨 이력 있어도 가입 가능해?",
            "고혈압인데 암보험 들 수 있어?",
            "병력 있는 고객 인수 가능 여부 확인",
            "55세 남성 기존 수술 이력 있는데 가입돼?",
            "기존 질환 있는데 보험 가입 될까?",
            "이 고객 병력으로 인수 심사 통과 가능해?",
        ),
        when_not_to_use=(
            "상품 규정상 가입 가능 나이 범위만 궁금하다 → eligibility_by_product_rule 사용",
            "어떤 조건이면 무조건 거절되는지 알고 싶다 → underwriting_knockout_rules 사용",
        ),
        tags=("인수심사", "가입가능여부", "병력", "인수적합성"),
    ),

    ToolCard(
        name="underwriting_questions_generator",
        purpose="상품별 인수심사에 필요한 고지 질문 목록을 생성한다.",
        when_to_use=(
            "암보험 가입 시 어떤 질문 해야 해?",
            "인수심사 고지 항목",
            "고지의무 질문 목록",
        ),
        tags=("인수심사", "고지의무", "질문목록"),
    ),

    ToolCard(
        name="underwriting_knockout_rules",
        purpose="상품별 인수 거절(녹아웃) 조건을 조회한다.",
        when_to_use=(
            "어떤 경우 가입 무조건 안 돼?",
            "거절 사유 목록",
            "이 상품 녹아웃 조건",
        ),
        when_not_to_use=(
            "특정 고객 병력으로 가입 가능한지 확인하고 싶다 → underwriting_precheck 사용",
        ),
        tags=("인수심사", "거절조건", "녹아웃"),
    ),

    ToolCard(
        name="underwriting_docs_required",
        purpose="가입 시 필요한 서류 목록을 확인한다.",
        when_to_use=(
            "가입에 필요한 서류가 뭐야?",
            "제출 서류 목록",
            "가입 서류 준비",
        ),
        tags=("인수심사", "서류"),
    ),

    ToolCard(
        name="underwriting_waiting_periods",
        purpose="상품의 면책기간·감액기간·보장개시일을 조회한다.",
        when_to_use=(
            "면책기간이 얼마야?",
            "가입하고 언제부터 보장돼?",
            "보장개시일",
            "감액기간 알려줘",
        ),
        tags=("면책기간", "보장개시", "감액기간"),
    ),

    ToolCard(
        name="underwriting_exclusions",
        purpose="상품의 보장 제외(면책) 사유를 조회한다. 보장이 안 되는 경우, 면책 항목 목록이 필요할 때 사용한다.",
        when_to_use=(
            "보장 안 되는 경우가 뭐야?",
            "면책 사유 목록",
            "이런 경우 보장 안 돼?",
            "보장 제외 사항 알려줘",
            "어떤 경우에 보험금 안 나와?",
            "면책 조항 목록",
        ),
        when_not_to_use=(
            "면책기간(기간 질문)이 궁금하다 → underwriting_waiting_periods 사용",
        ),
        tags=("면책", "보장제외", "면책사유"),
    ),

    ToolCard(
        name="underwriting_limitations",
        purpose="상품별 최소·최대 가입금액 한도를 조회한다.",
        when_to_use=(
            "최대 얼마까지 가입 가능해?",
            "가입금액 한도",
            "최소 가입금액",
        ),
        tags=("가입금액", "한도"),
    ),

    ToolCard(
        name="underwriting_reinstatement_rule",
        purpose="실효된 계약의 부활(복원) 규정을 조회한다.",
        when_to_use=(
            "보험 실효됐는데 다시 살릴 수 있어?",
            "부활 신청 조건",
            "계약 복원 방법",
        ),
        tags=("계약부활", "실효"),
    ),

    ToolCard(
        name="underwriting_renewal_eligibility",
        purpose="갱신 가능 여부 및 최대 갱신 가능 나이를 확인한다.",
        when_to_use=(
            "이 상품 갱신 가능해?",
            "몇 살까지 갱신돼?",
            "갱신 자격 확인",
        ),
        when_not_to_use=(
            "갱신 후 보험료가 얼마나 오를지 알고 싶다 → underwriting_renewal_premium_notice 사용",
        ),
        tags=("갱신", "갱신자격"),
    ),

    ToolCard(
        name="underwriting_renewal_premium_notice",
        purpose="갱신 시 예상 보험료 인상 범위를 안내한다.",
        when_to_use=(
            "갱신하면 보험료 얼마나 올라?",
            "갱신 보험료 변동 안내",
            "갱신 후 예상 금액",
        ),
        tags=("갱신", "보험료인상"),
    ),

    ToolCard(
        name="underwriting_high_risk_job_check",
        purpose="직업 위험도를 확인해 가입 제한 또는 보험료 할증 여부를 판단한다.",
        when_to_use=(
            "소방관도 가입 가능해?",
            "이 직업 위험직군이야?",
            "직업 때문에 가입 안 될 수 있어?",
            "직업 위험도 확인해줘",
            "이 직업 할증 대상이야?",
            "직업 위험등급 알려줘",
            "배달라이더 가입 가능해?",
        ),
        tags=("직업위험도", "인수심사"),
    ),

    ToolCard(
        name="underwriting_disclosure_risk_score",
        purpose="병력 요약을 입력해 고지 위험 점수를 산출하고 주의 항목을 안내한다.",
        when_to_use=(
            "이 병력 고지해야 해?",
            "고지 위험도 평가",
            "당뇨·고혈압 고지 필요 여부",
        ),
        tags=("고지의무", "위험점수"),
    ),

    # ── compliance.py ─────────────────────────────────────────
    ToolCard(
        name="compliance_required_disclosure",
        purpose="상품별 법적 필수 설명사항(중요사항 고지 의무)을 조회한다.",
        when_to_use=(
            "이 상품 필수 설명 항목이 뭐야?",
            "법적으로 설명해야 할 내용",
            "중요사항 고지 내용",
        ),
        tags=("컴플라이언스", "필수설명"),
    ),

    ToolCard(
        name="compliance_phrase_generator",
        purpose="상품·주제별 준법 멘트 스크립트(면책·감액·갱신·해약 등)를 생성한다. 준법 멘트나 안내 문구를 새로 만들어야 할 때 사용한다.",
        when_to_use=(
            "면책 관련 준법 멘트 만들어줘",
            "갱신 안내 스크립트",
            "해약 환급금 안내 문구 만들어줘",
            "준법 멘트 생성해줘",
            "이 주제로 안내 스크립트 작성해줘",
            "멘트 만들어줘",
        ),
        when_not_to_use=(
            "판매 스크립트에 금칙어가 있는지 검사하고 싶다 → compliance_misleading_check 사용",
        ),
        tags=("컴플라이언스", "스크립트", "멘트생성", "준법멘트"),
    ),

    ToolCard(
        name="compliance_misleading_check",
        purpose="판매 스크립트·멘트에 금칙어·과장표현·규정 위반 표현이 있는지 검사한다. 특정 문구나 표현의 사용 가능 여부를 확인할 때 사용한다.",
        when_to_use=(
            "이 문구 써도 돼?",
            "이 스크립트에 문제 있어?",
            "금칙어 검사해줘",
            "과장광고 여부 확인",
            "이 표현 사용해도 괜찮아?",
            "이 멘트에 금칙어 있어?",
            "규정 위반 문구인지 확인해줘",
            "이 말 써도 되는지 확인해줘",
            "이 표현 문제 없어?",
            "판매 문구 검사해줘",
        ),
        when_not_to_use=(
            "새로운 준법 멘트를 생성하고 싶다 → compliance_phrase_generator 사용",
        ),
        tags=("컴플라이언스", "금칙어", "스크립트검사", "문구검사"),
    ),

    ToolCard(
        name="comparison_disclaimer_generator",
        purpose="상품 비교 시 법적으로 필요한 면책 문구를 생성한다.",
        when_to_use=(
            "상품 비교할 때 면책 문구 만들어줘",
            "비교 광고 면책",
        ),
        tags=("컴플라이언스", "면책문구", "비교"),
    ),

    ToolCard(
        name="recording_notice_script",
        purpose="채널별(TM·CM) 녹취 고지 스크립트를 생성한다.",
        when_to_use=(
            "TM 녹취 고지 멘트",
            "CM 채널 녹취 스크립트",
            "전화 상담 녹취 안내",
        ),
        tags=("컴플라이언스", "녹취", "TM"),
    ),

    ToolCard(
        name="privacy_masking",
        purpose="텍스트에서 주민번호·전화번호·카드번호·이메일 등 개인정보를 마스킹한다.",
        when_to_use=(
            "이 텍스트에서 개인정보 지워줘",
            "주민번호 마스킹",
            "개인정보 비식별화",
            "주민번호 지워줘",
            "개인정보 숨겨줘",
            "전화번호 마스킹해줘",
        ),
        tags=("개인정보", "마스킹", "PII"),
    ),

    # ── claims.py ─────────────────────────────────────────────
    ToolCard(
        name="claim_guide",
        purpose="보험금 청구 유형(사망·진단·입원·수술)별 절차와 필요 사항을 안내한다. 청구하는 방법, 절차, 프로세스가 궁금할 때 사용한다.",
        when_to_use=(
            "보험금 청구 어떻게 해?",
            "암 진단 후 청구 절차",
            "사망보험금 청구 방법",
            "입원비 청구하려면?",
            "청구 방법 알려줘",
            "보험금 신청 방법",
            "보험 청구 프로세스",
            "입원비 청구 방법",
            "청구하는 방법 알고 싶어",
            "보험금 어떻게 받아?",
            "보험금 청구 절차 알려줘",
            "치과 치료비 청구 방법",
        ),
        when_not_to_use=(
            "어떤 상황이 보장되는지 내용이 궁금하다 → coverage_detail 또는 event_eligibility_check 사용",
            "청구 서류가 뭔지 알고 싶다 → claim_required_forms 사용",
        ),
        tags=("청구", "청구절차", "청구방법"),
    ),

    ToolCard(
        name="claim_required_forms",
        purpose="청구 유형별 필요 서류 및 양식을 조회한다.",
        when_to_use=(
            "청구할 때 필요한 서류가 뭐야?",
            "진단서 외에 뭐가 필요해?",
            "청구 서류 목록",
        ),
        tags=("청구", "서류"),
    ),

    ToolCard(
        name="contract_manage",
        purpose="계약 관리(조회·갱신·해지·대출·변경·부활) 관련 절차를 안내한다.",
        when_to_use=(
            "계약 해지하고 싶어",
            "보험계약대출 받으려면?",
            "수익자 변경 방법",
            "계약 조회",
        ),
        tags=("계약관리", "유지관리"),
    ),

    ToolCard(
        name="customer_followup_tasks",
        purpose="상담 사례 요약을 입력받아 후속 조치 To-Do 목록을 생성한다.",
        when_to_use=(
            "이 상담 후 다음에 뭘 해야 해?",
            "후속 조치 목록 만들어줘",
            "상담 후속 업무 정리",
        ),
        tags=("상담관리", "후속조치"),
    ),

    # ── customer_db.py ─────────────────────────────────────────
    ToolCard(
        name="customer_contract_lookup",
        purpose="고객 ID 또는 이름으로 기존 계약 목록을 조회한다.",
        when_to_use=(
            "홍길동 고객 계약 조회",
            "고객 ID C001의 계약 내역",
            "이 고객 어떤 보험 들었어?",
        ),
        tags=("고객DB", "계약조회"),
    ),

    ToolCard(
        name="duplicate_enrollment_check",
        purpose="고객의 기존 계약을 기반으로 특정 상품 추가 가입 가능 여부를 판단한다.",
        when_to_use=(
            "이 고객 암보험 중복 가입돼?",
            "이미 가입한 상품이랑 중복 여부",
            "동일 상품 또 들 수 있어?",
        ),
        tags=("고객DB", "중복가입"),
    ),

    ToolCard(
        name="customer_search",
        purpose="내부 DB에서 이름·나이·성별 조건으로 고객을 검색한다.",
        when_to_use=(
            "40대 남성 고객 찾아줘",
            "홍씨 성의 고객 목록",
            "DB에서 고객 검색",
        ),
        tags=("고객DB", "고객검색"),
    ),

    # ── rag_tools.py ─────────────────────────────────────────
    ToolCard(
        name="rag_terms_query_engine",
        purpose="약관·규정 문서에서 면책·예외·정의·고지의무 등 관련 내용을 검색한다. 약관 원문, 규정 문서를 검색해야 할 때 사용한다.",
        when_to_use=(
            "약관에서 면책 조건 찾아줘",
            "고지의무 규정이 약관에 어떻게 나와 있어?",
            "약관상 암의 정의",
            "보험 규정 문서에서 검색",
            "약관에서 찾아줘",
            "약관 내용 검색",
            "규정상 고지의무",
            "약관 원문 찾아줘",
            "보험 약관 조항 확인",
            "약관에서 면책 사유 찾아줘",
            "고지의무 약관 내용",
            "약관에 따르면",
        ),
        when_not_to_use=(
            "상품요약서나 회사 정보를 검색하고 싶다 → rag_product_info_query_engine 사용",
            "보장 금액이나 구조가 궁금하다 → coverage_summary 또는 coverage_detail 사용",
        ),
        tags=("RAG", "약관", "규정", "약관검색"),
    ),

    ToolCard(
        name="rag_product_info_query_engine",
        purpose="상품요약서·회사정보 문서에서 상품 관련 내용을 검색한다. 약관 문서는 제외된다.",
        when_to_use=(
            "상품요약서에서 보장 내용 찾아줘",
            "회사 소개 문서에서 찾아줘",
            "이 상품 요약서 내용",
        ),
        when_not_to_use=(
            "약관 원문이나 규정을 검색하고 싶다 → rag_terms_query_engine 사용",
            "정형화된 보장 데이터가 필요하다 → coverage_summary 사용",
        ),
        tags=("RAG", "상품요약서"),
    ),
]

# ──────────────────────────────────────────────────────────────
# 공개 인터페이스
# ──────────────────────────────────────────────────────────────
REGISTRY: dict[str, ToolCard] = {card.name: card for card in _CARDS}


def get_card(tool_name: str) -> ToolCard | None:
    """도구 이름으로 ToolCard를 반환한다. 없으면 None."""
    return REGISTRY.get(tool_name)


def all_cards() -> list[ToolCard]:
    """등록된 모든 ToolCard를 반환한다."""
    return list(REGISTRY.values())


def missing_cards(tool_names: list[str]) -> list[str]:
    """카드가 없는 도구 이름 목록을 반환한다. 신규 tool 추가 시 경고용."""
    return [n for n in tool_names if n not in REGISTRY]
