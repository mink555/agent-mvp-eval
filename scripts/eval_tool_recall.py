"""Tool Search Recall@k 평가 스크립트.

사용자 발화(query) → 정답 도구 이름(expected)을 정의하고
ChromaDB 검색이 top-k 안에 정답을 포함시키는지 측정합니다.

지표:
  ── Tool-Call 쿼리 (expected != None) ──
  Hit@1     — top-1 이 정답인 비율
  Recall@k  — top-k 안에 정답이 1개 이상 있는 비율
  MRR       — Mean Reciprocal Rank (정답이 처음 등장하는 순위의 역수 평균)
  Tool Acc  — top-1 이 정답인 비율 (= Hit@1)

  ── No-Call 쿼리 (expected == None) ──
  No-Call Acc — top-1 score 가 threshold 미만인 비율 (도구 불필요 판별)

  ── 전체 ──
  Overall Acc — (Tool Acc 정답 수 + No-Call Acc 정답 수) / 전체 쿼리 수

실행:
  python -m scripts.eval_tool_recall
  python -m scripts.eval_tool_recall --k 5 --verbose
  python -m scripts.eval_tool_recall --compare              # k=1,3,5,7,10 비교표
  python -m scripts.eval_tool_recall --compare --ks 3 5 7   # 커스텀 k 값
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

# ──────────────────────────────────────────────────────────────
# 테스트 케이스 정의
#
# 형식: (query, expected_tool_name | None)
#   expected=None → 도구를 호출하면 안 되는 쿼리 (No-Call)
# 혼동 쌍(confusion pair)은 주석으로 표시합니다.
# ──────────────────────────────────────────────────────────────
TEST_CASES: list[tuple[str, str | None]] = [

    # ══════════════════════════════════════════════════════════
    #  TOOL-CALL 케이스 (도구를 호출해야 하는 쿼리)
    # ══════════════════════════════════════════════════════════

    # ── product_search ──────────────────────────────────────
    ("우리 회사 상품 뭐 있어?", "product_search"),
    ("라이나생명 판매 상품 목록 알려줘", "product_search"),
    ("치아보험 있어?", "product_search"),
    ("암보험 상품 뭐가 있어?", "product_search"),
    ("전체 상품 리스트 보여줘", "product_search"),
    ("종신보험 상품 있어?", "product_search"),
    ("치매 관련 상품 있어?", "product_search"),
    ("간편심사 상품 목록", "product_search"),
    # 혼동: coverage_summary vs product_search
    ("어떤 보험 상품 파는지 알고 싶어", "product_search"),

    # ── coverage_summary ────────────────────────────────────
    ("이 상품 보장이 뭐야?", "coverage_summary"),
    ("B00197011 보장 내용 알려줘", "coverage_summary"),
    ("이 보험 뭘 보장해줘?", "coverage_summary"),
    # 혼동: coverage_summary vs product_search
    ("보장 범위 전체 보여줘", "coverage_summary"),

    # ── coverage_detail ─────────────────────────────────────
    ("암 진단금이 얼마야?", "coverage_detail"),
    ("치아 보장이 구체적으로 어떻게 돼?", "coverage_detail"),
    ("사망보험금 상세 내용", "coverage_detail"),
    # 혼동: coverage_detail vs coverage_summary
    ("이 상품에서 입원 보장만 따로 보고 싶어", "coverage_detail"),

    # ── premium_estimate ────────────────────────────────────
    ("이 상품 보험료 얼마야?", "premium_estimate"),
    ("40세 남성 보험료 계산해줘", "premium_estimate"),
    ("월 납입액이 얼마나 돼?", "premium_estimate"),
    # 혼동: premium_estimate vs plan_options
    ("보험료 산출해줘", "premium_estimate"),

    # ── plan_options ────────────────────────────────────────
    ("납입 기간 옵션 뭐 있어?", "plan_options"),
    ("10년납 20년납 중 선택 가능해?", "plan_options"),
    # 혼동: plan_options vs premium_estimate
    ("납입 방식 알려줘", "plan_options"),

    # ── underwriting_precheck ───────────────────────────────
    ("당뇨 이력 있어도 가입 가능해?", "underwriting_precheck"),
    ("고혈압인데 암보험 들 수 있어?", "underwriting_precheck"),
    ("55세 남성 기존 수술 이력 있는데 가입돼?", "underwriting_precheck"),
    # 혼동: underwriting_precheck vs eligibility_by_product_rule
    ("병력 있는 고객 인수 가능 여부 확인", "underwriting_precheck"),

    # ── eligibility_by_product_rule ─────────────────────────
    ("이 상품 몇 살까지 가입 가능해?", "eligibility_by_product_rule"),
    ("가입 가능 나이 범위", "eligibility_by_product_rule"),
    ("어떤 채널에서 팔아?", "eligibility_by_product_rule"),

    # ── claim_guide ─────────────────────────────────────────
    ("보험금 청구 어떻게 해?", "claim_guide"),
    ("암 진단 후 청구 절차", "claim_guide"),
    ("입원비 청구하려면?", "claim_guide"),
    # 혼동: claim_guide vs coverage_detail
    ("청구 방법 알려줘", "claim_guide"),

    # ── underwriting_waiting_periods ────────────────────────
    ("면책기간이 얼마야?", "underwriting_waiting_periods"),
    ("가입하고 언제부터 보장돼?", "underwriting_waiting_periods"),
    ("보장개시일이 언제야?", "underwriting_waiting_periods"),

    # ── underwriting_exclusions ─────────────────────────────
    ("보장 안 되는 경우가 뭐야?", "underwriting_exclusions"),
    ("면책 사유 목록", "underwriting_exclusions"),

    # ── rag_terms_query_engine ──────────────────────────────
    ("약관에서 면책 조건 찾아줘", "rag_terms_query_engine"),
    ("약관상 암의 정의", "rag_terms_query_engine"),
    # 혼동: rag_terms vs rag_product_info
    ("고지의무 규정이 약관에 어떻게 나와 있어?", "rag_terms_query_engine"),

    # ── rag_product_info_query_engine ───────────────────────
    ("상품요약서에서 보장 내용 찾아줘", "rag_product_info_query_engine"),
    ("이 상품 요약서 내용", "rag_product_info_query_engine"),

    # ── compliance ──────────────────────────────────────────
    ("이 문구 써도 돼?", "compliance_misleading_check"),
    ("이 스크립트에 금칙어 있어?", "compliance_misleading_check"),
    ("면책 관련 준법 멘트 만들어줘", "compliance_phrase_generator"),
    ("TM 녹취 고지 멘트", "recording_notice_script"),
    ("개인정보 마스킹해줘", "privacy_masking"),
    ("주민번호 지워줘", "privacy_masking"),

    # ── customer_db ─────────────────────────────────────────
    ("홍길동 고객 계약 조회", "customer_contract_lookup"),
    ("이 고객 중복 가입 돼?", "duplicate_enrollment_check"),

    # ── misc ────────────────────────────────────────────────
    ("갱신하면 보험료 얼마나 올라?", "renewal_premium_projection"),
    ("직업 위험도 확인해줘", "underwriting_high_risk_job_check"),
    ("소방관도 가입 가능해?", "underwriting_high_risk_job_check"),
    ("이 병력 고지해야 해?", "underwriting_disclosure_risk_score"),
    ("해약하면 돈 얼마 돌려받아?", "surrender_value_explain"),
    ("계약 해지하고 싶어", "contract_manage"),
    ("치아 보장 연간 몇 개까지야?", "benefit_limit_rules"),
    ("암 진단금 얼마 받아?", "benefit_amount_lookup"),
    ("ICD 코드 C50 이 무슨 병이야?", "icd_mapping_lookup"),
    ("고객 목표에 맞는 특약 추천해줘", "rider_bundle_recommend"),
    ("동일 치아 중복 청구 규칙", "multi_benefit_conflict_rule"),

    # ══════════════════════════════════════════════════════════
    #  NO-CALL 케이스 (도구를 호출하면 안 되는 쿼리)
    #
    #  보험 도메인 안이지만 특정 도구가 필요 없는 일반 질문,
    #  또는 인사/감사/확인 등 대화형 발화.
    # ══════════════════════════════════════════════════════════

    # ── 일반 보험 지식 (도구 없이 LLM이 직접 답할 수 있음) ──
    ("보험이란 무엇인가요?", None),
    ("종신보험이랑 정기보험 차이가 뭐야?", None),
    ("실손보험 뜻이 뭐야?", None),
    ("보험료와 보험금의 차이", None),
    ("보험 가입 시 주의사항이 뭐야?", None),

    # ── 대화형 발화 (도구 불필요) ──
    ("감사합니다 잘 알겠습니다", None),
    ("네 알겠어요", None),
    ("방금 말씀해주신 내용 요약해줘", None),
    ("좀 더 쉽게 설명해줄 수 있어?", None),
    ("다른 건 없어요 감사합니다", None),

    # ── 도메인 내이지만 모호한 질문 (특정 도구 매핑 불가) ──
    ("보험 들 때 뭘 확인해야 할까?", None),
    ("보험 설계사한테 뭘 물어봐야 해?", None),
    ("보험 하나만 들려면 뭐가 좋을까?", None),
    ("보험 해지하면 불이익이 있나요?", None),
    ("보험료를 아끼는 방법이 있을까?", None),
]


# ──────────────────────────────────────────────────────────────
# 평가 로직
# ──────────────────────────────────────────────────────────────
DEFAULT_NO_CALL_THRESHOLD = 0.86

@dataclass
class EvalResult:
    query: str
    expected: str | None
    ranked: list[str]
    scores: list[float] = field(default_factory=list)
    hit_rank: int | None = None

    @property
    def is_no_call(self) -> bool:
        return self.expected is None

    @property
    def top_score(self) -> float:
        return self.scores[0] if self.scores else 0.0


def _reciprocal_rank(result: EvalResult) -> float:
    return 1.0 / result.hit_rank if result.hit_rank else 0.0


def _run_search(k: int) -> list[EvalResult]:
    """TEST_CASES 를 실행하고 EvalResult 목록을 반환."""
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from app.tools import get_all_tools
    from app.tool_search.embedder import get_tool_search

    searcher = get_tool_search()
    all_tools = get_all_tools()
    searcher.index_tools(all_tools)

    results: list[EvalResult] = []
    for query, expected in TEST_CASES:
        candidates = searcher.search(query, top_k=k)
        ranked = [c.name for c in candidates]
        scores = [c.score for c in candidates]

        if expected is not None:
            hit_rank = next(
                (i + 1 for i, name in enumerate(ranked) if name == expected),
                None,
            )
        else:
            hit_rank = None

        results.append(EvalResult(
            query=query, expected=expected,
            ranked=ranked, scores=scores, hit_rank=hit_rank,
        ))

    return results


def _compute_metrics(results: list[EvalResult], k: int, threshold: float) -> dict:
    """결과 리스트에서 지표를 계산."""
    tool_call = [r for r in results if not r.is_no_call]
    no_call = [r for r in results if r.is_no_call]

    tc_total = len(tool_call)
    nc_total = len(no_call)
    total = len(results)

    hit1 = sum(1 for r in tool_call if r.hit_rank == 1)
    recall = sum(1 for r in tool_call if r.hit_rank is not None)
    mrr = sum(_reciprocal_rank(r) for r in tool_call) / tc_total if tc_total else 0.0

    nc_correct = sum(1 for r in no_call if r.top_score < threshold)

    tool_acc = hit1 / tc_total if tc_total else 0.0
    recall_at_k = recall / tc_total if tc_total else 0.0
    no_call_acc = nc_correct / nc_total if nc_total else 0.0
    overall_acc = (hit1 + nc_correct) / total if total else 0.0

    return {
        "k": k,
        "tc_total": tc_total,
        "nc_total": nc_total,
        "total": total,
        "hit1": hit1,
        "recall": recall,
        "nc_correct": nc_correct,
        "tool_acc": tool_acc,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "no_call_acc": no_call_acc,
        "overall_acc": overall_acc,
    }


# ──────────────────────────────────────────────────────────────
# 출력
# ──────────────────────────────────────────────────────────────
def _print_single(results: list[EvalResult], k: int, threshold: float,
                  verbose: bool) -> None:
    """단일 k 에 대한 상세 출력."""
    m = _compute_metrics(results, k, threshold)
    sep = "─" * 72

    print(f"\n{'=' * 72}")
    print(f"  Tool Search 평가  (k={k}, threshold={threshold})")
    print(f"  쿼리 수: tool-call {m['tc_total']}개 + no-call {m['nc_total']}개 = 총 {m['total']}개")
    print(f"{'=' * 72}")

    print(f"\n  ── Tool-Call 지표 ({m['tc_total']}개 쿼리) ──")
    print(f"  Tool Acc (Hit@1) : {m['tool_acc']:.1%}  ({m['hit1']}/{m['tc_total']})")
    print(f"  Recall@{k:<2}        : {m['recall_at_k']:.1%}  ({m['recall']}/{m['tc_total']})")
    print(f"  MRR              : {m['mrr']:.4f}")

    print(f"\n  ── No-Call 지표 ({m['nc_total']}개 쿼리, threshold={threshold}) ──")
    print(f"  No-Call Acc      : {m['no_call_acc']:.1%}  ({m['nc_correct']}/{m['nc_total']})")

    print(f"\n  ── 종합 ──")
    print(f"  Overall Acc      : {m['overall_acc']:.1%}  ({m['hit1'] + m['nc_correct']}/{m['total']})")
    print(sep)

    # 미탐 (tool-call 쿼리)
    tool_call = [r for r in results if not r.is_no_call]
    misses = [r for r in tool_call if r.hit_rank is None]
    if misses:
        print(f"\n  ❌ Tool-Call 미탐 ({len(misses)}개):")
        for r in misses:
            top3 = ", ".join(r.ranked[:3])
            print(f"    [{r.expected}]  '{r.query}'")
            print(f"      → top-3: {top3}  (scores: {', '.join(f'{s:.3f}' for s in r.scores[:3])})")
    else:
        print(f"\n  ✅ 모든 tool-call 쿼리가 top-{k} 안에 정답 포함")

    # No-Call 오판 (높은 점수로 도구가 매칭된 경우)
    no_call = [r for r in results if r.is_no_call]
    nc_fails = [r for r in no_call if r.top_score >= threshold]
    if nc_fails:
        print(f"\n  ⚠️  No-Call 오판 ({len(nc_fails)}개 — top-1 score ≥ {threshold}):")
        for r in nc_fails:
            print(f"    '{r.query}'")
            print(f"      → top-1: {r.ranked[0]} (score={r.top_score:.3f})")
    else:
        print(f"\n  ✅ 모든 no-call 쿼리가 threshold({threshold}) 미만")

    # No-Call 점수 분포
    if no_call:
        nc_scores = [r.top_score for r in no_call]
        print(f"\n  📊 No-Call top-1 score 분포:")
        print(f"     min={min(nc_scores):.3f}  avg={sum(nc_scores)/len(nc_scores):.3f}  max={max(nc_scores):.3f}")

    # Tool-Call 점수 분포
    if tool_call:
        tc_scores = [r.scores[0] for r in tool_call if r.scores]
        print(f"  📊 Tool-Call top-1 score 분포:")
        print(f"     min={min(tc_scores):.3f}  avg={sum(tc_scores)/len(tc_scores):.3f}  max={max(tc_scores):.3f}")

    if verbose:
        _print_verbose(results)

    print()


def _print_verbose(results: list[EvalResult]) -> None:
    """전체 결과 상세 출력."""
    tool_call = [r for r in results if not r.is_no_call]
    no_call = [r for r in results if r.is_no_call]

    print(f"\n  📋 Tool-Call 전체 결과:")
    print(f"  {'':>2} {'순위':>4}  {'score':>6}  {'정답 도구':<38}  쿼리")
    print(f"  {'':>2} {'─'*4}  {'─'*6}  {'─'*38}  {'─'*30}")
    for r in sorted(tool_call, key=lambda x: x.hit_rank or 9999):
        rank_str = f"#{r.hit_rank}" if r.hit_rank else "miss"
        score_str = f"{r.top_score:.3f}" if r.scores else "  -  "
        mark = "✅" if r.hit_rank and r.hit_rank <= 3 else ("⚠️" if r.hit_rank else "❌")
        print(f"  {mark} {rank_str:>4}  {score_str:>6}  {r.expected:<38}  {r.query}")

    print(f"\n  📋 No-Call 전체 결과:")
    print(f"  {'':>2} {'score':>6}  {'top-1 도구':<38}  쿼리")
    print(f"  {'':>2} {'─'*6}  {'─'*38}  {'─'*30}")
    for r in sorted(no_call, key=lambda x: -x.top_score):
        score_str = f"{r.top_score:.3f}" if r.scores else "  -  "
        mark = "✅" if r.top_score < DEFAULT_NO_CALL_THRESHOLD else "❌"
        top1 = r.ranked[0] if r.ranked else "-"
        print(f"  {mark} {score_str:>6}  {top1:<38}  {r.query}")


def _print_compare(ks: list[int], threshold: float) -> None:
    """여러 k 에 대한 비교표 출력."""
    print(f"\n{'=' * 72}")
    print(f"  Tool Search 비교 평가  (threshold={threshold})")
    print(f"{'=' * 72}")

    results_cache: dict[int, list[EvalResult]] = {}
    metrics_list: list[dict] = []

    for k_val in ks:
        results = _run_search(k_val)
        results_cache[k_val] = results
        metrics_list.append(_compute_metrics(results, k_val, threshold))

    m0 = metrics_list[0]
    print(f"\n  쿼리 수: tool-call {m0['tc_total']}개 + no-call {m0['nc_total']}개 = 총 {m0['total']}개\n")

    # 비교표
    k_header = "".join(f"{'k='+str(m['k']):>10}" for m in metrics_list)
    print(f"  {'지표':<20}{k_header}")
    print(f"  {'─'*20}{'─'*10*len(metrics_list)}")

    def _row(label: str, key: str, fmt: str = ".1%") -> str:
        vals = "".join(f"{format(m[key], fmt):>10}" for m in metrics_list)
        return f"  {label:<20}{vals}"

    print(_row("Tool Acc (Hit@1)", "tool_acc"))
    print(_row("Recall@k", "recall_at_k"))
    print(_row("MRR", "mrr", ".4f"))
    print(_row("No-Call Acc", "no_call_acc"))
    print(f"  {'─'*20}{'─'*10*len(metrics_list)}")
    print(_row("Overall Acc", "overall_acc"))

    print()

    # 미탐/오판 요약
    for k_val, results in results_cache.items():
        tool_misses = [r for r in results if not r.is_no_call and r.hit_rank is None]
        nc_fails = [r for r in results if r.is_no_call and r.top_score >= threshold]
        if tool_misses or nc_fails:
            print(f"  k={k_val}: 미탐 {len(tool_misses)}건, no-call 오판 {len(nc_fails)}건")
            for r in tool_misses:
                print(f"    ❌ [{r.expected}] '{r.query}' → top-1: {r.ranked[0] if r.ranked else '-'}")
            for r in nc_fails:
                print(f"    ⚠️  '{r.query}' → {r.ranked[0]}({r.top_score:.3f})")

    # 점수 분포
    last_results = results_cache[ks[-1]]
    tc = [r for r in last_results if not r.is_no_call]
    nc = [r for r in last_results if r.is_no_call]
    if tc and nc:
        tc_scores = [r.scores[0] for r in tc if r.scores]
        nc_scores = [r.top_score for r in nc]
        print(f"\n  📊 점수 분포 (k={ks[-1]} 기준):")
        print(f"     Tool-Call top-1 : min={min(tc_scores):.3f}  avg={sum(tc_scores)/len(tc_scores):.3f}  max={max(tc_scores):.3f}")
        print(f"     No-Call   top-1 : min={min(nc_scores):.3f}  avg={sum(nc_scores)/len(nc_scores):.3f}  max={max(nc_scores):.3f}")
        gap = min(tc_scores) - max(nc_scores)
        print(f"     분리 마진 (tool min - no-call max) = {gap:+.3f}")

    print()


# ──────────────────────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────────────────────
def run_eval(k: int = 10, verbose: bool = False, threshold: float = DEFAULT_NO_CALL_THRESHOLD) -> None:
    results = _run_search(k)
    _print_single(results, k, threshold, verbose)


def _run_card_validation() -> bool:
    """ToolCard 정합성 검증. 문제가 있으면 경고 출력 후 False 반환."""
    from app.tool_search.tool_cards import (
        validate_confusion_pairs,
        validate_duplicate_when_to_use,
    )

    print("=" * 60)
    print("  ToolCard 정합성 검증")
    print("=" * 60)

    warnings = validate_confusion_pairs() + validate_duplicate_when_to_use()
    if warnings:
        for w in warnings:
            print(f"  ⚠️  {w}")
        print(f"\n  총 {len(warnings)}건 경고\n")
        return False

    print("  ✅ 혼동 쌍 cross-reference 정상")
    print("  ✅ when_to_use 중복 발화 없음\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool Search 평가 (Tool Acc + No-Call Acc)")
    parser.add_argument("--k", type=int, default=10, help="top-k (기본값: 10)")
    parser.add_argument("--verbose", action="store_true", help="전체 결과 출력")
    parser.add_argument("--threshold", type=float, default=DEFAULT_NO_CALL_THRESHOLD,
                        help=f"No-Call 판정 임계값 (기본값: {DEFAULT_NO_CALL_THRESHOLD})")
    parser.add_argument("--compare", action="store_true", help="여러 k 에 대한 비교표 출력")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 7, 10],
                        help="비교할 k 값들 (기본값: 1 3 5 7 10)")
    args = parser.parse_args()

    _run_card_validation()

    if args.compare:
        _print_compare(args.ks, args.threshold)
    else:
        run_eval(k=args.k, verbose=args.verbose, threshold=args.threshold)
