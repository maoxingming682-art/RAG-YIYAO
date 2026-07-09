# -*- coding: utf-8 -*-
"""
Evaluate multi-turn RAG dialog behavior.

Default mode is deterministic and cheap:
  - rewrite follow-up question
  - expand retrieval queries
  - run retrieval
  - check evidence coverage

Optional live API mode calls /api/ask for end-to-end smoke testing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BASE_DIR
import app as rag_app


DEFAULT_CASES = Path(BASE_DIR) / "data" / "rag_dialog_eval_cases.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG multi-turn dialog quality.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--live-api", action="store_true", help="Call /api/ask instead of internal retrieval checks.")
    parser.add_argument("--api-url", default="http://localhost:5006/api/ask")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def load_cases(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def all_contained(text: str, needles: list[str]) -> tuple[bool, str]:
    for needle in needles or []:
        if needle not in text:
            return False, needle
    return True, ""


def any_contained(texts: list[str], needles: list[str]) -> tuple[bool, str]:
    for needle in needles or []:
        if not any(needle in text for text in texts):
            return False, needle
    return True, ""


def live_api_call(api_url: str, case: dict) -> dict:
    payload = json.dumps(
        {"question": case["question"], "history": case.get("history", [])},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_internal(case: dict, top_k: int) -> tuple[bool, list[str], dict]:
    failures: list[str] = []
    history = case.get("history", [])
    context = rag_app._dialog_context(history)
    rewritten, is_rewritten = rag_app.rewrite_query(case["question"], history)
    queries = rag_app.build_retrieval_queries(rewritten, context)
    retrieved = rag_app.retrieve_with_context(rewritten, context, top_k=top_k)
    evidence_ok, evidence_reason = rag_app.evidence_supports_question(rewritten, retrieved, context)

    ok, missing = all_contained(rewritten, case.get("expected_rewrite_contains", []))
    if not ok:
        failures.append(f"rewrite missing: {missing}; got={rewritten}")

    ok, missing = any_contained(queries, case.get("expected_query_contains", []))
    if not ok:
        failures.append(f"query missing: {missing}; got={queries}")

    sources = [r.get("source", "") for r in retrieved]
    ok, missing = any_contained(sources, case.get("expected_source_contains", []))
    if not ok:
        failures.append(f"source missing: {missing}; got={sources[:3]}")

    expected_evidence = case.get("expected_evidence_ok")
    if expected_evidence is not None and evidence_ok != expected_evidence:
        failures.append(f"evidence expected {expected_evidence}, got {evidence_ok}: {evidence_reason}")

    min_similarity = case.get("min_similarity")
    max_similarity = max((r.get("similarity", 0.0) for r in retrieved), default=0.0)
    if min_similarity is not None and max_similarity < float(min_similarity):
        failures.append(f"max similarity {max_similarity:.4f} < {min_similarity}")

    details = {
        "rewritten": rewritten,
        "is_rewritten": is_rewritten,
        "queries": queries,
        "max_similarity": round(float(max_similarity), 4),
        "sources": sources[:3],
        "evidence_ok": evidence_ok,
        "evidence_reason": evidence_reason,
    }
    return len(failures) == 0, failures, details


def run_live(case: dict, api_url: str) -> tuple[bool, list[str], dict]:
    failures: list[str] = []
    data = live_api_call(api_url, case)
    if not data.get("success"):
        return False, [f"api failed: {data}"], data

    result = data.get("result", {})
    answer = result.get("answer", "")
    steps = result.get("steps", [])
    retrieved = result.get("retrieved", [])
    sources = [r.get("source", "") for r in retrieved]

    ok, missing = any_contained(sources, case.get("expected_source_contains", []))
    if not ok:
        failures.append(f"source missing: {missing}; got={sources[:3]}")

    for needle in case.get("expected_answer_contains", []):
        if needle not in answer:
            failures.append(f"answer missing: {needle}")

    details = {
        "steps": steps,
        "sources": sources[:3],
        "answer_preview": answer[:160].replace("\n", " "),
    }
    return len(failures) == 0, failures, details


def main() -> int:
    args = parse_args()
    cases = load_cases(args.cases)

    passed = 0
    print("=" * 72)
    print("RAG multi-turn dialog evaluation")
    print(f"cases: {args.cases}")
    print(f"mode: {'live-api' if args.live_api else 'internal'}")
    print("=" * 72)

    for case in cases:
        if args.live_api:
            ok, failures, details = run_live(case, args.api_url)
        else:
            ok, failures, details = run_internal(case, args.top_k)

        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] {case['id']} - {case.get('description', '')}")
        if not ok:
            for failure in failures:
                print(f"  - {failure}")
        print(f"  details: {json.dumps(details, ensure_ascii=False)}")
        passed += 1 if ok else 0

    print("\n" + "=" * 72)
    print(f"passed: {passed}/{len(cases)}")
    print("=" * 72)
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
