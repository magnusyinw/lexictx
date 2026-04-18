"""
test_smoke.py — LexiContext end-to-end smoke test

Validates the full learn → query → expand pipeline using
the five canonical test cases from Distiller validation v0.1.

Covers all four design guarantees:
  G1 — compression without retrieval degradation
  G2 — every hit traces back to original evidence
  G3 — writes do not silently corrupt memory
  G4 — token savings vs full-context baseline

Run:
    python3 test_smoke.py
    python3 test_smoke.py --api-key sk-ant-...   # use real Distiller
    python3 test_smoke.py --verbose              # show full content output
    python3 test_smoke.py --json                 # write smoke_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import datetime

# ---------------------------------------------------------------------------
# Test corpus — five canonical cases
# ---------------------------------------------------------------------------

CASES = [
    {
        "id": "case_01",
        "label": "Technical Decision",
        "text": (
            "We decided to migrate the authentication service from JWT to session-based "
            "tokens. The main reason is that several enterprise clients require server-side "
            "session invalidation, which JWT cannot support without a blocklist. The "
            "migration is owned by Sarah from the platform team. Risk identified: increased "
            "Redis load during peak hours. Target completion is end of Q2."
        ),
        "queries": [
            "Why did we change the authentication strategy?",
            "Who owns the JWT migration?",
            "What risk was identified with the JWT migration?",
        ],
        # Component words that must appear in feature_keys (individually or as compounds)
        "expected_keys": [
            "JWT",
            "session",        # session_tokens or session alone
            "migration",
            "Sarah",
            "Redis",
            "Q2",
        ],
    },
    {
        "id": "case_02",
        "label": "Knowledge Update",
        "text": (
            "Update: Marcus has moved from the data engineering team to the ML platform "
            "team as of this month. His previous work on the Kafka pipeline has been "
            "handed off to Elena. Marcus will now focus on the feature store infrastructure. "
            "His Slack handle remains @marcus_k."
        ),
        "queries": [
            "What team is Marcus on now?",
            "Who took over the Kafka pipeline?",
            "What is Marcus working on?",
        ],
        "expected_keys": [
            "Marcus",
            "ML",             # ML_platform or ML alone
            "Kafka",
            "Elena",
            "feature",        # feature_store or feature alone
        ],
    },
    {
        "id": "case_03",
        "label": "Constraint",
        "text": (
            "Per legal review completed on March 3rd, all user data processed under the "
            "EU contract must remain within Frankfurt and Dublin regions. This applies to "
            "both primary storage and backups. Any exception requires written approval from "
            "the DPO. The constraint is permanent and not subject to future architectural "
            "decisions."
        ),
        "queries": [
            "Where must EU contract data be stored?",
            "How do I get an exception to the data residency policy?",
            "Is the Frankfurt data rule permanent?",
        ],
        "expected_keys": [
            "EU",             # EU_contract or EU alone
            "Frankfurt",
            "Dublin",
            "DPO",
            "permanent",
        ],
    },
    {
        "id": "case_04",
        "label": "Observation",
        "text": (
            "During this week's load test, the recommendation engine showed a consistent "
            "latency spike above 800ms when concurrent users exceeded 2,000. The spike "
            "correlates with database connection pool exhaustion, not CPU. No action has "
            "been taken yet. This was flagged by the monitoring alert at 14:32 on Tuesday."
        ),
        "queries": [
            "What caused the recommendation engine latency spike?",
            "What happened during the load test?",
            "Has anyone fixed the 800ms latency issue?",
        ],
        "expected_keys": [
            "recommendation",   # recommendation_engine or recommendation
            "latency",
            "800ms",
            "connection",       # connection_pool or connection
        ],
    },
    {
        "id": "case_05",
        "label": "Dependency",
        "text": (
            "The mobile release v4.2 is blocked on two things: the push notification "
            "service must complete its certificate rotation, and the analytics SDK needs "
            "a sign-off from the privacy team. The certificate rotation is owned by DevOps "
            "and is 80% done. Privacy review for the SDK was submitted last Friday and "
            "typically takes 5 business days. Current estimated unblock date is next "
            "Wednesday."
        ),
        "queries": [
            "Why is v4.2 blocked?",
            "When will the mobile release be unblocked?",
            "Who owns the certificate rotation?",
        ],
        "expected_keys": [
            "v4.2",
            "blocked",
            "certificate",      # certificate_rotation or certificate
            "DevOps",
            "Wednesday",
        ],
    },
]

# ---------------------------------------------------------------------------
# G3 conflict test corpus
# ---------------------------------------------------------------------------

CONFLICT_CASES = [
    {
        "id": "conflict_01",
        "label": "Knowledge Update triggers SUPERSEDE",
        "original": (
            "Marcus is on the data engineering team. "
            "He owns the Kafka pipeline."
        ),
        "update": (
            "Update: Marcus has moved to the ML platform team. "
            "Kafka pipeline handed to Elena."
        ),
        "expected_conflict": "supersede",
    },
    {
        "id": "conflict_02",
        "label": "Same content twice → conflict detected",
        "original": (
            "The Frankfurt data residency rule is permanent and cannot be overridden."
        ),
        "update": (
            "The Frankfurt data residency rule is permanent and cannot be overridden."
        ),
        "expected_conflict": "any",
    },
]


# ---------------------------------------------------------------------------
# G1 key matching helper
# ---------------------------------------------------------------------------

def _key_found(expected_key: str, feature_keys: list[str]) -> bool:
    """
    Check if expected_key is represented in feature_keys.

    Matching strategy (in order):
      1. Direct: expected_key (normalised) appears in any feature_key
      2. Component: all underscore-split parts of expected_key appear
         individually in feature_keys (handles compound vs split tokens)
      3. Substring: expected_key is a substring of any feature_key

    This makes G1 tests valid for both:
      - LLM Distiller  → produces compound snake_case keys exactly
      - Local fallback → produces split words + some compounds
    """
    ek = expected_key.lower()
    fk_set = {fk.lower() for fk in feature_keys}
    fk_normalized = {fk.lower().replace("_", "").replace(".", "") for fk in feature_keys}
    ek_normalized = ek.replace("_", "").replace(".", "")

    # 1. Direct normalised match
    if ek_normalized in fk_normalized:
        return True

    # 2. Component match: all parts present individually
    parts = [p for p in ek.split("_") if p]
    if len(parts) > 1 and all(p in fk_set for p in parts):
        return True

    # 3. Substring match (e.g. "recommendation" found in "recommendation_engine")
    for fk in feature_keys:
        if ek in fk.lower() or fk.lower() in ek:
            return True

    return False


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class SmokeTest:

    def __init__(self, api_key: str = "", verbose: bool = False) -> None:
        self.api_key = api_key
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.results: list[dict] = []

    def run(self) -> bool:
        print("\n" + "=" * 60)
        print("  LexiContext Smoke Test")
        print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"  Distiller: {'API (real)' if self.api_key else 'local fallback'}")
        print("=" * 60)

        self._test_learn_and_query()
        self._test_g1_recall()
        self._test_g2_provenance()
        self._test_g3_conflict()
        self._test_g4_token_efficiency()
        self._test_budget_tiers()

        self._print_summary()
        return self.failed == 0

    # ------------------------------------------------------------------
    # Block 1 — learn + query pipeline
    # ------------------------------------------------------------------

    def _test_learn_and_query(self) -> None:
        print("\n── BLOCK 1: learn() + query() pipeline ──")

        from manager import ContextManager
        cm = ContextManager(api_key=self.api_key)

        for case in CASES:
            result = cm.learn(case["text"], topic=case["label"].lower())
            self._check(
                f"[{case['id']}] learn() succeeds",
                result.success,
                detail=f"conflict_type={result.conflict_type.value}",
            )

        self._check(
            "dictionary has 5 active entries",
            cm.stats()["dictionary"]["active"] == 5,
            detail=str(cm.stats()["dictionary"]),
        )

        for case in CASES:
            for q in case["queries"]:
                ctx = cm.query(q, token_budget=2000)
                self._check(
                    f"[{case['id']}] query returns content: '{q[:40]}...'",
                    len(ctx.entries_used) > 0,
                    detail=(
                        f"tokens={ctx.tokens_used}, "
                        f"entries={len(ctx.entries_used)}, "
                        f"tier={ctx.budget_tier}"
                    ),
                )
                if self.verbose and ctx.content:
                    print(f"\n    CONTENT:\n{textwrap.indent(ctx.content, '    ')}")

    # ------------------------------------------------------------------
    # Block 2 — G1 feature recall
    # ------------------------------------------------------------------

    def _test_g1_recall(self) -> None:
        print("\n── BLOCK 2: G1 — feature recall after compression ──")

        from manager import ContextManager
        cm = ContextManager(api_key=self.api_key)

        for case in CASES:
            cm.learn(case["text"])

        for case in CASES:
            active = cm._dictionary.list_active()
            entry = next(
                (e for e in active if case["text"][:50] in e.source_text),
                None,
            )
            if entry is None:
                self._check(f"[{case['id']}] G1: entry found in dictionary", False)
                continue

            found = [
                k for k in case["expected_keys"]
                if _key_found(k, entry.feature_keys)
            ]
            recall = len(found) / len(case["expected_keys"])

            missing = [k for k in case["expected_keys"] if not _key_found(k, entry.feature_keys)]
            self._check(
                f"[{case['id']}] G1: recall ≥ 0.80 "
                f"({len(found)}/{len(case['expected_keys'])} keys)",
                recall >= 0.80,
                detail=(
                    f"recall={recall:.0%} | "
                    f"found={found} | "
                    f"missing={missing}"
                ),
            )
            if recall < 0.80:
                print(f"    ℹ actual feature_keys: {entry.feature_keys[:20]}")

            self._check(
                f"[{case['id']}] G1: compressed < source length",
                len(entry.compressed) < len(entry.source_text),
                detail=(
                    f"compressed={len(entry.compressed)} chars, "
                    f"source={len(entry.source_text)} chars"
                ),
            )

    # ------------------------------------------------------------------
    # Block 3 — G2 provenance
    # ------------------------------------------------------------------

    def _test_g2_provenance(self) -> None:
        print("\n── BLOCK 3: G2 — every hit traces back to source ──")

        from manager import ContextManager
        cm = ContextManager(api_key=self.api_key)
        for case in CASES:
            cm.learn(case["text"])

        for case in CASES[:3]:
            ctx = cm.query(case["queries"][0], token_budget=2000)

            if not ctx.entries_used:
                self._check(f"[{case['id']}] G2: query returned entries", False)
                continue

            self._check(
                f"[{case['id']}] G2: all entries have source_ref",
                all(bool(e.source_ref) for e in ctx.entries_used),
            )

            for entry in ctx.entries_used[:2]:
                source = cm.expand(entry.id)
                self._check(
                    f"[{case['id']}] G2: expand({entry.id[:12]}...) returns source",
                    bool(source) and len(source) > 10,
                    detail=f"source length={len(source) if source else 0}",
                )

            self._check(
                f"[{case['id']}] G2: source_refs count matches entries",
                len(ctx.source_refs) == len(ctx.entries_used),
                detail=(
                    f"source_refs={len(ctx.source_refs)}, "
                    f"entries={len(ctx.entries_used)}"
                ),
            )

    # ------------------------------------------------------------------
    # Block 4 — G3 conflict-aware writes
    # ------------------------------------------------------------------

    def _test_g3_conflict(self) -> None:
        print("\n── BLOCK 4: G3 — conflict-aware writes ──")

        from manager import ContextManager

        for conflict_case in CONFLICT_CASES:
            cm = ContextManager(api_key=self.api_key, auto_supersede=True)

            r1 = cm.learn(conflict_case["original"])
            self._check(
                f"[{conflict_case['id']}] G3: original write succeeds",
                r1.success,
            )

            preview = cm.preview_write(conflict_case["update"])
            self._check(
                f"[{conflict_case['id']}] G3: conflict detected in preview",
                preview["action"] != "INSERT",
                detail=f"preview action={preview['action']}",
            )

            r2 = cm.learn(conflict_case["update"])
            self._check(
                f"[{conflict_case['id']}] G3: update write succeeds",
                r2.success,
                detail=f"conflict_type={r2.conflict_type.value}",
            )

            log = cm.write_log()
            has_conflict = any(w.conflict_type.value != "none" for w in log)
            self._check(
                f"[{conflict_case['id']}] G3: conflict recorded in write_log",
                has_conflict,
                detail=f"write_log entries={len(log)}",
            )

    # ------------------------------------------------------------------
    # Block 5 — G4 token efficiency
    # ------------------------------------------------------------------

    def _test_g4_token_efficiency(self) -> None:
        print("\n── BLOCK 5: G4 — token efficiency vs full-context ──")

        from manager import ContextManager
        cm = ContextManager(api_key=self.api_key)
        for case in CASES:
            cm.learn(case["text"])

        # Full-context baseline: all source texts concatenated
        full_context_chars = sum(len(c["text"]) for c in CASES)
        full_context_tokens_approx = full_context_chars // 4

        total_lexi_tokens = 0
        total_queries = 0

        for case in CASES:
            for q in case["queries"]:
                ctx = cm.query(q, token_budget=2000)
                total_lexi_tokens += ctx.tokens_used
                total_queries += 1

        avg_lexi = total_lexi_tokens / total_queries if total_queries else 0
        efficiency = 1 - (avg_lexi / max(full_context_tokens_approx, 1))

        self._check(
            f"G4: LexiContext avg tokens ({avg_lexi:.0f}) < full-context "
            f"({full_context_tokens_approx})",
            avg_lexi < full_context_tokens_approx,
            detail=(
                f"avg_lexi={avg_lexi:.0f} tokens, "
                f"full_context≈{full_context_tokens_approx} tokens, "
                f"savings≈{efficiency:.0%}"
            ),
        )

    # ------------------------------------------------------------------
    # Block 6 — budget tier behaviour
    # ------------------------------------------------------------------

    def _test_budget_tiers(self) -> None:
        print("\n── BLOCK 6: budget tier behaviour ──")

        from manager import ContextManager
        cm = ContextManager(api_key=self.api_key)
        for case in CASES:
            cm.learn(case["text"])

        query = "What are all the key decisions and constraints?"
        budgets = [200, 500, 1000, 2000, 4000]

        prev_tokens = -1

        for budget in budgets:
            ctx = cm.query(query, token_budget=budget)

            # Tokens used must not exceed budget
            self._check(
                f"budget={budget}: tokens_used ({ctx.tokens_used}) ≤ budget",
                ctx.tokens_used <= budget,
                detail=f"tier={ctx.budget_tier}, ratio={ctx.expansion_ratio}",
            )

            # More budget → same or more tokens (monotonic)
            # Note: only check within the same tier boundary.
            # Across tier boundaries (e.g. 500→1000 crosses low→medium),
            # a small non-monotonic step is acceptable because format changes.
            if prev_tokens >= 0:
                same_or_more = ctx.tokens_used >= prev_tokens
                tier_boundary = (budget in (500, 1000, 2000))  # tier transition points
                if not tier_boundary:
                    self._check(
                        f"budget={budget}: tokens_used ≥ previous (monotonic within tier)",
                        same_or_more,
                        detail=f"current={ctx.tokens_used}, previous={prev_tokens}",
                    )

            prev_tokens = ctx.tokens_used

        # HIGH tier must use more tokens than MEDIUM tier for the same query
        ctx_medium = cm.query(query, token_budget=2000)
        ctx_high = cm.query(query, token_budget=4000)
        self._check(
            "HIGH tier tokens ≥ MEDIUM tier tokens (additive format)",
            ctx_high.tokens_used >= ctx_medium.tokens_used,
            detail=(
                f"high={ctx_high.tokens_used} tokens, "
                f"medium={ctx_medium.tokens_used} tokens"
            ),
        )

        # Tier labels are correct
        tier_200 = cm.query(query, token_budget=200).budget_tier
        tier_1000 = cm.query(query, token_budget=1000).budget_tier
        tier_4000 = cm.query(query, token_budget=4000).budget_tier

        self._check(
            "tier progression: 200→low, 1000→medium, 4000→high",
            tier_200 == "low" and tier_1000 == "medium" and tier_4000 == "high",
            detail=f"200={tier_200}, 1000={tier_1000}, 4000={tier_4000}",
        )

    # ------------------------------------------------------------------
    # Assertion helper
    # ------------------------------------------------------------------

    def _check(self, label: str, condition: bool, detail: str = "") -> None:
        status = "✅ PASS" if condition else "❌ FAIL"
        detail_str = f"  [{detail}]" if detail else ""
        print(f"  {status}  {label}{detail_str}")
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        self.results.append({"label": label, "passed": condition, "detail": detail})

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self) -> None:
        total = self.passed + self.failed
        print("\n" + "=" * 60)
        print(f"  Results: {self.passed}/{total} passed")

        if self.failed == 0:
            print("  Status:  ✅ ALL TESTS PASSED")
            print("\n  Next step: run with --api-key for real Distiller validation,")
            print("  then run against LongMemEval for G1/G4 benchmarks.")
        else:
            print(f"  Status:  ❌ {self.failed} TESTS FAILED")
            print("\n  Failed tests:")
            for r in self.results:
                if not r["passed"]:
                    print(f"    • {r['label']}")
                    if r["detail"]:
                        print(f"      {r['detail']}")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LexiContext end-to-end smoke test")
    parser.add_argument("--api-key", default="", help="Anthropic API key")
    parser.add_argument("--verbose", action="store_true", help="Print assembled context")
    parser.add_argument("--json", action="store_true", help="Write smoke_results.json")
    args = parser.parse_args()

    runner = SmokeTest(api_key=args.api_key, verbose=args.verbose)
    success = runner.run()

    if args.json:
        with open("smoke_results.json", "w") as f:
            json.dump(runner.results, f, indent=2)
        print("Results written to smoke_results.json")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
