"""
benchmark_longmemeval.py — LexiContext vs LongMemEval

Runs LexiContext against the LongMemEval benchmark and measures
token efficiency (G4) and retrieval coverage.

Usage:
    # Quick test — first 5 questions, local Distiller
    python3 benchmark_longmemeval.py --limit 5

    # 20 questions with Kimi Distiller
    python3 benchmark_longmemeval.py --api-key sk-xxx --limit 20

    # Full 500 questions (takes ~30 min with API)
    python3 benchmark_longmemeval.py --api-key sk-xxx --limit 500

Outputs:
    benchmark_results.json   per-question raw data
    benchmark_summary.json   aggregated G4 metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(data_dir: str, limit: int | None = None) -> list[dict]:
    data_path = Path(data_dir)
    for name in ["longmemeval_s_cleaned.json", "longmemeval_m_cleaned.json"]:
        path = data_path / name
        if path.exists():
            print(f"Loading: {path.name}")
            with open(path) as f:
                data = json.load(f)
            items = list(data.values()) if isinstance(data, dict) else data
            items = items[:limit] if limit else items
            print(f"Loaded {len(items)} questions")
            return items
    raise FileNotFoundError(
        f"No dataset found in {data_dir}/\n"
        "Expected: longmemeval_s_cleaned.json\n"
        "Run the download commands from the README."
    )


def load_oracle(data_dir: str) -> dict:
    path = Path(data_dir) / "longmemeval_oracle.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return {item["question_id"]: item.get("answer", "") for item in data}
    return data


def item_to_sessions(item: dict) -> list[dict]:
    for field in ["sessions", "history", "chat_history", "conversations"]:
        if field in item:
            return item[field]
    return []


def session_to_text(session: dict) -> str:
    lines = []
    sid = session.get("session_id", session.get("id", ""))
    if sid:
        lines.append(f"[Session {sid}]")
    turns = session.get("turns", session.get("conversation", []))
    for turn in turns:
        role = turn.get("role", turn.get("speaker", "user"))
        content = turn.get("content", turn.get("text", ""))
        if content:
            lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


def count_tokens(text: str) -> int:
    import re
    return max(1, len(re.findall(r"\S+", text)))


def full_context_token_count(item: dict) -> int:
    """
    Tokens in the complete conversation history as it would be sent
    to an LLM in a naive full-context approach.
    Includes session headers and turn formatting overhead.
    """
    sessions = item_to_sessions(item)
    sep = "\n\n"
    all_text = sep.join(session_to_text(s) for s in sessions)
    return count_tokens(all_text)


def avg_session_length(item: dict) -> int:
    """Average tokens per session — used to flag short-session warnings."""
    sessions = item_to_sessions(item)
    if not sessions:
        return 0
    lengths = [count_tokens(session_to_text(s)) for s in sessions]
    return sum(lengths) // max(len(lengths), 1)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class LongMemEvalBenchmark:

    def __init__(self, api_key: str = "", token_budget: int = 2000, delay: float = 1.5):
        self.api_key = api_key
        self.token_budget = token_budget
        self.delay = delay

    def run(self, items: list[dict], output_prefix: str = "benchmark") -> dict:
        from manager import ContextManager

        results = []
        n = len(items)
        print(f"\nBenchmark: {n} questions | budget={self.token_budget} | "
              f"distiller={'kimi' if self.api_key else 'local'}")
        print("-" * 70)

        for i, item in enumerate(items):
            qid = item.get("question_id", item.get("id", f"q{i:04d}"))
            question = item.get("question", item.get("query", ""))
            sessions = item_to_sessions(item)

            if not question or not sessions:
                continue

            # Ingest all sessions
            cm = ContextManager(api_key=self.api_key, default_topic="longmemeval")
            t0 = time.time()
            ingested = 0
            for sess in sessions:
                text = session_to_text(sess)
                if text.strip():
                    cm.learn(text, session_id=sess.get("session_id", f"s{ingested}"))
                    ingested += 1
                    if self.api_key:
                        time.sleep(self.delay)
            learn_sec = round(time.time() - t0, 2)

            # Query
            t1 = time.time()
            ctx = cm.query(question, token_budget=self.token_budget)
            query_sec = round(time.time() - t1, 3)

            # Baseline
            full_tokens = full_context_token_count(item)
            reduction = round((1 - ctx.tokens_used / max(full_tokens, 1)) * 100, 1)

            avg_sess_len = avg_session_length(item)
            short_session = avg_sess_len < 100  # warn if sessions are very short

            result = {
                "question_id": qid,
                "question": question[:120],
                "sessions": ingested,
                "lexi_tokens": ctx.tokens_used,
                "full_tokens": full_tokens,
                "reduction_pct": reduction,
                "entries": len(ctx.entries_used),
                "tier": ctx.budget_tier,
                "expansion_ratio": ctx.expansion_ratio,
                "learn_sec": learn_sec,
                "query_sec": query_sec,
                "avg_session_tokens": avg_sess_len,
                "short_session_flag": short_session,
                "dict_active": cm.stats()["dictionary"]["active"],
                "dict_deprecated": cm.stats()["dictionary"]["deprecated"],
            }
            results.append(result)

            warn = " ⚠ short" if avg_sess_len < 100 else ""
            print(f"[{i+1:3d}/{n}] {qid[:18]:18s} | "
                  f"sess={ingested:2d} | "
                  f"lexi={ctx.tokens_used:4d} | "
                  f"full={full_tokens:5d} | "
                  f"saved={reduction:5.1f}% | "
                  f"entries={len(ctx.entries_used):2d}{warn}")

        summary = self._summarise(results)
        self._save(results, summary, output_prefix)
        self._print_summary(summary)
        return summary

    def _summarise(self, results: list[dict]) -> dict:
        if not results:
            return {}

        def avg(key):
            vals = [r[key] for r in results]
            return round(sum(vals) / max(len(vals), 1), 1)

        def pct_above(key, threshold):
            vals = [r[key] for r in results]
            return round(100 * sum(1 for v in vals if v >= threshold) / max(len(vals), 1), 1)

        tiers = {}
        for r in results:
            tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1

        return {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "total": len(results),
            "distiller": "kimi_api" if self.api_key else "local_fallback",
            "token_budget": self.token_budget,

            # G4 — token efficiency (paper data)
            "avg_lexi_tokens": avg("lexi_tokens"),
            "avg_full_tokens": avg("full_tokens"),
            "avg_reduction_pct": avg("reduction_pct"),
            "pct_saving_over_50": pct_above("reduction_pct", 50),
            "pct_saving_over_80": pct_above("reduction_pct", 80),
            "pct_zero_retrieval": round(
                100 * sum(1 for r in results if r["entries"] == 0) / max(len(results), 1), 1
            ),

            # Retrieval
            "avg_entries_per_query": avg("entries"),
            "tier_distribution": tiers,

            # Timing
            "avg_learn_sec": avg("learn_sec"),
            "avg_query_sec": avg("query_sec"),
            "avg_session_tokens": avg("avg_session_tokens"),
            "pct_short_sessions": round(
                100 * sum(1 for r in results if r.get("short_session_flag")) / max(len(results), 1), 1
            ),
            "note": (
                "Short sessions detected — token savings increase significantly "
                "with longer sessions (real LongMemEval sessions are 50-500+ turns)."
                if any(r.get("short_session_flag") for r in results) else ""
            ),
        }

    def _save(self, results, summary, prefix):
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        with open(f"{prefix}_results.json", "w") as f:
            json.dump(results, f, indent=2)
        with open(f"{prefix}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {prefix}_results.json")
        print(f"Saved: {prefix}_summary.json")

    def _print_summary(self, s: dict):
        print("\n" + "=" * 60)
        print("  LexiContext × LongMemEval — Results")
        print("=" * 60)
        print(f"  Questions:          {s['total']}")
        print(f"  Distiller:          {s['distiller']}")
        print(f"  Token budget:       {s['token_budget']}")
        print()
        print("  G4 — Token Efficiency")
        print(f"  Avg LexiContext:     {s['avg_lexi_tokens']} tokens/query")
        print(f"  Avg Full-context:    {s['avg_full_tokens']} tokens/query")
        print(f"  Avg reduction:       {s['avg_reduction_pct']}%")
        print(f"  >50% saving:         {s['pct_saving_over_50']}% of questions")
        print(f"  >80% saving:         {s['pct_saving_over_80']}% of questions")
        print()
        print("  Retrieval")
        print(f"  Avg entries/query:   {s['avg_entries_per_query']}")
        print(f"  Zero retrieval:      {s['pct_zero_retrieval']}% of questions")
        print(f"  Tier distribution:   {s['tier_distribution']}")
        print()
        print("  Timing")
        print(f"  Avg ingest time:     {s['avg_learn_sec']}s/question")
        print(f"  Avg query time:      {s['avg_query_sec']}s/question")
        print("=" * 60)
        print()
        print("  Paper claim (G4):")
        print(f"  LexiContext uses {s['avg_reduction_pct']}% fewer tokens than")
        print(f"  full-context replay on LongMemEval-S ({s['total']} questions).")
        if s.get("note"):
            print()
            print(f"  Note: {s['note']}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--limit", type=int, default=5,
                        help="Questions to run (default 5 for quick test)")
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Seconds between API calls (avoid 429)")
    parser.add_argument("--output", default="benchmark")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  LexiContext × LongMemEval Benchmark")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)

    items = load_dataset(args.data_dir, limit=args.limit)

    bench = LongMemEvalBenchmark(
        api_key=args.api_key,
        token_budget=args.budget,
        delay=args.delay,
    )
    bench.run(items, output_prefix=args.output)


if __name__ == "__main__":
    main()
