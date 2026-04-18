"""
assembler.py — LexiContext Budget-Aware Context Assembler
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from models import AssembledContext, ContextEntry, StructuralRole
from matcher import ScoredEntry


def _default_token_counter(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))


@dataclass
class BudgetTiers:
    low_ceiling: int = 500
    medium_ceiling: int = 2000


_ROLE_PRIORITY: dict[str, int] = {
    StructuralRole.CONSTRAINT.value:        5,
    StructuralRole.DECISION.value:          4,
    StructuralRole.DEPENDENCY.value:        3,
    StructuralRole.KNOWLEDGE_UPDATE.value:  2,
    StructuralRole.OBSERVATION.value:       1,
    StructuralRole.OUTCOME.value:           1,
    StructuralRole.QUESTION.value:          0,
}


class BudgetAssembler:

    def __init__(
        self,
        tiers: Optional[BudgetTiers] = None,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        self._tiers = tiers or BudgetTiers()
        self._count = token_counter or _default_token_counter

    def assemble(
        self,
        query: str,
        scored_entries: list[ScoredEntry],
        token_budget: int,
    ) -> AssembledContext:
        if not scored_entries:
            return self._empty(query, token_budget)
        tier = self._select_tier(token_budget)
        ranked = self._rank(scored_entries)
        if tier == "low":
            return self._assemble_low(query, ranked, token_budget)
        elif tier == "medium":
            return self._assemble_medium(query, ranked, token_budget)
        else:
            return self._assemble_high(query, ranked, token_budget)

    def _select_tier(self, budget: int) -> str:
        if budget <= self._tiers.low_ceiling:
            return "low"
        elif budget <= self._tiers.medium_ceiling:
            return "medium"
        return "high"

    def _rank(self, scored_entries: list[ScoredEntry]) -> list[ScoredEntry]:
        return sorted(
            scored_entries,
            key=lambda se: (
                _ROLE_PRIORITY.get(se.entry.structural_role.value, 0),
                se.score,
                se.entry.metadata.freshness,
            ),
            reverse=True,
        )

    def _assemble_low(self, query, ranked, budget):
        blocks, used_entries, source_refs, tokens_used, scores = [], [], [], 0, {}
        for se in ranked:
            block = f"[{se.entry.structural_role.value.upper()}] {se.entry.compressed}"
            cost = self._count(block)
            if tokens_used + cost > budget:
                break
            blocks.append(block)
            used_entries.append(se.entry)
            source_refs.append(se.entry.source_ref)
            tokens_used += cost
            scores[se.entry.id] = se.score
        return AssembledContext(
            query=query, content="\n\n".join(blocks), entries_used=used_entries,
            source_refs=source_refs, tokens_used=tokens_used, token_budget=budget,
            budget_tier="low", expansion_ratio=round(tokens_used / max(budget, 1), 3),
            scores=scores,
        )

    def _assemble_medium(self, query, ranked, budget):
        tokens_used, scores = 0, {}
        compressed_pairs = []
        for se in ranked:
            line = f"[{se.entry.structural_role.value.upper()}] {se.entry.compressed}"
            cost = self._count(line)
            if tokens_used + cost > budget:
                break
            compressed_pairs.append((se, line))
            tokens_used += cost
            scores[se.entry.id] = se.score

        final_blocks, used_entries, source_refs = [], [], []
        for se, compressed_line in compressed_pairs:
            used_entries.append(se.entry)
            source_refs.append(se.entry.source_ref)
            source_line = f"SOURCE: {se.entry.source_text}"
            source_cost = self._count(source_line)
            if tokens_used + source_cost <= budget:
                final_blocks.append(f"{compressed_line}\n{source_line}")
                tokens_used += source_cost
            else:
                final_blocks.append(compressed_line)

        return AssembledContext(
            query=query, content="\n\n".join(final_blocks), entries_used=used_entries,
            source_refs=source_refs, tokens_used=tokens_used, token_budget=budget,
            budget_tier="medium", expansion_ratio=round(tokens_used / max(budget, 1), 3),
            scores=scores,
        )

    def _assemble_high(self, query, ranked, budget):
        """
        HIGH tier: always includes BOTH compressed AND full source for every entry.
        This makes HIGH a strict superset of MEDIUM, guaranteeing monotonicity.

        Format per entry:
            [ROLE | score=X.XX | topic=Y]
            compressed_text
            SOURCE: source_text
        """
        blocks, used_entries, source_refs, tokens_used, scores = [], [], [], 0, {}
        for se in ranked:
            entry = se.entry
            header = (
                f"[{entry.structural_role.value.upper()} | "
                f"score={se.score:.2f} | "
                f"topic={entry.metadata.topic}]"
            )
            # Full block = header + compressed + source (always both)
            full_block = f"{header}\n{entry.compressed}\nSOURCE: {entry.source_text}"
            full_cost = self._count(full_block)

            if tokens_used + full_cost <= budget:
                blocks.append(full_block)
                tokens_used += full_cost
            else:
                # Fallback: header + compressed only
                fallback = f"{header}\n{entry.compressed}"
                fallback_cost = self._count(fallback)
                if tokens_used + fallback_cost <= budget:
                    blocks.append(fallback)
                    tokens_used += fallback_cost
                else:
                    break

            used_entries.append(entry)
            source_refs.append(entry.source_ref)
            scores[entry.id] = se.score

        return AssembledContext(
            query=query, content="\n\n".join(blocks), entries_used=used_entries,
            source_refs=source_refs, tokens_used=tokens_used, token_budget=budget,
            budget_tier="high", expansion_ratio=round(tokens_used / max(budget, 1), 3),
            scores=scores,
        )

    def _empty(self, query, budget):
        return AssembledContext(
            query=query, content="", entries_used=[], source_refs=[],
            tokens_used=0, token_budget=budget, budget_tier=self._select_tier(budget),
            expansion_ratio=0.0, scores={},
        )

    def simulate(self, query, scored_entries, budgets=None):
        budgets = budgets or [200, 500, 1000, 2000, 4000]
        results = []
        for budget in budgets:
            ctx = self.assemble(query, scored_entries, budget)
            results.append({
                "budget": budget, "tier": ctx.budget_tier,
                "tokens_used": ctx.tokens_used,
                "expansion_ratio": ctx.expansion_ratio,
                "entries_included": len(ctx.entries_used),
            })
        return results
