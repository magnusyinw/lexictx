"""
matcher.py — LexiContext Feature Matcher

BM25-based retrieval over feature_keys and entities.
Feature keys are extracted from source text independently of compression,
so matching operates on source-derived signals rather than compressed text.

No external dependencies except rank_bm25.
Install: pip install rank-bm25
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from models import ContextEntry, EntryStatus


# ---------------------------------------------------------------------------
# ScoredEntry — retrieval result wrapper
# ---------------------------------------------------------------------------

@dataclass
class ScoredEntry:
    """A ContextEntry paired with its retrieval relevance score."""

    entry: ContextEntry
    score: float
    matched_keys: list[str] = field(default_factory=list)
    # Which feature_keys from the entry overlapped with the query tokens.
    # Used for result explainability and debug.


# ---------------------------------------------------------------------------
# BM25 — minimal self-contained implementation
# ---------------------------------------------------------------------------

class BM25:
    """
    Minimal BM25 implementation with no external dependencies.

    Operates over token lists (not raw strings).
    k1 and b are standard BM25 hyperparameters.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._corpus: list[list[str]] = []
        self._doc_freqs: list[dict[str, int]] = []
        self._idf: dict[str, float] = {}
        self._avg_dl: float = 0.0
        self._n_docs: int = 0

    def index(self, tokenised_docs: list[list[str]]) -> None:
        """Build the BM25 index from a list of pre-tokenised documents."""
        self._corpus = tokenised_docs
        self._n_docs = len(tokenised_docs)
        if self._n_docs == 0:
            return

        # Document frequencies
        self._doc_freqs = []
        df: dict[str, int] = {}
        total_len = 0

        for doc in tokenised_docs:
            total_len += len(doc)
            freqs: dict[str, int] = {}
            for token in doc:
                freqs[token] = freqs.get(token, 0) + 1
            self._doc_freqs.append(freqs)
            for token in freqs:
                df[token] = df.get(token, 0) + 1

        self._avg_dl = total_len / self._n_docs

        # IDF with smoothing
        self._idf = {}
        for token, freq in df.items():
            self._idf[token] = math.log(
                (self._n_docs - freq + 0.5) / (freq + 0.5) + 1
            )

    def score(self, query_tokens: list[str], doc_index: int) -> float:
        """Compute BM25 score for a single document given query tokens."""
        if self._n_docs == 0:
            return 0.0

        freqs = self._doc_freqs[doc_index]
        dl = len(self._corpus[doc_index])
        score = 0.0

        for token in query_tokens:
            if token not in freqs:
                continue
            tf = freqs[token]
            idf = self._idf.get(token, 0.0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * dl / max(self._avg_dl, 1)
            )
            score += idf * numerator / denominator

        return score

    def query(
        self,
        query_tokens: list[str],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Score all indexed documents against query_tokens.
        Returns list of (doc_index, score) sorted by score descending,
        filtered to top_k results with score > 0.
        """
        scores = [
            (i, self.score(query_tokens, i))
            for i in range(self._n_docs)
        ]
        scores = [(i, s) for i, s in scores if s > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# FeatureMatcher
# ---------------------------------------------------------------------------

class FeatureMatcher:
    """
    Indexes ContextEntry.feature_keys using BM25 for query matching.

    Design principle: feature_keys were extracted from source_text BEFORE
    compression, so this matcher operates on source-derived signals.
    A query that would have matched the original source will match here
    even if the term did not survive compression (G1).

    Two-hop retrieval:
      1. match()  — cheap index scan, returns semantic addresses
      2. expand() — called by BudgetAssembler when budget allows
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._bm25 = BM25(k1=k1, b=b)
        self._entries: list[ContextEntry] = []
        self._tokenised: list[list[str]] = []
        self._is_indexed = False

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    @staticmethod
    def tokenise(text: str) -> list[str]:
        """
        Tokenise with light suffix stemming so plural/verb query forms
        match base-form feature keys.
          "risks" -> also yields "risk"
          "authentication" -> also yields stem
        Originals are always preserved; stems added as extras.
        """
        text = text.replace("_", " ")
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        extra = []
        for t in tokens:
            if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
                extra.append(t[:-1])
            if len(t) > 6 and t.endswith("ing"):
                extra.append(t[:-3])
            if len(t) > 5 and t.endswith("ed"):
                extra.append(t[:-2])
        seen, result = set(), []
        for t in tokens + extra:
            if t not in seen and len(t) >= 2:
                seen.add(t)
                result.append(t)
        return result

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build(self, entries: list[ContextEntry]) -> None:
        """
        Build the BM25 index from a list of ContextEntry objects.

        Only ACTIVE entries are indexed. Deprecated or flagged entries
        are excluded from retrieval.

        Each entry is represented as the union of its feature_keys
        and entity tokens — the source-derived index signals.
        """
        active = [e for e in entries if e.status == EntryStatus.ACTIVE]
        self._entries = active

        self._tokenised = []
        for entry in active:
            # Combine feature_keys + entities as the indexable surface
            index_surface = " ".join(entry.feature_keys + entry.entities)
            self._tokenised.append(self.tokenise(index_surface))

        self._bm25.index(self._tokenised)
        self._is_indexed = True

    def add(self, entry: ContextEntry) -> None:
        """
        Incrementally add a single entry and rebuild the index.

        For production use, batch rebuilds via build() are more efficient.
        add() is provided for single-entry updates after ConflictWriter writes.
        """
        if entry.status != EntryStatus.ACTIVE:
            return
        entries = self._entries + [entry]
        self.build(entries)

    def remove(self, entry_id: str) -> None:
        """
        Remove an entry from the index (e.g. after deprecation).
        Triggers a full index rebuild.
        """
        entries = [e for e in self._entries if e.id != entry_id]
        self.build(entries)

    # ------------------------------------------------------------------
    # Retrieval — first hop
    # ------------------------------------------------------------------

    def match(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        role_filter: Optional[list[str]] = None,
    ) -> list[ScoredEntry]:
        """
        Match a natural language query against the feature key index.

        This is the first hop of two-hop retrieval:
          match() → semantic addresses (cheap)
          expand() → source slices (on demand, budget-gated)

        Args:
            query:       Natural language query string.
            top_k:       Maximum number of results to return.
            min_score:   Minimum BM25 score threshold (0.0 = no filter).
            role_filter: If set, restrict results to these structural roles.

        Returns:
            List of ScoredEntry objects sorted by relevance descending.
        """
        if not self._is_indexed or not self._entries:
            return []

        query_tokens = self.tokenise(query)
        if not query_tokens:
            return []

        raw_results = self._bm25.query(query_tokens, top_k=top_k * 2)

        scored: list[ScoredEntry] = []
        for doc_idx, score in raw_results:
            if score < min_score:
                continue

            entry = self._entries[doc_idx]

            if role_filter and entry.structural_role.value not in role_filter:
                continue

            # Find which feature_keys overlapped with query tokens
            entry_tokens = set(self._tokenised[doc_idx])
            matched_keys = [
                key for key in entry.feature_keys
                if any(t in entry_tokens for t in self.tokenise(key))
                and any(t in self.tokenise(key) for t in query_tokens)
            ]

            scored.append(ScoredEntry(
                entry=entry,
                score=round(score, 4),
                matched_keys=matched_keys,
            ))

        # Increment usage_count on matched entries
        for se in scored[:top_k]:
            se.entry.metadata.usage_count += 1

        return scored[:top_k]

    def match_by_entity(self, entity: str) -> list[ScoredEntry]:
        """
        Exact entity lookup. Returns all active entries containing
        the specified entity name (case-insensitive).

        Useful for provenance queries: "show everything about Sarah".
        """
        entity_lower = entity.lower()
        results = []
        for entry in self._entries:
            if any(e.lower() == entity_lower for e in entry.entities):
                results.append(ScoredEntry(
                    entry=entry,
                    score=1.0,
                    matched_keys=[entity],
                ))
        return results

    def match_by_role(self, role: str) -> list[ScoredEntry]:
        """
        Return all active entries with the specified structural_role.
        Useful for audit queries: "show all decisions" / "show all constraints".
        """
        return [
            ScoredEntry(entry=e, score=1.0)
            for e in self._entries
            if e.structural_role.value == role
        ]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def explain(self, query: str, entry_id: str) -> dict:
        """
        Return a debug breakdown of why (or why not) an entry matched
        a given query. Useful for evaluating Distiller quality.
        """
        entry = next((e for e in self._entries if e.id == entry_id), None)
        if not entry:
            return {"error": f"Entry {entry_id} not in active index."}

        query_tokens = set(self.tokenise(query))
        entry_tokens = set(self.tokenise(
            " ".join(entry.feature_keys + entry.entities)
        ))

        overlap = query_tokens & entry_tokens
        missed = query_tokens - entry_tokens

        doc_idx = next(
            (i for i, e in enumerate(self._entries) if e.id == entry_id), None
        )
        score = (
            self._bm25.score(list(query_tokens), doc_idx)
            if doc_idx is not None else 0.0
        )

        return {
            "entry_id": entry_id,
            "query_tokens": sorted(query_tokens),
            "entry_tokens": sorted(entry_tokens),
            "overlap": sorted(overlap),
            "missed_query_tokens": sorted(missed),
            "bm25_score": round(score, 4),
            "feature_keys": entry.feature_keys,
        }

    def stats(self) -> dict:
        """Index summary for monitoring."""
        return {
            "indexed_entries": len(self._entries),
            "is_indexed": self._is_indexed,
            "avg_feature_keys": (
                round(
                    sum(len(e.feature_keys) for e in self._entries)
                    / max(len(self._entries), 1),
                    1,
                )
            ),
        }
