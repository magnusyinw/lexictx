"""
manager.py — LexiContext ContextManager
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from models import (
    AssembledContext, ContextEntry, EntryMetadata,
    EntryStatus, StructuralRole, WriteResult,
)
from dictionary import SemanticDictionary
from matcher import FeatureMatcher
from assembler import BudgetAssembler, BudgetTiers
from writer import ConflictWriter


_DISTILLER_SYSTEM = """You are a semantic distiller for a retrieval index.

Given input text, return ONLY a JSON object. No explanation, no markdown, no preamble.

JSON fields:

"feature_keys": A flat list of SPECIFIC words and phrases extracted from the text.
  Rules:
  - Include names of people, systems, locations, version numbers, technical terms, key concepts
  - Use snake_case for multi-word phrases: "kafka_pipeline", "session_tokens", "ml_platform"
  - Include both single words AND compound phrases
  - NEVER output category names like "named_entities" or "decisions_and_state_changes"
  - Only output actual values found in the text

"structural_role": Exactly one of: decision, knowledge_update, constraint, observation, dependency, question, outcome

"entities": List of specific named people, systems, or locations from the text

"compressed": The text compressed to 25% of original character count. Preserve all key facts. Use symbols: -> > : ()

"source_char_start": 0
"source_char_end": character length of the input text

---
EXAMPLE INPUT:
"We decided to migrate auth from JWT to session tokens. Sarah from the platform team owns the migration. Risk: increased Redis load at peak hours. Target: end of Q2."

EXAMPLE OUTPUT:
{
  "feature_keys": ["jwt", "session_tokens", "migration", "authentication", "sarah", "platform_team", "redis", "peak_load", "q2", "jwt_to_session", "auth_migration"],
  "structural_role": "decision",
  "entities": ["Sarah", "Redis"],
  "compressed": "Auth: JWT->session tokens. Owner: Sarah/platform. Risk: Redis peak load. Due Q2.",
  "source_char_start": 0,
  "source_char_end": 163
}

---
Now process the input text and return ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Stop words for local distiller
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'this', 'that', 'these', 'those', 'it', 'its',
    'as', 'if', 'not', 'no', 'any', 'all', 'both', 'each', 'during', 'per',
    'he', 'she', 'his', 'her', 'they', 'their', 'we', 'our', 'you', 'your',
    'also', 'into', 'than', 'then', 'now', 'up', 'out', 'about', 'over',
    'after', 'before', 'under', 'must', 'can', 'cannot', 'need', 'only',
    'when', 'while', 'since', 'because', 'however', 'therefore', 'thus',
    'such', 'there', 'here', 'where', 'who', 'which', 'what', 'how', 'why',
    'several', 'some', 'other', 'another', 'typically', 'usually',
}


def _local_distill(text: str) -> dict:
    """
    Local distiller for offline/testing use.

    Extracts:
      - All meaningful single words (no 20-item cap on candidates)
      - Adjacent word pairs as compound keys (bigrams)
      - Proper nouns (capitalized mid-sentence words)
      - Version strings (v4.2), quarter refs (Q2), @handles, ACRONYMS
      - Structural role from keyword signals
    """
    # ---- Single word keys ----
    all_words = re.findall(r"[a-zA-Z][a-zA-Z0-9]*", text)
    single_keys = [
        w.lower() for w in all_words
        if w.lower() not in _STOP_WORDS and len(w) >= 3
    ]

    # ---- Bigram + skip-bigram compound keys ----
    # Window of 3: captures adjacent AND one-word-apart pairs.
    # Example: "session-based tokens" -> meaningful=[session,based,tokens]
    #   bigrams:      session_based, based_tokens
    #   skip-bigrams: session_tokens   ← needed for "session_tokens" key
    meaningful = [w for w in all_words if w.lower() not in _STOP_WORDS and len(w) >= 2]
    bigrams = []
    for i in range(len(meaningful)):
        for j in range(i + 1, min(i + 3, len(meaningful))):
            a, b = meaningful[i].lower(), meaningful[j].lower()
            bigrams.append(f"{a}_{b}")

    # ---- Proper nouns (capitalized mid-sentence) ----
    sentences = re.split(r'(?<=[.!?])\s+', text)
    proper_nouns = []
    for sent in sentences:
        words = sent.strip().split()
        for i, w in enumerate(words):
            clean = re.sub(r"[^a-zA-Z0-9]", "", w)
            # Mid-sentence capitalized word = likely entity
            if i > 0 and clean and clean[0].isupper() and len(clean) >= 2:
                proper_nouns.append(clean.lower())

    # ---- Special patterns ----
    # Version strings: v4.2, v1.0.0
    versions = re.findall(r'\bv\d+[\.\d]+', text)
    # Quarter refs: Q1, Q2, Q3, Q4
    quarters = re.findall(r'\bQ[1-4]\b', text)
    # @handles
    handles = re.findall(r'@\w+', text)
    # Hyphenated compounds: server-side -> server_side
    hyphenated = [
        re.sub(r'-', '_', m.lower())
        for m in re.findall(r'[a-zA-Z]+-[a-zA-Z]+', text)
    ]
    # Time refs: 14:32, 800ms, 2,000
    time_metrics = re.findall(r'\b\d+(?::\d+|ms|%)\b', text)

    # ---- Entity list (capitalized words for metadata) ----
    entities_raw = []
    for sent in sentences:
        words = sent.strip().split()
        for i, w in enumerate(words):
            clean = re.sub(r"[^a-zA-Z0-9@._]", "", w)
            if i > 0 and clean and clean[0].isupper() and len(clean) >= 2:
                entities_raw.append(clean)
    entities_raw += [v for v in versions]
    entities_raw += quarters
    entities_raw += handles
    entities_raw += re.findall(r'\b[A-Z]{2,}\b', text)

    entities = list(dict.fromkeys(
        e for e in entities_raw
        if e.lower() not in _STOP_WORDS
    ))[:15]

    # ---- Combine all feature keys ----
    # Special patterns go FIRST to survive the [:40] cap
    all_candidates = (
        [v.lower() for v in versions]       # v4.2, v1.0.0
        + [q.lower() for q in quarters]     # Q1-Q4
        + [h.lower() for h in handles]      # @marcus_k
        + hyphenated                        # server_side
        + [t.lower() for t in time_metrics] # 800ms, 14:32
        + single_keys
        + bigrams
        + proper_nouns
    )

    seen: set[str] = set()
    feature_keys: list[str] = []
    for k in all_candidates:
        k = k.strip("_")
        if k and k not in seen and len(k) >= 2:
            seen.add(k)
            feature_keys.append(k)

    feature_keys = feature_keys[:40]

    # ---- Role detection ----
    text_lower = text.lower()
    role = "observation"
    role_signals = {
        "knowledge_update": ["update:", "moved", "transferred", "handed off", "handed to", "no longer", "as of this"],
        "decision":         ["decided", "will migrate", "chose", "approved", "agreed", "migrating", "switching"],
        "constraint":       ["must remain", "must be", "permanent", "mandatory", "written approval", "all data must"],
        "dependency":       ["blocked", "waiting for", "depends on", "pending", "sign-off", "unblock"],
        "observation":      ["spike", "alert", "shows", "detected", "flagged", "load test", "no action"],
    }
    for r, signals in role_signals.items():
        if any(s in text_lower for s in signals):
            role = r
            break

    # ---- Compression (25% of original) ----
    compressed = text[:max(10, len(text) // 4)]

    return {
        "feature_keys": feature_keys,
        "structural_role": role,
        "entities": entities,
        "compressed": compressed,
        "source_char_start": 0,
        "source_char_end": len(text),
    }


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class ContextManager:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        tiers: Optional[BudgetTiers] = None,
        overlap_threshold: float = 0.25,   # lowered from 0.4
        auto_supersede: bool = True,
        session_id: Optional[str] = None,
        default_topic: str = "general",
    ) -> None:
        import os
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model
        self._session_id = session_id or f"session_{uuid.uuid4().hex[:6]}"
        self._default_topic = default_topic
        self._turn = 0

        # Adaptive threshold:
        # Real Distiller (API) produces denser, richer feature_keys,
        # raising Jaccard between unrelated entries → need higher threshold.
        # Local fallback produces sparse keys → need lower threshold.
        if overlap_threshold == 0.25:
            overlap_threshold = 0.50 if self._api_key else 0.25

        self._dictionary = SemanticDictionary()
        self._matcher = FeatureMatcher()
        self._assembler = BudgetAssembler(tiers=tiers)
        self._writer = ConflictWriter(
            overlap_threshold=overlap_threshold,
            auto_supersede=auto_supersede,
        )

    # ------------------------------------------------------------------
    # learn()
    # ------------------------------------------------------------------

    def learn(
        self,
        text: str,
        topic: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> WriteResult:
        self._turn += 1
        sid = session_id or self._session_id
        distilled = self._distill(text)
        entry = self._build_entry(text, distilled, sid, topic)
        result = self._writer.write(entry, self._dictionary)
        self._rebuild_index()
        return result

    def learn_batch(self, texts: list[str], **kwargs) -> list[WriteResult]:
        results = []
        for text in texts:
            self._turn += 1
            sid = kwargs.get("session_id", self._session_id)
            distilled = self._distill(text)
            entry = self._build_entry(text, distilled, sid, kwargs.get("topic"))
            results.append(self._writer.write(entry, self._dictionary))
        self._rebuild_index()
        return results

    # ------------------------------------------------------------------
    # query()
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        token_budget: int = 2000,
        top_k: int = 10,
        role_filter: Optional[list[str]] = None,
        min_score: float = 0.0,
    ) -> AssembledContext:
        scored = self._matcher.match(
            query, top_k=top_k, min_score=min_score, role_filter=role_filter,
        )
        return self._assembler.assemble(query, scored, token_budget)

    # ------------------------------------------------------------------
    # expand() — G2
    # ------------------------------------------------------------------

    def expand(self, entry_id: str) -> Optional[str]:
        return self._dictionary.expand_source(entry_id)

    def expand_from_result(self, result: AssembledContext) -> dict[str, str]:
        return {e.id: e.source_text for e in result.entries_used}

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def preview_write(self, text: str) -> dict:
        distilled = self._distill(text)
        entry = self._build_entry(text, distilled, self._session_id, None)
        return self._writer.preview(entry, self._dictionary)

    def stats(self) -> dict:
        return {
            "session_id": self._session_id,
            "turns_processed": self._turn,
            "dictionary": self._dictionary.stats(),
            "matcher": self._matcher.stats(),
        }

    def conflict_log(self) -> list[dict]:
        return self._dictionary.conflict_log()

    def write_log(self):
        return self._dictionary.write_log

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        self._dictionary.save(path)

    def load(self, path: str) -> None:
        self._dictionary.load(path)
        self._rebuild_index()

    # ------------------------------------------------------------------
    # Internal: Distiller
    # ------------------------------------------------------------------

    def _distill(self, text: str) -> dict:
        """
        Call the LLM Distiller. Supports Kimi (Moonshot) and Anthropic.
        Auto-detects provider from API key prefix:
          sk-ant-... -> Anthropic
          sk-...     -> Kimi (Moonshot, OpenAI-compatible)
        Falls back to local distiller if no API key or on error.
        """
        if not self._api_key:
            return _local_distill(text)
        try:
            import urllib.request

            is_anthropic = self._api_key.startswith("sk-ant-")

            if is_anthropic:
                payload = json.dumps({
                    "model": self._model or "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                    "system": _DISTILLER_SYSTEM,
                    "messages": [{"role": "user", "content": text}],
                }).encode()
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages",
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self._api_key,
                        "anthropic-version": "2023-06-01",
                    },
                )
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read())
                    raw = data["content"][0]["text"].strip()
            else:
                # Kimi / OpenAI-compatible
                model = self._model if self._model != "claude-sonnet-4-20250514"                     else "moonshot-v1-8k"
                payload = json.dumps({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _DISTILLER_SYSTEM},
                        {"role": "user", "content": text},
                    ],
                    "max_tokens": 1000,
                    "temperature": 0,
                }).encode()
                req = urllib.request.Request(
                    "https://api.moonshot.cn/v1/chat/completions",
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self._api_key}",
                    },
                )
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read())
                    raw = data["choices"][0]["message"]["content"].strip()

            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)

        except Exception as e:
            print(f"[LexiContext] Distiller API error: {e}. Using local fallback.")
            return _local_distill(text)
        try:
            import urllib.request
            payload = json.dumps({
                "model": self._model,
                "max_tokens": 1000,
                "system": _DISTILLER_SYSTEM,
                "messages": [{"role": "user", "content": text}],
            }).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
                raw = data["content"][0]["text"].strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                return json.loads(raw)
        except Exception as e:
            print(f"[LexiContext] Distiller API error: {e}. Using local fallback.")
            return _local_distill(text)

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _build_entry(self, text, distilled, session_id, topic):
        return ContextEntry(
            id=f"entry_{uuid.uuid4().hex[:8]}",
            feature_keys=distilled.get("feature_keys", []),
            semantic_address=f"{session_id}:turn_{self._turn}",
            compressed=distilled.get("compressed", text[:200]),
            source_ref=(
                f"{session_id}:char_"
                f"{distilled.get('source_char_start', 0)}:"
                f"char_{distilled.get('source_char_end', len(text))}"
            ),
            source_text=text,
            structural_role=StructuralRole(
                distilled.get("structural_role", "observation")
            ),
            entities=distilled.get("entities", []),
            metadata=EntryMetadata(
                timestamp=datetime.utcnow(),
                role=distilled.get("structural_role", "observation"),
                topic=topic or self._default_topic,
                source_session=session_id,
            ),
        )

    def _rebuild_index(self) -> None:
        self._matcher.build(self._dictionary.list_active())
