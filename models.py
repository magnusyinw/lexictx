"""
models.py — LexiContext core data structures

All persistent and runtime objects used across the LexiContext pipeline.
No external dependencies required.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StructuralRole(str, Enum):
    """Semantic role of the source text this entry was derived from."""
    DECISION       = "decision"        # A choice was made
    KNOWLEDGE_UPDATE = "knowledge_update"  # Prior fact was superseded
    CONSTRAINT     = "constraint"      # A rule or hard limit
    OBSERVATION    = "observation"     # Something noticed, not yet acted on
    DEPENDENCY     = "dependency"      # Blocked on one or more conditions
    QUESTION       = "question"        # An open question
    OUTCOME        = "outcome"         # Result of a prior decision or action


class EntryStatus(str, Enum):
    """Lifecycle status of a ContextEntry in the Semantic Dictionary."""
    ACTIVE      = "active"       # Current, retrievable
    DEPRECATED  = "deprecated"   # Superseded by a newer entry
    FLAGGED     = "flagged"      # Conflict detected, pending resolution


class ConflictType(str, Enum):
    """How a write conflict was classified by the ConflictWriter."""
    NONE        = "none"         # No conflict detected
    UPDATE      = "update"       # Same entry revised in place
    SUPERSEDE   = "supersede"    # Old entry deprecated, new entry written
    FLAG        = "flag"         # Conflict surfaced for manual review


# ---------------------------------------------------------------------------
# EntryMetadata
# ---------------------------------------------------------------------------

@dataclass
class EntryMetadata:
    """
    Provenance and operational metadata attached to every ContextEntry.
    Used for freshness scoring, audit trails, and budget prioritisation.
    """

    timestamp: datetime
    # When this entry was written into the Semantic Dictionary

    role: str
    # Free-text role label (mirrors StructuralRole but allows custom values)

    topic: str
    # Coarse topic tag for grouping, e.g. "infrastructure", "compliance"

    freshness: float = 1.0
    # Decay score in [0.0, 1.0]. Starts at 1.0, decremented over time or
    # on supersede. Used by BudgetAssembler to rank expansion priority.

    usage_count: int = 0
    # How many times this entry has been retrieved. High-usage entries
    # get expansion priority under tight token budgets.

    source_session: Optional[str] = None
    # Identifier of the conversation session this entry originated from.
    # Used for temporal reasoning and session-level conflict detection.

    deprecated_by: Optional[str] = None
    # If status == DEPRECATED, points to the id of the replacing entry.

    supersedes: Optional[str] = None
    # If this entry superseded another, points to the deprecated entry id.

    confidence: float = 1.0
    # Distiller confidence score in [0.0, 1.0]. Low-confidence entries
    # are flagged rather than written unconditionally.


# ---------------------------------------------------------------------------
# ContextEntry
# ---------------------------------------------------------------------------

@dataclass
class ContextEntry:
    """
    The atomic unit of the Semantic Dictionary.

    Three-layer design:
      - feature_keys       → Index Layer  (how to hit)
      - compressed         → Entry Layer  (how to store)
      - source_text        → Source Layer (what to return)

    feature_keys are extracted from source_text BEFORE compression.
    This decouples retrieval recall from compression depth.
    """

    id: str
    # Stable unique identifier. Format: "entry_{uuid4_hex[:8]}"

    feature_keys: list[str]
    # Retrieval index. Extracted from source_text by the Distiller,
    # independently of the compression step.
    # Example: ["JWT", "session_tokens", "migration", "Sarah", "Q2"]

    semantic_address: str
    # Stable pointer to this entry's position in the source corpus.
    # Format: "conv_{session_id}:turn_{n}" or "doc_{doc_id}:chunk_{n}"

    compressed: str
    # Minimal-token semantic form produced by the Distiller.
    # Target: ≤25% of source_text character length.
    # Must preserve all semantics retrievable via feature_keys.

    source_ref: str
    # Character-level pointer into the original source document.
    # Format: "{doc_or_session_id}:char_{start}:char_{end}"
    # Used to expand source_text on demand (G2).

    source_text: str
    # The original verbatim slice this entry was derived from.
    # Never compressed or modified. Retrieved only when budget allows.

    structural_role: StructuralRole
    # Semantic role of this entry. Used by BudgetAssembler to prioritise
    # DECISION and CONSTRAINT entries under tight budgets.

    metadata: EntryMetadata
    # Provenance, freshness, usage, and audit fields.

    status: EntryStatus = EntryStatus.ACTIVE
    # Lifecycle state. Only ACTIVE entries are returned by default queries.

    entities: list[str] = field(default_factory=list)
    # Named entities extracted by the Distiller.
    # Subset of feature_keys; used for entity-specific queries.

    token_count_compressed: Optional[int] = None
    # Approximate token count of `compressed`. Pre-computed by Distiller
    # to speed up BudgetAssembler calculations.

    token_count_source: Optional[int] = None
    # Approximate token count of `source_text`. Pre-computed at write time.


# ---------------------------------------------------------------------------
# WriteResult
# ---------------------------------------------------------------------------

@dataclass
class WriteResult:
    """
    Returned by ConflictWriter after every write attempt.
    Encodes whether the write succeeded and how any conflict was handled.
    """

    success: bool
    # True if the entry was written (or updated) in the dictionary.
    # False only on hard errors (schema violation, storage failure).

    conflict_type: ConflictType
    # NONE       → clean write, no conflicts detected
    # UPDATE     → existing entry was revised in place
    # SUPERSEDE  → old entry deprecated, new entry written with back-pointer
    # FLAG       → conflict surfaced, no automatic write performed

    entry_id: str
    # ID of the entry that was written, updated, or flagged.

    affected_entries: list[str] = field(default_factory=list)
    # IDs of entries whose status changed as a result of this write.
    # Populated when conflict_type is SUPERSEDE (deprecated entry ids)
    # or FLAG (conflicting entry ids surfaced for review).

    message: Optional[str] = None
    # Human-readable explanation of the conflict resolution taken.
    # Always populated when conflict_type != NONE.

    timestamp: datetime = field(default_factory=datetime.utcnow)
    # When this write operation occurred. Used for audit log.


# ---------------------------------------------------------------------------
# AssembledContext
# ---------------------------------------------------------------------------

@dataclass
class AssembledContext:
    """
    Output of the BudgetAssembler.
    Contains the prompt-ready context and full accounting of what was used.
    """

    query: str
    # The original query that triggered this assembly.

    content: str
    # The assembled context string, ready for injection into an LLM prompt.
    # Composed of compressed entries and/or expanded source slices,
    # depending on token budget and relevance scores.

    entries_used: list[ContextEntry]
    # All ContextEntry objects whose data appears in `content`.

    source_refs: list[str]
    # Source references for every entry included in `content`.
    # Enables full traceability back to original evidence (G2).

    tokens_used: int
    # Actual token count of `content`. Always ≤ token_budget.

    token_budget: int
    # The budget this assembly was constrained to.

    budget_tier: str
    # Which tier was applied:
    # "low"    → feature_keys + compressed entries only
    # "medium" → compressed entries + critical source slices
    # "high"   → full source slice expansion for all matched entries

    expansion_ratio: float
    # tokens_used / token_budget. Values near 1.0 indicate tight budget.
    # Used to tune budget_tier thresholds over time.

    scores: dict[str, float] = field(default_factory=dict)
    # Relevance score per entry_id, as returned by FeatureMatcher.
    # Useful for debugging retrieval quality.
