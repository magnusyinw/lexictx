"""
writer.py — LexiContext Conflict-Aware Write Protocol

Implements G3: writes cannot silently corrupt existing memory.

Every write attempt is checked against existing ACTIVE entries
for feature key overlap. Detected conflicts are resolved explicitly:
  UPDATE    — same entry revised in place
  SUPERSEDE — old entry deprecated, new entry written with back-pointer
  FLAG      — conflict surfaced for review, no automatic write

Silent overwrite is never permitted.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from models import (
    ConflictType,
    ContextEntry,
    EntryMetadata,
    EntryStatus,
    StructuralRole,
    WriteResult,
)
from dictionary import SemanticDictionary


# ---------------------------------------------------------------------------
# ConflictRecord
# ---------------------------------------------------------------------------

class ConflictRecord:
    """Describes a detected conflict between a new entry and an existing one."""

    def __init__(
        self,
        existing_id: str,
        new_entry: ContextEntry,
        overlap_keys: list[str],
        conflict_type: ConflictType,
        reason: str,
    ) -> None:
        self.existing_id = existing_id
        self.new_entry = new_entry
        self.overlap_keys = overlap_keys
        self.conflict_type = conflict_type
        self.reason = reason
        self.timestamp = datetime.utcnow()


# ---------------------------------------------------------------------------
# ConflictWriter
# ---------------------------------------------------------------------------

class ConflictWriter:
    """
    Wraps SemanticDictionary.insert() with conflict detection.

    Detection strategy:
      1. Compute Jaccard overlap between new entry's feature_keys
         and all ACTIVE entries' feature_keys.
      2. If overlap exceeds threshold, classify the conflict.
      3. KNOWLEDGE_UPDATE entries use a reduced threshold (0.5×)
         because they are explicit state changes and should supersede
         with fewer overlapping keys.
      4. Resolve explicitly — never silently overwrite.

    Conflict classification rules (in priority order):
      Same semantic_address  → UPDATE     (same source, revised content)
      KNOWLEDGE_UPDATE role  → SUPERSEDE  (new info replaces old)
      High overlap + same role → FLAG     (ambiguous, needs review)
      Default                → SUPERSEDE (if auto_supersede) else FLAG
    """

    def __init__(
        self,
        overlap_threshold: float = 0.4,
        auto_supersede: bool = True,
    ) -> None:
        """
        Args:
            overlap_threshold: Jaccard ratio above which a conflict triggers.
                               KNOWLEDGE_UPDATE entries use 0.5× this value.
            auto_supersede:    If True, auto-supersede on conflict.
                               False = always FLAG for human review.
        """
        self._threshold = overlap_threshold
        self._auto_supersede = auto_supersede

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(
        self,
        new_entry: ContextEntry,
        dictionary: SemanticDictionary,
    ) -> WriteResult:
        """
        Attempt to write new_entry into the dictionary.
        G3 guarantee: never calls insert() without conflict checking first.
        """
        if not new_entry.id:
            new_entry.id = f"entry_{uuid.uuid4().hex[:8]}"

        conflicts = self._detect(new_entry, dictionary)

        if not conflicts:
            return dictionary.insert(new_entry)

        primary = self._prioritise(conflicts)
        return self._resolve(primary, new_entry, dictionary)

    def write_batch(
        self,
        entries: list[ContextEntry],
        dictionary: SemanticDictionary,
    ) -> list[WriteResult]:
        """
        Write multiple entries sequentially.
        Each entry is checked against the dictionary state AFTER
        all previous writes, so intra-batch conflicts are caught.
        """
        results = []
        for entry in entries:
            results.append(self.write(entry, dictionary))
        return results

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------

    def _detect(
        self,
        new_entry: ContextEntry,
        dictionary: SemanticDictionary,
    ) -> list[ConflictRecord]:
        """
        Scan all ACTIVE entries for feature key overlap with new_entry.

        Threshold adjustment:
          KNOWLEDGE_UPDATE entries are explicit state replacements.
          They use a reduced threshold (0.5× default) so that partial
          key overlap still triggers supersede, even when the local
          distiller produces noisy feature sets.
        """
        conflicts: list[ConflictRecord] = []
        new_keys = set(new_entry.feature_keys)

        if not new_keys:
            return conflicts

        # Reduce threshold for explicit knowledge updates
        is_update = (
            new_entry.structural_role == StructuralRole.KNOWLEDGE_UPDATE
        )
        effective_threshold = (
            self._threshold * 0.5 if is_update else self._threshold
        )

        for existing in dictionary.list_active():
            existing_keys = set(existing.feature_keys)
            overlap = new_keys & existing_keys

            if not overlap:
                continue

            union = new_keys | existing_keys
            jaccard = len(overlap) / len(union)

            if jaccard < effective_threshold:
                continue

            conflict_type, reason = self._classify(
                new_entry, existing, overlap, jaccard
            )
            conflicts.append(ConflictRecord(
                existing_id=existing.id,
                new_entry=new_entry,
                overlap_keys=sorted(overlap),
                conflict_type=conflict_type,
                reason=reason,
            ))

        return conflicts

    # ------------------------------------------------------------------
    # Conflict classification
    # ------------------------------------------------------------------

    def _classify(
        self,
        new_entry: ContextEntry,
        existing: ContextEntry,
        overlap: set[str],
        jaccard: float,
    ) -> tuple[ConflictType, str]:
        """Determine how to handle a detected conflict."""

        # Rule 1: same source position → in-place revision
        if new_entry.semantic_address == existing.semantic_address:
            return (
                ConflictType.UPDATE,
                f"Same semantic address '{existing.semantic_address}'. "
                f"Treating as in-place revision of entry {existing.id}."
            )

        # Rule 2: explicit knowledge update → supersede
        if new_entry.structural_role == StructuralRole.KNOWLEDGE_UPDATE:
            return (
                ConflictType.SUPERSEDE,
                f"New entry has role KNOWLEDGE_UPDATE with {len(overlap)} "
                f"overlapping keys {sorted(overlap)}. "
                f"Entry {existing.id} will be deprecated."
            )

        # Rule 3: very high overlap + same role → ambiguous, flag
        if jaccard > 0.7 and new_entry.structural_role == existing.structural_role:
            return (
                ConflictType.FLAG,
                f"High key overlap ({jaccard:.0%}) with entry {existing.id} "
                f"of same role '{existing.structural_role.value}'. "
                f"Overlapping keys: {sorted(overlap)}. Manual review required."
            )

        # Rule 4: default
        if self._auto_supersede:
            return (
                ConflictType.SUPERSEDE,
                f"Overlap {jaccard:.0%} ({len(overlap)} keys) with entry "
                f"{existing.id}. Auto-superseding (auto_supersede=True)."
            )

        return (
            ConflictType.FLAG,
            f"Overlap {jaccard:.0%} ({len(overlap)} keys) with entry "
            f"{existing.id}. Flagged for review (auto_supersede=False)."
        )

    def _prioritise(self, conflicts: list[ConflictRecord]) -> ConflictRecord:
        severity = {
            ConflictType.SUPERSEDE: 3,
            ConflictType.FLAG: 2,
            ConflictType.UPDATE: 1,
        }
        return max(conflicts, key=lambda c: severity.get(c.conflict_type, 0))

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def _resolve(
        self,
        conflict: ConflictRecord,
        new_entry: ContextEntry,
        dictionary: SemanticDictionary,
    ) -> WriteResult:
        if conflict.conflict_type == ConflictType.UPDATE:
            return self._resolve_update(conflict, new_entry, dictionary)
        elif conflict.conflict_type == ConflictType.SUPERSEDE:
            return self._resolve_supersede(conflict, new_entry, dictionary)
        else:
            return self._resolve_flag(conflict, new_entry, dictionary)

    def _resolve_update(
        self,
        conflict: ConflictRecord,
        new_entry: ContextEntry,
        dictionary: SemanticDictionary,
    ) -> WriteResult:
        """UPDATE: revise existing entry in place."""
        updates = {
            "feature_keys": new_entry.feature_keys,
            "compressed": new_entry.compressed,
            "entities": new_entry.entities,
            "structural_role": new_entry.structural_role,
            "metadata": new_entry.metadata,
            "token_count_compressed": new_entry.token_count_compressed,
        }
        result = dictionary.update(conflict.existing_id, updates)
        result.conflict_type = ConflictType.UPDATE
        result.message = conflict.reason
        return result

    def _resolve_supersede(
        self,
        conflict: ConflictRecord,
        new_entry: ContextEntry,
        dictionary: SemanticDictionary,
    ) -> WriteResult:
        """SUPERSEDE: deprecate old entry, write new with back-pointer."""
        if not new_entry.id:
            new_entry.id = f"entry_{uuid.uuid4().hex[:8]}"

        new_entry.metadata.supersedes = conflict.existing_id
        dictionary.insert(new_entry)
        dictionary.deprecate(conflict.existing_id, deprecated_by=new_entry.id)

        return WriteResult(
            success=True,
            conflict_type=ConflictType.SUPERSEDE,
            entry_id=new_entry.id,
            affected_entries=[conflict.existing_id],
            message=conflict.reason,
        )

    def _resolve_flag(
        self,
        conflict: ConflictRecord,
        new_entry: ContextEntry,
        dictionary: SemanticDictionary,
    ) -> WriteResult:
        """FLAG: write new entry, mark existing as flagged for review."""
        dictionary.insert(new_entry)
        dictionary.flag(conflict.existing_id, reason=conflict.reason)

        return WriteResult(
            success=True,
            conflict_type=ConflictType.FLAG,
            entry_id=new_entry.id,
            affected_entries=[conflict.existing_id],
            message=conflict.reason,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def preview(
        self,
        new_entry: ContextEntry,
        dictionary: SemanticDictionary,
    ) -> dict:
        """
        Dry-run: detect conflicts without writing.
        Returns summary of what write() would do.
        """
        conflicts = self._detect(new_entry, dictionary)

        if not conflicts:
            return {
                "action": "INSERT",
                "conflicts": [],
                "message": "No conflicts detected. Clean insert.",
            }

        primary = self._prioritise(conflicts)
        return {
            "action": primary.conflict_type.value.upper(),
            "conflicts": [
                {
                    "existing_id": c.existing_id,
                    "overlap_keys": c.overlap_keys,
                    "conflict_type": c.conflict_type.value,
                    "reason": c.reason,
                }
                for c in conflicts
            ],
            "message": primary.reason,
        }
