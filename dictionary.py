"""
dictionary.py — LexiContext Semantic Dictionary

In-memory key-value store for ContextEntry objects.
Separates ACTIVE and DEPRECATED entries.
Supports JSON serialisation for persistence.
No external dependencies required.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from models import (
    AssembledContext,
    ConflictType,
    ContextEntry,
    EntryMetadata,
    EntryStatus,
    StructuralRole,
    WriteResult,
)


class SemanticDictionary:
    """
    The central store of the LexiContext pipeline.

    Maintains two internal stores:
      _active     — entries currently retrievable
      _deprecated — entries superseded or soft-deleted, retained for audit

    Every write operation appends to write_log, providing a full
    audit trail of inserts, updates, supersedes, and flags.
    """

    def __init__(self) -> None:
        self._active: dict[str, ContextEntry] = {}
        self._deprecated: dict[str, ContextEntry] = {}
        self.write_log: list[WriteResult] = []
        self._conflict_log: list[dict] = []

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def insert(self, entry: ContextEntry) -> WriteResult:
        """
        Write a new entry into the active store.

        Does NOT perform conflict detection — that is the responsibility
        of ConflictWriter. This method trusts the entry as given.

        Returns a WriteResult with conflict_type=NONE on clean insert.
        """
        self._active[entry.id] = entry

        result = WriteResult(
            success=True,
            conflict_type=ConflictType.NONE,
            entry_id=entry.id,
            message=f"Inserted entry {entry.id}.",
        )
        self.write_log.append(result)
        return result

    def update(self, entry_id: str, updates: dict) -> WriteResult:
        """
        Revise fields of an existing ACTIVE entry in place.

        `updates` is a dict of field_name → new_value.
        Immutable fields (id, source_ref, source_text) are ignored.

        Returns WriteResult with conflict_type=UPDATE.
        """
        IMMUTABLE = {"id", "source_ref", "source_text", "semantic_address"}

        if entry_id not in self._active:
            return WriteResult(
                success=False,
                conflict_type=ConflictType.NONE,
                entry_id=entry_id,
                message=f"Entry {entry_id} not found in active store.",
            )

        entry = self._active[entry_id]
        changed = []
        for field, value in updates.items():
            if field in IMMUTABLE:
                continue
            if hasattr(entry, field):
                setattr(entry, field, value)
                changed.append(field)

        result = WriteResult(
            success=True,
            conflict_type=ConflictType.UPDATE,
            entry_id=entry_id,
            message=f"Updated fields {changed} on entry {entry_id}.",
        )
        self.write_log.append(result)
        return result

    def deprecate(self, entry_id: str, deprecated_by: str) -> bool:
        """
        Move an ACTIVE entry to the deprecated store.

        Sets status=DEPRECATED and records the replacing entry id.
        The entry is retained for audit and provenance tracing (G2).

        Returns True if the entry was found and deprecated.
        """
        if entry_id not in self._active:
            return False

        entry = self._active.pop(entry_id)
        entry.status = EntryStatus.DEPRECATED
        entry.metadata.deprecated_by = deprecated_by
        entry.metadata.freshness = 0.0
        self._deprecated[entry_id] = entry

        result = WriteResult(
            success=True,
            conflict_type=ConflictType.SUPERSEDE,
            entry_id=deprecated_by,
            affected_entries=[entry_id],
            message=f"Entry {entry_id} deprecated, superseded by {deprecated_by}.",
        )
        self.write_log.append(result)
        return True

    def flag(self, entry_id: str, reason: str) -> WriteResult:
        """
        Mark an entry as FLAGGED without modifying or deprecating it.

        Used by ConflictWriter when a conflict cannot be auto-resolved.
        Flagged entries remain retrievable but are annotated for review.

        Returns WriteResult with conflict_type=FLAG.
        """
        if entry_id not in self._active:
            return WriteResult(
                success=False,
                conflict_type=ConflictType.FLAG,
                entry_id=entry_id,
                message=f"Entry {entry_id} not found.",
            )

        self._active[entry_id].status = EntryStatus.FLAGGED
        conflict_record = {
            "entry_id": entry_id,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._conflict_log.append(conflict_record)

        result = WriteResult(
            success=True,
            conflict_type=ConflictType.FLAG,
            entry_id=entry_id,
            affected_entries=[entry_id],
            message=f"Entry {entry_id} flagged: {reason}",
        )
        self.write_log.append(result)
        return result

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, entry_id: str) -> Optional[ContextEntry]:
        """
        Retrieve a single entry by id.
        Checks active store first, then deprecated (for provenance tracing).
        """
        return self._active.get(entry_id) or self._deprecated.get(entry_id)

    def list_active(self) -> list[ContextEntry]:
        """Return all ACTIVE entries, sorted by freshness descending."""
        return sorted(
            self._active.values(),
            key=lambda e: e.metadata.freshness,
            reverse=True,
        )

    def list_flagged(self) -> list[ContextEntry]:
        """Return all FLAGGED entries pending conflict resolution."""
        return [
            e for e in self._active.values()
            if e.status == EntryStatus.FLAGGED
        ]

    def list_deprecated(self) -> list[ContextEntry]:
        """Return all DEPRECATED entries (audit / provenance use)."""
        return list(self._deprecated.values())

    def expand_source(self, entry_id: str) -> Optional[str]:
        """
        Return the original source_text for a given entry id.
        Core mechanism for G2 (every hit traces back to original evidence).
        Returns None if the entry does not exist.
        """
        entry = self.get(entry_id)
        return entry.source_text if entry else None

    def conflict_log(self) -> list[dict]:
        """Return all recorded conflict events for review."""
        return list(self._conflict_log)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Summary counts for monitoring and debugging."""
        return {
            "active": len(self._active),
            "deprecated": len(self._deprecated),
            "flagged": len(self.list_flagged()),
            "write_operations": len(self.write_log),
            "unresolved_conflicts": len(self._conflict_log),
        }

    # ------------------------------------------------------------------
    # Persistence — JSON serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise the full dictionary state to a JSON file.
        Includes active entries, deprecated entries, write_log,
        and conflict_log for complete audit trail preservation.
        """
        def serialise_entry(e: ContextEntry) -> dict:
            return {
                "id": e.id,
                "feature_keys": e.feature_keys,
                "semantic_address": e.semantic_address,
                "compressed": e.compressed,
                "source_ref": e.source_ref,
                "source_text": e.source_text,
                "structural_role": e.structural_role.value,
                "status": e.status.value,
                "entities": e.entities,
                "token_count_compressed": e.token_count_compressed,
                "token_count_source": e.token_count_source,
                "metadata": {
                    "timestamp": e.metadata.timestamp.isoformat(),
                    "role": e.metadata.role,
                    "topic": e.metadata.topic,
                    "freshness": e.metadata.freshness,
                    "usage_count": e.metadata.usage_count,
                    "source_session": e.metadata.source_session,
                    "deprecated_by": e.metadata.deprecated_by,
                    "supersedes": e.metadata.supersedes,
                    "confidence": e.metadata.confidence,
                },
            }

        def serialise_result(r: WriteResult) -> dict:
            return {
                "success": r.success,
                "conflict_type": r.conflict_type.value,
                "entry_id": r.entry_id,
                "affected_entries": r.affected_entries,
                "message": r.message,
                "timestamp": r.timestamp.isoformat(),
            }

        payload = {
            "active": {k: serialise_entry(v) for k, v in self._active.items()},
            "deprecated": {k: serialise_entry(v) for k, v in self._deprecated.items()},
            "write_log": [serialise_result(r) for r in self.write_log],
            "conflict_log": self._conflict_log,
        }

        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def load(self, path: str) -> None:
        """
        Deserialise dictionary state from a JSON file produced by save().
        Replaces current in-memory state entirely.
        """
        def deserialise_entry(d: dict) -> ContextEntry:
            meta = d["metadata"]
            return ContextEntry(
                id=d["id"],
                feature_keys=d["feature_keys"],
                semantic_address=d["semantic_address"],
                compressed=d["compressed"],
                source_ref=d["source_ref"],
                source_text=d["source_text"],
                structural_role=StructuralRole(d["structural_role"]),
                status=EntryStatus(d["status"]),
                entities=d.get("entities", []),
                token_count_compressed=d.get("token_count_compressed"),
                token_count_source=d.get("token_count_source"),
                metadata=EntryMetadata(
                    timestamp=datetime.fromisoformat(meta["timestamp"]),
                    role=meta["role"],
                    topic=meta["topic"],
                    freshness=meta["freshness"],
                    usage_count=meta["usage_count"],
                    source_session=meta.get("source_session"),
                    deprecated_by=meta.get("deprecated_by"),
                    supersedes=meta.get("supersedes"),
                    confidence=meta.get("confidence", 1.0),
                ),
            )

        def deserialise_result(d: dict) -> WriteResult:
            return WriteResult(
                success=d["success"],
                conflict_type=ConflictType(d["conflict_type"]),
                entry_id=d["entry_id"],
                affected_entries=d.get("affected_entries", []),
                message=d.get("message"),
                timestamp=datetime.fromisoformat(d["timestamp"]),
            )

        payload = json.loads(Path(path).read_text())
        self._active = {
            k: deserialise_entry(v) for k, v in payload["active"].items()
        }
        self._deprecated = {
            k: deserialise_entry(v) for k, v in payload["deprecated"].items()
        }
        self.write_log = [deserialise_result(r) for r in payload["write_log"]]
        self._conflict_log = payload.get("conflict_log", [])
