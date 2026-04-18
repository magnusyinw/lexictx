# Section 3: Architecture

LexiContext is organized around a write path and a read path, mediated by a
Semantic Dictionary. Figure 1 shows the overall system architecture.

```
Write path:
  Raw input → Semantic Distiller → Index Layer + Entry Layer → Semantic Dictionary

Read path:
  New query → Feature Matcher → Budget Assembler → Source Layer → LLM prompt
```

We describe each component in turn.

---

## 3.1 The Semantic Dictionary

The Semantic Dictionary is the central data structure of LexiContext. It stores
ContextEntry objects, each of which maintains three independently managed
representations:

**Index Layer.** A list of feature keys — specific words, entity names,
compound phrases, version identifiers, temporal references, and structural
role labels — extracted from the original source text. Feature keys are the
sole basis for retrieval matching. They are extracted before compression and
are never derived from the compressed entry.

**Entry Layer.** A compressed semantic representation of the source text,
targeting 25% of the original character count. The compressed entry preserves
the meaning of the source but removes redundant phrasing, connective language,
and stylistic content. It is linked by a stable semantic address to the
source slice.

**Source Layer.** The original verbatim source text, stored locally and
retrievable on demand through the semantic address. The source is never
modified after ingestion.

Each entry also carries metadata: timestamp, structural role, topic tag,
freshness score, usage count, and lifecycle status (ACTIVE, DEPRECATED,
or FLAGGED).

The structural role classifies the semantic function of each entry: DECISION,
KNOWLEDGE_UPDATE, CONSTRAINT, OBSERVATION, DEPENDENCY, QUESTION, or OUTCOME.
This classification is used by the Budget Assembler to prioritize expansion
order under tight budgets (CONSTRAINT and DECISION entries are expanded first)
and by the Conflict Writer to determine resolution strategy.

---

## 3.2 Semantic Distiller

The Distiller is the write-time component that converts raw input text into
an Index Layer and an Entry Layer. It performs two operations independently
on the same source text:

**Feature extraction (Task A).** The Distiller identifies and extracts:
- Named entities: people, systems, locations, version numbers
- Decisions and state changes
- Constraints and risk conditions
- Intents and structural role signals
- Numeric metrics and time references

Extracted signals are normalized to snake_case compound keys
(e.g., `session_tokens`, `kafka_pipeline`, `q2_deadline`) and deduplicated.
The extraction operates on the original source text, not on the compressed
output.

**Semantic compression (Task B).** The Distiller compresses the source text
to approximately 25% of original character count, preserving complete semantic
content while removing stylistic redundancy. Symbol substitution is encouraged
(→ for transitions, > for thresholds, : for attribution) to maximize
information density within the character budget.

The two tasks are issued in a single LLM call with a structured output
requirement (JSON with fields: `feature_keys`, `structural_role`, `entities`,
`compressed`, `source_char_start`, `source_char_end`). The Distiller backend
is LLM-agnostic; the current implementation uses Kimi (moonshot-v1-8k) via an
OpenAI-compatible API. A local fallback distiller using BM25 tokenization with
skip-bigram extraction is available for offline use and testing.

The critical design constraint is the independence of Task A and Task B:
feature keys are extracted from source text, not from compressed text. This
is what allows the Index Layer to remain addressable even when the Entry Layer
is highly compressed.

---

## 3.3 Feature Matcher

The Feature Matcher implements the first hop of two-hop retrieval. It builds
a BM25 index over the feature keys of all ACTIVE entries in the Semantic
Dictionary and matches incoming queries against this index.

**Tokenization.** Feature keys are tokenized by splitting snake_case
components, handling camelCase transitions, and applying light suffix
stemming. The stemmer adds stem variants for common English suffixes (-s,
-ing, -ed, -tion) without removing the original token, so both `risk` and
`risks` match the feature key `risk`. Skip-bigrams with window size 3 are
generated at ingest time to capture non-adjacent meaningful word pairs
(e.g., `session_tokens` from "session-based tokens").

**BM25 scoring.** A self-contained BM25 implementation (k₁ = 1.5, b = 0.75)
scores each indexed entry against the query token set. Entries are ranked by
score and the top-k (default: 10) results above a minimum score threshold
(default: 0) are returned as ScoredEntry objects.

**Supplementary retrieval.** Two additional matching modes are available:
entity-exact lookup (returns all entries containing a specific named entity,
case-insensitive) and role-filter retrieval (returns all entries with a
specified structural role). These modes are used for audit queries ("show all
CONSTRAINT entries") and provenance tracing ("show all entries mentioning
Sarah").

The Feature Matcher returns ScoredEntry objects containing the matched
ContextEntry, the BM25 score, and the list of feature keys that overlapped
with the query tokens. The overlap list supports result explainability without
requiring access to the source text.

---

## 3.4 Budget Assembler

The Budget Assembler implements the second hop of two-hop retrieval. It takes
the ranked list of ScoredEntry objects from the Feature Matcher and assembles
a prompt-ready context string within the available token budget.

**Entry ranking.** Before assembly, entries are re-ranked by a composite key:
structural role priority (CONSTRAINT > DECISION > DEPENDENCY > KNOWLEDGE_UPDATE
> OBSERVATION > OUTCOME > QUESTION), BM25 score, and freshness. This ensures
that critical entries (constraints, decisions) are included first when budget
is limited.

**Three-tier expansion.** The assembly strategy is determined by the token
budget:

- **Low tier** (budget ≤ 500 tokens): Compressed entries only. Format:
  `[ROLE] compressed_text`. No source expansion.

- **Medium tier** (500 < budget ≤ 2,000 tokens): Two-pass assembly. Pass 1
  fits compressed entries up to the budget. Pass 2 expands source text for
  high-priority entries in order until the budget is exhausted. Format:
  `[ROLE] compressed_text\nSOURCE: source_text`.

- **High tier** (budget > 2,000 tokens): Full source expansion for all matched
  entries, with compressed entry included alongside source text. Falls back to
  compressed-only for entries where source expansion would exceed the budget.
  Format: `[ROLE | score=X.XX | topic=Y]\ncompressed_text\nSOURCE: source_text`.

The high tier is a strict superset of the medium tier output, guaranteeing
that token usage is monotonically non-decreasing as budget increases.

The Assembler returns an AssembledContext object recording: the assembled
content string, the list of ContextEntry objects included, their source
references, total tokens used, token budget, budget tier, and expansion ratio.
This accounting enables per-query G4 benchmarking without additional
instrumentation.

---

## 3.5 Conflict Writer

The Conflict Writer mediates all write operations to the Semantic Dictionary,
implementing G3 (no silent writes). Every ingest call goes through the
Conflict Writer before reaching the dictionary.

**Conflict detection.** For each new entry, the Conflict Writer scans all
ACTIVE entries for feature key overlap. Overlap is quantified as Jaccard
similarity: |new_keys ∩ existing_keys| / |new_keys ∪ existing_keys|. If
Jaccard similarity exceeds the configured threshold (default: 0.50 for API
Distiller, 0.25 for local fallback), a conflict is registered.

**Resolution classification.** Detected conflicts are classified by three
rules applied in priority order:

1. *Same semantic address*: The new entry references the same source position
   as an existing entry → **UPDATE** (revise in place).

2. *KNOWLEDGE_UPDATE structural role*: The new entry explicitly signals a
   state change → **SUPERSEDE** (deprecate old entry, write new entry with
   back-pointer). The Jaccard threshold for this rule is halved to catch
   partial-overlap updates.

3. *High overlap (> 0.70) with same structural role*: Ambiguous conflict,
   cannot be auto-resolved → **FLAG** (write new entry, mark existing as
   FLAGGED, surface to conflict log for human review).

**Audit trail.** SUPERSEDE resolutions retain the deprecated entry in a
separate store with status=DEPRECATED and a pointer to the replacing entry.
The replacing entry carries a `supersedes` pointer back to the deprecated
entry. Both directions are preserved for audit. The write log records every
write operation with its resolution type, affected entry IDs, and timestamp.

---

## 3.6 Context Manager

The Context Manager is the single public interface of LexiContext. It wires
together the Distiller, Semantic Dictionary, Feature Matcher, Budget Assembler,
and Conflict Writer behind a four-method API:

```python
cm = ContextManager(api_key="...")

# Write path
cm.learn(text)                        # ingest + distill + conflict-check + index

# Read path
ctx = cm.query(query, token_budget)   # match + assemble
source = cm.expand(entry_id)          # on-demand source retrieval (G2)

# Persistence
cm.save(path)                         # serialize to JSON
cm.load(path)                         # restore and rebuild index
```

The `learn()` method accepts raw text and an optional topic tag. Internally,
it calls the Distiller, constructs a ContextEntry, passes it through the
Conflict Writer, and triggers an index rebuild. The `query()` method returns
an AssembledContext object containing the prompt-ready content and full
accounting metadata. The `expand()` method retrieves the original source text
for a given entry ID, supporting G2 (traceability) without requiring source
text to be included in every query result.


---

# Section 4: Design Guarantees

LexiContext is designed around four properties that must hold jointly. We
state each guarantee formally and describe the architectural mechanism that
enforces it.

---

## 4.1 G1: Compression Without Material Loss of Addressability

**Statement.** A query that would match an entry based on the original source
text must match the corresponding Index Layer entry with equivalent recall,
regardless of the compression ratio applied to the Entry Layer.

**Mechanism.** Feature keys are extracted from source text in Task A of the
Distiller, which runs independently of and prior to the compression step
(Task B). The Feature Matcher operates exclusively against the Index Layer;
it never accesses the Entry Layer during matching. Compression depth therefore
has no effect on the feature key set available for retrieval.

This is the critical difference from systems where the retrieval index is
derived from the compressed output: in those systems, any vocabulary that does
not survive compression is no longer matchable. In LexiContext, the compression
step and the indexing step share only their input (the source text); their
outputs are independent.

**Condition for violation.** G1 can be violated only if the Distiller fails to
extract a relevant feature key from the source text during Task A. This is a
Distiller quality issue, not an architectural issue, and can be measured and
improved independently of the rest of the system through feature recall
evaluation on held-out annotated samples.

---

## 4.2 G2: Source Traceability

**Statement.** Every result returned by a query can be expanded to display
the original source text from which it was derived. Compressed entries are
never presented as primary sources.

**Mechanism.** Every ContextEntry maintains a `source_ref` field pointing to
the originating session and character range. The `source_text` field stores
the original verbatim slice at write time and is never modified. The
`expand()` method of the Context Manager retrieves source text by entry ID.

The Conflict Writer enforces a structural invariant: no entry can be written
to the Semantic Dictionary without a valid `source_ref`. The only exception
is the SUPERSEDE write path, where the new entry carries a `supersedes`
pointer to the deprecated entry, and the deprecated entry retains its original
`source_ref`, preserving the full provenance chain.

---

## 4.3 G3: Write Safety

**Statement.** A write operation that conflicts with an existing entry will
never silently overwrite it. Every detected conflict is resolved explicitly
through one of three operations: UPDATE (in-place revision), SUPERSEDE
(deprecation with audit pointer), or FLAG (surface for human review). Audit
history is always preserved.

**Mechanism.** The Conflict Writer scans the full ACTIVE entry set before
every write, computing pairwise Jaccard similarity between the new entry's
feature keys and all existing entries. Any overlap exceeding the threshold
triggers the resolution classifier. Silent overwrite is structurally impossible
because the `insert()` method of the Semantic Dictionary is never called
directly from user-facing code; all writes pass through the Conflict Writer.

**Audit preservation.** DEPRECATED entries are retained in a separate store
indefinitely. They are excluded from retrieval (not returned by `list_active()`)
but accessible through `expand()` for provenance queries. The write log
records every resolution with its type, affected entry IDs, reason string, and
timestamp.

---

## 4.4 G4: Token Efficiency

**Statement.** On long-session benchmarks, LexiContext must achieve
statistically significant token reduction relative to full-context replay,
without significant degradation of factual recall on the assembled content.

**Mechanism.** Token efficiency arises from two independent sources:

1. *Retrieval selectivity*: The Feature Matcher returns only entries relevant
   to the current query. All other entries in the Semantic Dictionary are
   excluded from the assembled context. As session count grows, the proportion
   of excluded entries increases, compounding the savings.

2. *Tier-gated expansion*: Within the matched entry set, the Budget Assembler
   exposes source text only as far as the token budget permits. Under the low
   and medium tiers, much of the content is delivered in compressed form,
   further reducing token consumption.

**Measurement.** G4 is measured as:

  reduction = 1 − (lexi_tokens / full_context_tokens)

where `lexi_tokens` is the token count of the assembled context and
`full_context_tokens` is the token count of the complete session history
concatenated. A positive reduction indicates savings over full-context replay.

**Condition for negative savings.** For very short sessions (source text
shorter than compressed + source overhead), the medium tier output can exceed
the raw source length, producing negative savings. This is detected automatically
via the `short_session_flag` field in AssembledContext and can be mitigated by
routing short sessions to the low tier or bypassing LexiContext entirely for
single-turn exchanges.
