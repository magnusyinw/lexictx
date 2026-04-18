# LexiContext: Decoupled Indexing, Compression, and Evidence Expansion
# for Long-Context LLM Systems

---

## Abstract

Long-context LLM systems commonly suffer from three coupled failure modes:
repeated token expenditure from full-history replay, retrieval quality
degradation under aggressive compression, and loss of traceability when
summaries replace original evidence. These failures share a common root cause:
existing approaches conflate indexing, storage, and content retrieval into a
single representation, forcing a tradeoff between compression depth and
retrieval quality.

We present LexiContext, a context orchestration layer that breaks this
coupling through a three-layer architecture: an Index Layer of feature keys
extracted from original source text before compression, an Entry Layer of
compressed semantic representations, and a Source Layer of original evidence
slices expanded on demand. Inspired by database index design, LexiContext
separates the routing cost of retrieval from the expansion cost of content
delivery, enabling precise token budget control through a two-hop retrieval
mechanism: query matching against lightweight feature keys (first hop), followed
by budget-gated source expansion (second hop).

We evaluate LexiContext across three domain scenarios covering 19 sessions and
validate four design guarantees experimentally. At an average compression ratio
of 85.6%, feature recall remains at 100%, confirming that compression depth
and retrieval quality are decoupled when indexing operates on source text
rather than compressed output. Against a full-context replay baseline,
LexiContext reduces prompt token consumption by an average of 72.9% while
maintaining complete source traceability and explicit conflict detection on
writes. LexiContext is released as an open-source Python library.

---

## 1. Introduction

Every turn of a long-running LLM conversation carries a hidden cost: the
system must decide how much of its history to include in the next prompt.
The dominant approaches — full-context replay, sliding window truncation,
retrieval-augmented generation (RAG), and summary memory — each solve part
of this problem while introducing their own failure modes.

Full-context replay preserves all information but scales linearly with
session length, making it economically infeasible for sustained interactions.
Sliding window truncation is cheap but discards potentially critical early
context without regard for relevance. Standard RAG retrieves text chunks by
embedding similarity, but this couples retrieval quality to chunk granularity
and cannot adapt content depth to available token budgets. Summary memory
reduces storage cost but sacrifices traceability: once original evidence is
compressed into a summary, it cannot be recovered, and contradictions between
old and new summaries are commonly silently overwritten.

A less-examined commonality underlies all of these approaches: they treat
indexing, storage, and content retrieval as a single operation. In RAG, the
chunk that is indexed is also the chunk that is stored and returned. In summary
memory, the summary serves simultaneously as the retrieval surface and the
delivered content. This conflation creates an inescapable tradeoff: the more
aggressively a system compresses, the less faithfully its index represents the
original content, and the weaker its retrieval becomes.

We observe that this tradeoff is not fundamental — it is architectural. Database
systems resolved the analogous problem decades ago by separating indexes from
data: a B-tree index points to row addresses, and rows are fetched only when
needed. The index is constructed from the full data at write time, not derived
from a compressed representation. Retrieval quality is therefore independent
of storage optimization.

LexiContext applies this principle to LLM context management. Instead of a
single representation, it maintains three independent layers:

- **Index Layer**: Feature keys and semantic addresses extracted from original
  source text at ingest time, before any compression is applied.
- **Entry Layer**: Compressed semantic representations for storage efficiency,
  each linked to a stable source reference.
- **Source Layer**: Original evidence slices, retrieved on demand within
  the available token budget.

This separation produces two structural advantages. First, because the Index
Layer is derived from original source text rather than compressed output,
retrieval recall is not bounded by compression quality — a query term that did
not survive compression can still match a feature key extracted from the
original. Second, because retrieval (first hop) and content delivery (second
hop) are separate operations, the system can match against the full index at
negligible cost and expand source evidence only as far as the token budget
permits.

Beyond retrieval, LexiContext addresses a practical failure mode of long-running
memory systems: write corruption. When new information conflicts with, updates,
or supersedes existing entries, naive systems either silently overwrite or
blindly append, accumulating contradictions over time. LexiContext implements
an explicit conflict-aware write protocol that classifies every write as a
clean insert, an in-place update, a supersede (deprecating the old entry with
an audit pointer), or a flagged conflict requiring human review. Silent
overwrite is never permitted.

We evaluate LexiContext through four design guarantees:

- **G1**: Compression without material loss of retrieval addressability
- **G2**: Every retrieved result traces back to original source evidence
- **G3**: Writes cannot silently corrupt existing memory
- **G4**: Token savings in real conversations without significant quality loss

Experimental results across three domain scenarios confirm that all four
guarantees hold under the described architecture. At 85.6% average compression,
feature recall remains 100%. Against full-context replay, average token
reduction is 72.9%. All retrieved entries maintain source references, and all
detected write conflicts are resolved explicitly.

The remainder of this paper is organized as follows. Section 2 reviews related
work on LLM context management and memory systems. Section 3 describes the
LexiContext architecture in detail. Section 4 formalizes the four design
guarantees. Section 5 presents experimental evaluation. Section 6 concludes
with directions for future work.

**Contributions.** This paper makes the following contributions:

1. A three-layer context representation that decouples indexing, storage, and
   evidence expansion, enabling compression depth and retrieval quality to be
   controlled independently.

2. A two-hop retrieval mechanism that separates routing cost (feature matching)
   from expansion cost (source delivery), enabling precise token budget control.

3. A conflict-aware write protocol that prevents silent memory corruption in
   long-running agent systems, with explicit audit trail preservation.

4. An open-source Python implementation (LexiContext) with a minimal dependency
   footprint, validated against four design guarantees through controlled
   experiments.
