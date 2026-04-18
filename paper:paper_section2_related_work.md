# Section 2: Related Work

Long-context management for LLM systems has been approached from several
directions. We organize prior work into four categories and describe how
LexiContext relates to each.

---

## 2.1 Context Window Extension

A natural response to the context management problem is to extend the model's
context window itself. Recent work has pushed context limits from 4K to 128K
tokens (GPT-4 Turbo, Claude 3), and beyond to 1M tokens (Gemini 1.5 Pro,
Kimi). In principle, a sufficiently large context window eliminates the need
for external memory management.

In practice, two problems persist. First, processing cost scales quadratically
with sequence length for attention-based architectures, making million-token
contexts economically prohibitive for sustained multi-session interactions.
Second, and more fundamentally, long-context LLMs exhibit the "lost-in-the-
middle" phenomenon [Liu et al., 2024]: models reliably attend to information
near the beginning and end of context but systematically underweight content
in the middle. Wu et al. [2024] report 30-60% accuracy drops on LongMemEval
when models are required to reason over full 115K-token histories without
targeted retrieval, even when the answer is present in the context.

LexiContext does not replace large context windows. It operates as an
orchestration layer that determines which evidence is worth injecting into
a context window of any size, addressing the attention dilution problem that
persists regardless of window capacity.

---

## 2.2 Retrieval-Augmented Generation (RAG)

Retrieval-augmented generation [Lewis et al., 2020] retrieves relevant
documents or passages at query time and injects them into the prompt. Standard
RAG systems represent content as fixed-size chunks, embed them using dense
retrieval models, and select top-k chunks by cosine similarity.

Several extensions have been proposed for conversational and multi-document
settings. Parent Document Retrieval retrieves small chunks for matching but
expands to larger parent documents for injection. Hierarchical RAG maintains
summaries at multiple granularities. Contextual Retrieval [Anthropic, 2024]
prepends chunk-level context to improve embedding quality.

LexiContext differs from RAG systems in two structural ways. First, the Index
Layer in LexiContext is built from explicitly extracted semantic signals
(feature keys, entity names, structural roles, temporal references) rather
than from dense embeddings of compressed or chunked text. This makes the
retrieval surface interpretable and auditable. Second, LexiContext performs
two-hop retrieval: the first hop matches against lightweight feature keys
(cheap), and the second hop expands source evidence only within the available
token budget (gated). Standard RAG collapses these into a single operation,
with no mechanism for budget-aware expansion depth.

The closest RAG variant to LexiContext is the fact-augmented key expansion
approach described in Wu et al. [2024], which augments retrieval keys with
extracted facts to improve index quality. LexiContext generalizes this insight
into a full architectural principle: the index is always derived from original
source text and is never a derivative of the compressed representation.

---

## 2.3 Memory-Augmented LLM Systems

Several systems have been proposed specifically for persistent agent memory
across sessions.

**MemGPT / Letta** [Packer et al., 2023] introduces a hierarchical memory
system with main context (in-window), external storage, and recursive
summarization. Memory is paged in and out of context using LLM-generated
function calls. MemGPT's core contribution is making memory management an
explicit part of the LLM's reasoning loop. LexiContext differs in that memory
management is handled externally by the orchestration layer rather than
delegated to the LLM, keeping the LLM's context clean and reducing the
reasoning overhead of self-directed memory operations.

**Mem0** [Mem0 AI, 2024] extracts user facts and preferences from
conversations and stores them in a vector database for retrieval. Like
LexiContext, Mem0 compresses raw dialogue into structured facts. However,
Mem0's compressed facts are also the primary retrieval surface: retrieval is
performed over compressed summaries, not over a separate index derived from
source text. This means compression quality directly bounds retrieval recall.
LexiContext addresses this by maintaining an independent Index Layer extracted
before compression.

**Zep** [Zep AI, 2024] maintains a temporal knowledge graph of user
information updated across sessions. Zep's graph structure enables temporal
reasoning about when facts were introduced or updated. LexiContext achieves
a simpler form of this through the conflict-aware write protocol (SUPERSEDE
with deprecated pointers), without requiring a full graph data structure.

**A-MEM** [Xu et al., 2024] proposes an Aho-Corasick based memory indexing
system for agentic LLM interactions, focusing on efficient string matching
over accumulated memory. LexiContext's BM25-based feature matching is
conceptually related but operates over extracted semantic keys rather than
raw text, reducing false positive matches from surface-level string overlap.

None of the above systems explicitly decouples the index from the compressed
entry from the source slice as three independent layers, nor do they implement
budget-aware two-hop expansion as a first-class architectural feature.

---

## 2.4 Long-Term Conversational Memory Benchmarks

**LongMemEval** [Wu et al., 2024] is the most rigorous existing benchmark for
evaluating long-term memory in LLM-based chat assistants. It contains 500
human-curated questions testing five abilities: information extraction,
multi-session reasoning, temporal reasoning, knowledge updates, and abstention.
LongMemEval-S configures chat histories at approximately 115K tokens per
question; LongMemEval-M scales to 500 sessions (~1.5M tokens). The benchmark
finds that commercial systems and long-context LLMs show 30-60% accuracy drops
relative to oracle retrieval conditions.

Wu et al. also propose a unified three-stage memory framework (indexing,
retrieval, reading) and identify four control points: value granularity,
key expansion, query expansion, and reading strategy. LexiContext can be
understood as an instantiation of this framework with specific design choices:
session-level granularity with turn-level compression, source-derived feature
keys for index expansion, and budget-gated source expansion for reading.

**LoCoMo** [Maharana et al., 2024] evaluates very long-term conversational
memory over up to 35 sessions and 300 turns, testing question answering,
event summarization, and multi-modal dialogue generation. LoCoMo focuses on
human-human conversations, whereas LexiContext is designed specifically for
human-assistant interaction patterns where the assistant's prior responses
are also part of the retrievable history.

---

## 2.5 Positioning

Table 4 summarizes how LexiContext compares to the most relevant prior work
across five dimensions.

| System | Index basis | Compression/recall coupling | Budget control | Write conflict | Provenance |
|---|---|---|---|---|---|
| Standard RAG | embedding similarity | high | weak (top-k) | none | medium |
| MemGPT / Letta | LLM-directed paging | medium | strong (implicit) | weak | medium |
| Mem0 | compressed fact embeddings | high | medium | none | weak |
| Zep | temporal knowledge graph | low | medium | partial | strong |
| **LexiContext** | **source-derived feature keys** | **decoupled** | **strong (explicit tiers)** | **explicit protocol** | **strong** |

Table 4: Comparison of LexiContext with related systems across five design
dimensions. "Compression/recall coupling" refers to the degree to which
retrieval quality degrades as compression ratio increases.

The key differentiator is the decoupled architecture: LexiContext is the first
system, to our knowledge, to treat the retrieval index, the compressed storage
entry, and the source evidence slice as three independently maintained
representations. This decoupling is what enables LexiContext to achieve high
compression ratios (85.6% average) without degrading retrieval recall, a
property that is structurally impossible in systems where the index is derived
from the compressed output.
