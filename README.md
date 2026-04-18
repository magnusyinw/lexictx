# LexiContext

**A Dictionary-Based Context Orchestration Layer for LLMs & Agents**

> Stop replaying full history into the model.  
> Turn context into an addressable semantic dictionary, then inject only the minimum evidence needed.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-green)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)]()

**LexiContext separates indexing, storage, and evidence expansion for long-context systems.**

---

## What is LexiContext?

LexiContext is a context orchestration engine for LLM applications and AI agents.

Instead of replaying raw conversation history or full documents into every prompt, LexiContext turns context into an addressable semantic dictionary with three independent layers:

- **Index Layer** — lightweight feature keys for matching
- **Entry Layer** — compressed semantic representations for storage
- **Source Layer** — original evidence slices for expansion and tracing

LexiContext is not just a summary tool, and it is not a replacement for RAG.  
It is a **context orchestration layer** that sits between storage and prompt construction.

---

## The Core Innovation

Most context systems couple three things that should be independent:

```
what to store  =  what to retrieve against  =  what to return
```

In standard RAG, the same chunk acts as the storage unit, retrieval unit, and returned content.  
In summary memory, the summary plays all three roles.

**LexiContext breaks this coupling.**

```
Index Layer (how to hit)  ≠  Entry Layer (how to store)  ≠  Source Layer (what to return)
```

Like a database index:

```
B-tree index  →  row address  →  row data
feature keys  →  semantic address  →  source slice
```

This separation allows LexiContext to:

- compress aggressively without relying on compressed text for retrieval
- control expansion cost independently from match cost
- preserve traceability back to original evidence

---

## Why it exists

Long-context systems repeatedly suffer from the same failure modes:

| Problem | Cause |
|---|---|
| Repeated token spend | The same history is re-sent every turn |
| Attention dilution | Longer context weakens focus on relevant signal |
| Lossy summaries | Compression reduces cost but weakens traceability |
| Retrieval without provenance | Standard RAG retrieves text, not managed context |
| Memory corruption on update | New information can silently contradict or overwrite prior entries |

These are not isolated issues.  
They share the same root cause: context is treated as a growing text dump rather than a managed semantic asset.

---

## Architecture

```
┌─────────────────────────────────────────┐
│              Raw Input                  │
│     (dialogue / documents / history)    │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│           Semantic Distiller            │
│                                         │
│  ┌──────────────┐   ┌─────────────────┐ │
│  │ Feature      │   │ Compression     │ │
│  │ Extraction   │   │ Engine          │ │
│  │ (from source)│   │ (minimal form)  │ │
│  └──────┬───────┘   └────────┬────────┘ │
└─────────┼─────────────────────┼──────────┘
          │                     │
          ▼                     ▼
┌──────────────────┐   ┌──────────────────┐
│   Index Layer    │   │   Entry Layer    │
│                  │   │                  │
│ feature keys     │   │ compressed entry │
│ semantic address │   │ metadata         │
└────────┬─────────┘   └────────┬─────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │  Semantic Dictionary │
         └──────────┬───────────┘
                    │
         ┌──────────▼──────────┐
         │     New Query       │
         └──────────┬──────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │    Feature Matcher   │
         │  (query → addresses) │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   Budget-Aware       │
         │ Context Assembler    │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │     Source Layer     │
         │ (evidence expansion) │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   LLM / Agent Prompt │
         └──────────────────────┘
```

---

## Core Pipeline

### 1. Distill

Convert raw text into the most compact form that still preserves retrievability.

The Distiller produces two independent outputs from the same source:

**Feature signals** for the Index Layer:
- named entities and relationships
- intents, decisions, and state changes
- structural roles such as question, answer, constraint, or outcome
- semantic labels and topic markers
- source position references

**Compressed entries** for the Entry Layer:
- minimal-token semantic form
- preserved meaning with redundant phrasing removed
- always linked to source references

> Feature extraction runs on original source text, not on compressed text.  
> This is what makes compression and retrieval independent.

### 2. Index

Store Distiller output as dictionary entries:

```json
{
  "id": "entry_0042",
  "feature_keys": ["locking_strategy", "sharded_locks", "tail_latency", "Alex"],
  "semantic_address": "addr:conv_14:turn_3",
  "compressed": "Locking strategy changed: global mutex → sharded locks. Reason: tail latency under burst. Risk: deadlock in cross-shard ops. Owner: Alex.",
  "source_ref": "conv_14:turn_3:char_0:char_210",
  "metadata": {
    "timestamp": "2025-01-08T14:32:00Z",
    "role": "decision",
    "topic": "infrastructure",
    "freshness": 1.0,
    "usage_count": 0
  }
}
```

### 3. Expand on demand

When a new query arrives:

1. Match the query against feature keys and semantic addresses
2. Retrieve the matched entry set and source references
3. Expand source slices only as far as the token budget allows

| Budget | Assembled context |
|---|---|
| Low | feature keys + compressed entries only |
| Medium | compressed entries + critical source slices |
| High | full source slice expansion for matched entries |

This two-step process separates routing cost from expansion cost.

---

## Design Guarantees

LexiContext is designed around four properties.

### G1 — Compression without material loss of addressability

Feature keys are extracted from original source text, independently of compression.

This means aggressive compression should not materially reduce retrievability,
because query matching operates on source-derived features rather than compressed text.

### G2 — Every hit can trace back to original evidence

Every entry maintains a semantic address and a source reference.

- retrieval returns both the compressed entry and its source pointer
- source text can be expanded on demand
- compressed entries are never treated as primary evidence

### G3 — Writes do not silently corrupt memory

LexiContext uses a conflict-aware write protocol:

- **Update** — revise an existing entry
- **Supersede** — deprecate the old entry and link to the new one
- **Flag** — surface the conflict without auto-resolving

Silent overwrite is not permitted.

### G4 — Token savings without unacceptable quality loss

LexiContext is intended to reduce token usage in long sessions while keeping factual recall
and traceability within acceptable bounds.

Its value should be measured against:
- full-context replay
- standard chunk-based RAG

---

## How it differs from existing approaches

| Dimension | Sliding Window | Standard RAG | Summary Memory | MemGPT / Letta | LexiContext |
|---|---|---|---|---|---|
| Primary unit | raw history | chunks | summaries | memory pages | semantic entries + source slices |
| Index basis | sequential position | embedding similarity | summary similarity | recency + relevance | independent feature extraction |
| Retrieval | replay | one-hop text | one-hop summary | page in / page out | address-first, then expand |
| Token control | weak | medium | medium | strong | strong, explicit |
| Provenance | high but expensive | medium | weak | medium | strong by design |
| Compression / recall coupling | — | high | high | medium | decoupled |
| Write conflict handling | none | none | none | limited | explicit protocol |
| Long-running agents | weak | medium | medium | strong | strong |

**In one sentence:**  
RAG retrieves text. MemGPT pages memory in and out.  
LexiContext retrieves semantic addresses that unfold into text.

---

## Best use cases

LexiContext is most valuable for:

- long-running AI agents with persistent memory
- enterprise copilots over evolving document corpora
- multi-session project, research, or strategy history
- meeting and decision history with audit requirements
- high-cost reasoning systems where token efficiency matters
- systems where context decisions must be explainable and traceable

Less useful for:

- short one-off chats
- prompts with minimal context
- scenarios that require full verbatim context at all times
- ultra-low-latency paths where two-step retrieval is too expensive

---

## Example

```python
from lexi_context import ContextManager

cm = ContextManager()

cm.learn("""
Last week we changed the locking strategy from global mutex to sharded locks.
The reason was tail latency under burst traffic.
Risk noted: deadlock risk in cross-shard operations.
Owner: Alex.
""")

response = cm.chat(
    query="Why did we change the locking strategy?",
    token_budget=2000
)

print(response.answer)
print(response.source_refs)
```

Example flow:

1. Feature match → `["locking_strategy", "change", "reason"]`
2. Semantic address hit → `addr:entry_0042`
3. Compressed entry retrieved
4. Budget permits source expansion
5. LLM answers with linked evidence

---

## Status

> LexiContext v0.1 is available as an open-source Python library.
>
> Core pipeline (Distiller, Matcher, Assembler, ConflictWriter) is implemented
> and validated. Benchmark results: 85.1% token savings, 100% retrieval recall
> across 50 queries and 10 domains.
---

## Roadmap

- [x] Reference Distiller with fidelity benchmark harness
- [x] Independent feature extraction pipeline
- [x] In-memory Semantic Dictionary backend
- [x] Address-first retrieval pipeline
- [x] Budget-Aware Context Assembler
- [x] Conflict-aware write protocol
- [ ] Persistent storage adapters
- [ ] Optional vector-augmented indexing
- [ ] MCP server interface for agent integration
- [ ] Long-session benchmark suite

---

## Contributing

Issues, PRs, and design feedback are welcome.

Especially useful areas of feedback:

- Distiller design and fidelity evaluation
- real-world token budget patterns
- write conflict patterns in long-running memory systems

---

## License

MIT
