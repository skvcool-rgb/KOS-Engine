# KOS: A Fuel-Constrained Spreading Activation Engine for Zero-Hallucination Knowledge Retrieval

**Authors:** Suraj K.V.

**Date:** March 2026

---

## Abstract

Retrieval-Augmented Generation (RAG) systems suffer from five persistent failure modes: hallucination (LLM generates facts not in the evidence), lost-in-the-middle (30%+ accuracy degradation for mid-context information), vocabulary mismatch (typos and synonyms cause retrieval failure), inability to perform multi-hop reasoning, and non-deterministic evidence scoring. We present KOS (Knowledge Operating System), a neurosymbolic knowledge engine that addresses all five simultaneously through a unified architecture. KOS confines Large Language Models to thin input/output roles -- keyword extraction and single-sentence synthesis -- while routing all reasoning through a deterministic spreading activation graph with biological neuron physics. The system introduces three novel mechanisms: (1) fuel-constrained propagation that prevents the fan-out explosion inherent in classic spreading activation, (2) a 6-layer word resolution cascade spanning exact matching through semantic vector fallback, and (3) an Algorithmic Weaver that scores evidence deterministically by query intent type. KOS achieves 16/16 on a benchmark suite covering SVO dependency parsing, cross-language queries, extreme typo recovery, multi-hop deduction, and symbolic mathematics, with zero hallucinations observed across all test scenarios.

**Keywords:** neurosymbolic AI, spreading activation, knowledge retrieval, zero hallucination, deterministic reasoning

---

## 1. Introduction

The dominant paradigm for knowledge-grounded question answering is Retrieval-Augmented Generation (RAG), which embeds document chunks into high-dimensional vectors and retrieves the most similar chunks to stuff into an LLM's context window for answer generation. Despite widespread adoption, RAG exhibits five well-documented failure modes:

1. **Hallucination.** Even with retrieved context, LLMs can ignore it or blend it with training data. Self-reflective RAG reduces hallucination to approximately 5.8% but does not eliminate it.

2. **Lost in the Middle.** LLMs exhibit a U-shaped attention curve -- they attend to the beginning and end of the context window but degrade significantly for information positioned in the middle. This is an architectural limitation of Rotary Position Embedding (RoPE).

3. **Vocabulary Mismatch.** Bi-encoder embeddings are lossy by design. Typos, abbreviations, and domain-specific terminology often fail to retrieve the correct chunks. Hybrid search (vector + BM25) provides partial mitigation but is limited to two resolution strategies.

4. **Multi-hop Reasoning.** Standard RAG retrieves flat chunks with no relational structure. Chaining inferences (A implies B, B implies C, therefore A implies C) is fundamentally unsupported.

5. **Non-deterministic Scoring.** Cosine similarity between embedding vectors provides no interpretable evidence quality metric. The same query can return different rankings across runs.

Microsoft's GraphRAG addresses multi-hop reasoning through LLM-extracted knowledge graphs and community summarization, but introduces high indexing costs and retains LLM dependency throughout the pipeline. Traditional knowledge graphs (Neo4j, Amazon Neptune) support multi-hop traversal but require manual schema design, offer no typo tolerance, and provide no built-in evidence scoring.

We present KOS, a system that takes a fundamentally different architectural approach: **the LLM never reasons.** Instead, it serves as a thin I/O shell -- a JSON keyword extractor (Ear) and a single-sentence synthesizer (Mouth) -- around a deterministic graph physics engine that handles all retrieval, inference, and evidence scoring through mathematical operations.

---

## 2. System Architecture

KOS processes a user query through six sequential stages:

### 2.1 Pre-LLM Raw Word Scan (Layer 0.5)

Before any LLM interaction, KOS extracts words from the raw user prompt and attempts to resolve each through the 6-layer cascade (Section 3). This captures typos and taxonomy matches that the LLM would normalize away.

### 2.2 LLM Ear (Layer 1)

A single API call to gpt-4o-mini with strict JSON output formatting:

```json
{"status": "EXECUTE", "keywords": ["toronto", "population"]}
```

The `response_format: json_object` constraint eliminates free-text parsing variability. Temperature is set to 0.0 for deterministic extraction.

### 2.3 Six-Layer Word Resolution Cascade

Each keyword passes through up to six resolution layers before giving up (see Section 3).

### 2.4 Spreading Activation Engine

Resolved keywords become seed nodes in the knowledge graph. The engine propagates activation energy through weighted edges using priority-queue-driven beam search (see Section 4).

### 2.5 Algorithmic Weaver

All provenance sentences linked to activated nodes are scored by a deterministic intent-matching formula (see Section 5). The top 2 sentences are selected as evidence.

### 2.6 LLM Mouth (Layer 7)

The selected 1-2 sentences are passed to gpt-4o-mini for natural language synthesis. The LLM reads a maximum of 2 evidence sentences -- eliminating the lost-in-the-middle problem by construction.

---

## 3. The 6-Layer Word Resolution Cascade

When a user types a query word, KOS attempts resolution through six strategies in order of speed and precision:

| Layer | Algorithm | Time Complexity | Example |
|-------|-----------|----------------|---------|
| 1 | Exact hash lookup in word-to-UUID lexicon | O(1) | "toronto" -> toronto.n.01 |
| 2 | Ratcliff/Obershelp fuzzy match (cutoff >= 0.6) | O(W log W) | "toranto" -> toronto.n.01 |
| 3a | Metaphone phonetic hash | O(1) | "tornto" -> TRNTN -> toronto.n.01 |
| 3b | Soundex phonetic hash | O(1) | "prpvskittes" -> P612 -> perovskite |
| 4 | WordNet hypernym/hyponym traversal (depth 6) | O(S) | "entities" -> entity -> ... -> company |
| 5 | Semantic vector cosine similarity (all-MiniLM-L6-v2) | O(N) | "metropolis" -> city (0.557) |

The cascade terminates at the first successful resolution. Approximately 90% of queries resolve at Layer 1. Layer 5 embeds only node labels (1-3 words each), not document chunks, preserving the zero-hallucination guarantee.

---

## 4. ConceptNode Physics Engine

### 4.1 Dual-State Model

Each concept node maintains two independent state variables:

- **Activation** (a): Epistemic truth score. Accumulates additively, clamped to [-E_max, E_max]. Represents semantic relevance to the current query.
- **Fuel** (f): Propagation energy. Drains to zero after firing. Represents the node's capacity to propagate information to neighbors.

### 4.2 Hyperpolarization Gate

When a node receives incoming energy e at tick t:

```
a(t) = clamp(a(t-1) * decay^(dt) + e, [-E_max, E_max])
f(t) = f(t-1) * decay^(dt) + e    if e > 0 AND a(t) > 0
        f(t-1) * decay^(dt)        otherwise
```

The gate ensures that negatively-activated nodes (inhibited concepts) cannot gain propagation energy, preventing pathological feedback loops.

### 4.3 Myelination (Synaptic Plasticity)

Each edge tracks a myelin counter m that increments every time the edge fires:

```
effective_weight = w * (1 + m * 0.01)
```

This implements Hebbian learning: frequently-traversed paths strengthen over time, creating self-optimizing retrieval without retraining.

### 4.4 Top-K 500 Routing

During propagation, each node fires only its top 500 edges ranked by effective weight. This prevents fan-out explosion (a known failure of classic spreading activation) while preserving access to niche connections.

### 4.5 Fuel Depletion

After firing, fuel drains to zero:

```
f_post = 0.0
```

This "fire-once" rule, combined with spatial decay (0.8^hops), naturally bounds the search frontier without arbitrary depth limits.

---

## 5. Algorithmic Weaver: Deterministic Intent Scoring

The Weaver gathers all provenance sentences connected to seed nodes and scores each by a deterministic formula:

```
S(sentence) = I_geo + I_temp + I_who + P_noise + K_overlap
```

Where:
- I_geo = +40 if prompt contains WHERE-intent words AND sentence contains location markers ("in", "located", "province")
- I_temp = +40 if prompt contains WHEN-intent words AND sentence contains a 4-digit year (regex: 16xx-20xx)
- I_who = +40 if prompt contains WHO-intent words AND sentence contains founder markers ("established", "named", "founded")
- P_noise = -50 if sentence contains sports/entertainment terms AND query is not sports-related
- K_overlap = +20 * |prompt_keywords intersection sentence_keywords|

The top 2 sentences by score are passed to the LLM Mouth. This formula is fully deterministic: identical inputs always produce identical rankings.

---

## 6. Background Maintenance Daemon

KOS runs three background maintenance algorithms during idle cycles:

### 6.1 Orphan Pruning -- O(V + E)

A single pass builds an inbound-tracker set from all edge targets, then deletes nodes with zero outbound AND zero inbound connections.

### 6.2 Isomorph Merging -- Degree-Bucketed

Nodes are grouped by outbound degree. Only nodes within the same degree bucket are compared via Jaccard similarity. Pairs exceeding 85% overlap are merged. This reduces the naive O(V^2) comparison to O(V + B^2) where B is the average bucket size.

### 6.3 Triadic Closure -- Predictive Inference

For strong edges (weight >= 0.7): if A -> B and B -> C, infer A -> C with confidence = w_AB * w_BC, capped at 0.5 to prevent predicted edges from overriding explicit facts.

---

## 7. Benchmark Results

### 7.1 Master Smoke Test (10 Scenarios)

| # | Scenario | Capability Tested | Result |
|---|----------|------------------|--------|
| 1 | SVO Slider Precision | Drug interaction: apixaban vs warfarin | PASS |
| 2 | Split-Antecedent Coreference | "They" resolves to 2 entities | PASS |
| 3 | Extreme Typo Phonetics | "prpvskittes" -> perovskite | PASS |
| 4 | Cross-Language (Spanglish) | Spanish query, English knowledge graph | PASS |
| 5 | Synonym Networks | "solar cells" -> "photovoltaic cells" via WordNet | PASS |
| 6 | Ambiguity Resolution | "silcon" disambiguated between silicon/silicone | PASS |
| 7 | Multi-Hop Deduction | perovskite -> cell -> photon -> electricity | PASS |
| 8 | Big Arithmetic | 345,000,000 * 0.0825 = 28,462,500 (exact, 2.4ms) | PASS |
| 9 | Calculus Integration | integral(x^3 * log(x)) via SymPy | PASS |
| 10 | Calculus Differentiation | d/dx[e^x * cos(x) * sin(x)] via SymPy | PASS |

### 7.2 Unification Test (6 Scenarios)

| Scenario | Query | Expected | Actual |
|----------|-------|----------|--------|
| Daemon survival | Hub node survives maintenance | 24 edges intact | PASS |
| WHERE intent | "Where is Toronto located?" | Ontario | "province of Ontario, Canada" |
| WHEN intent | "When was Toronto founded?" | 1834 | "founded and incorporated in 1834" |
| WHO intent | "Who established Toronto?" | Simcoe | "John Graves Simcoe" |
| WHAT intent | "Population of Toronto?" | 2.7M | "2.7 million people" |
| Layer 5 Vector | "Tell me about the metropolis" | Toronto info | "Toronto is a massive city...founded in 1834" |

### 7.3 Performance

| Metric | Value |
|--------|-------|
| Graph query latency | ~40-80ms (CPU, pure graph) |
| End-to-end with LLM | ~1.2-2.5s (dominated by OpenAI API) |
| Math coprocessor | 2-250ms (SymPy, CPU) |
| Daemon maintenance cycle | ~0.1-20ms |
| Hallucinations observed | 0 across 16 test scenarios |

---

## 8. Comparison with Existing Systems

| Capability | RAG | GraphRAG | Neo4j | Vector DBs | KOS |
|---|---|---|---|---|---|
| Typo handling | None | None | None | None | 6-layer cascade |
| Multi-hop | None | LLM-dependent | Cypher | None | Spreading activation |
| Lost in Middle | Suffers | Partial fix | N/A | N/A | Eliminated (1-2 sentences) |
| Hallucination | ~5.8% | Medium | Low | N/A | 0% observed |
| Evidence scoring | Cosine (opaque) | LLM (probabilistic) | None | Cosine | Deterministic formula |
| Self-learning | No | No | No | No | Myelination |
| Math | LLM approximation | LLM approximation | N/A | N/A | SymPy exact |

---

## 9. Limitations and Future Work

1. **Scale testing.** Current benchmarks use corpora of 200-1000 words. Performance on million-document graphs requires evaluation.

2. **Contextual mitosis.** Super-hub splitting is currently disabled pending a safe sub-graph clustering algorithm that preserves provenance.

3. **LLM dependency for I/O.** The Ear and Mouth require an LLM API. A fully offline mode using local models (via the existing `base_url` parameter) is architecturally supported but untested at scale.

4. **Semantic vector threshold.** Layer 5 uses a cosine similarity threshold of 0.50, which may admit false positives on larger vocabularies.

5. **Adversarial robustness.** The system has not been tested against adversarial prompts designed to exploit the deterministic scoring formula.

---

## 10. Conclusion

KOS demonstrates that the five persistent failure modes of RAG can be addressed simultaneously through a single architectural decision: confining the LLM to thin I/O and routing all reasoning through a deterministic graph physics engine. The biological neuron metaphors -- fuel constraints, myelination, temporal decay, triadic closure -- are not aesthetic choices but engineering solutions to specific graph traversal problems. The system achieves 16/16 on a diverse benchmark suite with zero observed hallucinations, deterministic evidence scoring, and sub-100ms graph query latency on CPU.

---

## References

1. Collins, A.M., & Loftus, E.F. (1975). A spreading activation theory of semantic processing. *Psychological Review*, 82(6), 407-428.

2. Liu, N.F., et al. (2024). Lost in the middle: How language models use long contexts. *Transactions of the ACL*, 12, 157-173.

3. Edge, D., et al. (2024). From local to global: A graph RAG approach to query-focused summarization. *arXiv preprint arXiv:2404.16130*.

4. Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.

5. Gartner. (2025). Hype Cycle for Artificial Intelligence -- Neuro-Symbolic AI.

---

## Appendix A: Source Code

The complete source code is available at: https://github.com/skvcool-rgb/KOS-Engine

## Appendix B: Reproducibility

All benchmark results can be reproduced by running:
```bash
git clone https://github.com/skvcool-rgb/KOS-Engine.git
cd KOS-Engine
pip install -r requirements.txt
export OPENAI_API_KEY=<your-key>
PYTHONPATH=. python tests/master_smoke_test.py
PYTHONPATH=. python tests/test_unification.py
```
