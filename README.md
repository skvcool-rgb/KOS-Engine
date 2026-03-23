# KOS - Knowledge Operating System

**A neurosymbolic engine that confines LLMs to thin I/O while routing all reasoning through deterministic graph physics. Zero hallucination. Provenance on every answer.**

[![Tests](https://img.shields.io/badge/tests-16%2F16%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## The Problem with RAG

| Failure Mode | Standard RAG | KOS |
|---|---|---|
| Hallucination | LLM invents facts from context | LLM never reasons - only reads 1-2 scored sentences |
| Lost in the Middle | 30%+ accuracy drop for mid-context facts | Weaver feeds exactly 1-2 sentences to LLM |
| Typo failure | "prpvskittes" returns nothing | 6-layer cascade rescues it to "perovskite" |
| Multi-hop reasoning | Cannot chain A->B->C | Spreading activation with spatial decay |
| Non-deterministic | Same query, different answers | Intent scoring formula: same input = same ranking |
| Math hallucination | "345M * 0.0825 = 28.4M" (wrong) | SymPy exact: 28,462,500.0000000 |

## Architecture

```
User Query
    |
[Pre-LLM Raw Scan] --- catches typos before LLM normalizes them
    |
[LLM Ear] --- JSON keyword extraction only (never reasoning)
    |
[6-Layer Resolution] --- exact > fuzzy > metaphone > soundex > hypernym > vector
    |
[Spreading Activation] --- dual-state physics, myelination, Top-K 500, 15-tick beam
    |
[Algorithmic Weaver] --- deterministic intent scoring (+40 WHERE/WHEN/WHO)
    |
[LLM Mouth] --- reads 1-2 precision-scored sentences, synthesizes answer
```

### The 6-Layer Word Resolution Cascade

Every user word passes through up to 6 resolution layers before giving up:

| Layer | Algorithm | What It Catches | Speed |
|---|---|---|---|
| 1 | Exact lexicon lookup | Known words | O(1) |
| 2 | Difflib fuzzy match (>=0.6) | Minor typos | O(W log W) |
| 3a | Metaphone phonetic hash | Sound-alike words | O(1) |
| 3b | Soundex phonetic hash | Broader phonetic net | O(1) |
| 4 | WordNet hypernym/hyponym walk | Taxonomy mismatch ("entities" -> "company") | O(S) |
| 5 | Semantic vector (all-MiniLM-L6-v2) | Concept synonyms ("metropolis" -> "city") | O(N) |

### The Physics Engine

Each concept node has **dual state** inspired by biological neurons:

- **Activation** (epistemic truth): accumulates, represents semantic relevance
- **Fuel** (mechanical spike): drains to zero after firing, prevents fan-out explosion
- **Myelination**: edges strengthen when used (Hebbian: "neurons that fire together wire together")
- **Temporal decay**: stale knowledge exponentially fades (0.7^ticks)

```python
# Hyperpolarization Gate: fuel only accumulates when BOTH conditions met
if incoming_energy > 0 and activation > 0:
    fuel += incoming_energy  # Node can propagate
```

### The Algorithmic Weaver

Deterministic evidence scoring - no probabilistic ranking:

```
sentence_score =
    + 40  (if WHERE intent: sentence contains "in", "located", "province")
    + 40  (if WHEN intent: sentence contains a 4-digit year)
    + 40  (if WHO intent: sentence contains "established", "founded", "named")
    - 50  (if sports noise detected and not a sports query)
    + 20 x (keyword overlap count)
```

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/kos-engine.git
cd kos-engine

# Install
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('wordnet')"

# Set your OpenAI key (used ONLY for keyword extraction + 1-sentence synthesis)
export OPENAI_API_KEY=sk-...

# Run tests
python tests/master_smoke_test.py
python tests/test_unification.py

# Launch the UI
streamlit run app.py
```

## Benchmark Results

### 10-Point Smoke Test (master_smoke_test.py)

| # | Test | What It Proves | Result |
|---|------|---|---|
| 1 | SVO Slider Precision | Drug interaction: apixaban vs warfarin | PASS |
| 2 | Split-Antecedent Coref | "They" resolves to 2 entities | PASS |
| 3 | Extreme Typo Phonetics | "prpvskittes" -> perovskite | PASS |
| 4 | Spanglish Filtering | Spanish query, English graph | PASS |
| 5 | Synonym Networks | "solar cells" -> "photovoltaic cells" | PASS |
| 6 | Ambiguity Clarification | "silcon" disambiguated | PASS |
| 7 | Multi-Hop Deduction | perovskite -> cell -> photon -> electricity | PASS |
| 8 | Big Arithmetic | 345,000,000 * 0.0825 = 28,462,500 (exact) | PASS |
| 9 | Calculus Integration | integral(x^3 * log(x)) = x^4*log(x)/4 - x^4/16 | PASS |
| 10 | Calculus Differentiation | d/dx[e^x * cos(x) * sin(x)] (exact product rule) | PASS |

### Unification Test (test_unification.py)

| Test | Query | Output |
|---|---|---|
| Daemon Survival | (hub node survives maintenance) | PASS - 24 edges intact |
| WHERE intent | "Where is Toronto located?" | Ontario, Canada |
| WHEN intent | "When was Toronto founded?" | 1834 |
| WHO intent | "Who established Toronto?" | John Graves Simcoe |
| WHAT intent | "Population of Toronto?" | 2.7 million |
| Layer 5 Vector | "Tell me about the metropolis" | Toronto info via semantic match |

## How KOS Compares

| | Standard RAG | GraphRAG | Neo4j | Vector DBs | **KOS** |
|---|---|---|---|---|---|
| Typo handling | None | None | None | None | **6-layer cascade** |
| Multi-hop | None | LLM-dependent | Cypher queries | None | **Spreading activation** |
| Lost in Middle | Suffers | Partial fix | N/A | N/A | **Eliminated** |
| Hallucination | High | Medium | Low | N/A | **Near-zero** |
| Evidence scoring | Cosine (opaque) | LLM (probabilistic) | None | Cosine | **Deterministic formula** |
| Learning | Static | Static | Static | Static | **Myelination** |
| Math | LLM guesses | LLM guesses | N/A | N/A | **SymPy exact** |

## Project Structure

```
kos-engine/
  kos/
    __init__.py          # Package exports
    node.py              # ConceptNode dual-state physics
    graph.py             # KOSKernel spreading activation
    lexicon.py           # Semantic DNS + phonetic indexing
    router.py            # KOSShell 6-layer cascade + pipeline
    weaver.py            # Intent scoring + evidence selection
    daemon.py            # O(V+E) maintenance (prune, merge, infer)
    drivers/
      text.py            # TextDriver SVO extraction + coreference
      math.py            # MathDriver SymPy zero-hallucination
      ast.py             # ASTDriver Python code parsing
      vision.py          # VisionDriver YOLO spatial topology
  kos_core_v4.py         # V4 kernel + KASM compiler + Z3 prover
  app.py                 # Streamlit web UI
  tests/
    master_smoke_test.py  # 10-scenario benchmark
    test_unification.py   # V4 verification + Layer 5
  requirements.txt
  setup.py
```

## Key Design Decisions

1. **LLM as thin I/O only**: The LLM extracts keywords (Ear) and synthesizes 1-sentence answers (Mouth). It never reasons, never sees more than 2 evidence sentences, and never generates facts.

2. **Node labels vectorized, not documents**: Layer 5 embeds only short concept labels (~1-3 words), not document chunks. This eliminates the core RAG hallucination vector (lossy paragraph embeddings).

3. **Fuel prevents fan-out**: Classic spreading activation (Collins & Loftus, 1975) has no resource constraint. KOS fuel drains to zero after firing, bounding the search space without arbitrary cutoffs.

4. **Myelination = self-optimizing retrieval**: Frequently-traversed edges get stronger. The graph learns which connections matter, unlike static knowledge graphs.

## License

MIT License - see LICENSE file.

## Citation

If you use KOS in research, please cite:

```bibtex
@software{kos2026,
  title={KOS: A Fuel-Constrained Spreading Activation Engine for Zero-Hallucination Knowledge Retrieval},
  author={Suraj},
  year={2026},
  url={https://github.com/YOUR_USERNAME/kos-engine}
}
```
