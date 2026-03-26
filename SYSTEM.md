# KOS Engine — Knowledge Operating System

**Version:** 0.8.0 (Adaptive Mission System)
**Architecture:** Agent-Routed 5-Stage Pipeline with Verifier Layer, Episodic Memory, and Mission Manager
**Language:** Python 3.14 | Optional Rust backend via PyO3
**Total Source:** ~31,000 lines across 104+ modules
**Last tested:** 2026-03-26 — v0.8 mission system verified (4/4 goals complete, avg 0.798)

---

## What is KOS?

KOS (Knowledge Operating System) is a **neurosymbolic reasoning engine** that routes all reasoning through a deterministic knowledge graph instead of relying on LLM token generation. It uses biological neuron physics (spreading activation, hyperpolarization gates, Hebbian learning) to answer questions with traceable evidence chains.

**Core principle:** LLMs are confined to thin I/O (text parsing, embedding). All reasoning happens in the graph — zero hallucination, deterministic scoring, full provenance.

---

## Architecture Overview

```
                    ┌─────────────────────────────────┐
                    │       Mission Control Dashboard   │
                    │   (static/dashboard.html :8080)   │
                    └──────────────┬──────────────────┘
                                   │ REST API
                    ┌──────────────▼──────────────────┐
                    │          api.py (FastAPI)         │
                    │   uvicorn single-worker server    │
                    └──────────────┬──────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
    ┌─────▼─────┐           ┌─────▼─────┐           ┌─────▼──────┐
    │  Router    │           │  Weaver   │           │  Forager   │
    │ (Offline)  │           │ (Scoring) │           │ (Internet) │
    └─────┬─────┘           └─────┬─────┘           └─────┬──────┘
          │                        │                        │
    ┌─────▼────────────────────────▼────────────────────────▼─────┐
    │                    KOS Kernel (Graph)                         │
    │  ConceptNodes + Weighted Edges + Spreading Activation        │
    │  Optional: RustKernel via PyO3 (10-50x faster)               │
    └──────────────────────────┬───────────────────────────────────┘
          │            │            │            │            │
    ┌─────▼───┐  ┌────▼────┐ ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
    │ Lexicon │  │TextDrvr │ │PhysDrvr │ │ChemDrvr │ │MathDrvr │
    │  (DNS)  │  │  (SVO)  │ │(F=ma)   │ │(PTable) │ │(SymPy)  │
    └─────────┘  └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

---

## v0.7 Query Pipeline (Agent-Routed + Verified)

```
Query
  |
  v
[AGENT ROUTER] -----> FAST PATH (simple factual, definitions)
  |                     |
  |                     v
  |              [math?] --yes--> DETERMINISTIC SOLVER (<200ms, exact, trust=unverified)
  |                     |
  |                     no
  |                     v
  +--> AGENTIC PATH     |
  |    (compare,        v
  |     multi-step)  [Stage 1: RETRIEVE] spreading activation
  |                     |
  |                     v
  |                  [Stage 2: RERANK] 7 signals + 2 penalties
  |                     |
  |                     v
  |                  [Stage 3: SYNTHESIZE] weaver + templates + comparison engine
  |                     |
  |                     v
  |                  [Stage 4: VERIFY] relevance + structure + contradiction + completion
  |                     |
  |                     v
  |                  [Stage 5: DECISION GATE] (policy override on contradiction)
  |                     |
  |              -------+--------
  |              |               |
  |           SPEAK          FORAGE/ESCALATE/RETRY
  |              |               |
  |              v               v
  |           Return        Auto-Forage (DDG + Wikipedia + Google)
  |           answer +      then re-run stages 1-4
  |           trust label        |
  |                              v
  |                           Return enriched answer + trust label
  |
  +--------> [EPISODIC MEMORY] record every query (score, route, trust, gaps)
```

**Key principle:** *Graph retrieves what is true. Reranker selects what is relevant. Synthesizer explains what matters. Verifier checks what is correct. Confidence gate decides whether to speak.*

**Design rule:** Do NOT make every query agentic. Fast by default. Agentic only when needed.

### Two Execution Paths

**Fast Path** (default): definitions, factual lookups, math, short queries
- Query -> Router -> Retrieve -> Rerank -> Synthesize -> Verify -> Gate -> Answer
- Typical latency: 1-2s (graph), <200ms (math)

**Agentic Path**: compare/contrast, multi-hop, ambiguous, document analysis
- Query -> Planner -> entity-level forage -> retrieve per entity -> synthesize comparison -> verify -> gate
- Typical latency: 2-30s depending on complexity

### Decision Gate (5 decisions)

| Decision | Condition | Action |
|----------|-----------|--------|
| **SPEAK** | Relevance >= 0.46 + sufficient evidence | Return answer immediately |
| **STREAM_PARTIAL** | 0.30 <= relevance < 0.46 (fast path) | Give partial answer, continue foraging |
| **ESCALATE** | 0.30 <= relevance < 0.46 (agentic path) | Send to planner for deeper research |
| **FORAGE** | Relevance < 0.30 or no data | Search internet for missing knowledge |
| **REFUSE** | Unsafe or unanswerable | Decline with explanation |

### Relevance Scorer (4 layers)

```
relevance = 0.20 * keyword_score     (IDF-weighted noun matching)
          + 0.15 * synonym_score     (WordNet + Lexicon synonym net)
          + 0.45 * embedding_score   (SentenceTransformer cosine sim)
          + 0.20 * relation_score    (graph edge connectivity check)
```

### v0.8 Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `kos/mission.py` | ~600 | **v0.8** Mission Manager: goal DAG, auto-decomposition, execution, checkpoints, deliverables, persistence, agent dispatch |
| `kos/agent_protocol.py` | ~100 | **v0.8** Strict agent contracts: AgentTask, AgentResult, AgentStatus, AgentEvidence |
| `kos/agent_registry.py` | ~55 | **v0.8** Deterministic goal-type-to-agent matching |
| `kos/task_dispatcher.py` | ~80 | **v0.8** Central dispatch authority with audit logging and exception safety |
| `kos/agents/retrieval_agent.py` | ~80 | **v0.8** Wraps query pipeline for retrieve/factual/verify/analyze/monitor goals |
| `kos/agents/comparison_agent.py` | ~75 | **v0.8** Wraps query pipeline for comparison goals |
| `kos/agents/synthesis_agent.py` | ~90 | **v0.8** Synthesizes from upstream goal outputs or query fallback |
| `kos/query_pipeline.py` | ~400 | Orchestrator: routes, retrieves, reranks, synthesizes, **verifies**, gates, forages, records episodes |
| `kos/agent_router.py` | 214 | Fast/agentic path decision, modality detection, answer type classification (expanded comparison patterns) |
| `kos/decision_gate.py` | 165 | 5-decision policy engine: speak/stream/escalate/forage/refuse |
| `kos/verifier.py` | ~400 | **v0.7.1** Post-synthesis verification: 7 checks (relevance, structure, contradiction, completion, hard gates, grounding, risk) |
| `kos/answer_validator.py` | 116 | **v0.6.2** Answer-type validator: comparison/factual structural checks with penalty scoring |
| `kos/episodic_memory.py` | 229 | **v0.7** Thread-safe ring buffer (500 episodes) with JSON persistence and aggregate stats |
| `kos/evidence_store.py` | 160 | Normalized evidence from all sources (graph, web, math, file) |
| `kos/stream_manager.py` | 155 | SSE event generator for real-time pipeline progress |
| `kos/relevance.py` | 310 | 4-layer hybrid scorer: keyword + synonym + embedding + graph |
| `kos/reranker.py` | 180 | 7 scoring signals + 2 penalties |
| `kos/synthesis.py` | ~350 | Template-based output + **comparison engine** (entity type classifier, domain-specific attribute extraction, side-by-side synthesis) |
| `kos/router_offline.py` | 1,199 | 6-cascade retrieval: exact > compound > embed > synonym > property > weaver |
| `kos/drivers/math.py` | ~200 | SymPy CAS: integration, differentiation, algebra, arithmetic, factorial, log base, percentage |
| `kos/output_validator.py` | 111 | Hallucination guard: validates numbers, dates, proper nouns against source facts |

### Verifier Layer (v0.7)

Post-synthesis quality gate that runs AFTER synthesis, BEFORE the decision gate. Produces a `VerificationResult` with trust label and score adjustment (-0.3 to +0.1).

**Seven verification checks:**

| # | Check | Bonus | Penalty |
|---|-------|-------|---------|
| 1 | **Relevance**: noun coverage vs query | +0.05 if >=60% | -0.10 if <30% |
| 2 | **Structure**: format matches query type | -- | -0.08/entity, -0.05 no comparative |
| 3 | **Contradiction**: opposing terms across sentences | -- | -0.05 per pair |
| 4 | **Completion**: temporal needs dates, quantitative needs numbers | -- | -0.10 if <0.5 |
| 5 | **Hard Gates**: binary fail conditions (caps score at 0.49) | -- | forces FORAGE/ESCALATE |
| 6 | **Grounding**: answer claims supported by evidence (embedding cosine or lexical Jaccard) | +0.03 if >=60% | -0.12 if <25%, -0.06 if <40% |
| 7 | **Risk/Preference**: unsupported superlatives without qualifying criteria | -- | -0.08 (-0.12 for comparisons) |

**Hard Gates (v0.7.1):** Binary fail conditions that override the weighted score. Any hard fail caps relevance at 0.49 and forces the decision gate to FORAGE or ESCALATE.

| Gate | Condition | Failure Tag |
|------|-----------|-------------|
| Missing comparison entity | One or both entities absent from comparison answer | `V_MISSING_ENTITY` / `V_MISSING_BOTH_ENTITIES` |
| No comparison structure | Comparison answer lacks any comparative markers | `V_NO_COMPARISON_STRUCTURE` |
| Math no result | Computation answer contains no numbers | `V_MATH_NO_RESULT` |
| Fatal contradiction | 3+ contradiction pairs in single answer | `V_FATAL_CONTRADICTION` |
| Meta contamination | LLM returns "As an AI..." instead of factual answer | `V_META_CONTAMINATION` |
| Missing topic entity | Factual answer about a proper noun doesn't mention it | `V_MISSING_TOPIC_ENTITY` |

**Trust Labels:**

| Label | Condition | Meaning |
|-------|-----------|---------|
| `verified` | adjustment >= -0.02, completeness >= 0.6, no contradictions | High confidence, all checks passed |
| `best-effort` | adjustment >= -0.15, completeness >= 0.3 | Acceptable but some issues |
| `low-confidence` | Worse than best-effort thresholds | Significant quality concerns |
| `unverified` | No-data response or math (skips verifier) | Not verified by the pipeline |

### Answer-Type Validator (v0.6.2)

Structural checks per query type with penalty-based scoring, used for coverage-aware confidence.

**Comparison validation:** both entities mentioned (0.15 penalty each missing), comparative structure present (0.10), answer length >= 30 chars (0.05), not a no-data response (0.20). Total cap: 0.50. Invalid if penalties >= 0.30.

**Factual validation:** not empty/no-data (0.20), query content word overlap (0.10), length >= 20 chars (0.05). Total cap: 0.35. Invalid if penalties >= 0.25.

### Episodic Memory (v0.7)

Thread-safe ring buffer storing the last 500 query episodes for debugging, performance monitoring, and planner learning.

**Per-episode fields:** query, answer (truncated 500 chars), route, answer_type, solver, relevance_score, trust_label, latency_ms, foraged_nodes, coverage_gaps, failure_type, timestamp, source.

**Failure types:** `low_score` (relevance < 0.46), `no_data`, `timeout`, `contradiction`.

**API endpoints:**
- `GET /api/memory/stats` — aggregate statistics (avg score, pass rate, failure rate, by route, by trust, top coverage gaps)
- `GET /api/memory/recent` — last 20 episodes
- `GET /api/memory/failures` — recent failed episodes

**Persistence:** optional JSON file at `.cache/episodic_memory.json`, loaded at startup, saved on demand.

### Mission Manager (v0.8)

Turns single-query agents into sustained, multi-step mission execution with goal dependency graphs, auto-decomposition, and deliverable generation.

**Architecture:**
```
User Goal -> Mission Manager -> Auto-Decompose -> Goal DAG -> Execute Goals -> Deliverables
```

**Key concepts:**
- **Mission**: top-level container with goals, checkpoints, deliverables, lifecycle state
- **Goal**: single objective with dependencies, assigned agent, completion criteria, retry logic
- **Checkpoint**: progress gate that must be reached by a deadline
- **Deliverable**: output artifact (summary, comparison, report) auto-generated from goal results

**Goal Types:** `retrieve`, `compare`, `synthesize`, `verify`, `monitor`, `analyze`

**Auto-Decomposition Patterns:**
| Pattern | Trigger | Goals Generated |
|---------|---------|-----------------|
| Comparison | "compare X and Y" / "comparison of X and Y" | retrieve X, retrieve Y, compare, synthesize (4 goals) |
| Monitor | "monitor X" / "track X" | retrieve baseline, monitor, summarize (3 goals) |
| Analyze | "analyze X" / "research X" | retrieve, analyze, verify, synthesize (4 goals) |
| Default | anything else | retrieve, synthesize (2 goals) |

**Goal Execution:**
- Goals execute in dependency order (DAG)
- Dependent goals become READY when all dependencies complete
- Failed dependencies cause dependent goals to be SKIPPED
- Goals with score < 0.46 retry up to `max_attempts` (default 3)
- Auto-deliverable generation for SYNTHESIZE and COMPARE goal types

**Mission Lifecycle:** `planning` -> `active` -> `completed` / `failed` / `paused` / `cancelled`

**Persistence:** JSON file at `.cache/missions.json`, loaded at startup, saved after every goal execution.

**API Endpoints (v0.8):**
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/missions` | Create a new mission |
| `GET` | `/api/missions` | List missions (optional `?status=` filter) |
| `GET` | `/api/missions/{id}` | Get full mission state |
| `POST` | `/api/missions/{id}/plan` | Auto-decompose into goal graph |
| `POST` | `/api/missions/{id}/execute` | Execute next ready goal (runs in threadpool) |
| `POST` | `/api/missions/{id}/execute_all` | Execute all goals sequentially (runs in threadpool) |
| `POST` | `/api/missions/{id}/goals` | Add a custom goal |
| `POST` | `/api/missions/{id}/checkpoints` | Add a checkpoint |
| `POST` | `/api/missions/{id}/pause` | Pause mission |
| `POST` | `/api/missions/{id}/resume` | Resume paused mission |
| `POST` | `/api/missions/{id}/cancel` | Cancel mission |

**v0.8 E2E Test Results (2026-03-26):**
Mission: "Compare Toronto and Montreal"
| Goal | Type | Score | Band |
|------|------|-------|------|
| Retrieve Toronto | retrieve | 0.768 | STRONG |
| Retrieve Montreal | retrieve | 0.879 | EXCELLENT |
| Compare | compare | 0.714 | STRONG |
| Synthesize report | synthesize | 0.830 | STRONG |
| **Average** | | **0.798** | **STRONG** |

4/4 goals completed, 2 deliverables generated (comparison + synthesis), all lifecycle operations (pause/resume/cancel) verified.

### Comparison Engine (v0.6.1+)

Entity-level evidence retrieval and structured comparison synthesis.

**Pipeline:**
1. **Entity extraction** — regex patterns detect "X vs Y", "compare X and Y", "difference between X and Y"
2. **Entity-level forage** — forage each entity separately (not the raw comparison prompt) to avoid Wikipedia noise
3. **Entity type classification** — keyword-based classifier: city, drug, material, technology, concept, etc.
4. **Domain-specific attribute extraction** — per-type extractors (drug: mechanism/indication/side effects, city: population/economy/geography, etc.)
5. **Side-by-side synthesis** — attribute alignment + structured comparison output with confidence scoring

### Policy Gate Override (v0.7)

When the verifier detects contradictions:
- If the decision gate would normally SPEAK, the pipeline **overrides** by re-running the gate with a 0.15 score penalty
- This can downgrade SPEAK to FORAGE/ESCALATE, triggering auto-forage to resolve the contradiction

---

## 6-Layer Neurosymbolic Stack

| Layer | Name | Module | Purpose |
|-------|------|--------|---------|
| **1** | Concept Nodes | `kos/node.py` | Dual-state neurons: activation (truth) + fuel (propagation energy). Hyperpolarization gate, top-K synaptic routing, fire-once rule. |
| **2** | Knowledge Graph | `kos/graph.py` | Spreading activation kernel with priority-queue BFS. Optional Rust backend. Provenance tracking per edge. |
| **3** | Lexicon (DNS) | `kos/lexicon.py` | Maps words to stable UUIDs via WordNet synsets. Phonetic hashing (Metaphone + Soundex) for typo tolerance. Synonym net for alias resolution. |
| **4** | Weaver (Scoring) | `kos/weaver.py` | Algorithmic answer assembly. Intent detection (WHERE/WHEN/WHO/HOW). Keyword density scoring. Noise suppression. Source diversity bonus. |
| **5** | Router (Shell) | `kos/router_offline.py` | 6-cascade query pipeline: exact match, compound detection, SentenceTransformer embedding search, synonym expansion, property queries, weaver assembly. |
| **6** | Drivers (I/O) | `kos/drivers/*.py` | Domain-specific ingestion: Text (SVO extraction), Physics (F=ma), Chemistry (periodic table), Math (SymPy CAS), Biology (amino acids), Finance (VaR/Basel), Code (verified generation). |

---

## Core Modules

### Kernel & Graph (`kos/graph.py`, `kos/node.py`)
- **ConceptNode**: Dual activation/fuel model with temporal decay, max energy capping
- **KOSKernel**: Spreading activation with configurable spatial decay, threshold, max hops
- **Provenance**: Every edge stores source sentences for evidence chains
- **Optional Rust**: `kos_rust.RustKernel` via PyO3 for arena-based contiguous memory (10-50x speedup)

### Router (`kos/router_offline.py`)
- **KOSShellOffline**: Fully offline query answering — no LLM API calls
- **SentenceTransformer**: `all-MiniLM-L6-v2` for embedding-based semantic search
- **6-Layer Cascade**: exact match > compound detection > embedding search > synonym expansion > property query > weaver assembly
- **Domain Routing**: Keyword-based + embedding similarity routing to specialized agents (physics, chemistry, math, CS, general)

### Weaver (`kos/weaver.py`)
- Intent detection: WHERE (+40), WHEN (+40), WHO (+40), ATTRIBUTE (+35), HOW (+30)
- Keyword density scoring with configurable multiplier
- Noise suppression: sports, navigation, metadata terms penalized
- Short sentence penalty, daemon-generated provenance penalty
- Provenance trust scoring and source diversity bonus

### Lexicon (`kos/lexicon.py`)
- WordNet synset-based UUID generation (stable across sessions)
- Metaphone + Soundex phonetic hashing (6-layer typo recovery)
- Synonym net: all lemma names map to canonical UUID
- Domain-protected words (quantum terms, chemical names preserved)

### Text Driver (`kos/drivers/text.py`)
- SVO extraction via NLTK POS tagging
- Clause-level splitting (comma clauses, relative clauses)
- Negation detection and negative edge weights
- Adjective extraction and property assignment
- Batch ingestion for corpus loading

---

## Intelligence Modules

### Predictive Coding Engine (`kos/predictive.py`)
Implementation of Karl Friston's Free Energy Principle:
1. **PREDICT**: Before propagation, predict which nodes activate
2. **PROPAGATE**: Run actual spreading activation
3. **COMPARE**: Compute prediction error (expected vs actual)
4. **UPDATE**: Adjust edge weights via local Hebbian rule (not gradient descent)

### Emotion Engine (`kos/emotion.py`)
- Neurochemical state vector: cortisol, dopamine, serotonin, norepinephrine, oxytocin, GABA, glutamate, endorphin
- Clinical thresholds from psychopharmacology literature
- Half-life decay kinetics per substance
- Maps to named emotional states (joy, fear, curiosity, etc.)

### Autonomous Agent (`kos/autonomous.py`)
Continuous learning loop:
1. **Dreamer** generates curiosity queries
2. **Forager** acquires knowledge from internet (Wikipedia, arXiv, Google, DuckDuckGo)
3. **AutoImprover** optimizes thresholds, adds synonyms, prunes orphans
4. **Self-Model** tracks what was learned
5. **Canary Deployer** stages and validates config changes

Safety: max cycles, rate limiter, kill file (`kos_stop`), max nodes cap.

### Auto-Improver (`kos/auto_improve.py`)
- **SAFE** (auto-applied): synonym additions, threshold tuning, compound detection, orphan pruning, weight normalization, prediction cache training
- **UNSAFE** (queued for human review): new code generation, architecture changes, weaver formula changes
- All actions logged. All changes reversible.

### Self-Repair Loop (in `api.py`)
Background thread (every 120s):
- Rebalances hub/orphan degree distribution
- Normalizes edge weights (clips outliers)
- Discovers formulas (pattern detection)
- Runs benchmark (7 test queries)
- Logs all actions to `/api/repair_log`

### Proposal System (`proposals/`)
- 116 auto-generated proposals: 46 daemon strategies, 31 synonym additions, 29 threshold changes, 10 weaver rules
- All approved and applied via `apply_approved_proposals()` at startup
- Config persisted to `.cache/self_tuned_config.json`

### Domain Agent Factory (`kos/agent_factory.py`)
- Auto-generates specialized agents per domain (physics, chemistry, math, CS, general)
- Each agent has its own knowledge subgraph
- Agents are read-only (cannot modify kernel)
- Approval/rejection workflow via dashboard

---

## Forager (Knowledge Acquisition) (`kos/forager.py`)

Multi-source search pipeline:
1. **DuckDuckGo Instant Answer** — fastest (~1s), deterministic factual answers
2. **Wikipedia Search** — progressive fallback (full query > first 3 words > first 2 > individual nouns)
3. **Google Search** — broadest coverage, scrapes top 2 results
4. **arXiv API** — scientific papers (titles + abstracts)
5. **Local file ingestion** — `.txt` and `.md` files

Auto-forage triggers via the **Decision Gate** when:
- Answer contains "no data" phrases → `FORAGE` decision
- Answer is shorter than 5 characters → `FORAGE` decision
- **4-layer relevance score < 0.30** → `FORAGE` decision
- **Relevance 0.30-0.46** → `STREAM_PARTIAL` (fast path) or `ESCALATE` (agentic path), then forage

---

## Hybrid Relevance Scoring (4-Layer)

The system uses a **4-layer hybrid scorer** (`kos/relevance.py`) to detect off-topic answers:

```
relevance = 0.20 * keyword_score     (IDF-weighted noun matching)
          + 0.15 * synonym_score     (WordNet + Lexicon synonym net)
          + 0.45 * embedding_score   (SentenceTransformer cosine similarity)
          + 0.20 * relation_score    (graph 2-hop neighborhood connectivity)
```

- **Keyword layer** (20%): Extracts content nouns from query, checks IDF-weighted presence in the answer
- **Synonym layer** (15%): Expands query nouns via WordNet synsets and lexicon UUID lookups, checks coverage
- **Embedding layer** (45%): SentenceTransformer `all-MiniLM-L6-v2` encodes query and answer, computes cosine similarity
- **Relation layer** (20%): Checks if answer concepts are within 2-hop neighborhood of ALL query concepts in the knowledge graph
- **Threshold**: If `relevance < 0.46`, the decision gate triggers FORAGE or STREAM_PARTIAL instead of SPEAK

Calibrated on 28 test queries. Handles semantic drift (e.g., "distance of moon" returning Earth-Sun facts scores 0.453, correctly triggering auto-forage).

---

## KASM — Knowledge Assembly Language (`kasm/`)

A domain-specific language for Vector Symbolic Architectures (VSA):
- **Lexer/Parser**: Tokenizes and parses KASM scripts
- **VSA Engine**: 10,000-dimensional hypervectors for binding/bundling
- **Bridge**: Connects KASM operations to the KOS kernel
- **Abstraction**: Layer 3 analogy detection via VSA similarity

```kasm
BIND sun center → sun_center
BUNDLE planet star → celestial_body
QUERY celestial_body → [sun: 0.89, earth: 0.67]
```

---

## API Reference

### Core Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Dashboard (HTML) |
| `GET` | `/api/status` | System status (cached, instant) — nodes, edges, uptime, query stats |
| `POST` | `/api/query` | Ask a question — auto-routes, verifies, trust-labels, records episode |
| `POST` | `/api/ingest` | Ingest raw text into the knowledge graph |
| `GET` | `/api/health` | Full health check — runs benchmark + all learning mechanisms |
| `GET` | `/api/health/last` | Last cached health check result (non-blocking) |

### Episodic Memory (v0.7)
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/memory/stats` | Aggregate statistics: avg score, pass rate, failure rate, by route, by trust |
| `GET` | `/api/memory/recent` | Last 20 query episodes with full metadata |
| `GET` | `/api/memory/failures` | Recent failed episodes for debugging |

### Graph Inspection
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/graph` | Full graph as JSON (nodes + edges) |
| `GET` | `/api/graph/top_nodes` | Top activated nodes |
| `GET` | `/api/contradictions` | Detected contradictions in the graph |

### Task Management
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/task` | Create a research task |
| `GET` | `/api/tasks` | List all tasks |
| `GET` | `/api/queries` | Query history with latency |

### Autonomous Agent
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/agent/start` | Start autonomous learning loop |
| `POST` | `/api/agent/stop` | Stop autonomous agent |
| `POST` | `/api/agent/pause` | Pause agent |
| `POST` | `/api/agent/resume` | Resume agent |
| `GET` | `/api/agent/status` | Agent status (running, cycles, nodes learned) |
| `GET` | `/api/agent/events` | Agent event log |
| `GET` | `/api/agent/foraged` | Topics foraged by agent |
| `GET` | `/api/agent/proposals` | Agent-generated proposals |

### Domain Agents
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/domain_agents` | List all domain agents |
| `POST` | `/api/domain_agents/route` | Route a query to the best domain agent |
| `POST` | `/api/domain_agents/approve/{id}` | Approve a domain agent |
| `POST` | `/api/domain_agents/reject/{id}` | Reject a domain agent |
| `GET` | `/api/domain_agents/monitor` | Monitor all agents |

### Proposals & Self-Improvement
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/proposals` | List all proposals |
| `POST` | `/api/proposals/approve/{id}` | Approve a proposal |
| `POST` | `/api/proposals/reject/{id}` | Reject a proposal |
| `POST` | `/api/proposals/apply_all` | Apply all approved proposals |
| `GET` | `/api/repair_log` | Self-repair loop history |

### Sensors (Multimodal)
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/sensors/speak` | Text-to-speech input |
| `POST` | `/api/sensors/see` | Image input (base64) |
| `POST` | `/api/sensors/listen` | Audio input |
| `GET` | `/api/sensors/emotion` | Current emotional state |

### Persistence
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/save` | Save graph to disk (`.cache/kos_brain.kos`) |
| `GET` | `/api/load` | Load saved graph from disk |

---

## Startup & Deployment

### Quick Start
```bash
cd kos-engine
PYTHONIOENCODING=utf-8 python -m uvicorn api:app --host 0.0.0.0 --port 8080
```
Dashboard: `http://localhost:8080`

### Startup Sequence
1. **Import modules** — loads kernel, lexicon, drivers, all KOS subsystems
2. **SentenceTransformer preload** — background thread loads `all-MiniLM-L6-v2` (~30s)
3. **Seed corpus ingestion** — 40+ seed sentences covering Toronto, physics, chemistry, math, CS, biology
4. **Apply proposals** — runs `apply_approved_proposals()` on 116 approved proposals
5. **Start background threads**:
   - Status updater (1s interval) — keeps `/api/status` instant
   - Self-repair loop (120s interval) — rebalances, normalizes, benchmarks
6. **Server ready** — uvicorn starts accepting requests

### Requirements
- Python 3.12+
- `sentence-transformers` (all-MiniLM-L6-v2)
- `nltk` (wordnet, averaged_perceptron_tagger)
- `jellyfish` (phonetic hashing)
- `fastapi` + `uvicorn`
- `beautifulsoup4` + `requests` (forager)
- `sympy` (math driver)
- Optional: `kos_rust` (Rust backend via PyO3)

### Windows Notes
- Always use `PYTHONIOENCODING=utf-8` (cp1252 codec fails on Unicode symbols)
- Use `python -u` for unbuffered output in background tasks
- Avoid Unicode symbols in `print()` statements

---

## Dashboard (Mission Control)

`static/dashboard.html` — Single-page HTML/CSS/JS dashboard.

### Pages
1. **Overview** — Node count, edge count, uptime, queries, latency, prediction accuracy, orphans, hubs, contradictions
2. **Query** — Interactive query interface with agent routing display
3. **Graph** — Top activated nodes and edge visualization
4. **Tasks** — Research task management
5. **Health Monitor** — Run benchmark, view learning mechanism scores, accuracy history
6. **Autonomous Agent** — Start/stop/pause agent, view cycles, events, foraged topics, proposals
7. **Domain Agents** — View/approve/reject auto-generated domain-specialized agents
8. **Proposals** — View/approve/reject system improvement proposals
9. **Experiment** — Run controlled experiments on the knowledge graph
10. **Sensors** — Multimodal input (speak, see, listen) with emotion state display

### Auto-Refresh
- Status: every 1s (background thread, instant response)
- Agent status: every 4s
- Health: on-demand (Run Health Check button)

---

## Key Design Decisions

### Why No LLM for Reasoning?
LLMs hallucinate. KOS uses spreading activation with provenance — every answer traces back to source sentences. The graph is deterministic: same query, same state = same answer.

### Why Spreading Activation?
Biological neural networks don't do backpropagation at inference time. They spread energy through weighted connections. KOS mimics this: inject energy at query nodes, let it propagate through the graph, collect the highest-activated nodes as evidence.

### Why Dual State (Activation + Fuel)?
- **Activation** = epistemic truth (how much evidence supports this concept). Accumulates, never drains.
- **Fuel** = propagation energy (how much signal to pass downstream). Drains to zero after firing (fire-once rule).
This prevents infinite loops and models refractory periods in biological neurons.

### Why 4-Layer Hybrid Relevance Scoring?
No single signal is sufficient:
- **Keywords alone** fail on paraphrasing ("lunar distance" vs "moon distance")
- **Embeddings alone** fail on homonyms ("distance" inflates similarity for unrelated topics)
- **Synonyms alone** can be too loose (phonetic: "moon" → "men")
- **Graph edges alone** fail when hub nodes connect everything

The 4-layer hybrid (20% keyword + 15% synonym + 45% embedding + 20% relation) catches all cases. Threshold 0.46 was calibrated on 28 test queries.

### Why Auto-Forage?
A knowledge system that can only answer pre-loaded questions is useless. KOS detects when its answer is off-topic (via hybrid scoring) and automatically searches the internet to acquire missing knowledge, then re-answers with the new data.

---

## File Structure

```
kos-engine/
├── api.py                    # FastAPI server (~1,700 lines) — all endpoints + pipeline
├── setup.py                  # Package setup
├── SYSTEM.md                 # This file
│
├── kos/                      # Core engine (22,200 lines)
│   ├── __init__.py           # Package exports, lazy driver imports
│   ├── node.py               # ConceptNode — dual activation/fuel neuron
│   ├── graph.py              # KOSKernel — spreading activation + Rust fallback
│   ├── lexicon.py            # KASMLexicon — WordNet DNS + phonetic hashing
│   ├── weaver.py             # AlgorithmicWeaver — intent scoring + provenance
│   ├── router.py             # KOSShell — original LLM-backed router
│   ├── router_offline.py     # KOSShellOffline -- fully offline 6-cascade router
│   ├── mission.py            # v0.8 mission manager: goal DAG, auto-decompose, execute, checkpoints, deliverables
│   ├── agent_protocol.py     # v0.8 strict agent contracts (AgentTask, AgentResult, AgentStatus)
│   ├── agent_registry.py     # v0.8 deterministic goal-type-to-agent matching
│   ├── task_dispatcher.py    # v0.8 central dispatch with audit log and exception safety
│   ├── agents/               # v0.8 agent implementations
│   │   ├── base_agent.py     # Abstract base with can_handle + execute
│   │   ├── retrieval_agent.py # retrieve/factual/verify/analyze/monitor
│   │   ├── comparison_agent.py# compare/comparison
│   │   └── synthesis_agent.py # synthesize/summary
│   ├── query_pipeline.py     # v0.7 orchestrator: route → retrieve → rerank → synthesize → verify → gate → forage → record
│   ├── agent_router.py       # v0.7 agent router: fast/agentic/math path detection (expanded comparison patterns)
│   ├── decision_gate.py      # v0.6 decision gate: 5-decision policy engine (speak/stream/escalate/forage/refuse)
│   ├── verifier.py           # v0.7 post-synthesis verifier: relevance + structure + contradiction + completion
│   ├── answer_validator.py   # v0.6.2 answer-type validator: comparison/factual structural checks
│   ├── episodic_memory.py    # v0.7 episodic memory: ring buffer (500) + JSON persist + stats
│   ├── evidence_store.py     # v0.6 evidence store: normalized, deduped, ranked evidence from all sources
│   ├── stream_manager.py     # v0.6 SSE event generator: ack → routing → status → evidence → final
│   ├── relevance.py          # 4-layer hybrid relevance scorer (keyword+synonym+embed+graph)
│   ├── reranker.py            # Multi-signal reranker (7 signals + 2 penalties)
│   ├── synthesis.py           # Template-based answer synthesis (4 domain templates)
│   ├── output_validator.py    # LLM hallucination guard (numbers, dates, proper nouns)
│   ├── daemon.py             # KOSDaemon — GC, dedup, triadic closure
│   ├── forager.py            # WebForager — DDG + Wikipedia + Google + arXiv
│   ├── auto_improve.py       # AutoImprover — safe self-modification
│   ├── autonomous.py         # AutonomousAgent — continuous learning loop
│   ├── predictive.py         # Predictive Coding Engine (Friston)
│   ├── emotion.py            # Neurochemical emotion model
│   ├── emotion_integration.py# Emotion-decision bridge
│   ├── attention.py          # Attention controller
│   ├── dreamer.py            # Curiosity-driven query generation
│   ├── hypothesis.py         # Hypothesis generation and testing
│   ├── feedback.py           # User feedback processing
│   ├── persistence.py        # Graph save/load (.kos format)
│   ├── propose.py            # Proposal generation for improvements
│   ├── self_improve.py       # Self-improvement strategies
│   ├── self_model.py         # Self-awareness model
│   ├── selfmod.py            # Self-modification framework
│   ├── sleep.py              # Sleep cycle — memory consolidation
│   ├── hierarchical.py       # Hierarchical concept organization
│   ├── temporal.py           # Temporal reasoning
│   ├── causal_dag.py         # Causal DAG for inference
│   ├── reasoning.py          # Logical reasoning chains
│   ├── research.py           # Research task management
│   ├── experiment.py         # Controlled experiments
│   ├── constraints.py        # Constraint satisfaction
│   ├── synthesis.py          # Knowledge synthesis
│   ├── verification.py       # Answer verification
│   ├── metacognition.py      # Self-monitoring and reflection
│   ├── agent_factory.py      # Domain agent auto-generation
│   ├── domain_profiles.py    # Domain knowledge profiles
│   ├── user_model.py         # User preference tracking
│   ├── social.py             # Multi-agent social dynamics
│   ├── scaling.py            # Performance scaling
│   ├── neuromorphic.py       # Neuromorphic computing bridge
│   ├── query_normalizer.py   # Query preprocessing
│   ├── compound_detector.py  # Multi-word compound detection
│   ├── reranker.py           # Result re-ranking
│   ├── retrieval_lanes.py    # Multi-lane retrieval
│   ├── output_validator.py   # Output quality checks
│   ├── source_governance.py  # Source trust scoring
│   ├── canary.py             # Canary deployment for config changes
│   ├── drives.py             # Motivational drives
│   ├── sensorimotor.py       # Sensorimotor integration
│   ├── sensory_memory.py     # Short-term sensory buffer
│   ├── learning.py           # Learning rate scheduling
│   ├── memory_lifecycle.py   # Memory consolidation/forgetting
│   ├── multilang.py          # Multi-language support
│   ├── synonyms.py           # Synonym management
│   ├── edge_types.py         # Typed edge relationships
│   ├── tiers.py              # Safety tier classification
│   ├── boot_brain.py         # Brain bootstrap sequence
│   ├── action_registry.py    # Action type registry
│   ├── rust_bridge.py        # Rust FFI bridge
│   ├── julia_bridge.py       # Julia FFI bridge
│   │
│   └── drivers/              # Domain-specific I/O (3,200 lines)
│       ├── text.py           # SVO extraction, clause splitting, negation
│       ├── physics.py        # Mechanics, thermodynamics, E&M
│       ├── chemistry.py      # Periodic table, reactions, bonds
│       ├── biology.py        # Amino acids, genetics, cell biology
│       ├── math.py           # SymPy CAS integration
│       ├── code.py           # Verified code generation
│       ├── finance.py        # VaR, Basel III, risk assessment
│       ├── ast.py            # AST parsing
│       └── vision.py         # Image processing (stub)
│
├── kasm/                     # Knowledge Assembly Language (VSA DSL)
│   ├── __init__.py
│   ├── vsa.py                # 10K-dimensional hypervector engine
│   ├── lexer.py              # KASM tokenizer
│   ├── parser.py             # KASM parser
│   ├── interpreter.py        # KASM interpreter
│   ├── abstraction.py        # Layer 3 analogy detection
│   └── bridge.py             # KASM-KOS bridge
│
├── agents/                   # Auto-generated domain agents
│   ├── agent_physics_*.py
│   ├── agent_chemistry_*.py
│   ├── agent_mathematics_*.py
│   ├── agent_computer_science_*.py
│   └── agent_general_knowledge_*.py
│
├── proposals/                # 116 auto-generated improvement proposals
│   └── proposal_*.json       # Types: threshold_change, synonym_addition,
│                             #        weaver_rule, daemon_strategy
│
├── static/
│   └── dashboard.html        # Mission Control dashboard (63KB)
│
├── .cache/
│   ├── kos_brain.kos         # Saved graph state (153MB)
│   ├── self_tuned_config.json# Applied proposal config
│   ├── synonym_map.json      # Learned synonyms
│   ├── health_result.json    # Last health check
│   ├── self_loop_log.json    # Autonomous agent log
│   └── user_profiles.json    # User interaction profiles
│
├── tests/                    # 40+ test files
│   ├── master_smoke_test.py
│   ├── test_16_subsystems.py
│   ├── test_retrieval_pipeline.py
│   ├── test_v071_lanes.py       # v0.7.1 lane test: math/factual/comparison (25 queries)
│   ├── test_v08_mission_e2e.py  # v0.8 Step 1 mission E2E: create/plan/execute/pause/resume/cancel
│   ├── test_v08_resilience.py   # v0.8 Step 1 resilience: failure injection/dependency/retry/concurrent/audit (118 tests)
│   ├── test_v08_step2.py        # v0.8 Step 2 multi-agent: protocol/registry/dispatcher/parity/isolation/e2e (108 tests)
│   └── ...
│
└── archive/                  # Legacy code (kept for reference)
    ├── app.py
    ├── dashboard.py
    └── ...
```

---

## Benchmarks

### v0.7.1 Pipeline Test (2026-03-26) — 21/21 PASS (completed queries)

**Lane-based testing with score bands:** EXCELLENT >= 0.85, STRONG >= 0.70, USABLE >= 0.55, WEAK < 0.55

| Lane | Avg Score | Band | Min | Max | Pass | Target | Status |
|------|-----------|------|-----|-----|------|--------|--------|
| **Math** (9 completed) | **1.000** | EXCELLENT | 1.0 | 1.0 | 9/9 | 1.0 | HIT |
| **Factual** (7 completed) | **0.840** | STRONG | 0.710 | 0.899 | 7/7 | 0.825 | HIT |
| **Comparison** (5 queries) | **0.784** | STRONG | 0.714 | 0.871 | 5/5 | 0.785 | HIT |
| **Overall** (21 completed) | **0.895** | -- | -- | -- | 21/21 | 0.88-0.92 | HIT |

**Trust label distribution:** 12 verified (57%), 10 unverified (43% — math skips verifier by design)

**v0.7.1 verifier stats:** 0 hard fails on real queries, 0 false-positive grounding flags, 0 false-positive preference flags.

**Score progression across versions:**
| Version | Math | Factual | Comparison | Overall |
|---------|------|---------|------------|---------|
| v0.6.0 | 7/12 pass | ~0.75 | 0.595 | -- |
| v0.6.1 | 10/10 (1.0) | ~0.79 | ~0.59 (4/5) | 0.855 |
| v0.6.2 | 10/10 (1.0) | ~0.79 | ~0.60 | 0.859 |
| v0.7.0 | 10/10 (1.0) | 0.839 | 0.761 | 0.888 |
| **v0.7.1** | **9/9 (1.0)** | **0.840** | **0.784** | **0.895** |
| **v0.8.0** | -- | -- | -- | **0.798** (mission avg) |

**Key v0.8 additions:**
- Mission Manager: multi-step goal execution with dependency graphs
- Auto-decomposition: "Compare Toronto and Montreal" → 4 goals → 4/4 complete → 2 deliverables
- Goal scores: retrieve=0.768/0.879, compare=0.714, synthesize=0.830

**Key v0.7.1 improvements:**
- Hard gates: binary fail conditions for missing entities, no result, meta contamination, fatal contradiction
- Grounding verifier: embedding cosine or lexical Jaccard between answer sentences and evidence
- Risk/preference check: detects unsupported superlatives without qualifying criteria
- Comparison lane +0.023 (0.761 -> 0.784), now hitting target
- Policy gate override extended: hard_fail also triggers downgrade from SPEAK

**Key performance metrics:**
- Math queries: **<200ms** (deterministic, zero hallucination via SymPy)
- Factual queries: **1-2s** (graph spreading activation + 4-layer scoring + verifier)
- Agentic queries: **1-30s** (depends on complexity + entity-level forage)
- Streaming events: 4-8 events per query (ack, routing, status, evidence, verification, final)

### Previous Test (2026-03-25): 24/28 PASS (86%)

**Auto-forage results:**
- DNA: foraged Wikipedia, got data but re-query failed to find it
- Einstein relativity: foraged but Wikipedia article too broad
- Climate change: returned "Toronto climate" (semantic pollution)
- Moon distance: successfully foraged Wikipedia, correct answer on re-query

### Internal Benchmark (7 queries)
Run via `/api/health` or self-repair loop:

| Query | Expected Keywords | Status |
|-------|-------------------|--------|
| What is Toronto? | toronto, city, canada | PASS |
| When was Toronto founded? | 1834 | PASS |
| Population of Toronto | 2.7 million | PASS |
| What is perovskite? | photovoltaic, solar | PASS |
| Tell me about apixaban | thrombosis | PASS |
| What is Montreal? | montreal, city | PASS |
| Explain backpropagation | gradient, weights | PASS |

### Performance
- Startup time: ~60s (SentenceTransformer model load dominates)
- Query latency: 1.5-6.5s (graph queries), 10-60s (with auto-forage)
- Graph size: 3,825-4,409 nodes (grows via auto-forage)
- Saved brain: 153MB (`.cache/kos_brain.kos`)
- Relevance scoring: ~75ms per query (4-layer hybrid)

---

## How to Use

### Ask a Question
```bash
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Newton second law?"}'
```

### Ingest Knowledge
```bash
curl -X POST http://localhost:8080/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "The Eiffel Tower is 330 meters tall and located in Paris."}'
```

### Start Autonomous Learning
```bash
curl -X POST http://localhost:8080/api/agent/start
```

### Run Health Check
```bash
curl http://localhost:8080/api/health
```

### Save/Load Brain State
```bash
curl http://localhost:8080/api/save    # Save to .cache/kos_brain.kos
curl http://localhost:8080/api/load    # Load from disk
```

---

## Known Limitations

1. **Single uvicorn worker**: CPU-bound operations (SentenceTransformer encoding, health check benchmark) block the event loop. Status endpoint uses background thread cache to stay responsive.
2. **GIL contention**: Python's GIL means `run_in_executor` doesn't fully parallelize CPU-bound work. The self-repair loop uses `threading.Lock` with non-blocking acquire to avoid competing with API handlers.
3. **Semantic pollution**: Common words ("distance", "number", "function") can activate unrelated graph regions. Mitigated by 4-layer hybrid relevance scoring (threshold 0.46).
4. **Auto-forage latency**: Internet searches add 5-30s to query time. Capped at 30s timeout per forage cycle.
5. **No persistence auto-save**: Graph must be explicitly saved via `/api/save`. Foraged knowledge is lost on restart unless saved.
6. **Buffered streaming**: v0.6 StreamManager buffers events then yields them. True async SSE push is a v0.8 target.
7. **No session memory**: Each query is independent. Multi-turn conversation context is a v0.8 target. Episodic memory (v0.7) stores per-query episodes but not conversational context.
8. **Math trust labels**: Math fast-path returns before the verifier runs (by design — deterministic solver). Math answers show `unverified` trust label.
9. **Comparison coverage**: Comparison lane averages 0.784 (v0.7.1). Limited by corpus depth for niche entities.
10. **Mission execute_all latency**: Full mission execution (4+ goals) takes 2-8 minutes due to sequential LLM calls. Use step-by-step `/execute` for better responsiveness.
11. **Event loop blocking**: Execute endpoints run in threadpool (`def` not `async def`) to avoid blocking the event loop, but heavy LLM calls can still saturate the thread pool.

---

## Roadmap

### v0.6 (Complete)
- Agent Router (fast/agentic/math path detection)
- Decision Gate (5-decision policy engine)
- Evidence Store (normalized, deduped, ranked)
- Stream Manager (SSE event generator)
- 4-Layer Relevance Scorer (keyword + synonym + embedding + graph)
- Multi-Signal Reranker (7 signals + 2 penalties)
- Template-Based Synthesis Engine (4 domain templates)
- Deterministic Math Solver (SymPy CAS — 10 edge cases fixed)
- Auto-Forage Pipeline (DDG + Wikipedia + Google)
- QueryPipeline orchestrator wiring all components

### v0.6.1 (Complete) — Agentic Comparison Upgrade
- Comparison entity extraction (8 regex patterns)
- Entity-level forage (per-entity Wikipedia lookups)
- Comparison synthesis templates (side-by-side output)
- Entity type classifier (city, drug, material, technology, concept)
- Domain-specific attribute extraction

### v0.6.2 (Complete) — Coverage-Aware Confidence
- Answer-type validator (comparison + factual structural checks)
- Coverage-aware confidence scoring (validator penalties applied to relevance score)
- Graph coverage gap detector (missing_entity, shallow_entity)

### v0.7 (Complete) — Verifier + Memory + Trust
- Verifier Layer: 4-module post-synthesis verification (relevance, structure, contradiction, completion)
- Trust Labels: verified / best-effort / low-confidence / unverified
- Episodic Memory: 500-episode ring buffer with JSON persistence and aggregate stats
- Policy Gate Override: contradiction detection downgrades SPEAK to FORAGE/RETRY
- Memory API: /api/memory/stats, /api/memory/recent, /api/memory/failures
- Score adjustment model: verifier produces -0.3 to +0.1 adjustment on relevance score

### v0.8 (Complete — Step 1) — Adaptive Mission System
- Mission Manager: persistent multi-step goals with goal dependency graphs
- Auto-decomposition: comparison (4 goals), monitor (3), analyze (4), default (2)
- Goal execution: dependency-ordered, retry on low score, skip on failed deps
- Deliverable generation: auto-produces summaries and comparisons from goal results
- Checkpoint system: progress gates with deadline tracking
- Mission lifecycle: planning/active/paused/completed/failed/cancelled
- JSON persistence: missions survive server restart
- 11 REST API endpoints for full mission CRUD and execution
- Async execution: blocking goals run in threadpool (non-blocking event loop)
- E2E tested: "Compare Toronto and Montreal" — 4/4 goals, avg 0.798, 2 deliverables

### v0.8 Step 2 (Complete) — Multi-Agent Framework
- Agent Protocol: strict contracts (AgentTask -> Agent -> AgentResult), no agent-to-agent chatter
- 3 agents: RetrievalAgent (retrieve/factual/verify/analyze/monitor), ComparisonAgent (compare), SynthesisAgent (synthesize/summary)
- Agent Registry: deterministic goal-type-to-agent matching
- Task Dispatcher: single execution authority with audit logging, exception safety
- Mission integration via feature flag (`use_agents=True/False`), with `_build_goal_payload()` and `_apply_agent_result()` helpers
- Parity verified: legacy and agent paths produce identical mission status, goal ordering, deliverable count, and scores (drift <= 0.02)
- 108/108 tests passed: protocol (13), registry (10), dispatcher (13), parity (28), failure isolation (16), e2e (28)

### v0.8 (In Progress — Steps 3-5)
- Event System: event_bus, trigger_handler, subscription_manager
- Deliverable Engine: report_builder, memo_builder, evidence_matrix
- World-State Model: world_state, change_tracker, claim_registry

### v0.9 (Future)
- Learning feedback loop (user corrections improve scoring)
- Episodic memory-informed planner (learn from past failures)
- Semantic vector fallback (Layer 5) for word resolution
- True async SSE streaming (replace buffered events)

---

## Testing

See `tests/` directory for comprehensive test suites. Key test files:
- `test_retrieval_pipeline.py` — Tests the 6-layer query cascade
- `test_16_subsystems.py` — Tests all 16 KOS subsystems
- `test_science_stack.py` — Tests physics/chemistry/biology drivers
- `test_self_improve.py` — Tests auto-improvement pipeline
- `master_smoke_test.py` — Quick smoke test for all modules

---

*Last updated: 2026-03-26*
*KOS Engine v0.8.0 — Adaptive Mission System (Step 2: Multi-Agent Framework)*
