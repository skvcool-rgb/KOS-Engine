"""
KOS v0.7 -- Query Pipeline Orchestrator

    Query
      |
      v
    [Agent Router] -> fast path / agentic path / math solver
      |
      v
    [Stage 1: RETRIEVE] -> spreading activation -> candidate nodes
      |
      v
    [Stage 2: RERANK] -> multi-signal scorer -> top-20 nodes
      |
      v
    [Stage 3: SYNTHESIZE] -> weaver + template -> candidate answer
      |
      v
    [Stage 4: VERIFY] -> relevance + structure + contradiction + completion
      |
      v
    [Stage 5: DECISION GATE] -> speak / stream_partial / forage / escalate / retry
      |
      v
    [Auto-forage if needed] -> DDG + Wikipedia + Google -> re-run stages 1-3
      |
      v
    [Final answer with trust label and citations]

Design principle:
    Graph retrieves what is true.
    Reranker selects what is relevant.
    Synthesizer explains what matters.
    Verifier checks what is correct.
    Confidence gate decides whether to speak.

Do NOT make every query agentic.
Fast by default. Agentic only when needed.
"""

import re
import time
import threading


# ── Comparison Entity Extraction ──────────────────────────────────
_COMPARISON_PATTERNS = [
    re.compile(r'(?i)^compare\s+(.+?)\s+and\s+(.+?)(?:\s*[\?\.]?\s*)$'),
    re.compile(r'(?i)(.+?)\s+vs\.?\s+(.+?)(?:\s*[\?\.]?\s*)$'),
    re.compile(r'(?i)(?:difference|differences)\s+between\s+(.+?)\s+and\s+(.+?)(?:\s*[\?\.]?\s*)$'),
    re.compile(r'(?i)(.+?)\s+compared\s+to\s+(.+?)(?:\s*[\?\.]?\s*)$'),
    re.compile(r'(?i)(?:which\s+is\s+(?:better|bigger|larger|smaller|faster|older|newer))\s+(.+?)\s+or\s+(.+?)(?:\s*[\?\.]?\s*)$'),
    re.compile(r'(?i)(?:compare|contrast)\s+(.+?)\s+(?:with|to)\s+(.+?)(?:\s*[\?\.]?\s*)$'),
    re.compile(r'(?i)(.+?)\s+(?:versus|vs\.?)\s+(.+?)(?:\s*[\?\.]?\s*)$'),
    re.compile(r'(?i)how\s+(?:does|do|is|are)\s+(.+?)\s+(?:compare|differ)\s+(?:to|from|with)\s+(.+?)(?:\s*[\?\.]?\s*)$'),
]


def _extract_comparison_entities(query: str):
    """
    Extract two entity names from a comparison query.

    Returns (entity_a, entity_b) or None if not a comparison.
    """
    q = query.strip().rstrip('?').rstrip('.')
    for pat in _COMPARISON_PATTERNS:
        m = pat.search(q)
        if m:
            a = m.group(1).strip().strip('"\'')
            b = m.group(2).strip().strip('"\'')
            if a and b:
                return (a, b)
    return None


class QueryPipeline:
    """
    v0.6 query orchestrator with agent routing, evidence store,
    decision gate, and optional SSE streaming.
    """

    def __init__(self, kernel, lexicon, shell, weaver=None, reranker=None,
                 synthesizer=None, relevance_scorer=None,
                 math_driver=None, forager_factory=None):
        self.kernel = kernel
        self.lexicon = lexicon
        self.shell = shell
        self.weaver = weaver
        self.reranker = reranker
        self.synthesizer = synthesizer
        self.scorer = relevance_scorer
        self.math_driver = math_driver
        self.forager_factory = forager_factory

        # Lazy-init v0.6 components
        self._router = None
        self._gate = None
        self._verifier = None

        # v0.7: Episodic memory (persists across queries)
        from .episodic_memory import EpisodicMemory
        self.memory = EpisodicMemory(
            max_episodes=500,
            persist_path=".cache/episodic_memory.json"
        )

    def _ensure_components(self):
        """Lazy-init pipeline components."""
        if self._router is None:
            from .agent_router import AgentRouter
            self._router = AgentRouter(math_driver=self.math_driver)

        if self.reranker is None:
            from .reranker import MultiSignalReranker
            self.reranker = MultiSignalReranker()

        if self.synthesizer is None:
            from .synthesis import SynthesisEngine
            self.synthesizer = SynthesisEngine()

        if self.scorer is None:
            try:
                from .relevance import RelevanceScorer
                from .router_offline import _get_embedder
                emb, st_util = _get_embedder()
                self.scorer = RelevanceScorer(
                    kernel=self.kernel, lexicon=self.lexicon,
                    embedder=emb, st_util=st_util,
                )
            except Exception:
                pass

        if self._gate is None:
            from .decision_gate import DecisionGate
            self._gate = DecisionGate(relevance_scorer=self.scorer)

        if self._verifier is None:
            from .verifier import AnswerVerifier
            # Pass embedder for grounding checks if available
            emb = getattr(self.scorer, 'embedder', None) if self.scorer else None
            st_u = getattr(self.scorer, 'st_util', None) if self.scorer else None
            self._verifier = AnswerVerifier(embedder=emb, st_util=st_u)

    def query(self, prompt, allow_forage=True, forage_timeout=30,
              stream=None, verbose=False):
        """
        Run the full v0.6 pipeline.

        Args:
            prompt: user query string
            allow_forage: enable auto-forage on low confidence
            forage_timeout: max seconds for foraging
            stream: optional StreamManager for SSE events
            verbose: print debug info

        Returns:
            dict with answer, source, confidence, relevance, stages, etc.
        """
        self._ensure_components()
        t0 = time.perf_counter()
        stages = {}

        from .evidence_store import EvidenceStore
        evidence_store = EvidenceStore()

        # ── Agent Router ──────────────────────────────────────────
        route = self._router.route(prompt)
        if stream:
            stream.routing(route.to_dict())
        if verbose:
            print(f"[PIPELINE] Route: {route.path} ({route.answer_type}, "
                  f"solver={route.solver}, conf={route.confidence:.2f})")

        stages["route"] = route.to_dict()

        # ── MATH FAST PATH ────────────────────────────────────────
        if route.solver == "math" and self.math_driver:
            if stream:
                stream.status("computing", "Deterministic math solver")
            result = self.math_driver.solve(prompt)
            if result.get("status") == "success":
                answer = f"{result['operation']}: {result['result']}"
                evidence_store.add_math_evidence(
                    answer, equation=result.get("equation", ""))
                latency = (time.perf_counter() - t0) * 1000
                stages["math"] = {"time_ms": round(latency, 1), **result}
                if stream:
                    stream.final(answer, citations=evidence_store.get_citations(),
                                 confidence=0.99, relevance=1.0,
                                 source="math", latency_ms=latency)
                return self._build_result(
                    prompt, answer, "math", evidence_store,
                    1.0, {}, 0, latency, stages, route)

        # ── Stage 1: RETRIEVE ─────────────────────────────────────
        t1 = time.perf_counter()
        if stream:
            stream.status("retrieving", "Spreading activation through graph")

        query_words = [w for w in re.findall(r'\w+', prompt.lower())
                       if len(w) > 2]

        raw_answer = self.shell.chat(prompt)

        # Get activated nodes for reranking
        activated = []
        if hasattr(self.shell, '_last_activated'):
            activated = self.shell._last_activated
        else:
            activated = sorted(
                ((uid, n.activation) for uid, n in self.kernel.nodes.items()
                 if n.activation > 0.01),
                key=lambda x: x[1], reverse=True
            )[:50]

        # Collect provenance evidence
        self._collect_evidence(query_words, evidence_store)

        stages["retrieve"] = {
            "time_ms": round((time.perf_counter() - t1) * 1000, 1),
            "candidates": len(activated),
            "evidence_found": len(evidence_store),
        }

        # ── Stage 2: RERANK ───────────────────────────────────────
        t2 = time.perf_counter()
        if stream:
            stream.status("reranking", f"Scoring {len(activated)} candidates")

        if activated and self.reranker:
            reranked = self.reranker.rerank(
                activated, self.kernel, query_words)
            top_nodes = reranked[:20]
        else:
            top_nodes = activated[:20] if activated else []

        stages["rerank"] = {
            "time_ms": round((time.perf_counter() - t2) * 1000, 1),
            "top_nodes": len(top_nodes),
        }

        # ── Stage 3: SYNTHESIZE ───────────────────────────────────
        t3 = time.perf_counter()
        if stream:
            stream.status("synthesizing", "Assembling answer from evidence")

        # --- Comparison-aware branch ---
        is_comparison = (route.answer_type == "comparison")
        comparison_entities = None
        if is_comparison:
            comparison_entities = _extract_comparison_entities(prompt)

        if is_comparison and comparison_entities and self.synthesizer:
            entity_a, entity_b = comparison_entities
            if stream:
                stream.status("synthesizing",
                              f"Comparing {entity_a} vs {entity_b}")

            # Collect per-entity evidence from provenance
            evidence_a = []
            evidence_b = []
            ea_lower, eb_lower = entity_a.lower(), entity_b.lower()
            for pair, texts in list(
                    getattr(self.kernel, 'provenance', {}).items()):
                for text in texts:
                    if len(text) <= 20:
                        continue
                    tl = text.lower()
                    has_a = ea_lower in tl
                    has_b = eb_lower in tl
                    if has_a and text not in evidence_a:
                        evidence_a.append(text)
                    if has_b and text not in evidence_b:
                        evidence_b.append(text)
                    if len(evidence_a) >= 8 and len(evidence_b) >= 8:
                        break

            # Also try per-entity graph retrieval for richer evidence
            for ent, elist, limit in [
                (entity_a, evidence_a, 8), (entity_b, evidence_b, 8)
            ]:
                if len(elist) < 3:
                    ent_words = [w for w in re.findall(r'\w+', ent.lower())
                                 if len(w) > 2]
                    for pair, texts in list(
                            getattr(self.kernel, 'provenance', {}).items()):
                        if any(w in str(pair).lower() for w in ent_words):
                            for text in texts:
                                if text not in elist and len(text) > 20:
                                    elist.append(text)
                                    if len(elist) >= limit:
                                        break
                        if len(elist) >= limit:
                            break

            synth_result = self.synthesizer.synthesize_comparison(
                entity_a=entity_a, entity_b=entity_b,
                evidence_a=evidence_a, evidence_b=evidence_b,
                raw_prompt=prompt)
            synth_answer = synth_result["response"]
            synth_confidence = synth_result["confidence"]
        else:
            # --- Default (non-comparison) synthesis ---
            # Collect provenance for reranked nodes
            synth_evidence = []
            for node_id, score in top_nodes[:10]:
                for pair, texts in list(
                        getattr(self.kernel, 'provenance', {}).items()):
                    if node_id in pair:
                        for text in texts:
                            if text not in synth_evidence and len(text) > 20:
                                synth_evidence.append(text)
                                if len(synth_evidence) >= 8:
                                    break
                    if len(synth_evidence) >= 8:
                        break

            if synth_evidence and self.synthesizer:
                synth_result = self.synthesizer.synthesize(
                    evidence=synth_evidence, raw_prompt=prompt)
                synth_answer = synth_result["response"]
                synth_confidence = synth_result["confidence"]
            else:
                synth_answer = raw_answer
                synth_confidence = 0.3

        # Choose best: raw shell answer vs synthesized
        # For comparisons, prefer synthesized (has structured side-by-side format)
        if is_comparison and comparison_entities:
            answer = synth_answer if synth_answer and len(synth_answer.strip()) > 10 else raw_answer
        else:
            answer = raw_answer if raw_answer and len(raw_answer.strip()) > 10 else synth_answer

        # Evidence count for stages dict
        if is_comparison and comparison_entities:
            _ev_count = len(evidence_a) + len(evidence_b) if 'evidence_a' in dir() else 0
        else:
            _ev_count = len(synth_evidence) if 'synth_evidence' in locals() else 0

        stages["synthesize"] = {
            "time_ms": round((time.perf_counter() - t3) * 1000, 1),
            "evidence_count": _ev_count,
            "synth_confidence": round(synth_confidence, 3),
        }

        # ── Stage 4: DECISION GATE ────────────────────────────────
        t4 = time.perf_counter()
        if stream:
            stream.status("evaluating", "4-layer relevance check")

        relevance_score = 0.5
        relevance_breakdown = {}
        if self.scorer and answer and len(answer.strip()) > 10:
            try:
                relevance_score, relevance_breakdown = self.scorer.score(
                    prompt, answer)
            except Exception:
                pass

        # ── Stage 4: VERIFY ──────────────────────────────────────
        t_verify = time.perf_counter()
        if stream:
            stream.status("verifying", "Post-synthesis verification")

        query_type = route.answer_type or "factual"
        entity_a_v = comparison_entities[0] if comparison_entities else None
        entity_b_v = comparison_entities[1] if comparison_entities else None

        # Collect evidence for grounding check
        verify_evidence = []
        if is_comparison and comparison_entities:
            if 'evidence_a' in dir() and 'evidence_b' in dir():
                verify_evidence = list(evidence_a) + list(evidence_b)
        elif 'synth_evidence' in locals() and synth_evidence:
            verify_evidence = list(synth_evidence)
        # Also pull from evidence_store if we have little
        if len(verify_evidence) < 2:
            for item in evidence_store.get_ranked(top_k=8):
                if hasattr(item, 'content') and item.content not in verify_evidence:
                    verify_evidence.append(item.content)

        # Extract factual topic entity for hard gate check
        if not entity_a_v and query_type == "factual":
            # Pull the main proper noun from the query
            _proper = re.findall(r'\b[A-Z][a-z]{2,}\b', prompt)
            if _proper:
                entity_a_v = _proper[0]

        verification = self._verifier.verify(
            query=prompt, answer=answer, query_type=query_type,
            entity_a=entity_a_v, entity_b=entity_b_v,
            evidence=verify_evidence)

        # Apply verification adjustment to relevance
        relevance_score = max(0.0, min(1.0,
            relevance_score + verification.score_adjustment))
        coverage_factor = max(0.5, 1.0 + verification.score_adjustment)

        # Hard fail caps the relevance score below SPEAK threshold
        if verification.hard_fail:
            relevance_score = min(relevance_score, 0.49)
            if verbose:
                print(f"[PIPELINE] HARD FAIL: {verification.failure_tags} "
                      f"-- score capped at {relevance_score:.3f}")

        stages["verify"] = {
            "time_ms": round((time.perf_counter() - t_verify) * 1000, 1),
            "trust_label": verification.trust_label,
            "score_adjustment": round(verification.score_adjustment, 3),
            "completeness": round(verification.completeness_score, 3),
            "issues": verification.issues,
            "contradictions": verification.contradiction_flags,
            "hard_fail": verification.hard_fail,
            "failure_tags": verification.failure_tags,
            "grounding_score": round(verification.grounding_score, 3),
        }

        if verbose and verification.issues:
            print(f"[PIPELINE] Verify: {verification.trust_label} "
                  f"(adj={verification.score_adjustment:+.3f}, "
                  f"issues={verification.issues})")

        # ── Detect coverage gaps (runs for ALL queries) ──
        coverage_gaps = self._detect_coverage_gaps(prompt, answer, route, verbose)
        missing_entities = [w for w, t in coverage_gaps if t == "missing_entity"]
        stages["gap_detector"] = {
            "gaps": [(w, t) for w, t in coverage_gaps],
            "missing": len(missing_entities),
        }

        # ── Stage 5: DECISION GATE (v0.7 policy upgrade) ────────
        from .decision_gate import Decision
        gate_result = self._gate.decide(
            query=prompt,
            answer=answer,
            evidence_count=len(evidence_store),
            relevance_score=relevance_score,
            route_path=route.path,
        )

        # v0.7.1 policy override: hard_fail or contradiction → downgrade
        _needs_override = (
            (verification.hard_fail and gate_result.decision == Decision.SPEAK)
            or (verification.contradiction_flags
                and gate_result.decision == Decision.SPEAK)
        )
        if _needs_override:
            override_score = max(0.3, relevance_score - 0.15)
            gate_result = self._gate.decide(
                query=prompt, answer=answer,
                evidence_count=len(evidence_store),
                relevance_score=override_score,
                route_path=route.path,
            )
            if verbose:
                reason = ("hard_fail" if verification.hard_fail
                          else "contradiction")
                print(f"[PIPELINE] Policy override: {reason} detected, "
                      f"downgraded to {gate_result.decision.value}")

        if stream:
            stream.gate(gate_result.to_dict())

        stages["gate"] = {
            "time_ms": round((time.perf_counter() - t4) * 1000, 1),
            "decision": gate_result.decision.value,
            "relevance": round(relevance_score, 3),
        }

        if verbose:
            print(f"[PIPELINE] Gate: {gate_result.decision.value} "
                  f"(relevance={relevance_score:.3f}, "
                  f"trust={verification.trust_label})")

        # ── SPEAK: return immediately ─────────────────────────────
        if gate_result.decision == Decision.SPEAK:
            latency = (time.perf_counter() - t0) * 1000
            if stream:
                stream.final(answer, citations=evidence_store.get_citations(),
                             confidence=gate_result.confidence,
                             relevance=relevance_score,
                             source="graph", latency_ms=latency,
                             breakdown=relevance_breakdown)
            result = self._build_result(
                prompt, answer, "graph", evidence_store,
                relevance_score, relevance_breakdown, 0, latency, stages, route,
                coverage_factor=coverage_factor,
                coverage_gaps=coverage_gaps,
                trust_label=verification.trust_label)
            self.memory.record_from_result(result)
            return result

        # ── STREAM PARTIAL (if applicable) ────────────────────────
        if gate_result.decision in (Decision.STREAM_PARTIAL, Decision.ESCALATE):
            if stream:
                stream.partial(answer, confidence=gate_result.confidence)

        # ── FORAGE ────────────────────────────────────────────────
        foraged_nodes = 0
        source = "graph"

        if allow_forage and self.forager_factory:
            if stream:
                stream.status("foraging", "Searching internet (DDG + Wikipedia + Google)")

            t_forage = time.perf_counter()

            # For comparisons, forage each entity separately
            if is_comparison and comparison_entities:
                entity_a, entity_b = comparison_entities
                foraged_nodes += self._do_forage(entity_a, forage_timeout // 2, verbose)
                foraged_nodes += self._do_forage(entity_b, forage_timeout // 2, verbose)
            elif missing_entities:
                # Targeted forage for missing concepts
                for entity in missing_entities[:3]:  # Cap at 3 to avoid timeout
                    foraged_nodes += self._do_forage(entity, forage_timeout // 3, verbose)
            else:
                foraged_nodes = self._do_forage(prompt, forage_timeout, verbose)

            if foraged_nodes > 0:
                if stream:
                    stream.status("re-retrieving",
                                  f"+{foraged_nodes} concepts, re-querying")

                # Invalidate embeddings and re-query
                self.shell.node_embeddings = None
                self.shell.embedded_uuids = []
                self.shell._word_emb_cache = {}
                self.shell._ensure_embeddings()

                # For comparisons, re-run comparison synthesis
                if is_comparison and comparison_entities and self.synthesizer:
                    entity_a, entity_b = comparison_entities
                    evidence_a, evidence_b = [], []
                    ea_lower, eb_lower = entity_a.lower(), entity_b.lower()
                    for pair, texts in list(
                            getattr(self.kernel, 'provenance', {}).items()):
                        for text in texts:
                            if len(text) <= 20:
                                continue
                            tl = text.lower()
                            if ea_lower in tl and text not in evidence_a:
                                evidence_a.append(text)
                            if eb_lower in tl and text not in evidence_b:
                                evidence_b.append(text)
                            if len(evidence_a) >= 10 and len(evidence_b) >= 10:
                                break
                    synth_result = self.synthesizer.synthesize_comparison(
                        entity_a=entity_a, entity_b=entity_b,
                        evidence_a=evidence_a, evidence_b=evidence_b,
                        raw_prompt=prompt)
                    answer = synth_result["response"]
                else:
                    answer = self.shell.chat(prompt)
                source = "internet"

                # Re-score
                if self.scorer:
                    try:
                        relevance_score, relevance_breakdown = self.scorer.score(
                            prompt, answer)
                    except Exception:
                        pass

            stages["forage"] = {
                "time_ms": round((time.perf_counter() - t_forage) * 1000, 1),
                "nodes_added": foraged_nodes,
                "triggered_by": gate_result.decision.value,
            }

        # ── Final Assembly ────────────────────────────────────────
        latency = (time.perf_counter() - t0) * 1000
        if stream:
            stream.final(answer, citations=evidence_store.get_citations(),
                         confidence=relevance_score, relevance=relevance_score,
                         source=source, foraged_nodes=foraged_nodes,
                         latency_ms=latency, breakdown=relevance_breakdown)

        result = self._build_result(
            prompt, answer, source, evidence_store,
            relevance_score, relevance_breakdown,
            foraged_nodes, latency, stages, route,
            coverage_gaps=coverage_gaps, coverage_factor=coverage_factor,
            trust_label=verification.trust_label)
        self.memory.record_from_result(result)
        return result

    def _collect_evidence(self, query_words, store):
        """Collect provenance evidence matching query words."""
        stop = {'what', 'is', 'the', 'a', 'an', 'of', 'in', 'to', 'for',
                'and', 'or', 'how', 'does', 'tell', 'me', 'about', 'are'}
        content_words = set(query_words) - stop
        try:
            for (src, tgt), texts in list(
                    getattr(self.kernel, 'provenance', {}).items()):
                for text in texts:
                    if any(w in text.lower() for w in content_words):
                        store.add_graph_evidence(src, text, 0.5, 0.7)
        except RuntimeError:
            pass

    def _do_forage(self, prompt, timeout, verbose):
        """Run foraging in a thread with timeout."""
        if not self.forager_factory:
            return 0
        result = [0]
        def _forage():
            try:
                forager = self.forager_factory()
                before = len(self.kernel.nodes)
                forager.forage_query(prompt, verbose=verbose)
                result[0] = len(self.kernel.nodes) - before
            except Exception as e:
                if verbose:
                    print(f"[PIPELINE] Forage error: {e}")
        t = threading.Thread(target=_forage)
        t.start()
        t.join(timeout=timeout)
        return result[0]

    def _detect_coverage_gaps(self, prompt, answer, route, verbose=False):
        """
        Detect missing concepts in the knowledge graph for this query.

        Returns:
            list of (concept_word, gap_type) tuples where gap_type is:
            - "missing_entity": query entity not in graph at all
            - "shallow_entity": entity exists but has < 3 provenance edges
            - "missing_attribute": entity exists but queried attribute not found

        Used to:
        1. Trigger targeted forage for specific missing concepts
        2. Log gaps for corpus improvement
        3. Adjust confidence based on coverage
        """
        gaps = []
        stop = {'what', 'is', 'the', 'a', 'an', 'of', 'in', 'to', 'for',
                'and', 'or', 'how', 'does', 'tell', 'me', 'about', 'are',
                'compare', 'between', 'difference', 'which', 'better',
                'vs', 'versus', 'compared', 'with', 'from', 'pros', 'cons'}

        query_words = [w for w in re.findall(r'\w+', prompt.lower())
                       if len(w) > 2 and w not in stop]

        for word in query_words:
            # Check if word resolves to a graph node
            uid = None
            if self.lexicon and word in self.lexicon.word_to_uuid:
                uid = self.lexicon.word_to_uuid[word]

            if uid is None:
                gaps.append((word, "missing_entity"))
                continue

            if uid not in self.kernel.nodes:
                gaps.append((word, "missing_entity"))
                continue

            # Check provenance depth
            prov_count = 0
            for pair, texts in getattr(self.kernel, 'provenance', {}).items():
                if uid in pair:
                    prov_count += len(texts)

            if prov_count < 3:
                gaps.append((word, "shallow_entity"))

        if verbose and gaps:
            print(f"[PIPELINE] Coverage gaps: {gaps}")

        return gaps

    def _build_result(self, prompt, answer, source, evidence_store,
                      relevance_score, breakdown, foraged, latency,
                      stages, route, coverage_gaps=None, coverage_factor=1.0,
                      trust_label="unverified"):
        return {
            "prompt": prompt,
            "answer": (answer or "I don't have data on this topic.").strip(),
            "source": source,
            "trust_label": trust_label,
            "latency_ms": round(latency, 1),
            "timestamp": time.time(),
            "nodes_activated": len(self.kernel.nodes),
            "foraged_nodes": foraged,
            "relevance_score": round(relevance_score, 3),
            "relevance_breakdown": breakdown,
            "coverage_factor": round(coverage_factor, 3),
            "off_topic_detected": relevance_score < 0.46,
            "evidence_count": len(evidence_store),
            "evidence_sources": evidence_store.summary().get("sources", {}),
            "coverage_gaps": coverage_gaps or [],
            "route": route.to_dict(),
            "stages": stages,
        }
