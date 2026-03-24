"""
KOS V6.0 — Proposal 1: Self-Referential Graph

KOS adds itself as a node in its own knowledge graph.
It can now model what it knows, what it believes, what it's
uncertain about, and what it has learned.

This enables:
- "What do you know about Toronto?" → queries self-model
- "What are you uncertain about?" → lists low-confidence beliefs
- "How did you learn this?" → traces provenance + learning path
- "What have you learned today?" → temporal learning log

All self-knowledge is queryable through the same graph physics
as any other knowledge. Meta-cognition IS data.

Safety: read-only self-model. KOS can observe itself but cannot
modify its own architecture through self-referential queries.
"""

import time
import re
from collections import defaultdict


class SelfModel:
    """
    KOS's model of itself. Maintains:
    - What it knows (belief inventory)
    - Confidence per belief (from prediction accuracy)
    - Learning history (when/how each fact was acquired)
    - Current state (active queries, entropy, emotion)
    - Capabilities inventory (what it can and cannot do)
    """

    def __init__(self, kernel, lexicon, pce=None):
        self.kernel = kernel
        self.lexicon = lexicon
        self.pce = pce

        # Learning log: uuid -> {learned_at, source, confidence, query_count}
        self._belief_log = {}
        self._learning_timeline = []  # (timestamp, event_type, details)
        self._query_history = []
        self._state_snapshots = []

        # Wire KOS as a node in its own graph
        self.kos_uid = lexicon.get_or_create_id("kos_self")
        kernel.add_node(self.kos_uid)

        # Log boot
        self._log_event("boot", "KOS Self-Model initialized")

    # ── Belief Tracking ──────────────────────────────────

    def register_belief(self, concept_uid: str, confidence: float = 0.5,
                        source: str = "ingestion"):
        """Register that KOS knows something, with confidence."""
        word = self.lexicon.get_word(concept_uid) if hasattr(self.lexicon, 'get_word') else str(concept_uid)

        self._belief_log[concept_uid] = {
            "word": word,
            "confidence": confidence,
            "source": source,
            "learned_at": time.time(),
            "query_count": 0,
            "last_queried": None,
        }

        # Wire self-referential edge: KOS -> knows -> concept
        self.kernel.add_connection(
            self.kos_uid, concept_uid, confidence * 0.5,
            "[SELF-MODEL] KOS believes: %s (confidence=%.2f)" % (word, confidence))

        self._log_event("learn", "Learned '%s' (confidence=%.2f, source=%s)" % (
            word, confidence, source))

    def update_confidence(self, concept_uid: str, new_confidence: float,
                           reason: str = "prediction_update"):
        """Update confidence in a belief."""
        if concept_uid in self._belief_log:
            old = self._belief_log[concept_uid]["confidence"]
            self._belief_log[concept_uid]["confidence"] = new_confidence
            self._log_event("confidence_change",
                            "Confidence in '%s': %.2f -> %.2f (%s)" % (
                                self._belief_log[concept_uid]["word"],
                                old, new_confidence, reason))

    def record_query(self, query: str, answer: str, latency_ms: float):
        """Record that a query was made and what was answered."""
        self._query_history.append({
            "query": query,
            "answer": answer[:100],
            "latency_ms": latency_ms,
            "timestamp": time.time(),
        })
        self._log_event("query", "Q: '%s' -> A: '%s' (%.0fms)" % (
            query[:50], answer[:50], latency_ms))

    # ── Self-Knowledge Queries ───────────────────────────

    def what_do_i_know(self, min_confidence: float = 0.0) -> list:
        """Return all beliefs above a confidence threshold."""
        beliefs = []
        for uid, info in self._belief_log.items():
            if info["confidence"] >= min_confidence:
                beliefs.append({
                    "concept": info["word"],
                    "confidence": round(info["confidence"], 3),
                    "source": info["source"],
                    "queries": info["query_count"],
                })
        beliefs.sort(key=lambda x: x["confidence"], reverse=True)
        return beliefs

    def what_am_i_uncertain_about(self, threshold: float = 0.4) -> list:
        """Return beliefs with confidence below threshold."""
        uncertain = []
        for uid, info in self._belief_log.items():
            if info["confidence"] < threshold:
                uncertain.append({
                    "concept": info["word"],
                    "confidence": round(info["confidence"], 3),
                    "source": info["source"],
                })
        uncertain.sort(key=lambda x: x["confidence"])
        return uncertain

    def what_did_i_learn_recently(self, minutes: int = 60) -> list:
        """Return beliefs learned in the last N minutes."""
        cutoff = time.time() - (minutes * 60)
        recent = []
        for uid, info in self._belief_log.items():
            if info["learned_at"] >= cutoff:
                recent.append({
                    "concept": info["word"],
                    "learned_ago_min": round((time.time() - info["learned_at"]) / 60, 1),
                    "source": info["source"],
                })
        return recent

    def how_did_i_learn(self, concept: str) -> dict:
        """Trace how a specific concept was learned."""
        uid = self.lexicon.word_to_uuid.get(concept.lower())
        if uid and uid in self._belief_log:
            info = self._belief_log[uid]
            # Find provenance
            provenance = set()
            for (a, b), sents in self.kernel.provenance.items():
                if uid in (a, b):
                    provenance.update(sents)
            return {
                "concept": concept,
                "source": info["source"],
                "confidence": info["confidence"],
                "provenance": list(provenance)[:5],
                "query_count": info["query_count"],
            }
        return {"concept": concept, "status": "unknown"}

    def my_capabilities(self) -> dict:
        """Inventory of what KOS can and cannot do."""
        return {
            "can_do": [
                "Factual retrieval with zero hallucination",
                "6-layer typo recovery",
                "Multi-hop graph reasoning (14 hops proven)",
                "Exact math via SymPy",
                "Cross-domain analogy via KASM",
                "Autonomous web foraging",
                "Self-diagnosis and fix proposals",
                "Predict activation patterns",
                "Detect and resolve contradictions",
                "Chemistry computations (bonds, reactions, pH)",
                "Physics computations (mechanics through relativity)",
                "Biology computations (genetics, pharmacology, ecology)",
                "Emotion modeling (neurochemical vectors)",
                "Social modeling (game theory, trust)",
                "Autonomous hypothesis testing",
            ],
            "cannot_do": [
                "Generate creative text (poetry, fiction)",
                "Translate between languages fluently",
                "Process images or audio",
                "Prove I am conscious",
                "Modify my own source code",
                "Guarantee real-world experimental outcomes",
            ],
            "uncertain_about": [
                "Whether I understand or just compute",
                "Whether my emotion model FEELS like anything",
                "Whether consciousness requires embodiment",
            ],
        }

    def my_current_state(self) -> dict:
        """Snapshot of current internal state."""
        total_nodes = len(self.kernel.nodes)
        total_edges = sum(len(n.connections) for n in self.kernel.nodes.values())
        total_beliefs = len(self._belief_log)
        avg_confidence = (
            sum(b["confidence"] for b in self._belief_log.values()) / total_beliefs
            if total_beliefs > 0 else 0
        )

        state = {
            "nodes": total_nodes,
            "edges": total_edges,
            "beliefs_tracked": total_beliefs,
            "avg_confidence": round(avg_confidence, 3),
            "queries_answered": len(self._query_history),
            "events_logged": len(self._learning_timeline),
            "uptime_events": len(self._learning_timeline),
        }

        # Add prediction stats if available
        if self.pce:
            pce_stats = self.pce.get_stats()
            state["prediction_accuracy"] = round(pce_stats["overall_accuracy"], 3)
            state["predictions_cached"] = pce_stats["cached_predictions"]

        return state

    # ── Sync with Graph ──────────────────────────────────

    def sync_beliefs_from_graph(self):
        """
        Scan the kernel and register all known concepts as beliefs.
        Called after ingestion to keep self-model in sync.
        """
        for uid in self.kernel.nodes:
            if uid != self.kos_uid and uid not in self._belief_log:
                word = self.lexicon.get_word(uid) if hasattr(self.lexicon, 'get_word') else None
                if word:
                    # Confidence based on number of connections
                    connections = len(self.kernel.nodes[uid].connections)
                    confidence = min(1.0, connections * 0.1)
                    self.register_belief(uid, confidence, "graph_sync")

    # ── Internal Logging ─────────────────────────────────

    def _log_event(self, event_type: str, details: str):
        """Log an internal event to the timeline."""
        self._learning_timeline.append({
            "time": time.time(),
            "type": event_type,
            "details": details,
        })

    def get_timeline(self, last_n: int = 20) -> list:
        """Get the last N events from the learning timeline."""
        return self._learning_timeline[-last_n:]

    # ── String Representation ────────────────────────────

    def introspect(self) -> str:
        """Full self-report as human-readable text."""
        state = self.my_current_state()
        uncertain = self.what_am_i_uncertain_about()
        recent = self.what_did_i_learn_recently(60)

        lines = []
        lines.append("=== KOS SELF-MODEL INTROSPECTION ===")
        lines.append("Nodes: %d | Edges: %d | Beliefs: %d" % (
            state["nodes"], state["edges"], state["beliefs_tracked"]))
        lines.append("Avg confidence: %.1f%%" % (state["avg_confidence"] * 100))
        lines.append("Queries answered: %d" % state["queries_answered"])

        if uncertain:
            lines.append("\nUncertain about (%d concepts):" % len(uncertain))
            for u in uncertain[:5]:
                lines.append("  - %s (confidence=%.1f%%)" % (
                    u["concept"], u["confidence"] * 100))

        if recent:
            lines.append("\nLearned recently (%d concepts):" % len(recent))
            for r in recent[:5]:
                lines.append("  - %s (%.0f min ago, source=%s)" % (
                    r["concept"], r["learned_ago_min"], r["source"]))

        return "\n".join(lines)
