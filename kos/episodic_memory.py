"""
KOS v0.7 -- Episodic Memory (Lightweight Query History)

Stores per-query episodes for:
  - Debugging (what went wrong?)
  - Planner learning (which routes work best?)
  - Performance monitoring (latency trends, score trends)
  - Failure recovery (retry failed patterns with better strategy)

Storage: in-memory ring buffer (last 500 queries) + optional JSON persist.
No external dependencies. Thread-safe.
"""

import time
import threading
import json
import os


class Episode:
    """Single query episode record."""
    __slots__ = ['query', 'answer', 'route', 'answer_type', 'solver',
                 'relevance_score', 'trust_label', 'latency_ms',
                 'foraged_nodes', 'coverage_gaps', 'failure_type',
                 'timestamp', 'source']

    def __init__(self, query, answer="", route="fast", answer_type="factual",
                 solver=None, relevance_score=0.0, trust_label="unverified",
                 latency_ms=0.0, foraged_nodes=0, coverage_gaps=None,
                 failure_type=None, source="graph"):
        self.query = query
        self.answer = answer[:500]  # Truncate for memory
        self.route = route
        self.answer_type = answer_type
        self.solver = solver
        self.relevance_score = relevance_score
        self.trust_label = trust_label
        self.latency_ms = latency_ms
        self.foraged_nodes = foraged_nodes
        self.coverage_gaps = coverage_gaps or []
        self.failure_type = failure_type  # None, "low_score", "no_data", "timeout", "contradiction"
        self.timestamp = time.time()
        self.source = source

    def to_dict(self):
        return {
            "query": self.query,
            "answer": self.answer,
            "route": self.route,
            "answer_type": self.answer_type,
            "solver": self.solver,
            "relevance_score": round(self.relevance_score, 3),
            "trust_label": self.trust_label,
            "latency_ms": round(self.latency_ms, 1),
            "foraged_nodes": self.foraged_nodes,
            "coverage_gaps": self.coverage_gaps,
            "failure_type": self.failure_type,
            "timestamp": self.timestamp,
            "source": self.source,
        }


class EpisodicMemory:
    """
    Thread-safe ring buffer of query episodes.

    Usage:
        memory = EpisodicMemory(max_episodes=500)
        memory.record(episode)
        stats = memory.stats()
        failures = memory.recent_failures(n=10)
    """

    def __init__(self, max_episodes=500, persist_path=None):
        self._episodes = []
        self._max = max_episodes
        self._lock = threading.Lock()
        self._persist_path = persist_path

        # Load from disk if available
        if persist_path and os.path.exists(persist_path):
            try:
                with open(persist_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for d in data[-max_episodes:]:
                    ep = Episode(
                        query=d.get("query", ""),
                        answer=d.get("answer", ""),
                        route=d.get("route", "fast"),
                        answer_type=d.get("answer_type", "factual"),
                        solver=d.get("solver"),
                        relevance_score=d.get("relevance_score", 0),
                        trust_label=d.get("trust_label", "unverified"),
                        latency_ms=d.get("latency_ms", 0),
                        foraged_nodes=d.get("foraged_nodes", 0),
                        coverage_gaps=d.get("coverage_gaps", []),
                        failure_type=d.get("failure_type"),
                        source=d.get("source", "graph"),
                    )
                    ep.timestamp = d.get("timestamp", 0)
                    self._episodes.append(ep)
            except Exception:
                pass

    def record(self, episode):
        """Add an episode to the ring buffer."""
        with self._lock:
            self._episodes.append(episode)
            if len(self._episodes) > self._max:
                self._episodes = self._episodes[-self._max:]

    def record_from_result(self, result):
        """Create and record an Episode from a pipeline result dict."""
        route_info = result.get("route", {})

        # Determine failure type
        failure_type = None
        rel = result.get("relevance_score", 0)
        answer = result.get("answer", "")
        if rel < 0.46:
            failure_type = "low_score"
        if any(p in answer.lower() for p in
               ["i don't have", "no data", "no information"]):
            failure_type = "no_data"

        trust = result.get("trust_label", "unverified")

        ep = Episode(
            query=result.get("prompt", ""),
            answer=answer,
            route=route_info.get("path", "fast"),
            answer_type=route_info.get("answer_type", "factual"),
            solver=route_info.get("solver"),
            relevance_score=rel,
            trust_label=trust,
            latency_ms=result.get("latency_ms", 0),
            foraged_nodes=result.get("foraged_nodes", 0),
            coverage_gaps=result.get("coverage_gaps", []),
            failure_type=failure_type,
            source=result.get("source", "graph"),
        )
        self.record(ep)
        return ep

    def recent(self, n=20):
        """Get the N most recent episodes."""
        with self._lock:
            return [e.to_dict() for e in self._episodes[-n:]]

    def recent_failures(self, n=10):
        """Get the N most recent failed episodes."""
        with self._lock:
            failures = [e for e in self._episodes if e.failure_type]
            return [e.to_dict() for e in failures[-n:]]

    def stats(self):
        """Compute aggregate statistics."""
        with self._lock:
            if not self._episodes:
                return {"total": 0}

            eps = self._episodes
            scores = [e.relevance_score for e in eps]
            latencies = [e.latency_ms for e in eps]

            # Count by route
            by_route = {}
            for e in eps:
                key = f"{e.route}/{e.answer_type}"
                if key not in by_route:
                    by_route[key] = {"count": 0, "avg_score": 0, "scores": []}
                by_route[key]["count"] += 1
                by_route[key]["scores"].append(e.relevance_score)

            for key in by_route:
                s = by_route[key]["scores"]
                by_route[key]["avg_score"] = round(sum(s) / len(s), 3)
                del by_route[key]["scores"]

            # Count by trust label
            by_trust = {}
            for e in eps:
                by_trust[e.trust_label] = by_trust.get(e.trust_label, 0) + 1

            # Failure analysis
            failure_count = sum(1 for e in eps if e.failure_type)
            failure_types = {}
            for e in eps:
                if e.failure_type:
                    failure_types[e.failure_type] = failure_types.get(e.failure_type, 0) + 1

            # Common coverage gaps
            gap_counts = {}
            for e in eps:
                for word, gtype in e.coverage_gaps:
                    gap_counts[word] = gap_counts.get(word, 0) + 1
            top_gaps = sorted(gap_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            return {
                "total": len(eps),
                "avg_score": round(sum(scores) / len(scores), 3),
                "min_score": round(min(scores), 3),
                "max_score": round(max(scores), 3),
                "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
                "pass_rate": round(sum(1 for s in scores if s >= 0.55) / len(scores), 3),
                "failure_count": failure_count,
                "failure_rate": round(failure_count / len(eps), 3),
                "failure_types": failure_types,
                "by_route": by_route,
                "by_trust": by_trust,
                "top_coverage_gaps": top_gaps,
            }

    def save(self):
        """Persist episodes to disk."""
        if not self._persist_path:
            return
        with self._lock:
            try:
                os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
                with open(self._persist_path, 'w', encoding='utf-8') as f:
                    json.dump([e.to_dict() for e in self._episodes], f)
            except Exception:
                pass

    def __len__(self):
        with self._lock:
            return len(self._episodes)
