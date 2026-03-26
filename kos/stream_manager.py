"""
KOS v0.6 -- Stream Manager (SSE Event Generator)

Generates Server-Sent Events for real-time query progress:
  - status: "retrieving", "reranking", "synthesizing", "foraging"
  - evidence: individual evidence cards as they are found
  - partial: partial answer (before forage completes)
  - final: complete answer with citations
  - error: failure with reason

Timing targets:
  <500ms: acknowledge + first status
  <2s: first evidence card
  <6s: final fast-path answer
  Long tasks: progress events while background continues
"""

import json
import time
from typing import Generator


class StreamEvent:
    """A single SSE event."""
    __slots__ = ['event', 'data', 'timestamp']

    def __init__(self, event: str, data: dict):
        self.event = event
        self.data = data
        self.timestamp = time.time()

    def to_sse(self) -> str:
        """Format as Server-Sent Event string."""
        payload = json.dumps(self.data, default=str)
        return f"event: {self.event}\ndata: {payload}\n\n"


class StreamManager:
    """
    Manages a stream of SSE events for a single query lifecycle.

    Usage in FastAPI:
        stream = StreamManager(query)
        stream.status("retrieving")
        stream.evidence({"content": "...", "source": "graph"})
        stream.partial("The Moon is...")
        stream.final("The Moon orbits Earth at 384,400 km.", citations=[...])

        # In endpoint:
        return StreamingResponse(stream.generate(), media_type="text/event-stream")
    """

    def __init__(self, query: str):
        self.query = query
        self.events: list[StreamEvent] = []
        self.start_time = time.time()
        self._closed = False

        # Auto-add acknowledge event
        self.events.append(StreamEvent("ack", {
            "query": query,
            "timestamp": self.start_time,
        }))

    def status(self, stage: str, detail: str = ""):
        """Emit a status update."""
        if self._closed:
            return
        self.events.append(StreamEvent("status", {
            "stage": stage,
            "detail": detail,
            "elapsed_ms": round((time.time() - self.start_time) * 1000, 1),
        }))

    def evidence(self, item: dict):
        """Emit an evidence card."""
        if self._closed:
            return
        self.events.append(StreamEvent("evidence", {
            "content": item.get("content", ""),
            "source": item.get("source", "unknown"),
            "trust": item.get("trust_score", 0.5),
            "citation": item.get("citation", ""),
        }))

    def partial(self, answer: str, confidence: float = 0.0):
        """Emit a partial answer (may be refined later)."""
        if self._closed:
            return
        self.events.append(StreamEvent("partial", {
            "answer": answer,
            "confidence": round(confidence, 3),
            "elapsed_ms": round((time.time() - self.start_time) * 1000, 1),
        }))

    def routing(self, route_info: dict):
        """Emit routing decision."""
        if self._closed:
            return
        self.events.append(StreamEvent("routing", route_info))

    def gate(self, gate_info: dict):
        """Emit confidence gate decision."""
        if self._closed:
            return
        self.events.append(StreamEvent("gate", gate_info))

    def final(self, answer: str, citations: list = None,
              confidence: float = 0.0, relevance: float = 0.0,
              source: str = "graph", foraged_nodes: int = 0,
              latency_ms: float = 0.0, breakdown: dict = None):
        """Emit final answer and close stream."""
        self.events.append(StreamEvent("final", {
            "answer": answer,
            "citations": citations or [],
            "confidence": round(confidence, 3),
            "relevance_score": round(relevance, 3),
            "source": source,
            "foraged_nodes": foraged_nodes,
            "latency_ms": round(latency_ms, 1),
            "relevance_breakdown": breakdown or {},
        }))
        self._closed = True

    def error(self, message: str):
        """Emit error and close stream."""
        self.events.append(StreamEvent("error", {
            "message": message,
            "elapsed_ms": round((time.time() - self.start_time) * 1000, 1),
        }))
        self._closed = True

    def generate(self) -> Generator[str, None, None]:
        """
        Generator that yields SSE events.
        Used with FastAPI StreamingResponse.

        NOTE: In v0.6, events are buffered then yielded.
        In v0.7+, this will be async with real-time event pushing.
        """
        for event in self.events:
            yield event.to_sse()

    def to_json(self) -> dict:
        """
        Return all events as a JSON-serializable dict.
        For non-streaming clients that want the full result.
        """
        if not self.events:
            return {"events": [], "query": self.query}

        # Find the final event
        final_event = None
        for e in reversed(self.events):
            if e.event == "final":
                final_event = e
                break

        return {
            "query": self.query,
            "events": [{"event": e.event, "data": e.data} for e in self.events],
            "answer": final_event.data.get("answer", "") if final_event else "",
            "latency_ms": final_event.data.get("latency_ms", 0) if final_event else 0,
            "source": final_event.data.get("source", "graph") if final_event else "graph",
            "foraged_nodes": final_event.data.get("foraged_nodes", 0) if final_event else 0,
            "relevance_score": final_event.data.get("relevance_score", 0) if final_event else 0,
        }
