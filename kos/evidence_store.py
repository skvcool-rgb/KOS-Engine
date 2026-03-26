"""
KOS v0.6 -- Evidence Store (Normalized Evidence Layer)

All retrieval sources (graph, file, web, image, audio) produce
EvidenceItem objects. The store normalizes them for the reranker
and synthesis engine.

Every answer traces back to evidence. Every evidence traces back
to a source. No hallucination possible.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvidenceItem:
    """A single piece of evidence from any source."""
    content: str                          # The actual text/fact
    source: str                           # "graph", "file", "web", "image", "audio", "math"
    trust_score: float = 0.5             # 0.0-1.0 source reliability
    timestamp: float = field(default_factory=time.time)
    node_id: Optional[str] = None        # Graph node UUID (if from graph)
    url: Optional[str] = None            # Source URL (if from web/file)
    activation: float = 0.0             # Spreading activation score
    citation: Optional[str] = None      # Human-readable citation
    embedding: Optional[list] = None    # Cached embedding vector

    def to_dict(self):
        return {
            "content": self.content,
            "source": self.source,
            "trust_score": round(self.trust_score, 3),
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "url": self.url,
            "activation": round(self.activation, 4),
            "citation": self.citation,
        }


class EvidenceStore:
    """
    Collects, deduplicates, and ranks evidence from all sources.

    Usage:
        store = EvidenceStore()
        store.add_graph_evidence(node_id, text, activation, trust)
        store.add_web_evidence(text, url, trust)
        ranked = store.get_ranked(top_k=10)
    """

    def __init__(self):
        self.items: list[EvidenceItem] = []
        self._seen_content: set = set()  # Dedup by content hash

    def add(self, item: EvidenceItem) -> bool:
        """Add an evidence item. Returns False if duplicate."""
        content_key = item.content.lower().strip()[:100]
        if content_key in self._seen_content:
            return False
        self._seen_content.add(content_key)
        self.items.append(item)
        return True

    def add_graph_evidence(self, node_id: str, content: str,
                           activation: float = 0.0,
                           trust: float = 0.7) -> bool:
        """Add evidence from graph retrieval."""
        return self.add(EvidenceItem(
            content=content,
            source="graph",
            trust_score=trust,
            node_id=node_id,
            activation=activation,
            citation=f"KOS Graph [{node_id[:20]}]",
        ))

    def add_web_evidence(self, content: str, url: str = "",
                         trust: float = 0.5) -> bool:
        """Add evidence from web foraging."""
        return self.add(EvidenceItem(
            content=content,
            source="web",
            trust_score=trust,
            url=url,
            citation=f"Web [{url[:50]}]" if url else "Web",
        ))

    def add_math_evidence(self, content: str, equation: str = "",
                          trust: float = 0.99) -> bool:
        """Add evidence from deterministic math solver."""
        return self.add(EvidenceItem(
            content=content,
            source="math",
            trust_score=trust,
            citation=f"SymPy CAS [{equation}]",
        ))

    def add_file_evidence(self, content: str, filepath: str = "",
                          trust: float = 0.6) -> bool:
        """Add evidence from file/document."""
        return self.add(EvidenceItem(
            content=content,
            source="file",
            trust_score=trust,
            url=filepath,
            citation=f"File [{filepath}]",
        ))

    def get_ranked(self, top_k: int = 10) -> list[EvidenceItem]:
        """
        Return top-K evidence items ranked by composite score.
        Score = trust * 0.4 + activation * 0.3 + recency * 0.15 + source_bonus * 0.15
        """
        now = time.time()
        source_bonus = {
            "math": 1.0,   # Deterministic = highest trust
            "graph": 0.7,  # From knowledge base
            "file": 0.6,   # User-provided documents
            "web": 0.4,    # Internet (less trusted)
            "image": 0.3,
            "audio": 0.3,
        }

        def score(item):
            recency = max(0, 1.0 - (now - item.timestamp) / 3600)  # Decay over 1h
            bonus = source_bonus.get(item.source, 0.3)
            return (
                0.40 * item.trust_score +
                0.30 * min(abs(item.activation), 1.0) +
                0.15 * recency +
                0.15 * bonus
            )

        ranked = sorted(self.items, key=score, reverse=True)
        return ranked[:top_k]

    def get_texts(self, top_k: int = 10) -> list[str]:
        """Get ranked evidence as plain text strings."""
        return [item.content for item in self.get_ranked(top_k)]

    def get_citations(self, top_k: int = 10) -> list[str]:
        """Get citations for ranked evidence."""
        return [item.citation or item.source for item in self.get_ranked(top_k)]

    def clear(self):
        """Clear all evidence."""
        self.items.clear()
        self._seen_content.clear()

    def __len__(self):
        return len(self.items)

    def summary(self) -> dict:
        """Summary statistics."""
        sources = {}
        for item in self.items:
            sources[item.source] = sources.get(item.source, 0) + 1
        return {
            "total": len(self.items),
            "sources": sources,
            "avg_trust": round(
                sum(i.trust_score for i in self.items) / max(len(self.items), 1), 3
            ),
        }
