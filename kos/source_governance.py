"""
KOS V8.0 -- Source Governance (Trust-Tiered Ingestion)

Not all sources are equal. A peer-reviewed paper is not Wikipedia
is not a random blog post. Source governance assigns trust classes
and provenance scoring to every piece of ingested knowledge.

Source Classes:
    AUTHORITATIVE (1.0) - Peer-reviewed journals, official databases, standards
    TRUSTED       (0.8) - Wikipedia, major news, textbooks, government sites
    SECONDARY     (0.5) - Blogs, forums, social media, uncited claims
    EXPLORATORY   (0.3) - Auto-generated, daemon inferences, unverified

Edges from lower-trust sources get lower weights and are candidates
for quarantine if they conflict with higher-trust sources.
"""

import re
from urllib.parse import urlparse


# ---- Source Trust Classes ------------------------------------------------

AUTHORITATIVE = 1.0
TRUSTED = 0.8
SECONDARY = 0.5
EXPLORATORY = 0.3

# ---- URL Pattern -> Trust Mapping ----------------------------------------

_AUTHORITATIVE_DOMAINS = {
    "pubmed.ncbi.nlm.nih.gov", "arxiv.org", "doi.org",
    "nature.com", "science.org", "ieee.org", "acm.org",
    "who.int", "cdc.gov", "nih.gov", "fda.gov",
    "sec.gov", "bis.org", "imf.org", "worldbank.org",
}

_TRUSTED_DOMAINS = {
    "en.wikipedia.org", "britannica.com",
    "reuters.com", "apnews.com", "bbc.com", "nytimes.com",
    "edu",  # Any .edu domain
    "gov",  # Any .gov domain
}

_SECONDARY_PATTERNS = [
    r'blog', r'forum', r'reddit\.com', r'quora\.com',
    r'medium\.com', r'stackoverflow\.com', r'twitter\.com',
    r'facebook\.com',
]


class SourceGovernor:
    """Assign trust classes and manage source reputation."""

    def __init__(self):
        self.source_history = {}  # url/source -> {trust, access_count, last_seen}
        self.quarantine = []      # Edges pending verification

    def classify_source(self, source_url: str) -> float:
        """Assign a trust score to a source URL or identifier."""
        if not source_url:
            return EXPLORATORY

        source_lower = source_url.lower()

        # Check for daemon-generated content
        if "[daemon" in source_lower or "automatically" in source_lower:
            return EXPLORATORY

        # Parse URL
        try:
            parsed = urlparse(source_url)
            domain = parsed.netloc.lower() or source_lower
        except Exception:
            domain = source_lower

        # Check authoritative
        for auth_domain in _AUTHORITATIVE_DOMAINS:
            if auth_domain in domain:
                return AUTHORITATIVE

        # Check trusted
        for trust_domain in _TRUSTED_DOMAINS:
            if trust_domain in domain:
                return TRUSTED

        # Check secondary patterns
        for pattern in _SECONDARY_PATTERNS:
            if re.search(pattern, domain):
                return SECONDARY

        # Local files get TRUSTED (user provided)
        if source_lower.startswith("/") or source_lower.startswith("c:"):
            return TRUSTED

        # Default
        return SECONDARY

    def classify_provenance(self, provenance_text: str) -> float:
        """Classify trust from provenance text (not URL)."""
        if not provenance_text:
            return EXPLORATORY

        text_lower = provenance_text.lower()

        # Daemon inference markers
        if "[daemon" in text_lower or "automatically predicted" in text_lower:
            return EXPLORATORY

        # Citation markers suggest higher trust
        if re.search(r'\(\d{4}\)|\[\d+\]|doi:|isbn:', text_lower):
            return AUTHORITATIVE

        # Statistical evidence
        if re.search(r'p\s*[<>=]\s*0\.\d|n\s*=\s*\d', text_lower):
            return AUTHORITATIVE

        # Standard factual text
        if len(provenance_text) > 50:
            return TRUSTED

        return SECONDARY

    def weight_adjustment(self, base_weight: float,
                          source_trust: float) -> float:
        """Adjust edge weight based on source trust."""
        return base_weight * source_trust

    def should_quarantine(self, edge_weight: float,
                          source_trust: float,
                          contradicts_existing: bool) -> bool:
        """Should this edge be quarantined for verification?"""
        if contradicts_existing and source_trust < TRUSTED:
            return True
        if source_trust <= EXPLORATORY and edge_weight > 0.7:
            return True  # High-confidence claim from low-trust source
        return False

    def quarantine_edge(self, source_id: str, target_id: str,
                        weight: float, provenance: str,
                        reason: str):
        """Add edge to quarantine for later verification."""
        self.quarantine.append({
            "source": source_id,
            "target": target_id,
            "weight": weight,
            "provenance": provenance,
            "reason": reason,
            "status": "pending",
        })

    def promote_from_quarantine(self, index: int) -> dict:
        """Promote a quarantined edge (verified by corroboration or human)."""
        if 0 <= index < len(self.quarantine):
            edge = self.quarantine[index]
            edge["status"] = "promoted"
            return edge
        return None

    def reject_from_quarantine(self, index: int) -> dict:
        """Reject a quarantined edge."""
        if 0 <= index < len(self.quarantine):
            edge = self.quarantine[index]
            edge["status"] = "rejected"
            return edge
        return None

    def pending_count(self) -> int:
        """Count edges pending verification."""
        return sum(1 for e in self.quarantine if e["status"] == "pending")

    def stats(self) -> dict:
        return {
            "total_quarantined": len(self.quarantine),
            "pending": self.pending_count(),
            "promoted": sum(1 for e in self.quarantine
                           if e["status"] == "promoted"),
            "rejected": sum(1 for e in self.quarantine
                           if e["status"] == "rejected"),
        }
