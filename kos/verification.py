"""
KOS V8.0 -- Verification Pipeline (Enterprise-Grade Ingestion)

Every piece of knowledge passes through a 5-stage pipeline before
being committed to the live graph:

    1. INGEST   - Parse raw input into candidate edges
    2. SCORE    - Assign source trust + edge type + weight
    3. CORROBORATE - Check if multiple sources agree
    4. QUARANTINE  - Hold low-trust/contradictory edges for review
    5. PROMOTE  - Commit verified edges to the live Rust arena

Non-negotiable for enterprise deployment.
"""


class VerificationPipeline:
    """5-stage verification pipeline for knowledge ingestion."""

    # Thresholds
    AUTO_PROMOTE_TRUST = 0.8     # Auto-promote if source trust >= this
    QUARANTINE_TRUST = 0.5       # Quarantine if source trust < this
    CORROBORATION_THRESHOLD = 2  # Need N sources to auto-promote low-trust

    def __init__(self, kernel, source_governor=None):
        self.kernel = kernel
        self.source_governor = source_governor
        self.staging = []       # Edges awaiting verification
        self.promoted = []      # Successfully promoted edges
        self.rejected = []      # Rejected edges
        self.stats_counter = {
            "ingested": 0,
            "auto_promoted": 0,
            "quarantined": 0,
            "corroborated": 0,
            "rejected": 0,
        }

    def ingest(self, source_id: str, target_id: str, weight: float,
               provenance: str = "", source_url: str = "",
               edge_type: int = None) -> dict:
        """
        Stage 1: Ingest a candidate edge.

        Returns status: "auto_promoted", "quarantined", or "rejected"
        """
        self.stats_counter["ingested"] += 1

        # Stage 2: Score
        source_trust = self._score_source(provenance, source_url)
        if edge_type is None:
            try:
                from .edge_types import infer_type
                edge_type = infer_type(provenance)
            except ImportError:
                edge_type = 0

        candidate = {
            "source_id": source_id,
            "target_id": target_id,
            "weight": weight,
            "provenance": provenance,
            "source_url": source_url,
            "source_trust": source_trust,
            "edge_type": edge_type,
            "status": "pending",
        }

        # Stage 3: Corroborate
        corroboration = self._check_corroboration(source_id, target_id)
        candidate["corroboration_count"] = corroboration

        # Stage 4: Decide — promote or quarantine
        contradicts = self._check_contradiction(source_id, target_id, weight)
        candidate["contradicts_existing"] = contradicts

        if contradicts and source_trust < self.AUTO_PROMOTE_TRUST:
            # Contradicts existing knowledge from a low-trust source
            candidate["status"] = "quarantined"
            candidate["reason"] = "contradicts existing + low trust"
            self.staging.append(candidate)
            self.stats_counter["quarantined"] += 1
            return {"status": "quarantined", "reason": candidate["reason"]}

        if source_trust >= self.AUTO_PROMOTE_TRUST:
            # High-trust source: auto-promote
            return self._promote(candidate, "auto_high_trust")

        if corroboration >= self.CORROBORATION_THRESHOLD:
            # Multiple sources agree: promote
            return self._promote(candidate, "corroborated")

        if source_trust < self.QUARANTINE_TRUST:
            # Low trust, no corroboration: quarantine
            candidate["status"] = "quarantined"
            candidate["reason"] = "low trust, no corroboration"
            self.staging.append(candidate)
            self.stats_counter["quarantined"] += 1
            return {"status": "quarantined", "reason": candidate["reason"]}

        # Medium trust: promote with reduced weight
        candidate["weight"] = weight * source_trust
        return self._promote(candidate, "medium_trust_adjusted")

    def _promote(self, candidate: dict, reason: str) -> dict:
        """Stage 5: Promote edge to live graph."""
        candidate["status"] = "promoted"
        candidate["promote_reason"] = reason

        # Commit to kernel
        self.kernel.add_connection(
            candidate["source_id"],
            candidate["target_id"],
            candidate["weight"],
            candidate["provenance"],
            edge_type=candidate["edge_type"],
        )

        self.promoted.append(candidate)
        self.stats_counter["auto_promoted"] += 1
        return {"status": "promoted", "reason": reason}

    def _score_source(self, provenance: str, source_url: str) -> float:
        """Score the source trust."""
        if self.source_governor:
            if source_url:
                return self.source_governor.classify_source(source_url)
            return self.source_governor.classify_provenance(provenance)
        # Default: medium trust
        return 0.6

    def _check_corroboration(self, source_id: str,
                              target_id: str) -> int:
        """Count how many existing sources agree with this edge."""
        pair = tuple(sorted([source_id, target_id]))
        provenance = getattr(self.kernel, 'provenance', {})
        existing = provenance.get(pair, set())
        return len(existing)

    def _check_contradiction(self, source_id: str, target_id: str,
                              weight: float) -> bool:
        """Check if this edge contradicts existing knowledge."""
        # Check existing contradictions
        for c in getattr(self.kernel, 'contradictions', []):
            if (source_id == c.get('source') and
                target_id in (c.get('existing_target'), c.get('new_target'))):
                return True

        # Check if opposite weight exists
        if source_id in self.kernel.nodes:
            existing_w = self.kernel.nodes[source_id].connections.get(target_id)
            if existing_w is not None:
                # Same edge exists with very different weight
                if abs(existing_w - weight) > 0.5:
                    return True

        return False

    def review_quarantine(self) -> list:
        """Return all quarantined edges for human review."""
        return [e for e in self.staging if e["status"] == "quarantined"]

    def approve_quarantined(self, index: int) -> dict:
        """Manually approve a quarantined edge."""
        pending = self.review_quarantine()
        if 0 <= index < len(pending):
            edge = pending[index]
            return self._promote(edge, "human_approved")
        return {"status": "error", "reason": "invalid index"}

    def reject_quarantined(self, index: int) -> dict:
        """Manually reject a quarantined edge."""
        pending = self.review_quarantine()
        if 0 <= index < len(pending):
            edge = pending[index]
            edge["status"] = "rejected"
            self.rejected.append(edge)
            self.stats_counter["rejected"] += 1
            return {"status": "rejected"}
        return {"status": "error", "reason": "invalid index"}

    def stats(self) -> dict:
        return dict(self.stats_counter)
