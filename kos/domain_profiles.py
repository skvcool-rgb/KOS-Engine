"""
KOS V6.1 — Domain-Specific Weaver Profiles.

Different query domains require different evidence weighting:
    - Medical queries need dosage/symptom/contraindication boosts
    - Legal queries need statute/precedent/jurisdiction boosts
    - Scientific queries need methodology/results/hypothesis boosts
    - Geographic queries use the existing WHERE boost
    - Historical queries use the existing WHEN boost
    - Technical queries need implementation/architecture boosts

The DomainProfiler detects the query domain from keywords and
temporarily adjusts the AlgorithmicWeaver's scoring weights for
that query, then restores defaults afterward.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Domain keyword detection maps
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: Dict[str, Set[str]] = {
    "medical": {
        "disease", "drug", "treatment", "symptom", "patient",
        "clinical", "therapy", "diagnosis", "dosage", "dose",
        "contraindication", "side effect", "pharmaceutical",
        "prescription", "pathology", "surgery", "infection",
        "antibody", "vaccine", "prognosis", "syndrome", "tumor",
        "receptor", "agonist", "antagonist", "metabolite",
        "pharmacokinetics", "bioavailability", "chronic", "acute",
        "health", "medicine", "doctor", "hospital", "cure",
        "blood", "organ", "tissue", "cell",
    },
    "legal": {
        "law", "legal", "court", "statute", "contract",
        "jurisdiction", "attorney", "plaintiff", "defendant",
        "precedent", "ruling", "verdict", "appeal", "tort",
        "liability", "negligence", "regulation", "compliance",
        "amendment", "constitution", "legislation", "judicial",
        "arbitration", "subpoena", "deposition", "litigation",
        "prosecution", "defense", "judge", "jury", "sentence",
        "felony", "misdemeanor", "penalty",
    },
    "scientific": {
        "experiment", "hypothesis", "molecule", "quantum",
        "physics", "chemistry", "biology", "research", "theory",
        "equation", "methodology", "results", "data", "analysis",
        "observation", "variable", "control", "sample", "peer",
        "review", "publish", "journal", "citation", "abstract",
        "study", "finding", "evidence", "correlation", "causation",
        "model", "simulation", "laboratory", "specimen",
        "genome", "protein", "enzyme", "catalyst", "reactor",
    },
    "geographic": {
        "city", "country", "river", "mountain", "continent",
        "ocean", "region", "capital", "population", "border",
        "island", "peninsula", "desert", "forest", "lake",
        "where", "located", "location", "geography", "area",
        "province", "state", "territory", "latitude", "longitude",
        "climate", "elevation", "north", "south", "east", "west",
    },
    "historical": {
        "history", "war", "century", "dynasty", "empire",
        "revolution", "ancient", "medieval", "colonial",
        "independence", "founded", "established", "era", "period",
        "civilization", "archaeological", "artifact", "reign",
        "monarch", "kingdom", "republic", "treaty", "battle",
        "conquest", "migration", "when", "year", "date",
    },
    "technical": {
        "algorithm", "code", "software", "api", "database",
        "server", "programming", "debug", "function", "class",
        "method", "protocol", "architecture", "implementation",
        "framework", "library", "runtime", "compiler", "interpreter",
        "memory", "cpu", "gpu", "thread", "process", "network",
        "security", "encryption", "authentication", "deploy",
        "container", "microservice", "cache", "index", "query",
    },
}


# ---------------------------------------------------------------------------
# Domain scoring profiles — evidence keyword boosts
# ---------------------------------------------------------------------------

DOMAIN_PROFILES: Dict[str, Dict[str, int]] = {
    "medical": {
        # Evidence keywords and their score boosts
        "dosage": 30,
        "dose": 30,
        "mg": 25,
        "symptom": 25,
        "symptoms": 25,
        "contraindication": 35,
        "contraindicated": 35,
        "side effect": 30,
        "adverse": 30,
        "interaction": 25,
        "treatment": 25,
        "therapy": 25,
        "diagnosis": 25,
        "prognosis": 20,
        "clinical": 20,
        "patient": 15,
        "receptor": 20,
        "mechanism of action": 30,
        "pharmacokinetics": 25,
        "bioavailability": 25,
        "half-life": 25,
        "efficacy": 25,
    },
    "legal": {
        "statute": 30,
        "section": 20,
        "subsection": 25,
        "precedent": 25,
        "jurisdiction": 20,
        "ruling": 25,
        "held that": 30,
        "court": 20,
        "amendment": 25,
        "liability": 25,
        "negligence": 25,
        "burden of proof": 30,
        "reasonable": 15,
        "pursuant to": 25,
        "under the law": 20,
        "constitutional": 25,
        "regulation": 20,
        "compliance": 20,
    },
    "scientific": {
        "methodology": 25,
        "method": 20,
        "results": 30,
        "result": 25,
        "finding": 25,
        "findings": 25,
        "hypothesis": 20,
        "observed": 20,
        "measured": 25,
        "experiment": 20,
        "data": 15,
        "p-value": 30,
        "significant": 20,
        "correlation": 20,
        "coefficient": 20,
        "sample size": 25,
        "control group": 25,
        "peer-reviewed": 30,
        "published": 15,
        "equation": 20,
    },
    "geographic": {
        # Reuses Weaver's WHERE logic — these are supplementary
        "located": 20,
        "latitude": 25,
        "longitude": 25,
        "elevation": 20,
        "population": 20,
        "area": 15,
        "km": 15,
        "miles": 15,
        "border": 20,
        "capital": 25,
        "province": 20,
        "region": 15,
    },
    "historical": {
        # Reuses Weaver's WHEN logic — these are supplementary
        "founded": 25,
        "established": 25,
        "century": 20,
        "era": 15,
        "period": 15,
        "dynasty": 20,
        "reign": 20,
        "battle": 20,
        "treaty": 25,
        "revolution": 20,
        "independence": 20,
        "archaeological": 25,
    },
    "technical": {
        "implementation": 25,
        "architecture": 25,
        "algorithm": 25,
        "complexity": 20,
        "runtime": 20,
        "performance": 20,
        "latency": 20,
        "throughput": 20,
        "syntax": 15,
        "api": 20,
        "endpoint": 20,
        "parameter": 15,
        "return": 15,
        "exception": 20,
        "protocol": 20,
        "specification": 25,
    },
}


# ---------------------------------------------------------------------------
# Domain Profiler
# ---------------------------------------------------------------------------

class DomainProfiler:
    """Detects query domain and applies domain-specific Weaver weights.

    The profiler examines query keywords to determine the most likely
    domain (medical, legal, scientific, geographic, historical, technical),
    then temporarily adjusts the AlgorithmicWeaver's scoring weights
    to prioritise domain-relevant evidence.

    Example::

        profiler = DomainProfiler()
        domain = profiler.detect_domain("What are the contraindications of aspirin?")
        # 'medical'
        profiler.apply_profile(weaver, domain)
        # Weaver now boosts dosage/symptom/contraindication evidence
        answer = weaver.weave(...)
        profiler.restore_profile(weaver)
        # Weaver weights restored to defaults
    """

    def __init__(self) -> None:
        """Initialise the profiler."""
        self._saved_weights: Dict[str, int] = {}
        self._active_domain: Optional[str] = None

    def detect_domain(self, query: str) -> Optional[str]:
        """Detect the most likely domain for a query.

        Counts keyword matches against each domain's keyword set.
        The domain with the most matches wins, provided it has
        at least 1 match. Ties are broken by domain priority:
        medical > legal > scientific > technical > geographic > historical.

        Args:
            query: The raw query string.

        Returns:
            Domain string ('medical', 'legal', 'scientific',
            'geographic', 'historical', 'technical') or None if
            no domain matches.
        """
        if not query:
            return None

        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))

        scores: Dict[str, int] = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            # Check single-word matches
            single_word_matches = len(query_words & keywords)
            # Check multi-word phrase matches
            phrase_matches = sum(
                1 for kw in keywords
                if ' ' in kw and kw in query_lower
            )
            total = single_word_matches + phrase_matches
            if total > 0:
                scores[domain] = total

        if not scores:
            return None

        # Priority order for tie-breaking
        priority = ["medical", "legal", "scientific",
                     "technical", "geographic", "historical"]

        max_score = max(scores.values())
        for domain in priority:
            if scores.get(domain) == max_score:
                return domain

        # Fallback (should not reach here)
        return max(scores, key=scores.get)

    def get_profile(self, domain: str) -> Dict[str, int]:
        """Get the scoring weight overrides for a domain.

        Args:
            domain: Domain string (e.g. 'medical').

        Returns:
            Dict mapping evidence keywords to their boost values.
            Empty dict if domain is not recognised.
        """
        return dict(DOMAIN_PROFILES.get(domain, {}))

    def apply_profile(self, weaver, domain: str) -> bool:
        """Temporarily adjust Weaver weights for a domain query.

        Saves the Weaver's current weights, then applies domain-specific
        boosts. Call restore_profile() after the query to undo.

        Specifically:
        - Increases ATTRIBUTE_BOOST for domains that rely on specific terms
        - Increases WHERE_BOOST for geographic queries
        - Increases WHEN_BOOST for historical queries
        - Increases HOW_BOOST for scientific/technical queries

        Args:
            weaver: AlgorithmicWeaver instance.
            domain: Domain to apply.

        Returns:
            True if profile was applied, False if domain unknown.
        """
        if domain not in DOMAIN_PROFILES:
            return False

        # Save current weights
        self._saved_weights = {
            'WHERE_BOOST': getattr(weaver, 'WHERE_BOOST', 40),
            'WHEN_BOOST': getattr(weaver, 'WHEN_BOOST', 40),
            'WHO_BOOST': getattr(weaver, 'WHO_BOOST', 40),
            'ATTRIBUTE_BOOST': getattr(weaver, 'ATTRIBUTE_BOOST', 35),
            'HOW_BOOST': getattr(weaver, 'HOW_BOOST', 30),
            'KEYWORD_MULTIPLIER': getattr(weaver, 'KEYWORD_MULTIPLIER', 20),
        }
        self._active_domain = domain

        # Apply domain-specific adjustments
        if domain == "geographic":
            weaver.WHERE_BOOST = 60    # Boost from 40 to 60
            weaver.ATTRIBUTE_BOOST = 45

        elif domain == "historical":
            weaver.WHEN_BOOST = 60     # Boost from 40 to 60
            weaver.ATTRIBUTE_BOOST = 45

        elif domain == "medical":
            weaver.ATTRIBUTE_BOOST = 55  # Strong attribute matching
            weaver.HOW_BOOST = 45        # Mechanism queries common
            weaver.KEYWORD_MULTIPLIER = 30  # Boost keyword density

        elif domain == "legal":
            weaver.ATTRIBUTE_BOOST = 55  # Statute/precedent matching
            weaver.WHO_BOOST = 50        # Parties matter
            weaver.KEYWORD_MULTIPLIER = 30

        elif domain == "scientific":
            weaver.HOW_BOOST = 50        # Methodology focus
            weaver.ATTRIBUTE_BOOST = 50  # Results/findings
            weaver.KEYWORD_MULTIPLIER = 25

        elif domain == "technical":
            weaver.HOW_BOOST = 50        # Implementation focus
            weaver.ATTRIBUTE_BOOST = 50
            weaver.KEYWORD_MULTIPLIER = 25

        return True

    def apply_to_weaver(self, weaver, query: str) -> str:
        """Detect domain from query and temporarily adjust weaver boosts.

        Convenience method that combines detect_domain and apply_profile
        in a single call. Returns the detected domain name so callers
        can log or display it.

        Args:
            weaver: AlgorithmicWeaver instance.
            query: The raw query string.

        Returns:
            The detected domain name string, or 'general' if no
            domain-specific profile was applied.
        """
        domain = self.detect_domain(query)
        if domain and self.apply_profile(weaver, domain):
            return domain
        return "general"

    def restore_profile(self, weaver) -> None:
        """Restore Weaver weights to their pre-profile values.

        Call this after the domain-adjusted query is complete.

        Args:
            weaver: AlgorithmicWeaver instance (same one passed to apply).
        """
        if not self._saved_weights:
            return

        for attr, value in self._saved_weights.items():
            setattr(weaver, attr, value)

        self._saved_weights.clear()
        self._active_domain = None

    def score_evidence(self, sentence: str, domain: str) -> int:
        """Score a single evidence sentence against a domain profile.

        Used for fine-grained scoring within the Weaver's pipeline.
        Checks the sentence for domain-specific evidence keywords
        and returns the cumulative boost.

        Args:
            sentence: Evidence sentence to score.
            domain: Domain to score against.

        Returns:
            Integer score boost (0 if no domain keywords found).
        """
        if domain not in DOMAIN_PROFILES:
            return 0

        profile = DOMAIN_PROFILES[domain]
        sentence_lower = sentence.lower()
        total_boost = 0

        for keyword, boost in profile.items():
            if keyword in sentence_lower:
                total_boost += boost

        return total_boost

    def get_active_domain(self) -> Optional[str]:
        """Return the currently active domain profile, or None."""
        return self._active_domain

    def list_domains(self) -> List[str]:
        """Return all available domain names."""
        return sorted(DOMAIN_PROFILES.keys())

    def __repr__(self) -> str:
        active = self._active_domain or "none"
        return f"DomainProfiler(active={active})"
