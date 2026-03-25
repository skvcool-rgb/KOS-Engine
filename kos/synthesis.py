"""
KOS V8.0 -- Synthesis Engine (Deterministic Output Layer)

NOT an LLM. A rhetorical planner that assembles structured output
from raw evidence using domain-aware templates.

Pipeline:
    evidence -> classify intent -> select template -> fill slots -> validate

Supports domain templates: general, finance, medical, industrial.
"""

import re


# ---- Template Library ----------------------------------------------------

_TEMPLATES = {
    "general": {
        "definition": "{entity} is {definition}.",
        "location": "{entity} is located in {location}.",
        "property": "{entity} has {property}: {value}.",
        "cause": "{cause} causes {effect}.",
        "temporal": "{event_a} occurred {relation} {event_b}.",
        "comparison": "{entity_a} and {entity_b} differ in {property}: "
                      "{value_a} vs {value_b}.",
        "list": "{entity} is associated with: {items}.",
        "fallback": "Based on available evidence: {evidence}",
    },
    "finance": {
        "definition": "{entity} is a financial instrument/entity: {definition}.",
        "metric": "{entity} reports {metric} of {value} ({period}).",
        "risk": "Risk assessment for {entity}: {risk_level}. {evidence}",
        "regulation": "{entity} is governed by {regulation}: {detail}.",
        "comparison": "{entity_a} vs {entity_b}: {metric} is {value_a} vs {value_b}.",
        "fallback": "Financial data indicates: {evidence}",
    },
    "medical": {
        "definition": "{entity}: {definition}.",
        "symptom": "{condition} presents with: {symptoms}.",
        "treatment": "Treatment for {condition}: {treatment}. {evidence}",
        "cause": "{cause} is a known factor in {condition}.",
        "contraindication": "Warning: {drug_a} and {drug_b} may interact. {evidence}",
        "fallback": "Clinical evidence suggests: {evidence}",
    },
    "industrial": {
        "definition": "{entity}: {definition}.",
        "process": "{process} involves: {steps}.",
        "failure": "Failure mode for {component}: {failure_mode}. {evidence}",
        "specification": "{entity} specification: {spec_name} = {spec_value}.",
        "safety": "Safety notice for {entity}: {hazard}. {mitigation}",
        "fallback": "Technical data indicates: {evidence}",
    },
}


class SynthesisEngine:
    """Deterministic output synthesis from structured evidence."""

    def __init__(self, domain: str = "general"):
        self.domain = domain
        self.templates = _TEMPLATES.get(domain, _TEMPLATES["general"])

    def set_domain(self, domain: str):
        self.domain = domain
        self.templates = _TEMPLATES.get(domain, _TEMPLATES["general"])

    def synthesize(self, evidence: list, intent: str = "general",
                   entities: list = None, raw_prompt: str = "") -> dict:
        """
        Synthesize a structured response from evidence.

        Args:
            evidence: list of evidence strings
            intent: query intent (from query_normalizer)
            entities: key entities mentioned
            raw_prompt: the original user query

        Returns:
            {
                "response": str,           # Human-readable output
                "raw_evidence": [str],     # Exact provenance citations
                "confidence": float,       # 0.0-1.0
                "template_used": str,      # Which template was applied
                "domain": str,
            }
        """
        if not evidence:
            return {
                "response": "No relevant information found in the knowledge base.",
                "raw_evidence": [],
                "confidence": 0.0,
                "template_used": "none",
                "domain": self.domain,
            }

        entities = entities or []
        entity = entities[0] if entities else self._extract_entity(raw_prompt)

        # Select template based on intent
        template_key = self._intent_to_template(intent, evidence)
        template = self.templates.get(template_key,
                                       self.templates.get("fallback", "{evidence}"))

        # Fill template slots
        slots = self._extract_slots(evidence, entity, intent)
        try:
            response = template.format(**slots)
        except KeyError:
            # Fallback if template slots don't match
            response = self.templates["fallback"].format(
                evidence=" ".join(evidence[:2]))

        # Compute confidence
        confidence = self._compute_confidence(evidence, entity, raw_prompt)

        return {
            "response": response,
            "raw_evidence": list(evidence),
            "confidence": confidence,
            "template_used": template_key,
            "domain": self.domain,
        }

    def build_contract(self, evidence: list, confidence: float) -> dict:
        """
        Build a strict JSON contract for LLM formatting.
        The LLM receives ONLY this -- no conversational context.
        """
        return {
            "facts": list(evidence),
            "confidence": round(confidence, 3),
            "prohibited_inference": True,
            "domain": self.domain,
            "instruction": ("Rephrase these facts into fluent English. "
                           "Do NOT add any information not in the facts array. "
                           "Do NOT infer, speculate, or add context. "
                           "Every number, date, and proper noun MUST appear "
                           "verbatim in the facts."),
        }

    def _intent_to_template(self, intent: str, evidence: list) -> str:
        """Map intent to best template key."""
        evidence_text = " ".join(evidence).lower()

        if intent == "where":
            return "location"
        elif intent == "causal" or intent == "how":
            return "cause"
        elif intent == "temporal":
            return "temporal"
        elif intent == "compare":
            return "comparison"

        # Auto-detect from evidence content
        if " is a " in evidence_text or " is an " in evidence_text:
            return "definition"
        if "located" in evidence_text or " in " in evidence_text[:50]:
            return "location"

        return "fallback"

    def _extract_entity(self, prompt: str) -> str:
        """Extract the main entity from the prompt."""
        words = re.findall(r'\b[A-Z][a-z]+\b', prompt)
        if words:
            return words[0]
        content = [w for w in prompt.split()
                   if len(w) > 3 and w.lower() not in
                   {"what", "where", "when", "tell", "about", "does", "have"}]
        return content[0] if content else "the subject"

    def _extract_slots(self, evidence: list, entity: str,
                       intent: str) -> dict:
        """Extract template slot values from evidence."""
        combined = " ".join(evidence[:3])
        return {
            "entity": entity,
            "definition": evidence[0] if evidence else "",
            "location": self._find_location(combined),
            "property": "characteristics",
            "value": evidence[0] if evidence else "",
            "cause": entity,
            "effect": evidence[0].split("causes")[-1].strip().rstrip(".")
                      if "causes" in evidence[0] else evidence[0],
            "event_a": entity,
            "event_b": evidence[0] if evidence else "",
            "relation": "before",
            "evidence": combined[:200],
            "entity_a": entity,
            "entity_b": "",
            "value_a": "",
            "value_b": "",
            "items": ", ".join(e[:60] for e in evidence[:5]),
            # Domain-specific
            "metric": "",
            "period": "",
            "risk_level": "",
            "regulation": "",
            "detail": "",
            "condition": entity,
            "symptoms": combined[:100],
            "treatment": combined[:100],
            "drug_a": "",
            "drug_b": "",
            "process": entity,
            "steps": combined[:150],
            "component": entity,
            "failure_mode": combined[:100],
            "spec_name": "",
            "spec_value": "",
            "hazard": combined[:100],
            "mitigation": "",
        }

    def _find_location(self, text: str) -> str:
        """Extract location from evidence text."""
        patterns = [
            r'(?:located in|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:province of|state of|country of)\s+([A-Z][a-z]+)',
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(1)
        return text[:50]

    def _compute_confidence(self, evidence: list, entity: str,
                            prompt: str) -> float:
        """Confidence based on evidence quality."""
        if not evidence:
            return 0.0

        score = 0.0
        # More evidence = higher confidence
        score += min(len(evidence) / 5.0, 0.4)

        # Entity mentioned in evidence
        entity_lower = entity.lower()
        mentions = sum(1 for e in evidence if entity_lower in e.lower())
        score += min(mentions / 3.0, 0.3)

        # Evidence length (longer = more detailed)
        avg_len = sum(len(e) for e in evidence) / max(len(evidence), 1)
        score += min(avg_len / 200.0, 0.3)

        return min(score, 1.0)
