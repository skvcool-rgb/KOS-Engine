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

    def synthesize_comparison(self, entity_a: str, entity_b: str,
                               evidence_a: list, evidence_b: list,
                               raw_prompt: str = "") -> dict:
        """
        Synthesize a structured comparison answer from per-entity evidence.

        Args:
            entity_a: first entity name
            entity_b: second entity name
            evidence_a: evidence strings about entity_a
            evidence_b: evidence strings about entity_b
            raw_prompt: original user query

        Returns:
            Same schema as synthesize(), with a side-by-side comparison response.
        """
        # Fall back to normal synthesize if we lack evidence for either entity
        if not evidence_a and not evidence_b:
            return self.synthesize([], intent="compare",
                                   entities=[entity_a, entity_b],
                                   raw_prompt=raw_prompt)
        if not evidence_a or not evidence_b:
            combined = evidence_a or evidence_b
            return self.synthesize(combined, intent="compare",
                                   entities=[entity_a, entity_b],
                                   raw_prompt=raw_prompt)

        # Classify entity types from evidence
        type_a = self._classify_entity_type(entity_a, evidence_a)
        type_b = self._classify_entity_type(entity_b, evidence_b)

        # Extract comparable attributes using domain-specific extractors
        attrs_a = self._extract_attributes(entity_a, evidence_a,
                                           entity_type=type_a)
        attrs_b = self._extract_attributes(entity_b, evidence_b,
                                           entity_type=type_b)

        # Build side-by-side structure for shared attribute keys
        all_keys = list(dict.fromkeys(list(attrs_a.keys()) + list(attrs_b.keys())))
        comparison_rows = []
        for key in all_keys:
            val_a = attrs_a.get(key, "N/A")
            val_b = attrs_b.get(key, "N/A")
            if val_a != "N/A" or val_b != "N/A":
                comparison_rows.append((key, val_a, val_b))

        # Build the response text
        parts = []

        # If entity types differ, note the cross-category comparison
        if type_a != type_b and type_a != "unknown" and type_b != "unknown":
            parts.append(
                f"Note: {entity_a} ({type_a}) and {entity_b} ({type_b}) "
                f"belong to different categories."
            )

        # Header summaries
        summary_a = self._one_line_summary(entity_a, attrs_a, evidence_a)
        summary_b = self._one_line_summary(entity_b, attrs_b, evidence_b)
        parts.append(f"{summary_a} vs {summary_b}.")

        if comparison_rows:
            diffs = []
            for key, val_a, val_b in comparison_rows:
                if val_a != "N/A" and val_b != "N/A" and val_a != val_b:
                    diffs.append(f"{key}: {val_a} ({entity_a}) vs {val_b} ({entity_b})")
            if diffs:
                parts.append("Key differences: " + "; ".join(diffs[:6]) + ".")

        response = " ".join(parts)

        all_evidence = list(evidence_a) + list(evidence_b)
        confidence = self._compute_comparison_confidence(
            entity_a, entity_b, evidence_a, evidence_b, comparison_rows)

        return {
            "response": response,
            "raw_evidence": all_evidence,
            "confidence": confidence,
            "template_used": "comparison_structured",
            "domain": self.domain,
            "comparison": {
                "entity_a": entity_a,
                "entity_b": entity_b,
                "type_a": type_a,
                "type_b": type_b,
                "attributes": {key: {"a": va, "b": vb}
                               for key, va, vb in comparison_rows},
            },
        }

    def _classify_entity_type(self, entity: str, evidence: list) -> str:
        """Classify an entity into a domain type based on keyword detection
        in evidence text.

        Returns one of: "city", "drug", "technology", "concept", "person",
        "organization", "language", "material", "unknown".
        """
        combined = " ".join(evidence).lower()

        # Keyword sets ordered by specificity (more specific first)
        _type_keywords = {
            "drug": [
                "anticoagulant", "medication", "dosage", "treatment",
                "side effect", "fda", "pharmaceutical", "drug", "clinical trial",
                "contraindication", "indication", "mechanism of action",
            ],
            "language": [
                "programming language", "syntax", "compiler", "interpreter",
                "statically typed", "dynamically typed", "garbage collection",
            ],
            "material": [
                "semiconductor", "photovoltaic", "alloy", "compound",
                "crystal", "tensile strength", "thermal conductivity",
                "material", "substrate",
            ],
            "technology": [
                "software", "programming", "algorithm", "computation",
                "framework", "processor", "open source", "api", "runtime",
            ],
            "city": [
                "population", "founded", "province", "capital",
                "metropolitan", "municipality", "city", "township",
            ],
            "person": [
                "born", "died", "nobel", "president", "scientist", "author",
                "biography", "birthplace",
            ],
            "organization": [
                "corporation", "company", "nonprofit", "founded in",
                "headquartered", "employees", "revenue", "ceo",
            ],
            "concept": [
                "theory", "principle", "method", "technique", "process",
                "approach", "paradigm", "hypothesis", "framework",
            ],
        }

        scores = {}
        for entity_type, keywords in _type_keywords.items():
            score = sum(1 for kw in keywords if kw in combined)
            if score > 0:
                scores[entity_type] = score

        if not scores:
            return "unknown"

        return max(scores, key=scores.get)

    def _extract_attributes(self, entity: str, evidence: list,
                            entity_type: str = "unknown") -> dict:
        """Extract key-value attributes from evidence for an entity.

        Uses domain-specific extractors when entity_type is known,
        falling back to city/general extractors otherwise.
        """
        attrs = {}
        combined = " ".join(evidence)

        # --- Domain-specific extractors ---
        if entity_type == "drug":
            return self._extract_drug_attributes(entity, combined, evidence)
        elif entity_type in ("technology", "language"):
            return self._extract_tech_attributes(entity, combined, evidence)
        elif entity_type == "material":
            return self._extract_material_attributes(entity, combined, evidence)
        elif entity_type == "concept":
            return self._extract_concept_attributes(entity, combined, evidence)

        # --- Default city / general extractors (original logic) ---

        # Population
        pop_m = re.search(
            r'population[:\s]+(?:of\s+)?(?:approximately\s+|about\s+|~)?'
            r'([\d,.]+\s*(?:million|billion|thousand|[MBKmk])?)',
            combined, re.I)
        if pop_m:
            attrs["population"] = pop_m.group(1).strip()

        # Founded / established year
        found_m = re.search(
            r'(?:founded|established|incorporated|settled)[:\s]+(?:in\s+)?(\d{3,4})',
            combined, re.I)
        if found_m:
            attrs["founded"] = found_m.group(1)

        # Location / country / region
        loc_m = re.search(
            r'(?:located in|situated in|in the province of|'
            r'(?:is|are) in|capital of)\s+([A-Z][\w\s,]+?)(?:\.|,|$)',
            combined)
        if loc_m:
            attrs["location"] = loc_m.group(1).strip()[:60]

        # Area
        area_m = re.search(
            r'area[:\s]+(?:of\s+)?([\d,.]+\s*(?:km2|sq\s*km|square\s*(?:kilo)?meters?|'
            r'sq\s*mi|square\s*miles?))',
            combined, re.I)
        if area_m:
            attrs["area"] = area_m.group(1).strip()

        # Known for / description snippet
        known_m = re.search(
            r'(?:known (?:for|as)|famous for|renowned for)\s+(.+?)(?:\.|$)',
            combined, re.I)
        if known_m:
            attrs["known for"] = known_m.group(1).strip()[:80]

        # Elevation / altitude
        elev_m = re.search(
            r'(?:elevation|altitude)[:\s]+([\d,.]+\s*(?:m|ft|meters?|feet))',
            combined, re.I)
        if elev_m:
            attrs["elevation"] = elev_m.group(1).strip()

        # Currency
        curr_m = re.search(
            r'(?:currency|monetary unit)[:\s]+([A-Z][\w\s]+?)(?:\.|,|$)',
            combined, re.I)
        if curr_m:
            attrs["currency"] = curr_m.group(1).strip()[:40]

        # Language
        lang_m = re.search(
            r'(?:official language|language)[:\s]+([A-Z][\w\s,]+?)(?:\.|$)',
            combined, re.I)
        if lang_m:
            attrs["language"] = lang_m.group(1).strip()[:60]

        # If no structured attributes found, use first evidence as description
        if not attrs and evidence:
            attrs["description"] = evidence[0][:100]

        return attrs

    # -- Domain-specific attribute extractors --------------------------------

    def _extract_drug_attributes(self, entity: str, combined: str,
                                  evidence: list) -> dict:
        """Extract drug/medical attributes from evidence."""
        attrs = {}

        mech_m = re.search(
            r'(?:mechanism of action|works by|acts by|inhibits|blocks)\s+'
            r'(.+?)(?:\.|$)', combined, re.I)
        if mech_m:
            attrs["mechanism"] = mech_m.group(1).strip()[:100]

        ind_m = re.search(
            r'(?:indicated for|used (?:for|to treat)|treatment of|treats)\s+'
            r'(.+?)(?:\.|$)', combined, re.I)
        if ind_m:
            attrs["indication"] = ind_m.group(1).strip()[:100]

        se_m = re.search(
            r'(?:side effects?|adverse (?:effects?|reactions?))[:\s]+(.+?)(?:\.|$)',
            combined, re.I)
        if se_m:
            attrs["side_effects"] = se_m.group(1).strip()[:120]

        dos_m = re.search(
            r'(?:dosage|dose|administered)[:\s]+(.+?)(?:\.|$)',
            combined, re.I)
        if dos_m:
            attrs["dosage"] = dos_m.group(1).strip()[:80]

        app_m = re.search(
            r'(?:approved|FDA approval|approval)[:\s]+(?:in\s+)?(\d{4})',
            combined, re.I)
        if app_m:
            attrs["approval_year"] = app_m.group(1)

        contra_m = re.search(
            r'(?:contraindicated|contraindication|do not use)[:\s]+(.+?)(?:\.|$)',
            combined, re.I)
        if contra_m:
            attrs["contraindications"] = contra_m.group(1).strip()[:100]

        if not attrs and evidence:
            attrs["description"] = evidence[0][:100]
        return attrs

    def _extract_tech_attributes(self, entity: str, combined: str,
                                  evidence: list) -> dict:
        """Extract technology / programming language attributes."""
        attrs = {}

        type_m = re.search(
            r'(?:is a|is an)\s+(.+?)(?:\.|,|$)', combined, re.I)
        if type_m:
            attrs["type"] = type_m.group(1).strip()[:80]

        para_m = re.search(
            r'(?:paradigm|paradigms)[:\s]+(.+?)(?:\.|$)', combined, re.I)
        if para_m:
            attrs["paradigm"] = para_m.group(1).strip()[:80]

        use_m = re.search(
            r'(?:used for|use cases?|applications?)[:\s]+(.+?)(?:\.|$)',
            combined, re.I)
        if use_m:
            attrs["use_cases"] = use_m.group(1).strip()[:100]

        perf_m = re.search(
            r'(?:performance|speed|benchmark)[:\s]+(.+?)(?:\.|$)',
            combined, re.I)
        if perf_m:
            attrs["performance"] = perf_m.group(1).strip()[:80]

        creator_m = re.search(
            r'(?:created by|developed by|author|designed by)\s+(.+?)(?:\.|,|$)',
            combined, re.I)
        if creator_m:
            attrs["creator"] = creator_m.group(1).strip()[:60]

        yr_m = re.search(
            r'(?:released|first appeared|launched|introduced)[:\s]+(?:in\s+)?(\d{4})',
            combined, re.I)
        if yr_m:
            attrs["year_released"] = yr_m.group(1)

        if not attrs and evidence:
            attrs["description"] = evidence[0][:100]
        return attrs

    def _extract_material_attributes(self, entity: str, combined: str,
                                      evidence: list) -> dict:
        """Extract material / semiconductor attributes."""
        attrs = {}

        eff_m = re.search(
            r'(?:efficiency)[:\s]+([\d.]+\s*%?)', combined, re.I)
        if eff_m:
            attrs["efficiency"] = eff_m.group(1).strip()

        cost_m = re.search(
            r'(?:cost|price)[:\s]+(.+?)(?:\.|$)', combined, re.I)
        if cost_m:
            attrs["cost"] = cost_m.group(1).strip()[:60]

        dur_m = re.search(
            r'(?:durability|lifespan|lifetime|degradation)[:\s]+(.+?)(?:\.|$)',
            combined, re.I)
        if dur_m:
            attrs["durability"] = dur_m.group(1).strip()[:80]

        app_m = re.search(
            r'(?:applications?|used (?:in|for))[:\s]+(.+?)(?:\.|$)',
            combined, re.I)
        if app_m:
            attrs["applications"] = app_m.group(1).strip()[:100]

        comp_m = re.search(
            r'(?:composed of|composition|made (?:of|from)|formula)[:\s]+(.+?)(?:\.|$)',
            combined, re.I)
        if comp_m:
            attrs["composition"] = comp_m.group(1).strip()[:80]

        if not attrs and evidence:
            attrs["description"] = evidence[0][:100]
        return attrs

    def _extract_concept_attributes(self, entity: str, combined: str,
                                     evidence: list) -> dict:
        """Extract concept / theory attributes."""
        attrs = {}

        def_m = re.search(
            r'(?:defined as|is a|refers to|describes)\s+(.+?)(?:\.|$)',
            combined, re.I)
        if def_m:
            attrs["definition"] = def_m.group(1).strip()[:120]

        origin_m = re.search(
            r'(?:proposed by|invented by|originated|developed by|introduced by)'
            r'\s+(.+?)(?:\.|,|$)', combined, re.I)
        if origin_m:
            attrs["origin"] = origin_m.group(1).strip()[:80]

        prop_m = re.search(
            r'(?:key propert(?:y|ies)|characteristics?|features?)[:\s]+(.+?)(?:\.|$)',
            combined, re.I)
        if prop_m:
            attrs["key_properties"] = prop_m.group(1).strip()[:100]

        app_m = re.search(
            r'(?:applications?|used (?:in|for)|applied to)[:\s]+(.+?)(?:\.|$)',
            combined, re.I)
        if app_m:
            attrs["applications"] = app_m.group(1).strip()[:100]

        if not attrs and evidence:
            attrs["description"] = evidence[0][:100]
        return attrs

    def _one_line_summary(self, entity: str, attrs: dict,
                          evidence: list) -> str:
        """Build a parenthetical summary like 'Toronto (pop 2.7M, founded 1834)'."""
        parts = []
        if "population" in attrs:
            parts.append(f"pop {attrs['population']}")
        if "founded" in attrs:
            parts.append(f"founded {attrs['founded']}")
        if "location" in attrs:
            parts.append(attrs["location"])
        if not parts and "description" in attrs:
            parts.append(attrs["description"][:50])
        if parts:
            return f"{entity} ({', '.join(parts[:3])})"
        return entity

    def _compute_comparison_confidence(self, entity_a, entity_b,
                                        evidence_a, evidence_b,
                                        comparison_rows) -> float:
        """Confidence for comparison answers."""
        score = 0.0
        # Both entities have evidence
        score += 0.25
        # More evidence per entity
        score += min(len(evidence_a) / 4.0, 0.2)
        score += min(len(evidence_b) / 4.0, 0.2)
        # Comparable attributes found
        shared = sum(1 for _, va, vb in comparison_rows
                     if va != "N/A" and vb != "N/A")
        score += min(shared / 3.0, 0.25)
        # Entity mentions in evidence
        a_mentions = sum(1 for e in evidence_a if entity_a.lower() in e.lower())
        b_mentions = sum(1 for e in evidence_b if entity_b.lower() in e.lower())
        score += min((a_mentions + b_mentions) / 4.0, 0.1)
        return min(score, 1.0)

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
