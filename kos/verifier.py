"""
KOS v0.7.1 -- Verifier Layer (Post-Synthesis Quality Gate)

Runs AFTER synthesis, BEFORE the decision gate.
Pipeline position:
    Retrieve -> Rerank -> Synthesize -> **VERIFY** -> Decision Gate -> Output

Seven verification checks:
  1. Relevance Verifier   -- noun coverage vs query
  2. Structure Verifier   -- format matches query type
  3. Contradiction Checker -- conflicting statements within the answer
  4. Completion Checker    -- did the answer actually answer the question?
  5. Hard Gates            -- binary fail conditions (missing entity, no result, fatal contradiction)
  6. Grounding Verifier    -- are answer claims supported by evidence?
  7. Risk/Preference Check -- unsupported superlatives in comparisons
"""

import re


class VerificationResult:
    """Immutable verification output."""
    __slots__ = ['trust_label', 'score_adjustment', 'issues',
                 'contradiction_flags', 'completeness_score',
                 'hard_fail', 'failure_tags', 'grounding_score']

    def __init__(self, trust_label="unverified", score_adjustment=0.0,
                 issues=None, contradiction_flags=None, completeness_score=0.0,
                 hard_fail=False, failure_tags=None, grounding_score=1.0):
        self.trust_label = trust_label
        self.score_adjustment = score_adjustment  # -0.3 to +0.1
        self.issues = issues or []
        self.contradiction_flags = contradiction_flags or []
        self.completeness_score = completeness_score  # 0.0-1.0
        self.hard_fail = hard_fail
        self.failure_tags = failure_tags or []
        self.grounding_score = grounding_score  # 0.0-1.0

    def to_dict(self):
        return {
            "trust_label": self.trust_label,
            "score_adjustment": round(self.score_adjustment, 3),
            "issues": self.issues,
            "contradiction_flags": self.contradiction_flags,
            "completeness_score": round(self.completeness_score, 3),
            "hard_fail": self.hard_fail,
            "failure_tags": self.failure_tags,
            "grounding_score": round(self.grounding_score, 3),
        }


class AnswerVerifier:
    """
    Post-synthesis answer verifier. Runs in <10ms (without grounding).
    Grounding check adds ~5-15ms if embedder is available.

    Returns VerificationResult with trust label, score adjustments,
    hard-fail flag, failure tags, and grounding score.
    """

    # Stop words for noun extraction
    _STOP = frozenset({
        'what', 'where', 'when', 'who', 'why', 'how', 'is', 'the', 'a', 'an',
        'does', 'it', 'are', 'do', 'of', 'in', 'to', 'for', 'and', 'or',
        'about', 'tell', 'me', 'please', 'can', 'from', 'this', 'that', 'be',
        'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would', 'with',
        'not', 'but', 'if', 'at', 'by', 'on', 'its', 'which', 'more', 'most',
        'compare', 'between', 'difference', 'versus', 'better', 'worse',
        'additionally', 'emotion', 'joy', 'neutral', 'curiosity',
    })

    # No-data indicators
    _NO_DATA = re.compile(
        r"(?i)(i don.t have|no data|no information|i.m not sure|"
        r"i cannot|i don.t know|not available|no relevant)"
    )

    # Contradiction indicators (pairs of opposing terms)
    _CONTRADICTIONS = [
        (r'\bincreas(?:e[sd]?|ing)\b', r'\bdecreas(?:e[sd]?|ing)\b'),
        (r'\bhigher\b', r'\blower\b'),
        (r'\bmore\b', r'\bless\b'),
        (r'\blarger\b', r'\bsmaller\b'),
        (r'\bfaster\b', r'\bslower\b'),
        (r'\btrue\b', r'\bfalse\b'),
        (r'\byes\b', r'\bno\b'),
        (r'\bgrow(?:s|ing|th)?\b', r'\bshrink(?:s|ing)?\b'),
        (r'\brise[sd]?\b', r'\bfall[s]?\b'),
    ]

    # Unsupported preference / superlative patterns
    _PREFERENCE_PATTERNS = re.compile(
        r'(?i)\b(best|better|superior|preferred|recommended|ideal|'
        r'optimal|worse|worst|inferior|safest|most\s+effective|'
        r'most\s+efficient|most\s+popular|should\s+(?:use|choose|pick|prefer))\b'
    )

    # Qualifying criteria that make preference claims acceptable
    _QUALIFYING_PATTERNS = re.compile(
        r'(?i)(?:because|due to|based on|according to|studies?\s+show|'
        r'evidence\s+suggests?|data\s+indicates?|research|clinical|'
        r'in terms of|with respect to|for\s+\w+\s+(?:cases?|patients?|scenarios?)|'
        r'when\s+\w+\s+(?:is|are)|if\s+\w+\s+(?:is|are|need))'
    )

    def __init__(self, embedder=None, st_util=None):
        """
        Args:
            embedder: SentenceTransformer model (optional, for grounding check)
            st_util: sentence_transformers.util module (optional)
        """
        self._embedder = embedder
        self._st_util = st_util

    def verify(self, query, answer, query_type="factual",
               entity_a=None, entity_b=None, evidence=None):
        """
        Run all 7 verification checks.

        Args:
            query: original user query
            answer: candidate answer text
            query_type: "factual", "comparison", "math", "procedural"
            entity_a: first entity (for comparisons)
            entity_b: second entity (for comparisons)
            evidence: list of evidence strings used to generate the answer

        Returns:
            VerificationResult
        """
        issues = []
        adjustment = 0.0
        contradictions = []
        failure_tags = []

        answer = answer or ""
        answer_lower = answer.lower()
        evidence = evidence or []

        # -- 1. Relevance Verifier --
        rel_adj, rel_issues = self._check_relevance(query, answer_lower)
        adjustment += rel_adj
        issues.extend(rel_issues)

        # -- 2. Structure Verifier --
        struct_adj, struct_issues = self._check_structure(
            answer, answer_lower, query_type, entity_a, entity_b)
        adjustment += struct_adj
        issues.extend(struct_issues)

        # -- 3. Contradiction Checker --
        contra_flags = self._check_contradictions(answer_lower)
        if contra_flags:
            adjustment -= 0.05 * len(contra_flags)
            contradictions = contra_flags
            issues.append(f"Found {len(contra_flags)} potential contradiction(s)")
            failure_tags.append("V_CONTRADICTION")

        # -- 4. Completion Checker --
        completeness, comp_issues = self._check_completeness(
            query, answer_lower, query_type)
        if completeness < 0.5:
            adjustment -= 0.10
        issues.extend(comp_issues)

        # -- 5. Hard Gates (binary fail conditions) --
        hard_fail, gate_tags, gate_issues = self._check_hard_gates(
            answer, answer_lower, query_type, entity_a, entity_b,
            contradictions, completeness)
        failure_tags.extend(gate_tags)
        issues.extend(gate_issues)

        # -- 6. Grounding Verifier (evidence support check) --
        grounding_score = 1.0  # Default: fully grounded (no evidence to check)
        if evidence:
            grounding_score, ground_adj, ground_issues, ground_tags = \
                self._check_grounding(answer, evidence)
            adjustment += ground_adj
            issues.extend(ground_issues)
            failure_tags.extend(ground_tags)

        # -- 7. Risk / Preference Check --
        risk_adj, risk_issues, risk_tags = self._check_preference_risk(
            answer, answer_lower, query_type)
        adjustment += risk_adj
        issues.extend(risk_issues)
        failure_tags.extend(risk_tags)

        # Hard fail caps score adjustment
        if hard_fail:
            adjustment = min(adjustment, -0.15)

        # Clamp adjustment
        adjustment = max(-0.3, min(0.1, adjustment))

        # Determine trust label
        if self._NO_DATA.search(answer):
            trust = "unverified"
        elif hard_fail:
            trust = "low-confidence"
        elif (adjustment >= -0.02 and completeness >= 0.6
              and not contradictions and grounding_score >= 0.4):
            trust = "verified"
        elif adjustment >= -0.15 and completeness >= 0.3:
            trust = "best-effort"
        else:
            trust = "low-confidence"

        return VerificationResult(
            trust_label=trust,
            score_adjustment=adjustment,
            issues=issues,
            contradiction_flags=contradictions,
            completeness_score=completeness,
            hard_fail=hard_fail,
            failure_tags=failure_tags,
            grounding_score=grounding_score,
        )

    # ── 1. Relevance ────────────────────────────────────────────────

    def _check_relevance(self, query, answer_lower):
        """Check noun coverage: are key query concepts in the answer?"""
        issues = []
        adjustment = 0.0

        q_nouns = [w for w in re.findall(r'\w+', query.lower())
                   if w not in self._STOP and len(w) > 2]

        if not q_nouns:
            return 0.0, []

        matched = sum(1 for n in q_nouns if n in answer_lower)
        coverage = matched / len(q_nouns)

        if coverage >= 0.6:
            adjustment += 0.05  # Good coverage bonus
        elif coverage < 0.3:
            adjustment -= 0.10
            missing = [n for n in q_nouns if n not in answer_lower]
            issues.append(f"Low noun coverage ({coverage:.0%}), missing: {missing[:3]}")

        return adjustment, issues

    # ── 2. Structure ────────────────────────────────────────────────

    def _check_structure(self, answer, answer_lower, query_type,
                         entity_a, entity_b):
        """Check answer structure matches query type."""
        issues = []
        adjustment = 0.0

        if query_type == "comparison":
            # Both entities should be mentioned
            if entity_a and entity_a.lower() not in answer_lower:
                adjustment -= 0.08
                issues.append(f"Missing entity: {entity_a}")
            if entity_b and entity_b.lower() not in answer_lower:
                adjustment -= 0.08
                issues.append(f"Missing entity: {entity_b}")

            # Should have comparative language
            comp_words = ['vs', 'compared', 'differ', 'while', 'whereas',
                         'unlike', 'contrast', 'more than', 'less than',
                         'key differences']
            if not any(w in answer_lower for w in comp_words):
                adjustment -= 0.05
                issues.append("No comparative language found")

        elif query_type == "factual":
            # Should have some definitive statement
            if len(answer.strip()) < 20:
                adjustment -= 0.05
                issues.append("Answer too short for factual query")

        elif query_type == "math":
            # Should contain numbers
            if not re.search(r'\d', answer):
                adjustment -= 0.10
                issues.append("Math answer contains no numbers")

        return adjustment, issues

    # ── 3. Contradictions ───────────────────────────────────────────

    def _check_contradictions(self, answer_lower):
        """Detect contradictory statements within the answer."""
        flags = []

        # Split into sentences
        sentences = re.split(r'[.!?]+', answer_lower)
        if len(sentences) < 2:
            return []

        for pattern_a, pattern_b in self._CONTRADICTIONS:
            has_a = any(re.search(pattern_a, s) for s in sentences)
            has_b = any(re.search(pattern_b, s) for s in sentences)
            if has_a and has_b:
                # Find the actual words
                word_a = re.search(pattern_a, answer_lower)
                word_b = re.search(pattern_b, answer_lower)
                if word_a and word_b:
                    flags.append(f"{word_a.group()} vs {word_b.group()}")

        return flags

    # ── 4. Completeness ─────────────────────────────────────────────

    def _check_completeness(self, query, answer_lower, query_type):
        """Did the answer actually answer the question?"""
        issues = []
        score = 0.0

        # Check answer isn't empty/no-data
        if self._NO_DATA.search(answer_lower):
            return 0.0, ["Answer indicates no data"]

        if len(answer_lower.strip()) < 10:
            return 0.0, ["Answer too short"]

        # Base score from length
        score += min(len(answer_lower) / 200.0, 0.3)

        # Query type specific completeness
        query_lower = query.lower()

        if 'when' in query_lower or 'year' in query_lower or 'founded' in query_lower:
            # Should contain a year/date
            if re.search(r'\b\d{4}\b', answer_lower):
                score += 0.4
            else:
                issues.append("Temporal query but no year/date in answer")
                score += 0.1

        elif 'population' in query_lower or 'how many' in query_lower:
            # Should contain a number
            if re.search(r'\d', answer_lower):
                score += 0.4
            else:
                issues.append("Quantitative query but no numbers in answer")
                score += 0.1

        elif query_type == "comparison":
            # Should address both entities
            score += 0.3  # Base for attempting comparison
            if 'vs' in answer_lower or 'key differences' in answer_lower:
                score += 0.2

        else:
            # General factual: just needs substance
            score += 0.3

        score = min(score, 1.0)
        return score, issues

    # ── 5. Hard Gates (NEW) ─────────────────────────────────────────

    def _check_hard_gates(self, answer, answer_lower, query_type,
                          entity_a, entity_b, contradictions, completeness):
        """
        Binary fail conditions that override the weighted score.

        Any hard fail caps the final score at 0.49 (below SPEAK threshold).
        This prevents structurally broken answers from reaching the user
        even if they score well on other dimensions.

        Returns:
            (hard_fail: bool, failure_tags: list, issues: list)
        """
        hard_fail = False
        tags = []
        issues = []

        # Gate 1: Comparison missing a required entity
        if query_type == "comparison":
            a_present = entity_a and entity_a.lower() in answer_lower
            b_present = entity_b and entity_b.lower() in answer_lower
            if entity_a and entity_b:
                if not a_present and not b_present:
                    hard_fail = True
                    tags.append("V_MISSING_BOTH_ENTITIES")
                    issues.append(
                        f"HARD FAIL: Neither {entity_a} nor {entity_b} "
                        f"in comparison answer")
                elif not a_present or not b_present:
                    missing = entity_a if not a_present else entity_b
                    hard_fail = True
                    tags.append("V_MISSING_ENTITY")
                    issues.append(
                        f"HARD FAIL: Entity '{missing}' absent from "
                        f"comparison answer")

            # Gate 2: Comparison has zero comparative structure
            comp_markers = [
                'vs', 'compared', 'differ', 'while', 'whereas', 'unlike',
                'contrast', 'more than', 'less than', 'key differences',
                'on the other hand', 'in contrast', 'however',
            ]
            if not any(m in answer_lower for m in comp_markers):
                # Only hard-fail if we also have no attribute alignment
                # (e.g., "X: val vs Y: val" patterns)
                if not re.search(r'\w+:\s*\w+.*vs\s+\w+', answer_lower):
                    hard_fail = True
                    tags.append("V_NO_COMPARISON_STRUCTURE")
                    issues.append(
                        "HARD FAIL: No comparative structure in "
                        "comparison answer")

        # Gate 3: Math answer missing a numeric result
        if query_type in ("computation", "math"):
            if not re.search(r'\d', answer):
                hard_fail = True
                tags.append("V_MATH_NO_RESULT")
                issues.append("HARD FAIL: Math answer contains no result")

        # Gate 4: Fatal contradiction (3+ contradiction pairs)
        if len(contradictions) >= 3:
            hard_fail = True
            tags.append("V_FATAL_CONTRADICTION")
            issues.append(
                f"HARD FAIL: {len(contradictions)} contradictions detected")

        # Gate 5: Meta/system contamination
        # LLM sometimes returns meta-statements instead of answers
        _META_PATTERNS = [
            r'^as an ai',
            r'^i am (?:a |an )?(?:language model|ai|assistant)',
            r'^i don.t have (?:access|the ability)',
            r'^based on (?:my|the) training',
        ]
        for pat in _META_PATTERNS:
            if re.search(pat, answer_lower.strip()):
                hard_fail = True
                tags.append("V_META_CONTAMINATION")
                issues.append(
                    "HARD FAIL: Answer is a meta/system statement, "
                    "not a factual response")
                break

        # Gate 6: Factual answer missing the topic entity entirely
        if query_type == "factual" and entity_a and not entity_b:
            if entity_a.lower() not in answer_lower:
                # Check for reasonable synonyms/abbreviations before failing
                # Only fail if the entity is a proper noun (capitalized)
                if entity_a[0].isupper() and len(entity_a) > 3:
                    hard_fail = True
                    tags.append("V_MISSING_TOPIC_ENTITY")
                    issues.append(
                        f"HARD FAIL: Topic entity '{entity_a}' not in "
                        f"factual answer")

        return hard_fail, tags, issues

    # ── 6. Grounding Verifier (NEW) ─────────────────────────────────

    def _check_grounding(self, answer, evidence):
        """
        Check whether answer claims are supported by retrieved evidence.

        Splits the answer into sentences, then for each sentence computes
        the best overlap with any evidence item. Uses either embedding
        cosine similarity (if embedder available) or lexical overlap
        (token Jaccard) as fallback.

        Returns:
            (grounding_score, adjustment, issues, failure_tags)
        """
        issues = []
        tags = []

        if not evidence or not answer.strip():
            return 1.0, 0.0, [], []

        # Split answer into sentences
        answer_sents = [s.strip() for s in re.split(r'[.!?]+', answer)
                        if len(s.strip()) > 15]

        if not answer_sents:
            return 1.0, 0.0, [], []

        # Compute per-sentence grounding
        if self._embedder is not None and self._st_util is not None:
            grounding_score = self._grounding_embedding(
                answer_sents, evidence)
        else:
            grounding_score = self._grounding_lexical(
                answer_sents, evidence)

        # Score interpretation
        adjustment = 0.0
        if grounding_score >= 0.6:
            adjustment += 0.03  # Well-grounded bonus
        elif grounding_score < 0.25:
            adjustment -= 0.12
            tags.append("V_LOW_GROUNDING")
            issues.append(
                f"Low grounding ({grounding_score:.0%}): answer claims "
                f"poorly supported by evidence")
        elif grounding_score < 0.4:
            adjustment -= 0.06
            issues.append(
                f"Moderate grounding ({grounding_score:.0%}): some claims "
                f"lack evidence support")

        return grounding_score, adjustment, issues, tags

    def _grounding_embedding(self, answer_sents, evidence):
        """Compute grounding via embedding cosine similarity."""
        try:
            a_embs = self._embedder.encode(answer_sents, convert_to_tensor=True)
            e_embs = self._embedder.encode(evidence, convert_to_tensor=True)
            # For each answer sentence, find max similarity to any evidence
            sim_matrix = self._st_util.cos_sim(a_embs, e_embs)
            # Best evidence match per sentence
            best_per_sent = sim_matrix.max(dim=1).values
            # Average grounding across all sentences
            return best_per_sent.mean().item()
        except Exception:
            # Fall back to lexical on any error
            return self._grounding_lexical(answer_sents, evidence)

    def _grounding_lexical(self, answer_sents, evidence):
        """Compute grounding via token Jaccard overlap (no model needed)."""
        scores = []
        for sent in answer_sents:
            sent_tokens = set(re.findall(r'\w+', sent.lower()))
            if len(sent_tokens) < 3:
                continue
            best = 0.0
            for ev in evidence:
                ev_tokens = set(re.findall(r'\w+', ev.lower()))
                if not ev_tokens:
                    continue
                intersection = sent_tokens & ev_tokens
                union = sent_tokens | ev_tokens
                jaccard = len(intersection) / len(union) if union else 0
                best = max(best, jaccard)
            scores.append(best)

        if not scores:
            return 1.0  # No scoreable sentences

        return sum(scores) / len(scores)

    # ── 7. Risk / Preference Check (NEW) ────────────────────────────

    def _check_preference_risk(self, answer, answer_lower, query_type):
        """
        Detect unsupported superlatives and preference claims.

        Comparison answers that say "X is better" or "X is recommended"
        without qualifying criteria are a silent failure mode. The answer
        sounds authoritative but has no basis.

        Returns:
            (adjustment, issues, failure_tags)
        """
        issues = []
        tags = []
        adjustment = 0.0

        # Only check comparisons and factual -- math doesn't apply
        if query_type in ("computation", "math"):
            return 0.0, [], []

        # Find preference/superlative claims
        pref_matches = self._PREFERENCE_PATTERNS.findall(answer)
        if not pref_matches:
            return 0.0, [], []

        # Check if the claims are qualified with reasoning
        has_qualifying = bool(self._QUALIFYING_PATTERNS.search(answer))

        if pref_matches and not has_qualifying:
            found = ", ".join(set(m.lower() for m in pref_matches[:3]))
            adjustment -= 0.08
            tags.append("V_UNSUPPORTED_PREFERENCE")
            issues.append(
                f"Unsupported preference language ({found}) without "
                f"qualifying criteria")

            # For comparisons, this is more serious
            if query_type == "comparison":
                adjustment -= 0.04  # Extra penalty
                issues.append(
                    "Comparison makes preference claim without "
                    "evidence-based justification")

        return adjustment, issues, tags
