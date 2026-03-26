"""
KOS Answer-Type Validator

Validates that answers match their expected query type before returning.
Used by the query pipeline for coverage-aware confidence adjustment.
"""

import re


class AnswerValidator:
    """Validates answers match their expected query type before returning."""

    _COMPARATIVE_PATTERNS = re.compile(
        r'(?i)\b(vs\.?|compared|differ|while|whereas|unlike|in contrast|'
        r'on the other hand|more than|less than|greater|smaller|larger|'
        r'faster|slower|higher|lower|better|worse)\b'
    )

    _NO_DATA_PATTERNS = re.compile(
        r'(?i)^(i don.t have|no data|no information|i.m not sure|'
        r'i cannot|i don.t know|unknown|n/a|not available|'
        r'i have no|there is no data)'
    )

    def validate_comparison(self, answer, entity_a, entity_b):
        """
        Returns (is_valid, penalties, issues) where:
        - is_valid: bool (True if answer passes minimum quality)
        - penalties: float (0.0-0.3 confidence penalty)
        - issues: list of strings describing problems

        Checks:
        1. Both entities mentioned in answer (0.15 penalty per missing)
        2. At least one comparative structure present (0.10 penalty)
        3. Answer length >= 30 chars (0.05 penalty if too short)
        4. Not a "no data" response (0.20 penalty)
        """
        penalties = 0.0
        issues = []
        answer_lower = (answer or "").lower()

        # Check 1: Both entities mentioned
        a_lower = entity_a.lower()
        b_lower = entity_b.lower()
        if a_lower not in answer_lower:
            penalties += 0.15
            issues.append(f"Entity '{entity_a}' not mentioned in answer")
        if b_lower not in answer_lower:
            penalties += 0.15
            issues.append(f"Entity '{entity_b}' not mentioned in answer")

        # Check 2: Comparative structure present
        has_numeric = bool(re.search(r'\d+\s*(?:vs|compared|than)', answer_lower))
        has_comparative = bool(self._COMPARATIVE_PATTERNS.search(answer or ""))
        if not has_comparative and not has_numeric:
            penalties += 0.10
            issues.append("No comparative structure found")

        # Check 3: Answer length
        if len((answer or "").strip()) < 30:
            penalties += 0.05
            issues.append("Answer too short for comparison (<30 chars)")

        # Check 4: No-data response
        if self._NO_DATA_PATTERNS.search((answer or "").strip()):
            penalties += 0.20
            issues.append("Answer is a 'no data' response")

        # Cap penalties at 0.5 (since max theoretical is 0.65)
        penalties = min(penalties, 0.5)
        is_valid = penalties < 0.3

        return (is_valid, penalties, issues)

    def validate_factual(self, answer, query):
        """
        Returns (is_valid, penalties, issues).

        Checks:
        1. Answer not empty or "no data" (0.20 penalty)
        2. At least one query content word in answer (0.10 penalty)
        3. Answer length >= 20 chars (0.05 penalty)
        """
        penalties = 0.0
        issues = []
        answer_stripped = (answer or "").strip()

        # Check 1: Not empty or no-data
        if not answer_stripped or self._NO_DATA_PATTERNS.search(answer_stripped):
            penalties += 0.20
            issues.append("Answer is empty or 'no data' response")

        # Check 2: Query content word overlap
        stop = {'what', 'is', 'the', 'a', 'an', 'of', 'in', 'to', 'for',
                'and', 'or', 'how', 'does', 'tell', 'me', 'about', 'are',
                'who', 'where', 'when', 'why', 'which', 'do', 'can', 'will',
                'was', 'were', 'has', 'have', 'been', 'be', 'with', 'that',
                'this', 'from', 'by', 'at', 'on', 'it', 'its', 'not', 'but'}
        query_words = {w for w in re.findall(r'\w+', (query or "").lower())
                       if len(w) > 2 and w not in stop}
        answer_lower = (answer or "").lower()
        if query_words and not any(w in answer_lower for w in query_words):
            penalties += 0.10
            issues.append("No query content words found in answer")

        # Check 3: Answer length
        if len(answer_stripped) < 20:
            penalties += 0.05
            issues.append("Answer too short (<20 chars)")

        penalties = min(penalties, 0.35)
        is_valid = penalties < 0.25

        return (is_valid, penalties, issues)
