"""
KOS V5.1 — Temporal Reasoning Module (Fix #3).

Enables before/after/during queries on dated facts.

"Was Toronto founded before Montreal?" →
    Toronto.properties['year'] = 1834
    Montreal.properties['year'] = 1642
    1834 > 1642 → "No, Montreal was founded first (1642 vs 1834)"

Also handles:
    - "What happened first?"
    - "What was founded in the 1800s?"
    - Chronological sorting of events
"""

import re


class TemporalReasoner:
    """
    Temporal comparison and reasoning on node properties.

    Works with the numeric properties extracted by TextDriver
    (Fix #8) — specifically 'year', 'founded', 'date' properties.
    """

    # Temporal query indicators
    BEFORE_WORDS = {"before", "earlier", "prior", "preceding", "older"}
    AFTER_WORDS = {"after", "later", "following", "newer", "younger"}
    FIRST_WORDS = {"first", "earliest", "oldest", "original"}
    LAST_WORDS = {"last", "latest", "newest", "most recent", "recent"}
    DURING_WORDS = {"during", "in the", "between", "century", "decade"}

    def detect_temporal_query(self, prompt: str) -> dict:
        """
        Detect if a query involves temporal reasoning.

        Returns: {type: "before"|"after"|"first"|"last"|"range"|None,
                  entities: [...], year_range: (start, end)}
        """
        prompt_lower = prompt.lower()
        words = set(prompt_lower.split())

        result = {"type": None, "entities": [], "year_range": None}

        if words & self.BEFORE_WORDS:
            result["type"] = "before"
        elif words & self.AFTER_WORDS:
            result["type"] = "after"
        elif words & self.FIRST_WORDS:
            result["type"] = "first"
        elif words & self.LAST_WORDS:
            result["type"] = "last"

        # Detect year ranges: "in the 1800s", "between 1800 and 1900"
        range_match = re.search(r'(\d{4})s', prompt_lower)
        if range_match:
            base = int(range_match.group(1))
            result["year_range"] = (base, base + 99)
            result["type"] = result["type"] or "range"

        between_match = re.search(r'between\s+(\d{4})\s+and\s+(\d{4})',
                                   prompt_lower)
        if between_match:
            result["year_range"] = (int(between_match.group(1)),
                                     int(between_match.group(2)))
            result["type"] = result["type"] or "range"

        return result

    def compare_temporal(self, kernel, node_a_id: str, node_b_id: str,
                         lexicon=None) -> dict:
        """
        Compare two nodes temporally.

        Returns which came first and the time difference.
        """
        node_a = kernel.nodes.get(node_a_id)
        node_b = kernel.nodes.get(node_b_id)

        if not node_a or not node_b:
            return {"error": "Nodes not found"}

        # Look for temporal properties
        time_props = ['year', 'founded', 'established', 'date',
                      'incorporated', 'created']

        year_a = None
        year_b = None

        for prop in time_props:
            if prop in node_a.properties and year_a is None:
                year_a = node_a.properties[prop]
            if prop in node_b.properties and year_b is None:
                year_b = node_b.properties[prop]

        if year_a is None or year_b is None:
            return {"error": "Temporal data not available for both nodes"}

        name_a = lexicon.get_word(node_a_id) if lexicon else node_a_id
        name_b = lexicon.get_word(node_b_id) if lexicon else node_b_id

        if year_a < year_b:
            return {
                "first": name_a, "first_year": year_a,
                "second": name_b, "second_year": year_b,
                "difference": year_b - year_a,
                "answer": f"{name_a} ({year_a:.0f}) came before "
                          f"{name_b} ({year_b:.0f}) by "
                          f"{year_b - year_a:.0f} years."
            }
        elif year_a > year_b:
            return {
                "first": name_b, "first_year": year_b,
                "second": name_a, "second_year": year_a,
                "difference": year_a - year_b,
                "answer": f"{name_b} ({year_b:.0f}) came before "
                          f"{name_a} ({year_a:.0f}) by "
                          f"{year_a - year_b:.0f} years."
            }
        else:
            return {
                "first": name_a, "first_year": year_a,
                "second": name_b, "second_year": year_b,
                "difference": 0,
                "answer": f"{name_a} and {name_b} both date to "
                          f"{year_a:.0f}."
            }

    def find_in_range(self, kernel, year_start: int, year_end: int,
                      lexicon=None) -> list:
        """Find all nodes with temporal properties in a year range."""
        results = []
        time_props = ['year', 'founded', 'established', 'date',
                      'incorporated', 'created']

        for nid, node in kernel.nodes.items():
            for prop in time_props:
                if prop in node.properties:
                    year = node.properties[prop]
                    try:
                        y = float(year)
                        if year_start <= y <= year_end:
                            name = lexicon.get_word(nid) if lexicon else nid
                            results.append((name, y, prop))
                    except (ValueError, TypeError):
                        continue

        results.sort(key=lambda x: x[1])
        return results

    def chronological_sort(self, kernel, node_ids: list,
                            lexicon=None) -> list:
        """Sort nodes chronologically by their temporal properties."""
        timed = []
        time_props = ['year', 'founded', 'established', 'date']

        for nid in node_ids:
            node = kernel.nodes.get(nid)
            if not node:
                continue
            for prop in time_props:
                if prop in node.properties:
                    name = lexicon.get_word(nid) if lexicon else nid
                    timed.append((name, float(node.properties[prop]), nid))
                    break

        timed.sort(key=lambda x: x[1])
        return timed
