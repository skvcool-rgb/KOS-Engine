"""
KOS V9.0 — 4-Layer Hybrid Relevance Engine

Production-grade answer relevance scoring:
  Layer 1: Keyword Weighting (fast filter, IDF-weighted)
  Layer 2: Synonym Expansion (WordNet + Lexicon synonym net)
  Layer 3: Embedding Similarity (SentenceTransformer cosine)
  Layer 4: Relation Validation (graph edge connectivity check)

Each layer returns a score in [0, 1]. The final relevance score is a
weighted combination that balances speed (keywords) with depth (graph).

Usage:
    scorer = RelevanceScorer(kernel, lexicon, embedder, st_util)
    score, breakdown = scorer.score(query, answer)
    if score < 0.45:
        trigger_auto_forage()
"""

import re
import math
from collections import Counter

try:
    from nltk.corpus import wordnet as wn
except Exception:
    wn = None


# Common English stop words — excluded from keyword extraction
_STOP_WORDS = frozenset({
    "what", "where", "when", "who", "why", "how", "is", "the", "a", "an",
    "does", "it", "are", "do", "of", "in", "to", "for", "and", "or",
    "about", "tell", "me", "please", "can", "could", "from", "this",
    "that", "be", "was", "were", "been", "have", "has", "had", "will",
    "would", "shall", "should", "may", "might", "much", "many", "some",
    "any", "all", "no", "not", "but", "if", "at", "by", "on", "with",
    "distance", "between", "far", "close", "near", "also", "very",
    "just", "than", "then", "there", "here", "these", "those", "its",
    "which", "more", "most", "other", "such", "only", "same", "into",
    "over", "after", "before", "through", "during", "each", "few",
    "emotion", "joy", "neutral", "curiosity", "additionally",
})


def _extract_nouns(text):
    """Extract content words from text, excluding stop words."""
    words = re.findall(r'\w+', text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 2]


class RelevanceScorer:
    """4-layer hybrid relevance scoring engine."""

    def __init__(self, kernel=None, lexicon=None, embedder=None, st_util=None):
        self.kernel = kernel
        self.lexicon = lexicon
        self.embedder = embedder
        self.st_util = st_util

        # Weights for each layer (must sum to 1.0)
        self.w_keyword = 0.20   # Layer 1: keyword weighting
        self.w_synonym = 0.15   # Layer 2: synonym expansion
        self.w_embedding = 0.45 # Layer 3: embedding similarity
        self.w_relation = 0.20  # Layer 4: relation validation

        # Off-topic threshold
        self.threshold = 0.46

    # ── Layer 1: Keyword Weighting (IDF-weighted) ──────────────────

    def _keyword_score(self, query, answer):
        """
        IDF-weighted keyword overlap.

        Rare words (like 'moon', 'apixaban') get higher weight than
        common words (like 'earth', 'system'). This ensures the scorer
        cares more about specific query terms.
        """
        q_nouns = _extract_nouns(query)
        if not q_nouns:
            return 0.5, {"q_nouns": [], "matched": [], "weights": {}}

        a_lower = answer.lower()

        # Compute IDF-like weight: rarer words in the answer get higher weight
        # Words NOT in the answer are the ones we care about most
        a_words = set(re.findall(r'\w+', a_lower))

        # Weight each query noun by inverse frequency in common English
        # Simple heuristic: shorter words are more common, domain-specific words are rare
        weights = {}
        for noun in q_nouns:
            # Base weight: length-based rarity estimate
            base = min(len(noun) / 10.0, 1.0)
            # Boost if it looks like a proper noun or domain term (>6 chars)
            if len(noun) > 6:
                base *= 1.5
            weights[noun] = base

        # Normalize weights
        total_w = sum(weights.values())
        if total_w == 0:
            return 0.5, {"q_nouns": q_nouns, "matched": [], "weights": {}}

        for k in weights:
            weights[k] /= total_w

        # Weighted match score
        matched = []
        score = 0.0
        for noun in q_nouns:
            if noun in a_lower:
                score += weights[noun]
                matched.append(noun)

        return score, {"q_nouns": q_nouns, "matched": matched, "weights": weights}

    # ── Layer 2: Synonym Expansion ─────────────────────────────────

    def _synonym_score(self, query, answer):
        """
        Expand query nouns via WordNet + lexicon synonym net.

        If query says 'moon' but answer says 'lunar', this layer catches it.
        If query says 'speed' but answer says 'velocity', this layer catches it.
        """
        q_nouns = _extract_nouns(query)
        if not q_nouns:
            return 0.5, {"expanded": {}}

        a_lower = answer.lower()
        expanded = {}
        matched_via_synonym = []

        for noun in q_nouns:
            synonyms = set()

            # Source 1: WordNet synonyms
            if wn is not None:
                try:
                    for ss in wn.synsets(noun, pos=wn.NOUN)[:3]:
                        for lemma in ss.lemma_names():
                            syn = lemma.lower().replace("_", " ")
                            if syn != noun:
                                synonyms.add(syn)
                except Exception:
                    pass

            # Source 2: Lexicon UUID reverse lookup (all words sharing same UUID)
            if self.lexicon and noun in self.lexicon.word_to_uuid:
                uuid = self.lexicon.word_to_uuid[noun]
                for word, uid in self.lexicon.word_to_uuid.items():
                    if uid == uuid and word != noun:
                        synonyms.add(word)

            # NOTE: Phonetic neighbors deliberately excluded —
            # too loose for relevance (e.g., moon -> men via Metaphone)

            expanded[noun] = synonyms

            # Check if any synonym appears in the answer
            if noun not in a_lower:
                for syn in synonyms:
                    # Only count multi-char synonyms that aren't stop words
                    if len(syn) > 3 and syn not in _STOP_WORDS and syn in a_lower:
                        matched_via_synonym.append((noun, syn))
                        break

        # Score: fraction of ALL query nouns matched (directly or via synonym)
        direct_matches = sum(1 for n in q_nouns if n in a_lower)
        synonym_matches = len(matched_via_synonym)
        total_matched = direct_matches + synonym_matches
        score = total_matched / len(q_nouns) if q_nouns else 0.5

        return score, {"expanded": expanded, "matched": matched_via_synonym,
                       "direct": direct_matches, "via_synonym": synonym_matches}

    # ── Layer 3: Embedding Similarity ──────────────────────────────

    def _embedding_score(self, query, answer):
        """
        SentenceTransformer cosine similarity between query and answer.
        This is the semantic backbone — catches paraphrases, rephrasing,
        and conceptual similarity that keywords miss.
        """
        if self.embedder is None or self.st_util is None:
            return 0.5, {"available": False}

        try:
            q_emb = self.embedder.encode(query, convert_to_tensor=True)
            # Truncate answer to 500 chars for speed
            a_emb = self.embedder.encode(answer[:500], convert_to_tensor=True)
            cos_sim = float(self.st_util.cos_sim(q_emb, a_emb)[0][0])
            return max(0.0, cos_sim), {"cos_sim": cos_sim}
        except Exception as e:
            return 0.5, {"error": str(e)}

    # ── Layer 4: Relation Validation (Graph Edge Check) ────────────

    def _relation_score(self, query, answer):
        """
        Concept Graph Matching — validates that the answer's evidence path
        actually connects query concepts, not just any concepts.

        Strategy:
        1. Resolve query nouns to graph UUIDs
        2. Resolve answer nouns to graph UUIDs
        3. Check: do answer nodes lie on the SHORTEST PATH between query nodes?
        4. If answer nodes are in a completely different graph region, score = low

        This catches: "moon + earth" query returning "sun + orbit + entanglement"
        (answer nodes are far from query node "moon" even if near "earth")
        """
        if self.kernel is None or self.lexicon is None:
            return 0.5, {"available": False}

        q_nouns = _extract_nouns(query)
        a_nouns = _extract_nouns(answer)

        if not q_nouns or not a_nouns:
            return 0.5, {"q_nodes": 0, "a_nodes": 0}

        # Resolve query nouns to graph node UUIDs
        q_uuids = set()
        q_missing = 0
        for noun in q_nouns:
            if noun in self.lexicon.word_to_uuid:
                uid = self.lexicon.word_to_uuid[noun]
                if uid in self.kernel.nodes:
                    q_uuids.add(uid)
                else:
                    q_missing += 1
            else:
                q_missing += 1

        # Resolve answer nouns to graph node UUIDs
        a_uuids = set()
        for noun in a_nouns[:15]:
            if noun in self.lexicon.word_to_uuid:
                uid = self.lexicon.word_to_uuid[noun]
                if uid in self.kernel.nodes:
                    a_uuids.add(uid)

        if not q_uuids:
            # Query concepts not in graph at all — can't validate
            return 0.2, {"q_nodes": 0, "a_nodes": len(a_uuids), "q_missing": q_missing}

        if not a_uuids:
            return 0.3, {"q_nodes": len(q_uuids), "a_nodes": 0}

        # For each query node, check how many answer nodes are within 2 hops
        # This measures: is the answer drawn from the NEIGHBORHOOD of query concepts?
        q_neighborhoods = {}
        total_proximity = 0
        checks = 0

        for q_uid in q_uuids:
            q_node = self.kernel.nodes.get(q_uid)
            if not q_node:
                continue
            # 1-hop neighbors
            neighbors_1 = set(q_node.connections.keys())
            # 2-hop neighbors (sample top 50 for speed)
            neighbors_2 = set()
            for n1_uid in list(neighbors_1)[:50]:
                n1_node = self.kernel.nodes.get(n1_uid)
                if n1_node:
                    neighbors_2.update(list(n1_node.connections.keys())[:20])

            neighborhood = neighbors_1 | neighbors_2 | {q_uid}
            q_neighborhoods[q_uid] = len(neighborhood)

            for a_uid in a_uuids:
                checks += 1
                if a_uid in neighborhood:
                    total_proximity += 1

        score = total_proximity / checks if checks > 0 else 0.0

        # Coverage penalty: if key query nouns aren't in graph
        if q_nouns and q_missing > 0:
            score *= (1.0 - q_missing / len(q_nouns) * 0.5)

        return min(1.0, score), {
            "q_nodes": len(q_uuids),
            "a_nodes": len(a_uuids),
            "proximity_hits": total_proximity,
            "checks": checks,
            "q_missing": q_missing,
        }

    # ── Combined Score ─────────────────────────────────────────────

    def score(self, query, answer):
        """
        Compute the 4-layer hybrid relevance score.

        Returns:
            (float, dict): Overall score [0-1] and per-layer breakdown.
        """
        kw_score, kw_detail = self._keyword_score(query, answer)
        syn_score, syn_detail = self._synonym_score(query, answer)
        emb_score, emb_detail = self._embedding_score(query, answer)
        rel_score, rel_detail = self._relation_score(query, answer)

        # Weighted combination
        final = (self.w_keyword * kw_score +
                 self.w_synonym * syn_score +
                 self.w_embedding * emb_score +
                 self.w_relation * rel_score)

        breakdown = {
            "keyword": {"score": round(kw_score, 3), "weight": self.w_keyword, **kw_detail},
            "synonym": {"score": round(syn_score, 3), "weight": self.w_synonym, **syn_detail},
            "embedding": {"score": round(emb_score, 3), "weight": self.w_embedding, **emb_detail},
            "relation": {"score": round(rel_score, 3), "weight": self.w_relation, **rel_detail},
            "final": round(final, 3),
            "threshold": self.threshold,
            "off_topic": final < self.threshold,
        }

        return final, breakdown

    def is_off_topic(self, query, answer):
        """Quick check: is this answer off-topic for this query?"""
        score, _ = self.score(query, answer)
        return score < self.threshold
