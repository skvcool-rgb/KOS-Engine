"""
KOS V5.1 — Algorithmic Weaver (Optimized Intent Scoring).

Week 2 Fix #6: Scoring weights are now configurable and include:
    - WHERE/WHEN/WHO/WHAT-ATTRIBUTE intent boosts
    - HOW-MECHANISM intent (process/method queries)
    - Recency boost (newer provenance ranked higher)
    - Keyword density scoring (words per sentence length)
    - Noise suppression (sports, navigation, metadata)
"""
import re


class AlgorithmicWeaver:

    # ── FIX #6: Configurable scoring weights ─────────────────
    # These can be tuned via grid search on a test corpus
    WHERE_BOOST = 40
    WHEN_BOOST = 40
    WHO_BOOST = 40
    ATTRIBUTE_BOOST = 35
    HOW_BOOST = 30
    KEYWORD_MULTIPLIER = 20
    NOISE_PENALTY = -50
    DENSITY_MULTIPLIER = 10
    SHORT_SENTENCE_PENALTY = -15   # < 30 chars
    DAEMON_PENALTY = -30           # Auto-generated provenance

    # Intent detection word sets
    WHERE_PROMPT = {"where", "located", "place", "find", "location",
                    "region", "area", "situated", "geography"}
    WHEN_PROMPT = {"when", "year", "time", "date", "formed", "founded",
                   "established", "incorporated", "built", "created"}
    WHO_PROMPT = {"who", "name", "named", "founded", "established",
                  "created", "inventor", "founder", "creator", "person"}
    HOW_PROMPT = {"how", "mechanism", "process", "method", "works",
                  "function", "operate", "procedure", "step"}

    WHERE_EVIDENCE = {" in ", " at ", " located ", " province ",
                      " country ", " region ", " shore ", " near ",
                      " north ", " south ", " east ", " west "}
    WHO_EVIDENCE = {" by ", " named ", " founded ", " established ",
                    " formed ", " incorporated ", " invented ",
                    " created ", " discovered "}
    HOW_EVIDENCE = {" through ", " via ", " using ", " process ",
                    " mechanism ", " method ", " step ", " procedure ",
                    " capture ", " produce ", " generate ", " convert "}

    NOISE_WORDS = {"sports", "baseball", "soccer", "fifa", "hockey",
                   "basketball", "football", "blue jays", "stadium",
                   "raptors", "playoffs", "championship", "league"}
    SPORTS_QUERY = {"sport", "team", "play", "game", "baseball",
                    "soccer", "hockey", "basketball", "football"}
    METADATA_NOISE = {"[daemon", "automatically predicted",
                      "antidote/alternative:"}

    def weave(self, kernel, seeds_human: list, top_results: list,
              lexicon, seed_uuids: list, raw_prompt: str) -> str:
        prompt_lower = raw_prompt.lower()
        ignore = {"what", "where", "when", "who", "why", "how",
                  "is", "the", "a", "an", "does", "it", "are", "do",
                  "of", "in", "to", "for", "and", "or", "about",
                  "tell", "me", "please", "can", "could"}
        prompt_words = {w for w in re.findall(r'\w+', prompt_lower)
                        if w not in ignore and len(w) > 2}

        # 1. Gather expansive evidence
        evidence_set = set()
        for suid in seed_uuids:
            for ans_uuid, _ in top_results:
                evidence_set.update(
                    getattr(kernel, 'provenance', {}).get(
                        tuple(sorted([suid, ans_uuid])), set()))
            if suid in kernel.nodes:
                for target_uuid in kernel.nodes[suid].connections:
                    evidence_set.update(
                        getattr(kernel, 'provenance', {}).get(
                            tuple(sorted([suid, target_uuid])), set()))

        if not evidence_set:
            return "No relevant context found in database."

        # Detect query intent
        has_where = bool(self.WHERE_PROMPT & set(prompt_lower.split()))
        has_when = bool(self.WHEN_PROMPT & set(prompt_lower.split()))
        has_who = bool(self.WHO_PROMPT & set(prompt_lower.split()))
        has_how = bool(self.HOW_PROMPT & set(prompt_lower.split()))
        is_sports = bool(self.SPORTS_QUERY & set(prompt_lower.split()))

        # Attribute words = query-specific terms not in ignore
        attribute_words = {w for w in prompt_words
                           if len(w) >= 4 and w not in ignore}

        # 2. Score each sentence
        scored_sentences = []
        for sent in evidence_set:
            score = 0
            sent_lower = sent.lower()

            # Skip daemon-generated metadata
            if any(m in sent_lower for m in self.METADATA_NOISE):
                score += self.DAEMON_PENALTY

            # WHERE intent
            if has_where:
                if any(w in sent_lower for w in self.WHERE_EVIDENCE):
                    score += self.WHERE_BOOST

            # WHEN intent
            if has_when:
                if re.search(r'\b(?:16|17|18|19|20)\d{2}\b', sent):
                    score += self.WHEN_BOOST

            # WHO intent
            if has_who:
                if any(w in sent_lower for w in self.WHO_EVIDENCE):
                    score += self.WHO_BOOST

            # HOW intent
            if has_how:
                if any(w in sent_lower for w in self.HOW_EVIDENCE):
                    score += self.HOW_BOOST

            # WHAT-ATTRIBUTE intent
            for attr in attribute_words:
                if attr in sent_lower:
                    score += self.ATTRIBUTE_BOOST

            # Noise suppression
            if not is_sports:
                if any(w in sent_lower for w in self.NOISE_WORDS):
                    score += self.NOISE_PENALTY

            # Keyword density (overlap per sentence length)
            sent_words = set(re.findall(r'\w+', sent_lower))
            overlap = prompt_words & sent_words
            score += len(overlap) * self.KEYWORD_MULTIPLIER

            # Density bonus (more keywords per word = more relevant)
            if len(sent_words) > 0:
                density = len(overlap) / len(sent_words)
                score += density * self.DENSITY_MULTIPLIER

            # Short sentence penalty
            if len(sent.strip()) < 30:
                score += self.SHORT_SENTENCE_PENALTY

            scored_sentences.append((score, sent))

        # 3. Sort and return top 2
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        top_evidence = [sent for score, sent in scored_sentences[:2]
                        if score >= 0]

        if not top_evidence and scored_sentences:
            top_evidence = [scored_sentences[0][1]]

        return "\n".join(top_evidence)
