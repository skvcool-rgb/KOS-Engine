"""
KOS V2.0 — Algorithmic Weaver (Discourse NLG).

Generates natural language responses from graph traversal results.
Gathers expansive evidence from ALL seed-connected edges, scores
sentences deterministically via intent matching (WHERE/WHEN/WHO +40),
and returns the top 2 raw quotes for the LLM Mouth.
"""
import re


class AlgorithmicWeaver:
    def weave(self, kernel, seeds_human: list, top_results: list,
              lexicon, seed_uuids: list, raw_prompt: str) -> str:
        prompt_lower = raw_prompt.lower()
        ignore = {"what", "where", "when", "who", "why", "how",
                  "is", "the", "a", "an", "does", "it", "are", "do"}
        prompt_words = {w for w in re.findall(r'\w+', prompt_lower)
                        if w not in ignore}

        # 1. Gather expansive evidence.
        # Don't just look at Top Results. Look at every sentence
        # that connects to our seeds!
        evidence_set = set()
        for suid in seed_uuids:
            # Gather evidence from the seed to the top results
            for ans_uuid, _ in top_results:
                evidence_set.update(
                    getattr(kernel, 'provenance', {}).get(
                        tuple(sorted([suid, ans_uuid])), set()))

            # Gather ALL general edges attached to this seed
            # (Massive net for niche facts)
            if suid in kernel.nodes:
                for target_uuid in kernel.nodes[suid].connections:
                    evidence_set.update(
                        getattr(kernel, 'provenance', {}).get(
                            tuple(sorted([suid, target_uuid])), set()))

        if not evidence_set:
            return "No relevant context found in database."

        # 2. Score the sentences deterministically
        scored_sentences = []
        for sent in evidence_set:
            sent_score = 0
            sent_lower = sent.lower()

            # CRITICAL INTENT SCORING
            if any(w in prompt_lower for w in
                   ["where", "located", "place", "find"]):
                if any(w in sent_lower for w in
                       [" in ", " at ", " located ", " province ",
                        " country ", " region ", " shore "]):
                    sent_score += 40

            if any(w in prompt_lower for w in
                   ["when", "year", "time", "date", "formed"]):
                if re.search(r'\b(?:16|17|18|19|20)\d{2}\b', sent):
                    sent_score += 40

            if any(w in prompt_lower for w in
                   ["who", "name", "named", "founded",
                    "established", "created"]):
                if any(w in sent_lower for w in
                       [" by ", " named ", " founded ",
                        " established ", " formed ",
                        " incorporated "]):
                    sent_score += 40

            # WHAT-ATTRIBUTE intent: if prompt asks about a specific
            # attribute (climate, population, economy), boost sentences
            # that contain that exact attribute word
            attribute_words = prompt_words - {w for w in prompt_words
                                               if w in ignore or len(w) < 4}
            for attr in attribute_words:
                if attr in sent_lower:
                    sent_score += 35  # Strong boost for exact attribute match

            # GLOBAL sports noise suppression — unless user explicitly
            # asks about sports, punish sports sentences in ALL queries
            is_sports_query = any(w in prompt_lower for w in
                                  ["sport", "team", "play", "game",
                                   "baseball", "soccer", "hockey",
                                   "basketball", "football"])
            if not is_sports_query:
                if any(w in sent_lower for w in
                       ["sports", "baseball", "soccer", "fifa",
                        "hockey", "basketball", "football",
                        "blue jays", "stadium"]):
                    sent_score -= 50

            # Keyword exact overlap
            overlap = prompt_words.intersection(
                set(re.findall(r'\w+', sent_lower)))
            sent_score += len(overlap) * 20

            scored_sentences.append((sent_score, sent))

        # 3. Sort highest scores first
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # 4. Return ONLY the top 2 highest scoring, raw sentences
        # for the LLM Mouth (filtering out negative score garbage)
        top_evidence = [sent for score, sent in scored_sentences[:2]
                        if score >= 0]

        # If no sentences passed the positive threshold, fall back
        if not top_evidence and scored_sentences:
            top_evidence = [scored_sentences[0][1]]

        return "\n".join(top_evidence)
