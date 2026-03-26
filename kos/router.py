"""
KOS V5.0 — Shell (LLM Ear + System 2 Brain + Active Inference + LLM Mouth).

The full cognitive pipeline:
    0.  Math Intercept (SymPy exact computation)
    0.5 Pre-LLM Raw Word Scan (typo rescue before normalization)
    1.  LLM Ear (JSON keyword extraction — deterministic)
    2.  6-Layer Word Resolution Cascade
    3.  System 2: ShadowKernel simulates multiple interpretations
    4.  Active Inference: if entropy too high, Forager learns autonomously
    5.  Algorithmic Weaver scores evidence deterministically
    6.  LLM Mouth synthesizes 1-2 sentence answer
"""
import os
import re
import json
import itertools
from openai import OpenAI


class KOSShell:
    # Entropy threshold — above this, Active Inference triggers
    ENTROPY_THRESHOLD = 15.0

    def __init__(self, kernel, lexicon, enable_forager: bool = True):
        self.kernel = kernel
        self.lexicon = lexicon
        self.client = OpenAI()

        # Math Coprocessor (0% hallucination — SymPy exact computation)
        from .drivers.math import MathDriver
        self.math = MathDriver()

        # System 2: Metacognitive Shadow Simulation
        from .metacognition import ShadowKernel
        self.shadow = ShadowKernel(kernel)

        # Phase 2: Proactive Attention Controller
        from .attention import AttentionController
        self.attention = AttentionController(kernel, lexicon)

        # Layer 5: Active Inference — Autonomous Web Forager
        self.forager = None
        if enable_forager:
            try:
                from .forager import WebForager
                self.forager = WebForager(kernel, lexicon)
            except Exception:
                pass  # Forager optional — runs without it

        # Semantic Vector Fallback (lazy-loaded)
        self.embedder = None
        self.node_embeddings = None
        self.embedded_uuids = []
        self._st_util = None
        self._word_emb_cache = {}  # Cache word -> embedding

    def _ensure_embeddings(self):
        """Lazily build/rebuild cached embeddings — incremental for new nodes."""
        if self.embedder is None:
            from kos.router_offline import _get_embedder
            self.embedder, self._st_util = _get_embedder()
            if self.embedder is None:
                return False
        current_uuids = list(self.kernel.nodes.keys())
        if self.node_embeddings is None:
            self.embedded_uuids = current_uuids
            plain_words = [self.lexicon.get_word(uid) for uid in current_uuids]
            self.node_embeddings = self.embedder.encode(
                plain_words, convert_to_tensor=True)
        elif len(current_uuids) != len(self.embedded_uuids):
            import torch
            old_set = set(self.embedded_uuids)
            new_uuids = [u for u in current_uuids if u not in old_set]
            if new_uuids:
                new_words = [self.lexicon.get_word(uid) for uid in new_uuids]
                new_embs = self.embedder.encode(new_words, convert_to_tensor=True)
                self.node_embeddings = torch.cat([self.node_embeddings, new_embs], dim=0)
                self.embedded_uuids = self.embedded_uuids + new_uuids
        return True

    def _resolve_word(self, w, known_words):
        """6-layer word->UUID resolution cascade."""
        import difflib
        import jellyfish

        # Layer 1: Exact lexicon match
        if w in self.lexicon.word_to_uuid:
            return self.lexicon.word_to_uuid[w]

        # Layer 2: Difflib fuzzy match (character similarity >= 0.6)
        matches = difflib.get_close_matches(w, known_words, n=1, cutoff=0.6)
        if matches:
            return self.lexicon.word_to_uuid[matches[0]]

        # Layer 3a: Metaphone phonetic rescue
        sound = jellyfish.metaphone(w)
        candidates = self.lexicon.sound_to_uuids.get(sound, set())
        if candidates:
            return next(iter(candidates))

        # Layer 3b: Soundex phonetic rescue (broader net)
        sdx = jellyfish.soundex(w)
        sdx_candidates = getattr(self.lexicon, 'soundex_to_uuids', {}).get(sdx, set())
        if sdx_candidates:
            return next(iter(sdx_candidates))

        # Layer 4: Hypernym/hyponym taxonomy walk (WordNet)
        resolved = self.lexicon.resolve_hypernym(w, self.kernel.nodes)
        if resolved:
            return resolved

        # Layer 5: Semantic vector fallback (all-MiniLM-L6-v2) with cache
        if self.kernel.nodes and self._ensure_embeddings():
            if w in self._word_emb_cache:
                w_emb = self._word_emb_cache[w]
            else:
                w_emb = self.embedder.encode(w, convert_to_tensor=True)
                self._word_emb_cache[w] = w_emb
            hits = self._st_util.cos_sim(w_emb, self.node_embeddings)[0]
            best_score = hits.max().item()
            best_idx = hits.argmax().item()
            if best_score > 0.50:
                return self.embedded_uuids[best_idx]

        return None

    def _extract_keywords(self, user_prompt: str) -> dict:
        """LLM Ear: strict JSON keyword extraction."""
        extraction_system = """
        You are a deterministic database router. Read the user's prompt.
        If it is casual conversation ('hello', 'how are you'), output: {"status": "CONVERSATION"}
        Otherwise, extract the 2 to 4 core entities/actions needed for a search query.
        Output ONLY valid JSON in this exact format:
        {"status": "EXECUTE", "keywords": ["keyword1", "keyword2"]}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": extraction_system},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception:
            return {"status": "EXECUTE", "keywords": []}

    def _resolve_seeds(self, user_prompt: str, extracted: dict) -> tuple:
        """
        Pre-LLM scan + LLM keyword resolution -> merged seed set.
        Also returns the raw words for Active Inference fallback.
        """
        stopwords = {"what", "where", "when", "who", "why", "how",
                     "is", "the", "a", "an", "does", "it", "are", "do",
                     "of", "in", "to", "for", "and", "or", "so", "about",
                     "that", "this", "was", "be", "has", "had", "will",
                     "can", "if", "my", "me", "i", "you", "we", "they",
                     "not", "no", "with", "from", "by", "at", "on", "its",
                     "wht", "gud", "que", "es", "mas", "el"}

        raw_words = [w.lower() for w in re.findall(r'[a-zA-Z]+', user_prompt)
                     if len(w) > 2 and w.lower() not in stopwords]

        known_words = list(self.lexicon.word_to_uuid.keys())

        # Pre-LLM scan
        pre_seeds = set()
        for w in raw_words:
            uid = self._resolve_word(w, known_words)
            if uid:
                pre_seeds.add(uid)

        # LLM-extracted keywords
        llm_seeds = set()
        if extracted.get("status") != "CONVERSATION":
            words = [w.strip().lower() for w in extracted.get("keywords", [])]
            for w in words:
                uid = self._resolve_word(w, known_words)
                if uid:
                    llm_seeds.add(uid)

        return list(pre_seeds | llm_seeds), raw_words

    def _weave_evidence(self, seeds: list, results: list,
                        user_prompt: str) -> str:
        """Run the Algorithmic Weaver on graph results."""
        from .weaver import AlgorithmicWeaver
        weaver = AlgorithmicWeaver()
        return weaver.weave(
            self.kernel,
            [self.lexicon.get_word(s) for s in seeds],
            results,
            self.lexicon,
            seeds,
            user_prompt
        )

    def chat(self, user_prompt: str) -> str:
        """
        The full cognitive pipeline.

        System 1 (fast): Math intercept + direct graph query
        System 2 (slow): Shadow simulation + contradiction detection
        Active Inference: Autonomous foraging if entropy too high
        """
        # ====================================================
        # 0. MATH INTERCEPT (fires before everything)
        # ====================================================
        if self.math.is_math_query(user_prompt):
            result = self.math.solve(user_prompt)
            if result.get("status") == "success":
                return (f"**{result['operation']}**\n\n"
                        f"Input: `{result['equation']}`\n\n"
                        f"Result: **{result['result']}**")

        # Track whether forager was already used this query (prevent loops)
        self._forager_attempted = False

        # ====================================================
        # 1. THE EAR (JSON keyword extraction)
        # ====================================================
        extracted = self._extract_keywords(user_prompt)

        if extracted.get("status") == "CONVERSATION":
            # Quick pre-scan check before dismissing as conversation
            stopwords = {"what", "where", "when", "who", "why", "how",
                         "is", "the", "a", "an", "does", "it", "are", "do"}
            raw = [w.lower() for w in re.findall(r'[a-zA-Z]+', user_prompt)
                   if len(w) > 2 and w.lower() not in stopwords]
            known = list(self.lexicon.word_to_uuid.keys())
            has_seeds = any(self._resolve_word(w, known) for w in raw)
            if not has_seeds:
                return self._talk_normally(
                    user_prompt,
                    "I am a factual knowledge system. I don't have access "
                    "to personal or conversational context.")

        # ====================================================
        # 2. SEED RESOLUTION (6-layer cascade)
        # ====================================================
        seeds, raw_words = self._resolve_seeds(user_prompt, extracted)

        # ====================================================
        # 3. SYSTEM 2: METACOGNITIVE SHADOW SIMULATION
        # ====================================================
        # Create multiple interpretation branches
        # Branch 1: All resolved seeds (full query)
        # Branch 2: First seed only (broad/fallback)
        # Branch 3: LLM keywords only (if different from pre-scan)
        branches = [seeds]
        if len(seeds) > 1:
            branches.append(seeds[:1])  # Broad fallback
            branches.append(seeds[1:])  # Alternate focus

        thought = self.shadow.think_before_speaking(branches)

        # ====================================================
        # 4. ACTIVE INFERENCE (Curiosity Trigger)
        # ====================================================
        if (thought is None or self.shadow.SYSTEM_ENTROPY > self.ENTROPY_THRESHOLD):
            entropy = self.shadow.SYSTEM_ENTROPY

            if self.forager and raw_words:
                print(f"\n[ACTIVE INFERENCE] System Entropy = {entropy:.1f} "
                      f"(threshold: {self.ENTROPY_THRESHOLD})")
                print("[ACTIVE INFERENCE] Insufficient knowledge. "
                      "Deploying autonomous forager...")

                # Formulate a search query from the raw words
                search_query = " ".join(raw_words[:4])
                self._forager_attempted = True
                new_nodes = self.forager.forage_query(search_query)

                if new_nodes > 0:
                    print(f"[ACTIVE INFERENCE] Acquired +{new_nodes} concepts. "
                          f"Re-thinking...")

                    # Re-resolve seeds after learning
                    seeds, raw_words = self._resolve_seeds(
                        user_prompt, extracted)
                    branches = [seeds]
                    if len(seeds) > 1:
                        branches.append(seeds[:1])

                    # Re-run System 2 with new knowledge
                    thought = self.shadow.think_before_speaking(branches)

                    if thought:
                        print(f"[ACTIVE INFERENCE] Uncertainty resolved! "
                              f"New entropy = {self.shadow.SYSTEM_ENTROPY:.1f}")
                    else:
                        print("[ACTIVE INFERENCE] Foraging did not resolve "
                              "the uncertainty.")
                else:
                    print("[ACTIVE INFERENCE] Forager returned no new data.")
            else:
                if not self.forager:
                    print(f"\n[ACTIVE INFERENCE] Entropy = {entropy:.1f} "
                          f"but Forager is offline.")

        # ====================================================
        # 5. WEAVER + MOUTH (Evidence scoring + synthesis)
        # ====================================================
        if thought and thought["results"]:
            best_seeds = thought["seeds"]
            best_results = thought["results"]
            evidence_text = self._weave_evidence(
                best_seeds, best_results, user_prompt)

            # POST-HOC ENTROPY CHECK: The graph activated, but does
            # the Weaver's evidence actually match the query intent?
            # Even if we have evidence, if the key query words are
            # completely absent from it, there's a semantic gap.
            evidence_lower = evidence_text.lower()
            no_evidence = "no relevant context" in evidence_lower

            # Semantic gap detection: check if ANY non-entity query
            # words appear in the evidence. If none do, the evidence
            # is about the right entity but wrong attribute.
            if not no_evidence and raw_words:
                # Get entity words (things that resolved to seeds)
                entity_words = set()
                known = list(self.lexicon.word_to_uuid.keys())
                for w in raw_words:
                    if self._resolve_word(w, known):
                        entity_words.add(w)
                # Attribute words = query words that aren't entities
                attribute_words = [w for w in raw_words
                                   if w not in entity_words and len(w) > 3]
                if attribute_words:
                    attr_in_evidence = sum(
                        1 for w in attribute_words if w in evidence_lower)
                    if attr_in_evidence == 0:
                        # None of the attribute words appear in evidence
                        # = semantic gap (knows entity, not the fact)
                        no_evidence = True
                        print(f"\n[SEMANTIC GAP] Query attributes "
                              f"{attribute_words} not found in evidence.")

            if no_evidence and self.forager and raw_words and not self._forager_attempted:
                print(f"\n[POST-HOC INFERENCE] Graph activated but Weaver "
                      f"found no matching evidence.")
                print(f"[POST-HOC INFERENCE] Semantic gap detected. "
                      f"Deploying Forager...")

                search_query = " ".join(raw_words[:4])
                new_nodes = self.forager.forage_query(search_query)

                if new_nodes > 0:
                    print(f"[POST-HOC INFERENCE] +{new_nodes} concepts acquired. "
                          f"Re-weaving...")

                    # Re-resolve and re-query with expanded knowledge
                    seeds, raw_words = self._resolve_seeds(
                        user_prompt, extracted)
                    if seeds:
                        new_results = self.kernel.query(seeds, top_k=10)
                        if new_results:
                            evidence_text = self._weave_evidence(
                                seeds, new_results, user_prompt)

            # Record query for Attention Controller (anticipation)
            self.attention.record_query(
                best_seeds, self.kernel.current_tick)

            # Annotate with confidence metadata
            if self.shadow.SYSTEM_ENTROPY > 10.0:
                evidence_text += (
                    f"\n[Note: System confidence is moderate. "
                    f"Entropy={self.shadow.SYSTEM_ENTROPY:.1f}]")

            return self._talk_normally(user_prompt, evidence_text)
        else:
            # Complete knowledge gap — even after foraging
            if self.forager:
                return self._talk_normally(
                    user_prompt,
                    "I attempted to resolve my uncertainty via autonomous "
                    "foraging, but could not find logically consistent data "
                    "for this query.")
            else:
                return self._talk_normally(
                    user_prompt,
                    "No relevant context found in database.")

    def _talk_normally(self, prompt, evidence):
        synthesis_system = f"""
        You are a factual answer engine. Your ONLY job is to answer the user's question from the evidence below.
        Rules:
        1. If evidence is provided, you MUST use it to answer. Synthesize multiple facts if needed.
        2. The user may use synonyms, slang, or typos — match their INTENT to the evidence, not exact words.
           Example: "metropolis" means "city", "produces" means the same as "produce".
        3. If the evidence contains the answer even indirectly, extract and state it clearly.
        4. ONLY say "I don't have the data" if the evidence literally says "No relevant context found".
        5. Keep answers concise — 1-2 sentences.
        6. If a confidence note is present, you may mention uncertainty but still answer.

        DATABASE EVIDENCE:
        {evidence}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": synthesis_system},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content
