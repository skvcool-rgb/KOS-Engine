"""
KOS V2.0 — Shell (LLM Ear + KOS Brain + LLM Mouth).

The LLM acts as the Ear to catch conversation and extract intent,
and the Mouth to speak naturally. Your KOS Kernel is the Brain.
"""
import os
from openai import OpenAI


class KOSShell:
    def __init__(self, kernel, lexicon):
        self.kernel = kernel
        self.lexicon = lexicon
        # Initialize the LLM (Requires OPENAI_API_KEY as environment variable)
        # OR point to a local model:
        # client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.client = OpenAI()

        # Math Coprocessor (0% hallucination — SymPy exact computation)
        from .drivers.math import MathDriver
        self.math = MathDriver()

        # Layer 5: Semantic Vector Fallback (node labels only, NOT documents)
        from sentence_transformers import SentenceTransformer, util
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.node_embeddings = None
        self.embedded_uuids = []
        self._st_util = util

    def _ensure_embeddings(self):
        """Lazily build/rebuild cached embeddings for all graph node labels."""
        current_uuids = list(self.kernel.nodes.keys())
        if self.node_embeddings is None or len(current_uuids) != len(self.embedded_uuids):
            self.embedded_uuids = current_uuids
            plain_words = [self.lexicon.get_word(uid) for uid in current_uuids]
            self.node_embeddings = self.embedder.encode(
                plain_words, convert_to_tensor=True)

    def _resolve_word(self, w, known_words):
        """6-layer word→UUID resolution cascade."""
        import difflib
        import jellyfish

        # Layer 1: Exact lexicon match
        if w in self.lexicon.word_to_uuid:
            return self.lexicon.word_to_uuid[w]

        # Layer 2: Difflib fuzzy match (character similarity >= 0.6)
        matches = difflib.get_close_matches(
            w, known_words, n=1, cutoff=0.6)
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

        # Layer 5: Semantic vector fallback (all-MiniLM-L6-v2)
        # Embeds node LABELS only (not documents) — zero hallucination risk
        if self.kernel.nodes:
            self._ensure_embeddings()
            w_emb = self.embedder.encode(w, convert_to_tensor=True)
            hits = self._st_util.cos_sim(w_emb, self.node_embeddings)[0]
            best_score = hits.max().item()
            best_idx = hits.argmax().item()
            if best_score > 0.50:
                return self.embedded_uuids[best_idx]

        return None

    def chat(self, user_prompt: str) -> str:
        # ====================================================
        # 0. MATH INTERCEPT (fires before the LLM Ear)
        # ====================================================
        if self.math.is_math_query(user_prompt):
            result = self.math.solve(user_prompt)
            if result.get("status") == "success":
                return (f"**{result['operation']}**\n\n"
                        f"Input: `{result['equation']}`\n\n"
                        f"Result: **{result['result']}**")
            # If math parse failed, fall through to the graph

        # ====================================================
        # 0.5 PRE-LLM RAW WORD SCAN
        # Catches typos and taxonomy matches BEFORE the LLM
        # normalizes them away. Any seeds found here supplement
        # the LLM-extracted seeds.
        # ====================================================
        import re
        import itertools

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
        pre_seeds = set()
        for w in raw_words:
            uid = self._resolve_word(w, known_words)
            if uid:
                pre_seeds.add(uid)

        # ====================================================
        # 1. THE EAR (Strict JSON extraction — deterministic)
        # ====================================================
        extraction_system = """
        You are a deterministic database router. Read the user's prompt.
        If it is casual conversation ('hello', 'how are you'), output: {"status": "CONVERSATION"}
        Otherwise, extract the 2 to 4 core entities/actions needed for a search query.
        Output ONLY valid JSON in this exact format:
        {"status": "EXECUTE", "keywords": ["keyword1", "keyword2"]}
        """

        import json
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
            extracted = json.loads(response.choices[0].message.content.strip())
        except Exception:
            extracted = {"status": "EXECUTE", "keywords": []}

        # Intercept casual conversation — BUT only if pre-scan found nothing
        if extracted.get("status") == "CONVERSATION" and not pre_seeds:
            return self._talk_normally(
                user_prompt,
                "I am a factual knowledge system. I don't have access "
                "to personal or conversational context."
            )

        # ====================================================
        # 2. THE BRAIN (Your KOS runs the exact math)
        # ====================================================
        # Resolve LLM-extracted keywords through the 6-layer cascade
        llm_seeds = set()
        if extracted.get("status") != "CONVERSATION":
            words = [w.strip().lower() for w in extracted.get("keywords", [])]
            for w in words:
                uid = self._resolve_word(w, known_words)
                if uid:
                    llm_seeds.add(uid)

        # Merge pre-LLM and LLM seeds (union — no duplicates)
        seeds = list(pre_seeds | llm_seeds)

        evidence_text = "No relevant context found in database."

        if seeds:
            results = self.kernel.query(seeds, top_k=10)

            # Collect ALL provenance evidence (inter-seed + traversal)
            all_evidence = set()

            # Inter-seed provenance (direct pairs)
            for s1, s2 in itertools.combinations(seeds, 2):
                pair = tuple(sorted([s1, s2]))
                all_evidence.update(
                    getattr(self.kernel, 'provenance', {}).get(
                        pair, set()))

            # Traversal provenance (seed→result pairs)
            if results:
                for ans_uuid, _ in results:
                    for s_uuid in seeds:
                        all_evidence.update(
                            getattr(self.kernel, 'provenance', {}).get(
                                tuple(sorted([s_uuid, ans_uuid])), set()))

            if results:
                # Weaver returns top-2 raw sentences scored by intent
                from .weaver import AlgorithmicWeaver
                weaver = AlgorithmicWeaver()
                evidence_text = weaver.weave(
                    self.kernel,
                    [self.lexicon.get_word(s) for s in seeds],
                    results,
                    self.lexicon,
                    seeds,
                    user_prompt
                )

        # ====================================================
        # 3. THE MOUTH (LLM formats the Weaver's scored evidence)
        # ====================================================
        return self._talk_normally(user_prompt, evidence_text)

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
