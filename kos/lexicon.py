"""
KOS V2.0 — KASM Lexicon (Semantic DNS + Phonetic Hashing + Synonym Net).

Maps human words to stable UUIDs via WordNet synsets.
Phonetic index (Metaphone) enables typo-tolerant lookup.
Synonym net maps all lemma names to the same canonical UUID.
"""
import hashlib
import jellyfish
import nltk
try:
    from nltk.corpus import wordnet as wn
except LookupError:
    nltk.download('wordnet', quiet=True)
    from nltk.corpus import wordnet as wn


class KASMLexicon:
    def __init__(self):
        self.word_to_uuid = {}
        self.uuid_to_word = {}
        # Phonetic Hash -> set of UUIDs
        self.sound_to_uuids = {}
        self.soundex_to_uuids = {}  # Soundex index (broader phonetic net)

    # Agent Fix 1: Domain words WordNet misclassifies
    _DOMAIN_PROTECTED = {
        "entanglement", "qubit", "qubits", "perovskite", "perovskites",
        "photovoltaic", "photovoltaics", "backpropagation", "nanotube",
        "nanotubes", "graphene", "blockchain", "cryptocurrency",
        "crispr", "mitochondria", "mitochondrion", "ribosome", "ribosomes",
        "genome", "genomic", "proteomics", "epigenetic",
        "neurotransmitter", "neurotransmitters", "serotonin", "dopamine",
        "apixaban", "warfarin", "thrombosis",
    }
    # Agent Fix 2: Plural normalization for non-English-origin words
    _PLURAL_MAP = {
        "qubits": "qubit", "perovskites": "perovskite",
        "photovoltaics": "photovoltaic", "nanotubes": "nanotube",
        "ribosomes": "ribosome", "neurotransmitters": "neurotransmitter",
    }

    def get_or_create_id(self, word: str) -> str:
        w_lower = word.lower()

        # Agent Fix 2: normalize plurals
        w_lower = self._PLURAL_MAP.get(w_lower, w_lower)

        if w_lower in self.word_to_uuid:
            return self.word_to_uuid[w_lower]

        # Agent Fix 1: skip WordNet for domain-protected words
        synsets = None
        if w_lower in self._DOMAIN_PROTECTED:
            uuid = f"KASM_{hashlib.sha256(w_lower.encode()).hexdigest()[:8]}"
        else:
            synsets = wn.synsets(w_lower, pos=wn.NOUN)
            if synsets:
                uuid = synsets[0].name()
            else:
                uuid = f"KASM_{hashlib.sha256(w_lower.encode()).hexdigest()[:8]}"

        self.word_to_uuid[w_lower] = uuid
        if uuid not in self.uuid_to_word:
            self.uuid_to_word[uuid] = w_lower

        # 1. Phonetic Hashing (Spellchecker — dual index)
        sound_hash = jellyfish.metaphone(w_lower)
        self._add_to_index(sound_hash, uuid)
        sdx = jellyfish.soundex(w_lower)
        self._add_to_soundex(sdx, uuid)

        # 2. Universal Synonym Net (Map synonyms to the SAME UUID)
        if synsets:
            for lemma in synsets[0].lemma_names():
                syn_lower = lemma.lower().replace("_", "")
                syn_hash = jellyfish.metaphone(syn_lower)
                self._add_to_index(syn_hash, uuid)
                syn_sdx = jellyfish.soundex(syn_lower)
                self._add_to_soundex(syn_sdx, uuid)

        return uuid

    def _add_to_index(self, sound_hash: str, uuid: str):
        if sound_hash not in self.sound_to_uuids:
            self.sound_to_uuids[sound_hash] = set()
        self.sound_to_uuids[sound_hash].add(uuid)

    def _add_to_soundex(self, sdx: str, uuid: str):
        if sdx not in self.soundex_to_uuids:
            self.soundex_to_uuids[sdx] = set()
        self.soundex_to_uuids[sdx].add(uuid)

    def resolve_hypernym(self, word: str, graph_nodes: dict) -> str:
        """
        WordNet Taxonomy Resolver (Is-A Relationships).

        Two-phase search:
        Phase 1 (Climb UP): If query word is more specific than graph.
          e.g., user says "medication", graph has "drug" (parent).
        Phase 2 (Descend DOWN): If query word is more abstract than graph.
          e.g., user says "entities", graph has "company" (child).
          WordNet knows company IS-A entity.

        Returns the first graph-matching UUID, or None.
        """
        w_lower = word.lower()
        synsets = wn.synsets(w_lower, pos=wn.NOUN)
        if not synsets:
            return None

        visited = set()

        # Phase 1: Walk UP hypernyms (query is specific, graph is general)
        queue = list(synsets[:3])
        for _ in range(5):
            next_queue = []
            for ss in queue:
                if ss.name() in visited:
                    continue
                visited.add(ss.name())

                if ss.name() in graph_nodes:
                    return ss.name()

                for lemma in ss.lemma_names():
                    lem_lower = lemma.lower()
                    if lem_lower in self.word_to_uuid:
                        uid = self.word_to_uuid[lem_lower]
                        if uid in graph_nodes:
                            return uid

                next_queue.extend(ss.hypernyms())

            queue = next_queue
            if not queue:
                break

        # Phase 2: Walk DOWN hyponyms (query is abstract, graph is specific)
        # e.g., "entities" → entity.n.01 → ... → company.n.01 (in graph)
        # Collect ALL matches within depth 6, then return the one with
        # the most graph connections (most central/relevant node).
        visited_down = set()
        queue = list(synsets[:3])
        candidates = []
        for _ in range(6):
            next_queue = []
            for ss in queue:
                if ss.name() in visited_down:
                    continue
                visited_down.add(ss.name())

                for hypo in ss.hyponyms():
                    if hypo.name() in graph_nodes:
                        candidates.append(hypo.name())

                    for lemma in hypo.lemma_names():
                        lem_lower = lemma.lower()
                        if lem_lower in self.word_to_uuid:
                            uid = self.word_to_uuid[lem_lower]
                            if uid in graph_nodes:
                                candidates.append(uid)

                    next_queue.append(hypo)

            queue = next_queue
            if not queue:
                break

        if candidates:
            # Pick the candidate with the most connections (most relevant)
            return max(candidates,
                       key=lambda c: len(graph_nodes[c].connections)
                       if c in graph_nodes else 0)

        return None

    def get_word(self, uuid: str) -> str:
        import re
        # 1. Clean WordNet suffix FIRST (toronto.n.01 -> Toronto)
        if re.search(r'\.[a-z]\.\d+$', uuid.lower()):
            return uuid.split('.')[0].replace("_", " ").title()

        # 2. Live memory lookup
        if uuid in self.uuid_to_word:
            return self.uuid_to_word[uuid].replace("_", " ").title()

        return uuid.replace("_", " ").title()
