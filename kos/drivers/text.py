"""
KOS V5.1 — Text Driver (SVO + Negation + Adjectives + Clause Splitting).

Ingests raw text into the kernel via the lexicon.
Pass 0: Clause-level splitting (comma clauses, relative clauses)
Pass 1: Pronoun resolution (split antecedent support)
Pass 2: Extraction with negation scope detection
Pass 3: Adjective extraction as property nodes
Pass 4: Sequential Triplet Slider for SVO dependency parsing

Week 1 Fixes:
    #4  Negation handling — "does NOT cause" → inhibitory edge
    #1  Adjective extraction — "cheap", "humid" become property nodes
    #10 Clause-level provenance — split on commas/relative clauses
"""
import re
import nltk
# Ensure NLTK data is available
for _pkg in ['punkt_tab', 'averaged_perceptron_tagger_eng', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{_pkg}' if 'punkt' in _pkg
                       else f'taggers/{_pkg}' if 'tagger' in _pkg
                       else f'corpora/{_pkg}')
    except LookupError:
        nltk.download(_pkg, quiet=True)
from nltk.stem import WordNetLemmatizer


# ── Protected Lemmatizer ─────────────────────────────────────

_PROTECTED_WORDS = {"kos", "os", "api", "gpu", "cpu", "llm", "ast"}
_LEMMATIZER = WordNetLemmatizer()


def _safe_lemmatize(word: str, pos: str = 'n') -> str:
    """Prevents the lemmatizer from destroying acronyms."""
    w = word.lower()
    if w in _PROTECTED_WORDS or len(w) <= 3:
        return w
    return _LEMMATIZER.lemmatize(w, pos)


# ── Negation Words ───────────────────────────────────────────

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "without",
    "hardly", "barely", "scarcely", "rarely", "seldom",
    "n't", "dont", "doesn", "didn", "isn", "aren",
    "wasn", "weren", "couldn", "wouldn", "shouldn",
    "cannot", "cant",
}

# ── Coreference Resolver ─────────────────────────────────────

class KOSResolver:
    """
    Pronoun resolver with recency-based working memory.

    Fix #7 upgrade:
    - Tracks sentence subject (first noun) for "it" resolution
    - Handles "the city" → resolves to last city-type entity
    - Handles "the former" / "the latter" in comparisons
    - Better split-antecedent for "They" across sentences
    """

    # Definite article phrases that refer back
    DEFINITE_REFS = {
        "the city", "the country", "the company", "the team",
        "the university", "the drug", "the material", "the system",
        "the process", "the method", "the tower", "the building",
    }

    def __init__(self):
        self.recent_singular = []
        self.recent_plural = []
        self.recent_proper = []
        self.sentence_topics = []
        self._current_sentence_nouns = []
        self._sentence_subject = None     # First noun in CURRENT sentence/clause
        self._last_sentence_subject = None # First noun from PREVIOUS sentence (persists!)
        self.PB = {
            "they", "them", "their", "theirs", "these",
            "he", "him", "his", "she", "her", "hers", "who",
            "it", "its", "itself", "this", "that", "which",
        }

    def reset(self):
        self.recent_singular.clear()
        self.recent_plural.clear()
        self.recent_proper.clear()
        self.sentence_topics.clear()
        self._current_sentence_nouns.clear()
        self._sentence_subject = None
        self._last_sentence_subject = None

    def mark_topic(self, w):
        if w not in self._current_sentence_nouns:
            self._current_sentence_nouns.append(w)
        # Fix #7: Track sentence subject (first noun = default "it" referent)
        if self._sentence_subject is None:
            self._sentence_subject = w

    def end_sentence(self):
        for noun in self._current_sentence_nouns:
            self.sentence_topics.append(noun)
        self._current_sentence_nouns.clear()
        # Carry subject forward: if this sentence had a subject,
        # save it as the "last sentence subject" for the next sentence.
        # This is what makes "Toronto is a city. It has..." work —
        # "It" in the next sentence resolves to Toronto.
        if self._sentence_subject:
            self._last_sentence_subject = self._sentence_subject
        self._sentence_subject = None

    def update_memory(self, w, pos):
        if w in self.PB:
            return
        self.mark_topic(w)
        if pos.startswith('NNP'):
            self.recent_proper = (self.recent_proper + [w])[-5:]
        elif pos in ['NNS', 'NNPS']:
            self.recent_plural = (self.recent_plural + [w])[-5:]
        elif pos == 'NN':
            self.recent_singular = (self.recent_singular + [w])[-5:]

    def resolve(self, w):
        if w in {"they", "them", "their", "theirs", "these"}:
            if self.recent_plural:
                return [self.recent_plural[-1]]
            unq = []
            for t in reversed(self.sentence_topics):
                if t not in unq:
                    unq.append(t)
                if len(unq) == 2:
                    return [unq[1], "and", unq[0]]
            return [w]
        if w in {"he", "him", "his", "she", "her", "hers", "who"}:
            if self.recent_proper:
                return [self.recent_proper[-1]]
            if self.recent_singular:
                return [self.recent_singular[-1]]
            return [w]
        # Fix #7: 4-level cascade for "it" resolution
        # 1. Current sentence subject (within same sentence/clause)
        # 2. Last sentence subject (carried forward across sentences)
        # 3. Most recent singular noun (NN tag)
        # 4. Most recent proper noun (NNP tag — catches "Toronto")
        if w in {"it", "its", "itself", "this", "that", "which"}:
            if self._sentence_subject:
                return [self._sentence_subject]
            if self._last_sentence_subject:
                return [self._last_sentence_subject]
            if self.recent_singular:
                return [self.recent_singular[-1]]
            if self.recent_proper:
                return [self.recent_proper[-1]]
            return [w]
        return [w]


# ── Text Driver ──────────────────────────────────────────────

class TextDriver:
    """Ingests natural language text into the KOS graph via the lexicon."""

    _IDENTIFIER_RE = re.compile(r'^[a-z]+_\d+$', re.IGNORECASE)

    # Clause splitting patterns
    _CLAUSE_SPLIT = re.compile(
        r',\s*(?:which|who|that|where|when|while|although|though|'
        r'founded|established|located|known|built|named|making|'
        r'having|being|including)\s+',
        re.IGNORECASE
    )

    def __init__(self, kernel, lexicon):
        self.kernel = kernel
        self.lexicon = lexicon
        self.resolver = KOSResolver()
        self.lemmatizer = _LEMMATIZER
        # Set lexicon ref on kernel for contradiction word lookup
        if hasattr(kernel, 'set_lexicon'):
            kernel.set_lexicon(lexicon)
        self.JUNK = {
            "def", "import", "from", "return", "print", "len",
            "true", "false", "none", "thing", "way",
        }
        self.INHIBITORY = {
            "unlike", "instead", "replace", "outperform", "contrast",
            "oppose", "compete", "prevent", "block", "inhibit",
            "reduce", "decrease", "reject", "eliminate", "suppress",
        }
        self.EXCITATORY = {
            "is", "use", "utilize", "require", "produce", "enable",
            "improve", "capture", "develop", "enhance", "increase",
            "support", "create", "generate", "contain", "provide",
            "achieve", "build", "form", "yield",
        }

    def _fix_pos_tags(self, word: str, tag: str) -> str:
        """Fix POS tags for synthetic identifiers and domain terms."""
        from nltk.corpus import wordnet as wn

        if self._IDENTIFIER_RE.match(word) and not tag.startswith('NN'):
            return 'NN'
        if not tag.startswith('NN') and not tag.startswith('VB'):
            if len(word) > 3 and not wn.synsets(word):
                return 'NN'
        if tag.startswith('NN') and len(word) > 3:
            verb_synsets = wn.synsets(word, pos=wn.VERB)
            noun_synsets = wn.synsets(word, pos=wn.NOUN)
            if verb_synsets and not noun_synsets:
                return 'VBZ'
        return tag

    # ── FIX #10: Clause-Level Provenance Splitting ───────────

    def _split_clauses(self, sentence: str) -> list:
        """
        Split complex sentences into clauses for finer provenance.

        "Toronto, founded in 1834 by Simcoe, is located in Ontario"
        becomes:
            ["Toronto founded in 1834 by Simcoe",
             "Toronto is located in Ontario"]

        The subject is carried forward to each clause.
        """
        # First, try to extract the main subject (first noun phrase)
        words = sentence.split()
        if len(words) < 5:
            return [sentence]  # Too short to split

        # Split on clause-boundary patterns
        parts = self._CLAUSE_SPLIT.split(sentence)

        if len(parts) <= 1:
            return [sentence]  # No clause boundaries found

        # Extract subject from the first part
        first_part = parts[0].strip().rstrip(',').strip()
        # Subject is typically everything before the first comma
        subject_match = re.match(r'^([\w\s]+?)(?:,|\s+is\s+|\s+was\s+|\s+has\s+)',
                                  first_part)
        subject = subject_match.group(1).strip() if subject_match else ""

        clauses = [first_part]
        for part in parts[1:]:
            part = part.strip().rstrip('.').strip()
            if part and len(part) > 10:
                # Prepend subject if the clause doesn't start with one
                first_word = part.split()[0].lower() if part.split() else ""
                if first_word in {'is', 'was', 'has', 'had', 'are', 'were',
                                   'in', 'on', 'at', 'by', 'for', 'with'}:
                    part = f"{subject} {part}"
                clauses.append(part)

        return clauses if len(clauses) > 1 else [sentence]

    # ── FIX #4: Negation Scope Detection ─────────────────────

    def _detect_negation(self, tokens: list, trigger_idx: int) -> bool:
        """
        Check if a trigger verb is under negation scope.

        Scans 3 tokens to the left of the trigger for negation words.
        "does NOT cause" → negation detected at "cause"
        "prevents" → no negation (the verb itself is inhibitory)

        Returns True if the trigger is negated.
        """
        start = max(0, trigger_idx - 3)
        for i in range(start, trigger_idx):
            token = tokens[i].lower() if isinstance(tokens[i], str) else ""
            # Handle "n't" attached to verbs (doesn't, isn't, etc.)
            if token in NEGATION_WORDS or token.endswith("n't"):
                return True
        return False

    # ── FIX #1: Adjective Extraction ─────────────────────────

    def _extract_adjectives(self, tagged_tokens: list, uids: list,
                             sentence: str):
        """
        Extract adjectives as property nodes and wire them to
        the nearest noun.

        "Perovskite is cheap" → [Perovskite] --(property:0.7)--> [Cheap]
        "Toronto has humid continental climate" →
            [Climate] --(property:0.7)--> [Humid]
            [Climate] --(property:0.7)--> [Continental]
        """
        adjective_count = 0

        for i, (word, tag) in enumerate(tagged_tokens):
            if not tag.startswith('JJ'):
                continue

            w_lower = word.lower()
            if len(w_lower) <= 2 or w_lower in self.JUNK:
                continue

            # Find the nearest noun (look right first, then left)
            nearest_noun_uid = None

            # Look right for the noun this adjective modifies
            for j in range(i + 1, min(i + 4, len(tagged_tokens))):
                if tagged_tokens[j][1].startswith('NN'):
                    lem = _safe_lemmatize(tagged_tokens[j][0].lower(), 'n')
                    if lem in self.lexicon.word_to_uuid:
                        nearest_noun_uid = self.lexicon.word_to_uuid[lem]
                        break

            # If no noun to the right, look left
            if not nearest_noun_uid:
                for j in range(i - 1, max(i - 4, -1), -1):
                    if tagged_tokens[j][1].startswith('NN'):
                        lem = _safe_lemmatize(tagged_tokens[j][0].lower(), 'n')
                        if lem in self.lexicon.word_to_uuid:
                            nearest_noun_uid = self.lexicon.word_to_uuid[lem]
                            break

            # If still no noun, use the first noun in the sentence
            if not nearest_noun_uid and uids:
                nearest_noun_uid = uids[0]

            if nearest_noun_uid:
                adj_uid = self.lexicon.get_or_create_id(w_lower)
                self.kernel.add_connection(
                    nearest_noun_uid, adj_uid, 0.7,
                    sentence
                )
                adjective_count += 1

        return adjective_count

    # ── FIX #8: Numeric Property Extraction ────────────────

    # Patterns: "2.7 million", "1834", "$50 billion", "30%", "9 °C"
    _NUM_PATTERNS = [
        # "2.7 million people" → value=2700000
        (re.compile(r'(\d+[\.,]?\d*)\s*(million|billion|trillion|thousand)',
                     re.IGNORECASE),
         lambda m: float(m.group(1).replace(',', '')) *
         {'million': 1e6, 'billion': 1e9, 'trillion': 1e12,
          'thousand': 1e3}[m.group(2).lower()]),

        # "1834" (standalone year)
        (re.compile(r'\b(1[5-9]\d{2}|20[0-2]\d)\b'),
         lambda m: int(m.group(1))),

        # "30%" or "30 percent"
        (re.compile(r'(\d+[\.,]?\d*)\s*(%|percent)', re.IGNORECASE),
         lambda m: float(m.group(1).replace(',', ''))),

        # "$50" or "50 dollars"
        (re.compile(r'\$\s*(\d+[\.,]?\d*)', re.IGNORECASE),
         lambda m: float(m.group(1).replace(',', ''))),

        # "9 °C" or "48 °F"
        (re.compile(r'(\d+[\.,]?\d*)\s*°\s*([CF])', re.IGNORECASE),
         lambda m: float(m.group(1).replace(',', ''))),

        # Generic large numbers: "2,700,000"
        (re.compile(r'\b(\d{1,3}(?:,\d{3})+)\b'),
         lambda m: float(m.group(1).replace(',', ''))),
    ]

    def _extract_numeric_properties(self, clause: str, uids: list):
        """
        Extract numeric values from text and attach as typed properties
        to the nearest noun node.

        "Toronto has a population of 2.7 million" →
            nodes['toronto'].properties['population'] = 2700000
        """
        if not uids:
            return

        for pattern, extractor in self._NUM_PATTERNS:
            for match in pattern.finditer(clause):
                try:
                    value = extractor(match)
                except (ValueError, KeyError):
                    continue

                # Find context word near the number (±5 words)
                start = max(0, match.start() - 50)
                end = min(len(clause), match.end() + 50)
                context = clause[start:end].lower()

                # Determine property name from context
                prop_name = "value"
                for keyword in ["population", "year", "temperature",
                                 "cost", "price", "area", "height",
                                 "distance", "speed", "weight",
                                 "percentage", "rate", "age",
                                 "founded", "incorporated", "established"]:
                    if keyword in context:
                        prop_name = keyword
                        break

                # Attach to first uid (primary subject)
                node_id = uids[0]
                if node_id in self.kernel.nodes:
                    self.kernel.nodes[node_id].properties[prop_name] = value

    # ── MAIN INGEST ──────────────────────────────────────────

    def ingest(self, text: str) -> dict:
        self.resolver.reset()
        sentences = nltk.sent_tokenize(text)
        total_nouns = 0
        total_svo = 0
        total_adj = 0
        total_clauses = 0

        for sent in sentences:
            # ── FIX #10: Split into clauses ──────────────────
            clauses = self._split_clauses(sent)
            total_clauses += len(clauses)

            for clause in clauses:
                self._ingest_clause(clause, sent)

            # End sentence for coreference tracking
            self.resolver.end_sentence()

        return {
            'sentences': len(sentences),
            'clauses': total_clauses,
            'concepts_found': total_nouns,
            'svo_edges': total_svo,
        }

    def _ingest_clause(self, clause: str, original_sentence: str):
        """Process a single clause (may be a sub-sentence)."""

        # Pass 0: Coreference resolution
        resolved_tokens = []
        raw_tokens = nltk.word_tokenize(clause)
        raw_tagged = nltk.pos_tag(raw_tokens)

        for w, p in raw_tagged:
            wl = w.lower()
            if wl in self.resolver.PB:
                resolved_tokens.extend(self.resolver.resolve(wl))
            elif p.startswith('NN'):
                self.resolver.update_memory(wl, p)
                resolved_tokens.append(wl)
            else:
                resolved_tokens.append(wl)

        # Pass 1: POS tag the resolved tokens
        r_tagged = nltk.pos_tag(resolved_tokens)

        sequence = []
        uids_only = []

        # ── FIX #4: Track negation scope ─────────────────────
        # Build a negation mask: True at positions under negation
        negation_active = False
        negation_mask = []
        for w, p in r_tagged:
            wl = w.lower()
            if wl in NEGATION_WORDS or wl.endswith("n't"):
                negation_active = True
                negation_mask.append(False)  # The negation word itself isn't negated
            elif negation_active and p.startswith('VB'):
                negation_mask.append(True)  # This verb IS negated
                negation_active = False  # Negation scope ends after the verb
            elif negation_active and wl in {',', '.', ';', 'and', 'but', 'or'}:
                negation_active = False  # Punctuation ends negation scope
                negation_mask.append(False)
            else:
                negation_mask.append(False)

        for idx, (w, p) in enumerate(r_tagged):
            p_safe = self._fix_pos_tags(w, p)
            w_lower = w.lower()

            # Identify Nodes (Nouns)
            if p_safe.startswith('NN'):
                lem = _safe_lemmatize(w_lower, 'n')
                if lem not in self.JUNK and len(lem) > 2:
                    uid = self.lexicon.get_or_create_id(lem)
                    sequence.append({"type": "node", "val": uid})
                    uids_only.append(uid)

            # Identify Triggers (Verbs / Polarity words)
            elif (p_safe.startswith('VB')
                  or w_lower in self.INHIBITORY | self.EXCITATORY):
                lem = _safe_lemmatize(w_lower, 'v')

                # Determine base polarity
                if lem in self.INHIBITORY or w_lower == "unlike":
                    polarity = "inhibitory"
                elif lem in self.EXCITATORY:
                    polarity = "excitatory"
                else:
                    polarity = "neutral"

                # FIX #4: Negation FLIPS polarity
                is_negated = (idx < len(negation_mask)
                              and negation_mask[idx])
                if is_negated:
                    if polarity == "excitatory":
                        polarity = "inhibitory"
                    elif polarity == "inhibitory":
                        polarity = "excitatory"

                sequence.append({
                    "type": "trigger",
                    "val": lem,
                    "polarity": polarity,
                    "negated": is_negated,
                })

                # ── VERB NODE CREATION ──────────────────────────
                # Create a node for meaningful verbs so queries like
                # "What produces ATP?" can resolve "produces" as a seed.
                # Skip trivial verbs (is, are, was, has, have, do, etc.)
                _TRIVIAL_VERBS = {
                    "be", "is", "are", "was", "were", "am", "been",
                    "have", "has", "had", "do", "does", "did",
                    "can", "could", "will", "would", "shall", "should",
                    "may", "might", "must", "get", "got", "let",
                    "make", "made", "use", "used",
                }
                if lem not in _TRIVIAL_VERBS and len(lem) > 2:
                    verb_uid = self.lexicon.get_or_create_id(lem)
                    self.kernel.add_node(verb_uid)
                    # Wire verb to all nouns in this clause
                    for noun_uid in uids_only:
                        self.kernel.add_connection(
                            verb_uid, noun_uid, 0.5, original_sentence)
                        self.kernel.add_connection(
                            noun_uid, verb_uid, 0.3, original_sentence)

                    # ── CROSS-SENTENCE VERB BRIDGING ──────────────
                    # If a verb like "learn" appears, connect it to
                    # ALL existing nodes that share a semantic domain.
                    # This creates bridges across sentences:
                    # "Neural networks learn" + "Backpropagation adjusts weights"
                    # → learn connects to backpropagation via shared "network" edges
                    # We do this by finding all nodes that share connections
                    # with any noun in this clause (2-hop bridge)
                    for noun_uid in uids_only:
                        if noun_uid in self.kernel.nodes:
                            for neighbor_uid in list(self.kernel.nodes[noun_uid].connections.keys()):
                                if neighbor_uid != verb_uid:
                                    # Weak bridge edge: verb → 2-hop neighbor
                                    self.kernel.add_connection(
                                        verb_uid, neighbor_uid, 0.2, original_sentence)

        # Wire Ambient Noise (Hebbian)
        for n1 in uids_only:
            self.kernel.add_node(n1)
            for n2 in uids_only:
                if n1 != n2:
                    self.kernel.add_connection(n1, n2, 0.4,
                                               original_sentence)

        # ── FIX #1: Extract adjectives as property nodes ─────
        self._extract_adjectives(r_tagged, uids_only, original_sentence)

        # ── FIX #8: Extract numeric properties ──────────────
        self._extract_numeric_properties(clause, uids_only)

        # Native Dependency Parsing: Sequential Triplet Slider
        for i, token in enumerate(sequence):
            if token["type"] == "trigger":

                # Look Left for Subject
                subj = None
                for j in range(i - 1, -1, -1):
                    if sequence[j]["type"] == "node":
                        subj = sequence[j]["val"]
                        break

                # Look Right for Object
                obj = None
                for j in range(i + 1, len(sequence)):
                    if sequence[j]["type"] == "node":
                        obj = sequence[j]["val"]
                        break

                # Rule 1: "Unlike" Sentence-Initial
                if (not subj and obj
                        and token["polarity"] == "inhibitory"):
                    right_nodes = [
                        t["val"] for t in sequence[i + 1:]
                        if t["type"] == "node"
                    ]
                    if len(right_nodes) >= 2:
                        legacy = right_nodes[0]
                        challenger = right_nodes[1]
                        self.kernel.add_connection(
                            challenger, legacy, -0.8, original_sentence)
                        self.kernel.add_connection(
                            legacy, challenger, 0.8,
                            f"Antidote/Alternative: {original_sentence}")

                # Rule 2: Standard SVO
                elif subj and obj:
                    if token["polarity"] == "inhibitory":
                        self.kernel.add_connection(
                            subj, obj, -0.8, original_sentence)
                        self.kernel.add_connection(
                            obj, subj, 0.8,
                            f"Antidote/Alternative: {original_sentence}")
                    elif token["polarity"] == "excitatory":
                        self.kernel.add_connection(
                            subj, obj, 0.9, original_sentence)
