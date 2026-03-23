"""
KOS V2.0 — Text Driver (Sequential Triplet Slider + Coreference).

Ingests raw text into the kernel via the lexicon.
Pass 0: Pronoun resolution (split antecedent support).
Pass 1: Extraction & Math Sequence Mapping — builds an ordered
        sequence of nodes (nouns) and triggers (verbs/polarity words),
        then uses a Sequential Triplet Slider for native dependency
        parsing: looks left for subject, right for object around each
        trigger. Handles sentence-initial "Unlike" edge case.
"""
import re
import nltk
# Ensure NLTK data is available (required on Streamlit Cloud)
for _pkg in ['punkt_tab', 'averaged_perceptron_tagger_eng', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{_pkg}' if 'punkt' in _pkg else f'taggers/{_pkg}' if 'tagger' in _pkg else f'corpora/{_pkg}')
    except LookupError:
        nltk.download(_pkg, quiet=True)
from nltk.stem import WordNetLemmatizer


# ── Protected Lemmatizer ─────────────────────────────────────

_PROTECTED_WORDS = {"kos", "os", "api", "gpu", "cpu", "llm", "ast"}
_LEMMATIZER = WordNetLemmatizer()


def _safe_lemmatize(word: str, pos: str = 'n') -> str:
    """Prevents the lemmatizer from destroying acronyms and domain terms."""
    w = word.lower()
    if w in _PROTECTED_WORDS or len(w) <= 3:
        return w
    return _LEMMATIZER.lemmatize(w, pos)


# ── Coreference Resolver ─────────────────────────────────────

class KOSResolver:
    """
    Lightweight pronoun resolver using recency-based working memory.
    Supports split antecedent resolution: "They" resolves to last
    two unique sentence topics when no plural antecedent exists.
    """

    def __init__(self):
        self.recent_singular = []
        self.recent_plural = []
        self.recent_proper = []
        self.sentence_topics = []
        self._current_sentence_nouns = []  # ALL nouns in current sentence
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

    def mark_topic(self, w):
        """Track every noun in the sentence for split-antecedent."""
        if w not in self._current_sentence_nouns:
            self._current_sentence_nouns.append(w)

    def end_sentence(self):
        """Push all unique nouns from this sentence into the topic list.
        This gives the split-antecedent resolver enough history to
        resolve 'They' → last 2 unique topics across sentences."""
        for noun in self._current_sentence_nouns:
            self.sentence_topics.append(noun)
        self._current_sentence_nouns.clear()

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
        # Plural pronouns — split antecedent
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

        # Person pronouns
        if w in {"he", "him", "his", "she", "her", "hers", "who"}:
            if self.recent_proper:
                return [self.recent_proper[-1]]
            if self.recent_singular:
                return [self.recent_singular[-1]]
            return [w]

        # Neutral pronouns
        if w in {"it", "its", "itself", "this", "that", "which"}:
            if self.recent_singular:
                return [self.recent_singular[-1]]
            return [w]

        return [w]


# ── Text Driver ──────────────────────────────────────────────

class TextDriver:
    """Ingests natural language text into the KOS graph via the lexicon."""

    _IDENTIFIER_RE = re.compile(r'^[a-z]+_\d+$', re.IGNORECASE)

    def __init__(self, kernel, lexicon):
        self.kernel = kernel
        self.lexicon = lexicon
        self.resolver = KOSResolver()
        self.lemmatizer = _LEMMATIZER
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
        """Fix POS tags for synthetic identifiers and unknown domain terms.

        Rule 1: Synthetic IDs (outcome_1) → NN.
        Rule 2: Unknown domain terms (apixaban) tagged as JJ/RB → NN.
                If no WordNet synsets exist for any POS, it's domain jargon.
        Rule 3: Known verbs mistagged as nouns (prevents→NNS) → VBZ.
                NLTK often mistaggs verbs after unknown words.
        """
        from nltk.corpus import wordnet as wn

        # Rule 1: Synthetic identifiers
        if self._IDENTIFIER_RE.match(word) and not tag.startswith('NN'):
            return 'NN'

        # Rule 2: Unknown domain term rescue
        if not tag.startswith('NN') and not tag.startswith('VB'):
            if len(word) > 3 and not wn.synsets(word):
                return 'NN'

        # Rule 3: Known verbs mistagged as nouns
        # e.g., "prevents" tagged NNS, but WordNet knows it's a verb
        if tag.startswith('NN') and len(word) > 3:
            verb_synsets = wn.synsets(word, pos=wn.VERB)
            noun_synsets = wn.synsets(word, pos=wn.NOUN)
            if verb_synsets and not noun_synsets:
                # It's ONLY a verb in WordNet → fix the tag
                return 'VBZ'

        return tag

    def ingest(self, text: str) -> dict:
        self.resolver.reset()
        sentences = nltk.sent_tokenize(text)
        total_nouns = 0
        total_svo = 0

        for sent in sentences:
            # Pass 0: Coreference resolution
            resolved_tokens = []
            for w, p in nltk.pos_tag(nltk.word_tokenize(sent)):
                wl = w.lower()
                if wl in self.resolver.PB:
                    resolved_tokens.extend(self.resolver.resolve(wl))
                elif p.startswith('NN'):
                    self.resolver.update_memory(wl, p)
                    resolved_tokens.append(wl)
                else:
                    resolved_tokens.append(wl)
            self.resolver.end_sentence()

            # Pass 1: Extraction & Math Sequence Mapping
            r_tagged = nltk.pos_tag(resolved_tokens)

            sequence = []    # Preserves the exact grammar order
            uids_only = []   # Used for ambient Hebbian noise

            for w, p in r_tagged:
                p_safe = self._fix_pos_tags(w, p)
                w_lower = w.lower()

                # Identify Nodes (Nouns)
                if p_safe.startswith('NN'):
                    lem = _safe_lemmatize(w_lower, 'n')
                    if lem not in self.JUNK and len(lem) > 2:
                        uid = self.lexicon.get_or_create_id(lem)
                        sequence.append({"type": "node", "val": uid})
                        uids_only.append(uid)

                # Identify Triggers (Verbs / Polarity Prepositions)
                elif (p_safe.startswith('VB')
                      or w_lower in self.INHIBITORY | self.EXCITATORY):
                    lem = _safe_lemmatize(w_lower, 'v')
                    if (lem in self.INHIBITORY
                            or w_lower == "unlike"):
                        polarity = "inhibitory"
                    elif lem in self.EXCITATORY:
                        polarity = "excitatory"
                    else:
                        polarity = "neutral"
                    sequence.append({
                        "type": "trigger",
                        "val": lem,
                        "polarity": polarity,
                    })

            total_nouns += len(uids_only)

            # Wire Ambient Noise (Hebbian)
            for n1 in uids_only:
                self.kernel.add_node(n1)
                for n2 in uids_only:
                    if n1 != n2:
                        self.kernel.add_connection(n1, n2, 0.4, sent)

            # Native Dependency Parsing: The Sequential Triplet Slider
            for i, token in enumerate(sequence):
                if token["type"] == "trigger":

                    # Look Left for the Subject
                    subj = None
                    for j in range(i - 1, -1, -1):
                        if sequence[j]["type"] == "node":
                            subj = sequence[j]["val"]
                            break

                    # Look Right for the Object
                    obj = None
                    for j in range(i + 1, len(sequence)):
                        if sequence[j]["type"] == "node":
                            obj = sequence[j]["val"]
                            break

                    # Rule 1: "Unlike" Sentence-Initial Edge Case
                    if (not subj and obj
                            and token["polarity"] == "inhibitory"):
                        right_nodes = [
                            t["val"] for t in sequence[i + 1:]
                            if t["type"] == "node"
                        ]
                        if len(right_nodes) >= 2:
                            legacy = right_nodes[0]
                            challenger = right_nodes[1]
                            # Old suppresses legacy
                            self.kernel.add_connection(
                                challenger, legacy, -0.8, sent)
                            # Reverse: legacy strongly calls for
                            # challenger as antidote/alternative
                            self.kernel.add_connection(
                                legacy, challenger, 0.8,
                                f"Antidote/Alternative: {sent}")
                            total_svo += 1

                    # Rule 2: Standard Center SVO
                    elif subj and obj:
                        if token["polarity"] == "inhibitory":
                            self.kernel.add_connection(
                                subj, obj, -0.8, sent)
                            # Reverse: obj strongly calls for subj
                            self.kernel.add_connection(
                                obj, subj, 0.8,
                                f"Antidote/Alternative: {sent}")
                            total_svo += 1
                        elif token["polarity"] == "excitatory":
                            self.kernel.add_connection(
                                subj, obj, 0.9, sent)
                            total_svo += 1

        return {
            'sentences': len(sentences),
            'concepts_found': total_nouns,
            'svo_edges': total_svo,
        }
