"""
KOS V8.0 -- Query Normalizer

Transforms raw user queries into canonical form before retrieval:
    1. Lowercase + strip punctuation
    2. Expand contractions (what's -> what is)
    3. Remove stop words
    4. Lemmatize content words
    5. Detect query intent (what/where/when/who/how/why/causal/compare)
    6. Select retrieval profile based on intent
"""

import re

# ---- Contractions --------------------------------------------------------
_CONTRACTIONS = {
    "what's": "what is", "where's": "where is", "who's": "who is",
    "how's": "how is", "it's": "it is", "that's": "that is",
    "there's": "there is", "here's": "here is", "he's": "he is",
    "she's": "she is", "let's": "let us", "can't": "cannot",
    "won't": "will not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "hasn't": "has not",
    "haven't": "have not", "hadn't": "had not", "wouldn't": "would not",
    "shouldn't": "should not", "couldn't": "could not",
    "i'm": "i am", "you're": "you are", "they're": "they are",
    "we're": "we are", "i've": "i have", "you've": "you have",
    "they've": "they have", "we've": "we have", "i'll": "i will",
    "you'll": "you will", "they'll": "they will", "we'll": "we will",
    "i'd": "i would", "you'd": "you would", "they'd": "they would",
}

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "shall", "may", "might", "can", "must", "am",
    "i", "me", "my", "we", "us", "our", "you", "your",
    "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "which", "who", "whom", "whose",
    "what", "where", "when", "how", "why",
    "and", "or", "but", "if", "then", "so", "because", "as", "while",
    "of", "in", "to", "for", "with", "on", "at", "by", "from", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over", "up", "down", "out", "off",
    "not", "no", "nor", "very", "just", "also", "too", "only",
    "tell", "me", "please", "could", "explain", "describe",
})

# ---- Intent Detection ---------------------------------------------------

_CAUSAL_WORDS = {"cause", "causes", "caused", "why", "because", "reason",
                 "leads", "lead", "result", "results", "trigger", "triggers",
                 "effect", "effects", "consequence"}
_COMPARE_WORDS = {"compare", "compared", "versus", "vs", "difference",
                  "differences", "between", "better", "worse", "similar",
                  "unlike", "contrast"}
_TEMPORAL_WORDS = {"when", "before", "after", "during", "timeline",
                   "history", "first", "last", "sequence", "order"}
_HOW_WORDS = {"how", "mechanism", "process", "method", "procedure",
              "step", "steps", "works", "function", "operate"}
_WHERE_WORDS = {"where", "located", "location", "place", "region",
                "area", "situated", "geography", "country", "city"}
_WHO_WORDS = {"who", "person", "people", "founder", "creator",
              "inventor", "author", "leader", "name", "named"}


def normalize(raw_query: str) -> dict:
    """
    Normalize a raw query and detect intent.

    Returns:
        {
            "raw": original query,
            "normalized": cleaned query string,
            "content_words": list of content words (no stop words),
            "intent": detected intent string,
            "profile": retrieval profile name,
        }
    """
    text = raw_query.strip().lower()

    # Expand contractions
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)

    # Strip punctuation (keep alphanumeric + spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Extract words
    words = text.split()

    # Content words (no stop words, len > 2)
    content_words = [w for w in words if w not in _STOP_WORDS and len(w) > 2]

    # Detect intent
    word_set = set(words)
    intent = _detect_intent(word_set)

    # Select retrieval profile
    profile = _INTENT_TO_PROFILE.get(intent, "default")

    return {
        "raw": raw_query,
        "normalized": text,
        "content_words": content_words,
        "intent": intent,
        "profile": profile,
    }


def _detect_intent(words: set) -> str:
    """Detect query intent from word set."""
    if words & _CAUSAL_WORDS:
        return "causal"
    if words & _COMPARE_WORDS:
        return "compare"
    if words & _TEMPORAL_WORDS:
        return "temporal"
    if words & _HOW_WORDS:
        return "how"
    if words & _WHERE_WORDS:
        return "where"
    if words & _WHO_WORDS:
        return "who"
    return "general"


# ---- Retrieval Profiles -------------------------------------------------
# Each profile defines beam search parameters and edge type filters.

PROFILES = {
    "default": {
        "beam_width": 32,
        "max_depth": 5,
        "top_k": 10,
        "allowed_edge_types": None,  # All types
    },
    "causal": {
        "beam_width": 24,
        "max_depth": 7,
        "top_k": 10,
        "allowed_edge_types": [2, 9, 10],  # CAUSES, TEMPORAL_BEFORE, TEMPORAL_AFTER
    },
    "taxonomic": {
        "beam_width": 32,
        "max_depth": 5,
        "top_k": 10,
        "allowed_edge_types": [1, 3, 12],  # IS_A, PART_OF, HAS_PROPERTY
    },
    "temporal": {
        "beam_width": 24,
        "max_depth": 5,
        "top_k": 10,
        "allowed_edge_types": [9, 10, 2],  # TEMPORAL_BEFORE, TEMPORAL_AFTER, CAUSES
    },
    "spatial": {
        "beam_width": 24,
        "max_depth": 3,
        "top_k": 10,
        "allowed_edge_types": [11, 3],  # LOCATED_IN, PART_OF
    },
}

_INTENT_TO_PROFILE = {
    "general": "default",
    "causal": "causal",
    "compare": "default",
    "temporal": "temporal",
    "how": "causal",       # "How" often implies causal/procedural
    "where": "spatial",
    "who": "default",
}


def get_profile(intent: str) -> dict:
    """Get retrieval profile for an intent."""
    name = _INTENT_TO_PROFILE.get(intent, "default")
    return PROFILES.get(name, PROFILES["default"])
