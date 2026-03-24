"""
KOS V5.1 — Multi-Language Support (Fix #2).

Language detection + language-specific NLP pipelines.
Each pipeline implements tokenize() and extract_keywords().

Currently supported:
    - English (full: NLTK + WordNet + phonetics)
    - Spanish (basic: regex + stopwords + translation hints)
    - French (basic: regex + stopwords + translation hints)
    - German (basic: regex + stopwords + translation hints)

The graph itself is language-agnostic — only the I/O layer changes.
All internal concepts are stored as English-lemmatized UUIDs.
"""

import re


# ── Language Detection ───────────────────────────────────────

# Simple statistical detection based on common word frequency
_LANG_INDICATORS = {
    'es': {'el', 'la', 'los', 'las', 'de', 'del', 'en', 'es',
           'que', 'un', 'una', 'por', 'con', 'para', 'como',
           'mas', 'pero', 'donde', 'cuando', 'quien', 'cual',
           'ciudad', 'tiene', 'esta'},
    'fr': {'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une',
           'est', 'dans', 'pour', 'avec', 'sur', 'par', 'que',
           'qui', 'mais', 'ou', 'ville', 'cette'},
    'de': {'der', 'die', 'das', 'ein', 'eine', 'ist', 'und',
           'mit', 'von', 'auf', 'fur', 'den', 'dem', 'nicht',
           'auch', 'sich', 'stadt', 'haben'},
}

# Translation hints: foreign word → English concept
_TRANSLATION_HINTS = {
    # Spanish
    'ciudad': 'city', 'poblacion': 'population', 'clima': 'climate',
    'fundada': 'founded', 'donde': 'where', 'cuando': 'when',
    'quien': 'who', 'tiene': 'has', 'esta': 'is', 'grande': 'big',
    'pequeno': 'small', 'rio': 'river', 'montana': 'mountain',
    'pais': 'country', 'capital': 'capital', 'gente': 'people',
    'ano': 'year', 'temperatura': 'temperature', 'lluvia': 'rain',
    'costa': 'coast', 'norte': 'north', 'sur': 'south',

    # French
    'ville': 'city', 'populacion': 'population', 'climat': 'climate',
    'fondee': 'founded', 'fleuve': 'river', 'montagne': 'mountain',
    'pays': 'country', 'capitale': 'capital', 'gens': 'people',
    'annee': 'year', 'temperature': 'temperature', 'pluie': 'rain',
    'cote': 'coast',

    # German
    'stadt': 'city', 'bevolkerung': 'population', 'klima': 'climate',
    'gegrundet': 'founded', 'fluss': 'river', 'berg': 'mountain',
    'land': 'country', 'hauptstadt': 'capital', 'leute': 'people',
    'jahr': 'year', 'temperatur': 'temperature', 'regen': 'rain',
    'kuste': 'coast',
}

# Foreign stopwords to remove
_FOREIGN_STOPWORDS = {
    # Spanish
    'el', 'la', 'los', 'las', 'de', 'del', 'en', 'es', 'que',
    'un', 'una', 'por', 'con', 'para', 'como', 'mas', 'pero',
    'se', 'su', 'al', 'lo', 'le',
    # French
    'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et',
    'est', 'dans', 'pour', 'avec', 'sur', 'par', 'qui', 'que',
    # German
    'der', 'die', 'das', 'ein', 'eine', 'und', 'ist', 'mit',
    'von', 'auf', 'fur', 'den', 'dem', 'nicht', 'auch', 'sich',
}


def detect_language(text: str) -> str:
    """
    Simple language detection based on word frequency.

    Returns: 'en', 'es', 'fr', 'de', or 'en' (default)
    """
    words = set(re.findall(r'[a-zA-Z]+', text.lower()))
    if len(words) < 3:
        return 'en'

    scores = {}
    for lang, indicators in _LANG_INDICATORS.items():
        overlap = words & indicators
        scores[lang] = len(overlap) / len(words)

    best_lang = max(scores, key=scores.get)
    if scores[best_lang] > 0.15:  # At least 15% indicator words
        return best_lang

    return 'en'


def translate_keywords(words: list, source_lang: str) -> list:
    """
    Translate foreign keywords to English using hint table.

    This is NOT a full translator — it maps known domain terms
    to their English equivalents for graph lookup.
    """
    if source_lang == 'en':
        return words

    translated = []
    for w in words:
        w_lower = w.lower()
        # Check translation hints
        if w_lower in _TRANSLATION_HINTS:
            translated.append(_TRANSLATION_HINTS[w_lower])
        elif w_lower not in _FOREIGN_STOPWORDS:
            # Keep as-is (might be a proper noun like "Toronto")
            translated.append(w_lower)

    return translated


def extract_multilang_keywords(text: str) -> dict:
    """
    Full pipeline: detect language → extract keywords → translate.

    Returns: {
        'language': 'es',
        'original_words': ['donde', 'esta', 'toronto', 'ciudad'],
        'english_keywords': ['where', 'is', 'toronto', 'city']
    }
    """
    lang = detect_language(text)

    # Tokenize
    words = re.findall(r'[a-zA-Z]+', text.lower())

    # Remove stopwords (both English and foreign)
    from kos.router_offline import STOPWORDS
    all_stops = STOPWORDS | _FOREIGN_STOPWORDS
    meaningful = [w for w in words if w not in all_stops and len(w) > 2]

    # Translate to English
    english_kw = translate_keywords(meaningful, lang)

    return {
        'language': lang,
        'original_words': meaningful,
        'english_keywords': english_kw,
    }
