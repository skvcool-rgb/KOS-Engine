"""
KOS V5.1 — Auto-Generated Synonym Map (Fix #18).

Replaces the hand-coded 50-entry synonym table with a
programmatically generated map from WordNet.

On first import, builds ~10K+ synonym pairs from WordNet lemmas.
Caches to disk for instant startup on subsequent runs.

Usage:
    from kos.synonyms import get_synonym_map
    synonyms = get_synonym_map()  # dict: word -> canonical_word
"""

import os
import json
import time

# Cache file location
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.cache')
_CACHE_FILE = os.path.join(_CACHE_DIR, 'synonym_map.json')

# Hand-coded domain synonyms that WordNet misses
_DOMAIN_SYNONYMS = {
    # People / Population
    "people": "population", "residents": "population",
    "inhabitants": "population", "citizens": "population",
    "lives": "population", "living": "population",
    "demographic": "population", "demographics": "population",

    # Location
    "place": "location", "situated": "located",
    "geography": "location", "area": "region",

    # Temperature / Climate
    "hot": "temperature", "cold": "temperature",
    "warm": "temperature", "cool": "temperature",
    "weather": "climate", "forecast": "climate",

    # Cost / Price
    "cost": "price", "costs": "price", "pricing": "price",
    "value": "price", "worth": "price",

    # Science
    "solar": "photovoltaic", "panels": "cell",
    "cells": "cell", "battery": "cell",
    "efficient": "efficiency", "effective": "efficiency",

    # Machine Learning (Agent Proposal A: learn <-> backpropagation)
    "learn": "backpropagation", "learning": "backpropagation",
    "train": "backpropagation", "training": "backpropagation",
    "optimize": "gradient", "optimization": "gradient",
    "convergence": "gradient", "descent": "gradient",

    # Medicine
    "drug": "medicine", "medication": "medicine",
    "treatment": "medicine", "therapy": "medicine",

    # General
    "biggest": "largest", "smallest": "smallest",
    "created": "founded", "built": "founded",
    "started": "founded", "established": "founded",
    "maker": "founder", "creator": "founder",
    "inventor": "founder",
    "metropolis": "city", "town": "city", "urban": "city",
    "nation": "country", "state": "country",
    "buy": "purchase", "bought": "purchase",
    "sell": "sale", "sold": "sale",
    "begin": "start", "commence": "start",
    "end": "finish", "conclude": "finish",
    "large": "big", "huge": "big", "massive": "big",
    "small": "little", "tiny": "little",
    "fast": "quick", "rapid": "quick", "swift": "quick",
    "slow": "sluggish",
    "old": "ancient", "aged": "ancient",
    "new": "recent", "modern": "recent",
    "important": "significant", "crucial": "significant",
    "difficult": "hard", "tough": "hard",
    "easy": "simple", "straightforward": "simple",
}


def _build_wordnet_synonyms() -> dict:
    """
    Build synonym map from WordNet.

    For each synset, all lemma names map to the first (canonical) lemma.
    Example: synset('car.n.01') has lemmas [car, auto, automobile, machine]
    Result: {"auto": "car", "automobile": "car", "machine": "car"}
    """
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        return {}

    synonym_map = {}
    count = 0

    for synset in wn.all_synsets():
        lemmas = [l.name().lower().replace('_', ' ')
                  for l in synset.lemmas()]
        if len(lemmas) < 2:
            continue

        canonical = lemmas[0]
        for alt in lemmas[1:]:
            if alt != canonical and len(alt) > 2 and len(canonical) > 2:
                # Don't overwrite existing entries (first synset wins)
                if alt not in synonym_map:
                    synonym_map[alt] = canonical
                    count += 1

    return synonym_map


def _load_or_build() -> dict:
    """Load cached synonyms or build from WordNet."""
    # Try cache first
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if len(cached) > 1000:  # Sanity check
                return cached
        except Exception:
            pass

    # Build from WordNet
    t0 = time.perf_counter()
    wn_synonyms = _build_wordnet_synonyms()
    elapsed = (time.perf_counter() - t0) * 1000

    # Merge: domain synonyms take priority
    merged = {**wn_synonyms, **_DOMAIN_SYNONYMS}

    # Cache to disk
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(merged, f)
    except Exception:
        pass

    return merged


# Module-level singleton
_SYNONYM_MAP = None


def get_synonym_map() -> dict:
    """Get the full synonym map (lazy-loaded, cached to disk)."""
    global _SYNONYM_MAP
    if _SYNONYM_MAP is None:
        _SYNONYM_MAP = _load_or_build()
    return _SYNONYM_MAP


def get_synonym(word: str) -> str:
    """Get the canonical synonym for a word, or the word itself."""
    m = get_synonym_map()
    return m.get(word.lower(), word)
