"""
KOS V6.1 — Automatic Compound Noun Detection.

Scans ingested text for frequently co-occurring adjacent noun pairs
and automatically registers them as compound nouns in the TextDriver.

This solves the problem of compound nouns like "machine learning" or
"climate change" being split into two separate nodes when they should
be one. Instead of maintaining a static list, the detector learns
new compounds from the corpus itself.

Uses NLTK POS tagging to identify adjacent NN/NNS/NNP pairs that
appear together 3+ times — a strong signal of compound noun status.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import nltk
# Ensure NLTK data is available
for _pkg in ['punkt_tab', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.data.find(f'tokenizers/{_pkg}' if 'punkt' in _pkg
                       else f'taggers/{_pkg}')
    except LookupError:
        nltk.download(_pkg, quiet=True)


# ---------------------------------------------------------------------------
# Compound Detector
# ---------------------------------------------------------------------------

class CompoundDetector:
    """Detects compound nouns from corpus frequency analysis.

    Scans sentences for adjacent noun pairs (both tagged NN, NNS, or NNP
    by NLTK's POS tagger). Pairs appearing 3+ times are classified as
    compounds and can be injected into the TextDriver's COMPOUND_NOUNS
    dictionary for future ingestion.

    Example::

        detector = CompoundDetector()
        sentences = [
            "Machine learning is a subset of artificial intelligence.",
            "Machine learning algorithms require large datasets.",
            "Deep learning is a type of machine learning.",
        ]
        compounds = detector.detect_from_corpus(sentences)
        # {('machine', 'learning'): 'machine_learning'}
        detector.update_textdriver(text_driver, compounds)
    """

    # POS tags considered "noun-like" for compound detection
    NOUN_TAGS: Set[str] = {'NN', 'NNS', 'NNP', 'NNPS'}

    # Minimum co-occurrence count to be classified as compound
    MIN_FREQUENCY: int = 3

    # Words too short or generic to form meaningful compounds
    STOP_WORDS: Set[str] = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
        'has', 'have', 'had', 'do', 'does', 'did', 'will',
        'can', 'may', 'it', 'its', 'this', 'that', 'these',
    }

    def __init__(self, min_frequency: int = 3,
                 min_word_length: int = 2) -> None:
        """Initialise the detector.

        Args:
            min_frequency: Minimum times a pair must appear to be
                           classified as a compound (default 3).
            min_word_length: Minimum character length per word in
                            the pair (default 2).
        """
        self.min_frequency = max(1, min_frequency)
        self.min_word_length = max(1, min_word_length)
        self._pair_counts: Counter = Counter()
        self._known_compounds: Dict[Tuple[str, str], str] = {}

    def detect_from_corpus(
        self, sentences: List[str]
    ) -> Dict[Tuple[str, str], str]:
        """Scan sentences for frequent adjacent noun pairs.

        Tokenises and POS-tags each sentence, then counts adjacent
        pairs where both words are tagged as nouns. Pairs meeting
        the frequency threshold are returned as compound candidates.

        Args:
            sentences: List of raw sentences to analyse.

        Returns:
            Dict mapping (word1, word2) -> compound_name for all
            detected compounds. compound_name is word1_word2 in
            lowercase with underscores.
        """
        if not sentences:
            return {}

        pair_counts: Counter = Counter()

        for sentence in sentences:
            if not sentence or not sentence.strip():
                continue

            try:
                tokens = nltk.word_tokenize(sentence)
                tagged = nltk.pos_tag(tokens)
            except Exception:
                continue

            # Scan for adjacent noun pairs
            for i in range(len(tagged) - 1):
                word1, tag1 = tagged[i]
                word2, tag2 = tagged[i + 1]

                # Both must be noun-tagged
                if tag1 not in self.NOUN_TAGS or tag2 not in self.NOUN_TAGS:
                    continue

                w1_lower = word1.lower()
                w2_lower = word2.lower()

                # Skip stop words and too-short words
                if (w1_lower in self.STOP_WORDS
                        or w2_lower in self.STOP_WORDS):
                    continue
                if (len(w1_lower) < self.min_word_length
                        or len(w2_lower) < self.min_word_length):
                    continue

                pair_counts[(w1_lower, w2_lower)] += 1

        # Merge with running counts
        self._pair_counts += pair_counts

        # Extract compounds meeting frequency threshold
        new_compounds: Dict[Tuple[str, str], str] = {}
        for pair, count in self._pair_counts.items():
            if count >= self.min_frequency and pair not in self._known_compounds:
                compound_name = f"{pair[0]}_{pair[1]}"
                new_compounds[pair] = compound_name
                self._known_compounds[pair] = compound_name

        return new_compounds

    def detect_incremental(
        self, sentences: List[str]
    ) -> Dict[Tuple[str, str], str]:
        """Incremental detection — only returns NEW compounds.

        Same as detect_from_corpus but only returns compounds that
        were not previously detected. Useful for batch-by-batch
        ingestion where you want to know what's new.

        Args:
            sentences: New sentences to analyse.

        Returns:
            Dict of only newly detected compounds.
        """
        return self.detect_from_corpus(sentences)

    def update_textdriver(
        self, text_driver, compounds: Dict[Tuple[str, str], str]
    ) -> int:
        """Add detected compounds to a TextDriver's COMPOUND_NOUNS dict.

        For each compound (word1, word2) -> compound_name, adds entries
        for both singular and common plural forms. This ensures the
        TextDriver merges these tokens on future ingestion.

        Args:
            text_driver: A TextDriver instance with a COMPOUND_NOUNS
                         class attribute (dict).
            compounds: Dict from detect_from_corpus.

        Returns:
            Number of new compound entries added.
        """
        if not compounds:
            return 0
        if not hasattr(text_driver, 'COMPOUND_NOUNS'):
            return 0

        added = 0
        for (w1, w2), compound_name in compounds.items():
            # Add the exact pair
            if (w1, w2) not in text_driver.COMPOUND_NOUNS:
                text_driver.COMPOUND_NOUNS[(w1, w2)] = compound_name
                added += 1

            # Add common plural variant (word2 + 's')
            plural_pair = (w1, w2 + 's')
            if (plural_pair not in text_driver.COMPOUND_NOUNS
                    and not w2.endswith('s')):
                text_driver.COMPOUND_NOUNS[plural_pair] = compound_name
                added += 1

            # If word2 already ends in 's', add singular variant
            if w2.endswith('s') and len(w2) > 2:
                singular_pair = (w1, w2[:-1])
                if singular_pair not in text_driver.COMPOUND_NOUNS:
                    text_driver.COMPOUND_NOUNS[singular_pair] = compound_name
                    added += 1

        return added

    def get_all_compounds(self) -> Dict[Tuple[str, str], str]:
        """Return all detected compounds so far.

        Returns:
            Dict mapping (word1, word2) -> compound_name.
        """
        return dict(self._known_compounds)

    def get_pair_counts(self, min_count: int = 1) -> List[Tuple[Tuple[str, str], int]]:
        """Return noun pair frequency counts.

        Useful for inspecting which pairs are close to the threshold
        or for debugging compound detection.

        Args:
            min_count: Only return pairs with at least this many
                       occurrences (default 1).

        Returns:
            List of ((word1, word2), count) sorted by count descending.
        """
        return [
            (pair, count) for pair, count in self._pair_counts.most_common()
            if count >= min_count
        ]

    def reset(self) -> None:
        """Reset all counts and detected compounds."""
        self._pair_counts.clear()
        self._known_compounds.clear()

    def __repr__(self) -> str:
        return (f"CompoundDetector(compounds={len(self._known_compounds)}, "
                f"tracked_pairs={len(self._pair_counts)})")
