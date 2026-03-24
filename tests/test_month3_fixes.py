"""
KOS V5.1 — Month 3 Fixes: Temporal Reasoning + Multi-Language.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.temporal import TemporalReasoner
from kos.multilang import detect_language, extract_multilang_keywords


def run_month3_test():
    print("=" * 70)
    print("  MONTH 3 FIXES: Temporal Reasoning + Multi-Language")
    print("=" * 70)

    # ── TEST #3: Temporal Reasoning ──────────────────────────
    print("\n[TEST #3] TEMPORAL REASONING")
    print("-" * 50)

    kernel = KOSKernel(enable_vsa=False)
    lexicon = KASMLexicon()
    driver = TextDriver(kernel, lexicon)

    driver.ingest("""
    Toronto was founded in the year 1834.
    Montreal was founded in the year 1642.
    Vancouver was incorporated in the year 1886.
    Ottawa was established in the year 1855.
    """)

    reasoner = TemporalReasoner()

    # Test temporal comparison
    toronto_id = lexicon.word_to_uuid.get('toronto')
    montreal_id = lexicon.word_to_uuid.get('montreal')
    vancouver_id = lexicon.word_to_uuid.get('vancouver')

    if toronto_id and montreal_id:
        result = reasoner.compare_temporal(kernel, toronto_id, montreal_id, lexicon)
        print(f"  Toronto vs Montreal: {result.get('answer', result)}")
        correct = result.get('first', '').lower() == 'montreal'
        print(f"  Montreal came first: {'PASS' if correct else 'FAIL'}")

    # Test chronological sort
    all_cities = [nid for nid in [toronto_id, montreal_id, vancouver_id]
                  if nid]
    sorted_cities = reasoner.chronological_sort(kernel, all_cities, lexicon)
    print(f"\n  Chronological order:")
    for name, year, _ in sorted_cities:
        print(f"    {year:.0f}: {name}")

    if sorted_cities:
        first_correct = sorted_cities[0][0].lower() == 'montreal'
        print(f"  First city (Montreal): {'PASS' if first_correct else 'FAIL'}")

    # Test range query
    cities_1800s = reasoner.find_in_range(kernel, 1800, 1899, lexicon)
    print(f"\n  Cities founded in 1800s: {len(cities_1800s)}")
    for name, year, prop in cities_1800s:
        print(f"    {year:.0f}: {name}")

    # Test temporal query detection
    queries = [
        "Was Toronto founded before Montreal?",
        "What was the first city established?",
        "Cities founded in the 1800s",
        "What happened after 1850?",
    ]
    for q in queries:
        detected = reasoner.detect_temporal_query(q)
        print(f"  '{q[:40]}' → type={detected['type']}")

    # ── TEST #2: Multi-Language ──────────────────────────────
    print("\n[TEST #2] MULTI-LANGUAGE SUPPORT")
    print("-" * 50)

    # Language detection
    lang_tests = [
        ("Where is Toronto located?", "en"),
        ("Donde esta Toronto ciudad?", "es"),
        ("Ou est la ville de Toronto?", "fr"),
        ("Wo ist die Stadt Toronto?", "de"),
    ]

    lang_passed = 0
    for text, expected_lang in lang_tests:
        detected = detect_language(text)
        ok = detected == expected_lang
        if ok:
            lang_passed += 1
        print(f"  [{detected}] '{text[:40]}' "
              f"{'PASS' if ok else f'FAIL (expected {expected_lang})'}")

    print(f"\n  Language detection: {lang_passed}/{len(lang_tests)}")

    # Keyword extraction + translation
    print(f"\n  Multilingual keyword extraction:")
    ml_tests = [
        "Donde esta Toronto ciudad?",
        "Quelle est la population de Toronto?",
        "Was ist das Klima von Toronto?",
    ]
    for text in ml_tests:
        result = extract_multilang_keywords(text)
        print(f"  [{result['language']}] '{text[:40]}'")
        print(f"       Original: {result['original_words']}")
        print(f"       English:  {result['english_keywords']}")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MONTH 3 SUMMARY")
    print("=" * 70)
    print(f"  #3 Temporal reasoning:  comparison, sorting, range queries")
    print(f"  #2 Multi-language:      {lang_passed}/{len(lang_tests)} "
          f"language detection")
    print(f"     Supported: English, Spanish, French, German")
    print("=" * 70)


if __name__ == "__main__":
    run_month3_test()
