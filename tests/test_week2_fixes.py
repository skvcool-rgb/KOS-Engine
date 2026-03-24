"""
KOS V5.1 — Week 2 Fixes Verification Test.

Tests:
    #6  Weaver scoring optimization (HOW intent, density, noise)
    #9  Contradiction detection at ingestion
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.router_offline import KOSShellOffline


def run_week2_test():
    print("=" * 70)
    print("  WEEK 2 FIXES: Weaver Optimization + Contradiction Detection")
    print("=" * 70)

    # ── TEST #6: Weaver Scoring Optimization ─────────────────
    print("\n[TEST #6] WEAVER SCORING OPTIMIZATION")
    print("-" * 50)

    kernel = KOSKernel(enable_vsa=False)
    lexicon = KASMLexicon()
    driver = TextDriver(kernel, lexicon)
    shell = KOSShellOffline(kernel, lexicon, enable_forager=False)

    corpus = """
    Toronto is a major city in the Canadian province of Ontario.
    Toronto was founded and incorporated in the year 1834.
    The city of Toronto has a population of approximately 2.7 million people.
    Toronto has a humid continental climate with warm summers and cold winters.
    John Graves Simcoe originally established the settlement of Toronto.
    Photovoltaic cells capture photons to produce electricity through a process called the photovoltaic effect.
    Perovskite solar cells convert sunlight into electricity using a thin film of perovskite material.
    The Toronto Blue Jays play professional baseball at Rogers Centre.
    """
    driver.ingest(corpus)

    intent_tests = [
        ("WHERE", "Where is Toronto located?",
         ["ontario", "province", "canadian"]),
        ("WHEN", "When was Toronto founded?",
         ["1834", "founded"]),
        ("WHO", "Who established Toronto?",
         ["simcoe", "john", "established"]),
        ("WHAT", "What is the population of Toronto?",
         ["2.7", "million", "population"]),
        ("WHAT-ATTR", "What is the climate of Toronto?",
         ["humid", "continental", "climate"]),
        ("HOW", "How do photovoltaic cells work?",
         ["photon", "electricity", "capture", "process"]),
        ("NOISE", "Tell me about Toronto city",
         ["city", "ontario"]),  # Should NOT return Blue Jays
    ]

    passed = 0
    for intent, query, expected in intent_tests:
        answer = shell.chat(query)
        answer_lower = answer.lower()
        hits = [kw for kw in expected if kw.lower() in answer_lower]
        ok = len(hits) >= 1

        # For NOISE test, also verify sports content is suppressed
        if intent == "NOISE":
            has_sports = any(w in answer_lower for w in
                             ["blue jays", "baseball", "rogers"])
            if has_sports:
                ok = False  # Sports should be suppressed

        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {intent:10s} | {query:40s} | {answer[:50]}")

    print(f"\n  Weaver intent tests: {passed}/{len(intent_tests)}")

    # ── TEST #9: Contradiction Detection ─────────────────────
    print("\n[TEST #9] CONTRADICTION DETECTION AT INGESTION")
    print("-" * 50)

    kernel2 = KOSKernel(enable_vsa=False)
    lexicon2 = KASMLexicon()
    driver2 = TextDriver(kernel2, lexicon2)

    # First: establish a belief
    driver2.ingest("Perovskite is an expensive material for solar cells.")
    print(f"  After 'expensive' ingestion: {len(kernel2.contradictions)} contradictions")

    # Then: inject contradicting evidence
    driver2.ingest("Perovskite is a cheap material for solar cells.")
    print(f"  After 'cheap' ingestion: {len(kernel2.contradictions)} contradictions")

    if kernel2.contradictions:
        for c in kernel2.contradictions:
            print(f"    DETECTED: {c.get('existing_word', '?')} <-> "
                  f"{c.get('new_word', '?')} "
                  f"(type: {c.get('type', '?')})")
    else:
        print(f"    No antonym contradictions detected "
              f"(WordNet may not have these as antonyms)")

    # Test with known WordNet antonyms
    driver2.ingest("The process is fast and efficient.")
    driver2.ingest("The process is slow and inefficient.")

    print(f"  After fast/slow ingestion: {len(kernel2.contradictions)} total contradictions")

    for c in kernel2.contradictions:
        print(f"    DETECTED: {c.get('existing_word', '?')} <-> "
              f"{c.get('new_word', '?')}")

    # ── Regression ───────────────────────────────────────────
    print("\n[REGRESSION] Running 20/20 offline test...")
    print("-" * 50)

    kernel3 = KOSKernel(enable_vsa=False)
    lexicon3 = KASMLexicon()
    driver3 = TextDriver(kernel3, lexicon3)
    shell3 = KOSShellOffline(kernel3, lexicon3, enable_forager=False)

    driver3.ingest("""
    Toronto is a major city in the Canadian province of Ontario.
    Toronto was founded and incorporated in the year 1834.
    The city of Toronto has a population of approximately 2.7 million people.
    Toronto has a humid continental climate with warm summers and cold winters.
    John Graves Simcoe originally established the settlement of Toronto.
    Perovskite is a highly efficient material used in modern photovoltaic cells.
    Photovoltaic cells capture photons to produce electricity efficiently.
    Silicon is a traditional semiconductor used in computing and solar panels.
    Perovskite solar cells are remarkably cheap and affordable to manufacture.
    Apixaban prevents thrombosis without requiring strict dietary restrictions.
    Unlike warfarin, apixaban does not require constant diet monitoring.
    A parent company acquired its subsidiary through a corporate merger.
    Corporate mergers face immediate antitrust regulation by government agencies.
    """)

    regression = [
        ("Where is Toronto?", ["ontario"]),
        ("When was Toronto founded?", ["1834"]),
        ("Population of Toronto?", ["million"]),
        ("Climate of Toronto?", ["humid"]),
        ("Who established Toronto?", ["simcoe"]),
        ("How do photovoltaic cells work?", ["photon"]),
        ("Tell me about apixaban", ["thrombosis"]),
        ("345000000 * 0.0825", ["28462500"]),
    ]

    reg_passed = 0
    for query, expected in regression:
        answer = shell3.chat(query)
        hits = [kw for kw in expected if kw.lower() in answer.lower()]
        ok = len(hits) >= 1
        if ok:
            reg_passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {query:35s}")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  WEEK 2 SUMMARY")
    print("=" * 70)
    print(f"  #6  Weaver optimization:     {passed}/{len(intent_tests)} intent tests pass")
    print(f"  #9  Contradiction detection: {len(kernel2.contradictions)} contradictions found")
    print(f"  Regression:                  {reg_passed}/{len(regression)} pass")
    print("=" * 70)


if __name__ == "__main__":
    run_week2_test()
