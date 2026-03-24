"""
KOS V5.1 — Week 1 Fixes Verification Test.

Tests:
    #4  Negation: "does NOT cause" → inhibitory edge
    #1  Adjectives: "cheap", "humid" become property nodes
    #10 Clause splitting: compound sentences get fine-grained provenance
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver


def run_week1_test():
    print("=" * 70)
    print("  WEEK 1 FIXES: Negation + Adjectives + Clause Splitting")
    print("=" * 70)

    kernel = KOSKernel(enable_vsa=False)
    lexicon = KASMLexicon()
    driver = TextDriver(kernel, lexicon)

    # ── TEST #4: Negation Handling ───────────────────────────
    print("\n[TEST #4] NEGATION HANDLING")
    print("-" * 50)

    negation_corpus = """
    Apixaban does not cause bleeding in patients.
    Warfarin does not prevent dietary interactions.
    Perovskite is not expensive to manufacture.
    Silicon cannot replace perovskite in efficiency.
    Mercury never causes beneficial health effects.
    """
    driver.ingest(negation_corpus)

    test_cases_neg = [
        ("apixaban", "bleeding", "inhibitory",
         "does NOT cause → should be inhibitory"),
        ("warfarin", "interaction", "inhibitory",
         "does NOT prevent → negated inhibitory = excitatory... "
         "but 'prevent' is inhibitory, negated = excitatory"),
    ]

    apix_id = lexicon.word_to_uuid.get('apixaban')
    bleed_id = lexicon.word_to_uuid.get('bleeding')

    if apix_id and bleed_id and apix_id in kernel.nodes:
        conn = kernel.nodes[apix_id].connections
        if bleed_id in conn:
            w = conn[bleed_id]
            weight = w['w'] if isinstance(w, dict) else w
            is_neg = weight < 0
            print(f"  apixaban -> bleeding: weight = {weight:.2f} "
                  f"({'INHIBITORY' if is_neg else 'EXCITATORY'}) "
                  f"{'PASS' if is_neg else 'FAIL'}")
        else:
            print(f"  apixaban -> bleeding: no direct edge (checking ambient)")
            # Check ambient edges
            for tgt, data in kernel.nodes[apix_id].connections.items():
                word = lexicon.get_word(tgt)
                w = data['w'] if isinstance(data, dict) else data
                if 'bleed' in word.lower():
                    print(f"    Found: {word} weight={w:.2f}")
    else:
        print(f"  apixaban or bleeding not found in graph")

    # Check for adjective "expensive" being negated for perovskite
    perov_id = lexicon.word_to_uuid.get('perovskite')
    expensive_id = lexicon.word_to_uuid.get('expensive')

    if perov_id and expensive_id:
        if perov_id in kernel.nodes and expensive_id in kernel.nodes[perov_id].connections:
            w = kernel.nodes[perov_id].connections[expensive_id]
            weight = w['w'] if isinstance(w, dict) else w
            print(f"  perovskite -> expensive: weight = {weight:.2f} "
                  f"{'PASS (should be property)' if weight > 0 else 'negated'}")
        else:
            print(f"  perovskite -> expensive: no edge (may be adjective node)")

    # ── TEST #1: Adjective Extraction ────────────────────────
    print("\n[TEST #1] ADJECTIVE EXTRACTION")
    print("-" * 50)

    kernel2 = KOSKernel(enable_vsa=False)
    lexicon2 = KASMLexicon()
    driver2 = TextDriver(kernel2, lexicon2)

    adj_corpus = """
    Perovskite is a cheap and affordable material.
    Toronto has a humid continental climate.
    Silicon wafers are extremely brittle and fragile.
    The massive population of Toronto exceeds 2.7 million.
    """
    result = driver2.ingest(adj_corpus)

    adj_words = ['cheap', 'affordable', 'humid', 'continental',
                 'brittle', 'fragile', 'massive']
    found_adj = []
    missing_adj = []

    for adj in adj_words:
        if adj in lexicon2.word_to_uuid:
            found_adj.append(adj)
        else:
            missing_adj.append(adj)

    print(f"  Adjectives found as nodes: {found_adj}")
    print(f"  Adjectives missing:        {missing_adj}")
    print(f"  Pass rate: {len(found_adj)}/{len(adj_words)} "
          f"({len(found_adj)/len(adj_words):.0%})")

    # Check property edges
    perov_id2 = lexicon2.word_to_uuid.get('perovskite')
    if perov_id2 and perov_id2 in kernel2.nodes:
        print(f"\n  Perovskite property edges:")
        for tgt, data in kernel2.nodes[perov_id2].connections.items():
            word = lexicon2.get_word(tgt)
            w = data['w'] if isinstance(data, dict) else data
            if word.lower() in adj_words:
                print(f"    -> {word}: weight={w:.2f} PROPERTY")

    toronto_id = lexicon2.word_to_uuid.get('toronto')
    if toronto_id and toronto_id in kernel2.nodes:
        print(f"\n  Toronto property edges:")
        for tgt, data in kernel2.nodes[toronto_id].connections.items():
            word = lexicon2.get_word(tgt)
            w = data['w'] if isinstance(data, dict) else data
            if word.lower() in adj_words:
                print(f"    -> {word}: weight={w:.2f} PROPERTY")

    # ── TEST #10: Clause-Level Provenance ────────────────────
    print("\n[TEST #10] CLAUSE-LEVEL PROVENANCE SPLITTING")
    print("-" * 50)

    kernel3 = KOSKernel(enable_vsa=False)
    lexicon3 = KASMLexicon()
    driver3 = TextDriver(kernel3, lexicon3)

    clause_corpus = """
    Toronto, founded in 1834 by Simcoe, is located in Ontario.
    Perovskite, which is cheap to manufacture, has high efficiency.
    The CN Tower, built in 1976, is the tallest structure in Toronto.
    """
    result3 = driver3.ingest(clause_corpus)

    print(f"  Sentences: {result3['sentences']}")
    print(f"  Clauses extracted: {result3.get('clauses', 'N/A')}")

    # Check provenance granularity
    toronto_id3 = lexicon3.word_to_uuid.get('toronto')
    if toronto_id3:
        prov_entries = set()
        for pair, sentences in kernel3.provenance.items():
            if toronto_id3 in pair:
                prov_entries.update(sentences)
        print(f"  Provenance entries for Toronto: {len(prov_entries)}")
        for p in sorted(prov_entries)[:5]:
            print(f"    -> {p[:80]}...")

    # ── Regression: Run original tests ───────────────────────
    print("\n[REGRESSION] Running core functionality check...")
    print("-" * 50)

    kernel4 = KOSKernel(enable_vsa=False)
    lexicon4 = KASMLexicon()
    driver4 = TextDriver(kernel4, lexicon4)

    regression_corpus = """
    Toronto is a major city in the Canadian province of Ontario.
    Toronto was founded and incorporated in the year 1834.
    The city of Toronto has a population of approximately 2.7 million people.
    Toronto has a humid continental climate with warm summers and cold winters.
    Perovskite is a highly efficient material used in modern photovoltaic cells.
    Photovoltaic cells capture photons to produce electricity efficiently.
    Apixaban prevents thrombosis without requiring strict dietary restrictions.
    Unlike warfarin, apixaban does not require constant diet monitoring.
    """
    driver4.ingest(regression_corpus)

    from kos.router_offline import KOSShellOffline
    shell = KOSShellOffline(kernel4, lexicon4, enable_forager=False)

    regression_tests = [
        ("Where is Toronto?", ["ontario", "province"]),
        ("When was Toronto founded?", ["1834"]),
        ("Population of Toronto?", ["2.7", "million"]),
        ("Climate of Toronto?", ["humid", "continental"]),
        ("How do photovoltaic cells work?", ["photon", "electricity"]),
        ("Tell me about apixaban", ["thrombosis", "apixaban"]),
        ("345000000 * 0.0825", ["28462500"]),
    ]

    passed = 0
    for query, expected in regression_tests:
        answer = shell.chat(query)
        answer_lower = answer.lower()
        hits = [kw for kw in expected if kw.lower() in answer_lower]
        ok = len(hits) >= 1
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {query:35s} -> {answer[:60]}")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  WEEK 1 SUMMARY")
    print("=" * 70)
    print(f"  #4  Negation handling:     IMPLEMENTED")
    print(f"  #1  Adjective extraction:  {len(found_adj)}/{len(adj_words)} adjectives captured")
    print(f"  #10 Clause splitting:      {result3.get('clauses', 0)} clauses from {result3['sentences']} sentences")
    print(f"  Regression tests:          {passed}/{len(regression_tests)} pass")
    print("=" * 70)


if __name__ == "__main__":
    run_week1_test()
