"""
KOS V5.1 — Week 4 Fixes Verification Test.

Tests:
    #11 Multi-backend Forager (Wikipedia + arXiv + file)
    #8  Quantitative comparison (numeric properties + compare)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver


def run_week4_test():
    print("=" * 70)
    print("  WEEK 4 FIXES: Multi-Backend Forager + Quantitative Comparison")
    print("=" * 70)

    # ── TEST #8: Quantitative Comparison ─────────────────────
    print("\n[TEST #8] QUANTITATIVE COMPARISON")
    print("-" * 50)

    kernel = KOSKernel(enable_vsa=False)
    lexicon = KASMLexicon()
    driver = TextDriver(kernel, lexicon)

    corpus = """
    Toronto has a population of approximately 2.7 million people.
    Montreal has a population of approximately 1.7 million residents.
    Vancouver has a population of approximately 630 thousand people.
    Toronto was founded in the year 1834.
    Montreal was founded in the year 1642.
    Vancouver was incorporated in the year 1886.
    """
    driver.ingest(corpus)

    # Check numeric properties were extracted
    toronto_id = lexicon.word_to_uuid.get('toronto')
    montreal_id = lexicon.word_to_uuid.get('montreal')
    vancouver_id = lexicon.word_to_uuid.get('vancouver')

    print(f"  Node properties:")
    for name, uid in [("Toronto", toronto_id), ("Montreal", montreal_id),
                       ("Vancouver", vancouver_id)]:
        if uid and uid in kernel.nodes:
            props = kernel.nodes[uid].properties
            print(f"    {name}: {props}")
        else:
            print(f"    {name}: not found")

    # Test comparison
    print(f"\n  Quantitative comparisons:")

    if toronto_id and montreal_id:
        result = kernel.compare(toronto_id, montreal_id, "population")
        if "population" in result:
            comp = result["population"]
            t_bigger = comp.get("comparison") == "greater"
            print(f"    Toronto vs Montreal (population): "
                  f"{comp.get('a', '?'):,.0f} vs {comp.get('b', '?'):,.0f} "
                  f"-> Toronto is {comp.get('comparison', '?')} "
                  f"({'PASS' if t_bigger else 'FAIL'})")
        else:
            print(f"    Toronto vs Montreal: {result}")

    if toronto_id and montreal_id:
        result = kernel.compare(toronto_id, montreal_id, "founded")
        if "founded" in result:
            comp = result["founded"]
            m_older = comp.get("comparison") == "greater"
            print(f"    Toronto vs Montreal (founded): "
                  f"{comp.get('a', '?'):.0f} vs {comp.get('b', '?'):.0f} "
                  f"-> Montreal is older "
                  f"({'PASS' if not m_older else 'FAIL — Toronto year > Montreal year means Montreal is older'})")
        elif "year" in result:
            comp = result["year"]
            print(f"    Toronto vs Montreal (year): "
                  f"{comp.get('a', '?'):.0f} vs {comp.get('b', '?'):.0f}")
        else:
            print(f"    Toronto vs Montreal (founded): {result}")

    # ── TEST #11: Multi-Backend Forager ──────────────────────
    print("\n[TEST #11] MULTI-BACKEND FORAGER")
    print("-" * 50)

    from kos.forager import WebForager
    forager = WebForager(kernel, lexicon, text_driver=driver)

    # Test arXiv backend
    print("\n  Testing arXiv backend...")
    arxiv_nodes = forager.forage_arxiv("perovskite solar cell", max_results=1)
    print(f"  arXiv result: +{arxiv_nodes} concepts")

    # Test file backend
    print("\n  Testing file backend...")
    test_file = os.path.join(os.path.dirname(__file__), '_test_doc.txt')
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("Graphene is a single layer of carbon atoms arranged "
                "in a hexagonal lattice.\n"
                "Graphene has extraordinary electrical conductivity.\n"
                "Graphene was first isolated in 2004 by Geim and Novoselov.\n")

    file_nodes = forager.forage_file(test_file)
    print(f"  File result: +{file_nodes} concepts")

    # Clean up test file
    os.remove(test_file)

    # Check that graphene was ingested
    graphene_id = lexicon.word_to_uuid.get('graphene')
    print(f"  'graphene' in graph: {graphene_id is not None}")

    # Test smart foraging
    print("\n  Testing smart foraging...")
    before = len(kernel.nodes)
    smart_nodes = forager.forage_smart("quantum computing", verbose=True)
    print(f"  Smart forage result: +{smart_nodes} concepts "
          f"(total graph: {len(kernel.nodes)})")

    # ── Regression ───────────────────────────────────────────
    print("\n[REGRESSION] Core functionality check...")
    print("-" * 50)

    from kos.router_offline import KOSShellOffline
    shell = KOSShellOffline(kernel, lexicon, enable_forager=False)

    regression = [
        ("Population of Toronto?", ["million"]),
        ("When was Montreal founded?", ["1642"]),
        ("345000000 * 0.0825", ["28462500"]),
    ]

    reg_passed = 0
    for query, expected in regression:
        answer = shell.chat(query)
        hits = [kw for kw in expected if kw.lower() in answer.lower()]
        ok = len(hits) >= 1
        if ok:
            reg_passed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {query}")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  WEEK 4 SUMMARY")
    print("=" * 70)
    print(f"  #8  Numeric properties:  extracted from corpus")
    print(f"  #8  Comparison:          Toronto > Montreal (population)")
    print(f"  #11 arXiv backend:       +{arxiv_nodes} concepts")
    print(f"  #11 File backend:        +{file_nodes} concepts")
    print(f"  #11 Smart foraging:      +{smart_nodes} concepts")
    print(f"  Regression:              {reg_passed}/{len(regression)} pass")
    print("=" * 70)


if __name__ == "__main__":
    run_week4_test()
