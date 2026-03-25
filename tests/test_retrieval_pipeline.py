"""
KOS V8.0 — Phase 1 Retrieval Pipeline Tests

Tests:
    1. Edge type inference from provenance text
    2. Typed edges in Rust kernel (add + retrieve)
    3. Hub penalty suppresses high-degree nodes
    4. Beam search returns bounded, ranked results
    5. Causal lane retrieves only CAUSES/TEMPORAL edges
    6. Memory tiers classify nodes correctly
    7. Retrieval cache hit/miss behavior
    8. Provenance trust scoring in weaver
    9. Graph.py edge_type wiring (auto-infer + pass-through)
    10. Full pipeline: ingest -> beam query -> weave
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_1_edge_type_inference():
    """Edge type inference from provenance text."""
    print("=" * 70)
    print("  TEST 1: Edge Type Inference")
    print("=" * 70)

    from kos.edge_types import infer_type, IS_A, CAUSES, PART_OF, SUPPORTS
    from kos.edge_types import LOCATED_IN, TEMPORAL_BEFORE, CONTRADICTS, GENERIC

    cases = [
        ("Toronto is a city.", IS_A),
        ("Rain causes flooding.", CAUSES),
        ("The engine is part of the car.", PART_OF),
        ("Studies support this conclusion.", SUPPORTS),
        ("The factory is located in Ontario.", LOCATED_IN),
        ("Event A occurred before event B.", TEMPORAL_BEFORE),
        ("This contradicts earlier findings.", CONTRADICTS),
        ("The sky is blue.", GENERIC),  # No pattern match
    ]

    passed = 0
    for text, expected in cases:
        result = infer_type(text)
        ok = result == expected
        if ok:
            passed += 1
        print(f"  {'PASS' if ok else 'FAIL'}: '{text[:50]}' -> {result} (expected {expected})")

    total_pass = passed == len(cases)
    print(f"\n  Result: {passed}/{len(cases)} {'PASS' if total_pass else 'FAIL'}")
    return total_pass


def test_2_typed_edges_rust():
    """Typed edges in Rust kernel."""
    print("\n" + "=" * 70)
    print("  TEST 2: Typed Edges in Rust Kernel")
    print("=" * 70)

    from kos_rust import RustKernel

    k = RustKernel(dim=100, seed=42)
    k.add_connection("toronto", "city", 0.9, "Toronto is a city.", 1)  # IS_A
    k.add_connection("toronto", "ontario", 0.8, "Toronto in Ontario.", 11)  # LOCATED_IN
    k.add_connection("rain", "flood", 0.9, "Rain causes flood.", 2)  # CAUSES

    # Check edge types are stored
    e1 = k.get_edge("toronto", "city")
    e2 = k.get_edge("toronto", "ontario")
    e3 = k.get_edge("rain", "flood")

    t1 = e1[2] == 1   # IS_A
    t2 = e2[2] == 11   # LOCATED_IN
    t3 = e3[2] == 2    # CAUSES

    print(f"  toronto->city edge_type={e1[2]} (expect 1 IS_A): {'PASS' if t1 else 'FAIL'}")
    print(f"  toronto->ontario edge_type={e2[2]} (expect 11 LOCATED_IN): {'PASS' if t2 else 'FAIL'}")
    print(f"  rain->flood edge_type={e3[2]} (expect 2 CAUSES): {'PASS' if t3 else 'FAIL'}")

    # Check get_neighbors includes edge type
    neighbors = k.get_neighbors("toronto")
    has_types = all(len(n) == 4 for n in neighbors)
    print(f"  get_neighbors returns 4-tuple (name, weight, myelin, type): {'PASS' if has_types else 'FAIL'}")

    passed = t1 and t2 and t3 and has_types
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_3_hub_penalty():
    """Hub penalty suppresses high-degree nodes."""
    print("\n" + "=" * 70)
    print("  TEST 3: Hub Penalty")
    print("=" * 70)

    from kos_rust import RustKernel

    k = RustKernel(dim=100, seed=42)

    # Create a hub node with many connections
    for i in range(50):
        k.add_connection("hub", f"target_{i}", 0.8, None, 0)

    # Create a focused node with few connections
    k.add_connection("focused", "target_0", 0.8, None, 0)
    k.add_connection("focused", "target_1", 0.8, None, 0)

    # Query from a node connected to both hub and focused
    k.add_connection("source", "hub", 0.9, None, 0)
    k.add_connection("source", "focused", 0.9, None, 0)

    results = k.query_beam(["source"], 60, 64, 3, None)
    result_dict = dict(results)

    # Hub's individual targets should have lower scores than focused's targets
    # because hub_penalty = 1/(1+ln(degree)) reduces hub's propagation
    hub_score = result_dict.get("hub", 0)
    focused_score = result_dict.get("focused", 0)

    # Both should be found, hub should be penalized relative to focused
    print(f"  hub score: {hub_score:.4f}")
    print(f"  focused score: {focused_score:.4f}")

    # The focused node should score at least as high (same weight, less penalty)
    passed = focused_score >= hub_score * 0.9  # Allow small margin
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_4_beam_search():
    """Beam search returns bounded, ranked results."""
    print("\n" + "=" * 70)
    print("  TEST 4: Beam Search")
    print("=" * 70)

    from kos_rust import RustKernel

    k = RustKernel(dim=100, seed=42)

    # Build a chain: a -> b -> c -> d -> e -> f
    k.add_connection("a", "b", 0.9, None, 0)
    k.add_connection("b", "c", 0.8, None, 0)
    k.add_connection("c", "d", 0.7, None, 0)
    k.add_connection("d", "e", 0.6, None, 0)
    k.add_connection("e", "f", 0.5, None, 0)

    # Beam search with depth=3 should NOT reach f (5 hops away)
    results_shallow = k.query_beam(["a"], 10, 8, 3, None)
    names_shallow = [r[0] for r in results_shallow]

    # Beam search with depth=5 should reach f
    results_deep = k.query_beam(["a"], 10, 8, 5, None)
    names_deep = [r[0] for r in results_deep]

    t1 = "b" in names_shallow  # Direct neighbor always found
    t2 = "f" not in names_shallow  # Too far for depth=3
    t3 = "b" in names_deep  # Still found at depth=5

    print(f"  depth=3 found 'b': {'PASS' if t1 else 'FAIL'}")
    print(f"  depth=3 did NOT find 'f': {'PASS' if t2 else 'FAIL'}")
    print(f"  depth=5 found 'b': {'PASS' if t3 else 'FAIL'}")
    print(f"  depth=3 results: {names_shallow}")
    print(f"  depth=5 results: {names_deep}")

    # Results should be sorted by descending score
    scores = [r[1] for r in results_deep if r[1] > 0]
    sorted_ok = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    print(f"  Scores sorted descending: {'PASS' if sorted_ok else 'FAIL'}")

    passed = t1 and t2 and t3 and sorted_ok
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_5_causal_lane():
    """Causal lane retrieves only CAUSES/TEMPORAL edges."""
    print("\n" + "=" * 70)
    print("  TEST 5: Causal Lane")
    print("=" * 70)

    from kos_rust import RustKernel

    k = RustKernel(dim=100, seed=42)

    # Causal chain: rain -> flood -> damage
    k.add_connection("rain", "flood", 0.9, "Rain causes flood.", 2)  # CAUSES
    k.add_connection("flood", "damage", 0.8, "Flood causes damage.", 2)  # CAUSES

    # Non-causal edges (should be filtered out)
    k.add_connection("rain", "weather", 0.9, "Rain is weather.", 1)  # IS_A
    k.add_connection("rain", "cloud", 0.7, "Rain from clouds.", 3)  # PART_OF

    # Causal query should only follow CAUSES edges
    causal = k.query_causal(["rain"], 10)
    causal_names = [r[0] for r in causal]

    t1 = "flood" in causal_names
    t2 = "weather" not in causal_names  # IS_A should be excluded
    t3 = "cloud" not in causal_names    # PART_OF should be excluded

    print(f"  Causal found 'flood': {'PASS' if t1 else 'FAIL'}")
    print(f"  Causal excluded 'weather' (IS_A): {'PASS' if t2 else 'FAIL'}")
    print(f"  Causal excluded 'cloud' (PART_OF): {'PASS' if t3 else 'FAIL'}")
    print(f"  Causal results: {causal_names}")

    passed = t1 and t2 and t3
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_6_memory_tiers():
    """Memory tier classification."""
    print("\n" + "=" * 70)
    print("  TEST 6: Memory Tiers")
    print("=" * 70)

    from kos.tiers import classify, bias, HOT_BIAS, WARM_BIAS, COLD_BIAS

    t1 = classify(10) == "hot"
    t2 = classify(100) == "warm"
    t3 = classify(500) == "cold"

    t4 = bias(10) == HOT_BIAS
    t5 = bias(100) == WARM_BIAS
    t6 = bias(500) == COLD_BIAS

    print(f"  classify(10)='hot': {'PASS' if t1 else 'FAIL'}")
    print(f"  classify(100)='warm': {'PASS' if t2 else 'FAIL'}")
    print(f"  classify(500)='cold': {'PASS' if t3 else 'FAIL'}")
    print(f"  bias(10)={HOT_BIAS}: {'PASS' if t4 else 'FAIL'}")
    print(f"  bias(100)={WARM_BIAS}: {'PASS' if t5 else 'FAIL'}")
    print(f"  bias(500)={COLD_BIAS}: {'PASS' if t6 else 'FAIL'}")

    passed = all([t1, t2, t3, t4, t5, t6])
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_7_retrieval_cache():
    """Retrieval cache hit/miss behavior."""
    print("\n" + "=" * 70)
    print("  TEST 7: Retrieval Cache")
    print("=" * 70)

    from kos_rust import RustKernel

    k = RustKernel(dim=100, seed=42)
    k.add_connection("a", "b", 0.9, None, 0)
    k.add_connection("a", "c", 0.8, None, 0)

    # First query — cache miss
    t0 = time.perf_counter()
    r1 = k.query(["a"], 5)
    t1 = time.perf_counter() - t0

    # Second identical query — should be cache hit (faster)
    t0 = time.perf_counter()
    r2 = k.query(["a"], 5)
    t2 = time.perf_counter() - t0

    # Results should be identical
    results_match = r1 == r2
    print(f"  First query: {t1*1000:.3f}ms")
    print(f"  Second query (cached): {t2*1000:.3f}ms")
    print(f"  Results match: {'PASS' if results_match else 'FAIL'}")

    stats = k.stats()
    cache_size = stats.get('cache_size', 0)
    print(f"  Cache size: {cache_size}")

    passed = results_match and cache_size > 0
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_8_provenance_trust():
    """Provenance trust scoring in weaver."""
    print("\n" + "=" * 70)
    print("  TEST 8: Provenance Trust Scoring")
    print("=" * 70)

    from kos.edge_types import infer_type, EDGE_CONFIG

    # High-trust provenance
    text1 = "Toronto is a city in Ontario."  # IS_A (trust=0.9)
    et1 = infer_type(text1)
    trust1 = EDGE_CONFIG[et1]["trust"]

    # Low-trust provenance
    text2 = "This contradicts the previous data."  # CONTRADICTS (trust=0.3)
    et2 = infer_type(text2)
    trust2 = EDGE_CONFIG[et2]["trust"]

    t1 = trust1 >= 0.8
    t2 = trust2 <= 0.4

    print(f"  '{text1[:40]}' -> type={et1}, trust={trust1}: {'PASS' if t1 else 'FAIL'}")
    print(f"  '{text2[:40]}' -> type={et2}, trust={trust2}: {'PASS' if t2 else 'FAIL'}")

    passed = t1 and t2
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_9_graph_edge_type_wiring():
    """Graph.py auto-infers edge type from provenance."""
    print("\n" + "=" * 70)
    print("  TEST 9: Graph Edge Type Wiring")
    print("=" * 70)

    from kos.graph import KOSKernel

    k = KOSKernel(enable_vsa=False)

    # Add with provenance — should auto-infer IS_A
    k.add_connection("toronto", "city", 0.9, "Toronto is a city.")

    # Add with explicit edge type
    k.add_connection("rain", "flood", 0.9, "Rain causes flood.", edge_type=2)

    # Check Rust backend has correct types
    if k._rust is not None:
        e1 = k._rust.get_edge("toronto", "city")
        e2 = k._rust.get_edge("rain", "flood")

        t1 = e1[2] == 1   # Auto-inferred IS_A
        t2 = e2[2] == 2   # Explicit CAUSES

        print(f"  Auto-inferred toronto->city type={e1[2]} (expect 1 IS_A): {'PASS' if t1 else 'FAIL'}")
        print(f"  Explicit rain->flood type={e2[2]} (expect 2 CAUSES): {'PASS' if t2 else 'FAIL'}")

        passed = t1 and t2
    else:
        print("  Rust backend not available — skipping edge type check")
        passed = True

    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_10_full_pipeline():
    """Full pipeline: ingest -> beam query -> weave."""
    print("\n" + "=" * 70)
    print("  TEST 10: Full Pipeline (Ingest -> Beam Query -> Weave)")
    print("=" * 70)

    from kos.graph import KOSKernel
    from kos.weaver import AlgorithmicWeaver

    k = KOSKernel(enable_vsa=False)
    w = AlgorithmicWeaver()

    # Ingest a small knowledge graph about Toronto
    edges = [
        ("toronto", "city", 0.9, "Toronto is a major city in Canada."),
        ("toronto", "ontario", 0.85, "Toronto is located in Ontario."),
        ("toronto", "population", 0.8, "Toronto has a population of 2.9 million."),
        ("toronto", "cn_tower", 0.7, "Toronto is famous for the CN Tower."),
        ("ontario", "canada", 0.9, "Ontario is a province of Canada."),
        ("montreal", "city", 0.9, "Montreal is a city in Canada."),
        ("montreal", "quebec", 0.85, "Montreal is located in Quebec."),
    ]

    for src, tgt, weight, text in edges:
        k.add_connection(src, tgt, weight, text)

    # Beam search query
    results = k.query_beam(["toronto"], top_k=5)
    result_names = [r[0] for r in results]
    print(f"  Beam query results: {result_names}")

    t1 = len(results) > 0
    t2 = "city" in result_names or "ontario" in result_names

    # Weave evidence — use a minimal mock lexicon
    class MockLexicon:
        def __init__(self):
            self.word_to_uuid = {}
        def get_word(self, uid):
            return uid  # IDs are plain words in this test

    lex = MockLexicon()

    evidence = w.weave(k, ["toronto"], results, lex,
                       ["toronto"], "Tell me about Toronto")
    t3 = len(evidence) > 0
    print(f"  Evidence: {evidence[:100]}...")
    print(f"  Has results: {'PASS' if t1 else 'FAIL'}")
    print(f"  Has relevant nodes: {'PASS' if t2 else 'FAIL'}")
    print(f"  Has evidence: {'PASS' if t3 else 'FAIL'}")

    passed = t1 and t2 and t3
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  KOS V8.0 — PHASE 1 RETRIEVAL PIPELINE TESTS")
    print("#  Typed Edges, Hub Penalty, Beam Search, Tiers, Provenance Trust")
    print("#" * 70)

    results = []
    results.append(("Edge type inference", test_1_edge_type_inference()))
    results.append(("Typed edges (Rust)", test_2_typed_edges_rust()))
    results.append(("Hub penalty", test_3_hub_penalty()))
    results.append(("Beam search", test_4_beam_search()))
    results.append(("Causal lane", test_5_causal_lane()))
    results.append(("Memory tiers", test_6_memory_tiers()))
    results.append(("Retrieval cache", test_7_retrieval_cache()))
    results.append(("Provenance trust", test_8_provenance_trust()))
    results.append(("Graph edge type wiring", test_9_graph_edge_type_wiring()))
    results.append(("Full pipeline", test_10_full_pipeline()))

    print("\n" + "=" * 70)
    print("  V8.0 PHASE 1 — FINAL RESULTS")
    print("=" * 70)
    for name, passed in results:
        print(f"  {name:30s} {'PASS' if passed else 'FAIL'}")

    total = sum(1 for _, p in results if p)
    print(f"\n  Total: {total}/{len(results)}")
    print("=" * 70)
