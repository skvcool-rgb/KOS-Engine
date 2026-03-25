"""
KOS V8.0 -- Phase 2 Retrieval Pipeline Tests

Tests:
    1. Query normalizer: contraction expansion + stop word removal
    2. Intent detection: causal, temporal, spatial, general
    3. Retrieval profiles: correct profile for each intent
    4. Working-memory tracking in graph.py
    5. Precomputed neighborhoods
    6. Multi-signal reranker: scoring and ordering
    7. Reranker hub penalty suppression
    8. Reranker working-memory bias boost
    9. Profile-guided beam search (end-to-end)
    10. Full Phase 2 pipeline: normalize -> profile -> beam -> rerank
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_1_query_normalizer():
    """Query normalizer: contractions + stop words."""
    print("=" * 70)
    print("  TEST 1: Query Normalizer")
    print("=" * 70)

    from kos.query_normalizer import normalize

    r = normalize("What's the population of Toronto?")
    print(f"  Raw: '{r['raw']}'")
    print(f"  Normalized: '{r['normalized']}'")
    print(f"  Content words: {r['content_words']}")

    t1 = "what's" not in r["normalized"]  # Contraction expanded
    t2 = "population" in r["content_words"]
    t3 = "toronto" in r["content_words"]
    t4 = "the" not in r["content_words"]  # Stop word removed

    print(f"  Contraction expanded: {'PASS' if t1 else 'FAIL'}")
    print(f"  'population' kept: {'PASS' if t2 else 'FAIL'}")
    print(f"  'toronto' kept: {'PASS' if t3 else 'FAIL'}")
    print(f"  'the' removed: {'PASS' if t4 else 'FAIL'}")

    passed = all([t1, t2, t3, t4])
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_2_intent_detection():
    """Intent detection: causal, temporal, spatial, general."""
    print("\n" + "=" * 70)
    print("  TEST 2: Intent Detection")
    print("=" * 70)

    from kos.query_normalizer import normalize

    cases = [
        ("Why does rain cause flooding?", "causal"),
        ("When was Toronto founded?", "temporal"),
        ("Where is the CN Tower located?", "where"),
        ("How does photosynthesis work?", "how"),
        ("Compare Toronto and Montreal", "compare"),
        ("Tell me about the ocean", "general"),
    ]

    passed = 0
    for query, expected in cases:
        r = normalize(query)
        ok = r["intent"] == expected
        if ok:
            passed += 1
        print(f"  {'PASS' if ok else 'FAIL'}: '{query[:40]}' -> {r['intent']} (expect {expected})")

    total_pass = passed == len(cases)
    print(f"\n  Result: {passed}/{len(cases)} {'PASS' if total_pass else 'FAIL'}")
    return total_pass


def test_3_retrieval_profiles():
    """Retrieval profiles: correct profile for each intent."""
    print("\n" + "=" * 70)
    print("  TEST 3: Retrieval Profiles")
    print("=" * 70)

    from kos.query_normalizer import get_profile, PROFILES

    # Causal profile should filter to CAUSES + TEMPORAL edges
    p = get_profile("causal")
    t1 = p["allowed_edge_types"] is not None
    t2 = 2 in p["allowed_edge_types"]  # CAUSES

    # Default profile allows all edges
    p2 = get_profile("general")
    t3 = p2["allowed_edge_types"] is None

    # Spatial profile should include LOCATED_IN
    p3 = get_profile("where")
    t4 = 11 in p3["allowed_edge_types"]  # LOCATED_IN

    print(f"  Causal has edge filter: {'PASS' if t1 else 'FAIL'}")
    print(f"  Causal includes CAUSES(2): {'PASS' if t2 else 'FAIL'}")
    print(f"  Default allows all types: {'PASS' if t3 else 'FAIL'}")
    print(f"  Spatial includes LOCATED_IN(11): {'PASS' if t4 else 'FAIL'}")

    passed = all([t1, t2, t3, t4])
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_4_working_memory():
    """Working-memory tracking in graph.py."""
    print("\n" + "=" * 70)
    print("  TEST 4: Working Memory")
    print("=" * 70)

    from kos.graph import KOSKernel

    k = KOSKernel(enable_vsa=False)
    k.add_connection("a", "b", 0.9, "A links to B.")
    k.add_connection("b", "c", 0.8, "B links to C.")
    k.add_connection("c", "d", 0.7, "C links to D.")

    # Query multiple times
    k.query(["a"], 5)
    k.query(["b"], 5)
    k.query(["c"], 5)

    wm = k.get_working_memory()
    t1 = "a" in wm
    t2 = "b" in wm
    t3 = "c" in wm
    t4 = len(wm) == 3

    print(f"  Working memory: {wm}")
    print(f"  Contains 'a': {'PASS' if t1 else 'FAIL'}")
    print(f"  Contains 'b': {'PASS' if t2 else 'FAIL'}")
    print(f"  Contains 'c': {'PASS' if t3 else 'FAIL'}")
    print(f"  Length=3: {'PASS' if t4 else 'FAIL'}")

    passed = all([t1, t2, t3, t4])
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_5_precomputed_neighborhoods():
    """Precomputed neighborhoods."""
    print("\n" + "=" * 70)
    print("  TEST 5: Precomputed Neighborhoods")
    print("=" * 70)

    from kos.graph import KOSKernel

    k = KOSKernel(enable_vsa=False)
    k.add_connection("a", "b", 0.9, "A->B")
    k.add_connection("a", "c", 0.8, "A->C")
    k.add_connection("a", "d", 0.7, "A->D")

    # First call computes and caches
    n1 = k.precompute_neighborhood("a")
    # Second call returns cached
    n2 = k.precompute_neighborhood("a")

    t1 = len(n1) == 3
    t2 = n1 == n2  # Same object from cache
    t3 = "a" in k._neighborhoods  # Cached

    print(f"  Neighbors of 'a': {n1}")
    print(f"  Correct count (3): {'PASS' if t1 else 'FAIL'}")
    print(f"  Cache hit (same result): {'PASS' if t2 else 'FAIL'}")
    print(f"  In cache dict: {'PASS' if t3 else 'FAIL'}")

    # Invalidate
    k.invalidate_neighborhoods()
    t4 = "a" not in k._neighborhoods
    print(f"  After invalidate, cache empty: {'PASS' if t4 else 'FAIL'}")

    passed = all([t1, t2, t3, t4])
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_6_reranker_scoring():
    """Multi-signal reranker: scoring and ordering."""
    print("\n" + "=" * 70)
    print("  TEST 6: Multi-Signal Reranker")
    print("=" * 70)

    from kos.graph import KOSKernel
    from kos.reranker import MultiSignalReranker

    k = KOSKernel(enable_vsa=False)
    k.add_connection("toronto", "city", 0.9, "Toronto is a major city.")
    k.add_connection("toronto", "ontario", 0.8, "Toronto is in Ontario.")
    k.add_connection("toronto", "hockey", 0.3, "Toronto has hockey teams.")

    # Simulate beam search results
    results = [
        ("city", 0.9),
        ("ontario", 0.8),
        ("hockey", 0.3),
    ]

    reranker = MultiSignalReranker()
    reranked = reranker.rerank(results, k, ["toronto", "city"])

    print(f"  Original order: {[r[0] for r in results]}")
    print(f"  Reranked order: {[r[0] for r in reranked]}")
    print(f"  Reranked scores: {[(r[0], f'{r[1]:.3f}') for r in reranked]}")

    # City should rank high (query match + high activation)
    t1 = len(reranked) == 3
    t2 = reranked[0][0] == "city"  # Best match for "toronto city" query

    print(f"  Has all results: {'PASS' if t1 else 'FAIL'}")
    print(f"  'city' ranked first: {'PASS' if t2 else 'FAIL'}")

    passed = t1 and t2
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_7_reranker_hub_penalty():
    """Reranker hub penalty suppression."""
    print("\n" + "=" * 70)
    print("  TEST 7: Reranker Hub Penalty")
    print("=" * 70)

    from kos.graph import KOSKernel
    from kos.reranker import MultiSignalReranker

    k = KOSKernel(enable_vsa=False)

    # Create a hub node with many connections (no provenance to isolate hub penalty)
    for i in range(20):
        k.add_connection("hub", f"t_{i}", 0.5)

    # Create a focused node with few connections (also no provenance)
    k.add_connection("focused", "target_x", 0.5)
    k.add_connection("focused", "target_y", 0.5)

    results = [("hub", 0.8), ("focused", 0.8)]  # Same activation
    reranker = MultiSignalReranker()
    reranked = reranker.rerank(results, k, ["something_unrelated"])

    scores = dict(reranked)
    print(f"  Hub score: {scores.get('hub', 0):.4f} (degree=20)")
    print(f"  Focused score: {scores.get('focused', 0):.4f} (degree=2)")

    # Hub penalty should make hub score lower than focused
    t1 = scores.get("focused", 0) > scores.get("hub", 0)
    print(f"  Focused > Hub: {'PASS' if t1 else 'FAIL'}")

    print(f"\n  Result: {'PASS' if t1 else 'FAIL'}")
    return t1


def test_8_reranker_wm_bias():
    """Reranker working-memory bias boost."""
    print("\n" + "=" * 70)
    print("  TEST 8: Working-Memory Bias in Reranker")
    print("=" * 70)

    from kos.graph import KOSKernel
    from kos.reranker import MultiSignalReranker

    k = KOSKernel(enable_vsa=False)
    k.add_connection("a", "b", 0.7, "A links B.")
    k.add_connection("a", "c", 0.7, "A links C.")

    results = [("b", 0.7), ("c", 0.7)]  # Same activation

    reranker = MultiSignalReranker()

    # Without working memory
    r1 = reranker.rerank(results, k, ["a"])

    # With working memory containing 'b'
    r2 = reranker.rerank(results, k, ["a"], working_memory=["b"])

    score_b_no_wm = dict(r1).get("b", 0)
    score_b_with_wm = dict(r2).get("b", 0)

    t1 = score_b_with_wm > score_b_no_wm
    print(f"  'b' score without WM: {score_b_no_wm:.4f}")
    print(f"  'b' score with WM:    {score_b_with_wm:.4f}")
    print(f"  WM boost applied: {'PASS' if t1 else 'FAIL'}")

    print(f"\n  Result: {'PASS' if t1 else 'FAIL'}")
    return t1


def test_9_profile_guided_beam():
    """Profile-guided beam search (end-to-end)."""
    print("\n" + "=" * 70)
    print("  TEST 9: Profile-Guided Beam Search")
    print("=" * 70)

    from kos.graph import KOSKernel
    from kos.query_normalizer import normalize, get_profile

    k = KOSKernel(enable_vsa=False)

    # Build a mixed graph
    k.add_connection("rain", "flood", 0.9, "Rain causes flooding.", edge_type=2)
    k.add_connection("rain", "weather", 0.9, "Rain is a type of weather.", edge_type=1)
    k.add_connection("flood", "damage", 0.8, "Flooding causes damage.", edge_type=2)
    k.add_connection("rain", "cloud", 0.7, "Rain falls from clouds.", edge_type=3)

    # Causal query should prefer CAUSES edges
    q = normalize("Why does rain cause flooding?")
    profile = get_profile(q["intent"])
    print(f"  Query: '{q['raw']}'")
    print(f"  Intent: {q['intent']}, Profile: {q['profile']}")
    print(f"  Allowed edge types: {profile['allowed_edge_types']}")

    results = k.query_beam(
        ["rain"],
        top_k=profile["top_k"],
        beam_width=profile["beam_width"],
        max_depth=profile["max_depth"],
        allowed_edge_types=profile["allowed_edge_types"],
    )
    names = [r[0] for r in results]
    print(f"  Results: {names}")

    t1 = "flood" in names  # Causal neighbor
    t2 = "weather" not in names  # IS_A should be filtered
    t3 = "cloud" not in names    # PART_OF should be filtered

    print(f"  'flood' found (CAUSES): {'PASS' if t1 else 'FAIL'}")
    print(f"  'weather' excluded (IS_A): {'PASS' if t2 else 'FAIL'}")
    print(f"  'cloud' excluded (PART_OF): {'PASS' if t3 else 'FAIL'}")

    passed = t1 and t2 and t3
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_10_full_phase2_pipeline():
    """Full Phase 2: normalize -> profile -> beam -> rerank."""
    print("\n" + "=" * 70)
    print("  TEST 10: Full Phase 2 Pipeline")
    print("=" * 70)

    from kos.graph import KOSKernel
    from kos.query_normalizer import normalize, get_profile
    from kos.reranker import MultiSignalReranker

    k = KOSKernel(enable_vsa=False)

    # Build knowledge graph
    edges = [
        ("toronto", "city", 0.9, "Toronto is a major city."),
        ("toronto", "ontario", 0.85, "Toronto is located in Ontario."),
        ("toronto", "population", 0.8, "Toronto has 2.9 million people."),
        ("toronto", "cn_tower", 0.7, "Toronto has the CN Tower."),
        ("ontario", "canada", 0.9, "Ontario is in Canada."),
        ("toronto", "hockey", 0.4, "Toronto has hockey."),
    ]
    for src, tgt, w, text in edges:
        k.add_connection(src, tgt, w, text)

    # Step 1: Normalize query
    q = normalize("Where is Toronto located?")
    print(f"  Query: '{q['raw']}'")
    print(f"  Intent: {q['intent']}, Content: {q['content_words']}")

    # Step 2: Get retrieval profile
    profile = get_profile(q["intent"])
    print(f"  Profile: beam_width={profile['beam_width']}, depth={profile['max_depth']}")

    # Step 3: Beam search with profile
    results = k.query_beam(
        ["toronto"],
        top_k=profile["top_k"],
        beam_width=profile["beam_width"],
        max_depth=profile["max_depth"],
        allowed_edge_types=profile["allowed_edge_types"],
    )
    print(f"  Beam results: {[r[0] for r in results]}")

    # Step 4: Rerank
    reranker = MultiSignalReranker()
    wm = k.get_working_memory()
    reranked = reranker.rerank(results, k, q["content_words"], wm)
    print(f"  Reranked: {[(r[0], f'{r[1]:.3f}') for r in reranked[:5]]}")

    t1 = len(reranked) > 0
    # Ontario should rank high for "where is Toronto located"
    top_names = [r[0] for r in reranked[:3]]
    t2 = "ontario" in top_names or "canada" in top_names

    print(f"  Has results: {'PASS' if t1 else 'FAIL'}")
    print(f"  Location node in top 3: {'PASS' if t2 else 'FAIL'}")

    passed = t1 and t2
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  KOS V8.0 -- PHASE 2 RETRIEVAL PIPELINE TESTS")
    print("#  Query Normalizer, Profiles, Working Memory, Reranker")
    print("#" * 70)

    results = []
    results.append(("Query normalizer", test_1_query_normalizer()))
    results.append(("Intent detection", test_2_intent_detection()))
    results.append(("Retrieval profiles", test_3_retrieval_profiles()))
    results.append(("Working memory", test_4_working_memory()))
    results.append(("Precomputed neighborhoods", test_5_precomputed_neighborhoods()))
    results.append(("Multi-signal reranker", test_6_reranker_scoring()))
    results.append(("Hub penalty (reranker)", test_7_reranker_hub_penalty()))
    results.append(("WM bias (reranker)", test_8_reranker_wm_bias()))
    results.append(("Profile-guided beam", test_9_profile_guided_beam()))
    results.append(("Full Phase 2 pipeline", test_10_full_phase2_pipeline()))

    print("\n" + "=" * 70)
    print("  V8.0 PHASE 2 -- FINAL RESULTS")
    print("=" * 70)
    for name, passed in results:
        print(f"  {name:30s} {'PASS' if passed else 'FAIL'}")

    total = sum(1 for _, p in results if p)
    print(f"\n  Total: {total}/{len(results)}")
    print("=" * 70)
