"""
KOS V8.0 -- Phase 3 Retrieval Pipeline Tests

Tests:
    1. Hypothesis forking with no contradictions (single branch)
    2. Hypothesis forking with contradictions (multiple branches)
    3. Hypothesis confidence scoring
    4. Causal lane: trace causal chains
    5. Causal lane: build ordered chain
    6. Temporal lane: timeline construction
    7. Analogical lane: VSA-based analogy finding
    8. Analogical lane: structural mapping
    9. Multi-lane retrieval: combine causal + temporal
    10. Full Phase 3: query -> lanes -> fork -> rerank
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_1_no_contradiction_fork():
    """Single hypothesis when no contradictions exist."""
    print("=" * 70)
    print("  TEST 1: Hypothesis Fork (No Contradictions)")
    print("=" * 70)

    from kos.graph import KOSKernel
    from kos.hypothesis import HypothesisForker

    k = KOSKernel(enable_vsa=False)
    k.add_connection("a", "b", 0.9, "A links B.")
    k.add_connection("a", "c", 0.8, "A links C.")

    results = [("b", 0.9), ("c", 0.8)]
    forker = HypothesisForker()
    hypotheses = forker.fork(results, k)

    t1 = len(hypotheses) == 1
    t2 = hypotheses[0].label == "primary"
    t3 = len(hypotheses[0].evidence) == 2
    t4 = hypotheses[0].confidence > 0.9

    print(f"  Hypotheses: {len(hypotheses)}")
    print(f"  Single primary branch: {'PASS' if t1 and t2 else 'FAIL'}")
    print(f"  All evidence in one branch: {'PASS' if t3 else 'FAIL'}")
    print(f"  Confidence > 0.9: {'PASS' if t4 else 'FAIL'} ({hypotheses[0].confidence:.2f})")

    passed = all([t1, t2, t3, t4])
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_2_contradiction_fork():
    """Multiple hypotheses when contradictions exist."""
    print("\n" + "=" * 70)
    print("  TEST 2: Hypothesis Fork (With Contradictions)")
    print("=" * 70)

    from kos.graph import KOSKernel
    from kos.hypothesis import HypothesisForker

    k = KOSKernel(enable_vsa=False)
    k.add_connection("claim", "good", 0.9, "Claim is good.")
    k.add_connection("claim", "bad", 0.8, "Claim is bad.")

    # Manually add a contradiction
    k.contradictions.append({
        'source': 'claim',
        'existing_target': 'good',
        'new_target': 'bad',
        'type': 'antonym_contradiction',
    })

    results = [("good", 0.9), ("bad", 0.8), ("neutral", 0.5)]
    forker = HypothesisForker()
    hypotheses = forker.fork(results, k)

    t1 = len(hypotheses) >= 2  # "good" and "bad" should be in separate branches

    # Check that "good" and "bad" are NOT in the same hypothesis
    in_same = False
    for h in hypotheses:
        nodes = set(e[0] for e in h.evidence)
        if "good" in nodes and "bad" in nodes:
            in_same = True

    t2 = not in_same

    print(f"  Hypotheses: {len(hypotheses)}")
    for h in hypotheses:
        nodes = [e[0] for e in h.evidence]
        print(f"    {h.label}: {nodes} (conf={h.confidence:.2f})")
    print(f"  Multiple branches: {'PASS' if t1 else 'FAIL'}")
    print(f"  Contradictions separated: {'PASS' if t2 else 'FAIL'}")

    passed = t1 and t2
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_3_hypothesis_confidence():
    """Hypothesis confidence scoring."""
    print("\n" + "=" * 70)
    print("  TEST 3: Hypothesis Confidence")
    print("=" * 70)

    from kos.hypothesis import Hypothesis

    h1 = Hypothesis("strong")
    h1.add_evidence("a", 0.9, "Strong evidence A.")
    h1.add_evidence("b", 0.8, "Strong evidence B.")
    h1.supporting = ["x", "y"]
    h1.compute_confidence()

    h2 = Hypothesis("weak")
    h2.add_evidence("c", 0.3, "Weak evidence C.")
    h2.contradicting = ["x", "y", "z"]
    h2.compute_confidence()

    t1 = h1.confidence > h2.confidence
    t2 = h1.confidence > 0.5
    t3 = h2.confidence < h1.confidence

    print(f"  Strong hypothesis: conf={h1.confidence:.3f}")
    print(f"  Weak hypothesis:   conf={h2.confidence:.3f}")
    print(f"  Strong > Weak: {'PASS' if t1 else 'FAIL'}")

    passed = t1 and t2 and t3
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_4_causal_lane_trace():
    """Causal lane: trace causal chains."""
    print("\n" + "=" * 70)
    print("  TEST 4: Causal Lane Trace")
    print("=" * 70)

    from kos.graph import KOSKernel
    from kos.retrieval_lanes import CausalLane

    k = KOSKernel(enable_vsa=False)
    k.add_connection("rain", "flood", 0.9, "Rain causes flood.", edge_type=2)
    k.add_connection("flood", "damage", 0.8, "Flood causes damage.", edge_type=2)
    k.add_connection("rain", "weather", 0.9, "Rain is weather.", edge_type=1)

    lane = CausalLane()
    results = lane.trace(k, ["rain"], max_depth=5, top_k=10)
    names = [r[0] for r in results]

    t1 = "flood" in names
    t2 = "damage" in names
    t3 = "weather" not in names  # IS_A filtered out

    print(f"  Causal trace: {names}")
    print(f"  Found 'flood': {'PASS' if t1 else 'FAIL'}")
    print(f"  Found 'damage': {'PASS' if t2 else 'FAIL'}")
    print(f"  Excluded 'weather': {'PASS' if t3 else 'FAIL'}")

    passed = t1 and t2 and t3
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_5_causal_chain():
    """Causal lane: build ordered chain."""
    print("\n" + "=" * 70)
    print("  TEST 5: Causal Chain Building")
    print("=" * 70)

    from kos.graph import KOSKernel
    from kos.retrieval_lanes import CausalLane

    k = KOSKernel(enable_vsa=False)
    k.add_connection("spark", "fire", 0.9, "Spark causes fire.", edge_type=2)
    k.add_connection("fire", "smoke", 0.8, "Fire causes smoke.", edge_type=2)
    k.add_connection("smoke", "alarm", 0.7, "Smoke triggers alarm.", edge_type=2)

    lane = CausalLane()
    chain = lane.build_chain(k, "spark", max_depth=5)
    names = [c[0] for c in chain]

    t1 = len(chain) >= 3
    # Chain should be in order: fire, smoke, alarm
    t2 = names == ["fire", "smoke", "alarm"] if len(names) >= 3 else False

    print(f"  Chain: {names}")
    print(f"  Length >= 3: {'PASS' if t1 else 'FAIL'}")
    print(f"  Correct order: {'PASS' if t2 else 'FAIL'}")

    passed = t1 and t2
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_6_temporal_timeline():
    """Temporal lane: timeline construction."""
    print("\n" + "=" * 70)
    print("  TEST 6: Temporal Timeline")
    print("=" * 70)

    from kos.graph import KOSKernel
    from kos.retrieval_lanes import TemporalLane

    k = KOSKernel(enable_vsa=False)
    # Timeline: plan -> build -> launch -> grow
    k.add_connection("launch", "build", 0.8, "Build before launch.", edge_type=9)  # BEFORE
    k.add_connection("launch", "grow", 0.7, "Grow after launch.", edge_type=10)  # AFTER
    k.add_connection("build", "plan", 0.9, "Plan before build.", edge_type=9)  # BEFORE

    lane = TemporalLane()
    timeline = lane.build_timeline(k, "launch", max_depth=3)

    before_names = [n for n, _ in timeline["before"]]
    after_names = [n for n, _ in timeline["after"]]

    t1 = "build" in before_names
    t2 = "grow" in after_names
    t3 = "plan" in before_names  # Transitive: plan before build before launch

    print(f"  Before 'launch': {before_names}")
    print(f"  After 'launch': {after_names}")
    print(f"  'build' before: {'PASS' if t1 else 'FAIL'}")
    print(f"  'grow' after: {'PASS' if t2 else 'FAIL'}")
    print(f"  'plan' before (transitive): {'PASS' if t3 else 'FAIL'}")

    passed = t1 and t2 and t3
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_7_analogical_lane():
    """Analogical lane: VSA-based analogy finding."""
    print("\n" + "=" * 70)
    print("  TEST 7: Analogical Lane (VSA)")
    print("=" * 70)

    from kos_rust import RustKernel

    k = RustKernel(dim=10000, seed=42)
    # Build similar structures
    k.add_connection("sun", "solar_system", 0.9, None, 3)
    k.add_connection("sun", "gravity", 0.8, None, 2)
    k.add_connection("nucleus", "atom", 0.9, None, 3)
    k.add_connection("nucleus", "strong_force", 0.8, None, 2)

    # Check resonate similarity
    sim = k.resonate("sun", "nucleus")
    print(f"  RESONATE(sun, nucleus) = {sim:.4f}")

    # The similarity may be low with random vectors, but the function should work
    t1 = isinstance(sim, float)
    print(f"  Returns float: {'PASS' if t1 else 'FAIL'}")

    # Test with the AnalogicalLane wrapper
    from kos.graph import KOSKernel
    from kos.retrieval_lanes import AnalogicalLane

    gk = KOSKernel(enable_vsa=False)
    gk.add_connection("sun", "solar_system", 0.9, "Sun is center of solar system.")
    gk.add_connection("nucleus", "atom", 0.9, "Nucleus is center of atom.")

    lane = AnalogicalLane()
    mapping = lane.structural_map(gk, "sun", "nucleus")

    t2 = "similarity" in mapping
    t3 = "mappings" in mapping

    print(f"  Structural mapping: {mapping}")
    print(f"  Has similarity: {'PASS' if t2 else 'FAIL'}")
    print(f"  Has mappings: {'PASS' if t3 else 'FAIL'}")

    passed = t1 and t2 and t3
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_8_structural_mapping():
    """Analogical lane: detailed structural mapping."""
    print("\n" + "=" * 70)
    print("  TEST 8: Structural Mapping")
    print("=" * 70)

    from kos.graph import KOSKernel
    from kos.retrieval_lanes import AnalogicalLane

    k = KOSKernel(enable_vsa=False)
    # Solar system structure
    k.add_connection("sun", "planet", 0.9, "Sun has planets.", edge_type=3)
    k.add_connection("sun", "gravity", 0.8, "Sun uses gravity.", edge_type=2)
    # Atom structure (parallel)
    k.add_connection("nucleus", "electron", 0.9, "Nucleus has electrons.", edge_type=3)
    k.add_connection("nucleus", "em_force", 0.8, "Nucleus uses EM force.", edge_type=2)

    lane = AnalogicalLane()
    mapping = lane.structural_map(k, "sun", "nucleus")

    print(f"  Similarity: {mapping['similarity']:.4f}")
    print(f"  Mappings:")
    for src, tgt, sim in mapping["mappings"]:
        print(f"    {src} <-> {tgt} (sim={sim:.2f})")

    # Should map planet<->electron and gravity<->em_force (same edge types)
    mapped_pairs = [(s, t) for s, t, _ in mapping["mappings"]]
    t1 = len(mapped_pairs) >= 2

    # Check that same-type edges are matched
    t2 = ("planet", "electron") in mapped_pairs or ("gravity", "em_force") in mapped_pairs

    print(f"  Has mappings: {'PASS' if t1 else 'FAIL'}")
    print(f"  Type-matched mapping: {'PASS' if t2 else 'FAIL'}")

    passed = t1 and t2
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_9_multi_lane():
    """Multi-lane retrieval: combine causal + temporal."""
    print("\n" + "=" * 70)
    print("  TEST 9: Multi-Lane Retrieval")
    print("=" * 70)

    from kos.graph import KOSKernel
    from kos.retrieval_lanes import CausalLane, TemporalLane

    k = KOSKernel(enable_vsa=False)
    # Causal: drought -> famine -> migration
    k.add_connection("drought", "famine", 0.9, "Drought causes famine.", edge_type=2)
    k.add_connection("famine", "migration", 0.8, "Famine causes migration.", edge_type=2)
    # Temporal: 2020 -> drought -> famine -> 2022
    k.add_connection("drought", "year_2020", 0.7, "Drought started 2020.", edge_type=9)
    k.add_connection("famine", "year_2022", 0.6, "Famine peaked 2022.", edge_type=10)

    causal = CausalLane()
    temporal = TemporalLane()

    c_results = causal.trace(k, ["drought"], max_depth=5, top_k=10)
    t_results = temporal.trace(k, ["drought"], max_depth=5, top_k=10)

    c_names = [r[0] for r in c_results]
    t_names = [r[0] for r in t_results]

    print(f"  Causal results: {c_names}")
    print(f"  Temporal results: {t_names}")

    # Merge results (union with max score)
    combined = {}
    for name, score in c_results + t_results:
        combined[name] = max(combined.get(name, 0), score)
    merged = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    merged_names = [n for n, _ in merged]

    print(f"  Merged results: {merged_names}")

    t1 = "famine" in merged_names
    t2 = len(merged_names) >= 2

    print(f"  'famine' in merged: {'PASS' if t1 else 'FAIL'}")
    print(f"  Multiple results: {'PASS' if t2 else 'FAIL'}")

    passed = t1 and t2
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_10_full_phase3():
    """Full Phase 3: query -> lanes -> fork -> rerank."""
    print("\n" + "=" * 70)
    print("  TEST 10: Full Phase 3 Pipeline")
    print("=" * 70)

    from kos.graph import KOSKernel
    from kos.query_normalizer import normalize, get_profile
    from kos.retrieval_lanes import CausalLane
    from kos.hypothesis import HypothesisForker
    from kos.reranker import MultiSignalReranker

    k = KOSKernel(enable_vsa=False)

    # Build a knowledge graph with causal + contradictory info
    k.add_connection("pollution", "warming", 0.9, "Pollution causes warming.", edge_type=2)
    k.add_connection("warming", "ice_melt", 0.85, "Warming causes ice melt.", edge_type=2)
    k.add_connection("ice_melt", "sea_rise", 0.8, "Ice melt causes sea level rise.", edge_type=2)
    k.add_connection("pollution", "smog", 0.7, "Pollution causes smog.", edge_type=2)
    k.add_connection("warming", "drought", 0.6, "Warming causes drought.", edge_type=2)

    # Step 1: Normalize
    q = normalize("Why does pollution cause sea level rise?")
    print(f"  Intent: {q['intent']}, Profile: {q['profile']}")

    # Step 2: Causal lane
    causal = CausalLane()
    c_results = causal.trace(k, ["pollution"], max_depth=7, top_k=10)
    print(f"  Causal results: {[r[0] for r in c_results]}")

    # Step 3: Hypothesis fork
    forker = HypothesisForker()
    hypotheses = forker.fork(c_results, k)
    print(f"  Hypotheses: {len(hypotheses)}")
    for h in hypotheses:
        print(f"    {h.label}: {[e[0] for e in h.evidence]} conf={h.confidence:.2f}")

    # Step 4: Rerank top hypothesis
    reranker = MultiSignalReranker()
    top_h = hypotheses[0]
    h_results = [(e[0], e[1]) for e in top_h.evidence]
    reranked = reranker.rerank(h_results, k, q["content_words"])
    print(f"  Reranked: {[r[0] for r in reranked]}")

    t1 = "warming" in [r[0] for r in c_results]
    t2 = "ice_melt" in [r[0] for r in c_results]
    t3 = len(hypotheses) >= 1
    t4 = len(reranked) >= 2

    print(f"  Causal found 'warming': {'PASS' if t1 else 'FAIL'}")
    print(f"  Causal found 'ice_melt': {'PASS' if t2 else 'FAIL'}")
    print(f"  Has hypotheses: {'PASS' if t3 else 'FAIL'}")
    print(f"  Reranked results: {'PASS' if t4 else 'FAIL'}")

    passed = all([t1, t2, t3, t4])
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  KOS V8.0 -- PHASE 3 RETRIEVAL PIPELINE TESTS")
    print("#  Hypothesis Forking, Causal/Temporal/Analogical Lanes")
    print("#" * 70)

    results = []
    results.append(("No-contradiction fork", test_1_no_contradiction_fork()))
    results.append(("Contradiction fork", test_2_contradiction_fork()))
    results.append(("Hypothesis confidence", test_3_hypothesis_confidence()))
    results.append(("Causal lane trace", test_4_causal_lane_trace()))
    results.append(("Causal chain build", test_5_causal_chain()))
    results.append(("Temporal timeline", test_6_temporal_timeline()))
    results.append(("Analogical lane (VSA)", test_7_analogical_lane()))
    results.append(("Structural mapping", test_8_structural_mapping()))
    results.append(("Multi-lane retrieval", test_9_multi_lane()))
    results.append(("Full Phase 3 pipeline", test_10_full_phase3()))

    print("\n" + "=" * 70)
    print("  V8.0 PHASE 3 -- FINAL RESULTS")
    print("=" * 70)
    for name, passed in results:
        print(f"  {name:30s} {'PASS' if passed else 'FAIL'}")

    total = sum(1 for _, p in results if p)
    print(f"\n  Total: {total}/{len(results)}")
    print("=" * 70)
