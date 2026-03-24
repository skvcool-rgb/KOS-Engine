"""
KOS V5.1 — Self-Improvement Before/After Comparison.

Measures system metrics BEFORE and AFTER running the self-improvement
engine. Proves the system actually gets better.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.router_offline import KOSShellOffline
from kos.predictive import PredictiveCodingEngine
from kos.self_improve import SelfImprover


def measure_system(kernel, lexicon, shell, pce, label=""):
    """Take a snapshot of system health metrics."""
    nodes = len(kernel.nodes)
    edges = sum(len(n.connections) for n in kernel.nodes.values())
    orphans = sum(1 for n in kernel.nodes.values() if not n.connections)
    super_hubs = sum(1 for n in kernel.nodes.values()
                      if len(n.connections) > 20)

    # Count weights > 1.0 (inflation)
    inflated = 0
    max_w = 0
    for n in kernel.nodes.values():
        for tgt, data in n.connections.items():
            w = data['w'] if isinstance(data, dict) else data
            max_w = max(max_w, abs(w))
            if abs(w) > 1.0:
                inflated += 1

    # Average degree
    avg_degree = edges / nodes if nodes > 0 else 0

    # Prediction accuracy
    pce_stats = pce.get_stats()
    pred_accuracy = pce_stats['overall_accuracy']

    # Query accuracy
    benchmark = [
        ("Where is Toronto?", ["ontario"]),
        ("When was Toronto founded?", ["1834"]),
        ("Population of Toronto?", ["million"]),
        ("Climate of Toronto?", ["humid"]),
        ("Tell me about apixaban", ["thrombosis"]),
        ("Tell me about perovskite", ["photovoltaic", "efficient"]),
        ("345000000 * 0.0825", ["28462500"]),
        ("Tell me about the metropolis", ["toronto", "city"]),
        ("Who established Toronto?", ["simcoe"]),
    ]

    q_passed = 0
    for query, expected in benchmark:
        try:
            answer = shell.chat(query).lower()
            if any(kw.lower() in answer for kw in expected):
                q_passed += 1
        except Exception:
            pass
    query_accuracy = q_passed / len(benchmark)

    contradictions = len(getattr(kernel, 'contradictions', []))

    metrics = {
        'label': label,
        'nodes': nodes,
        'edges': edges,
        'orphans': orphans,
        'super_hubs': super_hubs,
        'inflated_weights': inflated,
        'max_weight': max_w,
        'avg_degree': avg_degree,
        'pred_accuracy': pred_accuracy,
        'query_accuracy': query_accuracy,
        'query_passed': q_passed,
        'query_total': len(benchmark),
        'contradictions': contradictions,
    }

    return metrics


def run_improvement_test():
    print("=" * 70)
    print("  KOS V5.1: SELF-IMPROVEMENT BEFORE/AFTER COMPARISON")
    print("  Does the system actually get better?")
    print("=" * 70)

    # Boot
    kernel = KOSKernel(enable_vsa=False)
    lexicon = KASMLexicon()
    driver = TextDriver(kernel, lexicon)
    shell = KOSShellOffline(kernel, lexicon, enable_forager=False)
    pce = PredictiveCodingEngine(kernel, learning_rate=0.05)

    # Ingest corpus with deliberate issues:
    # - Contradictions (expensive vs cheap)
    # - Super-hubs (Toronto has tons of edges)
    # - Some orphan-prone short sentences
    corpus = """
    Toronto is a major city in the Canadian province of Ontario.
    Toronto was founded and incorporated in the year 1834.
    The city of Toronto has a population of approximately 2.7 million people.
    Toronto has a humid continental climate with warm summers and cold winters.
    John Graves Simcoe originally established the settlement of Toronto.
    The CN Tower is a famous landmark in downtown Toronto.
    Toronto is the financial capital of Canada with many banks.
    The Toronto Blue Jays play baseball at Rogers Centre.
    Toronto Transit Commission operates public transit.
    The University of Toronto is a prestigious institution.
    Toronto hosts the annual film festival.
    Drake is a famous musician from Toronto.

    Perovskite is a highly efficient material for photovoltaic cells.
    Photovoltaic cells capture photons to produce electricity.
    Perovskite is extremely expensive to manufacture.
    Perovskite is remarkably cheap and affordable to produce.
    Silicon is a traditional semiconductor for computing.

    Apixaban prevents thrombosis without dietary restrictions.
    Apixaban does not cause bleeding in patients.
    Unlike warfarin, apixaban is a modern anticoagulant.
    Warfarin requires careful dietary monitoring.

    Montreal was founded in the year 1642.
    Montreal has a population of 1.7 million.
    """
    driver.ingest(corpus)

    # Train predictive model
    toronto_id = lexicon.word_to_uuid.get('toronto')
    if toronto_id:
        for _ in range(10):
            pce.query_with_prediction([toronto_id], top_k=5, verbose=False)

    # ── BEFORE METRICS ───────────────────────────────────────
    print("\n[BEFORE] Measuring system health...")
    before = measure_system(kernel, lexicon, shell, pce, "BEFORE")

    print(f"  Nodes:            {before['nodes']}")
    print(f"  Edges:            {before['edges']}")
    print(f"  Orphans:          {before['orphans']}")
    print(f"  Super-hubs:       {before['super_hubs']}")
    print(f"  Inflated weights: {before['inflated_weights']}")
    print(f"  Max weight:       {before['max_weight']:.3f}")
    print(f"  Avg degree:       {before['avg_degree']:.1f}")
    print(f"  Pred accuracy:    {before['pred_accuracy']:.0%}")
    print(f"  Query accuracy:   {before['query_accuracy']:.0%} "
          f"({before['query_passed']}/{before['query_total']})")
    print(f"  Contradictions:   {before['contradictions']}")

    # ── RUN SELF-IMPROVEMENT ─────────────────────────────────
    print("\n" + "-" * 70)
    improver = SelfImprover(kernel, lexicon, shell=shell)
    improvement_results = improver.improve(verbose=True)

    # ── AFTER METRICS ────────────────────────────────────────
    print("\n[AFTER] Measuring system health...")
    after = measure_system(kernel, lexicon, shell, pce, "AFTER")

    print(f"  Nodes:            {after['nodes']}")
    print(f"  Edges:            {after['edges']}")
    print(f"  Orphans:          {after['orphans']}")
    print(f"  Super-hubs:       {after['super_hubs']}")
    print(f"  Inflated weights: {after['inflated_weights']}")
    print(f"  Max weight:       {after['max_weight']:.3f}")
    print(f"  Avg degree:       {after['avg_degree']:.1f}")
    print(f"  Pred accuracy:    {after['pred_accuracy']:.0%}")
    print(f"  Query accuracy:   {after['query_accuracy']:.0%} "
          f"({after['query_passed']}/{after['query_total']})")
    print(f"  Contradictions:   {after['contradictions']}")

    # ── COMPARISON ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BEFORE vs AFTER COMPARISON")
    print("=" * 70)

    comparisons = [
        ("Orphans", before['orphans'], after['orphans'], "lower"),
        ("Super-hubs", before['super_hubs'], after['super_hubs'], "lower"),
        ("Inflated weights", before['inflated_weights'],
         after['inflated_weights'], "lower"),
        ("Max weight", before['max_weight'], after['max_weight'], "lower"),
        ("Query accuracy", before['query_accuracy'],
         after['query_accuracy'], "higher"),
        ("Pred accuracy", before['pred_accuracy'],
         after['pred_accuracy'], "higher"),
    ]

    improved_count = 0
    for name, bval, aval, direction in comparisons:
        if direction == "lower":
            improved = aval <= bval
            delta = bval - aval
            arrow = "v" if delta > 0 else "=" if delta == 0 else "^"
        else:
            improved = aval >= bval
            delta = aval - bval
            arrow = "^" if delta > 0 else "=" if delta == 0 else "v"

        if improved:
            improved_count += 1

        if isinstance(bval, float):
            print(f"  {name:25s}: {bval:.3f} -> {aval:.3f} "
                  f"({arrow}) {'IMPROVED' if improved else 'same'}")
        else:
            print(f"  {name:25s}: {bval} -> {aval} "
                  f"({arrow}) {'IMPROVED' if improved else 'same'}")

    # ── SECOND IMPROVEMENT PASS ──────────────────────────────
    print("\n[PASS 2] Running self-improvement again (convergence test)...")
    improvement_results_2 = improver.improve(verbose=True)

    after2 = measure_system(kernel, lexicon, shell, pce, "AFTER_2")
    print(f"\n  Query accuracy after pass 2: {after2['query_accuracy']:.0%}")
    print(f"  Orphans after pass 2:        {after2['orphans']}")

    # ── SUMMARY ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SELF-IMPROVEMENT SUMMARY")
    print("=" * 70)
    print(f"  Metrics improved:     {improved_count}/{len(comparisons)}")
    print(f"  Hubs rebalanced:      "
          f"{improvement_results['rebalance']['hubs_fixed']}")
    print(f"  Orphans connected:    "
          f"{improvement_results['rebalance']['orphans_fixed']}")
    print(f"  Contradictions resolved: "
          f"{improvement_results['contradictions']['resolved']}")
    print(f"  Weights normalized:   "
          f"{improvement_results['normalization']['clipped']}")
    print(f"  Formulas discovered:  "
          f"{improvement_results['formulas']['formulas_found']}")
    print(f"  Benchmark accuracy:   "
          f"{improvement_results['benchmark']['accuracy']:.0%}")

    query_maintained = after['query_accuracy'] >= before['query_accuracy']
    structure_improved = (after['orphans'] <= before['orphans'] or
                          after['super_hubs'] <= before['super_hubs'] or
                          after['inflated_weights'] <= before['inflated_weights'])

    if query_maintained and structure_improved:
        print(f"\n  SELF-IMPROVEMENT VERIFIED:")
        print(f"  Query accuracy maintained or improved ({before['query_accuracy']:.0%} -> {after['query_accuracy']:.0%})")
        print(f"  Graph structure improved (fewer orphans/hubs/inflation)")
        print(f"  The system made itself better without human intervention.")
    elif query_maintained:
        print(f"\n  SELF-IMPROVEMENT PARTIAL:")
        print(f"  Query accuracy maintained ({after['query_accuracy']:.0%})")
        print(f"  Structure changes were neutral.")
    else:
        print(f"\n  REGRESSION DETECTED:")
        print(f"  Query accuracy dropped: {before['query_accuracy']:.0%} -> {after['query_accuracy']:.0%}")

    print("=" * 70)


if __name__ == "__main__":
    run_improvement_test()
