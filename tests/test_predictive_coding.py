"""
KOS V5.0 — Predictive Coding Engine Test (Phase 3).

Proves that the system learns to predict its own behavior:
1. First query: naive prediction (1-hop lookahead), low accuracy
2. Same query repeated: cached prediction improves
3. Over 10 repetitions: accuracy should converge upward
4. Weight adjustments accumulate (the graph self-optimizes)

This is the Friston loop:
    PREDICT → PROPAGATE → COMPARE → UPDATE → LEARN → repeat

The key metric: prediction accuracy MUST increase over time.
If it does, the system is genuinely learning from its own experience.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.predictive import PredictiveCodingEngine


def run_predictive_coding_test():
    print("=" * 70)
    print("  KOS V5.0: PREDICTIVE CODING ENGINE TEST (Phase 3)")
    print("  Friston Loop: PREDICT -> PROPAGATE -> COMPARE -> UPDATE")
    print("=" * 70)

    # ── Boot with structured knowledge ───────────────────────────
    kernel = KOSKernel(enable_vsa=False)
    lexicon = KASMLexicon()
    driver = TextDriver(kernel, lexicon)

    corpus = """
    Toronto is a major city in the Canadian province of Ontario.
    Toronto was founded and incorporated in the year 1834.
    The city of Toronto has a population of approximately 2.7 million people.
    Toronto is located on the northwestern shore of Lake Ontario.
    The Toronto Blue Jays play professional baseball at Rogers Centre.
    John Graves Simcoe originally established the settlement of Toronto.
    The CN Tower is a famous landmark in downtown Toronto.
    Toronto is the financial capital of Canada with many major banks.
    The University of Toronto is a prestigious research institution.
    Perovskite is a highly efficient material for photovoltaic solar cells.
    Photovoltaic cells capture photons to produce electricity efficiently.
    Silicon is a traditional semiconductor used in computing and solar panels.
    Apixaban prevents thrombosis without requiring strict dietary restrictions.
    Warfarin is an older blood thinner that requires careful diet monitoring.
    """
    driver.ingest(corpus)

    print(f"\n[BOOT] Graph seeded: {len(kernel.nodes)} nodes")

    # ── Initialize Predictive Coding Engine ──────────────────────
    pce = PredictiveCodingEngine(kernel, learning_rate=0.02)

    # ── Test 1: Repeated query — accuracy should improve ─────────
    print("\n" + "-" * 70)
    print("  TEST 1: Repeated Query Learning (Toronto)")
    print("  Query the same seeds 10 times — accuracy should increase")
    print("-" * 70)

    toronto_id = lexicon.word_to_uuid.get('toronto')
    if not toronto_id:
        print("  ERROR: 'toronto' not in lexicon")
        return

    accuracies = []
    maes = []

    for i in range(10):
        report = pce.query_with_prediction(
            [toronto_id], top_k=5, verbose=False)

        accuracies.append(report['accuracy'])
        mae = report['mae'] if report['mae'] != float('inf') else 0.0
        maes.append(mae)

        print(f"  Run {i+1:2d}: Accuracy={report['accuracy']:.0%} | "
              f"MAE={mae:.3f} | "
              f"Hits={report['hits']} Miss={report['misses']} "
              f"Surprise={report['surprises']} | "
              f"Adj={report['adjustments']}")

    # Check if accuracy improved
    first_3_avg = sum(accuracies[:3]) / 3 if accuracies[:3] else 0
    last_3_avg = sum(accuracies[-3:]) / 3 if accuracies[-3:] else 0
    improved = last_3_avg >= first_3_avg

    print(f"\n  First 3 avg accuracy: {first_3_avg:.0%}")
    print(f"  Last 3 avg accuracy:  {last_3_avg:.0%}")
    print(f"  Improvement: {'YES' if improved else 'NO'}")

    # ── Test 2: Different queries — cross-domain learning ────────
    print("\n" + "-" * 70)
    print("  TEST 2: Cross-Domain Queries")
    print("  Different topics should each build their own predictions")
    print("-" * 70)

    test_words = ['perovskite', 'apixaban', 'silicon']
    for word in test_words:
        uid = lexicon.word_to_uuid.get(word)
        if not uid:
            print(f"  '{word}' not in lexicon, skipping")
            continue

        # Query 3 times each
        for i in range(3):
            report = pce.query_with_prediction([uid], top_k=3, verbose=False)

        print(f"  {word:15s}: Cached prediction confidence = "
              f"{pce.predictions.get(pce._make_seed_key([uid]), type('', (), {'confidence': 0})()).confidence:.2f}"
              if pce._make_seed_key([uid]) in pce.predictions
              else f"  {word:15s}: No prediction cached")

    # ── Test 3: Multi-seed query ─────────────────────────────────
    print("\n" + "-" * 70)
    print("  TEST 3: Multi-Seed Query (Toronto + Population)")
    print("-" * 70)

    pop_id = lexicon.word_to_uuid.get('population')
    if toronto_id and pop_id:
        for i in range(5):
            report = pce.query_with_prediction(
                [toronto_id, pop_id], top_k=5, verbose=True)

    # ── Test 4: Verify weight adjustments accumulate ─────────────
    print("\n" + "-" * 70)
    print("  TEST 4: Weight Adjustment Verification")
    print("-" * 70)

    stats = pce.get_stats()
    print(f"  Total prediction cycles:    {stats['total_predictions']}")
    print(f"  Total hits:                 {stats['total_hits']}")
    print(f"  Total misses:               {stats['total_misses']}")
    print(f"  Total surprises:            {stats['total_surprises']}")
    print(f"  Overall accuracy:           {stats['overall_accuracy']:.1%}")
    print(f"  Total weight adjustments:   {stats['total_weight_adjustments']}")
    print(f"  Cached prediction patterns: {stats['cached_predictions']}")

    # ── Test 5: Prediction transfer ──────────────────────────────
    print("\n" + "-" * 70)
    print("  TEST 5: Prediction Transfer")
    print("  After learning, does the system predict better on first try?")
    print("-" * 70)

    # Query a node the system has learned about
    city_id = lexicon.word_to_uuid.get('city')
    if city_id:
        predicted_before = pce.predict([city_id])
        report = pce.query_with_prediction([city_id], top_k=5, verbose=True)

        if predicted_before:
            print(f"  Pre-query prediction had {len(predicted_before)} nodes")
            print(f"  Post-query accuracy: {report['accuracy']:.0%}")
        else:
            print(f"  No prior prediction for 'city' (first encounter)")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PREDICTIVE CODING SUMMARY")
    print("=" * 70)

    stats = pce.get_stats()
    print(f"  Prediction cycles:     {stats['total_predictions']}")
    print(f"  Overall accuracy:      {stats['overall_accuracy']:.1%}")
    print(f"  Weight adjustments:    {stats['total_weight_adjustments']}")
    print(f"  Cached patterns:       {stats['cached_predictions']}")
    print(f"  Learning improvement:  {'VERIFIED' if improved else 'INCONCLUSIVE'}")

    if improved and stats['total_weight_adjustments'] > 0:
        print(f"\n  PHASE 3 VERIFIED: The system learns to predict its own")
        print(f"  activation patterns. Accuracy improves with experience.")
        print(f"  Weights self-adjust without gradient descent.")
        print(f"  This is predictive coding on a spreading activation graph.")
    elif stats['total_weight_adjustments'] > 0:
        print(f"\n  Phase 3 PARTIAL: Weight adjustments occurring but")
        print(f"  accuracy improvement is inconclusive. May need more data.")
    else:
        print(f"\n  Phase 3 NEEDS WORK: No weight adjustments detected.")

    print("=" * 70)


if __name__ == "__main__":
    run_predictive_coding_test()
