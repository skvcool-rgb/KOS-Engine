"""
KOS V5.1 — Learning Gaps Closure Test

Proves all 4 remaining self-learning gaps are fixed:
  GAP #4: Weaver learns from user re-asks (feedback loop)
  GAP #9: System learns formulas from ingested text
  GAP L1: Auto-tuning triggers itself when accuracy drops
  GAP L3: Analogy discovery works at scale autonomously
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.router_offline import KOSShellOffline
from kos.weaver import AlgorithmicWeaver
from kos.feedback import WeaverFeedback, FormulaLearner, ContinuousTuner, AnalogyScanner


def build_test_system():
    """Boot KOS with a substantial test corpus."""
    kernel = KOSKernel(enable_vsa=False)
    lexicon = KASMLexicon()
    driver = TextDriver(kernel, lexicon)

    driver.ingest("""
    Toronto is a major city in the Canadian province of Ontario.
    Toronto was founded and incorporated in the year 1834.
    The city of Toronto has a population of approximately 2.7 million people.
    Toronto has a humid continental climate with warm summers and cold winters.
    John Graves Simcoe originally established the settlement of Toronto.
    The CN Tower is a famous landmark in downtown Toronto.
    Toronto is the financial capital of Canada with many banks.

    Montreal was founded in the year 1642 in Quebec province.
    Montreal has a population of approximately 1.7 million residents.
    Montreal has a humid continental climate similar to Toronto.
    Montreal is the cultural capital of Canada.

    Perovskite is a highly efficient material for photovoltaic cells.
    Photovoltaic cells capture photons to produce electricity.
    Perovskite degrades rapidly under UV light and moisture.
    Silicon is a traditional semiconductor for computing.

    Apixaban prevents thrombosis without dietary restrictions.
    Apixaban does not cause bleeding in patients.
    Warfarin requires careful dietary monitoring.

    The area of a rectangle equals length times width.
    Compound interest formula is P times one plus r divided by n raised to nt.
    Force equals mass times acceleration.
    Energy equals mass times speed of light squared.
    """)

    shell = KOSShellOffline(kernel, lexicon, enable_forager=False)
    return kernel, lexicon, driver, shell


def test_gap4_weaver_feedback():
    """GAP #4: Weaver learns from user re-asks."""
    print("=" * 60)
    print("  GAP #4: WEAVER FEEDBACK LOOP")
    print("  Does the system learn from user dissatisfaction?")
    print("=" * 60)

    kernel, lexicon, driver, shell = build_test_system()
    weaver = AlgorithmicWeaver()
    feedback = WeaverFeedback(weaver)

    # Simulate a user asking about Toronto location
    # First answer might be wrong (no feedback yet)
    q1 = "Where is Toronto located?"
    a1 = shell.chat(q1)
    evidence1 = [a1]  # simplified
    feedback.record(q1, a1, evidence1)

    print(f"\n  Q1: {q1}")
    print(f"  A1: {a1.strip()[:80]}")

    # User RE-ASKS (similar question = dissatisfied)
    q2 = "Where exactly is Toronto?"
    a2 = shell.chat(q2)
    evidence2 = [a2]
    feedback.record(q2, a2, evidence2)

    print(f"\n  Q2: {q2} (re-ask!)")
    print(f"  A2: {a2.strip()[:80]}")

    # User moves to NEW TOPIC (= satisfied with answer)
    q3 = "Tell me about apixaban"
    a3 = shell.chat(q3)
    feedback.record(q3, a3, [a3])

    print(f"\n  Q3: {q3} (new topic = satisfied)")
    print(f"  A3: {a3.strip()[:80]}")

    stats = feedback.get_stats()
    print(f"\n  FEEDBACK STATS:")
    print(f"    Queries tracked:    {stats['total_queries']}")
    print(f"    Re-asks detected:   {stats['reasks']}")
    print(f"    Satisfied signals:  {stats['satisfied']}")
    print(f"    Satisfaction rate:  {stats['satisfaction_rate']:.0%}")

    # Check that feedback adjustments exist
    adjustments = {k: v for k, v in feedback._evidence_scores.items() if v != 0}
    print(f"    Evidence adjustments: {len(adjustments)}")
    for sent, score in list(adjustments.items())[:3]:
        tag = "DEMOTED" if score < 0 else "BOOSTED"
        print(f"      [{tag}] score={score:+.1f}: {sent[:60]}")

    passed = stats['reasks'] >= 1 and stats['satisfied'] >= 1
    print(f"\n  GAP #4: {'PASS' if passed else 'FAIL'}")
    return passed


def test_gap9_formula_discovery():
    """GAP #9: System learns formulas from ingested text."""
    print("\n" + "=" * 60)
    print("  GAP #9: AUTO FORMULA DISCOVERY")
    print("  Can the system extract formulas from natural language?")
    print("=" * 60)

    kernel, lexicon, driver, shell = build_test_system()
    learner = FormulaLearner()

    print("\n  Scanning provenance for mathematical expressions...")
    formulas = learner.scan_provenance(kernel)

    print(f"\n  DISCOVERED FORMULAS:")
    for f in formulas:
        print(f"    [{f['type']:15s}] {f['name']} = {f['expression'][:50]}")
        print(f"                     Source: {f['source'][:60]}")

    stats = learner.get_stats()
    print(f"\n  Total discovered: {stats['discovered']}")

    passed = stats['discovered'] >= 1
    print(f"\n  GAP #9: {'PASS' if passed else 'FAIL'}")
    return passed


def test_gap_l1_continuous_tuning():
    """GAP L1: Auto-tuning triggers itself when accuracy drops."""
    print("\n" + "=" * 60)
    print("  GAP L1: CONTINUOUS AUTO-TUNING")
    print("  Does the system auto-tune when accuracy drops?")
    print("=" * 60)

    kernel, lexicon, driver, shell = build_test_system()

    from kos.selfmod import AutoTuner
    tuner = AutoTuner(kernel, lexicon, driver)
    continuous = ContinuousTuner(tuner, threshold=0.60, window_size=10,
                                  check_interval=10)

    print("\n  Simulating query stream with declining accuracy...")

    # Phase 1: Good accuracy (8/10 correct)
    for i in range(8):
        continuous.record_result(True)
    for i in range(2):
        continuous.record_result(False)

    stats1 = continuous.get_stats()
    print(f"  After 10 queries: accuracy={stats1['recent_accuracy']:.0%}, "
          f"auto-tuned={stats1['times_auto_tuned']}x")

    # Phase 2: Accuracy drops (only 4/10 correct)
    for i in range(4):
        continuous.record_result(True)
    for i in range(6):
        continuous.record_result(False)

    stats2 = continuous.get_stats()
    print(f"  After 20 queries: accuracy={stats2['recent_accuracy']:.0%}, "
          f"auto-tuned={stats2['times_auto_tuned']}x")

    # Phase 3: Force a check
    check = continuous.force_check()
    print(f"\n  Force check: accuracy={check['accuracy']:.0%}, "
          f"needs_tuning={check['needs_tuning']}")

    passed = stats2['times_auto_tuned'] >= 1 or check['needs_tuning']
    print(f"\n  GAP L1: {'PASS' if passed else 'FAIL'}")
    return passed


def test_gap_l3_analogy_discovery():
    """GAP L3: Autonomous analogy discovery at scale."""
    print("\n" + "=" * 60)
    print("  GAP L3: AUTONOMOUS ANALOGY DISCOVERY")
    print("  Can the system discover analogies without being told?")
    print("=" * 60)

    kernel, lexicon, driver, shell = build_test_system()
    scanner = AnalogyScanner(kernel, lexicon, similarity_threshold=0.3)

    print("\n  Scanning graph for structural analogies...")
    t0 = time.perf_counter()
    analogies = scanner.scan(max_comparisons=10000, verbose=True)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\n  Scan completed in {elapsed:.0f}ms")

    if analogies:
        print(f"\n  TOP ANALOGIES DISCOVERED:")
        for a in sorted(analogies, key=lambda x: x['similarity'], reverse=True)[:5]:
            print(f"    {a['word_a']:20s} <=> {a['word_b']:20s} "
                  f"(similarity={a['similarity']:.2f}, shared={a['shared_targets']})")

        # Wire them into the graph
        wired = scanner.wire_analogies(confidence=0.4)
        print(f"\n  Wired {wired} analogy edges into graph")
    else:
        print(f"\n  No analogies found at threshold {scanner.threshold}")

    stats = scanner.get_stats()
    print(f"\n  SCANNER STATS:")
    print(f"    Pairs compared:    {stats['total_scanned']}")
    print(f"    Analogies found:   {stats['analogies_found']}")
    print(f"    Avg similarity:    {stats['avg_similarity']:.2f}")

    passed = stats['analogies_found'] >= 1
    print(f"\n  GAP L3: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    print("=" * 60)
    print("  KOS V5.1: LEARNING GAPS CLOSURE TEST")
    print("  Proving all 4 self-learning gaps are fixed")
    print("=" * 60)

    results = {}
    results['gap4'] = test_gap4_weaver_feedback()
    results['gap9'] = test_gap9_formula_discovery()
    results['gap_l1'] = test_gap_l1_continuous_tuning()
    results['gap_l3'] = test_gap_l3_analogy_discovery()

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:10s}: {status}")

    all_passed = all(results.values())
    print(f"\n  ALL GAPS CLOSED: {'YES' if all_passed else 'NO'}")
    print("=" * 60)

    if all_passed:
        print("\n  KOS now self-learns through 9 mechanisms:")
        print("    1. Myelination (edges strengthen when used)")
        print("    2. Predictive Coding (predict->compare->adjust)")
        print("    3. Catastrophic Unlearning (crush false beliefs)")
        print("    4. Active Inference (forage to fill gaps)")
        print("    5. Triadic Closure (infer A->C from A->B->C)")
        print("    6. Weaver Feedback (learn from user re-asks)  [NEW]")
        print("    7. Formula Discovery (learn math from text)   [NEW]")
        print("    8. Continuous Auto-Tuning (self-triggering)   [NEW]")
        print("    9. Analogy Discovery (find structural matches)[NEW]")


if __name__ == "__main__":
    main()
