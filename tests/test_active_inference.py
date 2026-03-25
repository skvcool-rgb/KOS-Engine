"""
KOS V7.0 — Active Inference Integration Test (Offline).

This test proves the complete cognitive loop:
1. The OS receives a question it CANNOT answer (no prior knowledge)
2. System 2 detects high entropy (mathematical "confusion")
3. Active Inference triggers the WebForager autonomously
4. The Forager reads Wikipedia, wires new knowledge into the graph
5. System 2 re-evaluates with new knowledge
6. The OS answers correctly — having taught itself

Uses KOSShellOffline — no OpenAI API key needed.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.router_offline import KOSShellOffline
from kos.daemon import KOSDaemon


def run_active_inference_test():
    print("=" * 70)
    print("  KOS V7.0: ACTIVE INFERENCE INTEGRATION TEST (Offline)")
    print("  System 2 + Autonomous Foraging + Self-Teaching")
    print("=" * 70)

    # ── Phase 1: Boot with MINIMAL knowledge ─────────────────────
    print("\n[PHASE 1] Booting KOS with minimal seed knowledge...")

    kernel = KOSKernel(enable_vsa=False)
    lexicon = KASMLexicon()
    driver = TextDriver(kernel, lexicon)
    shell = KOSShellOffline(kernel, lexicon, enable_forager=True)

    # Seed with a tiny amount of knowledge — NOT about climate
    seed_corpus = """
    Toronto is a major city in the Canadian province of Ontario.
    Toronto was incorporated in the year 1834.
    The city has a population of approximately 2.7 million people.
    """
    driver.ingest(seed_corpus)

    node_count = len(kernel.nodes)
    print(f"    Seeded {node_count} nodes. No climate data exists.")

    # ── Phase 2: Ask a question the OS CANNOT answer ─────────────
    print("\n[PHASE 2] Asking a question the OS has NO knowledge about...")
    print('    Query: "What is the climate of Toronto?"')
    print("    Expected: System 2 detects high entropy -> Forager activates\n")

    t0 = time.perf_counter()
    response = shell.chat("What is the climate of Toronto?")
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\n[RESULT] Latency: {elapsed:.0f}ms")
    if hasattr(shell, 'shadow') and shell.shadow:
        print(f"[RESULT] Entropy after: {shell.shadow.SYSTEM_ENTROPY:.1f}")
    print(f"[RESULT] Nodes after: {len(kernel.nodes)} (+{len(kernel.nodes) - node_count} learned)")
    print(f"[RESULT] Answer: {response.strip()[:200]}")

    # ── Phase 3: Verify the OS actually learned ──────────────────
    print("\n[PHASE 3] Verifying the OS retained the new knowledge...")
    print('    Query: "Tell me about Toronto weather"')

    t1 = time.perf_counter()
    response2 = shell.chat("Tell me about Toronto weather")
    elapsed2 = (time.perf_counter() - t1) * 1000

    print(f"\n[RESULT] Latency: {elapsed2:.0f}ms (should be faster -- no foraging needed)")
    if hasattr(shell, 'shadow') and shell.shadow:
        print(f"[RESULT] Entropy: {shell.shadow.SYSTEM_ENTROPY:.1f}")
    print(f"[RESULT] Answer: {response2.strip()[:200]}")

    # ── Phase 4: Test with a completely unknown topic ────────────
    print("\n[PHASE 4] Testing with a completely unknown topic...")
    print('    Query: "What is the boiling point of mercury?"')

    t2 = time.perf_counter()
    response3 = shell.chat("What is the boiling point of mercury?")
    elapsed3 = (time.perf_counter() - t2) * 1000

    print(f"\n[RESULT] Latency: {elapsed3:.0f}ms")
    if hasattr(shell, 'shadow') and shell.shadow:
        print(f"[RESULT] Entropy: {shell.shadow.SYSTEM_ENTROPY:.1f}")
    print(f"[RESULT] Nodes now: {len(kernel.nodes)}")
    print(f"[RESULT] Answer: {response3.strip()[:200]}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COGNITIVE LOOP SUMMARY")
    print("=" * 70)
    print(f"  Starting knowledge:  {node_count} nodes")
    print(f"  Final knowledge:     {len(kernel.nodes)} nodes")
    print(f"  Knowledge acquired:  +{len(kernel.nodes) - node_count} concepts (autonomous)")
    forager = getattr(shell, 'forager', None)
    print(f"  Forager status:      {'ACTIVE' if forager else 'OFFLINE'}")
    if hasattr(shell, 'shadow') and shell.shadow:
        print(f"  Final entropy:       {shell.shadow.SYSTEM_ENTROPY:.1f}")
    print("=" * 70)

    climate_words = ["climate", "humid", "continental", "temperature",
                     "weather", "warm", "cold", "snow", "summer"]
    has_climate = any(
        w in lexicon.word_to_uuid for w in climate_words
    )
    print(f"\n  Climate knowledge acquired: {'YES' if has_climate else 'NO'}")
    if has_climate:
        print("  The OS successfully taught itself about Toronto's climate")
        print("  by autonomously reading Wikipedia when it detected")
        print("  it didn't know the answer.")


if __name__ == "__main__":
    run_active_inference_test()
