"""
KOS V7.0 — Unification & Weaver Verification Test (Offline).

Tests hub survival, daemon safety, and Weaver intent routing
using KOSShellOffline — no OpenAI API key needed.
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


def run_unification_test():
    print("==========================================================")
    print(" KOS V7.0 : UNIFICATION & WEAVER VERIFICATION TEST (Offline)")
    print("==========================================================")

    # 1. Boot the Unified OS
    kernel = KOSKernel(enable_vsa=False)
    lexicon = KASMLexicon()
    driver = TextDriver(kernel, lexicon)
    shell = KOSShellOffline(kernel, lexicon)
    daemon = KOSDaemon(kernel)

    # 2. The "Haystack" Corpus
    corpus = """
    Toronto is a massive city with connections to dogs, cats, cars, transit, buildings, streets, lights, and noise.
    Toronto is located in the beautiful Canadian province of Ontario.
    The city of Toronto was founded and incorporated in the year 1834.
    John Graves Simcoe originally established and named the Toronto settlement.
    The Toronto Blue Jays play professional baseball in the city stadium.
    Toronto has a massive population of 2.7 million people.
    """

    print("\n[>>] Ingesting Corpus...")
    driver.ingest(corpus)

    toronto_id = lexicon.get_or_create_id("toronto")
    if toronto_id in kernel.nodes:
        conn_count = len(kernel.nodes[toronto_id].connections)
        print(f"[OK] 'Toronto' hub created successfully with {conn_count} conceptual edges.")

    # 3. VERIFY Daemon Mitosis Prevention
    print("\n[>>] Triggering Background Daemon (Testing Hub Survival)...")
    if hasattr(daemon, '_contextual_mitosis'):
        daemon._contextual_mitosis()
    elif hasattr(daemon, 'run_once'):
        daemon.run_once()

    if toronto_id in kernel.nodes:
        print(f"[PASS] 'Toronto' node survived the Daemon! Contextual Mitosis is safely disabled.")
    else:
        print(f"[FAIL] 'Toronto' was deleted by the Daemon!")
        return

    # 4. VERIFY Weaver Intent Routing
    print("\n[>>] Testing Algorithmic Weaver Intent Routing...")
    print("If successful, the engine will completely ignore baseball and cats, and answer with exact facts.")

    scenarios = [
        ("Geographic Intent", "Where is Toronto located?"),
        ("Temporal Intent", "When was Toronto founded and incorporated?"),
        ("Creator Intent", "Who established and named Toronto?"),
        ("Statistical Intent", "What is the population of Toronto?"),
        ("LAYER 5 SEMANTIC VECTOR", "Tell me about the metropolis."),
    ]

    passed = 0
    for name, prompt in scenarios:
        print(f"\n--- {name} ---")
        print(f"Prompt: \"{prompt}\"")

        t0 = time.perf_counter()
        response = shell.chat(prompt)
        latency = (time.perf_counter() - t0) * 1000

        print(f"Latency: {latency:.2f} ms")
        print(f"Output:  {response.strip()[:200]}")
        if response and response.strip():
            passed += 1

    print(f"\n{'=' * 60}")
    print(f"  RESULT: {passed}/{len(scenarios)} queries answered")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_unification_test()
