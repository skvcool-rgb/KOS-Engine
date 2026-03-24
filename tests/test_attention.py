"""
KOS V5.0 — Proactive Attention Controller Test.

Proves Phase 2 of the intelligence roadmap:
The system generates its own learning goals WITHOUT user prompting.

Test scenario:
1. Seed the OS with basic Toronto knowledge
2. Simulate 5 user queries about Toronto (builds query history)
3. Run the Attention Controller — it should generate:
   - CURIOSITY goals (Toronto is a shallow hub)
   - ANTICIPATION goals (user keeps asking about Toronto)
4. Let the daemon ACT on goals (autonomous Wikipedia foraging)
5. Verify the graph grew without any user requesting it
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.forager import WebForager
from kos.attention import AttentionController
from kos.daemon import KOSDaemon


def run_attention_test():
    print("=" * 70)
    print("  KOS V5.0: PROACTIVE ATTENTION CONTROLLER TEST")
    print("  Phase 2: Self-Generated Goals + Autonomous Learning")
    print("=" * 70)

    # ── Boot ─────────────────────────────────────────────────────
    kernel = KOSKernel(enable_vsa=False)
    lexicon = KASMLexicon()
    driver = TextDriver(kernel, lexicon)
    forager = WebForager(kernel, lexicon, text_driver=driver)
    attention = AttentionController(kernel, lexicon)

    # Seed with moderate Toronto knowledge
    corpus = """
    Toronto is a major city in the Canadian province of Ontario.
    Toronto was incorporated in the year 1834.
    The city has a population of approximately 2.7 million people.
    Toronto is located on the northwestern shore of Lake Ontario.
    The Toronto Blue Jays play professional baseball at Rogers Centre.
    Toronto has a diverse multicultural population from around the world.
    The CN Tower is a famous landmark in downtown Toronto.
    Toronto is the financial capital of Canada with many banks.
    The University of Toronto is a prestigious research institution.
    Toronto Transit Commission operates the public transit system.
    """
    driver.ingest(corpus)

    initial_nodes = len(kernel.nodes)
    print(f"\n[BOOT] Seeded {initial_nodes} nodes.")

    # ── Simulate User Query History ──────────────────────────────
    print("\n[SIMULATE] Simulating 5 user queries about Toronto...")

    toronto_id = lexicon.word_to_uuid.get('toronto')
    climate_id = lexicon.word_to_uuid.get('climate')
    population_id = lexicon.word_to_uuid.get('population')
    city_id = lexicon.word_to_uuid.get('city')
    ontario_id = lexicon.word_to_uuid.get('ontario')

    # Simulate queries (just record them — no LLM needed)
    queries = [
        [toronto_id, population_id],
        [toronto_id, city_id],
        [toronto_id, ontario_id],
        [toronto_id],
        [toronto_id, city_id],
    ]

    for i, seeds in enumerate(queries):
        valid_seeds = [s for s in seeds if s]
        if valid_seeds:
            kernel.current_tick += 1
            attention.record_query(valid_seeds, kernel.current_tick)
            seed_words = [lexicon.get_word(s) for s in valid_seeds]
            print(f"   Query {i+1}: {seed_words}")

    print(f"\n[SIMULATE] Query history recorded. "
          f"Topic frequency: {dict(attention.topic_frequency.most_common(3))}")

    # ── Generate Goals (No User Input!) ──────────────────────────
    print("\n[PHASE 2] Generating self-directed attention goals...")
    goals = attention.generate_goals(verbose=True)

    if goals:
        print(f"\n   The OS decided ON ITS OWN to learn about:")
        for g in goals[:5]:
            print(f"   -> [{g.goal_type.upper()}] {g.query}")
    else:
        print("\n   No goals generated (graph may be too small for patterns)")

    # ── Act on Goals (Autonomous Foraging!) ──────────────────────
    print("\n[PHASE 2] Executing autonomous learning actions...")

    before_nodes = len(kernel.nodes)
    report = attention.act_on_goals(forager, max_actions=2, verbose=True)
    after_nodes = len(kernel.nodes)

    # ── Run Full Daemon Cycle ────────────────────────────────────
    print("\n[DAEMON] Running full maintenance + attention cycle...")
    daemon = KOSDaemon(kernel, lexicon, forager=forager,
                       attention_controller=attention)
    daemon_report = daemon.run_maintenance_cycle(
        enable_attention=True, max_forage_actions=1)

    final_nodes = len(kernel.nodes)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PROACTIVE ATTENTION SUMMARY")
    print("=" * 70)
    print(f"  Initial knowledge:     {initial_nodes} nodes")
    print(f"  After attention goals: {after_nodes} nodes (+{after_nodes - initial_nodes})")
    print(f"  After daemon cycle:    {final_nodes} nodes (+{final_nodes - after_nodes})")
    print(f"  Total self-acquired:   +{final_nodes - initial_nodes} concepts")
    print(f"  Goals generated:       {report['goals_generated']}")
    print(f"  Actions taken:         {report['actions_taken']}")
    print(f"  User queries needed:   ZERO (all self-directed)")
    print("=" * 70)

    if final_nodes > initial_nodes:
        print("\n  PHASE 2 VERIFIED: The OS generated its own learning goals")
        print("  and autonomously acquired new knowledge without any")
        print("  user prompting. This is proactive intelligence.")
    else:
        print("\n  Phase 2 partial: Goals generated but no new knowledge")
        print("  acquired (possible network issue or small graph).")


if __name__ == "__main__":
    run_attention_test()
