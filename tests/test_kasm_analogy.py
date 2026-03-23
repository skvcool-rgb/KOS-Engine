"""
KASM Proof-of-Concept: Analogical Reasoning via Hyperdimensional Computing

Demonstrates that KASM can detect structural isomorphism (metaphor/analogy)
between two systems that share ZERO surface-level similarity, using only
algebraic operations on 10,000-dimensional bipolar vectors.

Test case: Solar System <=> Atom
    sun      ↔ nucleus        (both play the "center" role)
    planet   ↔ electron       (both play the "orbiter" role)
    gravity  ↔ electromagnetism (both play the "binding force" role)

The system discovers these mappings through pure algebra — no training,
no embeddings, no neural network.

Reference: Kanerva (2009), "Hyperdimensional Computing"
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kasm.vsa import KASMEngine


def run_analogy_test():
    print("=" * 70)
    print("  KASM Phase 1: Analogical Reasoning Proof-of-Concept")
    print("  Vector Symbolic Architecture — 10,000-D Bipolar Vectors")
    print("=" * 70)

    engine = KASMEngine(dimensions=10_000, seed=42)

    # ── Step 1: NODE — Spawn atomic concepts ─────────────────────────

    print("\n[1] NODE: Spawning atomic concepts...")

    # Domain A: Astrophysics
    engine.node_batch("sun", "planet", "gravity")

    # Domain B: Quantum Physics
    engine.node_batch("nucleus", "electron", "electromagnetism")

    # Structural Roles (domain-independent)
    engine.node_batch("role_center", "role_orbiter", "role_force")

    print(f"    Created 9 atomic vectors in {engine.D}-D space")
    print(f"    Memory: {engine.stats()['memory_mb']} MB")

    # Verify orthogonality — random vectors should have ~0 similarity
    ortho_check = engine.resonate(engine.get("sun"), engine.get("nucleus"))
    print(f"    Orthogonality check: cos(sun, nucleus) = {ortho_check:.4f}  (expect ~0.00)")

    # ── Step 2: BIND — Associate concepts with roles ─────────────────

    print("\n[2] BIND: Associating concepts with structural roles...")

    # Solar System role bindings
    r_sun    = engine.store("r_sun",    engine.bind(engine.get("sun"),    engine.get("role_center")))
    r_planet = engine.store("r_planet", engine.bind(engine.get("planet"), engine.get("role_orbiter")))
    r_grav   = engine.store("r_grav",   engine.bind(engine.get("gravity"),engine.get("role_force")))

    # Atom role bindings
    r_nuc    = engine.store("r_nuc",    engine.bind(engine.get("nucleus"),       engine.get("role_center")))
    r_elec   = engine.store("r_elec",   engine.bind(engine.get("electron"),      engine.get("role_orbiter")))
    r_em     = engine.store("r_em",     engine.bind(engine.get("electromagnetism"), engine.get("role_force")))

    # Verify binding orthogonality — bound vector ≠ either parent
    bind_check = engine.resonate(r_sun, engine.get("sun"))
    print(f"    Binding orthogonality: cos(r_sun, sun) = {bind_check:.4f}  (expect ~0.00)")

    # ── Step 3: SUPERPOSE — Create composite manifolds ───────────────

    print("\n[3] SUPERPOSE: Building composite system representations...")

    solar_system = engine.store("solar_system", engine.superpose(r_sun, r_planet, r_grav))
    atom         = engine.store("atom",         engine.superpose(r_nuc, r_elec, r_em))

    # Verify: solar_system should be similar to its components
    ss_to_rsun   = engine.resonate(solar_system, r_sun)
    ss_to_rplanet = engine.resonate(solar_system, r_planet)
    ss_to_rgrav  = engine.resonate(solar_system, r_grav)

    print(f"    cos(solar_system, r_sun)    = {ss_to_rsun:.4f}   (expect ~0.33)")
    print(f"    cos(solar_system, r_planet) = {ss_to_rplanet:.4f}   (expect ~0.33)")
    print(f"    cos(solar_system, r_grav)   = {ss_to_rgrav:.4f}   (expect ~0.33)")

    # Verify: solar_system should NOT be similar to raw concepts
    ss_to_sun_raw = engine.resonate(solar_system, engine.get("sun"))
    print(f"    cos(solar_system, sun_raw)  = {ss_to_sun_raw:.4f}   (expect ~0.00)")

    # ── Step 4: RESONATE — Direct comparison ─────────────────────────

    print("\n[4] RESONATE: Comparing systems...")

    # Raw noun comparison — should be ~0 (no shared nouns)
    raw_sim = engine.resonate(engine.get("sun"), engine.get("nucleus"))
    print(f"    cos(sun, nucleus)           = {raw_sim:.4f}   (raw nouns: orthogonal)")

    # Direct system comparison
    system_sim = engine.resonate(solar_system, atom)
    print(f"    cos(solar_system, atom)     = {system_sim:.4f}   (structural match!)")
    print(f"    --> Systems share ~{abs(system_sim)*100:.1f}% structural similarity")

    # ── Step 5: THE BREAKTHROUGH — Analogical Mapping ────────────────

    print("\n[5] ANALOGICAL REASONING: Discovering cross-domain mappings...")
    print("    Question: 'What is the Sun of an Atom?'")
    print()

    t0 = time.perf_counter()

    # The mapping vector: bind the two systems together
    # Because bind is self-inverse, the shared roles cancel out,
    # leaving the concept-to-concept correspondences
    mapping = engine.bind(solar_system, atom)

    # Query: unbind "sun" from the mapping to find its analog
    # Math: mapping * sun ≈ (sun*nucleus + planet*electron + gravity*em) * sun
    #     = nucleus + noise  (because sun*sun = 1, other terms stay noisy)
    query_result = engine.unbind(mapping, engine.get("sun"))

    # Cleanup: find the nearest known concept
    matches = engine.cleanup(query_result, threshold=0.05)

    elapsed = (time.perf_counter() - t0) * 1000

    print(f"    Mapping vector computed in {elapsed:.2f} ms")
    print(f"    Query: unbind(mapping, sun) -> closest known concepts:")
    for name, score in matches[:5]:
        marker = " <-- ANSWER" if name == "nucleus" else ""
        print(f"      {name:25s}  similarity = {score:.4f}{marker}")

    # ── Step 6: Full analogy extraction ──────────────────────────────

    print("\n[6] FULL ANALOGY TABLE:")
    print(f"    {'Query':20s} {'Answer':20s} {'Similarity':>10s}  {'Correct':>8s}")
    print("    " + "-" * 62)

    analogy_pairs = [
        ("sun",     "nucleus"),
        ("planet",  "electron"),
        ("gravity", "electromagnetism"),
    ]

    all_correct = True
    for query_name, expected_name in analogy_pairs:
        result_vec = engine.unbind(mapping, engine.get(query_name))
        top_matches = engine.cleanup(result_vec, threshold=0.01)

        # Find the best non-role, non-bound match
        answer = "???"
        score = 0.0
        for name, s in top_matches:
            if not name.startswith("r_") and not name.startswith("role_"):
                if name not in ("solar_system", "atom"):
                    answer = name
                    score = s
                    break

        correct = answer == expected_name
        if not correct:
            all_correct = False
        status = "PASS" if correct else "FAIL"
        print(f"    {query_name:20s} {answer:20s} {score:>10.4f}  {status:>8s}")

    # ── Step 7: Reverse analogy ──────────────────────────────────────

    print("\n[7] REVERSE ANALOGY (Atom -> Solar System):")
    print(f"    {'Query':20s} {'Answer':20s} {'Similarity':>10s}  {'Correct':>8s}")
    print("    " + "-" * 62)

    for query_name, expected_name in [("nucleus", "sun"), ("electron", "planet"), ("electromagnetism", "gravity")]:
        result_vec = engine.unbind(mapping, engine.get(query_name))
        top_matches = engine.cleanup(result_vec, threshold=0.01)

        answer = "???"
        score = 0.0
        for name, s in top_matches:
            if not name.startswith("r_") and not name.startswith("role_"):
                if name not in ("solar_system", "atom"):
                    answer = name
                    score = s
                    break

        correct = answer == expected_name
        if not correct:
            all_correct = False
        status = "PASS" if correct else "FAIL"
        print(f"    {query_name:20s} {answer:20s} {score:>10.4f}  {status:>8s}")

    # ── Step 8: Permute — Sequence encoding ──────────────────────────

    print("\n[8] PERMUTE: Sequence encoding test...")

    dog  = engine.node("dog")
    bite = engine.node("bite")
    man  = engine.node("man")

    # "dog bites man" = permute(dog, 0) + permute(bite, 1) + permute(man, 2)
    s1 = engine.superpose(
        engine.permute(dog, 0),
        engine.permute(bite, 1),
        engine.permute(man, 2)
    )

    # "man bites dog" = permute(man, 0) + permute(bite, 1) + permute(dog, 2)
    s2 = engine.superpose(
        engine.permute(man, 0),
        engine.permute(bite, 1),
        engine.permute(dog, 2)
    )

    seq_sim = engine.resonate(s1, s2)
    print(f"    cos('dog bites man', 'man bites dog') = {seq_sim:.4f}")
    print(f"    --> Sequences are {'DIFFERENT' if abs(seq_sim) < 0.8 else 'SAME'} (expect DIFFERENT)")

    # Same sentence should match itself
    self_sim = engine.resonate(s1, s1)
    print(f"    cos('dog bites man', 'dog bites man') = {self_sim:.4f}  (expect 1.00)")

    # ── Summary ──────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Dimensions:          {engine.D:,}")
    print(f"  Symbols created:     {engine.stats()['symbols']}")
    print(f"  Memory used:         {engine.stats()['memory_mb']} MB")
    print(f"  Analogy mapping:     {'ALL CORRECT' if all_correct else 'SOME FAILURES'}")
    print(f"  Sequence encoding:   {'PASS' if abs(seq_sim) < 0.8 else 'FAIL'}")
    print(f"  ML training used:    ZERO")
    print(f"  Neural networks:     ZERO")
    print("=" * 70)

    if all_correct:
        print("\n  KASM Phase 1 VERIFIED: Analogical reasoning works")
        print("  through pure algebra on bipolar hypervectors.")
        print("  No gradient descent. No backpropagation. No training data.")
        print("  Just math.")


if __name__ == "__main__":
    run_analogy_test()
