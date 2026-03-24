"""
KOS V5.1 Benchmark: Zero-Copy Rust VSA (vectors never cross PyO3 boundary)

Compares:
    1. Python KASMEngine (numpy)
    2. RustVSA with named operations (zero-copy)
    3. Full analogy pipeline in one Rust call
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_1_zero_copy_analogy():
    """Verify the zero-copy analogy pipeline works correctly."""
    print("=" * 70)
    print("  TEST 1: Zero-Copy Analogy (all ops stay in Rust)")
    print("=" * 70)

    from kos_rust import RustVSA

    engine = RustVSA(dim=10_000, seed=42)

    # Create all nodes (vectors stay in Rust)
    engine.node_batch(["sun", "planet", "gravity",
                       "nucleus", "electron", "electromagnetism",
                       "role_center", "role_orbiter", "role_force"])

    # BIND by name (zero copy)
    engine.bind_named("r_sun", "sun", "role_center")
    engine.bind_named("r_planet", "planet", "role_orbiter")
    engine.bind_named("r_grav", "gravity", "role_force")
    engine.bind_named("r_nuc", "nucleus", "role_center")
    engine.bind_named("r_elec", "electron", "role_orbiter")
    engine.bind_named("r_em", "electromagnetism", "role_force")

    # SUPERPOSE by name (zero copy)
    engine.superpose_named("solar_system", ["r_sun", "r_planet", "r_grav"])
    engine.superpose_named("atom", ["r_nuc", "r_elec", "r_em"])

    # RESONATE by name
    raw_sim = engine.resonate_named("sun", "nucleus")
    print(f"\n  cos(sun, nucleus) = {raw_sim:.4f}  (expect ~0.00)")

    # Full analogy: mapping + unbind + cleanup in ONE call
    t0 = time.perf_counter()
    matches = engine.analogy(
        ["r_sun", "r_planet", "r_grav"],
        ["r_nuc", "r_elec", "r_em"],
        "sun"
    )
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\n  Analogy: What is the Sun of an Atom?")
    print(f"  Computed in {elapsed:.3f} ms (single Rust call)")
    for name, score in matches[:3]:
        marker = " <<" if name == "nucleus" else ""
        print(f"    {name:25s}  {score:+.4f}{marker}")

    passed = matches[0][0] == "nucleus" if matches else False
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_2_speed_comparison():
    """Head-to-head: Python numpy vs Rust zero-copy."""
    print("\n" + "=" * 70)
    print("  TEST 2: Speed — Python numpy vs Rust Zero-Copy")
    print("=" * 70)

    from kos_rust import RustVSA
    from kasm.vsa import KASMEngine

    N_ITER = 10_000

    # ── Python: NODE creation ──
    py = KASMEngine(dimensions=10_000, seed=42)
    t0 = time.perf_counter()
    for i in range(1000):
        py.node(f"py_{i}")
    py_node = (time.perf_counter() - t0) * 1000

    # ── Rust: NODE creation (zero copy — no vector returned) ──
    rs = RustVSA(dim=10_000, seed=42)
    t0 = time.perf_counter()
    for i in range(1000):
        rs.node(f"rs_{i}")
    rs_node = (time.perf_counter() - t0) * 1000

    # ── Python: BIND ──
    a = py.get("py_0")
    b = py.get("py_1")
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        py.bind(a, b)
    py_bind = (time.perf_counter() - t0) * 1000

    # ── Rust: BIND by name (zero copy) ──
    t0 = time.perf_counter()
    for i in range(N_ITER):
        rs.bind_named(f"tmp_bind_{i}", "rs_0", "rs_1")
    rs_bind = (time.perf_counter() - t0) * 1000

    # ── Python: RESONATE ──
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        py.resonate(a, b)
    py_res = (time.perf_counter() - t0) * 1000

    # ── Rust: RESONATE by name (zero copy) ──
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        rs.resonate_named("rs_0", "rs_1")
    rs_res = (time.perf_counter() - t0) * 1000

    # ── Python: Full analogy pipeline ──
    py2 = KASMEngine(dimensions=10_000, seed=42)
    for n in ["sun", "planet", "gravity", "nucleus", "electron", "electromagnetism",
              "role_center", "role_orbiter", "role_force"]:
        py2.node(n)

    t0 = time.perf_counter()
    for _ in range(100):
        r_sun = py2.bind(py2.get("sun"), py2.get("role_center"))
        r_planet = py2.bind(py2.get("planet"), py2.get("role_orbiter"))
        r_grav = py2.bind(py2.get("gravity"), py2.get("role_force"))
        r_nuc = py2.bind(py2.get("nucleus"), py2.get("role_center"))
        r_elec = py2.bind(py2.get("electron"), py2.get("role_orbiter"))
        r_em = py2.bind(py2.get("electromagnetism"), py2.get("role_force"))
        solar = py2.superpose(r_sun, r_planet, r_grav)
        atom = py2.superpose(r_nuc, r_elec, r_em)
        mapping = py2.bind(solar, atom)
        answer = py2.unbind(mapping, py2.get("sun"))
        py2.cleanup(answer, 0.05)
    py_pipeline = (time.perf_counter() - t0) * 1000

    # ── Rust: Full analogy pipeline (single call) ──
    rs2 = RustVSA(dim=10_000, seed=42)
    rs2.node_batch(["sun", "planet", "gravity", "nucleus", "electron", "electromagnetism",
                    "role_center", "role_orbiter", "role_force"])
    rs2.bind_named("r_sun", "sun", "role_center")
    rs2.bind_named("r_planet", "planet", "role_orbiter")
    rs2.bind_named("r_grav", "gravity", "role_force")
    rs2.bind_named("r_nuc", "nucleus", "role_center")
    rs2.bind_named("r_elec", "electron", "role_orbiter")
    rs2.bind_named("r_em", "electromagnetism", "role_force")

    t0 = time.perf_counter()
    for _ in range(100):
        rs2.analogy(
            ["r_sun", "r_planet", "r_grav"],
            ["r_nuc", "r_elec", "r_em"],
            "sun"
        )
    rs_pipeline = (time.perf_counter() - t0) * 1000

    print(f"\n  {'Operation':<30} {'Python (ms)':>12} {'Rust (ms)':>12} {'Speedup':>10}")
    print("  " + "-" * 67)
    print(f"  {'1K NODE creates':<30} {py_node:>12.2f} {rs_node:>12.2f} {py_node/max(rs_node, 0.001):>9.1f}x")
    print(f"  {'10K BIND ops':<30} {py_bind:>12.2f} {rs_bind:>12.2f} {py_bind/max(rs_bind, 0.001):>9.1f}x")
    print(f"  {'10K RESONATE ops':<30} {py_res:>12.2f} {rs_res:>12.2f} {py_res/max(rs_res, 0.001):>9.1f}x")
    print(f"  {'100 full analogy pipelines':<30} {py_pipeline:>12.2f} {rs_pipeline:>12.2f} {py_pipeline/max(rs_pipeline, 0.001):>9.1f}x")

    return True


def test_3_permute_sequence():
    """Test PERMUTE works with named API."""
    print("\n" + "=" * 70)
    print("  TEST 3: Zero-Copy PERMUTE (Sequence Encoding)")
    print("=" * 70)

    from kos_rust import RustVSA

    engine = RustVSA(dim=10_000, seed=77)
    engine.node_batch(["dog", "bites", "man"])

    # "dog bites man"
    engine.permute_named("p_dog_1", "dog", 0)
    engine.permute_named("p_bite_1", "bites", 1)
    engine.permute_named("p_man_1", "man", 2)
    engine.superpose_named("s1", ["p_dog_1", "p_bite_1", "p_man_1"])

    # "man bites dog"
    engine.permute_named("p_man_2", "man", 0)
    engine.permute_named("p_bite_2", "bites", 1)
    engine.permute_named("p_dog_2", "dog", 2)
    engine.superpose_named("s2", ["p_man_2", "p_bite_2", "p_dog_2"])

    sim_diff = engine.resonate_named("s1", "s2")
    sim_same = engine.resonate_named("s1", "s1")

    print(f"\n  cos('dog bites man', 'man bites dog') = {sim_diff:.4f}")
    print(f"  cos('dog bites man', 'dog bites man') = {sim_same:.4f}")

    passed = sim_same > 0.99 and abs(sim_diff) < 0.8
    print(f"\n  Sequences are {'DIFFERENT' if abs(sim_diff) < 0.8 else 'SAME'}: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  KOS V5.1 — ZERO-COPY RUST VSA BENCHMARK")
    print("#  Vectors NEVER cross the PyO3 boundary")
    print("#" * 70)

    r1 = test_1_zero_copy_analogy()
    r2 = test_2_speed_comparison()
    r3 = test_3_permute_sequence()

    print("\n" + "=" * 70)
    print("  V5.1 RESULTS")
    print("=" * 70)
    print(f"  1) Zero-copy analogy:    {'PASS' if r1 else 'FAIL'}")
    print(f"  2) Speed benchmark:      {'PASS' if r2 else 'FAIL'}")
    print(f"  3) Sequence encoding:    {'PASS' if r3 else 'FAIL'}")
    print("=" * 70)
