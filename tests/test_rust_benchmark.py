"""
KOS V5.0 Benchmark: Rust Arena Engine vs Python Engine

Tests:
    1. Basic functionality — does the Rust engine produce correct results?
    2. VSA operations — BIND, SUPERPOSE, RESONATE, CLEANUP
    3. Spreading activation — seed + propagate + query
    4. Speed benchmark — Rust vs Python on identical workloads
    5. Scale test — 10K, 25K, 50K nodes
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_1_rust_vsa():
    """Test RustVSA: analogical reasoning (same as Python test)."""
    print("=" * 70)
    print("  TEST 1: RustVSA — Analogical Reasoning")
    print("=" * 70)

    from kos_rust import RustVSA

    engine = RustVSA(dim=10_000, seed=42)

    # Create concepts
    sun = engine.node("sun")
    planet = engine.node("planet")
    gravity = engine.node("gravity")
    nucleus = engine.node("nucleus")
    electron = engine.node("electron")
    em = engine.node("electromagnetism")
    role_center = engine.node("role_center")
    role_orbiter = engine.node("role_orbiter")
    role_force = engine.node("role_force")

    # Build structures
    r_sun = engine.bind(sun, role_center)
    r_planet = engine.bind(planet, role_orbiter)
    r_grav = engine.bind(gravity, role_force)

    r_nuc = engine.bind(nucleus, role_center)
    r_elec = engine.bind(electron, role_orbiter)
    r_em = engine.bind(em, role_force)

    solar = engine.superpose([r_sun, r_planet, r_grav])
    atom = engine.superpose([r_nuc, r_elec, r_em])

    # Analogy
    mapping = engine.bind(solar, atom)
    answer = engine.bind(mapping, sun)  # unbind

    engine.store("answer", answer)
    engine.store("nucleus", nucleus)
    engine.store("electron", electron)
    engine.store("electromagnetism", em)

    matches = engine.cleanup(answer, 0.05)

    # Filter to concept names only
    concepts = [m for m in matches if m[0] in ("nucleus", "electron", "electromagnetism", "sun", "planet", "gravity")]

    print(f"\n  Question: What is the Sun of an Atom?")
    print(f"  Top matches:")
    for name, score in concepts[:3]:
        print(f"    {name:25s}  {score:+.4f}")

    passed = concepts[0][0] == "nucleus" if concepts else False
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_2_rust_kernel():
    """Test RustKernel: basic graph operations + spreading activation."""
    print("\n" + "=" * 70)
    print("  TEST 2: RustKernel — Graph + Spreading Activation")
    print("=" * 70)

    from kos_rust import RustKernel

    kernel = RustKernel(dim=10_000)

    # Build a small knowledge graph
    kernel.add_connection("toronto", "city", 0.9, "Toronto is a city.")
    kernel.add_connection("toronto", "ontario", 0.8, "Toronto is in Ontario.")
    kernel.add_connection("toronto", "canada", 0.7, "Toronto is in Canada.")
    kernel.add_connection("toronto", "population", 0.6, "Toronto has a large population.")
    kernel.add_connection("montreal", "city", 0.9, "Montreal is a city.")
    kernel.add_connection("montreal", "quebec", 0.8, "Montreal is in Quebec.")
    kernel.add_connection("montreal", "canada", 0.7, "Montreal is in Canada.")

    print(f"\n  Nodes: {kernel.node_count()}")
    print(f"  Edges: {kernel.edge_count()}")
    print(f"  Stats: {kernel.stats()}")

    # Query
    results = kernel.query(["toronto"], 5)
    print(f"\n  Query: 'toronto' -> top 5:")
    for name, score in results:
        print(f"    {name:20s}  activation={score:.4f}")

    # Verify city is in results
    result_names = [r[0] for r in results]
    passed = "city" in result_names or "ontario" in result_names
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    # Test RESONATE
    sim = kernel.resonate("toronto", "montreal")
    print(f"\n  RESONATE(toronto, montreal) = {sim:.4f}")
    print(f"  (Structural similarity via shared edges)")

    return passed


def test_3_speed_benchmark():
    """Benchmark: Rust vs Python on graph construction + query."""
    print("\n" + "=" * 70)
    print("  TEST 3: Speed Benchmark — Rust vs Python")
    print("=" * 70)

    from kos_rust import RustKernel, RustVSA
    from kasm.vsa import KASMEngine

    # ── VSA Benchmark ──
    print("\n  --- VSA Operations (10,000-D vectors) ---")

    # Python
    py_engine = KASMEngine(dimensions=10_000, seed=42)
    t0 = time.perf_counter()
    for i in range(1000):
        py_engine.node(f"py_{i}")
    py_node_time = (time.perf_counter() - t0) * 1000

    a = py_engine.get("py_0")
    b = py_engine.get("py_1")
    t0 = time.perf_counter()
    for _ in range(10_000):
        py_engine.bind(a, b)
    py_bind_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(10_000):
        py_engine.resonate(a, b)
    py_res_time = (time.perf_counter() - t0) * 1000

    # Rust
    rs_engine = RustVSA(dim=10_000, seed=42)
    t0 = time.perf_counter()
    for i in range(1000):
        rs_engine.node(f"rs_{i}")
    rs_node_time = (time.perf_counter() - t0) * 1000

    a = rs_engine.get("rs_0")
    b = rs_engine.get("rs_1")
    t0 = time.perf_counter()
    for _ in range(10_000):
        rs_engine.bind(a, b)
    rs_bind_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(10_000):
        rs_engine.resonate(a, b)
    rs_res_time = (time.perf_counter() - t0) * 1000

    print(f"\n  {'Operation':<25} {'Python (ms)':>12} {'Rust (ms)':>12} {'Speedup':>10}")
    print("  " + "-" * 62)
    print(f"  {'1000 NODE creates':<25} {py_node_time:>12.2f} {rs_node_time:>12.2f} {py_node_time/max(rs_node_time, 0.001):>9.1f}x")
    print(f"  {'10K BIND ops':<25} {py_bind_time:>12.2f} {rs_bind_time:>12.2f} {py_bind_time/max(rs_bind_time, 0.001):>9.1f}x")
    print(f"  {'10K RESONATE ops':<25} {py_res_time:>12.2f} {rs_res_time:>12.2f} {py_res_time/max(rs_res_time, 0.001):>9.1f}x")

    # ── Graph Benchmark ──
    print("\n  --- Graph Construction + Query ---")

    n_edges = 5000

    # Python graph
    from kos.graph import KOSKernel as PyKernel
    py_k = PyKernel(enable_vsa=False)  # disable VSA for fair comparison
    t0 = time.perf_counter()
    for i in range(n_edges):
        py_k.add_connection(f"src_{i % 100}", f"tgt_{i % 200}", 0.5 + (i % 10) * 0.05)
    py_build_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(100):
        py_k.query(["src_0"], top_k=10)
    py_query_time = (time.perf_counter() - t0) * 1000

    # Rust graph
    rs_k = RustKernel(dim=100)  # small dim for speed
    t0 = time.perf_counter()
    for i in range(n_edges):
        rs_k.add_connection(f"src_{i % 100}", f"tgt_{i % 200}", 0.5 + (i % 10) * 0.05, None)
    rs_build_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(100):
        rs_k.query(["src_0"], 10)
    rs_query_time = (time.perf_counter() - t0) * 1000

    print(f"  {'5K edge construction':<25} {py_build_time:>12.2f} {rs_build_time:>12.2f} {py_build_time/max(rs_build_time, 0.001):>9.1f}x")
    print(f"  {'100 queries':<25} {py_query_time:>12.2f} {rs_query_time:>12.2f} {py_query_time/max(rs_query_time, 0.001):>9.1f}x")

    return True


def test_4_scale():
    """Scale test: how fast is Rust at 10K, 25K, 50K nodes?"""
    print("\n" + "=" * 70)
    print("  TEST 4: Scale Test — Rust Performance at Scale")
    print("=" * 70)

    from kos_rust import RustKernel

    scales = [1_000, 5_000, 10_000, 25_000, 50_000]

    print(f"\n  {'Nodes':>8} {'Edges':>8} {'Build (ms)':>12} {'Query (ms)':>12} {'VSA MB':>8}")
    print("  " + "-" * 52)

    for n in scales:
        kernel = RustKernel(dim=1000, seed=42)  # D=1000 for scale test

        t0 = time.perf_counter()
        for i in range(n):
            src = f"n_{i}"
            # Each node connects to 3-5 random others
            for j in range(min(5, n)):
                tgt = f"n_{(i * 7 + j * 13) % n}"
                if src != tgt:
                    kernel.add_connection(src, tgt, 0.5 + (j % 5) * 0.1, None)
        build_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        kernel.query(["n_0"], 10)
        query_ms = (time.perf_counter() - t0) * 1000

        stats = kernel.stats()

        print(f"  {int(stats['nodes']):>8,} {int(stats['edges']):>8,} "
              f"{build_ms:>12.1f} {query_ms:>12.2f} {stats['vsa_mb']:>8.1f}")

    return True


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  KOS V5.0 — RUST ENGINE BENCHMARK")
    print("#  Arena Allocation + SIMD-Ready Hypervectors")
    print("#" * 70)

    r1 = test_1_rust_vsa()
    r2 = test_2_rust_kernel()
    r3 = test_3_speed_benchmark()
    r4 = test_4_scale()

    print("\n" + "=" * 70)
    print("  V5.0 FINAL RESULTS")
    print("=" * 70)
    print(f"  1) RustVSA analogy:      {'PASS' if r1 else 'FAIL'}")
    print(f"  2) RustKernel graph:     {'PASS' if r2 else 'FAIL'}")
    print(f"  3) Speed benchmark:      {'PASS' if r3 else 'FAIL'}")
    print(f"  4) Scale test:           {'PASS' if r4 else 'FAIL'}")
    print("=" * 70)
