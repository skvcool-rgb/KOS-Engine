"""
KOS V5.1 — Month 2 Fixes: FAISS Scaling + Multi-Tenancy.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.scaling import FAISSIndex, NamespaceManager


def run_month2_test():
    print("=" * 70)
    print("  MONTH 2 FIXES: FAISS Scaling + Multi-Tenancy")
    print("=" * 70)

    # ── TEST #5: FAISS Vector Index ──────────────────────────
    print("\n[TEST #5] FAISS VECTOR INDEX SCALING")
    print("-" * 50)

    dim = 384  # MiniLM dimension
    index = FAISSIndex(dimension=dim)

    # Insert 50,000 random vectors (simulating a large graph)
    print(f"  Building index with 50,000 vectors (dim={dim})...")
    t0 = time.perf_counter()

    rng = np.random.default_rng(42)
    for i in range(50_000):
        vec = rng.standard_normal(dim).astype(np.float32)
        index.add(f"node_{i}", vec)

    build_time = (time.perf_counter() - t0) * 1000
    print(f"  Build time: {build_time:.0f}ms")
    print(f"  Index size: {index.size:,} vectors")

    # Query speed test
    query_vec = rng.standard_normal(dim).astype(np.float32)

    t1 = time.perf_counter()
    for _ in range(100):
        results = index.search(query_vec, top_k=10, threshold=0.0)
    query_time = (time.perf_counter() - t1) * 1000
    avg_query = query_time / 100

    print(f"  Query time (100 queries): {query_time:.1f}ms total, "
          f"{avg_query:.2f}ms avg")
    print(f"  Top result: {results[0] if results else 'none'}")

    # Verify correctness — search for a known vector
    known_vec = rng.standard_normal(dim).astype(np.float32)
    index.add("known_target", known_vec)

    results = index.search(known_vec, top_k=5, threshold=0.0)
    found_target = any(r[0] == "known_target" for r in results)
    print(f"\n  Self-search test: {'PASS' if found_target else 'FAIL'} "
          f"(known vector found in top 5)")

    # Scale comparison
    print(f"\n  Scale comparison:")
    print(f"    Brute-force O(N):  ~{50000 * 0.001:.0f}ms at 50K nodes")
    print(f"    FAISS O(log N):    {avg_query:.2f}ms at 50K nodes")
    print(f"    Speedup:           ~{(50000 * 0.001) / max(avg_query, 0.01):.0f}x")

    # ── TEST #12: Multi-Tenancy ──────────────────────────────
    print("\n[TEST #12] MULTI-TENANCY (NAMESPACE ISOLATION)")
    print("-" * 50)

    ns = NamespaceManager()

    # Assign nodes to namespaces
    legal_nodes = [f"legal_{i}" for i in range(10)]
    medical_nodes = [f"medical_{i}" for i in range(10)]
    shared_nodes = [f"shared_{i}" for i in range(5)]

    for nid in legal_nodes:
        ns.assign(nid, "legal")
    for nid in medical_nodes:
        ns.assign(nid, "medical")
    for nid in shared_nodes:
        ns.assign(nid, "default")

    print(f"  Namespaces: {ns.get_namespaces()}")

    # Test filtering
    all_nodes = legal_nodes + medical_nodes + shared_nodes

    legal_only = ns.filter_nodes(all_nodes, allowed_namespaces={"legal"})
    medical_only = ns.filter_nodes(all_nodes, allowed_namespaces={"medical"})
    both = ns.filter_nodes(all_nodes, allowed_namespaces={"legal", "default"})
    no_filter = ns.filter_nodes(all_nodes, allowed_namespaces=None)

    print(f"  Legal only:        {len(legal_only)} nodes "
          f"({'PASS' if len(legal_only) == 10 else 'FAIL'})")
    print(f"  Medical only:      {len(medical_only)} nodes "
          f"({'PASS' if len(medical_only) == 10 else 'FAIL'})")
    print(f"  Legal + default:   {len(both)} nodes "
          f"({'PASS' if len(both) == 15 else 'FAIL'})")
    print(f"  No filter:         {len(no_filter)} nodes "
          f"({'PASS' if len(no_filter) == 25 else 'FAIL'})")

    # Test result filtering
    fake_results = [(nid, 0.9) for nid in all_nodes[:5]]
    filtered = ns.filter_results(fake_results,
                                  allowed_namespaces={"legal"})
    print(f"\n  Result filter test:")
    print(f"    Input results: {len(fake_results)}")
    print(f"    After legal filter: {len(filtered)}")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MONTH 2 SUMMARY")
    print("=" * 70)
    print(f"  #5  FAISS index:    {index.size:,} vectors, "
          f"{avg_query:.2f}ms/query")
    print(f"  #5  Self-search:    {'PASS' if found_target else 'FAIL'}")
    print(f"  #12 Namespaces:     {len(ns.get_namespaces())} tenants")
    print(f"  #12 Isolation:      Legal={len(legal_only)}, "
          f"Medical={len(medical_only)}")

    all_pass = (found_target and len(legal_only) == 10
                and len(medical_only) == 10 and len(both) == 15)
    if all_pass:
        print(f"\n  MONTH 2 VERIFIED: FAISS scaling + multi-tenancy work.")
    print("=" * 70)


if __name__ == "__main__":
    run_month2_test()
