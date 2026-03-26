"""
Standalone health check worker — runs in a separate process to avoid GIL blocking.
Writes results to a JSON file that the API reads.
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    result_path = os.path.join(os.path.dirname(__file__), ".cache", "health_result.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    # Write "running" status
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"status": "running", "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f)

    try:
        from kos.graph import KOSKernel
        from kos.lexicon import KASMLexicon
        from kos.router_offline import KOSShellOffline
        from kos.self_improve import SelfImprover

        # Load graph
        kernel = KOSKernel()
        lexicon = KASMLexicon()
        brain_path = os.path.join(os.path.dirname(__file__), "data", "brain.kos")
        if os.path.exists(brain_path):
            kernel.load(brain_path)

        shell = KOSShellOffline(kernel, lexicon)
        # Build embeddings so the 6-layer resolver works (not just exact match)
        shell._ensure_embeddings()

        t0 = time.perf_counter()

        # Run benchmark only (skip rebalance/normalize to be fast)
        improver = SelfImprover(kernel, lexicon, shell)
        bm = improver.run_benchmark(verbose=False)

        elapsed = (time.perf_counter() - t0) * 1000

        result = {
            "status": "completed",
            "benchmark": bm,
            "time_ms": round(elapsed, 1),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "nodes": len(kernel.nodes),
            "rebalance": {"hubs_fixed": 0, "orphans_fixed": 0, "skipped": True},
            "normalization": {"clipped": 0, "max_weight": 0, "skipped": True},
            "contradictions": {"contradictions_total": len(kernel.contradictions), "resolved": 0},
            "feedback": {"reasks": 0},
            "formulas": {"formulas_found": 0},
            "predictive": {"cached": 0, "accuracy": 0.0, "adjustments": 0},
        }

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"Health check complete: {bm.get('passed', 0)}/{bm.get('total', 0)} in {elapsed:.0f}ms")

    except Exception as e:
        import traceback
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"status": "error", "error": str(e), "trace": traceback.format_exc()}, f)
        print(f"Health check error: {e}")

if __name__ == "__main__":
    main()
