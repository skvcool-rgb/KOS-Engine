"""
KOS 4-Stage Pipeline — 50-Query Full System Test

Tests across all domains: physics, chemistry, mathematics, CS,
biology, general knowledge, and edge cases.

Scores each answer on:
  - relevance_score (from 4-layer scorer)
  - keyword_hit (does answer contain expected terms)
  - latency
  - source (graph vs internet)
"""

import requests
import time
import json
import sys

API = "http://localhost:8080/api/query"

# ── Test Queries ──────────────────────────────────────────────────
# Format: (query, expected_keywords, domain)

TESTS = [
    # ── General Knowledge (10) ────────────────────────────────────
    ("What is Toronto?", ["toronto", "city"], "general"),
    ("When was Toronto founded?", ["1834"], "general"),
    ("Population of Toronto", ["2.7", "million"], "general"),
    ("What is the CN Tower?", ["tower", "toronto"], "general"),
    ("What is Montreal?", ["montreal", "city"], "general"),
    ("What is DNA?", ["genetic", "helix"], "general"),
    ("What is photosynthesis?", ["carbon", "glucose"], "general"),
    ("What is the human heart?", ["blood", "pump"], "general"),
    ("What is climate change?", ["greenhouse", "gas"], "general"),
    ("What is the water cycle?", ["evaporat", "precipit"], "general"),

    # ── Physics (10) ──────────────────────────────────────────────
    ("What is Newton second law?", ["force", "mass", "acceleration"], "physics"),
    ("What is the speed of light?", ["300", "kilometer"], "physics"),
    ("What is gravity?", ["attract", "mass"], "physics"),
    ("What is momentum?", ["mass", "velocity"], "physics"),
    ("What is Einstein special relativity?", ["light", "time"], "physics"),
    ("What is nuclear fusion?", ["hydrogen", "helium"], "physics"),
    ("What is entropy?", ["disorder", "thermodynamic"], "physics"),
    ("What is quantum entanglement?", ["qubit", "correl"], "physics"),
    ("What is Newton first law?", ["rest", "force"], "physics"),
    ("What is time dilation?", ["time", "speed"], "physics"),

    # ── Chemistry (8) ─────────────────────────────────────────────
    ("What is a covalent bond?", ["electron", "shar"], "chemistry"),
    ("What is an ionic bond?", ["transfer", "electron"], "chemistry"),
    ("What is the periodic table?", ["element", "atomic"], "chemistry"),
    ("What is oxygen?", ["oxygen", "element"], "chemistry"),
    ("What is electrolysis?", ["water", "hydrogen"], "chemistry"),
    ("What is a chemical reaction?", ["react", "product"], "chemistry"),
    ("What is water made of?", ["hydrogen", "oxygen"], "chemistry"),
    ("Tell me about apixaban", ["thrombosis", "anticoagulant"], "chemistry"),

    # ── Mathematics (7) ───────────────────────────────────────────
    ("What is the Pythagorean theorem?", ["squared", "triangle"], "mathematics"),
    ("What is pi?", ["3.14", "circumference"], "mathematics"),
    ("What is a derivative?", ["rate", "change"], "mathematics"),
    ("What is calculus?", ["integral", "deriv"], "mathematics"),
    ("What is an algorithm?", ["step", "procedure"], "mathematics"),
    ("What is probability?", ["likelihood", "event"], "mathematics"),
    ("What is a polynomial?", ["variable", "coefficient"], "mathematics"),

    # ── Computer Science (5) ──────────────────────────────────────
    ("What is backpropagation?", ["gradient", "weight"], "cs"),
    ("What is machine learning?", ["data", "pattern"], "cs"),
    ("What is Python?", ["python", "programming"], "cs"),
    ("What is artificial intelligence?", ["intelligence", "computer"], "cs"),
    ("What is encryption?", ["data", "protect"], "cs"),

    # ── Biology (5) ───────────────────────────────────────────────
    ("What are mitochondria?", ["atp", "energy"], "biology"),
    ("What is perovskite?", ["photovoltaic", "solar"], "biology"),
    ("What are coral reefs?", ["marine", "species"], "biology"),
    ("What is the human brain?", ["neuron", "billion"], "biology"),
    ("What is silicon?", ["semiconductor", "computing"], "biology"),

    # ── Edge Cases (5) ────────────────────────────────────────────
    ("Distance of moon from earth?", ["moon", "384"], "edge"),
    ("What is the Milky Way?", ["galaxy", "star"], "edge"),
    ("How fast is sound?", ["343", "meter"], "edge"),
    ("Tell me about black holes", ["gravity", "escape"], "edge"),
    ("What is Mars?", ["planet", "fourth"], "edge"),
]


def run_test(query, expected, domain, timeout=60):
    """Run a single test query and evaluate the result."""
    try:
        t0 = time.time()
        resp = requests.post(API, json={"prompt": query}, timeout=timeout)
        wall_time = time.time() - t0

        if resp.status_code != 200:
            return {
                "query": query, "domain": domain, "status": "ERROR",
                "error": f"HTTP {resp.status_code}", "wall_time": wall_time
            }

        data = resp.json()
        answer = data.get("answer", "").lower()

        # Check if expected keywords are in the answer
        hits = sum(1 for kw in expected if kw.lower() in answer)
        keyword_hit = hits > 0
        keyword_ratio = hits / len(expected) if expected else 0

        return {
            "query": query,
            "domain": domain,
            "status": "PASS" if keyword_hit else "FAIL",
            "answer": data.get("answer", "")[:120],
            "relevance": data.get("relevance_score", 0),
            "confidence": data.get("confidence", 0),
            "source": data.get("source", "?"),
            "foraged": data.get("foraged_nodes", 0),
            "latency_ms": data.get("latency_ms", 0),
            "wall_time": round(wall_time, 1),
            "keyword_ratio": round(keyword_ratio, 2),
            "off_topic": data.get("off_topic_detected", False),
        }

    except requests.exceptions.Timeout:
        return {
            "query": query, "domain": domain, "status": "TIMEOUT",
            "wall_time": timeout
        }
    except Exception as e:
        return {
            "query": query, "domain": domain, "status": "ERROR",
            "error": str(e), "wall_time": 0
        }


def main():
    print("=" * 70)
    print("KOS 4-Stage Pipeline — 50-Query Full System Test")
    print("=" * 70)

    # Check server is up
    try:
        r = requests.get("http://localhost:8080/api/status", timeout=5)
        status = r.json()
        print(f"Server: ONLINE | Nodes: {status['nodes']} | Uptime: {status['uptime_human']}")
    except Exception:
        print("ERROR: Server not reachable at localhost:8080")
        sys.exit(1)

    print(f"\nRunning {len(TESTS)} tests...\n")

    results = []
    passed = 0
    failed = 0
    errors = 0
    foraged = 0

    for i, (query, expected, domain) in enumerate(TESTS):
        result = run_test(query, expected, domain)
        results.append(result)

        status = result["status"]
        if status == "PASS":
            passed += 1
            icon = "OK"
        elif status == "FAIL":
            failed += 1
            icon = "FAIL"
        else:
            errors += 1
            icon = "ERR"

        if result.get("foraged", 0) > 0:
            foraged += 1

        rel = result.get("relevance", 0)
        lat = result.get("latency_ms", 0)
        src = result.get("source", "?")[:4]
        ans = result.get("answer", "")[:60]

        print(f"[{i+1:2d}/{len(TESTS)}] [{icon:4s}] [{domain:10s}] rel={rel:.2f} lat={lat:7.0f}ms src={src} | {query[:40]}")
        if status == "FAIL":
            print(f"         Answer: {ans}...")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{len(TESTS)} PASSED | {failed} FAILED | {errors} ERRORS")
    print(f"Foraged: {foraged} queries required internet lookup")

    avg_lat = sum(r.get("latency_ms", 0) for r in results) / len(results)
    avg_rel = sum(r.get("relevance", 0) for r in results) / len(results)
    print(f"Avg Latency: {avg_lat:.0f}ms | Avg Relevance: {avg_rel:.3f}")

    # Per-domain breakdown
    domains = {}
    for r in results:
        d = r["domain"]
        if d not in domains:
            domains[d] = {"pass": 0, "fail": 0, "err": 0}
        if r["status"] == "PASS":
            domains[d]["pass"] += 1
        elif r["status"] == "FAIL":
            domains[d]["fail"] += 1
        else:
            domains[d]["err"] += 1

    print("\nPer-Domain:")
    for d, counts in sorted(domains.items()):
        total = counts["pass"] + counts["fail"] + counts["err"]
        pct = counts["pass"] / total * 100 if total else 0
        print(f"  {d:12s}: {counts['pass']}/{total} ({pct:.0f}%)")

    # Save results
    with open("tests/pipeline_50_results.json", "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total": len(TESTS),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "avg_latency_ms": round(avg_lat, 1),
            "avg_relevance": round(avg_rel, 3),
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to tests/pipeline_50_results.json")

    return 0 if failed + errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
