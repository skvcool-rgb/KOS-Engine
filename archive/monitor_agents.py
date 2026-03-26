"""
KOS Agent Monitor — Load all approved agents, test them, and monitor activity.

Runs in a loop:
1. Loads all APPROVED agents from registry
2. Fires domain-specific test queries at each agent
3. Routes random queries to see which agent handles them
4. Logs all agent activity, hit rates, and routing decisions
5. Generates a periodic report

Usage:
    python monitor_agents.py                  # Run once with report
    python monitor_agents.py --loop 60        # Repeat every 60 seconds
    python monitor_agents.py --loop 120 &     # Background monitoring
"""

import os
import sys
import time
import json
import random
import argparse
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONIOENCODING"] = "utf-8"

LOG_FILE = "agent_monitor.log"


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "[%s] %s" % (ts, msg)
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ── Boot KOS ──────────────────────────────────────────────

def boot_engine():
    """Boot kernel, load corpora, return subsystems."""
    from kos.graph import KOSKernel
    from kos.lexicon import KASMLexicon
    from kos.drivers.text import TextDriver
    from kos.boot_brain import BOOT_CORPUS
    import glob

    kernel = KOSKernel(enable_vsa=False)
    lexicon = KASMLexicon()
    driver = TextDriver(kernel, lexicon)
    driver.ingest(BOOT_CORPUS)

    # Feed corpora
    for f in sorted(glob.glob("corpora/**/*.txt", recursive=True)):
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                driver.ingest(fh.read())
        except Exception:
            pass

    log("Engine booted: %d nodes" % len(kernel.nodes))
    return kernel, lexicon


# ── Test Queries per Domain ───────────────────────────────

DOMAIN_QUERIES = {
    "physics": [
        "What is Newton second law?",
        "How does gravity work?",
        "What is the speed of light?",
        "Explain quantum entanglement",
        "What is the Heisenberg uncertainty principle?",
        "How do black holes form?",
        "What is electromagnetic induction?",
        "Explain the photoelectric effect",
        "What is nuclear fission?",
        "How does a superconductor work?",
    ],
    "chemistry": [
        "What is a covalent bond?",
        "How do catalysts work?",
        "What is the pH scale?",
        "Explain oxidation and reduction",
        "What are amino acids?",
        "How does ATP store energy?",
        "What is electronegativity?",
        "Explain Le Chatelier principle",
        "What are enantiomers?",
        "How do enzymes lower activation energy?",
    ],
    "computer science": [
        "What is Big O notation?",
        "How does binary search work?",
        "What is a neural network?",
        "Explain the transformer architecture",
        "What is TCP/IP?",
        "How does a hash table work?",
        "What is reinforcement learning?",
        "Explain public key cryptography",
        "What is a relational database?",
        "How does gradient descent optimize?",
    ],
    "mathematics": [
        "What is the derivative?",
        "Explain eigenvalues and eigenvectors",
        "What is the central limit theorem?",
        "How does integration work?",
        "What is a matrix?",
        "Explain Bayes theorem",
        "What is the Fourier transform?",
        "How does linear regression work?",
        "What is a vector space?",
        "Explain the chain rule in calculus",
    ],
    "general knowledge": [
        "What is DNA?",
        "How does the immune system work?",
        "What causes climate change?",
        "How do tectonic plates move?",
        "What is the Big Bang?",
        "How does photosynthesis work?",
        "What is GDP?",
        "Explain supply and demand",
        "What is natural selection?",
        "How do solar panels generate electricity?",
    ],
}

# Cross-domain queries to test routing
ROUTING_QUERIES = [
    "What is the relationship between energy and mass?",
    "How do drugs interact with receptors?",
    "What causes earthquakes?",
    "How does machine learning differ from traditional programming?",
    "What is the role of mitochondria?",
    "How do interest rates affect the economy?",
    "What is the Schrodinger equation?",
    "How does CRISPR gene editing work?",
    "What is dark matter?",
    "How do semiconductors work?",
    "What is the greenhouse effect?",
    "How does the heart pump blood?",
    "What is an algorithm?",
    "How do antibiotics kill bacteria?",
    "What is the Pythagorean theorem?",
]


def run_monitor(kernel, lexicon, verbose=True):
    """Load agents, fire queries, generate report."""
    from kos.agent_factory import AgentRegistry

    registry = AgentRegistry(kernel, lexicon)

    log("")
    log("=" * 70)
    log("  KOS AGENT MONITOR")
    log("=" * 70)

    # List all agents
    all_agents = registry.list_agents()
    approved = [a for a in all_agents if a.get("status") == "APPROVED"]
    log("Registry: %d agents total, %d approved" % (len(all_agents), len(approved)))

    if not approved:
        log("No approved agents. Nothing to monitor.")
        return

    # Load all approved agents
    loaded = []
    for agent_meta in approved:
        aid = agent_meta["id"]
        try:
            agent = registry.load_agent(aid)
            if agent:
                loaded.append((aid, agent_meta, agent))
                log("  Loaded: %s (%s)" % (agent_meta["name"], agent_meta["domain"]))
            else:
                log("  FAILED to load: %s" % aid)
        except Exception as e:
            log("  ERROR loading %s: %s" % (aid, str(e)[:50]))

    if not loaded:
        log("No agents could be loaded.")
        return

    log("")
    log("--- PHASE 1: Domain-Specific Testing ---")
    log("")

    # Test each agent with its domain queries
    agent_results = {}
    for aid, meta, agent in loaded:
        domain = meta["domain"]
        queries = DOMAIN_QUERIES.get(domain, DOMAIN_QUERIES["general knowledge"])
        test_queries = random.sample(queries, min(5, len(queries)))

        log("[%s] Testing %d queries for domain: %s" % (
            meta["name"], len(test_queries), domain))

        hits = 0
        total_confidence = 0.0

        for q in test_queries:
            try:
                result = agent.query(q)
                confidence = result.get("confidence", 0)
                total_confidence += confidence
                evidence_count = len(result.get("evidence", []))

                if confidence > 0.1 or evidence_count > 0:
                    hits += 1
                    if verbose:
                        answer = str(result.get("answer", ""))[:80]
                        log("  Q: %s" % q)
                        log("    A: %s" % answer)
                        log("    Confidence: %.2f | Evidence: %d" % (
                            confidence, evidence_count))
                else:
                    if verbose:
                        log("  Q: %s -> NO MATCH (confidence=%.2f)" % (q, confidence))
            except Exception as e:
                log("  Q: %s -> ERROR: %s" % (q, str(e)[:40]))

        hit_rate = hits / max(len(test_queries), 1)
        avg_confidence = total_confidence / max(len(test_queries), 1)

        agent_results[aid] = {
            "name": meta["name"],
            "domain": domain,
            "queries_tested": len(test_queries),
            "hits": hits,
            "hit_rate": round(hit_rate, 3),
            "avg_confidence": round(avg_confidence, 3),
        }

        log("  Result: %d/%d hits (%.0f%%), avg confidence: %.3f" % (
            hits, len(test_queries), hit_rate * 100, avg_confidence))
        log("")

    # Phase 2: Cross-domain routing
    log("--- PHASE 2: Cross-Domain Query Routing ---")
    log("")

    routing_log = []
    test_routing = random.sample(ROUTING_QUERIES, min(10, len(ROUTING_QUERIES)))

    for q in test_routing:
        # Score all agents
        scores = []
        for aid, meta, agent in loaded:
            try:
                score = agent.can_handle(q)
                scores.append((aid, meta["name"], meta["domain"], score))
            except Exception:
                pass

        scores.sort(key=lambda x: -x[3])
        best = scores[0] if scores else None

        if best and best[3] > 0.1:
            # Route to best agent
            try:
                result = scores[0]
                agent_instance = dict((a[0], a[2]) for a in loaded).get(best[0])
                if agent_instance:
                    answer = agent_instance.query(q)
                    routing_entry = {
                        "query": q,
                        "routed_to": best[1],
                        "domain": best[2],
                        "score": best[3],
                        "confidence": answer.get("confidence", 0),
                        "evidence": len(answer.get("evidence", [])),
                    }
                    routing_log.append(routing_entry)

                    log("  Q: %s" % q)
                    log("    -> %s (score=%.2f) | confidence=%.2f | evidence=%d" % (
                        best[1], best[3],
                        answer.get("confidence", 0),
                        len(answer.get("evidence", []))))

                    # Show runner-up
                    if len(scores) > 1 and scores[1][3] > 0.05:
                        log("    Runner-up: %s (score=%.2f)" % (
                            scores[1][1], scores[1][3]))
            except Exception as e:
                log("  Q: %s -> ROUTING ERROR: %s" % (q, str(e)[:40]))
        else:
            log("  Q: %s -> NO AGENT MATCHED" % q)
            routing_log.append({"query": q, "routed_to": None, "score": 0})

    log("")

    # Phase 3: Agent stats summary
    log("--- PHASE 3: Agent Performance Report ---")
    log("")

    for aid, meta, agent in loaded:
        try:
            stats = agent.get_stats()
            r = agent_results.get(aid, {})
            log("[%s]" % stats.get("name", aid))
            log("  Domain: %s" % stats.get("domain", "?"))
            log("  Core concepts: %d" % stats.get("core_concepts", 0))
            log("  Queries handled: %d" % stats.get("queries", 0))
            log("  Hit rate: %.0f%%" % (r.get("hit_rate", 0) * 100))
            log("  Avg confidence: %.3f" % r.get("avg_confidence", 0))
            log("  Capabilities: %s" % ", ".join(stats.get("capabilities", [])[:3]))
        except Exception as e:
            log("[%s] Stats error: %s" % (aid, str(e)[:40]))
        log("")

    # Routing summary
    routed = [r for r in routing_log if r.get("routed_to")]
    unrouted = [r for r in routing_log if not r.get("routed_to")]
    log("--- Routing Summary ---")
    log("  Total queries: %d" % len(routing_log))
    log("  Routed: %d (%.0f%%)" % (len(routed),
        len(routed) / max(len(routing_log), 1) * 100))
    log("  Unrouted: %d" % len(unrouted))

    if routed:
        domain_counts = {}
        for r in routed:
            d = r["domain"]
            domain_counts[d] = domain_counts.get(d, 0) + 1
        log("  By domain: %s" % ", ".join(
            "%s=%d" % (d, c) for d, c in sorted(domain_counts.items(), key=lambda x: -x[1])))

    log("")
    log("=" * 70)

    # Save report
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "agents_loaded": len(loaded),
        "agent_results": agent_results,
        "routing_log": routing_log,
        "routing_rate": len(routed) / max(len(routing_log), 1),
    }
    report_path = os.path.join("agents", "monitor_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log("Report saved: %s" % report_path)

    return report


def main():
    parser = argparse.ArgumentParser(description="KOS Agent Monitor")
    parser.add_argument("--loop", type=int, default=0,
                        help="Repeat every N seconds (0 = run once)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    kernel, lexicon = boot_engine()

    if args.loop > 0:
        log("Agent monitor running every %ds. Ctrl+C to stop." % args.loop)
        cycle = 0
        while True:
            cycle += 1
            log("\n--- Monitor Cycle %d ---" % cycle)
            try:
                run_monitor(kernel, lexicon, verbose=not args.quiet)
            except Exception as e:
                log("Monitor error: %s" % str(e)[:60])
            time.sleep(args.loop)
    else:
        run_monitor(kernel, lexicon, verbose=not args.quiet)


if __name__ == "__main__":
    main()
