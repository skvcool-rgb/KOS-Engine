"""
KOS Domain Trainer — Feed clean domain knowledge, then let the engine think.

Phase 1: FEED — Ingest structured domain corpora (text files, inline text, URLs)
Phase 2: THINK — Autonomous dreaming + self-healing (no new external data)
Phase 3: FORAGE — Optional: let the engine explore the internet for gaps

Usage:
    # Feed a directory of .txt files, then think for 50 cycles:
    python train_domain.py --domain ./corpora/physics --think 50

    # Feed multiple domains:
    python train_domain.py --domain ./corpora/biology ./corpora/chemistry --think 100

    # Feed + forage (let it search the internet for gaps):
    python train_domain.py --domain ./corpora/medicine --think 30 --forage 20

    # Just think on existing graph (no new data):
    python train_domain.py --think 100

    # Feed a single file:
    python train_domain.py --file paper.txt --think 20
"""

import os
import sys
import time
import argparse
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.boot_brain import BOOT_CORPUS


def build_engine(load_saved=True):
    """Boot the KOS engine and return all subsystems."""
    print("=" * 60)
    print("  KOS Domain Trainer")
    print("=" * 60)

    kernel = KOSKernel(enable_vsa=False)
    lexicon = KASMLexicon()
    driver = TextDriver(kernel, lexicon)

    # Boot corpus — foundational knowledge
    driver.ingest(BOOT_CORPUS)
    print("[BOOT] Base ontology: %d nodes" % len(kernel.nodes))

    # Load saved graph if it exists
    if load_saved:
        try:
            from kos.persistence import GraphPersistence
            gp = GraphPersistence()
            gp.load(kernel, lexicon)
            print("[BOOT] Loaded saved graph: %d nodes" % len(kernel.nodes))
        except Exception:
            pass

    return kernel, lexicon, driver


def phase_feed(kernel, lexicon, driver, sources, verbose=True):
    """
    Phase 1: FEED — Ingest clean domain knowledge.

    Sources can be:
    - Directory paths (all .txt/.md files inside)
    - File paths (.txt, .md)
    - Raw text strings
    """
    print("\n" + "=" * 60)
    print("  PHASE 1: FEED (Domain Ingestion)")
    print("=" * 60)

    before = len(kernel.nodes)
    files_ingested = 0
    t0 = time.perf_counter()

    for source in sources:
        if os.path.isdir(source):
            # Directory: ingest all .txt and .md files
            patterns = ["*.txt", "*.md"]
            files = []
            for pat in patterns:
                files.extend(glob.glob(os.path.join(source, "**", pat), recursive=True))
            files.sort()

            if not files:
                print("[FEED] No .txt/.md files found in: %s" % source)
                continue

            print("[FEED] Directory: %s (%d files)" % (source, len(files)))
            for fpath in files:
                nodes_before = len(kernel.nodes)
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                        text = f.read()
                    if text.strip():
                        driver.ingest(text)
                        added = len(kernel.nodes) - nodes_before
                        files_ingested += 1
                        if verbose:
                            print("  +%d nodes from %s" % (added, os.path.basename(fpath)))
                except Exception as e:
                    print("  [ERROR] %s: %s" % (os.path.basename(fpath), str(e)[:60]))

        elif os.path.isfile(source):
            # Single file
            nodes_before = len(kernel.nodes)
            try:
                with open(source, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                if text.strip():
                    driver.ingest(text)
                    added = len(kernel.nodes) - nodes_before
                    files_ingested += 1
                    if verbose:
                        print("[FEED] +%d nodes from %s" % (added, os.path.basename(source)))
            except Exception as e:
                print("[FEED ERROR] %s: %s" % (source, str(e)[:60]))

        else:
            # Raw text string
            nodes_before = len(kernel.nodes)
            driver.ingest(source)
            added = len(kernel.nodes) - nodes_before
            if verbose:
                print("[FEED] +%d nodes from inline text" % added)

    total_new = len(kernel.nodes) - before
    elapsed = time.perf_counter() - t0
    print("\n[FEED COMPLETE] +%d nodes from %d files in %.1fs" % (
        total_new, files_ingested, elapsed))
    print("[GRAPH] Total: %d nodes" % len(kernel.nodes))

    return total_new


def phase_think(kernel, lexicon, cycles=50, verbose=True):
    """
    Phase 2: THINK — Let the engine dream, connect, and self-heal.

    No new external data. Pure internal cognition:
    - Dreamer finds novel connections between existing nodes
    - Predictive coding strengthens/weakens edges
    - Self-execution loop detects and repairs graph issues
    - Learning coordinator runs Hebbian reinforcement
    """
    print("\n" + "=" * 60)
    print("  PHASE 2: THINK (Autonomous Cognition) — %d cycles" % cycles)
    print("=" * 60)

    from kos.router_offline import KOSShellOffline
    from kos.predictive import PredictiveCodingEngine

    shell = KOSShellOffline(kernel, lexicon, enable_forager=False)
    pce = PredictiveCodingEngine(kernel, learning_rate=0.05)

    # Optional subsystems
    dreamer = None
    try:
        from kos.dreamer import Dreamer, DreamerConfig
        cfg = DreamerConfig()
        cfg.max_cycles = cycles
        cfg.cycle_interval_sec = 0  # No delay between cycles
        cfg.dream_seeds = 5         # More seeds = more connections
        dreamer = Dreamer(kernel, lexicon, pce=pce, config=cfg)
    except Exception as e:
        print("[WARN] Dreamer unavailable: %s" % str(e)[:40])

    self_loop = None
    try:
        from kos.architect import create_self_loop
        self_loop = create_self_loop(kernel=kernel, lexicon=lexicon, pce=pce)
    except Exception as e:
        print("[WARN] Self-loop unavailable: %s" % str(e)[:40])

    learning = None
    try:
        from kos.learning import LearningCoordinator
        learning = LearningCoordinator(kernel, lexicon, pce)
    except Exception as e:
        print("[WARN] Learning coordinator unavailable: %s" % str(e)[:40])

    before = len(kernel.nodes)
    t0 = time.perf_counter()
    discoveries = 0
    repairs = 0

    for cycle in range(1, cycles + 1):
        cycle_info = []

        # Dream: find novel connections
        if dreamer:
            try:
                result = dreamer.think_once()
                if result and result.get("discoveries"):
                    discoveries += len(result["discoveries"])
                    cycle_info.append("dream:%d" % len(result["discoveries"]))
            except Exception:
                pass

        # Self-heal: detect and fix graph issues
        if self_loop and cycle % 5 == 0:
            try:
                loop_result = self_loop.run_cycle(verbose=False)
                actions = loop_result.get("actions_executed", 0)
                if actions > 0:
                    repairs += actions
                    cycle_info.append("repair:%d" % actions)
                health = loop_result.get("health_after", 0)
                cycle_info.append("health:%.2f" % health)
            except Exception:
                pass

        # Learning: Hebbian reinforcement
        if learning and cycle % 3 == 0:
            try:
                learning.consolidate()
                cycle_info.append("consolidate")
            except Exception:
                pass

        if verbose and (cycle % 10 == 0 or cycle == 1):
            info = " | ".join(cycle_info) if cycle_info else "quiet"
            print("[THINK %d/%d] %d nodes | %s" % (
                cycle, cycles, len(kernel.nodes), info))

    total_new = len(kernel.nodes) - before
    elapsed = time.perf_counter() - t0
    print("\n[THINK COMPLETE] %d cycles in %.1fs" % (cycles, elapsed))
    print("  Discoveries: %d | Repairs: %d | New nodes: %d" % (
        discoveries, repairs, total_new))
    print("[GRAPH] Total: %d nodes" % len(kernel.nodes))

    return {"discoveries": discoveries, "repairs": repairs, "new_nodes": total_new}


def phase_forage(kernel, lexicon, cycles=20, verbose=True):
    """
    Phase 3: FORAGE — Let the engine explore the internet for gaps.

    The dreamer generates curiosity queries from uncertain/sparse areas,
    then the forager fetches knowledge from Wikipedia/arXiv/PubMed.
    """
    print("\n" + "=" * 60)
    print("  PHASE 3: FORAGE (Internet Exploration) — %d cycles" % cycles)
    print("=" * 60)

    from kos.router_offline import KOSShellOffline
    from kos.autonomous import AutonomousAgent

    shell = KOSShellOffline(kernel, lexicon, enable_forager=True)
    driver = TextDriver(kernel, lexicon)

    agent = AutonomousAgent(kernel, lexicon, shell, driver)
    agent.persistence_interval = 0   # We'll save once at the end
    agent.max_cycles = cycles
    agent.cycle_interval_sec = 2     # Light delay to respect APIs
    agent.max_forage_per_cycle = 2

    result = agent.run(max_cycles=cycles, verbose=verbose)

    print("\n[FORAGE COMPLETE] %d cycles" % cycles)
    print("  Nodes learned: %d | Foraged: %d | Errors: %d" % (
        result.get("nodes_learned", 0),
        result.get("topics_foraged", 0),
        result.get("errors", 0)))
    print("[GRAPH] Total: %d nodes" % len(kernel.nodes))

    return result


def save_graph(kernel, lexicon):
    """Persist the trained graph to disk."""
    try:
        from kos.persistence import GraphPersistence
        gp = GraphPersistence()
        gp.save(kernel, lexicon)
        print("\n[SAVED] Graph persisted to disk")
    except Exception as e:
        print("\n[SAVE ERROR] %s" % str(e)[:60])


def print_summary(kernel):
    """Print final graph statistics."""
    total_edges = sum(len(n.connections) for n in kernel.nodes.values())
    orphans = sum(1 for n in kernel.nodes.values() if not n.connections)
    hubs = sum(1 for n in kernel.nodes.values() if len(n.connections) > 15)
    max_conns = max((len(n.connections) for n in kernel.nodes.values()), default=0)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print("  Nodes:   %d" % len(kernel.nodes))
    print("  Edges:   %d" % total_edges)
    print("  Orphans: %d" % orphans)
    print("  Hubs:    %d (>15 connections)" % hubs)
    print("  Max connections on a single node: %d" % max_conns)
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="KOS Domain Trainer: Feed domain knowledge, then let the engine think.")

    parser.add_argument("--domain", nargs="+", default=[],
                        help="Directories or files of domain text to ingest")
    parser.add_argument("--file", nargs="+", default=[],
                        help="Individual text files to ingest")
    parser.add_argument("--text", type=str, default=None,
                        help="Inline text to ingest (quote the string)")
    parser.add_argument("--think", type=int, default=0,
                        help="Number of autonomous thinking cycles after feeding")
    parser.add_argument("--forage", type=int, default=0,
                        help="Number of internet foraging cycles after thinking")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save the graph after training")
    parser.add_argument("--no-load", action="store_true",
                        help="Start with a fresh graph (ignore saved state)")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")

    args = parser.parse_args()

    # Validate: at least one action required
    sources = args.domain + args.file
    has_text = args.text is not None
    has_feed = len(sources) > 0 or has_text
    has_think = args.think > 0
    has_forage = args.forage > 0

    if not has_feed and not has_think and not has_forage:
        parser.print_help()
        print("\nExample:")
        print("  python train_domain.py --domain ./corpora/physics --think 50")
        print("  python train_domain.py --file notes.txt --think 20 --forage 10")
        print("  python train_domain.py --think 100  # just think on existing graph")
        return

    verbose = not args.quiet

    # Boot
    kernel, lexicon, driver = build_engine(load_saved=not args.no_load)

    # Phase 1: Feed
    if has_feed:
        all_sources = sources[:]
        if has_text:
            all_sources.append(args.text)
        phase_feed(kernel, lexicon, driver, all_sources, verbose=verbose)

    # Phase 2: Think
    if has_think:
        phase_think(kernel, lexicon, cycles=args.think, verbose=verbose)

    # Phase 3: Forage
    if has_forage:
        phase_forage(kernel, lexicon, cycles=args.forage, verbose=verbose)

    # Save
    if not args.no_save:
        save_graph(kernel, lexicon)

    # Summary
    print_summary(kernel)


if __name__ == "__main__":
    main()
