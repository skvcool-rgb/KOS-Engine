"""
KOS 6-Hour Autonomous Run — Full cognitive loop with live logging.

Runs all 7 phases continuously:
  1. Dream (curiosity queries)
  2. Forage (internet knowledge acquisition)
  3. Self-Improve (graph optimization)
  4. Self-Model (track what was learned)
  5. Self-Execution Loop (architecture review + auto-repair)
  6. Canary Deployer (staged config changes)
  7. Persist (save every 10 cycles)

Logs to: kos_6hr.log (tail -f kos_6hr.log to watch)
Stop:    create 'kos_stop' file or Ctrl+C
"""

import os
import sys
import time
import signal
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONIOENCODING"] = "utf-8"

# ── Logging setup ─────────────────────────────────────────
LOG_FILE = "kos_6hr.log"

class TeeLogger:
    """Write to both stdout and log file. Fully compatible with libraries."""
    def __init__(self, logfile, stream):
        self.terminal = stream
        self.log = open(logfile, "a", encoding="utf-8", buffering=1)
        self.encoding = "utf-8"
        self.errors = "replace"
        self.name = getattr(stream, "name", "<tee>")
    def write(self, message):
        try:
            self.terminal.write(message)
        except Exception:
            pass
        self.log.write(message)
    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        self.log.flush()
    def isatty(self):
        return False
    def fileno(self):
        return self.terminal.fileno()
    def readable(self):
        return False
    def writable(self):
        return True
    def seekable(self):
        return False

sys.stdout = TeeLogger(LOG_FILE, sys.__stdout__)
sys.stderr = TeeLogger(LOG_FILE, sys.__stderr__)


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[%s] %s" % (ts, msg))


# ── Boot ──────────────────────────────────────────────────
log("=" * 70)
log("  KOS ENGINE — 6-HOUR AUTONOMOUS RUN")
log("=" * 70)
log("Log file: %s" % os.path.abspath(LOG_FILE))
log("Stop: create 'kos_stop' file or Ctrl+C")
log("")

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.router_offline import KOSShellOffline
from kos.predictive import PredictiveCodingEngine
from kos.boot_brain import BOOT_CORPUS

kernel = KOSKernel(enable_vsa=False)
lexicon = KASMLexicon()
driver = TextDriver(kernel, lexicon)

# Ingest boot corpus
driver.ingest(BOOT_CORPUS)
log("Boot corpus: %d nodes" % len(kernel.nodes))

# Load saved graph if it exists
saved_loaded = False
try:
    from kos.persistence import GraphPersistence
    gp = GraphPersistence()
    if gp.exists():
        gp.load(kernel, lexicon)
        if len(kernel.nodes) > 300:  # Sanity check: loaded meaningful data
            saved_loaded = True
            log("Loaded saved brain: %d nodes, %d edges" % (
                len(kernel.nodes),
                sum(len(n.connections) for n in kernel.nodes.values())))
except Exception as e:
    log("Brain load error (will re-feed corpora): %s" % str(e)[:60])

# If no saved brain, feed all domain corpora
if not saved_loaded:
    import glob as globmod
    corpora_dir = os.path.join(os.path.dirname(__file__), "corpora")
    txt_files = sorted(globmod.glob(os.path.join(corpora_dir, "**", "*.txt"), recursive=True))
    log("Feeding %d corpus files from corpora/..." % len(txt_files))
    for fpath in txt_files:
        try:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            if text.strip():
                before = len(kernel.nodes)
                driver.ingest(text)
                added = len(kernel.nodes) - before
                log("  +%d nodes from %s" % (added, os.path.basename(fpath)))
        except Exception as e:
            log("  [ERROR] %s: %s" % (os.path.basename(fpath), str(e)[:40]))
    log("Corpus feed complete: %d nodes" % len(kernel.nodes))

# Shell + PCE
shell = KOSShellOffline(kernel, lexicon, enable_forager=True)
pce = PredictiveCodingEngine(kernel, learning_rate=0.05)

# ── Wire all subsystems ──────────────────────────────────
from kos.autonomous import AutonomousAgent

agent = AutonomousAgent(
    kernel=kernel,
    lexicon=lexicon,
    shell=shell,
    driver=driver,
)

# Configure for 6-hour run
HOURS = 6
CYCLE_SEC = 30          # 30 seconds between cycles
MAX_CYCLES = int((HOURS * 3600) / CYCLE_SEC)  # ~720 cycles

agent.max_cycles = MAX_CYCLES
agent.cycle_interval_sec = CYCLE_SEC
agent.persistence_interval = 10    # Save every 10 cycles (~5 min)
agent.self_loop_interval = 5       # Self-heal every 5 cycles
agent.canary_check_interval = 3    # Check canary every 3 cycles
agent.max_forage_per_cycle = 2
agent.max_nodes = 100000

log("")
log("CONFIGURATION:")
log("  Duration: %d hours" % HOURS)
log("  Cycle interval: %ds" % CYCLE_SEC)
log("  Max cycles: %d" % MAX_CYCLES)
log("  Persistence: every %d cycles" % agent.persistence_interval)
log("  Self-heal: every %d cycles" % agent.self_loop_interval)
log("  Max nodes: %d" % agent.max_nodes)
log("  Starting nodes: %d" % len(kernel.nodes))
log("")

# ── Graceful shutdown on Ctrl+C ──────────────────────────
def handle_signal(sig, frame):
    log("")
    log("SIGNAL RECEIVED — shutting down gracefully...")
    agent.stop()

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# ── Wire self-improvement subsystems ─────────────────────
improver = None
try:
    from kos.self_improve import SelfImprover
    improver = SelfImprover(kernel, lexicon, shell)
    agent.improver = improver
    log("SelfImprover: wired")
except Exception as e:
    log("SelfImprover: unavailable (%s)" % str(e)[:40])

proposer = None
try:
    from kos.propose import CodeProposer
    proposer = CodeProposer(kernel, lexicon, pce)
    log("CodeProposer: wired")
except Exception as e:
    log("CodeProposer: unavailable (%s)" % str(e)[:40])

self_loop = None
try:
    from kos.architect import create_self_loop
    self_loop = create_self_loop(kernel=kernel, lexicon=lexicon, pce=pce)
    agent._self_loop = self_loop
    log("SelfExecutionLoop: wired")
except Exception as e:
    log("SelfExecutionLoop: unavailable (%s)" % str(e)[:40])

# ── Observability: Health + Self-Improvement Tracker ─────
class SystemObserver:
    """Monitors health, self-improvement actions, and code proposals."""

    def __init__(self):
        self.snapshots = []
        self.proposals_seen = set()
        self.improvements_log = []
        self.self_loop_log = []

    def snapshot(self, cycle, kernel, agent):
        total_edges = sum(len(n.connections) for n in kernel.nodes.values())
        orphans = sum(1 for n in kernel.nodes.values() if not n.connections)
        hubs = sum(1 for n in kernel.nodes.values() if len(n.connections) > 15)
        max_conns = max((len(n.connections) for n in kernel.nodes.values()), default=0)

        snap = {
            "cycle": cycle,
            "time": datetime.datetime.now().isoformat(),
            "nodes": len(kernel.nodes),
            "edges": total_edges,
            "orphans": orphans,
            "hubs": hubs,
            "max_connections": max_conns,
            "errors": len(agent._errors),
            "foraged": len(agent._foraged_topics),
        }
        self.snapshots.append(snap)

        log("")
        log("=" * 60)
        log("  SYSTEM OBSERVATION — Cycle %d" % cycle)
        log("=" * 60)

        # Graph health
        log("[GRAPH]")
        log("  Nodes: %d | Edges: %d" % (snap["nodes"], snap["edges"]))
        log("  Orphans: %d | Hubs(>15): %d | Max-conns: %d" % (
            snap["orphans"], snap["hubs"], snap["max_connections"]))
        if len(self.snapshots) > 1:
            prev = self.snapshots[-2]
            dn = snap["nodes"] - prev["nodes"]
            de = snap["edges"] - prev["edges"]
            do = snap["orphans"] - prev["orphans"]
            log("  Delta: %+d nodes, %+d edges, %+d orphans" % (dn, de, do))

        # Self-execution loop health
        if agent._self_loop and agent._self_loop_results:
            last = agent._self_loop_results[-1]
            health = last.get("health_after", "?")
            actions = last.get("actions_executed", 0)
            total_actions = sum(r.get("actions_executed", 0) for r in agent._self_loop_results)
            log("[SELF-HEAL]")
            log("  Last health: %.3f | Actions this cycle: %d | Total repairs: %d" % (
                health, actions, total_actions))
            # Log specific repairs
            problems = last.get("problems_found", [])
            if problems:
                log("  Problems detected: %s" % ", ".join(str(p)[:50] for p in problems[:5]))
            repairs = last.get("repairs_executed", [])
            if repairs:
                for r in repairs[:5]:
                    log("  REPAIR: %s" % str(r)[:70])

        # Self-improvement suggestions
        if improver:
            log("[SELF-IMPROVE]")
            if agent._improvements:
                recent = agent._improvements[-3:]  # last 3
                for imp in recent:
                    total = imp.get("total_applied", 0)
                    queued = imp.get("total_queued", 0)
                    log("  Applied: %d | Queued: %d" % (total, queued))
                    if imp.get("rebalance"):
                        rb = imp["rebalance"]
                        log("    Rebalance: hubs_fixed=%s orphans_fixed=%s" % (
                            rb.get("hubs_fixed", 0), rb.get("orphans_fixed", 0)))
            else:
                log("  No improvements applied yet")

        # Code proposals (from CodeProposer)
        if proposer:
            log("[CODE PROPOSALS]")
            proposals_dir = os.path.join(os.path.dirname(__file__), "proposals")
            if os.path.exists(proposals_dir):
                import json as jsonmod
                proposal_files = sorted(
                    [f for f in os.listdir(proposals_dir) if f.endswith(".json")],
                    key=lambda f: os.path.getmtime(os.path.join(proposals_dir, f)),
                    reverse=True)
                new_proposals = [f for f in proposal_files if f not in self.proposals_seen]
                if new_proposals:
                    log("  NEW proposals since last check:")
                    for pf in new_proposals[:5]:
                        self.proposals_seen.add(pf)
                        try:
                            with open(os.path.join(proposals_dir, pf), 'r') as fp:
                                prop = jsonmod.load(fp)
                            log("    [%s] %s" % (
                                prop.get("type", "unknown"),
                                prop.get("description", prop.get("title", pf))[:60]))
                            if prop.get("rationale"):
                                log("      Rationale: %s" % str(prop["rationale"])[:70])
                        except Exception:
                            log("    %s (could not read)" % pf)
                else:
                    log("  No new proposals (total: %d)" % len(proposal_files))
            else:
                log("  No proposals directory")

        # Foraged topics
        if agent._foraged_topics:
            recent_topics = agent._foraged_topics[-5:]
            log("[FORAGED]")
            for t in recent_topics:
                if isinstance(t, dict):
                    log("  %s (+%d nodes)" % (
                        t.get("topic", "?")[:40], t.get("nodes_added", 0)))
                else:
                    log("  %s" % str(t)[:50])

        # Errors
        if agent._errors:
            recent_errs = agent._errors[-3:]
            log("[ERRORS] Last %d:" % len(recent_errs))
            for e in recent_errs:
                log("  %s" % str(e)[:70])

        log("=" * 60)
        log("")

observer = SystemObserver()

# ── Monkey-patch cycle callback for observation ──────────
_original_run_one_cycle = agent._run_one_cycle

def _patched_run_one_cycle(verbose=True):
    result = _original_run_one_cycle(verbose)
    # Full observation every 25 cycles (~12.5 min)
    if agent._cycle % 25 == 0:
        observer.snapshot(agent._cycle, kernel, agent)
    # Run code proposer every 100 cycles to generate suggestions
    if proposer and agent._cycle % 100 == 0:
        try:
            log("[PROPOSER] Generating improvement proposals at cycle %d..." % agent._cycle)
            proposer.propose_synonym_additions()
        except Exception as e:
            log("[PROPOSER] Error: %s" % str(e)[:50])
    return result

agent._run_one_cycle = _patched_run_one_cycle

# ── RUN ───────────────────────────────────────────────────
log("STARTING AUTONOMOUS LOOP...")
log("=" * 70)

start_time = time.time()

try:
    result = agent.run(
        max_cycles=MAX_CYCLES,
        verbose=True,
    )
except KeyboardInterrupt:
    log("KeyboardInterrupt — stopping...")
    agent.stop()
    result = agent.get_status()
except Exception as e:
    log("FATAL ERROR: %s" % str(e))
    result = agent.get_status()

# ── Final report ──────────────────────────────────────────
elapsed = time.time() - start_time
hours = elapsed / 3600

log("")
log("=" * 70)
log("  6-HOUR RUN COMPLETE")
log("=" * 70)
log("  Runtime: %.1f hours (%.0f minutes)" % (hours, elapsed / 60))
log("  Cycles completed: %d / %d" % (result.get("cycle", 0), MAX_CYCLES))
log("  Final nodes: %d" % len(kernel.nodes))
log("  Final edges: %d" % sum(len(n.connections) for n in kernel.nodes.values()))
log("  Nodes learned: %d" % result.get("nodes_learned", 0))
log("  Topics foraged: %d" % result.get("topics_foraged", 0))
log("  Self-loop cycles: %d" % result.get("self_loop_cycles", 0))
log("  Self-loop actions: %d" % result.get("self_loop_actions", 0))
log("  Errors: %d" % result.get("errors", 0))

if observer.snapshots:
    log("")
    log("HEALTH TREND:")
    for s in observer.snapshots:
        log("  Cycle %d: %d nodes, %d edges, %d orphans, %d foraged" % (
            s["cycle"], s["nodes"], s["edges"], s["orphans"], s["foraged"]))

# Self-improvement summary
log("")
log("SELF-IMPROVEMENT SUMMARY:")
total_repairs = sum(r.get("actions_executed", 0) for r in agent._self_loop_results) if agent._self_loop_results else 0
log("  Self-loop cycles: %d" % len(agent._self_loop_results))
log("  Total auto-repairs: %d" % total_repairs)
if agent._self_loop_results:
    health_start = agent._self_loop_results[0].get("health_after", "?")
    health_end = agent._self_loop_results[-1].get("health_after", "?")
    log("  Health trend: %.3f -> %.3f" % (health_start, health_end))
log("  Improvements applied: %d" % len(agent._improvements))

# Code proposals summary
proposals_dir = os.path.join(os.path.dirname(__file__), "proposals")
if os.path.exists(proposals_dir):
    prop_count = len([f for f in os.listdir(proposals_dir) if f.endswith(".json")])
    log("  Code proposals generated: %d (in proposals/)" % prop_count)

# Final save
try:
    gp = GraphPersistence()
    gp.save(kernel, lexicon)
    log("")
    log("Brain saved to disk.")
except Exception:
    pass

log("=" * 70)
log("Done. Log: %s" % os.path.abspath(LOG_FILE))
