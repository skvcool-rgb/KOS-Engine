"""
KOS V7.0 — Autonomous Agent Loop

Runs continuously:
1. Dreamer generates curiosity queries
2. Forager acquires knowledge from internet
3. AutoImprover optimizes the system
4. Self-Model tracks what was learned
5. Dashboard monitors everything in real-time

Safety:
- Max cycles before auto-stop
- Rate limiter (min seconds between cycles)
- Kill file: create 'kos_stop' to halt
- Max nodes cap (prevents unbounded growth)
- All changes logged for audit
"""

import time
import os
import threading
import json


class AutonomousAgent:
    """
    The full autonomous loop. Combines all KOS modules into
    a self-directed learning agent.
    """

    def __init__(self, kernel, lexicon, shell, driver,
                 forager=None, auto_improver=None, dreamer=None,
                 self_model=None, pce=None):
        self.kernel = kernel
        self.lexicon = lexicon
        self.shell = shell
        self.driver = driver
        self.forager = forager
        self.improver = auto_improver
        self.dreamer = dreamer
        self.self_model = self_model
        self.pce = pce

        # State
        self._cycle = 0
        self._running = False
        self._paused = False
        self._thread = None
        self._events = []
        self._foraged_topics = []
        self._improvements = []
        self._errors = []

        # Config
        self.max_cycles = 100
        self.cycle_interval_sec = 30
        self.max_nodes = 50000
        self.max_forage_per_cycle = 2
        self.stop_file = "kos_stop"

        # Stats
        self._nodes_start = len(kernel.nodes)
        self._start_time = None

    # ── MAIN LOOP ────────────────────────────────────────

    def run(self, max_cycles=None, cycle_interval=None, verbose=True):
        """Run the autonomous loop synchronously."""
        if max_cycles:
            self.max_cycles = max_cycles
        if cycle_interval:
            self.cycle_interval_sec = cycle_interval

        self._running = True
        self._start_time = time.time()

        if verbose:
            print("[AGENT] Starting autonomous loop")
            print("  Max cycles: %d | Interval: %ds | Max nodes: %d" % (
                self.max_cycles, self.cycle_interval_sec, self.max_nodes))
            print("  Kill: create '%s' file to stop" % self.stop_file)

        while self._running and self._cycle < self.max_cycles:
            if os.path.exists(self.stop_file):
                self._log("STOPPED by kill file")
                if verbose:
                    print("[AGENT] Kill file detected. Stopping.")
                break

            if self._paused:
                time.sleep(1)
                continue

            if len(self.kernel.nodes) >= self.max_nodes:
                self._log("STOPPED: max nodes reached (%d)" % self.max_nodes)
                if verbose:
                    print("[AGENT] Max nodes reached. Stopping.")
                break

            self._cycle += 1
            result = self._run_one_cycle(verbose)

            if verbose and self._cycle % 5 == 0:
                self._print_status()

            if self.cycle_interval_sec > 0:
                time.sleep(self.cycle_interval_sec)

        self._running = False
        if verbose:
            print("\n[AGENT] Stopped after %d cycles" % self._cycle)
            self._print_status()

        return self.get_status()

    def run_background(self, **kwargs):
        """Run in a daemon thread."""
        self._thread = threading.Thread(
            target=self.run, kwargs=kwargs, daemon=True)
        self._thread.start()
        return "Agent started in background"

    def stop(self):
        self._running = False
        self._log("STOPPED by user")

    def pause(self):
        self._paused = True
        self._log("PAUSED")

    def resume(self):
        self._paused = False
        self._log("RESUMED")

    # ── ONE CYCLE ────────────────────────────────────────

    def _run_one_cycle(self, verbose) -> dict:
        """Execute one full cycle of the autonomous loop."""
        t0 = time.perf_counter()
        result = {
            "cycle": self._cycle,
            "foraged": 0,
            "improved": 0,
            "dreamed": 0,
            "nodes_added": 0,
            "errors": 0,
        }

        nodes_before = len(self.kernel.nodes)

        # ── Phase 1: Dream (generate curiosity) ─────────
        curiosity_query = None
        if self.dreamer:
            try:
                dream_result = self.dreamer.think_once(verbose=False)
                result["dreamed"] = dream_result.get("discoveries", 0)
                if dream_result.get("curiosity_query"):
                    curiosity_query = dream_result["curiosity_query"]
            except Exception as e:
                self._errors.append(str(e)[:60])
                result["errors"] += 1

        # ── Phase 2: Forage (acquire knowledge) ─────────
        if self.forager:
            topics_to_forage = []

            # Use dreamer curiosity
            if curiosity_query:
                topics_to_forage.append(curiosity_query)

            # Use self-model uncertainty
            if self.self_model:
                try:
                    uncertain = self.self_model.what_am_i_uncertain_about(0.2)
                    for u in uncertain[:1]:
                        topics_to_forage.append(u["concept"])
                except Exception:
                    pass

            # Directed learning: cycle through priority topics
            if hasattr(self, 'learning_curriculum') and self.learning_curriculum:
                idx = self._cycle % len(self.learning_curriculum)
                topics_to_forage.append(self.learning_curriculum[idx])

            # Forage each topic
            for topic in topics_to_forage[:self.max_forage_per_cycle]:
                try:
                    # Extract clean search term
                    clean = topic.replace("What is ", "").replace("Tell me about ", "").replace("?", "").strip()
                    if len(clean) < 3:
                        continue

                    # Let the forager decide the best source
                    if hasattr(self.forager, 'forage_smart'):
                        new_nodes = self.forager.forage_smart(clean, verbose=False)
                    else:
                        new_nodes = self.forager.forage_query(clean, verbose=False)
                    if new_nodes > 0:
                        result["foraged"] += new_nodes
                        self._foraged_topics.append({
                            "topic": clean,
                            "nodes": new_nodes,
                            "time": time.time(),
                        })
                        self._log("Foraged '%s': +%d nodes" % (clean, new_nodes))

                        if verbose:
                            print("  [FORAGE] '%s' -> +%d nodes" % (clean, new_nodes))
                except Exception as e:
                    self._errors.append("Forage '%s': %s" % (topic[:20], str(e)[:40]))
                    result["errors"] += 1

        # ── Phase 3: Self-Improve ────────────────────────
        if self.improver:
            try:
                improve_result = self.improver.improve(verbose=False)
                result["improved"] = improve_result.get("total_applied", 0)
                if result["improved"] > 0:
                    self._improvements.append(improve_result)
            except Exception as e:
                self._errors.append("Improve: %s" % str(e)[:40])
                result["errors"] += 1

        # ── Phase 4: Sync Self-Model ─────────────────────
        if self.self_model:
            try:
                self.self_model.sync_beliefs_from_graph()
            except Exception:
                pass

        # ── Phase 5: Save (auto-persist) ─────────────────
        try:
            from .persistence import GraphPersistence
            gp = GraphPersistence()
            gp.save(self.kernel, self.lexicon)
        except Exception:
            pass

        result["nodes_added"] = len(self.kernel.nodes) - nodes_before
        result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["total_nodes"] = len(self.kernel.nodes)

        self._events.append(result)

        if verbose:
            print("[CYCLE %d] +%d nodes | %d foraged | %d improved | %dms" % (
                self._cycle, result["nodes_added"], result["foraged"],
                result["improved"], result["elapsed_ms"]))

        return result

    # ── MONITORING ───────────────────────────────────────

    def get_status(self) -> dict:
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "running": self._running,
            "paused": self._paused,
            "cycle": self._cycle,
            "max_cycles": self.max_cycles,
            "total_nodes": len(self.kernel.nodes),
            "nodes_learned": len(self.kernel.nodes) - self._nodes_start,
            "topics_foraged": len(self._foraged_topics),
            "improvements_applied": sum(
                r.get("total_applied", 0) for r in self._improvements),
            "errors": len(self._errors),
            "uptime_sec": round(uptime, 1),
            "cycle_interval_sec": self.cycle_interval_sec,
        }

    def get_events(self, last_n=20) -> list:
        return self._events[-last_n:]

    def get_foraged(self, last_n=20) -> list:
        return self._foraged_topics[-last_n:]

    def get_improvements(self) -> list:
        return self._improvements

    def get_errors(self) -> list:
        return self._errors[-20:]

    def get_queued_proposals(self) -> list:
        if self.improver:
            return self.improver.get_queued()
        return []

    def approve_proposal(self, index: int) -> dict:
        if self.improver:
            return self.improver.approve_queued(index)
        return {"error": "No improver"}

    def reject_proposal(self, index: int) -> dict:
        if self.improver:
            return self.improver.reject_queued(index)
        return {"error": "No improver"}

    def _log(self, msg):
        self._events.append({
            "cycle": self._cycle,
            "message": msg,
            "time": time.time(),
        })

    def _print_status(self):
        s = self.get_status()
        print("\n  [STATUS] Cycle %d/%d | Nodes: %d (+%d) | Foraged: %d | Errors: %d" % (
            s["cycle"], s["max_cycles"], s["total_nodes"],
            s["nodes_learned"], s["topics_foraged"], s["errors"]))
