"""
KOS V9.0 — Autonomous Agent Loop

Runs continuously:
1. Dreamer generates curiosity queries
2. Forager acquires knowledge from internet
3. AutoImprover optimizes the system
4. Self-Model tracks what was learned
5. Self-Execution Loop reviews architecture + auto-repairs
6. Canary Deployer advances/rollbacks staged config changes
7. Dashboard monitors everything in real-time

Safety:
- Max cycles before auto-stop
- Rate limiter (min seconds between cycles)
- Kill file: create 'kos_stop' to halt
- Max nodes cap (prevents unbounded growth)
- All changes logged for audit
- Self-loop respects 3-tier safety model (SAFE/SUPERVISED/RESTRICTED)
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
                 self_model=None, pce=None,
                 self_loop=None, canary_deployer=None,
                 canary_evaluator=None,
                 self_improver_legacy=None,
                 learning_coord=None,
                 memory_lifecycle=None,
                 verification=None):
        self.kernel = kernel
        self.lexicon = lexicon
        self.shell = shell
        self.driver = driver
        self.forager = forager
        self.improver = auto_improver
        self.dreamer = dreamer
        self.self_model = self_model
        self.pce = pce

        # V9: Self-Execution Loop + Canary
        self._self_loop = self_loop              # SelfExecutionLoop instance
        self._canary_deployer = canary_deployer  # CanaryDeployer instance
        self._canary_eval = canary_evaluator     # ShadowEvaluator instance

        # V9: References for lazy self-loop creation
        self._self_improver_legacy = self_improver_legacy
        self._learning_coord = learning_coord
        self._memory_lifecycle = memory_lifecycle
        self._verification = verification

        # V9: Self-loop config
        self.self_loop_interval = 5   # Run self-loop every N cycles
        self.canary_check_interval = 3  # Check canary every N cycles
        self.persistence_interval = 10  # Save to disk every N cycles (0 = never)

        # State
        self._cycle = 0
        self._running = False
        self._paused = False
        self._thread = None
        self._events = []
        self._foraged_topics = []
        self._improvements = []
        self._self_loop_results = []
        self._canary_results = []
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
        if max_cycles is not None:
            self.max_cycles = max_cycles
        if cycle_interval is not None:
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
            try:
                result = self._run_one_cycle(verbose)
            except Exception as e:
                self._errors.append("Cycle %d crash: %s" % (self._cycle, str(e)[:60]))
                self._log("ERROR in cycle %d: %s" % (self._cycle, str(e)[:60]))
                if verbose:
                    print("[AGENT] Cycle %d error: %s (continuing)" % (self._cycle, str(e)[:60]))

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
        """Run in a daemon thread with crash recovery."""
        def _safe_run():
            try:
                self.run(**kwargs)
            except Exception as e:
                self._errors.append("Thread crash: %s" % str(e)[:60])
                self._log("THREAD CRASH: %s" % str(e)[:60])
                self._running = False

        self._thread = threading.Thread(target=_safe_run, daemon=True)
        self._thread.start()
        return "Agent started in background"

    def stop(self):
        self._running = False
        # Save-on-shutdown: persist final state if persistence is enabled
        if self.persistence_interval > 0:
            try:
                from .persistence import GraphPersistence
                gp = GraphPersistence()
                gp.save(self.kernel, self.lexicon)
            except Exception:
                pass
        self._log("STOPPED by user")

    def pause(self):
        self._paused = True
        self._log("PAUSED")

    def resume(self):
        self._paused = False
        self._log("RESUMED")

    # ── ONE CYCLE ────────────────────────────────────────

    def _ensure_self_loop(self):
        """Lazily create the self-execution loop from available subsystems."""
        if self._self_loop is not None:
            return
        try:
            from .architect import create_self_loop
            self._self_loop = create_self_loop(
                kernel=self.kernel,
                lexicon=self.lexicon,
                pce=self.pce,
                self_improver=self._self_improver_legacy,
                learning_coord=self._learning_coord,
                canary_evaluator=self._canary_eval,
                canary_deployer=self._canary_deployer,
                memory_lifecycle=self._memory_lifecycle,
                verification=self._verification,
                auto_mode=True,
            )
        except Exception:
            pass  # Self-loop is optional

    def _run_one_cycle(self, verbose) -> dict:
        """Execute one full cycle of the autonomous loop."""
        t0 = time.perf_counter()
        result = {
            "cycle": self._cycle,
            "foraged": 0,
            "improved": 0,
            "dreamed": 0,
            "nodes_added": 0,
            "self_loop": None,
            "canary": None,
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

        # ── Phase 5: Self-Execution Loop (architecture review + repair)
        if self._cycle % self.self_loop_interval == 0:
            self._ensure_self_loop()
            if self._self_loop:
                try:
                    loop_result = self._self_loop.run_cycle(verbose=False)
                    result["self_loop"] = {
                        "health_before": loop_result.get("health_before"),
                        "health_after": loop_result.get("health_after"),
                        "actions_executed": loop_result.get("actions_executed", 0),
                        "actions_skipped": loop_result.get("actions_skipped", 0),
                        "problems_found": loop_result.get("problems_found", 0),
                    }
                    self._self_loop_results.append(result["self_loop"])
                    if verbose and loop_result.get("actions_executed", 0) > 0:
                        delta = loop_result.get("health_delta", 0)
                        sign = "+" if delta >= 0 else ""
                        print("  [SELF-LOOP] Health: %.1f%% -> %.1f%% (%s%.1f%%) | "
                              "%d actions executed" % (
                                  loop_result["health_before"] * 100,
                                  loop_result["health_after"] * 100,
                                  sign, delta * 100,
                                  loop_result["actions_executed"]))
                except Exception as e:
                    self._errors.append("Self-loop: %s" % str(e)[:40])
                    result["errors"] += 1

        # ── Phase 6: Canary Deployer (advance/rollback staged configs)
        if (self._canary_deployer and
                self._cycle % self.canary_check_interval == 0):
            try:
                deployer = self._canary_deployer
                if deployer.current_stage >= 0:
                    # Check if stage duration has elapsed
                    elapsed_since_stage = time.time() - deployer.stage_start_time
                    if elapsed_since_stage >= deployer.STAGE_DURATION:
                        # Evaluate current health as metrics signal
                        metrics_ok = True
                        if self._self_loop and self._self_loop.history:
                            last_health = self._self_loop.history[-1].get(
                                "health_after", 1.0)
                            metrics_ok = last_health >= 0.7

                        advance_result = deployer.advance_stage(metrics_ok)
                        result["canary"] = advance_result
                        self._canary_results.append(advance_result)

                        if verbose:
                            status = advance_result.get("status", "?")
                            if status == "fully_deployed":
                                print("  [CANARY] Config fully deployed!")
                            elif status == "rolled_back":
                                print("  [CANARY] ROLLED BACK - metrics regressed")
                            elif status == "advanced":
                                frac = advance_result.get("fraction", 0)
                                print("  [CANARY] Advanced to %.0f%%" % (frac * 100))
            except Exception as e:
                self._errors.append("Canary: %s" % str(e)[:40])
                result["errors"] += 1

        # ── Phase 7: Save (decoupled persistence) ─────────
        if (self.persistence_interval > 0
                and self._cycle % self.persistence_interval == 0):
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

        # Self-loop health trend
        last_health = None
        self_loop_actions = 0
        if self._self_loop_results:
            last_health = self._self_loop_results[-1].get("health_after")
            self_loop_actions = sum(
                r.get("actions_executed", 0) for r in self._self_loop_results)

        # Canary status
        canary_status = None
        if self._canary_deployer:
            canary_status = self._canary_deployer.status()

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
            "self_loop_cycles": len(self._self_loop_results),
            "self_loop_actions": self_loop_actions,
            "last_health": last_health,
            "canary": canary_status,
            "errors": len(self._errors),
            "uptime_sec": round(uptime, 1),
            "cycle_interval_sec": self.cycle_interval_sec,
            "self_loop_interval": self.self_loop_interval,
            "canary_check_interval": self.canary_check_interval,
            "persistence_interval": self.persistence_interval,
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
