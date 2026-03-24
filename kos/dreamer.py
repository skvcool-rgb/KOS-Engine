"""
KOS V6.0 — Proposal 2: Continuous Background Thinking (Dreamer)

KOS thinks without being asked. It runs a background loop that:
1. Scans for stale knowledge (staleness detection)
2. Generates curiosity queries (self-directed learning)
3. Dreams: random seed activation to find novel connections
4. Pre-caches predictions for high-traffic concepts
5. Runs self-benchmarks to monitor health

Safety controls:
- PAUSE/RESUME via flag
- MAX cycles per session
- Rate limiter (min seconds between cycles)
- CPU budget (max ms per cycle)
- Full event log for monitoring
- Kill switch via stop file
"""

import time
import random
import os
import threading


class DreamerConfig:
    """Configuration for the autonomous thinking loop."""

    def __init__(self):
        self.enabled = True
        self.cycle_interval_sec = 30       # Min seconds between cycles
        self.max_cycles = 100              # Max cycles before auto-stop
        self.max_cycle_ms = 500            # Max ms per cycle (CPU budget)
        self.dream_seeds = 3               # Random seeds per dream cycle
        self.staleness_threshold_min = 60  # Flag nodes not queried in N min
        self.curiosity_probability = 0.3   # Chance of generating curiosity query
        self.stop_file = "kos_dreamer.stop"  # Create this file to kill


class DreamEvent:
    """A single event from the dreaming process."""

    def __init__(self, event_type: str, details: str, data: dict = None):
        self.time = time.time()
        self.type = event_type
        self.details = details
        self.data = data or {}

    def __repr__(self):
        return "[%.0f] %s: %s" % (self.time, self.type, self.details[:80])


class Dreamer:
    """
    Autonomous background thinking for KOS.

    The Dreamer is NOT a separate thread by default — it runs
    synchronously when called. Use run_cycle() in a timer loop
    or call think_once() for a single cycle.

    For background operation, use start_background() which
    creates a daemon thread.
    """

    def __init__(self, kernel, lexicon, self_model=None, pce=None,
                 config: DreamerConfig = None):
        self.kernel = kernel
        self.lexicon = lexicon
        self.self_model = self_model
        self.pce = pce
        self.config = config or DreamerConfig()

        self._cycle_count = 0
        self._events = []
        self._discoveries = []  # Novel connections found
        self._paused = False
        self._stopped = False
        self._thread = None
        self._last_cycle_time = 0

    # ── Control ──────────────────────────────────────────

    def pause(self):
        """Pause the dreamer (can be resumed)."""
        self._paused = True
        self._log("control", "PAUSED by user")

    def resume(self):
        """Resume the dreamer."""
        self._paused = False
        self._log("control", "RESUMED by user")

    def stop(self):
        """Permanently stop the dreamer."""
        self._stopped = True
        self._log("control", "STOPPED by user")

    @property
    def is_active(self) -> bool:
        return (self.config.enabled
                and not self._paused
                and not self._stopped
                and self._cycle_count < self.config.max_cycles)

    # ── Single Cycle ─────────────────────────────────────

    def think_once(self, verbose: bool = False) -> dict:
        """
        Run one thinking cycle. Returns summary of what happened.
        """
        if not self.is_active:
            return {"status": "inactive", "reason": self._inactive_reason()}

        # Check stop file
        if os.path.exists(self.config.stop_file):
            self.stop()
            return {"status": "stopped", "reason": "Stop file detected"}

        # Rate limit
        elapsed = time.time() - self._last_cycle_time
        if elapsed < self.config.cycle_interval_sec:
            return {"status": "rate_limited",
                    "wait_sec": round(self.config.cycle_interval_sec - elapsed, 1)}

        t0 = time.perf_counter()
        self._cycle_count += 1
        self._last_cycle_time = time.time()

        results = {
            "cycle": self._cycle_count,
            "stale_found": 0,
            "dreams": 0,
            "discoveries": 0,
            "predictions_cached": 0,
        }

        if verbose:
            print("[DREAMER] Cycle %d starting..." % self._cycle_count)

        # ── Phase 1: Staleness Scan ──────────────────────
        stale = self._scan_staleness()
        results["stale_found"] = len(stale)
        if stale and verbose:
            print("  [STALE] %d concepts need refresh" % len(stale))

        # ── Phase 2: Dream (random activation) ───────────
        if self._budget_remaining(t0):
            discoveries = self._dream(verbose)
            results["dreams"] = self.config.dream_seeds
            results["discoveries"] = len(discoveries)

        # ── Phase 3: Pre-cache Predictions ───────────────
        if self._budget_remaining(t0) and self.pce:
            cached = self._precache_predictions()
            results["predictions_cached"] = cached

        # ── Phase 4: Curiosity Query ─────────────────────
        if (self._budget_remaining(t0)
                and random.random() < self.config.curiosity_probability):
            curiosity = self._generate_curiosity()
            if curiosity:
                results["curiosity_query"] = curiosity
                if verbose:
                    print("  [CURIOSITY] Generated: '%s'" % curiosity)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        results["elapsed_ms"] = round(elapsed_ms, 1)
        results["status"] = "completed"

        self._log("cycle", "Cycle %d: %d stale, %d discoveries, %.0fms" % (
            self._cycle_count, results["stale_found"],
            results["discoveries"], elapsed_ms))

        if verbose:
            print("[DREAMER] Cycle %d complete: %s" % (self._cycle_count, results))

        return results

    # ── Dream: Random Activation ─────────────────────────

    def _dream(self, verbose: bool = False) -> list:
        """
        Activate random seeds and look for unexpected connections.
        This is the computational equivalent of dreaming —
        exploring the graph without a specific query.
        """
        discoveries = []
        all_uids = list(self.kernel.nodes.keys())

        if len(all_uids) < 2:
            return discoveries

        for _ in range(self.config.dream_seeds):
            # Pick random seed
            seed = random.choice(all_uids)
            seed_word = self.lexicon.get_word(seed) if hasattr(self.lexicon, 'get_word') else "?"

            # Run activation
            results = self.kernel.query([seed], top_k=5)

            if results:
                for target_uid, energy in results:
                    if target_uid != seed and energy > 0.5:
                        target_word = self.lexicon.get_word(target_uid) if hasattr(self.lexicon, 'get_word') else "?"

                        # Check if this connection is surprising
                        # (high energy but not a direct neighbor)
                        is_direct = target_uid in self.kernel.nodes.get(seed, type('', (), {'connections': {}})()).connections if seed in self.kernel.nodes else False

                        if not is_direct and energy > 1.0:
                            discovery = {
                                "seed": seed_word,
                                "target": target_word,
                                "energy": round(energy, 2),
                                "hops": "indirect",
                            }
                            discoveries.append(discovery)
                            self._discoveries.append(discovery)

                            if verbose:
                                print("  [DREAM] Unexpected: %s -> %s (energy=%.2f)" % (
                                    seed_word, target_word, energy))

                            self._log("dream_discovery",
                                      "%s -> %s (energy=%.2f)" % (
                                          seed_word, target_word, energy))

        return discoveries

    # ── Staleness Scan ───────────────────────────────────

    def _scan_staleness(self) -> list:
        """Find concepts that haven't been queried recently."""
        stale = []
        if not self.self_model:
            return stale

        cutoff = time.time() - (self.config.staleness_threshold_min * 60)

        for uid, info in self.self_model._belief_log.items():
            last_q = info.get("last_queried")
            if last_q and last_q < cutoff:
                stale.append({
                    "concept": info["word"],
                    "last_queried_min_ago": round((time.time() - last_q) / 60, 1),
                })

        return stale

    # ── Pre-cache Predictions ────────────────────────────

    def _precache_predictions(self) -> int:
        """Pre-cache predictions for high-degree nodes."""
        if not self.pce:
            return 0

        cached = 0
        # Find top 5 most-connected nodes
        ranked = sorted(
            self.kernel.nodes.items(),
            key=lambda x: len(x[1].connections),
            reverse=True
        )[:5]

        for uid, node in ranked:
            self.pce.query_with_prediction([uid], top_k=3, verbose=False)
            cached += 1

        return cached

    # ── Curiosity Generation ─────────────────────────────

    def _generate_curiosity(self) -> str:
        """
        Generate a curiosity query — something KOS wants to know
        but hasn't been asked about.
        """
        if not self.self_model:
            return None

        # Find uncertain beliefs
        uncertain = self.self_model.what_am_i_uncertain_about(0.3)
        if uncertain:
            target = random.choice(uncertain)
            return "What is %s?" % target["concept"]

        # Find orphan nodes (disconnected concepts)
        orphans = [uid for uid, node in self.kernel.nodes.items()
                   if not node.connections]
        if orphans:
            uid = random.choice(orphans)
            word = self.lexicon.get_word(uid) if hasattr(self.lexicon, 'get_word') else "?"
            return "Tell me about %s" % word

        return None

    # ── Budget Check ─────────────────────────────────────

    def _budget_remaining(self, t0: float) -> bool:
        """Check if we still have CPU budget for this cycle."""
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return elapsed_ms < self.config.max_cycle_ms

    # ── Background Thread ────────────────────────────────

    def start_background(self, verbose: bool = False):
        """Start dreaming in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            return "Already running"

        self._stopped = False
        self._paused = False

        def _loop():
            while self.is_active:
                if os.path.exists(self.config.stop_file):
                    self.stop()
                    break
                self.think_once(verbose=verbose)
                time.sleep(self.config.cycle_interval_sec)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        self._log("control", "Background dreaming STARTED")
        return "Started"

    def stop_background(self):
        """Stop background dreaming."""
        self.stop()
        if self._thread:
            self._thread.join(timeout=5)
        self._log("control", "Background dreaming STOPPED")

    # ── Monitoring ───────────────────────────────────────

    def _log(self, event_type: str, details: str):
        self._events.append(DreamEvent(event_type, details))

    def _inactive_reason(self) -> str:
        if self._stopped:
            return "Permanently stopped"
        if self._paused:
            return "Paused"
        if self._cycle_count >= self.config.max_cycles:
            return "Max cycles reached (%d)" % self.config.max_cycles
        if not self.config.enabled:
            return "Disabled in config"
        return "Unknown"

    def get_status(self) -> dict:
        """Full status for dashboard monitoring."""
        return {
            "active": self.is_active,
            "paused": self._paused,
            "stopped": self._stopped,
            "cycles_completed": self._cycle_count,
            "max_cycles": self.config.max_cycles,
            "discoveries": len(self._discoveries),
            "events": len(self._events),
            "last_cycle": self._last_cycle_time,
            "cycle_interval_sec": self.config.cycle_interval_sec,
        }

    def get_discoveries(self, last_n: int = 20) -> list:
        """Get recent dream discoveries."""
        return self._discoveries[-last_n:]

    def get_events(self, last_n: int = 20) -> list:
        """Get recent events."""
        return [{"time": e.time, "type": e.type, "details": e.details}
                for e in self._events[-last_n:]]
