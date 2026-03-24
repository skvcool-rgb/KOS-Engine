"""
KOS V5.0 — Sensorimotor Loop (Phase 4: World Grounding).

A live agent that monitors real-world URLs, detects when its
internal beliefs diverge from external reality, and autonomously
self-corrects through the predictive coding loop.

The "body" is the internet connection.
The "environment" is the live web.
Actions (foraging) change the system's state (knowledge graph).
The system monitors whether its internal model matches reality.

The Loop (runs every N seconds):
    1. OBSERVE: Check monitored URLs for content changes
    2. PREDICT: What does the graph currently believe about this topic?
    3. SENSE:   Ingest the actual page content
    4. COMPARE: Compute prediction error (belief vs reality)
    5. UPDATE:  Hebbian weight correction if beliefs are wrong
    6. ACT:     Alert the user if a significant belief changed

Graceful shutdown:
    - Ctrl+C: caught by signal handler, finishes current cycle
    - STOP file: agent checks for kos_agent.stop before each cycle
    - taskkill: always works as last resort
"""

import os
import re
import time
import signal
import hashlib
import datetime
from collections import defaultdict
from pathlib import Path


class WorldMonitor:
    """
    Tracks a set of URLs and detects when their content changes.
    Uses content hashing to avoid re-processing unchanged pages.
    """

    def __init__(self):
        self.watchlist = {}       # url -> {topic, check_interval, last_hash, last_check}
        self.content_cache = {}   # url -> last extracted text

    def add_watch(self, url: str, topic: str,
                  check_interval: int = 300):
        """Add a URL to the watchlist."""
        self.watchlist[url] = {
            'topic': topic,
            'check_interval': check_interval,
            'last_hash': None,
            'last_check': 0,
            'change_count': 0,
        }

    def remove_watch(self, url: str):
        """Remove a URL from the watchlist."""
        self.watchlist.pop(url, None)
        self.content_cache.pop(url, None)

    def get_due_urls(self) -> list:
        """Return URLs that are due for a check."""
        now = time.time()
        due = []
        for url, meta in self.watchlist.items():
            elapsed = now - meta['last_check']
            if elapsed >= meta['check_interval']:
                due.append(url)
        return due

    def check_changed(self, url: str, new_text: str) -> bool:
        """Check if content has changed since last observation."""
        new_hash = hashlib.md5(new_text.encode('utf-8')).hexdigest()
        meta = self.watchlist.get(url, {})
        old_hash = meta.get('last_hash')

        if old_hash is None:
            # First observation
            meta['last_hash'] = new_hash
            meta['last_check'] = time.time()
            self.content_cache[url] = new_text
            return True  # First time = always "changed"

        changed = new_hash != old_hash
        meta['last_hash'] = new_hash
        meta['last_check'] = time.time()
        if changed:
            meta['change_count'] = meta.get('change_count', 0) + 1
        self.content_cache[url] = new_text
        return changed


class BeliefChangeAlert:
    """A record of a detected belief change."""

    def __init__(self, topic: str, url: str, change_type: str,
                 details: str, surprises: int, adjustments: int,
                 timestamp: str = None):
        self.topic = topic
        self.url = url
        self.change_type = change_type  # "new_knowledge" | "belief_revision" | "confirmation"
        self.details = details
        self.surprises = surprises
        self.adjustments = adjustments
        self.timestamp = timestamp or datetime.datetime.now().isoformat()

    def __repr__(self):
        return (f"[{self.timestamp}] {self.change_type.upper()}: "
                f"{self.topic} — {self.details}")


class SensoriMotorAgent:
    """
    The Phase 4 grounded agent.

    Runs a continuous loop that:
    1. Monitors real-world URLs for changes
    2. Predicts what the graph believes about each topic
    3. Ingests new content and computes prediction error
    4. Self-corrects through Hebbian weight adjustment
    5. Alerts the user of significant belief changes

    Designed for graceful shutdown via:
    - Ctrl+C (signal handler)
    - STOP file (checked every cycle)
    - External process kill
    """

    STOP_FILE = "kos_agent.stop"

    def __init__(self, kernel, lexicon, forager, predictive_engine,
                 text_driver=None, log_file: str = "kos_agent.log"):
        self.kernel = kernel
        self.lexicon = lexicon
        self.forager = forager
        self.pce = predictive_engine
        self.driver = text_driver or forager.driver

        self.monitor = WorldMonitor()
        self.alerts = []           # History of all alerts
        self.cycle_count = 0
        self.total_changes = 0
        self.total_new_concepts = 0
        self.log_file = log_file

        # Graceful shutdown
        self._running = True
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Ctrl+C handler — finish current cycle, then stop."""
        print("\n[AGENT] Shutdown signal received. "
              "Finishing current cycle...")
        self._running = False

    def _check_stop_file(self) -> bool:
        """Check for external stop signal via file."""
        if os.path.exists(self.STOP_FILE):
            print(f"[AGENT] Stop file detected ({self.STOP_FILE}). "
                  f"Shutting down...")
            try:
                os.remove(self.STOP_FILE)
            except OSError:
                pass
            return True
        return False

    def _log(self, message: str):
        """Log to both console and file."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(line + "\n")
        except OSError:
            pass

    def add_watch(self, url: str, topic: str,
                  check_interval: int = 300):
        """Add a URL to monitor."""
        self.monitor.add_watch(url, topic, check_interval)
        self._log(f"[WATCH] Added: {topic} -> {url} "
                  f"(every {check_interval}s)")

    def _get_topic_seeds(self, topic: str) -> list:
        """Resolve a topic string to graph seed UUIDs."""
        words = [w.lower().strip() for w in topic.split()
                 if len(w) > 2]
        seeds = []
        for w in words:
            uid = self.lexicon.word_to_uuid.get(w)
            if uid:
                seeds.append(uid)
        return seeds

    def _observe_and_compare(self, url: str) -> BeliefChangeAlert:
        """
        The core sensorimotor cycle for one URL:
        OBSERVE -> PREDICT -> SENSE -> COMPARE -> UPDATE -> ALERT
        """
        meta = self.monitor.watchlist[url]
        topic = meta['topic']

        # 1. PREDICT: What does the graph currently believe?
        seeds = self._get_topic_seeds(topic)
        prediction = {}
        if seeds:
            prediction = self.pce.predict(seeds)

        nodes_before = len(self.kernel.nodes)

        # 2. SENSE: Fetch the actual page content
        text = self.forager._fetch_and_clean(url)
        if not text:
            return None

        # 3. Check if content changed
        changed = self.monitor.check_changed(url, text)
        if not changed and meta.get('last_hash') is not None:
            return None  # No change — skip processing

        # 4. INGEST: Wire new knowledge into graph
        self.driver.ingest(text)
        nodes_after = len(self.kernel.nodes)
        new_concepts = nodes_after - nodes_before

        # 5. COMPARE: Run predictive coding loop
        seeds = self._get_topic_seeds(topic)  # Re-resolve (may have new nodes)
        surprises = 0
        adjustments = 0

        if seeds:
            report = self.pce.query_with_prediction(
                seeds, top_k=10, verbose=False)
            surprises = report['surprises']
            adjustments = report['adjustments']

        # 6. Classify the change
        if new_concepts > 0 and surprises > 5:
            change_type = "belief_revision"
            details = (f"+{new_concepts} concepts, {surprises} surprises, "
                       f"{adjustments} weight corrections")
        elif new_concepts > 0:
            change_type = "new_knowledge"
            details = f"+{new_concepts} new concepts acquired"
        else:
            change_type = "confirmation"
            details = "Content unchanged or already known"

        alert = BeliefChangeAlert(
            topic=topic, url=url, change_type=change_type,
            details=details, surprises=surprises,
            adjustments=adjustments
        )

        self.alerts.append(alert)
        self.total_changes += 1
        self.total_new_concepts += new_concepts

        return alert

    def run_single_cycle(self, verbose: bool = True) -> list:
        """Run one observation cycle across all due URLs."""
        self.cycle_count += 1
        due_urls = self.monitor.get_due_urls()

        if not due_urls:
            if verbose:
                self._log(f"[CYCLE {self.cycle_count}] "
                          f"No URLs due for check.")
            return []

        if verbose:
            self._log(f"[CYCLE {self.cycle_count}] "
                      f"Checking {len(due_urls)} URL(s)...")

        cycle_alerts = []

        for url in due_urls:
            topic = self.monitor.watchlist[url]['topic']
            if verbose:
                self._log(f"  Observing: {topic}")

            alert = self._observe_and_compare(url)

            if alert:
                cycle_alerts.append(alert)
                if alert.change_type != "confirmation":
                    self._log(f"  >> {alert}")

        if verbose and not cycle_alerts:
            self._log(f"  No changes detected.")

        return cycle_alerts

    def run(self, max_cycles: int = None, cycle_interval: int = 60,
            verbose: bool = True):
        """
        The main sensorimotor loop. Runs until stopped.

        Stop methods:
            - Ctrl+C
            - Create file: kos_agent.stop
            - max_cycles parameter
        """
        self._log("=" * 60)
        self._log("  KOS SENSORIMOTOR AGENT — PHASE 4")
        self._log(f"  Monitoring {len(self.monitor.watchlist)} URL(s)")
        self._log(f"  Cycle interval: {cycle_interval}s")
        self._log(f"  Stop: Ctrl+C or create '{self.STOP_FILE}'")
        self._log("=" * 60)

        cycles_run = 0

        while self._running:
            # Check stop conditions
            if self._check_stop_file():
                break
            if max_cycles and cycles_run >= max_cycles:
                self._log(f"[AGENT] Reached max cycles ({max_cycles}). "
                          f"Stopping.")
                break

            # Run one cycle
            alerts = self.run_single_cycle(verbose=verbose)
            cycles_run += 1

            # Print significant alerts prominently
            for alert in alerts:
                if alert.change_type == "belief_revision":
                    self._log(f"\n  *** BELIEF REVISION DETECTED ***")
                    self._log(f"  Topic: {alert.topic}")
                    self._log(f"  {alert.details}")
                    self._log(f"  The graph self-corrected.\n")

            # Wait for next cycle (interruptible)
            if self._running and (not max_cycles or cycles_run < max_cycles):
                try:
                    for _ in range(cycle_interval):
                        if not self._running or self._check_stop_file():
                            break
                        time.sleep(1)
                except KeyboardInterrupt:
                    self._running = False

        # Restore original signal handler
        signal.signal(signal.SIGINT, self._original_sigint)

        # Final report
        self._log("\n" + "=" * 60)
        self._log("  SENSORIMOTOR AGENT — SESSION REPORT")
        self._log("=" * 60)
        self._log(f"  Cycles completed:    {cycles_run}")
        self._log(f"  URLs monitored:      {len(self.monitor.watchlist)}")
        self._log(f"  Changes detected:    {self.total_changes}")
        self._log(f"  Concepts acquired:   +{self.total_new_concepts}")
        self._log(f"  Graph size:          {len(self.kernel.nodes)} nodes")
        self._log(f"  Alerts generated:    {len(self.alerts)}")

        pce_stats = self.pce.get_stats()
        self._log(f"  Prediction accuracy: {pce_stats['overall_accuracy']:.1%}")
        self._log(f"  Weight adjustments:  {pce_stats['total_weight_adjustments']}")
        self._log("=" * 60)

        return {
            'cycles': cycles_run,
            'changes': self.total_changes,
            'concepts_acquired': self.total_new_concepts,
            'alerts': self.alerts,
            'graph_size': len(self.kernel.nodes),
        }
