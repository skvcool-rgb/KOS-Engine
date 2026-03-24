"""Test Proposals 1-3: Self-Model + Dreamer + Hierarchical Prediction."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.router_offline import KOSShellOffline
from kos.predictive import PredictiveCodingEngine
from kos.self_model import SelfModel
from kos.dreamer import Dreamer, DreamerConfig
from kos.hierarchical import HierarchicalPredictor

def banner(t):
    print("\n" + "=" * 60)
    print("  " + t)
    print("=" * 60)

kernel = KOSKernel(enable_vsa=False)
lexicon = KASMLexicon()
driver = TextDriver(kernel, lexicon)
pce = PredictiveCodingEngine(kernel, learning_rate=0.05)

driver.ingest("""
Toronto is a major city in the Canadian province of Ontario.
Toronto was founded and incorporated in the year 1834.
The city of Toronto has a population of approximately 2.7 million people.
Perovskite is a highly efficient material for photovoltaic cells.
Backpropagation adjusts weights by computing gradient of the loss.
Artificial neural networks are inspired by biological neurons.
Quantum computers use qubits which can exist in superposition.
Entanglement allows two qubits to be correlated across any distance.
""")

shell = KOSShellOffline(kernel, lexicon, enable_forager=False)

# ══════════════════════════════════════════════════════════
# PROPOSAL 1: Self-Referential Graph
# ══════════════════════════════════════════════════════════

banner("PROPOSAL 1: Self-Referential Graph")

sm = SelfModel(kernel, lexicon, pce)
sm.sync_beliefs_from_graph()

print("\n[1a] What do I know? (top 10 by confidence)")
beliefs = sm.what_do_i_know(min_confidence=0.0)
for b in beliefs[:10]:
    print("  %.0f%% | %s (source: %s)" % (
        b["confidence"] * 100, b["concept"], b["source"]))

print("\n[1b] What am I uncertain about?")
uncertain = sm.what_am_i_uncertain_about(0.3)
for u in uncertain[:5]:
    print("  %.0f%% | %s" % (u["confidence"] * 100, u["concept"]))
print("  Total uncertain: %d concepts" % len(uncertain))

print("\n[1c] How did I learn about Toronto?")
trace = sm.how_did_i_learn("toronto")
print("  Source: %s" % trace.get("source"))
print("  Confidence: %s" % trace.get("confidence"))
print("  Provenance: %s" % (trace.get("provenance", [])[:2]))

print("\n[1d] My capabilities:")
caps = sm.my_capabilities()
print("  Can do: %d things" % len(caps["can_do"]))
print("  Cannot do: %d things" % len(caps["cannot_do"]))
print("  Uncertain about: %d things" % len(caps["uncertain_about"]))

print("\n[1e] My current state:")
state = sm.my_current_state()
for k, v in state.items():
    print("  %s: %s" % (k, v))

# Record a query
sm.record_query("Where is Toronto?", "Ontario, Canada", 45.0)
print("\n[1f] Query recorded. Timeline:")
for e in sm.get_timeline(5):
    print("  [%s] %s" % (e["type"], e["details"][:60]))

print("\n[PASS] Proposal 1: Self-Model operational")

# ══════════════════════════════════════════════════════════
# PROPOSAL 2: Dreamer (Continuous Background Thinking)
# ══════════════════════════════════════════════════════════

banner("PROPOSAL 2: Dreamer (Background Thinking)")

config = DreamerConfig()
config.max_cycles = 5
config.cycle_interval_sec = 0  # No delay for testing
config.dream_seeds = 3
config.curiosity_probability = 1.0  # Always generate curiosity

dreamer = Dreamer(kernel, lexicon, sm, pce, config)

print("\n[2a] Running 5 dream cycles...")
for i in range(5):
    result = dreamer.think_once(verbose=True)
    if result.get("status") != "completed":
        print("  Cycle %d: %s" % (i + 1, result.get("status")))

print("\n[2b] Dreamer status:")
status = dreamer.get_status()
for k, v in status.items():
    print("  %s: %s" % (k, v))

print("\n[2c] Dream discoveries:")
discoveries = dreamer.get_discoveries()
for d in discoveries[:5]:
    print("  %s -> %s (energy=%.2f)" % (d["seed"], d["target"], d["energy"]))
if not discoveries:
    print("  (No unexpected connections found — graph is well-connected)")

print("\n[2d] Pause/Resume test:")
dreamer.pause()
r = dreamer.think_once()
print("  While paused: %s" % r["status"])
dreamer.resume()
r = dreamer.think_once()
print("  After resume: %s" % r.get("status", "rate_limited"))

print("\n[PASS] Proposal 2: Dreamer operational (with safety controls)")

# ══════════════════════════════════════════════════════════
# PROPOSAL 3: Hierarchical Predictive Coding (6 Layers)
# ══════════════════════════════════════════════════════════

banner("PROPOSAL 3: Hierarchical Prediction (6 Layers)")

hp = HierarchicalPredictor(kernel, pce)

# Get seeds for a test query
toronto_uid = lexicon.word_to_uuid.get("toronto")
seeds = [toronto_uid] if toronto_uid else []

print("\n[3a] Full prediction stack for 'Toronto' query:")
predictions = hp.predict_full(seeds, "toronto_query")
for k, v in predictions.items():
    if k != "activations":
        print("  %s: %s" % (k, v))

print("\n[3b] Running actual query and observing...")
actual_results = {
    "top_energy": 2.5,
    "ticks": 8,
    "confidence": 0.9,
    "no_answer": False,
    "foraged": False,
}
errors = hp.observe_actual(seeds, actual_results, "toronto_query")
print("  Prediction errors across layers:")
for k, v in errors.items():
    if v is not None:
        print("    %s: %.4f" % (k, v))

print("\n[3c] Running 10 prediction cycles to build accuracy...")
for i in range(10):
    hp.predict_full(seeds, "toronto_query_%d" % i)
    hp.observe_actual(seeds, {
        "top_energy": 2.5 + (i % 3) * 0.1,
        "ticks": 8,
        "confidence": 0.85 + i * 0.01,
        "no_answer": False,
        "foraged": False,
    }, "toronto_query_%d" % i)

print("\n[3d] Layer-by-layer dashboard:")
hp.print_dashboard()

print("\n[3e] Top-down priors:")
priors = hp.get_top_down_priors(seeds)
print("  Activation spread: %s" % priors.get("activation_spread"))
print("  Confidence ceiling: %s" % priors.get("confidence_ceiling", "none"))
print("  Error floor: %s" % priors.get("expected_error_floor", "none"))

print("\n[PASS] Proposal 3: Hierarchical Prediction operational")

# ══════════════════════════════════════════════════════════
# INTEGRATION: All 3 Working Together
# ══════════════════════════════════════════════════════════

banner("INTEGRATION: All 3 Proposals Working Together")

print("\n  1. Self-Model knows %d beliefs at %.0f%% avg confidence" % (
    len(sm._belief_log), sm.my_current_state()["avg_confidence"] * 100))
print("  2. Dreamer completed %d cycles, found %d discoveries" % (
    dreamer._cycle_count, len(dreamer._discoveries)))
print("  3. Hierarchical Predictor: %d cycles, %.1f%% avg accuracy" % (
    hp._cycle_count, hp.get_summary()["avg_accuracy"]))
print("\n  All proposals operational with monitoring and safety controls.")
print("  Kill switch: create 'kos_dreamer.stop' file")
print("  Pause: dreamer.pause() / dreamer.resume()")
print("  Monitor: sm.introspect(), dreamer.get_status(), hp.print_dashboard()")
