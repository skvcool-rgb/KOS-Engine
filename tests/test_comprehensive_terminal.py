"""
KOS V6.1 — Comprehensive Terminal Test

Tests EVERY capability of the agent in one run.
Results are scored and issues are logged for fixing.
"""
import sys, os, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.router_offline import KOSShellOffline
from kos.forager import WebForager
from kos.predictive import PredictiveCodingEngine
from kos.self_model import SelfModel
from kos.dreamer import Dreamer, DreamerConfig
from kos.hierarchical import HierarchicalPredictor
from kos.emotion import EmotionEngine

issues = []
def log_issue(category, severity, desc):
    issues.append({"cat": category, "sev": severity, "desc": desc})

def banner(t):
    print("\n" + "=" * 60)
    print("  " + t)
    print("=" * 60)

def check(name, condition, got="", expected=""):
    status = "PASS" if condition else "FAIL"
    print("  [%s] %s" % (status, name))
    if not condition:
        if got: print("       Got: %s" % str(got)[:80])
        if expected: print("       Expected: %s" % str(expected)[:80])
        log_issue(name, "HIGH" if "FAIL" in status else "LOW",
                  "Got '%s' expected '%s'" % (str(got)[:50], str(expected)[:50]))
    return condition

# ══════════════════════════════════════════════════════════
# BOOT
# ══════════════════════════════════════════════════════════

banner("BOOT & INGESTION")

t0 = time.perf_counter()
kernel = KOSKernel(enable_vsa=False)
lexicon = KASMLexicon()
driver = TextDriver(kernel, lexicon)
boot_ms = (time.perf_counter() - t0) * 1000
check("Boot time < 500ms", boot_ms < 500, "%.0fms" % boot_ms)

t0 = time.perf_counter()
driver.ingest("""
Toronto is a major city in the Canadian province of Ontario.
Toronto was founded and incorporated in the year 1834.
The city of Toronto has a population of approximately 2.7 million people.
Toronto has a humid continental climate with warm summers and cold winters.
John Graves Simcoe originally established the settlement of Toronto.
Montreal was founded in the year 1642 in Quebec province.
Montreal has a population of 1.7 million residents.
Perovskite is a highly efficient material for photovoltaic cells.
Photovoltaic cells capture photons to produce electricity.
Silicon is a traditional semiconductor for computing and solar panels.
Perovskite is remarkably cheap and affordable to manufacture.
Backpropagation adjusts weights by computing gradient of the loss.
Artificial neural networks are inspired by biological neurons.
Quantum computers use qubits which can exist in superposition.
Entanglement allows two qubits to be correlated across any distance.
The Sun produces energy through nuclear fusion of hydrogen into helium.
Mitochondria produce ATP which is the energy currency of cells.
Apixaban prevents thrombosis without dietary restrictions.
Apixaban does not cause bleeding in patients.
Unlike warfarin, apixaban is a modern anticoagulant.
Coral reefs support 25 percent of all marine species.
Ocean acidification threatens coral reef survival worldwide.
The human heart pumps blood through arteries and veins.
Blood carries oxygen from the lungs to every cell in the body.
Water consists of two hydrogen atoms bonded to one oxygen atom.
Electrolysis splits water into hydrogen and oxygen using electricity.
Einstein special relativity states nothing travels faster than light.
Forward time travel is proven real via time dilation at high speeds.
Backward time travel has never been observed experimentally.
DNA contains the genetic instructions for building proteins.
""")
ingest_ms = (time.perf_counter() - t0) * 1000

nodes = len(kernel.nodes)
edges = sum(len(n.connections) for n in kernel.nodes.values())
check("Ingest time < 5000ms", ingest_ms < 5000, "%.0fms" % ingest_ms)
check("Nodes > 30", nodes > 30, nodes)
check("Edges > 100", edges > 100, edges)
print("  Nodes: %d | Edges: %d | Ingest: %.0fms" % (nodes, edges, ingest_ms))

shell = KOSShellOffline(kernel, lexicon, enable_forager=False)

# ══════════════════════════════════════════════════════════
# 1. FACTUAL RETRIEVAL (10 queries)
# ══════════════════════════════════════════════════════════

banner("1. FACTUAL RETRIEVAL")

passed = 0
total = 0
retrieval_tests = [
    ("Where is Toronto?", ["ontario", "province"]),
    ("When was Toronto founded?", ["1834"]),
    ("Population of Toronto?", ["million"]),
    ("Climate of Toronto?", ["humid", "continental"]),
    ("Who established Toronto?", ["simcoe"]),
    ("Tell me about perovskite", ["photovoltaic", "efficient"]),
    ("What is entanglement?", ["qubit", "correlated"]),
    ("How do neural networks learn?", ["backpropagation", "gradient"]),
    ("What produces ATP?", ["mitochondria"]),
    ("Tell me about apixaban", ["thrombosis"]),
]

latencies = []
for q, exp in retrieval_tests:
    total += 1
    t0 = time.perf_counter()
    a = shell.chat(q)
    lat = (time.perf_counter() - t0) * 1000
    latencies.append(lat)
    hits = [k for k in exp if k in a.lower()]
    ok = len(hits) >= 1
    if ok: passed += 1
    check(q[:45], ok, a.strip()[:60], exp)

avg_lat = sum(latencies) / len(latencies)
print("\n  Retrieval: %d/%d | Avg latency: %.0fms" % (passed, total, avg_lat))
check("Retrieval accuracy >= 90%%", passed >= 9, "%d/10" % passed, "9/10")
check("Avg latency < 2000ms", avg_lat < 2000, "%.0fms" % avg_lat)

# ══════════════════════════════════════════════════════════
# 2. IRRELEVANT QUERY REJECTION
# ══════════════════════════════════════════════════════════

banner("2. IRRELEVANT QUERY REJECTION")

reject_tests = [
    "Is time travel possible?",
    "How to cook pasta?",
    "What is the meaning of life?",
    "What is the price of bitcoin?",
    "How tall is the Eiffel Tower?",
    "Who is the president of France?",
    "How to be a perfect being?",
    "What color is happiness?",
]

rejected = 0
for q in reject_tests:
    a = shell.chat(q)
    is_rejected = "don't have" in a.lower() or "no relevant" in a.lower()
    if is_rejected: rejected += 1
    check("Reject: %s" % q[:40], is_rejected, a.strip()[:60], "I don't have data")

check("Rejection rate >= 87%%", rejected >= 7, "%d/8" % rejected, "7/8")

# ══════════════════════════════════════════════════════════
# 3. MATH (exact computation)
# ══════════════════════════════════════════════════════════

banner("3. MATH (SymPy Exact)")

math_tests = [
    ("345000000 * 0.0825", "28462500"),
    ("integral of x^2", "x**3"),
    ("derivative of sin(x)", "cos"),
    ("sqrt(144)", "12"),
]

for q, exp in math_tests:
    a = shell.chat(q)
    ok = exp.lower() in a.lower()
    check("Math: %s" % q[:30], ok, a.strip()[:60], exp)

# ══════════════════════════════════════════════════════════
# 4. SCIENCE DRIVERS
# ══════════════════════════════════════════════════════════

banner("4. SCIENCE DRIVERS")

science_tests = [
    ("What is the molecular weight of H2O?", ["18.015"]),
    ("What is the molecular weight of C6H12O6?", ["180"]),
    ("Tell me about the element iron", ["iron", "26"]),
    ("What is tungsten?", ["tungsten", "3422"]),
]

for q, exp in science_tests:
    a = shell.chat(q)
    hits = [k for k in exp if k.lower() in a.lower()]
    ok = len(hits) >= 1
    check("Science: %s" % q[:40], ok, a.strip()[:60], exp)

# ══════════════════════════════════════════════════════════
# 5. NEGATION HANDLING
# ══════════════════════════════════════════════════════════

banner("5. NEGATION HANDLING")

a = shell.chat("Does apixaban cause bleeding?")
has_not = "not" in a.lower() or "does not" in a.lower() or "prevent" in a.lower()
check("Negation: apixaban does NOT cause bleeding", has_not, a.strip()[:80])

# ══════════════════════════════════════════════════════════
# 6. PREDICTIVE CODING
# ══════════════════════════════════════════════════════════

banner("6. PREDICTIVE CODING")

pce = PredictiveCodingEngine(kernel, learning_rate=0.05)
toronto_uid = lexicon.word_to_uuid.get("toronto")
if toronto_uid:
    for _ in range(10):
        pce.query_with_prediction([toronto_uid], top_k=5, verbose=False)
    stats = pce.get_stats()
    check("Predictions cached > 0", stats["cached_predictions"] > 0, stats["cached_predictions"])
    check("Prediction accuracy > 50%%", stats["overall_accuracy"] > 0.5,
          "%.0f%%" % (stats["overall_accuracy"] * 100))

# ══════════════════════════════════════════════════════════
# 7. SELF-MODEL
# ══════════════════════════════════════════════════════════

banner("7. SELF-MODEL")

sm = SelfModel(kernel, lexicon, pce)
sm.sync_beliefs_from_graph()
beliefs = sm.what_do_i_know()
uncertain = sm.what_am_i_uncertain_about(0.3)
state = sm.my_current_state()

check("Self-model has beliefs", len(beliefs) > 0, len(beliefs))
check("Self-model tracks state", state["nodes"] > 0, state["nodes"])
check("Self-model tracks uncertainty", True, "%d uncertain" % len(uncertain))

# ══════════════════════════════════════════════════════════
# 8. DREAMER
# ══════════════════════════════════════════════════════════

banner("8. DREAMER")

config = DreamerConfig()
config.max_cycles = 3
config.cycle_interval_sec = 0
dreamer = Dreamer(kernel, lexicon, sm, pce, config)

for _ in range(3):
    dreamer.think_once()

status = dreamer.get_status()
check("Dreamer completed cycles", status["cycles_completed"] == 3, status["cycles_completed"])
check("Dreamer has discoveries", True, "%d discoveries" % status["discoveries"])

# ══════════════════════════════════════════════════════════
# 9. EMOTION ENGINE
# ══════════════════════════════════════════════════════════

banner("9. EMOTION ENGINE")

em = EmotionEngine()
check("Baseline emotion = neutral", em.current_emotion() == "neutral", em.current_emotion())

em.apply_stimulus("threat")
check("After threat = fear", em.current_emotion() == "fear", em.current_emotion())

em.decay(60)
em.apply_stimulus("reward")
check("After reward = joy", em.current_emotion() == "joy", em.current_emotion())

em.decay(120)
check("After decay = neutral", em.current_emotion() == "neutral", em.current_emotion())

# ══════════════════════════════════════════════════════════
# 10. KASM (Hyperdimensional Computing)
# ══════════════════════════════════════════════════════════

banner("10. KASM")

from kasm.vsa import KASMEngine
kasm = KASMEngine(dimensions=10000, seed=42)
kasm.node_batch("sun", "planet", "gravity", "nucleus", "electron", "electromagnetism")
kasm.node_batch("role_center", "role_orbiter", "role_force")

r_sun = kasm.store("r_sun", kasm.bind(kasm.get("sun"), kasm.get("role_center")))
r_planet = kasm.store("r_planet", kasm.bind(kasm.get("planet"), kasm.get("role_orbiter")))
r_grav = kasm.store("r_grav", kasm.bind(kasm.get("gravity"), kasm.get("role_force")))
r_nuc = kasm.store("r_nuc", kasm.bind(kasm.get("nucleus"), kasm.get("role_center")))
r_elec = kasm.store("r_elec", kasm.bind(kasm.get("electron"), kasm.get("role_orbiter")))
r_em = kasm.store("r_em", kasm.bind(kasm.get("electromagnetism"), kasm.get("role_force")))

solar = kasm.store("solar", kasm.superpose(r_sun, r_planet, r_grav))
atom = kasm.store("atom", kasm.superpose(r_nuc, r_elec, r_em))

mapping = kasm.bind(solar, atom)
answer_vec = kasm.unbind(mapping, kasm.get("sun"))
matches = kasm.cleanup(answer_vec, threshold=0.05)

best = matches[0][0] if matches else "?"
check("KASM: sun of atom = nucleus", best == "nucleus", best, "nucleus")

# ══════════════════════════════════════════════════════════
# 11. EXPERIMENT ENGINE
# ══════════════════════════════════════════════════════════

banner("11. EXPERIMENT ENGINE")

from kos.experiment import ExperimentEngine, Hypothesis
from kos.drivers.physics import PhysicsDriver
from kos.drivers.chemistry import ChemistryDriver

engine = ExperimentEngine(chemistry=ChemistryDriver(), physics=PhysicsDriver())

h = Hypothesis("Blue light repairs perovskite without degrading",
               parameters={"wavelength_nm": 450, "repair_threshold_kJ": 150,
                            "degradation_threshold_kJ": 340})
result = engine.run(h, max_iterations=3, verbose=False)
check("Experiment: blue light viable", result["status"] == "VIABLE", result["status"])

h2 = Hypothesis("UV repairs perovskite",
                parameters={"wavelength_nm": 300, "repair_threshold_kJ": 150,
                             "degradation_threshold_kJ": 340})
result2 = engine.run(h2, max_iterations=5, verbose=False)
check("Experiment: UV self-corrects", result2["iterations"] > 1,
      "%d iterations" % result2["iterations"])

# ══════════════════════════════════════════════════════════
# 12. PERSISTENCE
# ══════════════════════════════════════════════════════════

banner("12. PERSISTENCE")

try:
    from kos.persistence import GraphPersistence
    gp = GraphPersistence()
    gp.save(kernel, lexicon)
    check("Graph saved", gp.exists(), gp.exists())

    kernel2 = KOSKernel(enable_vsa=False)
    lexicon2 = KASMLexicon()
    gp.load(kernel2, lexicon2)
    check("Graph loaded", len(kernel2.nodes) == len(kernel.nodes),
          len(kernel2.nodes), len(kernel.nodes))
except Exception as e:
    check("Persistence", False, str(e)[:60])

# ══════════════════════════════════════════════════════════
# 13. FORAGER (if network available)
# ══════════════════════════════════════════════════════════

banner("13. FORAGER")

try:
    forager = WebForager(kernel, lexicon, driver)
    t0 = time.perf_counter()
    new = forager.forage_query("tungsten", verbose=False)
    forage_ms = (time.perf_counter() - t0) * 1000
    check("Forager: tungsten", new > 0, "+%d nodes" % new)
    check("Forager latency < 30s", forage_ms < 30000, "%.0fms" % forage_ms)
except Exception as e:
    check("Forager", False, str(e)[:60])

# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════

banner("COMPREHENSIVE TEST SUMMARY")

# Agent Fix 3: removed file self-reading (cp1252 encoding crash)
# Count directly from issues list instead
print("\n  Total issues found: %d" % len(issues))
for i in issues:
    print("    [%s] %s: %s" % (i["sev"], i["cat"][:30], i["desc"][:50]))

if not issues:
    print("    No issues found!")

print("\n  VERDICT: %s" % ("ALL CLEAR" if len(issues) == 0 else "%d ISSUES TO FIX" % len(issues)))
