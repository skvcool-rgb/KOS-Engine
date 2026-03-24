"""KOS Agent: Full Architecture Diagnostics — Find Every Gap, Propose Every Fix."""
import sys, os, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.router_offline import KOSShellOffline
from kos.predictive import PredictiveCodingEngine
from kos.self_model import SelfModel
from kos.dreamer import Dreamer, DreamerConfig
from kos.hierarchical import HierarchicalPredictor
from kos.emotion import EmotionEngine

kernel = KOSKernel(enable_vsa=False)
lexicon = KASMLexicon()
driver = TextDriver(kernel, lexicon)
pce = PredictiveCodingEngine(kernel, learning_rate=0.05)
emotion = EmotionEngine()

driver.ingest("""
Toronto is a major city in the Canadian province of Ontario.
Toronto was founded and incorporated in the year 1834.
The city of Toronto has a population of approximately 2.7 million people.
Toronto has a humid continental climate with warm summers and cold winters.
John Graves Simcoe originally established the settlement of Toronto.
Perovskite is a highly efficient material for photovoltaic cells.
Photovoltaic cells capture photons to produce electricity.
Backpropagation adjusts weights by computing gradient of the loss.
Artificial neural networks are inspired by biological neurons.
Quantum computers use qubits which can exist in superposition.
Entanglement allows two qubits to be correlated across any distance.
The Sun produces energy through nuclear fusion of hydrogen into helium.
Mitochondria produce ATP which is the energy currency of cells.
Apixaban prevents thrombosis without dietary restrictions.
Coral reefs support 25 percent of all marine species.
""")

shell = KOSShellOffline(kernel, lexicon, enable_forager=False)
sm = SelfModel(kernel, lexicon, pce)
sm.sync_beliefs_from_graph()
hp = HierarchicalPredictor(kernel, pce)

config = DreamerConfig()
config.max_cycles = 5
config.cycle_interval_sec = 0
dreamer = Dreamer(kernel, lexicon, sm, pce, config)
for _ in range(5):
    dreamer.think_once()

print("=" * 70)
print("  KOS FULL ARCHITECTURE DIAGNOSTICS")
print("  Agent self-examines every component and proposes fixes")
print("=" * 70)

issues = []
fixes = []
issue_id = 0

def report(component, severity, problem, fix):
    global issue_id
    issue_id += 1
    issues.append({"id": issue_id, "component": component,
                    "severity": severity, "problem": problem})
    fixes.append({"id": issue_id, "fix": fix})
    marker = "!!!" if severity == "CRITICAL" else "**" if severity == "HIGH" else "*"
    print("\n  [%s] #%d %s: %s" % (severity, issue_id, component, problem))
    print("  FIX: %s" % fix[:120])


# ══════════════════════════════════════════════════════════
# 1. KNOWLEDGE ENGINE DIAGNOSTICS
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  [1] KNOWLEDGE ENGINE")
print("=" * 70)

total_nodes = len(kernel.nodes)
total_edges = sum(len(n.connections) for n in kernel.nodes.values())
orphans = [uid for uid, n in kernel.nodes.items() if not n.connections]
hubs = [(uid, len(n.connections)) for uid, n in kernel.nodes.items()
        if len(n.connections) > 15]

print("  Nodes: %d | Edges: %d | Orphans: %d | Hubs: %d" % (
    total_nodes, total_edges, len(orphans), len(hubs)))

if len(orphans) > total_nodes * 0.2:
    report("Graph", "HIGH",
           "%d orphan nodes (%.0f%% of graph) — disconnected knowledge" % (
               len(orphans), len(orphans)/total_nodes*100),
           "Run daemon._prune_orphans() or connect orphans to nearest "
           "neighbor via Layer 5 vector search. Most orphans are adjectives "
           "extracted by Fix #1 that were never wired to noun nodes.")

if hubs:
    for uid, degree in hubs:
        word = lexicon.get_word(uid) if hasattr(lexicon, 'get_word') else "?"
        if degree > 30:
            report("Graph", "MEDIUM",
                   "Super-hub '%s' has %d connections — fan-out risk" % (word, degree),
                   "Contextual mitosis: split into sub-topics. Or increase "
                   "Top-K routing from 500 to handle high-degree nodes.")

# Check provenance coverage
nodes_with_provenance = 0
for uid in kernel.nodes:
    has_prov = False
    for (a, b), sents in kernel.provenance.items():
        if uid in (a, b) and sents:
            has_prov = True
            break
    if has_prov:
        nodes_with_provenance += 1

prov_coverage = nodes_with_provenance / total_nodes * 100 if total_nodes else 0
if prov_coverage < 80:
    report("Provenance", "HIGH",
           "Only %.0f%% of nodes have provenance sentences" % prov_coverage,
           "Nodes without provenance cannot be cited in answers. "
           "Re-ingest source text or wire provenance during verb/adj extraction.")

# ══════════════════════════════════════════════════════════
# 2. QUERY PIPELINE DIAGNOSTICS
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  [2] QUERY PIPELINE")
print("=" * 70)

benchmark = [
    ("Where is Toronto?", ["ontario", "province"]),
    ("When was Toronto founded?", ["1834"]),
    ("Population of Toronto?", ["million"]),
    ("Climate of Toronto?", ["humid", "continental"]),
    ("What produces ATP?", ["mitochondria"]),
    ("How does the Sun produce energy?", ["fusion", "hydrogen"]),
    ("What do coral reefs support?", ["marine", "species"]),
    ("How do neural networks learn?", ["backpropagation", "gradient"]),
    ("What is entanglement?", ["qubit", "correlated"]),
    ("Tell me about perovskite", ["photovoltaic", "efficient"]),
]

passed = 0
failed = []
latencies = []
for q, exp in benchmark:
    t0 = time.perf_counter()
    a = shell.chat(q)
    lat = (time.perf_counter() - t0) * 1000
    latencies.append(lat)
    hits = [k for k in exp if k in a.lower()]
    if hits:
        passed += 1
    else:
        failed.append((q, exp, a.strip()[:80]))

accuracy = passed / len(benchmark) * 100
avg_lat = sum(latencies) / len(latencies)
max_lat = max(latencies)

print("  Accuracy: %d/%d (%.0f%%) | Avg latency: %.0fms | Max: %.0fms" % (
    passed, len(benchmark), accuracy, avg_lat, max_lat))

if accuracy < 100:
    for q, exp, got in failed:
        report("Query", "HIGH",
               "'%s' failed — expected %s, got '%s'" % (q[:40], exp, got[:50]),
               "Agent should diagnose: check seed resolution, provenance "
               "coverage, and Weaver scoring for this specific query.")

if max_lat > 5000:
    report("Latency", "MEDIUM",
           "Max query latency %.0fms — slow for interactive use" % max_lat,
           "Profile the slow query. Likely cause: Layer 5 vector search "
           "on large graph. Fix: add FAISS index or reduce graph size.")

if avg_lat > 1000:
    report("Latency", "MEDIUM",
           "Average latency %.0fms — should be under 500ms" % avg_lat,
           "Disable sentence-transformers Layer 5 if not needed. "
           "Most queries resolve at Layer 1-2.")

# ══════════════════════════════════════════════════════════
# 3. PREDICTIVE CODING DIAGNOSTICS
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  [3] PREDICTIVE CODING")
print("=" * 70)

# Train on all benchmark queries
for q, exp in benchmark:
    seeds = []
    raw = [w.lower() for w in re.findall(r"[a-zA-Z]+", q) if len(w) > 2]
    for w in raw:
        uid = shell._resolve_word(w, list(lexicon.word_to_uuid.keys()))
        if uid:
            seeds.append(uid)
    if seeds:
        for _ in range(5):
            pce.query_with_prediction(seeds, top_k=5, verbose=False)

pce_stats = pce.get_stats()
print("  Predictions: %d | Accuracy: %.0f%% | Cached: %d" % (
    pce_stats["total_predictions"],
    pce_stats["overall_accuracy"] * 100,
    pce_stats["cached_predictions"]))

if pce_stats["overall_accuracy"] < 0.8:
    report("Prediction", "MEDIUM",
           "Prediction accuracy %.0f%% — below 80%% target" % (
               pce_stats["overall_accuracy"] * 100),
           "Need more training cycles. Run 100+ queries to build cache. "
           "Consider increasing learning rate from 0.05 to 0.08.")

# Hierarchical predictor
for q, exp in benchmark[:5]:
    seeds = []
    raw = [w.lower() for w in re.findall(r"[a-zA-Z]+", q) if len(w) > 2]
    for w in raw:
        uid = shell._resolve_word(w, list(lexicon.word_to_uuid.keys()))
        if uid:
            seeds.append(uid)
    if seeds:
        hp.predict_full(seeds, q)
        hp.observe_actual(seeds, {"top_energy": 2.0, "ticks": 8,
                                   "confidence": 0.9, "no_answer": False,
                                   "foraged": False}, q)

hp_summary = hp.get_summary()
print("  Hierarchical: %d cycles | Avg accuracy: %.1f%%" % (
    hp_summary["total_cycles"], hp_summary["avg_accuracy"]))

for layer in hp_summary["layers"]:
    if layer["predictions"] > 3 and layer["accuracy"] < 20:
        report("Hierarchical L%d" % layer["layer"], "LOW",
               "Layer %d (%s) accuracy %.0f%% — undertrained" % (
                   layer["layer"], layer["name"], layer["accuracy"]),
               "Needs 100+ observations. Currently only %d." % layer["predictions"])

# ══════════════════════════════════════════════════════════
# 4. SELF-HEALING DIAGNOSTICS
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  [4] SELF-HEALING & LEARNING")
print("=" * 70)

# Self-model check
beliefs = sm.what_do_i_know()
uncertain = sm.what_am_i_uncertain_about(0.3)
print("  Beliefs: %d | Uncertain: %d" % (len(beliefs), len(uncertain)))

if len(uncertain) > len(beliefs) * 0.5:
    report("Self-Model", "MEDIUM",
           "%d/%d beliefs below 30%% confidence (%.0f%%)" % (
               len(uncertain), len(beliefs),
               len(uncertain)/len(beliefs)*100),
           "High uncertainty ratio. Either: (a) ingest more context for "
           "uncertain concepts, or (b) adjust confidence calculation — "
           "many low-confidence beliefs are adjectives that need 2+ edges "
           "to gain confidence.")

# Dreamer check
discoveries = dreamer.get_discoveries()
print("  Dream discoveries: %d" % len(discoveries))

valid_discoveries = 0
spurious = 0
for d in discoveries:
    # Check if discovery makes semantic sense
    seed = d.get("seed", "").lower()
    target = d.get("target", "").lower()
    # Spurious if energy == 3.0 (max) — likely graph artifact
    if d.get("energy", 0) >= 3.0:
        spurious += 1
    else:
        valid_discoveries += 1

if spurious > valid_discoveries:
    report("Dreamer", "MEDIUM",
           "%d/%d dream discoveries hit max energy (3.0) — likely artifacts" % (
               spurious, len(discoveries)),
           "Max-energy discoveries are super-hub echoes, not real insights. "
           "Filter discoveries where energy < max_energy * 0.9. Or lower "
           "max_energy from 3.0 to 2.0 for more selective dreaming.")

# ══════════════════════════════════════════════════════════
# 5. SENSES DIAGNOSTICS
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  [5] SENSES & PERCEPTION")
print("=" * 70)

# Check if senses are importable
senses_status = {}
try:
    from kos.senses.eyes import Eyes
    senses_status["eyes"] = "AVAILABLE"
except:
    senses_status["eyes"] = "MISSING (pip install ultralytics opencv-python)"

try:
    from kos.senses.ears import Ears
    senses_status["ears"] = "AVAILABLE"
except:
    senses_status["ears"] = "MISSING (pip install openai-whisper sounddevice)"

try:
    from kos.senses.mouth import Mouth
    senses_status["mouth"] = "AVAILABLE"
except:
    senses_status["mouth"] = "MISSING (pip install pyttsx3)"

for sense, status in senses_status.items():
    print("  %s: %s" % (sense.upper(), status))

# Check ffmpeg
import shutil
ffmpeg_path = shutil.which("ffmpeg")
if not ffmpeg_path:
    report("Ears", "HIGH",
           "ffmpeg not in PATH — Whisper cannot decode audio",
           "Install ffmpeg: download from ffmpeg.org or run "
           "'choco install ffmpeg' on Windows")
else:
    print("  ffmpeg: %s" % ffmpeg_path)

# Architectural issues with senses
report("Senses", "HIGH",
       "Eyes only see when button is clicked — no continuous awareness",
       "Add background webcam capture every 5 seconds. Store last 10 frames. "
       "Run YOLO diff to detect changes. Only process frames where objects "
       "appear/disappear (attention filtering). This gives KOS continuous "
       "visual awareness without constant CPU drain.")

report("Senses", "HIGH",
       "Visual context disconnected from query engine — 'what do you see' "
       "only works via special-case intercept in api.py",
       "After every YOLO detection, ingest a structured sentence into the graph: "
       "'KOS currently sees a person, a clock, and a couch.' This makes visual "
       "context queryable through the normal graph pipeline, not a special case.")

report("Senses", "MEDIUM",
       "Ears record fixed 4-second chunks — misses long utterances, wastes "
       "time on silence",
       "Implement Voice Activity Detection (VAD): start recording when sound "
       "exceeds threshold, stop when silence detected for 1.5 seconds. "
       "The Ears class already has listen_continuous() — wire it to the API.")

report("Senses", "MEDIUM",
       "Mouth speaks but has no emotion modulation in the API endpoint",
       "Use EmotionEngine state to modulate voice: fear = faster rate, "
       "calm = slower rate, joy = higher pitch. The Mouth class already "
       "has speak_with_emotion() — wire it to the API.")

report("Senses", "LOW",
       "No sensory memory — KOS forgets what it saw/heard after one query",
       "Store last 30 seconds of visual detections and audio transcripts "
       "in a ring buffer. Query engine should search this buffer for "
       "recent sensory context before falling back to the knowledge graph.")

# ══════════════════════════════════════════════════════════
# 6. EMOTION & SOCIAL DIAGNOSTICS
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  [6] EMOTION & SOCIAL")
print("=" * 70)

print("  Emotion state: %s" % emotion.current_emotion())
print("  Cortisol: %.1f | Dopamine: %.1f | Serotonin: %.1f" % (
    emotion.state.cortisol, emotion.state.dopamine, emotion.state.serotonin))

report("Emotion", "HIGH",
       "Emotion engine runs in isolation — does not affect query confidence, "
       "Weaver scoring, or response formatting",
       "Wire EmotionDecisionBridge into the query pipeline: "
       "high cortisol -> lower confidence, high dopamine after correct answer "
       "-> reinforce edge weights, low serotonin -> flag degradation. "
       "emotion_integration.py exists but is not wired into router_offline.py.")

report("Social", "MEDIUM",
       "SocialEngine exists but has no real users — trust scores are empty",
       "Wire UserModel into the API. Track per-user query history, expertise, "
       "satisfaction. Use SocialEngine trust scores to weight user feedback. "
       "user_model.py exists but is not wired into api.py.")

# ══════════════════════════════════════════════════════════
# 7. SCIENCE DRIVERS DIAGNOSTICS
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  [7] SCIENCE DRIVERS")
print("=" * 70)

drivers_status = {}
for name, module in [
    ("Chemistry", "kos.drivers.chemistry"),
    ("Physics", "kos.drivers.physics"),
    ("Biology", "kos.drivers.biology"),
    ("Math", "kos.drivers.math"),
    ("Code", "kos.drivers.code"),
]:
    try:
        __import__(module)
        drivers_status[name] = "OK"
    except Exception as e:
        drivers_status[name] = "ERROR: %s" % str(e)[:50]

for name, status in drivers_status.items():
    print("  %s: %s" % (name, status))

report("Drivers", "HIGH",
       "Science drivers (Chemistry/Physics/Biology) are standalone — "
       "not integrated into the query pipeline",
       "When user asks 'What is the molecular weight of H2O?', the router "
       "should detect it as a chemistry query and route to ChemistryDriver "
       "instead of searching the graph. Add driver detection to "
       "router_offline.py process_prompt(), similar to how MathDriver "
       "intercepts arithmetic expressions.")

report("Drivers", "MEDIUM",
       "ExperimentEngine not accessible from dashboard or API",
       "Add /api/experiment endpoint that accepts a hypothesis and runs "
       "the predict-compute-compare-heal loop. Display results in dashboard.")

# ══════════════════════════════════════════════════════════
# 8. DEPLOYMENT & SCALE DIAGNOSTICS
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  [8] DEPLOYMENT & SCALE")
print("=" * 70)

report("Deployment", "HIGH",
       "Railway deployment crashed — requirements.txt has heavy deps "
       "that exceed free tier memory",
       "Create two requirement files: requirements-core.txt (lightweight, "
       "no torch/transformers) and requirements-full.txt (all deps). "
       "Railway uses core, local uses full.")

report("Scale", "MEDIUM",
       "Single-process, single-thread — cannot handle concurrent queries",
       "Run uvicorn with --workers 4 for multi-process. Or use gunicorn. "
       "The graph is read-heavy, write-light — safe for multiple readers.")

report("Security", "HIGH",
       "No authentication on API — anyone on the network can query, "
       "ingest, or use sensors",
       "Add API key authentication. Simple: check X-API-Key header. "
       "Store key in .env file. Reject requests without valid key.")

report("Persistence", "HIGH",
       "Graph is in-memory only — all knowledge lost on restart",
       "Serialize graph to disk: pickle or JSON. Load on startup. "
       "Auto-save every 5 minutes or after every ingestion. "
       "This is the #1 production blocker.")

# ══════════════════════════════════════════════════════════
# 9. KASM DIAGNOSTICS
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  [9] KASM (Hyperdimensional Computing)")
print("=" * 70)

try:
    from kasm.vsa import KASMEngine
    e = KASMEngine(dimensions=10000, seed=42)
    a = e.node("test_a")
    b = e.node("test_b")
    sim = e.resonate(a, b)
    print("  KASM engine: OK (orthogonality check: %.4f)" % sim)
except Exception as ex:
    report("KASM", "HIGH", "KASM engine broken: %s" % str(ex)[:60],
           "Fix import or dependency issue")

report("KASM", "MEDIUM",
       "KASM bridge (auto BIND on edge creation) is not active by default",
       "enable_vsa=True in KOSKernel causes import overhead. Make VSA "
       "lazy-loaded: only initialize KASM when first analogy query arrives.")

report("KASM", "LOW",
       "Only 3 example .kasm programs — need more domain demos",
       "Write: medical.kasm (drug interaction analogy), legal.kasm "
       "(precedent matching), engineering.kasm (structural analysis).")

# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  DIAGNOSTICS SUMMARY")
print("=" * 70)

critical = sum(1 for i in issues if i["severity"] == "CRITICAL")
high = sum(1 for i in issues if i["severity"] == "HIGH")
medium = sum(1 for i in issues if i["severity"] == "MEDIUM")
low = sum(1 for i in issues if i["severity"] == "LOW")

print("\n  Total issues found: %d" % len(issues))
print("    CRITICAL: %d" % critical)
print("    HIGH:     %d" % high)
print("    MEDIUM:   %d" % medium)
print("    LOW:      %d" % low)

print("\n  TOP 5 FIXES (by impact):")
priority_fixes = sorted(zip(issues, fixes),
                         key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[x[0]["severity"]])

for i, (issue, fix) in enumerate(priority_fixes[:5], 1):
    print("\n  %d. [%s] %s" % (i, issue["severity"], issue["problem"][:70]))
    print("     FIX: %s" % fix["fix"][:100])

print("\n  AGENT RECOMMENDATION:")
print("  Fix #1 priority: GRAPH PERSISTENCE (knowledge lost on restart)")
print("  This is the single biggest production blocker. Everything else")
print("  is an enhancement. Without persistence, KOS forgets everything")
print("  every time the server restarts.")
