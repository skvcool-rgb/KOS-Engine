"""
KOS V5.1 — FastAPI Backend

REST API for the KOS Cognitive Agent.
Serves the HTML dashboard and provides endpoints for:
- Querying the knowledge graph
- Ingesting new knowledge
- Running health checks
- Monitoring agent status
- Task assignment

Run: uvicorn api:app --host 0.0.0.0 --port 8080 --reload
"""

import os
import sys
import time
import re
import asyncio
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.router_offline import KOSShellOffline, warm_preload_all
from kos.weaver import AlgorithmicWeaver
from kos.self_improve import SelfImprover
from kos.feedback import WeaverFeedback, FormulaLearner, ContinuousTuner, AnalogyScanner
from kos.predictive import PredictiveCodingEngine

# ── Warm preload: start loading heavy models in background ──
# SentenceTransformer loads in parallel with graph construction below.
warm_preload_all()

# Optional imports (may not be installed yet)
try:
    from kos.user_model import UserModel
    user_model = UserModel()
except Exception:
    user_model = None

try:
    from kos.persistence import GraphPersistence
    graph_persistence = GraphPersistence()
except Exception:
    graph_persistence = None

# ══════════════════════════════════════════════════════════
# BOOT THE KOS ENGINE
# ══════════════════════════════════════════════════════════

kernel = KOSKernel(enable_vsa=False)
lexicon = KASMLexicon()
driver = TextDriver(kernel, lexicon)
shell = KOSShellOffline(kernel, lexicon, enable_forager=False)
pce = PredictiveCodingEngine(kernel, learning_rate=0.05)
weaver = AlgorithmicWeaver()
feedback = WeaverFeedback(weaver)
improver = SelfImprover(kernel, lexicon, shell)
scanner = AnalogyScanner(kernel, lexicon)
formula_learner = FormulaLearner()

# ── v0.6 Pipeline Components ──
try:
    from kos.drivers.math import MathDriver
    math_driver = MathDriver()
except Exception:
    math_driver = None

def _forager_factory():
    """Create a fresh WebForager instance (avoids circular imports)."""
    from kos.forager import WebForager
    return WebForager(kernel, lexicon, driver)

try:
    from kos.query_pipeline import QueryPipeline
    pipeline = QueryPipeline(
        kernel=kernel, lexicon=lexicon, shell=shell,
        weaver=weaver, reranker=None, synthesizer=None,
        relevance_scorer=None, math_driver=math_driver,
        forager_factory=_forager_factory,
    )
except Exception as e:
    print(f"[KOS] Pipeline init error: {e}")
    pipeline = None

# v0.8: Mission Manager
try:
    from kos.mission import MissionManager
    def _pipeline_query(prompt):
        p = _get_pipeline()
        if p:
            return p.query(prompt)
        return {"answer": "Pipeline unavailable", "relevance_score": 0}
    mission_manager = MissionManager(
        query_fn=_pipeline_query,
        persist_path=".cache/missions.json",
    )
    print(f"[KOS] Mission Manager loaded ({len(mission_manager)} missions)")
except Exception as e:
    print(f"[KOS] Mission Manager init error: {e}")
    mission_manager = None

# Auto-seed benchmark corpus
SEED_CORPUS = """
Toronto is a major city in the Canadian province of Ontario.
Toronto was founded and incorporated in the year 1834.
The city of Toronto has a population of approximately 2.7 million people.
Toronto has a humid continental climate with warm summers and cold winters.
John Graves Simcoe originally established the settlement of Toronto.
The CN Tower is a famous landmark in downtown Toronto.
Toronto is the financial capital of Canada with many banks.
Perovskite is a highly efficient material for photovoltaic cells.
Photovoltaic cells capture photons to produce electricity.
Perovskite is remarkably cheap and affordable to produce.
Silicon is a traditional semiconductor for computing.
Apixaban prevents thrombosis without dietary restrictions.
Apixaban does not cause bleeding in patients.
Unlike warfarin, apixaban is a modern anticoagulant.
Montreal was founded in the year 1642.
Montreal has a population of 1.7 million.
Backpropagation adjusts weights by computing gradient of the loss.
Artificial neural networks are inspired by biological neurons.
Quantum computers use qubits which can exist in superposition.
Entanglement allows two qubits to be correlated across any distance.
The Sun produces energy through nuclear fusion of hydrogen into helium.
Mitochondria produce ATP which is the energy currency of cells.
Coral reefs support 25 percent of all marine species.
The human heart pumps blood through arteries and veins.
Water consists of two hydrogen atoms bonded to one oxygen atom.
Electrolysis splits water into hydrogen and oxygen using electricity.
DNA contains the genetic instructions for building proteins.
Einstein special relativity states nothing travels faster than light.
Forward time travel is proven real via time dilation at high speeds.
Backward time travel has never been observed experimentally.
Newton second law states that force equals mass times acceleration.
Newton first law states an object at rest stays at rest unless acted on by a force.
Gravity is the force that attracts objects with mass toward each other.
The speed of light in a vacuum is approximately 300000 kilometers per second.
A covalent bond is a chemical bond formed by sharing electrons between atoms.
An ionic bond is formed by the transfer of electrons between atoms.
The periodic table organizes elements by their atomic number and properties.
Oxygen is the most abundant element in the Earth crust by mass.
Photosynthesis converts carbon dioxide and water into glucose using sunlight.
Humans digest food through mechanical and chemical breakdown in the stomach and intestines.
The human brain contains approximately 86 billion neurons.
Python is a popular programming language used for artificial intelligence.
An algorithm is a step by step procedure for solving a problem.
Machine learning enables computers to learn patterns from data without explicit programming.
The derivative measures the rate of change of a function.
Pi is approximately 3.14159 and represents the ratio of circumference to diameter.
The Pythagorean theorem states that a squared plus b squared equals c squared.
Climate change is caused by increasing greenhouse gas concentrations in the atmosphere.
The Earth orbits the Sun at an average distance of 150 million kilometers.
Momentum is the product of mass and velocity of a moving object.
The periodic table organizes all known chemical elements by atomic number.
DNA is a double helix molecule that stores genetic information in all living organisms.
Coral reefs are underwater ecosystems built by colonies of tiny animals called coral polyps.
Machine learning is a branch of artificial intelligence that uses data to improve predictions.
Encryption converts readable data into coded form to protect information from unauthorized access.
The water cycle describes how water evaporates, forms clouds, and falls as precipitation.
Artificial intelligence is the simulation of human intelligence by computer systems.
Solar energy comes from the Sun and can be converted to electricity using photovoltaic panels.
"""
# Try to load saved graph (restores previous session state)
# NOTE: Disabled for demo — large graphs (3000+ nodes) slow down queries.
# Re-enable for production use.
# if graph_persistence:
#     try:
#         graph_persistence.load(kernel, lexicon)
#     except Exception:
#         pass  # No saved graph yet, that's fine

# Seed corpus ingested AFTER persistence load so it's always available
driver.ingest(SEED_CORPUS)

# State tracking
query_log = []
task_log = []
health_log = []
boot_time = time.time()

# ══════════════════════════════════════════════════════════
# PROPOSAL EXECUTOR — Apply approved proposals to live system
# ══════════════════════════════════════════════════════════

def apply_approved_proposals():
    """Apply all approved but unapplied proposals to the running system."""
    import json as jmod
    proposals_dir = Path(__file__).parent / "proposals"
    if not proposals_dir.exists():
        return {"applied": 0, "errors": 0}

    applied = 0
    errors = 0
    config_path = Path(__file__).parent / ".cache" / "self_tuned_config.json"
    config_path.parent.mkdir(exist_ok=True)

    # Load existing config
    config = {}
    if config_path.exists():
        try:
            config = jmod.load(open(config_path, "r", encoding="utf-8"))
        except Exception:
            pass

    for f in sorted(proposals_dir.glob("*.json")):
        try:
            p = jmod.load(open(f, "r", encoding="utf-8"))
        except Exception:
            continue

        if p.get("status") != "APPROVED" or p.get("applied"):
            continue

        ptype = p.get("type", "")

        try:
            if ptype == "threshold_change":
                # Apply threshold to config
                desc = p.get("description", "")
                m = re.match(r'Change (\w+): ([\d.]+) -> ([\d.]+)', desc)
                if m:
                    param, old_val, new_val = m.group(1), m.group(2), m.group(3)
                    config[param] = {"value": float(new_val), "old": float(old_val)}
                    applied += 1

            elif ptype == "synonym_addition":
                # Apply synonym to lexicon
                code = p.get("code", "")
                pairs = re.findall(r'"(\w+)".*?"(\w+)"', code)
                for a, b in pairs:
                    uuid_a = lexicon.get_uuid(a) if hasattr(lexicon, 'get_uuid') else None
                    uuid_b = lexicon.get_uuid(b) if hasattr(lexicon, 'get_uuid') else None
                    if uuid_a and uuid_b and uuid_a in kernel.nodes and uuid_b in kernel.nodes:
                        kernel.link(uuid_a, uuid_b, weight=0.5)
                applied += 1

            elif ptype == "weaver_rule":
                # Store weaver rules in config for later use
                config.setdefault("weaver_rules", [])
                config["weaver_rules"].append(p.get("description", ""))
                applied += 1

            elif ptype == "daemon_strategy":
                # Log daemon strategies (these are architectural proposals)
                config.setdefault("daemon_strategies", [])
                config["daemon_strategies"].append(p.get("description", ""))
                applied += 1

            # Mark as applied
            p["applied"] = True
            p["applied_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            with open(f, "w", encoding="utf-8") as fh:
                jmod.dump(p, fh, indent=2)

        except Exception:
            errors += 1

    # Save updated config
    try:
        with open(config_path, "w", encoding="utf-8") as fh:
            jmod.dump(config, fh, indent=2)
    except Exception:
        pass

    return {"applied": applied, "errors": errors, "config_keys": list(config.keys())}


# ══════════════════════════════════════════════════════════
# SELF-REPAIR LOOP — Auto-detect and fix failures
# ══════════════════════════════════════════════════════════

import threading

_engine_lock = threading.Lock()  # Prevents concurrent shell/kernel access
_repair_log = []

def _self_repair_loop():
    """Background thread — lightweight stats + structural fixes every 120s.
    Skips heavy operations (benchmark, rebalance) when agent is active to avoid GIL contention."""

    while True:
        try:
            time.sleep(120)

            agent_active = _agent and hasattr(_agent, '_running') and _agent._running
            action = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "actions_taken": [],
                "nodes": len(kernel.nodes),
                "agent_active": agent_active,
            }

            if agent_active:
                # Agent is running — only log lightweight stats, no kernel mutations
                action["actions_taken"].append(
                    f"Stats: {len(kernel.nodes)} nodes (agent active, skipping heavy ops)")
            else:
                # Agent idle — safe to run full structural repairs
                try:
                    rebalance = improver.rebalance_degrees(verbose=False)
                    action["actions_taken"].append(
                        f"Rebalanced: {rebalance.get('hubs_fixed',0)} hubs, "
                        f"{rebalance.get('orphans_fixed',0)} orphans")
                except Exception as e:
                    action["actions_taken"].append(f"Rebalance error: {e}")

                try:
                    norm = improver.normalize_weights(verbose=False)
                    action["actions_taken"].append(
                        f"Normalized: {norm.get('clipped',0)} weights clipped")
                except Exception as e:
                    action["actions_taken"].append(f"Normalize error: {e}")

                try:
                    result = improver.run_benchmark(verbose=False)
                    action["accuracy"] = result.get('accuracy', 0)
                    action["failed"] = result.get('failed', [])
                    action["actions_taken"].append(
                        f"Benchmark: {result.get('passed',0)}/{result.get('total',0)}")
                except Exception:
                    pass

            _repair_log.append(action)
            if len(_repair_log) > 100:
                _repair_log.pop(0)

        except Exception:
            pass


# ══════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════

app = FastAPI(title="KOS Agent API", version="7.0.0")


@app.on_event("startup")
async def _on_startup():
    """Force-load SentenceTransformer at server start so first query is fast."""
    from kos.router_offline import _get_embedder
    _get_embedder()  # Blocks until model is loaded
    # Pre-build embeddings for the seed graph
    shell._ensure_embeddings()

    # Apply all approved proposals
    result = apply_approved_proposals()
    print(f"[KOS] Applied {result['applied']} proposals ({result['errors']} errors)")

    # Start self-repair background thread
    t = threading.Thread(target=_self_repair_loop, daemon=True, name="kos-self-repair")
    t.start()
    print("[KOS] Self-repair loop started (checks every 60s)")
    print("[KOS] Models warm. Ready to serve.")

# ── API Key Authentication ────────────────────────────
KOS_API_KEY = os.environ.get("KOS_API_KEY", None)

if KOS_API_KEY:
    class APIKeyMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if request.url.path.startswith("/api/"):
                key = request.headers.get("X-API-Key")
                if key != KOS_API_KEY:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing API key"},
                    )
            return await call_next(request)

    app.add_middleware(APIKeyMiddleware)

# Serve static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Models ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    prompt: str

class IngestRequest(BaseModel):
    text: str

class TaskRequest(BaseModel):
    task: str
    priority: Optional[str] = "normal"


# ── Dashboard HTML ───────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_path = Path(__file__).parent / "static" / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Dashboard not found. Place dashboard.html in static/</h1>")


# ── API Endpoints ────────────────────────────────────────

_status_cache = {
    "status": "STARTING",
    "uptime_seconds": 0,
    "uptime_human": "0h 0m 0s",
    "nodes": 0, "edges": 0, "orphans": 0, "super_hubs": 0,
    "queries_total": 0, "avg_latency_ms": 0,
    "predictions_cached": 0, "prediction_accuracy": 0.0,
    "contradictions": 0, "tasks_pending": 0, "tasks_completed": 0,
    "agent_running": False, "agent_cycles": 0, "agent_nodes_learned": 0,
    "repair_cycles": 0,
}

def _status_update_loop():
    """Background thread that refreshes _status_cache every 5s.
    This keeps the /api/status endpoint instant — it always returns the cache."""
    global _status_cache
    while True:
        try:
            time.sleep(5)
            now = time.time()
            uptime = now - boot_time

            # These are cheap O(1) lookups
            node_count = len(kernel.nodes)

            # Expensive iteration — wrap in try/except for concurrent modification
            try:
                nodes_snapshot = list(kernel.nodes.values())
                total_edges = sum(len(n.connections) for n in nodes_snapshot)
                orphans = sum(1 for n in nodes_snapshot if not n.connections)
                hubs = sum(1 for n in nodes_snapshot if len(n.connections) > 15)
            except (RuntimeError, TypeError):
                total_edges = _status_cache.get("edges", 0)
                orphans = _status_cache.get("orphans", 0)
                hubs = _status_cache.get("super_hubs", 0)

            # Agent stats
            agent_queries = 0
            agent_nodes_learned = 0
            agent_cycles = 0
            agent_running = False
            try:
                if _agent and hasattr(_agent, 'get_status'):
                    a_st = _agent.get_status()
                    agent_cycles = a_st.get('cycle', 0)
                    agent_nodes_learned = a_st.get('nodes_learned', 0)
                    agent_running = a_st.get('running', False)
                    agent_queries = a_st.get('topics_foraged', 0)
            except Exception:
                pass

            repair_cycles = len(_repair_log)
            last_benchmark_acc = 0
            if _repair_log:
                last_benchmark_acc = _repair_log[-1].get('accuracy', 0) * 100

            total_queries = len(query_log) + agent_queries
            avg_lat = round(sum(q['latency_ms'] for q in query_log) / len(query_log), 1) if query_log else 0

            try:
                pce_stats = pce.get_stats()
                pred_acc = round(pce_stats['overall_accuracy'] * 100, 1)
                pred_cached = pce_stats['cached_predictions']
            except Exception:
                pred_acc = 0.0
                pred_cached = 0

            if pred_acc == 0 and last_benchmark_acc > 0:
                pred_acc = round(last_benchmark_acc, 1)

            _status_cache = {
                "status": "ONLINE",
                "uptime_seconds": round(uptime, 1),
                "uptime_human": f"{int(uptime//3600)}h {int((uptime%3600)//60)}m {int(uptime%60)}s",
                "nodes": node_count,
                "edges": total_edges,
                "orphans": orphans,
                "super_hubs": hubs,
                "queries_total": total_queries,
                "avg_latency_ms": avg_lat,
                "predictions_cached": pred_cached,
                "prediction_accuracy": pred_acc,
                "contradictions": len(kernel.contradictions),
                "tasks_pending": sum(1 for t in task_log if t['status'] == 'pending'),
                "tasks_completed": sum(1 for t in task_log if t['status'] == 'completed'),
                "agent_running": agent_running,
                "agent_cycles": agent_cycles,
                "agent_nodes_learned": agent_nodes_learned,
                "repair_cycles": repair_cycles,
            }
        except Exception:
            pass

# Start status updater thread at module load
threading.Thread(target=_status_update_loop, daemon=True, name="status-updater").start()

@app.get("/api/status")
async def get_status():
    """System overview — always returns instantly from background-updated cache."""
    return _status_cache


_NO_DATA_PHRASES = [
    "i don't have data",
    "no data on this",
    "i don't have information",
    "no information on",
    "i don't know",
    "cannot answer",
    "no answer found",
]

# ── v0.6 Query Pipeline (lazy singleton) ──────────────────────────
_pipeline_v6 = None

def _get_pipeline():
    """Build the v0.6 pipeline (lazy singleton)."""
    global _pipeline_v6
    if _pipeline_v6 is not None:
        return _pipeline_v6

    from kos.query_pipeline import QueryPipeline

    _pipeline_v6 = QueryPipeline(
        kernel=kernel, lexicon=lexicon, shell=shell,
        weaver=weaver, reranker=None, synthesizer=None,
        relevance_scorer=None, math_driver=math_driver,
        forager_factory=_forager_factory,
    )
    return _pipeline_v6


@app.post("/api/query")
def query(req: QueryRequest):
    """
    v0.6 Query Pipeline:
      Route -> Retrieve -> Rerank -> Synthesize -> Decision Gate -> [Forage] -> Answer

      Graph retrieves what is true.
      Reranker selects what is relevant.
      Synthesizer explains what matters.
      Confidence gate decides whether to speak.
    """
    p = _get_pipeline()
    result = p.query(req.prompt, verbose=True)

    entry = {
        "prompt": req.prompt,
        "answer": result["answer"],
        "trust_label": result.get("trust_label", "unverified"),
        "latency_ms": result["latency_ms"],
        "timestamp": time.time(),
        "nodes_activated": result.get("nodes_activated", len(kernel.nodes)),
        "source": result["source"],
        "foraged_nodes": result["foraged_nodes"],
        "relevance_score": result["relevance_score"],
        "relevance_breakdown": result.get("relevance_breakdown", {}),
        "coverage_factor": result.get("coverage_factor", 1.0),
        "off_topic_detected": result.get("off_topic_detected", False),
        "evidence_count": result.get("evidence_count", 0),
        "coverage_gaps": result.get("coverage_gaps", []),
        "route": result.get("route", {}),
        "stages": result.get("stages", {}),
    }
    query_log.append(entry)

    # Track feedback (re-ask detection)
    if len(query_log) >= 2:
        prev = query_log[-2]['prompt'].lower().split()
        curr = req.prompt.lower().split()
        overlap = len(set(prev) & set(curr))
        if overlap >= 2 and len(curr) >= 2:
            feedback._reask_count += 1

    # Update user model
    if user_model:
        try:
            user_model.update_from_interaction("default", req.prompt, result["answer"], True)
        except Exception:
            pass

    return entry


@app.post("/api/query/stream")
def query_stream(req: QueryRequest):
    """
    v0.6 Streaming Query — returns all pipeline events as JSON.
    (SSE endpoint will be /api/query/sse in v0.7)
    """
    from kos.stream_manager import StreamManager
    p = _get_pipeline()
    stream = StreamManager(req.prompt)
    p.query(req.prompt, stream=stream, verbose=True)
    return stream.to_json()


@app.get("/api/memory/stats")
async def memory_stats():
    """v0.7 Episodic memory statistics — scores, failures, coverage gaps."""
    p = _get_pipeline()
    return p.memory.stats()


@app.get("/api/memory/recent")
async def memory_recent():
    """v0.7 Last 20 query episodes."""
    p = _get_pipeline()
    return {"episodes": p.memory.recent(20)}


@app.get("/api/memory/failures")
async def memory_failures():
    """v0.7 Recent failed queries for debugging."""
    p = _get_pipeline()
    return {"failures": p.memory.recent_failures(10)}


# ══════════════════════════════════════════════════════════
# v0.8 MISSION ENDPOINTS
# ══════════════════════════════════════════════════════════

class MissionCreateRequest(BaseModel):
    name: str
    description: str = ""
    tags: list = []
    deadline: float = None

class MissionPlanRequest(BaseModel):
    goals: list = None  # Optional list of goal specs

class GoalAddRequest(BaseModel):
    description: str
    goal_type: str = "retrieve"
    query: str = None
    dependencies: list = []
    priority: int = 1

class CheckpointAddRequest(BaseModel):
    description: str
    required_goals: list = []
    deadline: float = None
    condition: str = ""

@app.post("/api/missions")
async def create_mission(req: MissionCreateRequest):
    """v0.8 Create a new mission."""
    if mission_manager is None:
        return JSONResponse({"error": "Mission manager not available"}, 500)
    m = mission_manager.create_mission(
        name=req.name, description=req.description or req.name,
        tags=req.tags, deadline=req.deadline)
    return {"mission": m.to_dict()}

@app.get("/api/missions")
async def list_missions(status: str = None):
    """v0.8 List all missions, optionally filtered by status."""
    if mission_manager is None:
        return JSONResponse({"error": "Mission manager not available"}, 500)
    return {"missions": mission_manager.list_missions(status=status)}

@app.get("/api/missions/{mission_id}")
async def get_mission(mission_id: str):
    """v0.8 Get full mission state."""
    if mission_manager is None:
        return JSONResponse({"error": "Mission manager not available"}, 500)
    try:
        return {"mission": mission_manager.get_mission(mission_id)}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, 404)

@app.post("/api/missions/{mission_id}/plan")
async def plan_mission(mission_id: str, req: MissionPlanRequest = None):
    """v0.8 Decompose mission into goals. Auto-decomposes if no goals provided."""
    if mission_manager is None:
        return JSONResponse({"error": "Mission manager not available"}, 500)
    try:
        goals = req.goals if req else None
        m = mission_manager.plan(mission_id, goals=goals)
        return {"mission": m.to_dict()}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, 404)

@app.post("/api/missions/{mission_id}/execute")
def execute_mission_step(mission_id: str):
    """v0.8 Execute the next ready goal in the mission (sync, runs in threadpool)."""
    if mission_manager is None:
        return JSONResponse({"error": "Mission manager not available"}, 500)
    try:
        result = mission_manager.execute_step(mission_id, verbose=True)
        return {"result": result}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, 404)

@app.post("/api/missions/{mission_id}/execute_all")
def execute_mission_all(mission_id: str):
    """v0.8 Execute all goals until mission completes or stalls (sync, runs in threadpool)."""
    if mission_manager is None:
        return JSONResponse({"error": "Mission manager not available"}, 500)
    try:
        result = mission_manager.execute_all(mission_id, verbose=True)
        return result
    except ValueError as e:
        return JSONResponse({"error": str(e)}, 404)

@app.post("/api/missions/{mission_id}/goals")
async def add_mission_goal(mission_id: str, req: GoalAddRequest):
    """v0.8 Add a goal to an existing mission."""
    if mission_manager is None:
        return JSONResponse({"error": "Mission manager not available"}, 500)
    try:
        goal = mission_manager.add_goal(
            mission_id, description=req.description,
            goal_type=req.goal_type, query=req.query,
            dependencies=req.dependencies, priority=req.priority)
        return {"goal": goal}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, 404)

@app.post("/api/missions/{mission_id}/checkpoints")
async def add_mission_checkpoint(mission_id: str, req: CheckpointAddRequest):
    """v0.8 Add a checkpoint to a mission."""
    if mission_manager is None:
        return JSONResponse({"error": "Mission manager not available"}, 500)
    try:
        cp = mission_manager.add_checkpoint(
            mission_id, description=req.description,
            required_goals=req.required_goals,
            deadline=req.deadline, condition=req.condition)
        return {"checkpoint": cp}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, 404)

@app.post("/api/missions/{mission_id}/pause")
async def pause_mission(mission_id: str):
    """v0.8 Pause a mission."""
    if mission_manager is None:
        return JSONResponse({"error": "Mission manager not available"}, 500)
    mission_manager.pause(mission_id)
    return {"status": "paused"}

@app.post("/api/missions/{mission_id}/resume")
async def resume_mission(mission_id: str):
    """v0.8 Resume a paused mission."""
    if mission_manager is None:
        return JSONResponse({"error": "Mission manager not available"}, 500)
    mission_manager.resume(mission_id)
    return {"status": "resumed"}

@app.post("/api/missions/{mission_id}/cancel")
async def cancel_mission(mission_id: str):
    """v0.8 Cancel a mission."""
    if mission_manager is None:
        return JSONResponse({"error": "Mission manager not available"}, 500)
    mission_manager.cancel(mission_id)
    return {"status": "cancelled"}


@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    """Ingest new knowledge into the graph."""
    t0 = time.perf_counter()
    nodes_before = len(kernel.nodes)
    driver.ingest(req.text)
    nodes_after = len(kernel.nodes)
    latency = (time.perf_counter() - t0) * 1000

    # Auto-save graph after ingest
    if graph_persistence:
        try:
            graph_persistence.save(kernel, lexicon)
        except Exception:
            pass

    return {
        "status": "OK",
        "nodes_before": nodes_before,
        "nodes_after": nodes_after,
        "nodes_added": nodes_after - nodes_before,
        "latency_ms": round(latency, 1),
    }


_health_running = False
_health_results = None

def _run_health_thread():
    """Run health check in a background thread. Results cached for dashboard polling."""
    global _health_running, _health_results
    try:
        t0 = time.perf_counter()
        stats = improver.improve(verbose=False, quick=True)
        elapsed = (time.perf_counter() - t0) * 1000

        pce_stats = pce.get_stats()
        _health_results = {
            "status": "completed",
            "benchmark": stats.get("benchmark", {}),
            "time_ms": round(elapsed, 1),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rebalance": stats.get("rebalance", {}),
            "normalization": stats.get("normalization", {}),
            "contradictions": stats.get("contradictions", {}),
            "feedback": stats.get("feedback", {}),
            "formulas": stats.get("formulas", {}),
            "predictive": {
                "cached": pce_stats.get("cached_predictions", 0),
                "accuracy": round(pce_stats.get("overall_accuracy", 0) * 100, 1),
                "adjustments": pce_stats.get("total_weight_adjustments", 0),
            },
        }
    except Exception as e:
        _health_results = {"status": "error", "message": str(e)}
    finally:
        _health_running = False

@app.get("/api/health")
async def health_check():
    """Run benchmark in a background thread. Dashboard polls /api/health/last.
    Note: GIL means status endpoint may be slow during benchmark (~10s total).
    The status background cache ensures the dashboard recovers quickly."""
    global _health_running
    if _health_running:
        return {"status": "running", "message": "Health check already in progress."}

    _health_running = True
    t = threading.Thread(target=_run_health_thread, daemon=True, name="health-check")
    t.start()
    return {"status": "started", "message": "Health check running in background (~10s)."}

@app.get("/api/health/last")
async def last_health():
    """Dashboard polls this every 3s to see if benchmark finished."""
    if _health_running:
        return {"running": True, "result": {"status": "running"}}
    if _health_results is None:
        return {"running": False, "result": {"status": "not_run_yet"}}
    return {"running": False, "result": _health_results}


@app.get("/api/graph")
async def get_graph():
    """Get graph structure for visualization."""
    nodes = []
    edges = []

    for nid, node in kernel.nodes.items():
        word = lexicon.get_word(nid) if hasattr(lexicon, 'get_word') else str(nid)
        nodes.append({
            "id": str(nid),
            "label": word or str(nid)[:8],
            "connections": len(node.connections),
            "activation": round(node.activation, 3),
        })
        for tgt, data in node.connections.items():
            tgt_word = lexicon.get_word(tgt) if hasattr(lexicon, 'get_word') else str(tgt)
            edges.append({
                "source": str(nid),
                "target": str(tgt),
                "weight": round(data['w'], 3),
                "myelin": data.get('myelin', 0),
            })

    return {"nodes": nodes, "edges": edges}


@app.get("/api/graph/top_nodes")
async def get_top_nodes():
    """Get top 20 most connected nodes."""
    ranked = sorted(
        kernel.nodes.items(),
        key=lambda x: len(x[1].connections),
        reverse=True
    )[:20]

    return [{
        "word": lexicon.get_word(nid) or str(nid)[:8],
        "connections": len(node.connections),
        "activation": round(node.activation, 3),
        "myelin_total": sum(d.get('myelin', 0) if isinstance(d, dict) else 0 for d in node.connections.values()),
    } for nid, node in ranked]


@app.post("/api/task")
async def create_task(req: TaskRequest):
    """Assign a task to the KOS agent."""
    task = {
        "id": len(task_log) + 1,
        "task": req.task,
        "priority": req.priority,
        "status": "pending",
        "created": time.time(),
        "result": None,
    }

    # Auto-execute certain task types
    task_lower = req.task.lower().strip()
    if task_lower.startswith("ingest:"):
        url_or_text = req.task[7:].strip()
        nodes_before = len(kernel.nodes)
        driver.ingest(url_or_text)
        task['status'] = 'completed'
        task['result'] = f"Ingested. Nodes: {nodes_before} -> {len(kernel.nodes)}"
    elif task_lower.startswith("query:"):
        query_text = req.task[6:].strip()
        answer = shell.chat(query_text)
        task['status'] = 'completed'
        task['result'] = answer.strip()
    elif task_lower.startswith("health"):
        result = improver.improve(verbose=False)
        task['status'] = 'completed'
        task['result'] = f"Accuracy: {result.get('benchmark', {}).get('accuracy', 0):.0%}"
    else:
        # Auto-answer any question-like task via the knowledge graph
        try:
            answer = shell.chat(req.task)
            task['status'] = 'completed'
            task['result'] = answer.strip() if answer else 'No answer found'
        except Exception as e:
            task['status'] = 'pending'
            task['result'] = f'Processing error: {str(e)[:80]}'

    task_log.append(task)
    return task


@app.get("/api/tasks")
async def get_tasks():
    """Get all tasks."""
    return task_log[-20:]  # Last 20 tasks


@app.get("/api/queries")
async def get_queries():
    """Get query history."""
    return query_log[-50:]  # Last 50 queries


@app.get("/api/user/profile")
async def get_user_profile():
    """Get current user profile."""
    if user_model:
        try:
            profile = user_model.get_profile("default")
            return {
                "user_id": profile.user_id,
                "expertise": profile.expertise,
                "detail_level": profile.detail_level,
                "domains": list(profile.domains) if hasattr(profile, 'domains') else [],
                "query_count": profile.query_count if hasattr(profile, 'query_count') else 0,
                "satisfaction": profile.satisfaction if hasattr(profile, 'satisfaction') else 0,
            }
        except Exception as e:
            return {"error": str(e)[:100]}
    return {"error": "UserModel not available"}


@app.get("/api/health/history")
async def get_health_history():
    """Get health check history."""
    return health_log[-20:]


@app.get("/api/feedback/stats")
async def get_feedback_stats():
    """Get user feedback / satisfaction stats."""
    return feedback.get_stats()


@app.get("/api/contradictions")
async def get_contradictions():
    """Get all detected contradictions."""
    try:
        contras = getattr(kernel, 'contradictions', [])
        if isinstance(contras, dict):
            return [{"node_a": str(a), "node_b": str(b), "details": str(d)}
                    for (a, b), d in contras.items()]
        elif isinstance(contras, list):
            return [{"details": str(c)} for c in contras]
        return []
    except Exception:
        return []


# ══════════════════════════════════════════════════════════
# PERSISTENCE ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.get("/api/save")
async def save_graph():
    """Save the current graph to disk."""
    if graph_persistence:
        try:
            graph_persistence.save(kernel, lexicon)
            return {"status": "OK", "nodes": len(kernel.nodes)}
        except Exception as e:
            return {"status": "ERROR", "error": str(e)[:100]}
    return {"status": "ERROR", "error": "GraphPersistence not available"}


@app.get("/api/load")
async def load_graph():
    """Load graph from disk."""
    if graph_persistence:
        try:
            graph_persistence.load(kernel, lexicon)
            return {"status": "OK", "nodes": len(kernel.nodes)}
        except Exception as e:
            return {"status": "ERROR", "error": str(e)[:100]}
    return {"status": "ERROR", "error": "GraphPersistence not available"}


# ══════════════════════════════════════════════════════════
# EXPERIMENT ENDPOINT
# ══════════════════════════════════════════════════════════

class ExperimentRequest(BaseModel):
    statement: str
    parameters: dict = {}

@app.post("/api/experiment")
async def run_experiment(req: ExperimentRequest):
    """Run a hypothesis through the ExperimentEngine."""
    try:
        from kos.experiment import ExperimentEngine, Hypothesis
    except ImportError:
        return {"error": "ExperimentEngine not available"}

    # Build science drivers
    chem_driver = None
    phys_driver = None
    try:
        from kos.drivers.chemistry import ChemistryDriver
        chem_driver = ChemistryDriver()
    except Exception:
        pass
    try:
        from kos.drivers.physics import PhysicsDriver
        phys_driver = PhysicsDriver()
    except Exception:
        pass

    hypothesis = Hypothesis(statement=req.statement, parameters=req.parameters)
    engine = ExperimentEngine(
        chemistry=chem_driver, physics=phys_driver,
        kernel=kernel, lexicon=lexicon
    )
    result = engine.run(hypothesis, max_iterations=10, verbose=False)

    # Serialize result (ExperimentResult objects are not JSON-serializable)
    serialized = {
        "status": result.get("status", "UNKNOWN"),
        "iterations": result.get("iterations", 0),
        "confidence": result.get("confidence", 0),
        "hypothesis": str(result.get("final_hypothesis", "")),
    }
    return serialized


# ══════════════════════════════════════════════════════════
# AUTONOMOUS AGENT ENDPOINTS
# ══════════════════════════════════════════════════════════

_agent = None

def _get_agent():
    global _agent
    if _agent is None:
        try:
            from kos.autonomous import AutonomousAgent
            from kos.auto_improve import AutoImprover
            from kos.dreamer import Dreamer, DreamerConfig
            from kos.self_model import SelfModel
            from kos.predictive import PredictiveCodingEngine

            _pce = PredictiveCodingEngine(kernel, learning_rate=0.05)
            _sm = SelfModel(kernel, lexicon, _pce)
            _sm.sync_beliefs_from_graph()

            _cfg = DreamerConfig()
            _cfg.max_cycles = 200
            _cfg.cycle_interval_sec = 0
            _cfg.curiosity_probability = 0.8
            _dreamer = Dreamer(kernel, lexicon, _sm, _pce, _cfg)

            _ai = AutoImprover(kernel, lexicon, shell, _pce)

            _forager = None
            try:
                from kos.forager import WebForager
                _forager = WebForager(kernel, lexicon, driver)
            except Exception:
                pass

            _agent = AutonomousAgent(
                kernel, lexicon, shell, driver,
                forager=_forager, auto_improver=_ai,
                dreamer=_dreamer, self_model=_sm, pce=_pce)

            # Directed learning curriculum
            _agent.learning_curriculum = [
                # Self-knowledge
                "artificial intelligence self improvement",
                "knowledge graph spreading activation",
                "hyperdimensional computing vector symbolic architecture",
                "predictive coding free energy principle",
                "neuromorphic computing Intel Loihi",
                # Human challenges
                "global energy crisis renewable solutions",
                "climate change carbon dioxide solutions",
                "water scarcity desalination technology",
                "air pollution health effects solutions",
                "ocean plastic pollution cleanup technology",
                "deforestation reforestation carbon capture",
                "sustainable agriculture food security",
                "nuclear fusion energy ITER tokamak",
                "solar energy perovskite efficiency record",
                "hydrogen fuel cell green hydrogen production",
                "battery technology solid state lithium",
                "resource depletion circular economy recycling",
                "biodiversity loss species extinction conservation",
                "microplastics health environmental impact",
                "carbon capture direct air technology",
            ]
        except Exception as e:
            print("[AGENT] Init error: %s" % e)
    return _agent

@app.post("/api/agent/start")
def agent_start():
    """Start the autonomous agent."""
    agent = _get_agent()
    if not agent:
        return {"error": "Agent init failed"}
    if agent._running:
        return {"status": "already_running", **agent.get_status()}
    agent.max_cycles = 1000
    agent.cycle_interval_sec = 15
    agent._running = False  # Reset if previously stopped
    agent._cycle = 0
    agent.run_background(max_cycles=1000, cycle_interval=15, verbose=True)
    return {"status": "started", **agent.get_status()}

@app.post("/api/agent/stop")
def agent_stop():
    agent = _get_agent()
    if agent:
        agent.stop()
        return {"status": "stopped", **agent.get_status()}
    return {"error": "No agent"}

@app.post("/api/agent/pause")
def agent_pause():
    agent = _get_agent()
    if agent:
        agent.pause()
        return {"status": "paused"}
    return {"error": "No agent"}

@app.post("/api/agent/resume")
def agent_resume():
    agent = _get_agent()
    if agent:
        agent.resume()
        return {"status": "resumed"}
    return {"error": "No agent"}

@app.get("/api/agent/status")
def agent_status():
    agent = _get_agent()
    if agent:
        return agent.get_status()
    return {"running": False, "error": "No agent"}

@app.get("/api/agent/events")
def agent_events():
    agent = _get_agent()
    if agent:
        return agent.get_events(30)
    return []

@app.get("/api/agent/foraged")
def agent_foraged():
    agent = _get_agent()
    if agent:
        return agent.get_foraged(20)
    return []

@app.get("/api/agent/proposals")
def agent_proposals():
    agent = _get_agent()
    if agent:
        return agent.get_queued_proposals()
    return []

@app.post("/api/agent/approve/{index}")
def agent_approve(index: int):
    agent = _get_agent()
    if agent:
        return agent.approve_proposal(index)
    return {"error": "No agent"}

@app.post("/api/agent/reject/{index}")
def agent_reject(index: int):
    agent = _get_agent()
    if agent:
        return agent.reject_proposal(index)
    return {"error": "No agent"}


# ══════════════════════════════════════════════════════════
# SENSOR ENDPOINTS
# ══════════════════════════════════════════════════════════

# Lazy sensor state
_sensors = {"eyes": None, "ears": None, "mouth": None, "emotion": None}
_last_visual = []  # Store last YOLO detections
_last_audio = ""   # Store last transcription

def _get_emotion():
    if _sensors["emotion"] is None:
        from kos.emotion import EmotionEngine
        _sensors["emotion"] = EmotionEngine()
    return _sensors["emotion"]

# ══════════════════════════════════════════════════════════
# DOMAIN AGENTS (Agent Factory) ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.get("/api/domain_agents")
async def list_domain_agents():
    """List all agents from the agent registry."""
    registry_path = Path(__file__).parent / "agents" / "registry.json"
    if not registry_path.exists():
        return {"agents": [], "total": 0}
    try:
        import json as jsonmod
        with open(registry_path, "r", encoding="utf-8") as f:
            data = jsonmod.load(f)
        agents = data.get("agents", [])
        # Enrich with blueprint info
        for a in agents:
            bp_path = Path(__file__).parent / "agents" / (a["id"] + ".json")
            if bp_path.exists():
                with open(bp_path, "r", encoding="utf-8") as f:
                    bp = jsonmod.load(f)
                a["description"] = bp.get("description", "")
                a["capabilities"] = bp.get("capabilities", [])
                a["core_concepts"] = len(bp.get("knowledge_nodes", []))
        return {"agents": agents, "total": len(agents), "last_updated": data.get("last_updated")}
    except Exception as e:
        return {"agents": [], "error": str(e)[:100]}


@app.post("/api/domain_agents/approve/{agent_id}")
async def approve_domain_agent(agent_id: str):
    """Approve a pending agent."""
    registry_path = Path(__file__).parent / "agents" / "registry.json"
    if not registry_path.exists():
        return {"error": "No registry"}
    import json as jsonmod
    with open(registry_path, "r", encoding="utf-8") as f:
        data = jsonmod.load(f)
    for a in data.get("agents", []):
        if a["id"] == agent_id:
            a["status"] = "APPROVED"
            a["approved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            break
    else:
        return {"error": "Agent not found"}
    with open(registry_path, "w", encoding="utf-8") as f:
        jsonmod.dump(data, f, indent=2)
    return {"status": "OK", "agent_id": agent_id}


@app.post("/api/domain_agents/reject/{agent_id}")
async def reject_domain_agent(agent_id: str):
    """Reject a pending agent."""
    registry_path = Path(__file__).parent / "agents" / "registry.json"
    if not registry_path.exists():
        return {"error": "No registry"}
    import json as jsonmod
    with open(registry_path, "r", encoding="utf-8") as f:
        data = jsonmod.load(f)
    for a in data.get("agents", []):
        if a["id"] == agent_id:
            a["status"] = "REJECTED"
            break
    else:
        return {"error": "Agent not found"}
    with open(registry_path, "w", encoding="utf-8") as f:
        jsonmod.dump(data, f, indent=2)
    return {"status": "OK", "agent_id": agent_id}


@app.get("/api/domain_agents/monitor")
async def domain_agents_monitor():
    """Get latest agent monitoring report."""
    report_path = Path(__file__).parent / "agents" / "monitor_report.json"
    if not report_path.exists():
        return {"error": "No monitor report yet"}
    import json as jsonmod
    with open(report_path, "r", encoding="utf-8") as f:
        return jsonmod.load(f)


_agent_registry = None

def _get_agent_registry():
    global _agent_registry
    if _agent_registry is None:
        try:
            from kos.agent_factory import AgentRegistry
            _agent_registry = AgentRegistry(kernel, lexicon, shell)
            # Auto-load all approved agents
            for a in _agent_registry.list_agents():
                if a.get("status") == "APPROVED":
                    try:
                        _agent_registry.load_agent(a["id"])
                    except Exception:
                        pass
        except Exception:
            pass
    return _agent_registry

@app.post("/api/domain_agents/route")
async def route_to_agent(req: QueryRequest):
    """Route a query to the best domain agent. Uses agent for domain classification, shell for answer."""
    try:
        # Domain keyword matching for routing (more reliable than agent scoring)
        _DOMAIN_KEYWORDS = {
            "physics": {"newton", "gravity", "force", "mass", "velocity", "acceleration",
                        "energy", "momentum", "quantum", "relativity", "photon", "wave",
                        "particle", "electron", "proton", "neutron", "thermodynamics",
                        "entropy", "magnetic", "electric", "light", "optics", "nuclear",
                        "maxwell", "schrodinger", "dirac", "electromagnetic", "navier",
                        "stokes", "fluid", "turbulence", "mechanics", "dynamics"},
            "chemistry": {"molecule", "atom", "bond", "reaction", "element", "compound",
                          "acid", "base", "ion", "periodic", "oxidation", "catalyst",
                          "solution", "ph", "organic", "inorganic", "polymer", "isotope",
                          "molar", "arrhenius", "gibbs", "enthalpy", "entropy",
                          "valence", "covalent", "chemical", "stoichiometry"},
            "computer science": {"algorithm", "computer", "software", "programming", "code",
                                 "data structure", "binary", "compiler", "cpu", "memory",
                                 "neural network", "machine learning", "database", "encryption"},
            "mathematics": {"equation", "theorem", "calculus", "integral", "derivative",
                           "algebra", "geometry", "probability", "statistics", "matrix",
                           "function", "prime", "logarithm", "polynomial", "topology",
                           "trigonometry", "sine", "cosine", "tangent", "proof", "conjecture",
                           "hypothesis", "zeta", "manifold", "elliptic", "number theory",
                           "riemann", "euler", "fourier", "laplace", "differential"},
            "general knowledge": {"who", "what", "where", "when", "why", "how",
                                  "history", "geography", "culture", "language", "economy"},
        }

        # Find best domain match
        ql = req.prompt.lower()
        best_domain = None
        best_match = 0
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in ql)
            if matches > best_match:
                best_match = matches
                best_domain = domain

        # Map domain to agent name
        _DOMAIN_TO_AGENT = {
            "physics": "physicsAgent",
            "chemistry": "chemistryAgent",
            "computer science": "computerscienceAgent",
            "mathematics": "mathematicsAgent",
            "general knowledge": "generalknowledgeAgent",
        }

        # Always use the KOS shell for the actual answer (most reliable)
        try:
            answer = shell.chat(req.prompt)
        except Exception:
            answer = ""

        agent_name = _DOMAIN_TO_AGENT.get(best_domain, "KOS Core") if best_match > 0 else "KOS Core"
        domain = best_domain or "general"

        # Check if answer is useful
        answer_lower = (answer or "").strip().lower()
        has_answer = answer and len(answer.strip()) > 15 and not any(
            p in answer_lower for p in _NO_DATA_PHRASES
        )

        if has_answer:
            return {
                "agent_name": agent_name,
                "domain": domain,
                "answer": answer.strip(),
                "confidence": 0.9 if best_match > 0 else 0.5,
                "routing_score": min(best_match * 0.3, 1.0),
            }

        # Fallback: Internet foraging
        foraged_nodes = 0
        try:
            from kos.forager import WebForager
            forager = WebForager(kernel, lexicon, driver)
            nodes_before = len(kernel.nodes)
            forager.forage_query(req.prompt, verbose=False)
            foraged_nodes = len(kernel.nodes) - nodes_before
            if foraged_nodes > 0:
                shell.node_embeddings = None
                shell.embedded_uuids = []
                shell._word_emb_cache = {}
                shell._ensure_embeddings()
                answer = shell.chat(req.prompt)
        except Exception:
            pass

        # Use keyword-matched agent even for foraged answers
        foraged_agent = agent_name if best_match > 0 else "KOS Forager"
        foraged_domain = domain if best_match > 0 else "internet"

        return {
            "agent_name": foraged_agent,
            "domain": foraged_domain,
            "answer": answer.strip() if answer else "No answer found",
            "confidence": 0.5 if best_match > 0 else 0.3,
            "routing_score": min(best_match * 0.3, 1.0) if best_match > 0 else 0.0,
            "fallback": "internet" if foraged_nodes > 0 else None,
            "foraged_nodes": foraged_nodes,
        }
    except Exception as e:
        return {"error": str(e)[:200]}


# ══════════════════════════════════════════════════════════
# SELF-REPAIR & PROPOSAL ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.get("/api/repair_log")
async def get_repair_log():
    """Get the self-repair loop log."""
    return {"entries": _repair_log[-20:], "total": len(_repair_log)}


@app.post("/api/proposals/apply_all")
async def apply_all_proposals():
    """Apply all approved but unapplied proposals."""
    result = apply_approved_proposals()
    return result


# ══════════════════════════════════════════════════════════
# CODE PROPOSALS ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.get("/api/proposals")
async def list_proposals():
    """List all code proposals from the proposer."""
    proposals_dir = Path(__file__).parent / "proposals"
    if not proposals_dir.exists():
        return {"proposals": [], "total": 0}
    import json as jsonmod
    proposals = []
    for f in sorted(proposals_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                p = jsonmod.load(fh)
            p["filename"] = f.name
            proposals.append(p)
        except Exception:
            pass
    return {
        "proposals": proposals[:200],
        "total": len(proposals),
        "pending": sum(1 for p in proposals if p.get("status") == "PENDING"),
        "approved": sum(1 for p in proposals if p.get("status") == "APPROVED"),
        "rejected": sum(1 for p in proposals if p.get("status") == "REJECTED"),
    }


@app.post("/api/proposals/approve/{proposal_id}")
async def approve_proposal_endpoint(proposal_id: str):
    """Approve a code proposal."""
    proposals_dir = Path(__file__).parent / "proposals"
    import json as jsonmod
    pfile = proposals_dir / (proposal_id + ".json")
    if not pfile.exists():
        return {"error": "Proposal not found"}
    with open(pfile, "r", encoding="utf-8") as f:
        p = jsonmod.load(f)
    p["status"] = "APPROVED"
    p["approved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(pfile, "w", encoding="utf-8") as f:
        jsonmod.dump(p, f, indent=2)
    # Log approval
    log_path = proposals_dir / "approved.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("%s APPROVED %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), proposal_id))
    return {"status": "OK", "proposal_id": proposal_id}


@app.post("/api/proposals/reject/{proposal_id}")
async def reject_proposal_endpoint(proposal_id: str):
    """Reject a code proposal."""
    proposals_dir = Path(__file__).parent / "proposals"
    import json as jsonmod
    pfile = proposals_dir / (proposal_id + ".json")
    if not pfile.exists():
        return {"error": "Proposal not found"}
    with open(pfile, "r", encoding="utf-8") as f:
        p = jsonmod.load(f)
    p["status"] = "REJECTED"
    with open(pfile, "w", encoding="utf-8") as f:
        jsonmod.dump(p, f, indent=2)
    return {"status": "OK", "proposal_id": proposal_id}


# ══════════════════════════════════════════════════════════
# ENGINE LOG TAILING
# ══════════════════════════════════════════════════════════

@app.get("/api/engine_log")
async def get_engine_log():
    """Tail the kos_6hr.log file for live monitoring."""
    log_path = Path(__file__).parent / "kos_6hr.log"
    if not log_path.exists():
        return {"lines": [], "total_lines": 0}
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        # Return last 200 lines
        tail = lines[-200:]
        return {"lines": [l.rstrip() for l in tail], "total_lines": len(lines)}
    except Exception as e:
        return {"lines": [], "error": str(e)[:100]}


@app.get("/api/engine_log/search")
async def search_engine_log(q: str = ""):
    """Search the engine log for a keyword."""
    log_path = Path(__file__).parent / "kos_6hr.log"
    if not log_path.exists() or not q:
        return {"lines": [], "total": 0}
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        matches = [l.rstrip() for l in lines if q.lower() in l.lower()]
        return {"lines": matches[-100:], "total": len(matches), "query": q}
    except Exception as e:
        return {"lines": [], "error": str(e)[:100]}


# ══════════════════════════════════════════════════════════
# SENSOR ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.post("/api/sensors/speak")
def sensor_speak(req: QueryRequest):
    """Make KOS speak text through speakers."""
    import threading
    def _speak(text):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            # Emotion-modulated speech rate
            em_state = _get_emotion().current_emotion()
            rate = 160  # default
            if em_state == "fear": rate = 200
            elif em_state == "calm": rate = 140
            elif em_state == "joy": rate = 175
            elif em_state == "depression": rate = 120
            engine.setProperty('rate', rate)
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    t = threading.Thread(target=_speak, args=(req.prompt,), daemon=True)
    t.start()
    return {"status": "OK", "text": req.prompt, "engine": "pyttsx3"}

@app.post("/api/sensors/see")
def sensor_see():
    """Capture one frame from webcam and detect objects."""
    try:
        import cv2
        from ultralytics import YOLO

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return {"status": "ERROR", "error": "Cannot read webcam"}

        h, w = frame.shape[:2]

        # Save frame
        frame_path = str(Path(__file__).parent / "static" / "last_frame.jpg")
        cv2.imwrite(frame_path, frame)

        # YOLO detection
        model = YOLO("yolov8n.pt")
        results = model(frame, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                detections.append({
                    "label": label,
                    "confidence": round(conf, 3),
                })

        # Store detections for context queries
        global _last_visual
        _last_visual = detections

        # Ingest YOLO detections as graph sentence
        if detections:
            objects = [d["label"] for d in detections]
            sentence = "KOS currently sees " + ", ".join(objects) + " in the visual field."
            driver.ingest(sentence)

        # Trigger emotions from visual detections
        from kos.senses.perception import EmotionGrounding
        grounding = EmotionGrounding(_get_emotion())
        fake_dets = [{"label": d["label"]} for d in detections]
        triggers = grounding.process_visual(fake_dets)

        # Ingest detected objects into graph
        for d in detections:
            uid = lexicon.get_or_create_id(d["label"])
            kernel.add_node(uid)

        return {
            "status": "OK",
            "resolution": "%dx%d" % (w, h),
            "detections": detections,
            "emotion_triggers": triggers,
            "emotion_state": _get_emotion().current_emotion(),
            "frame_url": "/static/last_frame.jpg",
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)[:100]}

@app.post("/api/sensors/listen")
def sensor_listen():
    """Record from microphone and transcribe."""
    try:
        import sounddevice as sd
        import numpy as np
        import wave
        import tempfile

        duration = 4
        sample_rate = 16000

        audio = sd.rec(int(duration * sample_rate),
                       samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()

        peak = float(np.max(np.abs(audio)))

        # Save to temp wav
        wav_path = tempfile.mktemp(suffix=".wav")
        with wave.open(wav_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        # Transcribe
        text = ""
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(wav_path, fp16=False)
            text = result.get("text", "").strip()
        except Exception as we:
            text = "(Whisper error: %s)" % str(we)[:60]

        # Store for context queries
        global _last_audio
        if text and not text.startswith("("):
            _last_audio = text

        # Ingest transcript into graph if we got text
        if text and len(text) > 3 and not text.startswith("("):
            driver.ingest(text)

        # Clean up
        try:
            os.remove(wav_path)
        except:
            pass

        return {
            "status": "OK",
            "duration_sec": duration,
            "peak_amplitude": round(peak, 4),
            "transcript": text,
            "ingested": bool(text and len(text) > 3),
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)[:100]}

@app.get("/api/sensors/emotion")
def sensor_emotion():
    """Get current emotion state."""
    em = _get_emotion()
    state = em.state
    return {
        "emotion": em.current_emotion(),
        "cortisol": round(state.cortisol, 1),
        "adrenaline": round(state.adrenaline, 1),
        "dopamine": round(state.dopamine, 1),
        "serotonin": round(state.serotonin, 1),
        "oxytocin": round(state.oxytocin, 1),
        "gaba": round(state.gaba, 1),
        "endorphin": round(state.endorphin, 1),
    }

@app.post("/api/sensors/speak_answer")
def speak_answer(req: QueryRequest):
    """Query KOS then speak the answer."""
    t0 = time.perf_counter()
    q = req.prompt.lower()

    # Intercept visual context queries
    visual_words = {"see", "looking", "picture", "photo", "camera", "webcam",
                    "visible", "detect", "detected", "objects", "view", "image",
                    "show", "showing", "front", "watching"}
    audio_words = {"hear", "heard", "said", "listen", "sound", "audio",
                   "spoke", "speaking", "voice", "recording"}

    if any(w in q for w in visual_words) and _last_visual:
        objects = [d["label"] + " (" + str(round(d["confidence"]*100)) + "%)"
                   for d in _last_visual]
        answer = "I can see: " + ", ".join(objects) + "."
    elif any(w in q for w in audio_words) and _last_audio:
        answer = "I heard you say: " + _last_audio
    else:
        # Check if user wants to use voice input — listen first, then answer
        voice_trigger = {"ask by voice", "voice input", "listen and answer",
                         "use mic", "use microphone", "speak to me",
                         "listen to me", "voice query"}
        if any(t in q for t in voice_trigger):
            # Record and transcribe
            try:
                import sounddevice as sd
                import numpy as np
                import wave
                import tempfile

                sample_rate = 16000
                duration = 5
                audio = sd.rec(int(duration * sample_rate),
                               samplerate=sample_rate, channels=1, dtype='float32')
                sd.wait()

                wav_path = tempfile.mktemp(suffix=".wav")
                with wave.open(wav_path, 'w') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes((audio * 32767).astype(np.int16).tobytes())

                import whisper
                model = whisper.load_model("base")
                result = model.transcribe(wav_path, fp16=False)
                heard = result.get("text", "").strip()

                try:
                    os.remove(wav_path)
                except:
                    pass

                if heard:
                    _last_audio = heard
                    answer = shell.chat(heard)
                    answer = "You said: '" + heard + "'. " + answer
                else:
                    answer = "I listened but did not hear any speech. Please try again."
            except Exception as e:
                answer = "Microphone error: " + str(e)[:80]
        else:
            answer = shell.chat(req.prompt)

    latency = (time.perf_counter() - t0) * 1000

    # Speak it in background thread
    import threading
    def _speak(text):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            # Emotion-modulated speech rate
            em_state = _get_emotion().current_emotion()
            rate = 160  # default
            if em_state == "fear": rate = 200
            elif em_state == "calm": rate = 140
            elif em_state == "joy": rate = 175
            elif em_state == "depression": rate = 120
            engine.setProperty('rate', rate)
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    t = threading.Thread(target=_speak, args=(answer.strip(),), daemon=True)
    t.start()
    spoken = True

    return {
        "prompt": req.prompt,
        "answer": answer.strip(),
        "latency_ms": round(latency, 1),
        "spoken": spoken,
    }
