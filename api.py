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
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.router_offline import KOSShellOffline
from kos.weaver import AlgorithmicWeaver
from kos.self_improve import SelfImprover
from kos.feedback import WeaverFeedback, FormulaLearner, ContinuousTuner, AnalogyScanner
from kos.predictive import PredictiveCodingEngine

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
"""
driver.ingest(SEED_CORPUS)

# State tracking
query_log = []
task_log = []
health_log = []
boot_time = time.time()

# ══════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════

app = FastAPI(title="KOS Agent API", version="5.1.0")

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

@app.get("/api/status")
async def get_status():
    """System overview — node count, uptime, query stats."""
    total_edges = sum(len(n.connections) for n in kernel.nodes.values())
    orphans = sum(1 for n in kernel.nodes.values() if not n.connections)
    hubs = sum(1 for n in kernel.nodes.values() if len(n.connections) > 15)
    uptime = time.time() - boot_time

    pce_stats = pce.get_stats()

    return {
        "status": "ONLINE",
        "uptime_seconds": round(uptime, 1),
        "uptime_human": f"{int(uptime//3600)}h {int((uptime%3600)//60)}m {int(uptime%60)}s",
        "nodes": len(kernel.nodes),
        "edges": total_edges,
        "orphans": orphans,
        "super_hubs": hubs,
        "queries_total": len(query_log),
        "avg_latency_ms": round(sum(q['latency_ms'] for q in query_log) / len(query_log), 1) if query_log else 0,
        "predictions_cached": pce_stats['cached_predictions'],
        "prediction_accuracy": round(pce_stats['overall_accuracy'] * 100, 1),
        "contradictions": len(kernel.contradictions),
        "tasks_pending": sum(1 for t in task_log if t['status'] == 'pending'),
        "tasks_completed": sum(1 for t in task_log if t['status'] == 'completed'),
    }


@app.post("/api/query")
def query(req: QueryRequest):
    """Ask the KOS a question."""
    t0 = time.perf_counter()
    answer = shell.chat(req.prompt)
    latency = (time.perf_counter() - t0) * 1000

    entry = {
        "prompt": req.prompt,
        "answer": answer.strip(),
        "latency_ms": round(latency, 1),
        "timestamp": time.time(),
        "nodes_activated": len(kernel.nodes),
    }
    query_log.append(entry)

    # Track feedback (re-ask detection)
    if len(query_log) >= 2:
        prev = query_log[-2]['prompt'].lower().split()
        curr = req.prompt.lower().split()
        overlap = len(set(prev) & set(curr))
        if overlap >= 2 and len(curr) >= 2:
            feedback._reask_count += 1

    return entry


@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    """Ingest new knowledge into the graph."""
    t0 = time.perf_counter()
    nodes_before = len(kernel.nodes)
    driver.ingest(req.text)
    nodes_after = len(kernel.nodes)
    latency = (time.perf_counter() - t0) * 1000

    return {
        "status": "OK",
        "nodes_before": nodes_before,
        "nodes_after": nodes_after,
        "nodes_added": nodes_after - nodes_before,
        "latency_ms": round(latency, 1),
    }


@app.get("/api/health")
async def health_check():
    """Run full health check — benchmark + all 9 learning mechanisms."""
    t0 = time.perf_counter()
    result = improver.improve(verbose=False)
    result['time_ms'] = (time.perf_counter() - t0) * 1000

    # Add predictive coding stats
    pce_stats = pce.get_stats()
    result['predictive'] = {
        'cached': pce_stats['cached_predictions'],
        'accuracy': round(pce_stats['overall_accuracy'] * 100, 1),
        'adjustments': pce_stats.get('total_weight_adjustments', 0),
    }

    health_log.append({
        'time': time.time(),
        'accuracy': result.get('benchmark', {}).get('accuracy', 0),
    })

    return result


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
        "myelin_total": sum(d.get('myelin', 0) for d in node.connections.values()),
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
    if req.task.lower().startswith("ingest:"):
        url_or_text = req.task[7:].strip()
        nodes_before = len(kernel.nodes)
        driver.ingest(url_or_text)
        task['status'] = 'completed'
        task['result'] = f"Ingested. Nodes: {nodes_before} -> {len(kernel.nodes)}"
    elif req.task.lower().startswith("query:"):
        query_text = req.task[6:].strip()
        answer = shell.chat(query_text)
        task['status'] = 'completed'
        task['result'] = answer.strip()
    elif req.task.lower().startswith("health"):
        result = improver.improve(verbose=False)
        task['status'] = 'completed'
        task['result'] = f"Accuracy: {result.get('benchmark', {}).get('accuracy', 0):.0%}"
    else:
        task['status'] = 'pending'
        task['result'] = 'Task queued for manual processing'

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
    return [
        {"node_a": str(a), "node_b": str(b), "details": str(d)}
        for (a, b), d in kernel.contradictions.items()
    ] if hasattr(kernel, 'contradictions') else []


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

@app.post("/api/sensors/speak")
def sensor_speak(req: QueryRequest):
    """Make KOS speak text through speakers."""
    import threading
    def _speak(text):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
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
        answer = "I heard: " + _last_audio
    else:
        answer = shell.chat(req.prompt)

    latency = (time.perf_counter() - t0) * 1000

    # Speak it in background thread
    import threading
    def _speak(text):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
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
