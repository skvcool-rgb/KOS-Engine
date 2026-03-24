"""
KOS V5.1 — Agent Dashboard

Real-time monitoring, task assignment, and health tracking
for the KOS Cognitive Agent.

Run: streamlit run dashboard.py
"""

import streamlit as st
import time
import json
import os
import sys
import re
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.router_offline import KOSShellOffline
from kos.weaver import AlgorithmicWeaver
from kos.self_improve import SelfImprover
from kos.feedback import WeaverFeedback, FormulaLearner, ContinuousTuner, AnalogyScanner
from kos.predictive import PredictiveCodingEngine
from kos.attention import AttentionController

# ══════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════

def init_session():
    if 'kernel' not in st.session_state:
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

        st.session_state.kernel = kernel
        st.session_state.lexicon = lexicon
        st.session_state.driver = driver
        st.session_state.shell = shell
        st.session_state.pce = pce
        st.session_state.feedback = feedback
        st.session_state.improver = improver
        st.session_state.scanner = scanner
        st.session_state.formula_learner = formula_learner
        st.session_state.query_log = []
        st.session_state.task_log = []
        st.session_state.health_log = []
        st.session_state.agent_status = "IDLE"
        st.session_state.total_queries = 0
        st.session_state.total_correct = 0


init_session()


# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="KOS Agent Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333;
    }
    .status-online { color: #00ff88; font-weight: bold; }
    .status-busy { color: #ffaa00; font-weight: bold; }
    .status-error { color: #ff4444; font-weight: bold; }
    .stMetric { background: #1a1a2e; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# SIDEBAR — NAVIGATION
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🧠 KOS Dashboard")
    st.caption("Knowledge Operating System V5.1")

    page = st.radio("Navigation", [
        "📊 Overview",
        "💬 Query & Chat",
        "📋 Task Manager",
        "🏥 Health Monitor",
        "🕸️ Graph Explorer",
        "📖 Knowledge Ingest",
        "🧬 Self-Improvement",
        "🔬 Learning Mechanisms",
    ])

    st.divider()

    # Quick stats
    k = st.session_state.kernel
    st.metric("Nodes", f"{len(k.nodes):,}")
    st.metric("Edges", f"{sum(len(n.connections) for n in k.nodes.values()):,}")
    st.metric("Queries", st.session_state.total_queries)
    st.metric("Status", st.session_state.agent_status)


# ══════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.header("📊 Agent Overview")

    k = st.session_state.kernel

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Nodes", f"{len(k.nodes):,}")
    with col2:
        total_edges = sum(len(n.connections) for n in k.nodes.values())
        st.metric("Total Edges", f"{total_edges:,}")
    with col3:
        st.metric("Provenance Sentences",
                   f"{sum(len(v) for v in k.provenance.values()):,}")
    with col4:
        contradictions = len(getattr(k, 'contradictions', []))
        st.metric("Contradictions", contradictions)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Agent Status")
        status = st.session_state.agent_status
        if status == "IDLE":
            st.success("🟢 Agent is IDLE — ready for queries")
        elif status == "BUSY":
            st.warning("🟡 Agent is BUSY — processing")
        else:
            st.info(f"Status: {status}")

        st.subheader("9 Learning Mechanisms")
        mechanisms = [
            ("Myelination", True, "Edges strengthen when used"),
            ("Predictive Coding", True, "Predict → compare → adjust"),
            ("Catastrophic Unlearning", True, "Crush false beliefs"),
            ("Active Inference", True, "Forage web for gaps"),
            ("Triadic Closure", True, "Infer A→C from A→B→C"),
            ("Weaver Feedback", True, "Learn from re-asks"),
            ("Formula Discovery", True, "Extract math from text"),
            ("Continuous Auto-Tuning", True, "Self-trigger optimizer"),
            ("Analogy Discovery", True, "Find structural matches"),
        ]
        for name, active, desc in mechanisms:
            icon = "✅" if active else "❌"
            st.markdown(f"{icon} **{name}** — {desc}")

    with col2:
        st.subheader("Recent Activity")
        if st.session_state.query_log:
            for entry in reversed(st.session_state.query_log[-10:]):
                with st.expander(f"Q: {entry['query'][:50]}...", expanded=False):
                    st.write(f"**Answer:** {entry['answer'][:200]}")
                    st.write(f"**Latency:** {entry['latency_ms']:.0f}ms")
        else:
            st.info("No queries yet. Go to 'Query & Chat' to start.")


# ══════════════════════════════════════════════════════════
# PAGE 2: QUERY & CHAT
# ══════════════════════════════════════════════════════════

elif page == "💬 Query & Chat":
    st.header("💬 Query the Knowledge Engine")

    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Ask KOS anything...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        st.session_state.agent_status = "BUSY"

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                t0 = time.perf_counter()
                answer = st.session_state.shell.chat(prompt)
                latency = (time.perf_counter() - t0) * 1000

            st.write(answer)
            st.caption(f"Latency: {latency:.0f}ms | Nodes: {len(st.session_state.kernel.nodes)}")

        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Track
        st.session_state.query_log.append({
            'query': prompt, 'answer': answer,
            'latency_ms': latency, 'time': time.time(),
        })
        st.session_state.total_queries += 1
        st.session_state.feedback.record(prompt, answer, [answer])
        st.session_state.agent_status = "IDLE"


# ══════════════════════════════════════════════════════════
# PAGE 3: TASK MANAGER
# ══════════════════════════════════════════════════════════

elif page == "📋 Task Manager":
    st.header("📋 Task Manager")
    st.caption("Assign tasks to the KOS agent")

    col1, col2 = st.columns([2, 1])

    with col1:
        task_type = st.selectbox("Task Type", [
            "Ingest Knowledge",
            "Run Self-Benchmark",
            "Run Self-Improvement Cycle",
            "Discover Analogies",
            "Discover Formulas",
            "Resolve Contradictions",
            "Normalize Weights",
            "Rebalance Graph Degrees",
            "Custom Query Batch",
        ])

        if task_type == "Ingest Knowledge":
            corpus = st.text_area("Paste knowledge to ingest:", height=200,
                                   placeholder="Toronto is the capital of Ontario...")
            if st.button("📥 Ingest", type="primary"):
                if corpus.strip():
                    st.session_state.agent_status = "BUSY"
                    t0 = time.perf_counter()
                    st.session_state.driver.ingest(corpus)
                    elapsed = (time.perf_counter() - t0) * 1000
                    st.session_state.agent_status = "IDLE"
                    st.success(f"Ingested in {elapsed:.0f}ms. "
                               f"Graph now has {len(st.session_state.kernel.nodes)} nodes.")
                    st.session_state.task_log.append({
                        'type': 'ingest', 'time': time.time(),
                        'result': f'{len(st.session_state.kernel.nodes)} nodes'
                    })

        elif task_type == "Run Self-Benchmark":
            if st.button("🏃 Run Benchmark", type="primary"):
                st.session_state.agent_status = "BUSY"
                result = st.session_state.improver.run_benchmark(verbose=False)
                st.session_state.agent_status = "IDLE"

                accuracy = result['accuracy']
                if accuracy >= 0.9:
                    st.success(f"Benchmark: {result['passed']}/{result['total']} "
                               f"({accuracy:.0%})")
                elif accuracy >= 0.7:
                    st.warning(f"Benchmark: {result['passed']}/{result['total']} "
                               f"({accuracy:.0%})")
                else:
                    st.error(f"Benchmark: {result['passed']}/{result['total']} "
                             f"({accuracy:.0%})")

                if result.get('failed'):
                    st.write("**Failed queries:**")
                    for q in result['failed']:
                        st.write(f"  - {q}")

                st.session_state.task_log.append({
                    'type': 'benchmark', 'time': time.time(),
                    'result': f"{accuracy:.0%}"
                })

        elif task_type == "Run Self-Improvement Cycle":
            if st.button("🧬 Run Self-Improvement", type="primary"):
                st.session_state.agent_status = "BUSY"
                with st.spinner("Running all 6 improvement proposals..."):
                    result = st.session_state.improver.improve(verbose=False)
                st.session_state.agent_status = "IDLE"

                st.success(f"Self-improvement complete in {result['time_ms']:.0f}ms")
                st.json(result)

                st.session_state.task_log.append({
                    'type': 'self_improve', 'time': time.time(),
                    'result': f"{result['time_ms']:.0f}ms"
                })

        elif task_type == "Discover Analogies":
            threshold = st.slider("Similarity threshold", 0.2, 0.8, 0.3, 0.05)
            if st.button("🔍 Scan for Analogies", type="primary"):
                st.session_state.agent_status = "BUSY"
                scanner = st.session_state.scanner
                scanner.threshold = threshold
                analogies = scanner.scan(verbose=False)
                st.session_state.agent_status = "IDLE"

                if analogies:
                    st.success(f"Found {len(analogies)} analogies!")
                    for a in sorted(analogies, key=lambda x: x['similarity'],
                                     reverse=True)[:10]:
                        st.write(f"  **{a['word_a']}** <=> **{a['word_b']}** "
                                 f"(similarity: {a['similarity']:.2f})")
                else:
                    st.info("No analogies found at this threshold.")

        elif task_type == "Discover Formulas":
            if st.button("🔢 Scan for Formulas", type="primary"):
                learner = st.session_state.formula_learner
                formulas = learner.scan_provenance(st.session_state.kernel)
                if formulas:
                    st.success(f"Discovered {len(formulas)} formulas!")
                    for f in formulas:
                        st.write(f"  **{f['name']}** = {f['expression']}")
                        st.caption(f"Source: {f['source'][:60]}")
                else:
                    st.info("No formulas found in current provenance.")

        elif task_type == "Resolve Contradictions":
            if st.button("⚖️ Resolve Contradictions", type="primary"):
                result = st.session_state.improver.resolve_contradictions(verbose=False)
                st.write(f"Total contradictions: {result['contradictions_total']}")
                st.write(f"Resolved: {result['resolved']}")

        elif task_type == "Normalize Weights":
            if st.button("📏 Normalize", type="primary"):
                result = st.session_state.improver.normalize_weights(verbose=False)
                st.write(f"Weights clipped: {result['clipped']}")
                st.write(f"Max effective weight: {result['max_weight']:.3f}")

        elif task_type == "Rebalance Graph Degrees":
            if st.button("⚖️ Rebalance", type="primary"):
                result = st.session_state.improver.rebalance_degrees(verbose=False)
                st.write(f"Hubs weakened: {result['hubs_fixed']}")
                st.write(f"Orphans connected: {result['orphans_fixed']}")

        elif task_type == "Custom Query Batch":
            queries = st.text_area("One query per line:", height=150,
                                    placeholder="Where is Toronto?\nPopulation of Montreal?")
            if st.button("▶️ Run Batch", type="primary"):
                if queries.strip():
                    lines = [l.strip() for l in queries.split('\n') if l.strip()]
                    st.session_state.agent_status = "BUSY"
                    for q in lines:
                        t0 = time.perf_counter()
                        answer = st.session_state.shell.chat(q)
                        latency = (time.perf_counter() - t0) * 1000
                        st.write(f"**Q:** {q}")
                        st.write(f"**A:** {answer.strip()[:200]}")
                        st.caption(f"Latency: {latency:.0f}ms")
                        st.divider()
                    st.session_state.agent_status = "IDLE"

    with col2:
        st.subheader("Task History")
        if st.session_state.task_log:
            for entry in reversed(st.session_state.task_log[-20:]):
                st.write(f"**{entry['type']}** — {entry['result']}")
        else:
            st.info("No tasks executed yet.")


# ══════════════════════════════════════════════════════════
# PAGE 4: HEALTH MONITOR
# ══════════════════════════════════════════════════════════

elif page == "🏥 Health Monitor":
    st.header("🏥 System Health Monitor")

    k = st.session_state.kernel

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nodes", f"{len(k.nodes):,}")
    with col2:
        total_edges = sum(len(n.connections) for n in k.nodes.values())
        st.metric("Edges", f"{total_edges:,}")
    with col3:
        orphans = sum(1 for n in k.nodes.values() if not n.connections)
        st.metric("Orphan Nodes", orphans)
    with col4:
        super_hubs = sum(1 for n in k.nodes.values() if len(n.connections) > 20)
        st.metric("Super-Hubs (>20)", super_hubs)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Graph Health")

        # Edge weight distribution
        weights = []
        myelins = []
        for node in k.nodes.values():
            for tgt, data in node.connections.items():
                if isinstance(data, dict):
                    weights.append(data['w'])
                    myelins.append(data.get('myelin', 0))
                else:
                    weights.append(data)

        if weights:
            avg_w = sum(abs(w) for w in weights) / len(weights)
            max_w = max(abs(w) for w in weights)
            avg_m = sum(myelins) / len(myelins) if myelins else 0

            st.write(f"**Avg edge weight:** {avg_w:.3f}")
            st.write(f"**Max edge weight:** {max_w:.3f}")
            st.write(f"**Avg myelination:** {avg_m:.1f}")
            st.write(f"**Contradictions:** {len(getattr(k, 'contradictions', []))}")

            # Weight health check
            if max_w > 1.0:
                st.warning("⚠️ Edge weights exceed 1.0 — normalization recommended")
            else:
                st.success("✅ Edge weights within normal range")

    with col2:
        st.subheader("Feedback Health")
        stats = st.session_state.feedback.get_stats()
        st.write(f"**Total queries tracked:** {stats['total_queries']}")
        st.write(f"**Re-asks (dissatisfaction):** {stats['reasks']}")
        st.write(f"**Satisfaction signals:** {stats['satisfied']}")
        st.write(f"**Satisfaction rate:** {stats['satisfaction_rate']:.0%}")
        st.write(f"**Evidence patterns tracked:** {stats['tracked_evidence']}")

        if stats['satisfaction_rate'] >= 0.8:
            st.success("✅ User satisfaction is HIGH")
        elif stats['satisfaction_rate'] >= 0.5:
            st.warning("⚠️ User satisfaction is MODERATE")
        elif stats['total_queries'] > 5:
            st.error("❌ User satisfaction is LOW — self-improvement needed")

    st.divider()

    st.subheader("Predictive Coding Health")
    pce_stats = st.session_state.pce.get_stats()
    st.write(f"**Cached predictions:** {pce_stats['cached_predictions']}")
    st.write(f"**Total predictions made:** {pce_stats['total_predictions']}")
    st.write(f"**Overall accuracy:** {pce_stats['overall_accuracy']:.0%}")
    st.write(f"**Weight adjustments:** {pce_stats['total_adjustments']}")

    if st.button("🔄 Run Health Check Now"):
        st.session_state.agent_status = "BUSY"
        result = st.session_state.improver.improve(verbose=False)
        st.session_state.agent_status = "IDLE"
        st.success(f"Health check complete in {result['time_ms']:.0f}ms")
        st.json(result)


# ══════════════════════════════════════════════════════════
# PAGE 5: GRAPH EXPLORER
# ══════════════════════════════════════════════════════════

elif page == "🕸️ Graph Explorer":
    st.header("🕸️ Knowledge Graph Explorer")

    k = st.session_state.kernel
    lex = st.session_state.lexicon

    # Search for a node
    search = st.text_input("Search for a concept:", placeholder="toronto")

    if search:
        search_lower = search.lower().strip()
        uid = lex.word_to_uuid.get(search_lower)

        if uid and uid in k.nodes:
            node = k.nodes[uid]
            word = lex.get_word(uid)

            st.subheader(f"Node: {word}")
            st.write(f"**UUID:** `{uid}`")
            st.write(f"**Connections:** {len(node.connections)}")
            st.write(f"**Activation:** {node.activation:.4f}")
            st.write(f"**Fuel:** {node.fuel:.4f}")

            # Show connections
            st.subheader("Connections")
            connections = []
            for tgt, data in sorted(node.connections.items(),
                                      key=lambda x: abs(x[1]['w'] if isinstance(x[1], dict) else x[1]),
                                      reverse=True):
                tgt_word = lex.get_word(tgt)
                w = data['w'] if isinstance(data, dict) else data
                m = data.get('myelin', 0) if isinstance(data, dict) else 0
                connections.append({
                    'Target': tgt_word,
                    'Weight': f"{w:.3f}",
                    'Myelin': m,
                    'Effective': f"{w * (1 + m * 0.01):.3f}",
                })

            if connections:
                st.dataframe(connections, use_container_width=True)

            # Show provenance
            st.subheader("Provenance (Source Sentences)")
            prov_set = set()
            for tgt in node.connections:
                key = tuple(sorted([uid, tgt]))
                prov_set.update(k.provenance.get(key, set()))

            for sent in list(prov_set)[:20]:
                st.write(f"  - {sent}")
        else:
            st.warning(f"Node '{search}' not found in graph")

    else:
        # Show top nodes by degree
        st.subheader("Top Nodes by Connectivity")
        top_nodes = sorted(k.nodes.items(),
                            key=lambda x: len(x[1].connections), reverse=True)[:20]
        data = []
        for nid, node in top_nodes:
            data.append({
                'Concept': lex.get_word(nid),
                'Connections': len(node.connections),
                'Activation': f"{node.activation:.3f}",
            })
        if data:
            st.dataframe(data, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 6: KNOWLEDGE INGEST
# ══════════════════════════════════════════════════════════

elif page == "📖 Knowledge Ingest":
    st.header("📖 Knowledge Ingestion")

    tab1, tab2 = st.tabs(["📝 Text Input", "🌐 URL Foraging"])

    with tab1:
        corpus = st.text_area("Paste text to ingest:", height=300,
                               placeholder="Enter knowledge here...\n\nExample:\nToronto is a major city in Ontario.\nIt was founded in 1834.")

        if st.button("📥 Ingest Text", type="primary"):
            if corpus.strip():
                t0 = time.perf_counter()
                st.session_state.driver.ingest(corpus)
                elapsed = (time.perf_counter() - t0) * 1000

                st.success(f"Ingested in {elapsed:.0f}ms")
                st.metric("Total Nodes", len(st.session_state.kernel.nodes))

    with tab2:
        url = st.text_input("Wikipedia URL:", placeholder="https://en.wikipedia.org/wiki/Toronto")
        if st.button("🌐 Forage URL"):
            if url.strip():
                try:
                    from kos.forager import WebForager
                    forager = WebForager(
                        st.session_state.kernel,
                        st.session_state.lexicon,
                        st.session_state.driver,
                    )
                    with st.spinner(f"Foraging {url}..."):
                        forager.forage(url)
                    st.success(f"Foraging complete! Nodes: {len(st.session_state.kernel.nodes)}")
                except Exception as e:
                    st.error(f"Foraging failed: {e}")


# ══════════════════════════════════════════════════════════
# PAGE 7: SELF-IMPROVEMENT
# ══════════════════════════════════════════════════════════

elif page == "🧬 Self-Improvement":
    st.header("🧬 Self-Improvement Engine")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Run Improvement Cycle")
        if st.button("🚀 Run All 6 Improvements", type="primary"):
            with st.spinner("Running improvements..."):
                result = st.session_state.improver.improve(verbose=False)
            st.success(f"Complete in {result['time_ms']:.0f}ms")

            for key, val in result.items():
                if key != 'time_ms':
                    st.write(f"**{key}:** {val}")

        st.divider()

        st.subheader("Analogy Scanner")
        if st.button("🔍 Discover Analogies"):
            scanner = st.session_state.scanner
            analogies = scanner.scan(verbose=False)
            if analogies:
                for a in analogies[:10]:
                    st.write(f"  **{a['word_a']}** <=> **{a['word_b']}** "
                             f"({a['similarity']:.2f})")
            else:
                st.info("No analogies found.")

    with col2:
        st.subheader("Formula Discovery")
        if st.button("🔢 Scan Formulas"):
            learner = st.session_state.formula_learner
            formulas = learner.scan_provenance(st.session_state.kernel)
            if formulas:
                for f in formulas:
                    st.write(f"  **{f['name']}** = `{f['expression']}`")
            else:
                st.info("No formulas found.")

        st.divider()

        st.subheader("Self-Benchmark")
        if st.button("📊 Run Benchmark"):
            result = st.session_state.improver.run_benchmark(verbose=False)
            accuracy = result['accuracy']
            st.metric("Accuracy", f"{accuracy:.0%}")
            if result.get('failed'):
                st.write("**Failed:**")
                for q in result['failed']:
                    st.write(f"  - {q}")


# ══════════════════════════════════════════════════════════
# PAGE 8: LEARNING MECHANISMS
# ══════════════════════════════════════════════════════════

elif page == "🔬 Learning Mechanisms":
    st.header("🔬 9 Learning Mechanisms — Live Status")

    mechanisms = [
        {
            'name': 'Myelination',
            'icon': '🧠',
            'description': 'Edges strengthen when used (Hebbian learning)',
            'status': 'ACTIVE',
            'metric': f"{sum(d.get('myelin', 0) for n in st.session_state.kernel.nodes.values() for d in n.connections.values() if isinstance(d, dict))} total myelin",
        },
        {
            'name': 'Predictive Coding',
            'icon': '🔮',
            'description': 'Predict activation patterns, compare to reality, adjust weights',
            'status': 'ACTIVE',
            'metric': f"{st.session_state.pce.get_stats()['cached_predictions']} cached predictions",
        },
        {
            'name': 'Catastrophic Unlearning',
            'icon': '💥',
            'description': 'Crush false beliefs when prediction error is persistent',
            'status': 'ACTIVE',
            'metric': f"Threshold: consecutive errors > 5",
        },
        {
            'name': 'Active Inference',
            'icon': '🌐',
            'description': 'Autonomously forage web when knowledge gaps detected',
            'status': 'STANDBY',
            'metric': 'Triggers on high system entropy',
        },
        {
            'name': 'Triadic Closure',
            'icon': '🔺',
            'description': 'Infer A→C from A→B and B→C',
            'status': 'ACTIVE',
            'metric': 'Runs during daemon maintenance',
        },
        {
            'name': 'Weaver Feedback',
            'icon': '📊',
            'description': 'Learn from user re-asks to demote bad evidence',
            'status': 'ACTIVE',
            'metric': f"{st.session_state.feedback.get_stats()['reasks']} re-asks detected",
        },
        {
            'name': 'Formula Discovery',
            'icon': '🔢',
            'description': 'Extract mathematical formulas from ingested text',
            'status': 'ACTIVE',
            'metric': f"{st.session_state.formula_learner.get_stats()['discovered']} formulas found",
        },
        {
            'name': 'Continuous Auto-Tuning',
            'icon': '⚙️',
            'description': 'Self-trigger threshold optimization when accuracy drops',
            'status': 'ACTIVE',
            'metric': 'Monitors rolling accuracy window',
        },
        {
            'name': 'Analogy Discovery',
            'icon': '🔬',
            'description': 'Find structural matches across domains',
            'status': 'ACTIVE',
            'metric': f"{st.session_state.scanner.get_stats()['analogies_found']} analogies found",
        },
    ]

    for m in mechanisms:
        with st.expander(f"{m['icon']} {m['name']} — {m['status']}", expanded=False):
            st.write(f"**Description:** {m['description']}")
            st.write(f"**Current metric:** {m['metric']}")
            if m['status'] == 'ACTIVE':
                st.success("✅ Online and learning")
            else:
                st.info("💤 Standing by — activates on trigger")
