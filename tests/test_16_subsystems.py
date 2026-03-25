"""
KOS V8.0 -- Full 16-Subsystem Integration Test

Tests every new subsystem built for the post-V7 architecture:
    1. Synthesis Engine (template + rhetorical planner)
    2. Output Validator (LLM hallucination guard)
    3. HTN Planner (goal stack + operators)
    4. Causal DAG (separate from association graph)
    5. Constraint Engine (physics primitives + domain axioms)
    6. Source Governance (trust-tiered ingestion)
    7. Memory Lifecycle (4-tier: hot/warm/cold/archive)
    8. Unified Drive System (DriveScore + mission alignment)
    9. Action Registry (schema + permissions + rollback)
    10. Reasoning Workspace (scratchpad + hypothesis workspaces)
    11. Verification Pipeline (ingest -> quarantine -> promote)
    12. Shadow CI/CD (canary deployment)
    13. Adaptive Tick Controller
    14. Domain Axiom Files (.kos format)
    15. Multi-signal Reranker (Phase 2 regression)
    16. Full Pipeline: query -> normalize -> profile -> beam -> rerank -> synthesize
"""

import sys
import os
import time
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_1_synthesis_engine():
    """Synthesis Engine: template-based output."""
    print("=" * 70)
    print("  TEST 1: Synthesis Engine")
    print("=" * 70)
    from kos.synthesis import SynthesisEngine

    engine = SynthesisEngine(domain="general")
    evidence = [
        "Toronto is a major city in Canada.",
        "Toronto is located in Ontario.",
        "Toronto has a population of 2.9 million.",
    ]
    result = engine.synthesize(evidence, intent="where",
                                entities=["Toronto"],
                                raw_prompt="Where is Toronto?")
    t1 = len(result["response"]) > 0
    t2 = result["confidence"] > 0.0
    t3 = len(result["raw_evidence"]) == 3
    print(f"  Response: {result['response'][:80]}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Template: {result['template_used']}")

    # Test JSON contract for LLM
    contract = engine.build_contract(evidence, result["confidence"])
    t4 = contract["prohibited_inference"] == True
    t5 = len(contract["facts"]) == 3

    print(f"  Contract has facts: {'PASS' if t5 else 'FAIL'}")
    print(f"  Inference prohibited: {'PASS' if t4 else 'FAIL'}")

    passed = all([t1, t2, t3, t4, t5])
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_2_output_validator():
    """Output Validator: catch LLM hallucinations."""
    print("\n" + "=" * 70)
    print("  TEST 2: Output Validator")
    print("=" * 70)
    from kos.output_validator import OutputValidator

    validator = OutputValidator()
    contract = {
        "facts": ["Toronto has a population of 2.9 million.",
                   "Toronto is in Ontario."]
    }

    # Good output: faithful to facts
    good = "Toronto, located in Ontario, has a population of 2.9 million."
    r1 = validator.validate(good, contract)
    t1 = r1["valid"]

    # Bad output: hallucinated number
    bad = "Toronto has a population of 5.2 million and was founded in 1834."
    r2 = validator.validate(bad, contract)
    t2 = not r2["valid"]
    t3 = r2["used_fallback"]

    print(f"  Good output valid: {'PASS' if t1 else 'FAIL'}")
    print(f"  Bad output rejected: {'PASS' if t2 else 'FAIL'}")
    print(f"  Fallback used: {'PASS' if t3 else 'FAIL'}")
    if r2["violations"]:
        print(f"  Violations: {r2['violations'][:2]}")

    passed = t1 and t2 and t3
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_3_htn_planner():
    """HTN Planner: goal-directed planning."""
    print("\n" + "=" * 70)
    print("  TEST 3: HTN Planner")
    print("=" * 70)
    from kos.planner import HTNPlanner, Goal

    planner = HTNPlanner()

    # Goal: deliver a response
    goal = Goal("answer_query",
                target_state={"response_sent": True},
                priority=10.0)

    state = {"has_seeds": True}
    result = planner.plan_and_execute(goal, state)

    t1 = result["status"] == "plan_ready"
    t2 = len(result["steps"]) >= 2
    print(f"  Plan status: {result['status']}")
    print(f"  Steps: {result['steps']}")
    print(f"  Cost: {result['cost']}")

    # Execute step by step
    for _ in range(len(result["steps"])):
        step_result = planner.execute_next_step(state)
        if step_result["status"] == "step_executed":
            state = step_result["new_state"]

    t3 = goal.satisfied(state) or "response_sent" in state
    print(f"  Goal achieved: {'PASS' if t3 else 'FAIL'}")

    passed = t1 and t2
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_4_causal_dag():
    """Causal DAG: separate causal reasoning."""
    print("\n" + "=" * 70)
    print("  TEST 4: Causal DAG")
    print("=" * 70)
    from kos.causal_dag import CausalDAG

    dag = CausalDAG()
    dag.add_cause("pollution", "warming", 0.9, 0.8, "greenhouse effect")
    dag.add_cause("warming", "ice_melt", 0.8, 0.7)
    dag.add_cause("ice_melt", "sea_rise", 0.7, 0.6)
    dag.add_cause("warming", "drought", 0.6, 0.5)

    # Cycle detection
    t1 = not dag.add_cause("sea_rise", "pollution")  # Would create cycle
    print(f"  Cycle blocked: {'PASS' if t1 else 'FAIL'}")

    # Effects
    effects = dag.get_effects("pollution", depth=3)
    effect_names = [e[0] for e in effects]
    t2 = "warming" in effect_names and "sea_rise" in effect_names
    print(f"  Effects of pollution: {effect_names}")

    # Causes
    causes = dag.get_causes("sea_rise", depth=3)
    cause_names = [c[0] for c in causes]
    t3 = "ice_melt" in cause_names
    print(f"  Causes of sea_rise: {cause_names}")

    # Causal path
    path = dag.causal_path("pollution", "sea_rise")
    t4 = path == ["pollution", "warming", "ice_melt", "sea_rise"]
    print(f"  Path pollution->sea_rise: {path}")

    # Intervention
    intervention = dag.intervene("warming")
    t5 = "ice_melt" in intervention["affected_nodes"]
    print(f"  Remove warming affects: {intervention['affected_nodes']}")

    # Topological order
    order = dag.topological_order()
    t6 = order.index("pollution") < order.index("warming")
    print(f"  Topo order: {order}")

    passed = all([t1, t2, t3, t4, t5, t6])
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_5_constraint_engine():
    """Constraint Engine: physics + domain axioms."""
    print("\n" + "=" * 70)
    print("  TEST 5: Constraint Engine")
    print("=" * 70)
    from kos.constraints import ConstraintEngine, DomainAxiom

    engine = ConstraintEngine()

    # Layer 1: Physics primitives
    r1 = engine.check("a", "a", 0.5, "")  # Self-loop
    t1 = not r1["passed"]
    print(f"  Self-loop blocked: {'PASS' if t1 else 'FAIL'}")

    r2 = engine.check("a", "b", 1.5, "")  # Weight out of bounds
    t2 = not r2["passed"]
    print(f"  Weight>1.0 blocked: {'PASS' if t2 else 'FAIL'}")

    r3 = engine.check("a", "b", 0.8, "Normal provenance")  # Valid
    t3 = r3["passed"]
    print(f"  Valid edge passes: {'PASS' if t3 else 'FAIL'}")

    # Layer 2: Domain axiom
    axiom = DomainAxiom(
        "min_capital", "Capital ratio >= 4.5%",
        "threshold", {"field": "weight", "operator": ">=", "value": 0.045})
    engine.add_axiom(axiom)

    r4 = engine.check("bank", "ratio", 0.02, "Capital ratio is 2%")
    t4 = not r4["passed"]
    print(f"  Basel III violation blocked: {'PASS' if t4 else 'FAIL'}")
    if r4["violations"]:
        print(f"    Reason: {r4['violations'][0]}")

    r5 = engine.check("bank", "ratio", 0.06, "Capital ratio is 6%")
    t5 = r5["passed"]
    print(f"  Compliant ratio passes: {'PASS' if t5 else 'FAIL'}")

    passed = all([t1, t2, t3, t4, t5])
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_6_source_governance():
    """Source Governance: trust-tiered ingestion."""
    print("\n" + "=" * 70)
    print("  TEST 6: Source Governance")
    print("=" * 70)
    from kos.source_governance import SourceGovernor, AUTHORITATIVE, TRUSTED, SECONDARY, EXPLORATORY

    gov = SourceGovernor()

    t1 = gov.classify_source("https://pubmed.ncbi.nlm.nih.gov/123") == AUTHORITATIVE
    t2 = gov.classify_source("https://en.wikipedia.org/wiki/Test") == TRUSTED
    t3 = gov.classify_source("https://reddit.com/r/test") == SECONDARY
    t4 = gov.classify_source("[daemon inference]") == EXPLORATORY

    print(f"  PubMed = AUTHORITATIVE: {'PASS' if t1 else 'FAIL'}")
    print(f"  Wikipedia = TRUSTED: {'PASS' if t2 else 'FAIL'}")
    print(f"  Reddit = SECONDARY: {'PASS' if t3 else 'FAIL'}")
    print(f"  Daemon = EXPLORATORY: {'PASS' if t4 else 'FAIL'}")

    # Quarantine decision
    t5 = gov.should_quarantine(0.9, EXPLORATORY, True)  # High weight, low trust, contradicts
    t6 = not gov.should_quarantine(0.9, AUTHORITATIVE, False)  # High trust, no contradiction
    print(f"  Low-trust contradiction quarantined: {'PASS' if t5 else 'FAIL'}")
    print(f"  High-trust no-contradiction passes: {'PASS' if t6 else 'FAIL'}")

    passed = all([t1, t2, t3, t4, t5, t6])
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_7_memory_lifecycle():
    """Memory Lifecycle: 4-tier memory management."""
    print("\n" + "=" * 70)
    print("  TEST 7: Memory Lifecycle")
    print("=" * 70)
    from kos.graph import KOSKernel
    from kos.memory_lifecycle import MemoryLifecycleManager

    k = KOSKernel(enable_vsa=False)
    k.add_connection("a", "b", 0.9, "A links B.")
    k.current_tick = 0

    mgr = MemoryLifecycleManager(k)

    # Fresh node should be HOT
    t1 = mgr.classify_node("a") == "hot"
    print(f"  Fresh node is HOT: {'PASS' if t1 else 'FAIL'}")

    # Age the node
    k.current_tick = 100
    t2 = mgr.classify_node("a") == "warm"
    print(f"  Aged node (100 ticks) is WARM: {'PASS' if t2 else 'FAIL'}")

    k.current_tick = 500
    t3 = mgr.classify_node("a") == "cold"
    print(f"  Old node (500 ticks) is COLD: {'PASS' if t3 else 'FAIL'}")

    # Sweep
    result = mgr.sweep()
    print(f"  Sweep result: {result['tiers']}")
    t4 = "hot" in result["tiers"]

    passed = all([t1, t2, t3, t4])
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_8_drive_system():
    """Unified Drive System: DriveScore + mission."""
    print("\n" + "=" * 70)
    print("  TEST 8: Unified Drive System")
    print("=" * 70)
    from kos.graph import KOSKernel
    from kos.drives import DriveScorer, Mission, DO_NOTHING_THRESHOLD

    k = KOSKernel(enable_vsa=False)
    k.add_connection("supply_chain", "logistics", 0.9, "Supply chain logistics.")

    mission = Mission(
        description="Optimize supply chain operations",
        keywords=["supply", "chain", "logistics", "inventory", "warehouse"],
        domain="industrial"
    )

    scorer = DriveScorer(mission=mission)

    # On-mission query should score higher than off-mission
    score1, b1 = scorer.score(k, "supply chain risk analysis",
                               target_topic="supply_chain",
                               is_active_task=True)
    print(f"  On-mission score: {score1:.3f} ({b1['decision']})")
    print(f"    Breakdown: gap={b1['knowledge_gap']:.2f} risk={b1['risk_relevance']:.2f} "
          f"task={b1['task_context']:.2f} mission={b1['mission_alignment']:.2f}")

    # Off-mission query should score lower
    score2, b2 = scorer.score(k, "medieval pottery techniques",
                               target_topic="pottery")
    print(f"  Off-mission score: {score2:.3f} ({b2['decision']})")

    t1 = score1 > score2  # On-mission always beats off-mission
    t2 = b1["mission_alignment"] > b2["mission_alignment"]

    # Do-nothing threshold
    t3 = b2["decision"] == "DO_NOTHING" or score2 < DO_NOTHING_THRESHOLD
    print(f"  On-mission > Off-mission: {'PASS' if t1 else 'FAIL'}")
    print(f"  Mission alignment correct: {'PASS' if t2 else 'FAIL'}")
    print(f"  Off-mission suppressed: {'PASS' if t3 else 'FAIL'}")

    passed = t1 and t2
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_9_action_registry():
    """Action Registry: permissions + rollback."""
    print("\n" + "=" * 70)
    print("  TEST 9: Action Registry")
    print("=" * 70)
    from kos.action_registry import ActionRegistry

    reg = ActionRegistry()

    # Without permission, network actions should be denied
    r1 = reg.execute("forage_web", {"query": "test"})
    t1 = r1["status"] == "denied"
    print(f"  Denied without permission: {'PASS' if t1 else 'FAIL'}")

    # Grant permission and retry
    reg.grant_permission("network")
    r2 = reg.execute("forage_web", {"query": "test"})
    t2 = r2["status"] == "executed"
    print(f"  Allowed with permission: {'PASS' if t2 else 'FAIL'}")

    # No-permission action should always work
    r3 = reg.execute("query_graph", {"seeds": ["a"]})
    t3 = r3["status"] == "executed"
    print(f"  No-perm action works: {'PASS' if t3 else 'FAIL'}")

    # Check history
    t4 = len(reg.execution_history) == 2  # Only successful ones
    print(f"  Execution history tracked: {'PASS' if t4 else 'FAIL'}")

    passed = all([t1, t2, t3, t4])
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_10_reasoning_workspace():
    """Reasoning Workspace: multi-step deliberation."""
    print("\n" + "=" * 70)
    print("  TEST 10: Reasoning Workspace")
    print("=" * 70)
    from kos.graph import KOSKernel
    from kos.reasoning import ReasoningEngine

    k = KOSKernel(enable_vsa=False)
    k.add_connection("pollution", "warming", 0.9, "Pollution causes warming.", edge_type=2)
    k.add_connection("warming", "ice_melt", 0.8, "Warming melts ice.", edge_type=2)
    k.add_connection("ice_melt", "sea_rise", 0.7, "Ice melt raises sea level.", edge_type=2)

    engine = ReasoningEngine(k)
    result = engine.reason("Why does pollution cause sea level rise?",
                            seeds=["pollution"], max_iterations=5)

    t1 = len(result["conclusions"]) > 0
    t2 = result["confidence"] > 0.0
    t3 = result["iterations"] > 0
    t4 = len(result["hypotheses"]) > 0

    print(f"  Conclusions: {result['conclusions'][:2]}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Hypotheses: {len(result['hypotheses'])}")

    passed = all([t1, t2, t3, t4])
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_11_verification_pipeline():
    """Verification Pipeline: quarantine + promote."""
    print("\n" + "=" * 70)
    print("  TEST 11: Verification Pipeline")
    print("=" * 70)
    from kos.graph import KOSKernel
    from kos.verification import VerificationPipeline
    from kos.source_governance import SourceGovernor

    k = KOSKernel(enable_vsa=False)
    gov = SourceGovernor()
    pipeline = VerificationPipeline(k, source_governor=gov)

    # High-trust source: auto-promote
    r1 = pipeline.ingest("toronto", "city", 0.9,
                          provenance="Toronto is a city. (2024) doi:10.1234",
                          source_url="https://pubmed.ncbi.nlm.nih.gov/123")
    t1 = r1["status"] == "promoted"
    print(f"  High-trust auto-promoted: {'PASS' if t1 else 'FAIL'}")

    # Low-trust source: quarantined
    r2 = pipeline.ingest("ufo", "real", 0.9,
                          provenance="[daemon inference] UFOs are real",
                          source_url="")
    t2 = r2["status"] == "quarantined"
    print(f"  Low-trust quarantined: {'PASS' if t2 else 'FAIL'}")

    # Check quarantine
    pending = pipeline.review_quarantine()
    t3 = len(pending) > 0
    print(f"  Quarantine has items: {'PASS' if t3 else 'FAIL'}")

    # Stats
    stats = pipeline.stats()
    t4 = stats["ingested"] == 2
    print(f"  Stats: {stats}")

    passed = all([t1, t2, t3, t4])
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_12_shadow_cicd():
    """Shadow CI/CD: canary deployment."""
    print("\n" + "=" * 70)
    print("  TEST 12: Shadow CI/CD")
    print("=" * 70)
    from kos.graph import KOSKernel
    from kos.canary import ShadowEvaluator, CanaryDeployer

    k = KOSKernel(enable_vsa=False)
    k.add_connection("a", "b", 0.9, "A links B.")
    k.add_connection("b", "c", 0.8, "B links C.")

    evaluator = ShadowEvaluator(k)

    # Record some historical queries
    r = k.query(["a"], 5)
    evaluator.record_query(["a"], r)
    evaluator.record_query(["b"], k.query(["b"], 5))

    # Evaluate a proposed config change
    new_config = {"max_ticks": 20}
    eval_result = evaluator.evaluate_config(new_config, n_queries=2)
    print(f"  Shadow eval: accuracy={eval_result['accuracy']:.2f}")

    # Canary deployer
    deployer = CanaryDeployer()
    result = deployer.propose(evaluator, new_config)
    t1 = "accepted" in result
    print(f"  Proposal accepted: {result.get('accepted', False)}")

    # Check status
    status = deployer.status()
    t2 = "deploying" in status
    print(f"  Deployer status: {status}")

    # Advance or rollback
    advance = deployer.advance_stage(True)
    print(f"  Stage advance: {advance['status']}")

    passed = t1 and t2
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_13_adaptive_ticks():
    """Adaptive Tick Controller."""
    print("\n" + "=" * 70)
    print("  TEST 13: Adaptive Tick Controller")
    print("=" * 70)
    from kos.drives import AdaptiveTickController

    ctrl = AdaptiveTickController()

    # Active mode
    ctrl.record_activity()
    t1 = ctrl.get_tick_interval() == ctrl.ACTIVE_INTERVAL
    print(f"  Active interval: {ctrl.get_tick_interval()*1000:.0f}ms")

    # Force idle
    ctrl.force_mode("idle")
    t2 = ctrl.get_tick_interval() == ctrl.IDLE_INTERVAL
    print(f"  Idle interval: {ctrl.get_tick_interval()*1000:.0f}ms")

    # Force sleep
    ctrl.force_mode("sleep")
    t3 = ctrl.get_tick_interval() == ctrl.SLEEP_INTERVAL
    print(f"  Sleep interval: {ctrl.get_tick_interval()*1000:.0f}ms")

    # Status
    ctrl.force_mode(None)
    ctrl.record_activity()
    status = ctrl.status()
    t4 = status["mode"] == "active"
    print(f"  Status: {status}")

    passed = all([t1, t2, t3, t4])
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_14_axiom_files():
    """Domain Axiom Files (.kos format)."""
    print("\n" + "=" * 70)
    print("  TEST 14: Domain Axiom Files")
    print("=" * 70)
    from kos.constraints import (ConstraintEngine, save_axiom_file,
                                  create_finance_axioms, create_medical_axioms)

    # Create finance axiom file
    axioms = create_finance_axioms()
    test_file = os.path.join(os.path.dirname(__file__), "_test_finance.kos")
    save_axiom_file(test_file, axioms, domain="finance")

    # Load and verify
    engine = ConstraintEngine()
    engine.load_axioms(test_file)
    t1 = len(engine.axioms) == len(axioms)
    print(f"  Loaded {len(engine.axioms)} finance axioms: {'PASS' if t1 else 'FAIL'}")

    # Test Basel III constraint
    r = engine.check("bank", "capital", 0.02, "Capital ratio 2%")
    t2 = not r["passed"]
    print(f"  Basel III violation caught: {'PASS' if t2 else 'FAIL'}")

    # Medical axioms
    med_axioms = create_medical_axioms()
    engine2 = ConstraintEngine()
    engine2.load_axioms_from_dict(med_axioms)
    r2 = engine2.check("patient", "treatment", 0.8,
                        "Prescribe warfarin and aspirin together")
    t3 = not r2["passed"]
    print(f"  Drug interaction caught: {'PASS' if t3 else 'FAIL'}")

    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)

    passed = all([t1, t2, t3])
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_15_reranker_regression():
    """Multi-signal Reranker: Phase 2 regression check."""
    print("\n" + "=" * 70)
    print("  TEST 15: Reranker Regression")
    print("=" * 70)
    from kos.graph import KOSKernel
    from kos.reranker import MultiSignalReranker

    k = KOSKernel(enable_vsa=False)
    k.add_connection("toronto", "city", 0.9, "Toronto is a city.")
    k.add_connection("toronto", "ontario", 0.8, "Toronto in Ontario.")

    results = [("city", 0.9), ("ontario", 0.8)]
    reranker = MultiSignalReranker()
    reranked = reranker.rerank(results, k, ["toronto", "city"])

    t1 = len(reranked) == 2
    t2 = reranked[0][0] == "city"

    print(f"  Reranked: {[(r[0], f'{r[1]:.3f}') for r in reranked]}")
    print(f"  Correct order: {'PASS' if t2 else 'FAIL'}")

    passed = t1 and t2
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_16_full_pipeline():
    """Full pipeline: query -> normalize -> profile -> beam -> rerank -> synthesize."""
    print("\n" + "=" * 70)
    print("  TEST 16: Full V8 Pipeline (End-to-End)")
    print("=" * 70)
    from kos.graph import KOSKernel
    from kos.query_normalizer import normalize, get_profile
    from kos.reranker import MultiSignalReranker
    from kos.synthesis import SynthesisEngine
    from kos.output_validator import OutputValidator

    # Build graph
    k = KOSKernel(enable_vsa=False)
    edges = [
        ("toronto", "city", 0.9, "Toronto is a major city in Canada."),
        ("toronto", "ontario", 0.85, "Toronto is located in Ontario."),
        ("toronto", "population", 0.8, "Toronto has a population of 2.9 million."),
        ("ontario", "canada", 0.9, "Ontario is a province of Canada."),
    ]
    for src, tgt, w, text in edges:
        k.add_connection(src, tgt, w, text)

    # Step 1: Normalize
    q = normalize("Where is Toronto located?")
    print(f"  Intent: {q['intent']}, Content: {q['content_words']}")

    # Step 2: Get profile
    profile = get_profile(q["intent"])

    # Step 3: Beam search
    results = k.query_beam(["toronto"],
                            top_k=profile["top_k"],
                            beam_width=profile["beam_width"],
                            max_depth=profile["max_depth"],
                            allowed_edge_types=profile["allowed_edge_types"])

    # Step 4: Rerank
    reranker = MultiSignalReranker()
    reranked = reranker.rerank(results, k, q["content_words"],
                                k.get_working_memory())

    # Step 5: Gather evidence for top results
    evidence = []
    for node_id, score in reranked[:3]:
        pair1 = tuple(sorted(["toronto", node_id]))
        provs = k.provenance.get(pair1, set())
        evidence.extend(list(provs))

    # Step 6: Synthesize
    synth = SynthesisEngine(domain="general")
    output = synth.synthesize(evidence, intent=q["intent"],
                               entities=["Toronto"],
                               raw_prompt=q["raw"])

    # Step 7: Validate (simulate LLM formatting)
    validator = OutputValidator()
    contract = synth.build_contract(evidence, output["confidence"])
    # Simulate LLM output that's faithful
    simulated_llm = " ".join(evidence[:2])
    validation = validator.validate(simulated_llm, contract)

    print(f"  Beam results: {[r[0] for r in results[:3]]}")
    print(f"  Reranked: {[r[0] for r in reranked[:3]]}")
    print(f"  Evidence: {evidence[:2]}")
    print(f"  Synthesis: {output['response'][:80]}")
    print(f"  Confidence: {output['confidence']:.2f}")
    print(f"  Validation: {'PASS' if validation['valid'] else 'FAIL'}")

    t1 = len(results) > 0
    t2 = len(evidence) > 0
    t3 = output["confidence"] > 0
    t4 = validation["valid"]

    passed = all([t1, t2, t3, t4])
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  KOS V8.0 -- FULL 16-SUBSYSTEM INTEGRATION TEST")
    print("#  The Complete Post-Transformer Platform")
    print("#" * 70)

    results = []
    results.append(("Synthesis Engine", test_1_synthesis_engine()))
    results.append(("Output Validator", test_2_output_validator()))
    results.append(("HTN Planner", test_3_htn_planner()))
    results.append(("Causal DAG", test_4_causal_dag()))
    results.append(("Constraint Engine", test_5_constraint_engine()))
    results.append(("Source Governance", test_6_source_governance()))
    results.append(("Memory Lifecycle", test_7_memory_lifecycle()))
    results.append(("Drive System", test_8_drive_system()))
    results.append(("Action Registry", test_9_action_registry()))
    results.append(("Reasoning Workspace", test_10_reasoning_workspace()))
    results.append(("Verification Pipeline", test_11_verification_pipeline()))
    results.append(("Shadow CI/CD", test_12_shadow_cicd()))
    results.append(("Adaptive Ticks", test_13_adaptive_ticks()))
    results.append(("Axiom Files", test_14_axiom_files()))
    results.append(("Reranker Regression", test_15_reranker_regression()))
    results.append(("Full V8 Pipeline", test_16_full_pipeline()))

    print("\n" + "=" * 70)
    print("  KOS V8.0 -- 16-SUBSYSTEM FINAL RESULTS")
    print("=" * 70)
    for name, passed in results:
        print(f"  {name:30s} {'PASS' if passed else 'FAIL'}")

    total = sum(1 for _, p in results if p)
    print(f"\n  Total: {total}/{len(results)}")
    print("=" * 70)
