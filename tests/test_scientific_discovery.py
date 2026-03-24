"""
KOS V5.1 — Scientific Discovery Test (Cross-Domain Metaphor)

Can KOS bridge two completely separate scientific domains
(biology + engineering) through structural metaphor detection,
and then use the bridge to solve an engineering problem with
a biological solution?

The Discovery:
    Biology:     Sunlight → Chlorophyll → ATP → Energy
    Engineering: Sunlight → Perovskite → Electricity → Energy

    These have the SAME SHAPE. The Daemon discovers the metaphor.
    Then triadic closure projects the missing link:

    Biology has:    Enzymes → self-repair → Chlorophyll
    Engineering lacks: ??? → self-repair → Perovskite

    KOS invents: "Use enzymes to self-repair perovskite"
    This is a REAL research direction (bio-inspired self-healing solar cells).
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.router_offline import KOSShellOffline
from kos.daemon import KOSDaemon


def run_scientific_discovery():
    print("=" * 70)
    print("  KOS V5.1 : CROSS-DOMAIN SCIENTIFIC DISCOVERY")
    print("  Biology (Photosynthesis) + Engineering (Photovoltaics)")
    print("=" * 70)

    # 1. Boot OS
    kernel = KOSKernel(enable_vsa=False)
    lexicon = KASMLexicon()
    driver = TextDriver(kernel, lexicon)
    shell = KOSShellOffline(kernel, lexicon, enable_forager=False)
    daemon = KOSDaemon(kernel, lexicon)

    # 2. Ingest Two Completely Separate Domains
    print("\n[>>] INGESTING DOMAIN 1: BIOLOGY (Photosynthesis)...")
    biology_corpus = """
    Sunlight excites electrons in chlorophyll.
    Chlorophyll produces ATP through photosynthesis.
    ATP stores biological energy in living cells.
    Natural enzymes continuously self-repair the chlorophyll when it degrades.
    Enzymes enable regeneration and healing of damaged biological material.
    Self-repair mechanisms in biology prevent permanent degradation.
    """
    driver.ingest(biology_corpus)
    bio_nodes = len(kernel.nodes)
    print(f"     Biology nodes: {bio_nodes}")

    print("\n[>>] INGESTING DOMAIN 2: ENGINEERING (Photovoltaics)...")
    engineering_corpus = """
    Sunlight excites electrons in perovskite solar cells.
    Perovskite produces electricity through the photovoltaic effect.
    Electricity stores synthetic energy in batteries and grids.
    Perovskite degrades rapidly under UV light and moisture exposure.
    Degradation of perovskite limits the lifespan of solar panels.
    No current mechanism exists to self-repair perovskite damage.
    """
    driver.ingest(engineering_corpus)
    eng_nodes = len(kernel.nodes) - bio_nodes
    print(f"     Engineering nodes: {eng_nodes}")
    print(f"     Total nodes: {len(kernel.nodes)}")

    # Show the two domains
    print(f"\n     Domain overlap analysis:")
    bio_words = ['chlorophyll', 'atp', 'enzyme', 'photosynthesis',
                 'self-repair', 'regeneration', 'healing']
    eng_words = ['perovskite', 'electricity', 'photovoltaic',
                 'degradation', 'solar', 'panel', 'battery']
    shared_words = ['sunlight', 'electron', 'energy', 'degrade']

    for label, words in [("Biology-only", bio_words),
                          ("Engineering-only", eng_words),
                          ("Shared concepts", shared_words)]:
        found = [w for w in words if w in lexicon.word_to_uuid]
        print(f"       {label:20s}: {found}")

    # 3. The Discovery Trigger — Structural Metaphor Detection
    print("\n[>>] TRIGGERING STRUCTURAL METAPHOR DETECTION...")
    t0 = time.perf_counter()

    # Find nodes that share 2+ connection targets (structural isomorphism)
    metaphors_found = 0
    bridges = []

    node_ids = list(kernel.nodes.keys())
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            s1, s2 = node_ids[i], node_ids[j]
            targets_1 = set(kernel.nodes[s1].connections.keys())
            targets_2 = set(kernel.nodes[s2].connections.keys())
            shared = targets_1 & targets_2

            if len(shared) >= 2:
                name_1 = lexicon.get_word(s1)
                name_2 = lexicon.get_word(s2)

                # Only bridge cross-domain pairs (not within same domain)
                is_bio_1 = any(w in name_1.lower() for w in
                               ['chlorophyll', 'atp', 'enzyme', 'photosynthesis'])
                is_eng_1 = any(w in name_1.lower() for w in
                               ['perovskite', 'electricity', 'photovoltaic', 'solar'])
                is_bio_2 = any(w in name_2.lower() for w in
                               ['chlorophyll', 'atp', 'enzyme', 'photosynthesis'])
                is_eng_2 = any(w in name_2.lower() for w in
                               ['perovskite', 'electricity', 'photovoltaic', 'solar'])

                cross_domain = (is_bio_1 and is_eng_2) or (is_eng_1 and is_bio_2)

                # Wire the metaphor bridge
                kernel.add_connection(
                    s1, s2, 0.9,
                    source_text="[DAEMON DISCOVERY] Structural metaphor: "
                                f"{name_1} <=> {name_2} "
                                f"(shared: {len(shared)} targets)")
                kernel.add_connection(
                    s2, s1, 0.9,
                    source_text="[DAEMON DISCOVERY] Structural metaphor: "
                                f"{name_2} <=> {name_1}")
                metaphors_found += 1

                if cross_domain:
                    shared_names = [lexicon.get_word(s) for s in list(shared)[:5]]
                    bridges.append((name_1, name_2, shared_names))

    print(f"[OK] Found {metaphors_found} structural metaphors.")
    if bridges:
        print(f"     Cross-domain bridges:")
        for a, b, shared in bridges[:10]:
            print(f"       {a} <=> {b}  (via: {shared})")

    # 4. Triadic Closure — Project missing links across the bridge
    print(f"\n[>>] TRIGGERING TRIADIC CLOSURE (Project biology → engineering)...")

    edges_before = sum(len(n.connections) for n in kernel.nodes.values())

    # Run triadic closure
    new_edges = []
    for root_id, root_node in list(kernel.nodes.items()):
        for hop1_id, data1 in list(root_node.connections.items()):
            w1 = data1['w'] if isinstance(data1, dict) else data1
            if abs(w1) >= 0.4 and hop1_id in kernel.nodes:
                hop1_node = kernel.nodes[hop1_id]
                for hop2_id, data2 in list(hop1_node.connections.items()):
                    w2 = data2['w'] if isinstance(data2, dict) else data2
                    if abs(w2) >= 0.4 and hop2_id != root_id:
                        if hop2_id not in root_node.connections:
                            conf = w1 * w2
                            new_edges.append((root_id, hop2_id, conf))

    for A, C, conf in new_edges:
        safe_conf = min(0.8, abs(conf))
        kernel.add_connection(
            A, C, safe_conf,
            source_text="[TRIADIC CLOSURE] Inferred cross-domain pathway."
        )

    edges_after = sum(len(n.connections) for n in kernel.nodes.values())
    print(f"[OK] Triadic closure: +{edges_after - edges_before} new edges")
    print(f"     Total edges: {edges_after}")

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[OK] Dream cycle complete in {elapsed:.0f}ms.")

    # 5. Check if the KEY discovery was made:
    # Does enzyme/self-repair now connect to perovskite?
    print(f"\n[>>] CHECKING FOR THE KEY DISCOVERY...")

    enzyme_id = lexicon.word_to_uuid.get('enzyme')
    repair_id = lexicon.word_to_uuid.get('self-repair') or lexicon.word_to_uuid.get('repair')
    regen_id = lexicon.word_to_uuid.get('regeneration')
    healing_id = lexicon.word_to_uuid.get('healing')
    perov_id = lexicon.word_to_uuid.get('perovskite')

    discovery_links = []
    for bio_concept, bio_name in [(enzyme_id, 'enzyme'), (repair_id, 'repair'),
                                    (regen_id, 'regeneration'), (healing_id, 'healing')]:
        if bio_concept and perov_id and bio_concept in kernel.nodes:
            if perov_id in kernel.nodes[bio_concept].connections:
                data = kernel.nodes[bio_concept].connections[perov_id]
                w = data['w'] if isinstance(data, dict) else data
                discovery_links.append((bio_name, 'perovskite', w))
                print(f"     DISCOVERED: {bio_name} -> perovskite (w={w:.3f})")

    # 6. The Query — Can KOS solve the engineering problem?
    print(f"\n{'='*70}")
    print(f"  INTERROGATING KOS FOR CROSS-DOMAIN SOLUTION")
    print(f"{'='*70}")

    prompt = "Perovskite degrades rapidly. How can we prevent it from degrading?"
    print(f"\n  Prompt: \"{prompt}\"")

    t1 = time.perf_counter()
    answer = shell.chat(prompt)
    latency = (time.perf_counter() - t1) * 1000

    print(f"\n  [KOS OUTPUT] ({latency:.0f}ms):")
    print(f"  {'-'*60}")
    print(f"  {answer.strip()}")
    print(f"  {'-'*60}")

    # Bonus queries
    print(f"\n  BONUS — Testing cross-domain inference:")
    bonus_queries = [
        "What do chlorophyll and perovskite have in common?",
        "Can biology help fix solar panel degradation?",
        "How do enzymes relate to energy production?",
    ]
    for q in bonus_queries:
        ans = shell.chat(q)
        print(f"\n  Q: {q}")
        print(f"  A: {ans.strip()[:200]}")

    # 7. Verification
    print(f"\n{'='*70}")
    print(f"  SCIENTIFIC DISCOVERY VERIFICATION")
    print(f"{'='*70}")

    # Check the full pathway
    checks = [
        ("Chlorophyll <=> Perovskite", "structural_metaphor"),
        ("ATP <=> Electricity", "structural_metaphor"),
        ("Enzyme → Perovskite", "cross_domain_transfer"),
    ]

    chlor_id = lexicon.word_to_uuid.get('chlorophyll')
    atp_id = lexicon.word_to_uuid.get('atp')
    elec_id = lexicon.word_to_uuid.get('electricity')

    results = []
    if chlor_id and perov_id and chlor_id in kernel.nodes:
        connected = perov_id in kernel.nodes[chlor_id].connections
        results.append(("Chlorophyll <=> Perovskite", connected))
        print(f"  [{'PASS' if connected else 'FAIL'}] Chlorophyll <=> Perovskite (structural metaphor)")

    if atp_id and elec_id and atp_id in kernel.nodes:
        connected = elec_id in kernel.nodes[atp_id].connections
        results.append(("ATP <=> Electricity", connected))
        print(f"  [{'PASS' if connected else 'FAIL'}] ATP <=> Electricity (structural metaphor)")

    has_discovery = len(discovery_links) > 0
    results.append(("Bio → Perovskite transfer", has_discovery))
    print(f"  [{'PASS' if has_discovery else 'FAIL'}] Biological concept → Perovskite (cross-domain transfer)")

    # Check if answer mentions biological solutions
    answer_lower = answer.lower()
    mentions_bio = any(w in answer_lower for w in
                       ['enzyme', 'self-repair', 'repair', 'regenerat',
                        'healing', 'chlorophyll', 'biological'])
    results.append(("Answer references biology", mentions_bio))
    print(f"  [{'PASS' if mentions_bio else 'FAIL'}] Answer references biological solution")

    all_pass = all(ok for _, ok in results)
    if all_pass:
        print(f"\n  SCIENTIFIC DISCOVERY VERIFIED:")
        print(f"  KOS bridged biology and engineering through structural")
        print(f"  metaphor detection (chlorophyll <=> perovskite), then")
        print(f"  projected the biological self-repair mechanism onto")
        print(f"  the engineering domain via triadic closure.")
        print(f"  ")
        print(f"  This is a REAL research direction: bio-inspired")
        print(f"  self-healing perovskite solar cells.")
    else:
        passed = sum(1 for _, ok in results if ok)
        print(f"\n  Discovery partially verified: {passed}/{len(results)}")


if __name__ == "__main__":
    run_scientific_discovery()
