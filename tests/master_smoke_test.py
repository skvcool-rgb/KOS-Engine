import time
from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.router import KOSShell

def run_master_diagnostics():
    print("==========================================================")
    print(" KOS V2.0 : MASTER ENTERPRISE SMOKE TEST (10 SCENARIOS)")
    print("==========================================================")

    # 1. Boot OS Components
    t0 = time.perf_counter()
    kernel = KOSKernel()
    lexicon = KASMLexicon()
    text_driver = TextDriver(kernel, lexicon)
    shell = KOSShell(kernel, lexicon)
    boot_time = (time.perf_counter() - t0) * 1000
    print(f"[OK] System Kernel & Lexicon Booted ({boot_time:.2f} ms)")

    # 2. Ingest Cross-Domain Enterprise Data
    print("\n[>>] INGESTING UNSTRUCTURED KNOWLEDGE BASE...")

    corpus = """
    The corporate merger joined the parent company and the subsidiary enterprise. They face immediate antitrust regulation.
    Unlike traditional warfarin, apixaban prevents thrombosis without requiring strict diets.
    Traditional silicon is used in old computing. Unlike traditional silicon, perovskite is highly efficient for modern photovoltaic cells.
    Photovoltaic cells capture photons. Photons produce electricity.
    A silicon wafer is incredibly brittle. Silicone rubber is highly flexible and used for sealing.
    """

    t1 = time.perf_counter()
    text_driver.ingest(corpus)
    ingest_time = (time.perf_counter() - t1) * 1000

    print(f"[OK] Data Pipeline Ingested ({ingest_time:.2f} ms)")
    print(f"    - Total Unique Math Concepts (UUIDs): {len(kernel.nodes)}")
    print(f"    - Structural Physics Edges: {sum(len(n.connections) for n in kernel.nodes.values())}")

    # 3. The 10 Extreme Scenarios
    scenarios = [
        {
            "category": "1. SVO SLIDER PRECISION",
            "intent": "Tests the new dependency slider! Ensures 'diet' does not defeat 'apixaban' in preventing thrombosis.",
            "prompt": "What drug prevents thrombosis instead of traditional warfarin?"
        },
        {
            "category": "2. SPLIT-ANTECEDENT COREF",
            "intent": "Tests if 'They' accurately maps backward to both 'company' and 'subsidiary' simultaneously.",
            "prompt": "What entities face antitrust regulation?"
        },
        {
            "category": "3. EXTREME TYPO PHONETICS",
            "intent": "Tests if the Metaphone hash rescues completely butchered spelling.",
            "prompt": "wht is so gud about prpvskittes?"
        },
        {
            "category": "4. SPANGLISH & JUNK FILTERING",
            "intent": "Tests multi-lingual stopword removal and phonetic execution without an LLM.",
            "prompt": "que material es mas eficiente que el silicon tradicional?"
        },
        {
            "category": "5. SYNONYM NETWORKS",
            "intent": "WordNet integration test. The text says 'photovoltaic cells', but user asks for 'solar cells'.",
            "prompt": "What captures photons to power solar cells?"
        },
        {
            "category": "6. AMBIGUITY CLARIFICATION LOOP",
            "intent": "User types 'silcon'. The system knows 'silicon wafer' and 'silicone rubber'. It must refuse to guess and ask for clarity.",
            "prompt": "Tell me about the silcon."
        },
        {
            "category": "7. MULTI-HOP DEDUCTION",
            "intent": "Evaluates traversal: Perovskite -> Photovoltaic Cell -> Photons -> Electricity.",
            "prompt": "If I use perovskite, what exactly produces the electricity?"
        },
        {
            "category": "8. BIG ARITHMETIC (ALU)",
            "intent": "Triggers the ALU coprocessor for float math, avoiding transformer digit hallucination.",
            "prompt": "calculate 345000000 * 0.0825"
        },
        {
            "category": "9. CALCULUS INTEGRATION (ALU)",
            "intent": "Tests analytical Abstract Syntax Tree formatting for university-level calculus.",
            "prompt": "calculate integrate x**3 * log(x)"
        },
        {
            "category": "10. CALCULUS DIFFERENTIATION (ALU)",
            "intent": "Tests differential execution and product rule via SymPy exact math.",
            "prompt": "derivative of exp(x) * cos(x) * sin(x)"
        }
    ]

    print("\n==========================================================")
    print(" EXECUTING 10-STAGE DIAGNOSTICS")
    print("==========================================================")

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[TEST {i}] {scenario['category']}")
        print(f"   Goal:   {scenario['intent']}")
        print(f"   Prompt: \"{scenario['prompt']}\"")

        t2 = time.perf_counter()
        answer = shell.chat(scenario['prompt'])
        query_latency = (time.perf_counter() - t2) * 1000

        print(f"   Time:   {query_latency:.2f} ms")
        print(f"   OUTPUT:\n{'-'*60}\n{answer.strip()}\n{'-'*60}")

if __name__ == "__main__":
    run_master_diagnostics()
