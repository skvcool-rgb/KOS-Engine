import time
from kos_core_v4 import KOSKernel, KOSDaemonV4
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.router import KOSShell

def run_unification_test():
    print("==========================================================")
    print(" KOS V4.1 : UNIFICATION & WEAVER VERIFICATION TEST")
    print("==========================================================")

    # 1. Boot the Unified OS
    kernel = KOSKernel()
    lexicon = KASMLexicon()
    driver = TextDriver(kernel, lexicon)
    shell = KOSShell(kernel, lexicon)
    daemon = KOSDaemonV4(kernel)

    # 2. The "Haystack" Corpus
    # We are creating a deliberate Super-Hub around "Toronto" to test
    # the Top-500 routing scaling, the Daemon, and the Weaver's intent scoring.
    corpus = """
    Toronto is a massive city with connections to dogs, cats, cars, transit, buildings, streets, lights, and noise.
    Toronto is located in the beautiful Canadian province of Ontario.
    The city of Toronto was founded and incorporated in the year 1834.
    John Graves Simcoe originally established and named the Toronto settlement.
    The Toronto Blue Jays play professional baseball in the city stadium.
    Toronto has a massive population of 2.7 million people.
    """

    print("\n[>>] Ingesting Corpus...")
    driver.ingest(corpus)

    toronto_id = lexicon.get_or_create_id("toronto")
    if toronto_id in kernel.nodes:
        conn_count = len(kernel.nodes[toronto_id].connections)
        print(f"[OK] 'Toronto' hub created successfully with {conn_count} conceptual edges.")

    # 3. VERIFY ROADBLOCK 2 IS FIXED (Daemon Mitosis Prevention)
    print("\n[>>] Triggering Background Daemon (Testing Hub Survival)...")
    daemon._contextual_mitosis()  # Manually trigger the previously destructive function

    if toronto_id in kernel.nodes:
        print(f"[PASS] 'Toronto' node survived the Daemon! Contextual Mitosis is safely disabled.")
    else:
        print(f"[FAIL] 'Toronto' was deleted by the Daemon!")
        return

    # 4. VERIFY ROADBLOCK 4 IS FIXED (Weaver Intent Routing)
    print("\n[>>] Testing Algorithmic Weaver Intent Routing...")
    print("If successful, the LLM will completely ignore baseball and cats, and answer with exact facts.")

    scenarios = [
        ("Geographic Intent", "Where is Toronto located?"),
        ("Temporal Intent", "When was Toronto founded and incorporated?"),
        ("Creator Intent", "Who established and named Toronto?"),
        ("Statistical Intent", "What is the population of Toronto?"),
        ("LAYER 5 SEMANTIC VECTOR", "Tell me about the metropolis."),
    ]

    for name, prompt in scenarios:
        print(f"\n--- {name} ---")
        print(f"Prompt: \"{prompt}\"")

        t0 = time.perf_counter()
        response = shell.chat(prompt)
        latency = (time.perf_counter() - t0) * 1000

        print(f"Latency: {latency:.2f} ms")
        print(f"Output:  {response.strip()}")

if __name__ == "__main__":
    run_unification_test()
