"""
KOS Engine — One-Click Deployment Script.

Run this to install everything and verify the system works:
    python deploy.py

Options:
    python deploy.py --install      Install dependencies only
    python deploy.py --test         Run all tests
    python deploy.py --agent        Start the live sensorimotor agent
    python deploy.py --ui           Launch Streamlit web UI
    python deploy.py --all          Install + test + launch UI
"""

import os
import sys
import subprocess
import argparse


def run(cmd, desc="", check=True, timeout=300):
    """Run a command with nice output."""
    if desc:
        print(f"\n  [{desc}]")
    print(f"  $ {cmd}")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        timeout=timeout, encoding='utf-8', errors='replace'
    )
    if result.returncode != 0 and check:
        print(f"  FAILED: {result.stderr[:500]}")
        return False
    return True


def install():
    """Install all dependencies."""
    print("=" * 60)
    print("  KOS ENGINE — INSTALLING DEPENDENCIES")
    print("=" * 60)

    run("pip install -r requirements.txt", "Core dependencies")
    run('python -c "import nltk; nltk.download(\'punkt_tab\', quiet=True); '
        'nltk.download(\'averaged_perceptron_tagger_eng\', quiet=True); '
        'nltk.download(\'wordnet\', quiet=True)"',
        "NLTK data")

    print("\n  Installation complete.")
    return True


def test_all():
    """Run all test suites."""
    print("=" * 60)
    print("  KOS ENGINE — RUNNING ALL TESTS")
    print("=" * 60)

    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    env['PYTHONIOENCODING'] = 'utf-8'

    tests = [
        ("KASM VSA Analogical Reasoning", "tests/test_kasm_analogy.py"),
        ("Predictive Coding", "tests/test_predictive_coding.py"),
        ("Belief Revision", "tests/test_belief_revision.py"),
    ]

    # Tests that need OpenAI API key
    api_tests = [
        ("Master Smoke Test (10-point)", "tests/master_smoke_test.py"),
        ("Unification Test", "tests/test_unification.py"),
        ("Active Inference", "tests/test_active_inference.py"),
        ("Proactive Attention", "tests/test_attention.py"),
        ("Sensorimotor Agent", "tests/test_sensorimotor.py"),
    ]

    has_api_key = bool(os.environ.get('OPENAI_API_KEY'))

    results = {}
    for name, path in tests:
        print(f"\n  --- {name} ---")
        ok = run(f"python -u {path}", check=False)
        results[name] = "PASS" if ok else "FAIL"

    if has_api_key:
        for name, path in api_tests:
            print(f"\n  --- {name} ---")
            ok = run(f"python -u {path}", check=False)
            results[name] = "PASS" if ok else "FAIL"
    else:
        print("\n  OPENAI_API_KEY not set — skipping LLM-dependent tests.")
        print("  Set it with: export OPENAI_API_KEY=sk-...")

    print("\n" + "=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)
    for name, status in results.items():
        icon = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{icon}] {name}")
    print("=" * 60)


def launch_agent():
    """Start the live sensorimotor agent."""
    print("=" * 60)
    print("  KOS ENGINE — STARTING SENSORIMOTOR AGENT")
    print("  Stop with Ctrl+C or create file: kos_agent.stop")
    print("=" * 60)

    subprocess.run(
        [sys.executable, "-u", "-c", """
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.forager import WebForager
from kos.predictive import PredictiveCodingEngine
from kos.sensorimotor import SensoriMotorAgent

kernel = KOSKernel(enable_vsa=False)
lexicon = KASMLexicon()
driver = TextDriver(kernel, lexicon)
forager = WebForager(kernel, lexicon, text_driver=driver)
pce = PredictiveCodingEngine(kernel, learning_rate=0.05)

agent = SensoriMotorAgent(kernel, lexicon, forager, pce, driver)

# Default watchlist
agent.add_watch("https://en.wikipedia.org/wiki/Artificial_intelligence",
                "artificial intelligence", check_interval=300)
agent.add_watch("https://en.wikipedia.org/wiki/Perovskite_solar_cell",
                "perovskite solar cell", check_interval=300)
agent.add_watch("https://en.wikipedia.org/wiki/Toronto",
                "toronto city canada", check_interval=300)

print("\\nAdd more URLs by editing the watchlist in this script.")
print("Agent will check each URL every 5 minutes.\\n")

agent.run(cycle_interval=300)
"""],
        encoding='utf-8', errors='replace'
    )


def launch_ui():
    """Launch the Streamlit web UI."""
    print("=" * 60)
    print("  KOS ENGINE — LAUNCHING WEB UI")
    print("  Open http://localhost:8501 in your browser")
    print("=" * 60)
    subprocess.run(["streamlit", "run", "app.py"])


def main():
    parser = argparse.ArgumentParser(
        description="KOS Engine — Deploy, Test, and Run")
    parser.add_argument('--install', action='store_true',
                        help='Install dependencies')
    parser.add_argument('--test', action='store_true',
                        help='Run all tests')
    parser.add_argument('--agent', action='store_true',
                        help='Start the live sensorimotor agent')
    parser.add_argument('--ui', action='store_true',
                        help='Launch Streamlit web UI')
    parser.add_argument('--all', action='store_true',
                        help='Install + test + launch UI')

    args = parser.parse_args()

    if args.all:
        install()
        test_all()
        launch_ui()
    elif args.install:
        install()
    elif args.test:
        test_all()
    elif args.agent:
        launch_agent()
    elif args.ui:
        launch_ui()
    else:
        # No args — show quick start
        print("=" * 60)
        print("  KOS ENGINE v5.0 — Knowledge Operating System")
        print("=" * 60)
        print()
        print("  Quick Start:")
        print("    python deploy.py --install    Install dependencies")
        print("    python deploy.py --test       Run all tests")
        print("    python deploy.py --agent      Start live web agent")
        print("    python deploy.py --ui         Launch web UI")
        print("    python deploy.py --all        Everything")
        print()
        print("  Components:")
        print("    kos/         Core engine (graph, nodes, lexicon)")
        print("    kasm/        KASM language (VSA hypervectors)")
        print("    kos_rust/    Rust acceleration (PyO3)")
        print("    tests/       Full test suite (8 test files)")
        print("    app.py       Streamlit web UI")
        print()
        print("  Requires: Python 3.10+, OPENAI_API_KEY env var")
        print("=" * 60)


if __name__ == "__main__":
    main()
