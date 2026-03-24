# KOS Engine — Quick Start Guide

## Install (2 minutes)

```bash
git clone https://github.com/skvcool-rgb/KOS-Engine.git
cd KOS-Engine
python deploy.py --install
```

## 4 Ways to Use KOS

---

### 1. Chat Mode (Ask questions about any topic)

```bash
python deploy.py --ui
```
Opens Streamlit at http://localhost:8501. Paste text, ask questions.

**Or use Python directly:**

```python
from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.router_offline import KOSShellOffline

# Boot
kernel = KOSKernel(enable_vsa=False)
lexicon = KASMLexicon()
driver = TextDriver(kernel, lexicon)
shell = KOSShellOffline(kernel, lexicon)

# Feed it knowledge
driver.ingest("""
Toronto is a major city in Ontario, Canada.
Toronto was founded in 1834.
The population is 2.7 million people.
""")

# Ask questions (zero LLM, zero API calls, zero cost)
print(shell.chat("When was Toronto founded?"))
# Output: Toronto was founded and incorporated in the year 1834.

print(shell.chat("What is the population?"))
# Output: The city of Toronto has a population of approximately 2.7 million people.

# Math (SymPy exact — never wrong)
print(shell.chat("345000000 * 0.0825"))
# Output: Result: 28462500.0000000
```

---

### 2. Live Agent Mode (Monitors the web autonomously)

```bash
python deploy.py --agent
```

**Or configure your own watchlist:**

```python
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

# Add URLs to monitor
agent.add_watch("https://en.wikipedia.org/wiki/Perovskite_solar_cell",
                "perovskite solar", check_interval=300)
agent.add_watch("https://en.wikipedia.org/wiki/Toronto",
                "toronto city", check_interval=300)

# Run (checks every 5 minutes, stop with Ctrl+C)
agent.run(cycle_interval=300)
```

---

### 3. Knowledge Ingestion (Feed it your documents)

```python
from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.forager import WebForager

kernel = KOSKernel(enable_vsa=False)
lexicon = KASMLexicon()
driver = TextDriver(kernel, lexicon)
forager = WebForager(kernel, lexicon, text_driver=driver)

# From text
driver.ingest("Your document text here...")

# From a local file
forager.forage_file("/path/to/document.txt")

# From Wikipedia
forager.forage_query("quantum computing")

# From arXiv (scientific papers)
forager.forage_arxiv("perovskite solar cell efficiency")

# From any URL
forager.forage("https://example.com/article")

# Save the brain for later
kernel.save_brain("my_brain.pkl")

# Load it back
kernel.load_brain("my_brain.pkl")
```

---

### 4. Self-Improvement Mode (System optimizes itself)

```python
from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.selfmod import AutoTuner, PluginManager, FormulaEvolver
from kos.propose import CodeProposer, HumanGate, Deployer

kernel = KOSKernel(enable_vsa=False)
lexicon = KASMLexicon()
driver = TextDriver(kernel, lexicon)

# Feed it knowledge first
driver.ingest("Your corpus here...")

# Level 1: Auto-tune thresholds
tuner = AutoTuner(kernel, lexicon, driver)
tuner.tune()  # Finds optimal activation_threshold, spatial_decay, etc.

# Level 2: Auto-enable modules
plugins = PluginManager(kernel, lexicon)
plugins.evaluate()  # Enables FAISS if >50K nodes, multilang if foreign queries

# Level 3: Evolve scoring formula
evolver = FormulaEvolver(kernel, lexicon, driver)
evolver.evolve()  # Genetic algorithm optimizes Weaver weights

# Level 3.5: System proposes improvements
proposer = CodeProposer(kernel, lexicon)
proposals = proposer.auto_propose()

gate = HumanGate(auto_mode=False)  # YOU must approve
for p in proposals:
    if gate.review(p):  # Shows code, asks Y/N
        result = Deployer.deploy(p, kernel, lexicon)
        print(f"Deployed: {result}")
```

---

## Key Commands

| Command | What It Does |
|---|---|
| `python deploy.py --install` | Install all dependencies |
| `python deploy.py --test` | Run all tests |
| `python deploy.py --ui` | Launch web UI |
| `python deploy.py --agent` | Start live web monitoring agent |
| `python deploy.py --all` | Install + test + UI |

## Stop the Agent

| Method | How |
|---|---|
| **Ctrl+C** | In the terminal |
| **Stop file** | Create a file called `kos_agent.stop` |
| **Kill process** | `taskkill /F /IM python.exe` (Windows) |

## Key Files

| File | What It Is |
|---|---|
| `kos/router_offline.py` | Zero-LLM query engine (20/20 tests) |
| `kos/router.py` | LLM-assisted query engine (needs OpenAI key) |
| `kos/graph.py` | The brain (spreading activation + physics) |
| `kos/predictive.py` | Predictive coding (Friston loop) |
| `kos/sensorimotor.py` | Live web agent |
| `kos/forager.py` | Web/arXiv/file knowledge acquisition |
| `kos/selfmod.py` | Self-optimization (Levels 1-3) |
| `kos/propose.py` | Supervised code proposals (Level 3.5) |
| `kasm/vsa.py` | Hyperdimensional vector engine |
| `.cache/self_tuned_config.json` | Self-tuned parameters |
