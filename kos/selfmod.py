"""
KOS V5.1 — Self-Modification Module (Levels 1-3).

Level 1: AutoTuner — adjusts thresholds based on benchmark results.
Level 2: PluginManager — enables/disables modules based on graph state.
Level 3: FormulaEvolver — evolves Weaver scoring weights via genetic programming.

SAFETY BOUNDARY: This module NEVER writes Python source code.
All tuning results are stored in a JSON config file.
The system optimizes itself but cannot modify its own architecture.
"""

import os
import json
import time
import random
import copy
from collections import defaultdict


# Config file path — all self-tuned parameters go here, NOT in source
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.cache')
_CONFIG_FILE = os.path.join(_CONFIG_DIR, 'self_tuned_config.json')


def _load_config() -> dict:
    """Load self-tuned config or return defaults."""
    if os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_config(config: dict):
    """Save self-tuned config to disk."""
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    with open(_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


# ═════════════════════════════════════════════════════════════
# LEVEL 1: AUTO-TUNER
# ═════════════════════════════════════════════════════════════

class AutoTuner:
    """
    Automatically tunes system thresholds by running benchmark
    queries at different parameter values and selecting the best.

    Tunable parameters:
        - activation_threshold (default 0.1)
        - spatial_decay (default 0.8)
        - base_threshold (default 0.05)
        - max_ticks (default 15)
        - weaver keyword_multiplier (default 20)

    The tuner NEVER modifies source code. It writes optimal
    values to a JSON config file that the system reads at startup.
    """

    # Parameters to tune with their search ranges
    TUNABLE_PARAMS = {
        'activation_threshold': {
            'default': 0.1,
            'range': [0.01, 0.05, 0.08, 0.10, 0.15, 0.20],
            'description': 'Minimum activation to include in results',
        },
        'spatial_decay': {
            'default': 0.8,
            'range': [0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
            'description': 'Energy loss per hop during propagation',
        },
        'base_threshold': {
            'default': 0.05,
            'range': [0.01, 0.02, 0.05, 0.08, 0.10],
            'description': 'Minimum energy to continue propagation',
        },
        'max_ticks': {
            'default': 15,
            'range': [10, 15, 20, 25, 30],
            'description': 'Maximum propagation depth',
        },
    }

    def __init__(self, kernel, lexicon, driver):
        self.kernel = kernel
        self.lexicon = lexicon
        self.driver = driver
        self.best_config = _load_config().get('auto_tuner', {})
        self.benchmark_results = []

    def _create_benchmark_corpus(self) -> list:
        """Create a set of test queries with known correct answers."""
        return [
            ("Where is Toronto located?", ["ontario", "province", "canadian"]),
            ("When was Toronto founded?", ["1834"]),
            ("What is the population?", ["million", "2.7"]),
            ("Climate of Toronto?", ["humid", "continental"]),
            ("How do photovoltaic cells work?", ["photon", "electricity"]),
            ("Tell me about apixaban", ["thrombosis"]),
        ]

    def _evaluate_config(self, params: dict, test_queries: list) -> float:
        """
        Run benchmark queries with given parameters and return accuracy.

        Returns: accuracy score (0.0 to 1.0)
        """
        # Temporarily apply parameters
        old_max_ticks = self.kernel.max_ticks
        self.kernel.max_ticks = params.get('max_ticks', 15)

        from .router_offline import KOSShellOffline
        shell = KOSShellOffline(self.kernel, self.lexicon, enable_forager=False)

        correct = 0
        total = len(test_queries)

        for query, expected_keywords in test_queries:
            try:
                answer = shell.chat(query)
                answer_lower = answer.lower()
                hits = sum(1 for kw in expected_keywords
                           if kw.lower() in answer_lower)
                if hits >= 1:
                    correct += 1
            except Exception:
                pass

        # Restore
        self.kernel.max_ticks = old_max_ticks

        return correct / total if total > 0 else 0.0

    def tune(self, corpus: str = None, verbose: bool = True) -> dict:
        """
        Run the auto-tuning loop.

        For each tunable parameter:
        1. Try each value in the search range
        2. Run benchmark queries
        3. Select the value with highest accuracy
        4. Save to config file

        Returns: dict of optimal parameter values
        """
        if verbose:
            print("\n[AUTO-TUNER] Starting self-optimization...")

        # Ingest test corpus if provided
        if corpus:
            self.driver.ingest(corpus)

        test_queries = self._create_benchmark_corpus()
        optimal = {}

        for param_name, param_info in self.TUNABLE_PARAMS.items():
            if verbose:
                print(f"\n  Tuning: {param_name} "
                      f"(default={param_info['default']})")

            best_value = param_info['default']
            best_accuracy = 0.0

            for value in param_info['range']:
                params = {param_name: value}
                accuracy = self._evaluate_config(params, test_queries)

                if verbose:
                    bar = "#" * int(accuracy * 20)
                    print(f"    {value:>6.2f} -> {accuracy:.0%} {bar}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_value = value

            optimal[param_name] = {
                'value': best_value,
                'accuracy': best_accuracy,
            }

            if verbose:
                changed = best_value != param_info['default']
                tag = " [CHANGED]" if changed else " [DEFAULT]"
                print(f"    OPTIMAL: {best_value}{tag}")

        # Save to config
        config = _load_config()
        config['auto_tuner'] = optimal
        config['auto_tuner_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        _save_config(config)

        self.best_config = optimal

        if verbose:
            print(f"\n[AUTO-TUNER] Config saved to {_CONFIG_FILE}")

        return optimal

    def get_tuned_value(self, param_name: str):
        """Get the tuned value for a parameter, or its default."""
        if param_name in self.best_config:
            return self.best_config[param_name]['value']
        if param_name in self.TUNABLE_PARAMS:
            return self.TUNABLE_PARAMS[param_name]['default']
        return None


# ═════════════════════════════════════════════════════════════
# LEVEL 2: PLUGIN MANAGER (Auto-Architecture)
# ═════════════════════════════════════════════════════════════

class PluginManager:
    """
    Automatically enables/disables KOS modules based on
    graph state and usage patterns.

    Rules:
        graph > 50K nodes → enable FAISS scaling
        graph < 1K nodes → disable FAISS (overhead)
        multi-language queries → enable multilang
        no temporal queries → disable temporal
        high contradiction rate → enable contradiction alerts
        low entropy → disable Active Inference

    The manager doesn't install new code — it activates/deactivates
    existing modules from the kos/ directory.
    """

    def __init__(self, kernel, lexicon=None):
        self.kernel = kernel
        self.lexicon = lexicon

        # Plugin registry: name → {module, enabled, condition}
        self.plugins = {
            'faiss_scaling': {
                'enabled': False,
                'module': 'kos.scaling',
                'reason': None,
            },
            'temporal': {
                'enabled': True,
                'module': 'kos.temporal',
                'reason': None,
            },
            'multilang': {
                'enabled': False,
                'module': 'kos.multilang',
                'reason': None,
            },
            'predictive_coding': {
                'enabled': True,
                'module': 'kos.predictive',
                'reason': None,
            },
            'attention_controller': {
                'enabled': True,
                'module': 'kos.attention',
                'reason': None,
            },
            'sensorimotor': {
                'enabled': False,
                'module': 'kos.sensorimotor',
                'reason': None,
            },
            'vsa_backplane': {
                'enabled': False,
                'module': 'kasm.bridge',
                'reason': None,
            },
        }

        # Usage counters
        self.temporal_query_count = 0
        self.multilang_query_count = 0
        self.total_query_count = 0

    def evaluate(self, verbose: bool = True) -> dict:
        """
        Evaluate graph state and adjust plugin activation.

        Returns dict of changes made.
        """
        changes = {}
        node_count = len(self.kernel.nodes)
        edge_count = sum(len(n.connections) for n in self.kernel.nodes.values())
        contradiction_count = len(getattr(self.kernel, 'contradictions', []))

        if verbose:
            print(f"\n[PLUGINS] Evaluating graph state...")
            print(f"  Nodes: {node_count:,} | Edges: {edge_count:,} | "
                  f"Contradictions: {contradiction_count}")

        # Rule 1: FAISS scaling
        if node_count > 50_000 and not self.plugins['faiss_scaling']['enabled']:
            self.plugins['faiss_scaling']['enabled'] = True
            self.plugins['faiss_scaling']['reason'] = (
                f"Graph exceeded 50K nodes ({node_count:,})")
            changes['faiss_scaling'] = 'ENABLED'
        elif node_count < 10_000 and self.plugins['faiss_scaling']['enabled']:
            self.plugins['faiss_scaling']['enabled'] = False
            self.plugins['faiss_scaling']['reason'] = (
                f"Graph below 10K nodes ({node_count:,})")
            changes['faiss_scaling'] = 'DISABLED'

        # Rule 2: Multi-language (activate after 3+ foreign queries)
        if self.multilang_query_count >= 3:
            if not self.plugins['multilang']['enabled']:
                self.plugins['multilang']['enabled'] = True
                self.plugins['multilang']['reason'] = (
                    f"Detected {self.multilang_query_count} multi-language queries")
                changes['multilang'] = 'ENABLED'

        # Rule 3: VSA backplane (enable if graph is large enough for analogies)
        if node_count > 100 and not self.plugins['vsa_backplane']['enabled']:
            self.plugins['vsa_backplane']['enabled'] = True
            self.plugins['vsa_backplane']['reason'] = (
                f"Graph has {node_count} nodes — enough for analogy detection")
            changes['vsa_backplane'] = 'ENABLED'

        # Rule 4: Sensorimotor (enable if forager is active)
        # Left as manual activation for safety

        if verbose:
            for name, status in changes.items():
                reason = self.plugins[name]['reason']
                print(f"  [{status}] {name}: {reason}")
            if not changes:
                print(f"  No changes needed.")

        # Save state
        config = _load_config()
        config['plugins'] = {name: info['enabled']
                              for name, info in self.plugins.items()}
        _save_config(config)

        return changes

    def is_enabled(self, plugin_name: str) -> bool:
        return self.plugins.get(plugin_name, {}).get('enabled', False)

    def record_query(self, query: str):
        """Track query patterns for plugin decisions."""
        self.total_query_count += 1

        # Detect multi-language
        from .multilang import detect_language
        lang = detect_language(query)
        if lang != 'en':
            self.multilang_query_count += 1

        # Detect temporal
        query_lower = query.lower()
        temporal_words = {'when', 'before', 'after', 'year', 'founded',
                          'first', 'oldest', 'newest', 'century'}
        if temporal_words & set(query_lower.split()):
            self.temporal_query_count += 1

    def get_status(self) -> dict:
        return {name: {'enabled': info['enabled'], 'reason': info['reason']}
                for name, info in self.plugins.items()}


# ═════════════════════════════════════════════════════════════
# LEVEL 3: FORMULA EVOLVER (Genetic Programming)
# ═════════════════════════════════════════════════════════════

class FormulaEvolver:
    """
    Evolves the Weaver's scoring formula using genetic programming.

    The genome: a dict of scoring weights
        {WHERE_BOOST: 40, WHEN_BOOST: 40, WHO_BOOST: 40,
         ATTRIBUTE_BOOST: 35, HOW_BOOST: 30, SOLVE_BOOST: 45,
         KEYWORD_MULTIPLIER: 20, NOISE_PENALTY: -50,
         DENSITY_MULTIPLIER: 10}

    Evolution loop:
        1. Generate population of 50 random weight variations
        2. Evaluate each against benchmark queries
        3. Select top 10 (tournament selection)
        4. Crossover + mutation → next generation
        5. Repeat for N generations
        6. Deploy winner to config (NOT to source code)

    SAFETY: The evolver can only modify NUMERIC WEIGHTS.
    It cannot add new scoring rules, change code logic,
    or modify any Python file.
    """

    # The genome template (parameter name → default value)
    GENOME_TEMPLATE = {
        'WHERE_BOOST': 40,
        'WHEN_BOOST': 40,
        'WHO_BOOST': 40,
        'ATTRIBUTE_BOOST': 35,
        'HOW_BOOST': 30,
        'SOLVE_BOOST': 45,
        'KEYWORD_MULTIPLIER': 20,
        'NOISE_PENALTY': -50,
        'DENSITY_MULTIPLIER': 10,
        'SHORT_SENTENCE_PENALTY': -15,
        'DAEMON_PENALTY': -30,
    }

    # Mutation ranges (how much each parameter can shift)
    MUTATION_RANGE = {
        'WHERE_BOOST': (-15, 15),
        'WHEN_BOOST': (-15, 15),
        'WHO_BOOST': (-15, 15),
        'ATTRIBUTE_BOOST': (-15, 15),
        'HOW_BOOST': (-15, 15),
        'SOLVE_BOOST': (-15, 15),
        'KEYWORD_MULTIPLIER': (-10, 10),
        'NOISE_PENALTY': (-20, 20),
        'DENSITY_MULTIPLIER': (-5, 5),
        'SHORT_SENTENCE_PENALTY': (-10, 10),
        'DAEMON_PENALTY': (-15, 15),
    }

    def __init__(self, kernel, lexicon, driver,
                 population_size: int = 30,
                 generations: int = 20,
                 mutation_rate: float = 0.3):
        self.kernel = kernel
        self.lexicon = lexicon
        self.driver = driver
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.rng = random.Random(42)
        self.best_genome = dict(self.GENOME_TEMPLATE)
        self.best_fitness = 0.0
        self.evolution_log = []

    def _create_benchmark(self) -> list:
        """Benchmark queries with known correct answers."""
        return [
            ("Where is Toronto located?", ["ontario", "province"]),
            ("When was Toronto founded?", ["1834"]),
            ("Who established Toronto?", ["simcoe"]),
            ("What is the population of Toronto?", ["million", "2.7"]),
            ("Climate of Toronto?", ["humid", "continental"]),
            ("How do photovoltaic cells work?", ["photon", "electricity"]),
            ("Tell me about apixaban", ["thrombosis"]),
            ("Tell me about the metropolis", ["toronto", "city"]),
        ]

    def _apply_genome_to_weaver(self, genome: dict):
        """Temporarily apply a genome's weights to the Weaver."""
        from .weaver import AlgorithmicWeaver
        AlgorithmicWeaver.WHERE_BOOST = genome['WHERE_BOOST']
        AlgorithmicWeaver.WHEN_BOOST = genome['WHEN_BOOST']
        AlgorithmicWeaver.WHO_BOOST = genome['WHO_BOOST']
        AlgorithmicWeaver.ATTRIBUTE_BOOST = genome['ATTRIBUTE_BOOST']
        AlgorithmicWeaver.HOW_BOOST = genome['HOW_BOOST']
        AlgorithmicWeaver.KEYWORD_MULTIPLIER = genome['KEYWORD_MULTIPLIER']
        AlgorithmicWeaver.NOISE_PENALTY = genome['NOISE_PENALTY']
        AlgorithmicWeaver.DENSITY_MULTIPLIER = genome['DENSITY_MULTIPLIER']
        AlgorithmicWeaver.SHORT_SENTENCE_PENALTY = genome['SHORT_SENTENCE_PENALTY']
        AlgorithmicWeaver.DAEMON_PENALTY = genome['DAEMON_PENALTY']

    def _evaluate_genome(self, genome: dict, benchmark: list) -> float:
        """Evaluate a genome's fitness against benchmark."""
        self._apply_genome_to_weaver(genome)

        from .router_offline import KOSShellOffline
        shell = KOSShellOffline(self.kernel, self.lexicon, enable_forager=False)

        correct = 0
        for query, expected in benchmark:
            try:
                answer = shell.chat(query).lower()
                if any(kw.lower() in answer for kw in expected):
                    correct += 1
            except Exception:
                pass

        return correct / len(benchmark) if benchmark else 0.0

    def _mutate(self, genome: dict) -> dict:
        """Mutate a genome by randomly shifting weights."""
        child = dict(genome)
        for param in child:
            if self.rng.random() < self.mutation_rate:
                lo, hi = self.MUTATION_RANGE.get(param, (-10, 10))
                delta = self.rng.randint(lo, hi)
                child[param] = child[param] + delta
        return child

    def _crossover(self, parent_a: dict, parent_b: dict) -> dict:
        """Single-point crossover of two genomes."""
        child = {}
        params = list(parent_a.keys())
        split = self.rng.randint(1, len(params) - 1)
        for i, param in enumerate(params):
            if i < split:
                child[param] = parent_a[param]
            else:
                child[param] = parent_b[param]
        return child

    def evolve(self, corpus: str = None, verbose: bool = True) -> dict:
        """
        Run the genetic evolution loop.

        Returns the best genome (optimal Weaver weights).
        """
        if verbose:
            print("\n[EVOLVER] Starting genetic formula evolution...")
            print(f"  Population: {self.pop_size} | "
                  f"Generations: {self.generations} | "
                  f"Mutation rate: {self.mutation_rate:.0%}")

        if corpus:
            self.driver.ingest(corpus)

        benchmark = self._create_benchmark()

        # Initialize population
        population = [dict(self.GENOME_TEMPLATE)]  # Include default
        for _ in range(self.pop_size - 1):
            population.append(self._mutate(dict(self.GENOME_TEMPLATE)))

        t0 = time.perf_counter()

        for gen in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for genome in population:
                fitness = self._evaluate_genome(genome, benchmark)
                fitness_scores.append((fitness, genome))

            # Sort by fitness (descending)
            fitness_scores.sort(key=lambda x: x[0], reverse=True)

            best_fitness = fitness_scores[0][0]
            best_genome = fitness_scores[0][1]
            avg_fitness = sum(f for f, _ in fitness_scores) / len(fitness_scores)

            self.evolution_log.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
            })

            if verbose and (gen % 5 == 0 or gen == self.generations - 1):
                print(f"  Gen {gen:3d}: best={best_fitness:.0%} "
                      f"avg={avg_fitness:.0%} "
                      f"{'#' * int(best_fitness * 20)}")

            # Early stop if perfect
            if best_fitness >= 1.0:
                if verbose:
                    print(f"  PERFECT FITNESS at generation {gen}!")
                break

            # Selection: top 30% survive
            survivors = [g for _, g in fitness_scores[:max(2, self.pop_size // 3)]]

            # Breed next generation
            next_gen = list(survivors)  # Elitism: keep survivors
            while len(next_gen) < self.pop_size:
                parent_a = self.rng.choice(survivors)
                parent_b = self.rng.choice(survivors)
                child = self._crossover(parent_a, parent_b)
                child = self._mutate(child)
                next_gen.append(child)

            population = next_gen

        elapsed = (time.perf_counter() - t0) * 1000

        # Apply best genome permanently
        self.best_genome = best_genome
        self.best_fitness = best_fitness
        self._apply_genome_to_weaver(best_genome)

        # Save to config
        config = _load_config()
        config['evolved_formula'] = best_genome
        config['evolved_fitness'] = best_fitness
        config['evolved_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        _save_config(config)

        if verbose:
            print(f"\n[EVOLVER] Evolution complete in {elapsed:.0f}ms")
            print(f"  Best fitness: {best_fitness:.0%}")
            print(f"  Evolved weights vs defaults:")
            for param in self.GENOME_TEMPLATE:
                default = self.GENOME_TEMPLATE[param]
                evolved = best_genome[param]
                changed = " [EVOLVED]" if default != evolved else ""
                print(f"    {param:25s}: {default:>5d} -> {evolved:>5d}{changed}")
            print(f"  Config saved to {_CONFIG_FILE}")

        return best_genome
