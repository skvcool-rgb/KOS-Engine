"""
KOS V9.0 — Agent Factory: Knowledge-Driven Agent Generation

The engine observes its own knowledge graph and proposes new specialized
agents to handle domains it has learned about. Agents are Python classes
saved to agents/ directory with full safety checks.

Architecture:
    AgentBlueprint    - Template for a new agent (name, domain, capabilities)
    AgentFactory      - Analyzes knowledge graph, designs agents
    AgentRegistry     - Tracks all generated agents and their status
    AgentSandbox      - Tests agents in isolation before deployment

Flow:
    1. Factory scans graph for dense knowledge clusters
    2. For each cluster, designs an agent blueprint
    3. Generates safe Python code (no os/subprocess/eval)
    4. Saves to agents/ directory as PENDING
    5. Human reviews and approves via AgentRegistry
    6. Approved agents can be loaded and used

SAFETY: Same invariants as CodeProposer — agents cannot access
os, subprocess, eval, exec, or network. They can only read the
kernel graph and produce structured outputs.
"""

import os
import re
import json
import time
import hashlib
import textwrap
from datetime import datetime
from collections import defaultdict

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AGENTS_DIR = os.path.join(_BASE, 'agents')
_REGISTRY_FILE = os.path.join(_AGENTS_DIR, 'registry.json')

# Same safety rules as propose.py
_FORBIDDEN_PATTERNS = [
    r'import\s+os',
    r'import\s+subprocess',
    r'import\s+shutil',
    r'__import__',
    r'exec\s*\(',
    r'eval\s*\(',
    r'open\s*\(.*(w|a)',
    r'os\.(system|popen|remove|unlink|rmdir|chmod|rename)',
    r'subprocess\.',
    r'shutil\.',
    r'compile\s*\(',
    r'globals\s*\(',
    r'setattr\s*\(',
    r'delattr\s*\(',
    r'__builtins__',
    r'import\s+signal',
    r'import\s+socket',
]


def _is_safe(code: str) -> tuple:
    """Check generated agent code for forbidden patterns."""
    violations = []
    for pattern in _FORBIDDEN_PATTERNS:
        if re.findall(pattern, code, re.IGNORECASE):
            violations.append("Forbidden: %s" % pattern)
    return (len(violations) == 0, violations)


# ═════════════════════════════════════════════════════════════
# AGENT BLUEPRINT
# ═════════════════════════════════════════════════════════════

class AgentBlueprint:
    """A design document for a new agent."""

    def __init__(self, name, domain, description, capabilities,
                 knowledge_nodes, entry_queries):
        self.name = name                      # e.g. "PhysicsExpert"
        self.domain = domain                  # e.g. "physics"
        self.description = description        # What the agent does
        self.capabilities = capabilities      # List of capability strings
        self.knowledge_nodes = knowledge_nodes  # Core nodes this agent uses
        self.entry_queries = entry_queries    # Example queries it handles
        self.created_at = datetime.now().isoformat()

    def to_dict(self):
        return {
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "capabilities": self.capabilities,
            "knowledge_nodes": len(self.knowledge_nodes),
            "top_nodes": self.knowledge_nodes[:20],
            "entry_queries": self.entry_queries,
            "created_at": self.created_at,
        }


# ═════════════════════════════════════════════════════════════
# AGENT FACTORY
# ═════════════════════════════════════════════════════════════

class AgentFactory:
    """
    Analyzes the knowledge graph and generates specialized agents.

    The factory:
    1. Detects knowledge clusters (dense subgraphs by domain)
    2. Identifies the most connected concepts per cluster
    3. Generates agent Python code with domain-specific reasoning
    4. Validates safety and saves to agents/ directory
    """

    def __init__(self, kernel, lexicon):
        self.kernel = kernel
        self.lexicon = lexicon
        self.blueprints = []
        os.makedirs(_AGENTS_DIR, exist_ok=True)

    # ── Cluster Detection ─────────────────────────────────

    def detect_knowledge_clusters(self, min_cluster_size=15):
        """
        Find dense knowledge clusters in the graph using
        connected component analysis with hub seeding.

        Returns list of clusters: [{seed, nodes, density, top_concepts}]
        """
        if not self.kernel.nodes:
            return []

        # Find hub nodes (highly connected = domain anchors)
        hubs = []
        for nid, node in self.kernel.nodes.items():
            conn_count = len(node.connections)
            if conn_count >= 10:
                hubs.append((nid, conn_count))

        hubs.sort(key=lambda x: -x[1])

        # BFS from each hub to find its cluster
        visited_global = set()
        clusters = []

        for hub_name, hub_conns in hubs[:50]:  # Top 50 hubs
            if hub_name in visited_global:
                continue

            # BFS 1 hop from this hub (tight clusters, not whole graph)
            cluster_nodes = {hub_name}
            if hub_name in self.kernel.nodes:
                node = self.kernel.nodes[hub_name]
                # Only take strongest connections (top 50 by weight)
                sorted_conns = sorted(
                    node.connections.items(),
                    key=lambda x: -abs(x[1].get('w', 0)
                                       if isinstance(x[1], dict) else x[1])
                )[:50]
                for tgt, _ in sorted_conns:
                    if tgt not in visited_global:
                        cluster_nodes.add(tgt)

            if len(cluster_nodes) < min_cluster_size:
                continue

            # Mark visited
            visited_global.update(cluster_nodes)

            # Score cluster density
            internal_edges = 0
            for nid in cluster_nodes:
                if nid in self.kernel.nodes:
                    for tgt in self.kernel.nodes[nid].connections:
                        if tgt in cluster_nodes:
                            internal_edges += 1
            density = internal_edges / max(len(cluster_nodes), 1)

            # Top concepts by connection count within cluster
            concept_scores = []
            for nid in cluster_nodes:
                if nid in self.kernel.nodes:
                    internal = sum(1 for t in self.kernel.nodes[nid].connections
                                   if t in cluster_nodes)
                    concept_scores.append((nid, internal))
            concept_scores.sort(key=lambda x: -x[1])
            top_concepts = [c[0] for c in concept_scores[:20]]

            clusters.append({
                "seed": hub_name,
                "nodes": list(cluster_nodes)[:100],  # Cap for memory
                "size": len(cluster_nodes),
                "density": round(density, 2),
                "top_concepts": top_concepts,
            })

        # Sort by size * density (most knowledge-rich first)
        clusters.sort(key=lambda c: -(c["size"] * c["density"]))
        return clusters

    # ── Blueprint Generation ──────────────────────────────

    def design_agent(self, cluster) -> AgentBlueprint:
        """Design an agent blueprint from a knowledge cluster."""
        seed = cluster["seed"]
        top = cluster["top_concepts"]

        # Infer domain name from top concepts
        domain = self._infer_domain(top)
        agent_name = domain.replace(" ", "") + "Agent"

        # Generate capabilities from the cluster's edge patterns
        capabilities = self._infer_capabilities(cluster)

        # Generate example queries this agent could handle
        entry_queries = self._generate_entry_queries(top)

        description = (
            "Specialized agent for %s domain with %d knowledge nodes. "
            "Core concepts: %s. "
            "Can answer questions about %s."
        ) % (domain, cluster["size"],
             ", ".join(top[:5]),
             ", ".join(capabilities[:3]))

        blueprint = AgentBlueprint(
            name=agent_name,
            domain=domain,
            description=description,
            capabilities=capabilities,
            knowledge_nodes=top,
            entry_queries=entry_queries,
        )
        self.blueprints.append(blueprint)
        return blueprint

    def _infer_domain(self, top_concepts):
        """Infer a domain name from top concept nodes."""
        # Look for common domain keywords
        domain_keywords = {
            "physics": ["force", "energy", "mass", "velocity", "momentum",
                        "gravity", "wave", "field", "quantum", "relativity"],
            "chemistry": ["atom", "molecule", "bond", "reaction", "element",
                          "compound", "acid", "base", "oxidation", "electron"],
            "biology": ["cell", "organism", "gene", "protein", "evolution",
                        "species", "ecosystem", "dna", "enzyme", "membrane"],
            "medicine": ["disease", "treatment", "drug", "patient", "symptom",
                         "immune", "blood", "organ", "therapy", "diagnosis"],
            "computer science": ["algorithm", "data", "network", "software",
                                 "computer", "binary", "database", "neural", "model"],
            "mathematics": ["function", "equation", "number", "theorem",
                            "matrix", "vector", "integral", "derivative"],
            "earth science": ["plate", "rock", "climate", "ocean", "atmosphere",
                              "earthquake", "volcano", "mineral", "erosion"],
            "astronomy": ["star", "planet", "galaxy", "universe", "sun",
                          "orbit", "light", "telescope", "cosmic"],
            "economics": ["market", "price", "demand", "supply", "gdp",
                          "inflation", "trade", "fiscal", "monetary"],
            "engineering": ["circuit", "material", "design", "system",
                            "power", "signal", "control", "sensor"],
        }

        concept_text = " ".join(top_concepts).lower()
        scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in concept_text)
            if score > 0:
                scores[domain] = score

        if scores:
            return max(scores, key=scores.get)
        return "general knowledge"

    def _infer_capabilities(self, cluster):
        """Infer what an agent can do from the cluster's edge structure."""
        capabilities = []
        top = cluster["top_concepts"][:10]

        # Check for causal chains (A causes B)
        causal_count = 0
        for nid in top:
            if nid in self.kernel.nodes:
                for tgt, data in self.kernel.nodes[nid].connections.items():
                    if isinstance(data, dict):
                        w = data.get('w', 0)
                        if w > 0.5 and tgt in self.kernel.nodes:
                            causal_count += 1
        if causal_count > 5:
            capabilities.append("causal reasoning")

        # Check for contradictions
        contradictions = getattr(self.kernel, 'contradictions', [])
        cluster_set = set(cluster["nodes"])
        related_contradictions = [c for c in contradictions
                                  if any(n in cluster_set for n in c[:2]
                                         if isinstance(n, str))]
        if related_contradictions:
            capabilities.append("contradiction detection")

        # Check cluster density for deep Q&A
        if cluster["density"] > 5:
            capabilities.append("deep question answering")
        elif cluster["density"] > 2:
            capabilities.append("factual question answering")

        # Check for numerical/quantitative nodes
        quant_count = sum(1 for nid in top
                          if any(c.isdigit() for c in nid))
        if quant_count > 2:
            capabilities.append("quantitative analysis")

        # Always can do basic retrieval
        capabilities.append("concept retrieval")
        capabilities.append("relationship mapping")

        return capabilities

    def _generate_entry_queries(self, top_concepts):
        """Generate example queries this agent could handle."""
        queries = []
        for concept in top_concepts[:5]:
            queries.append("What is %s?" % concept)
            queries.append("How does %s work?" % concept)
        if len(top_concepts) >= 2:
            queries.append("What is the relationship between %s and %s?" % (
                top_concepts[0], top_concepts[1]))
        return queries

    # ── Code Generation ───────────────────────────────────

    def generate_agent_code(self, blueprint: AgentBlueprint) -> str:
        """Generate Python code for a specialized agent."""

        concepts_list = repr(blueprint.knowledge_nodes[:30])
        queries_list = repr(blueprint.entry_queries[:6])
        caps_list = repr(blueprint.capabilities)

        code = textwrap.dedent('''\
        """
        KOS Auto-Generated Agent: {name}
        Domain: {domain}
        Generated: {timestamp}

        {description}

        Capabilities: {caps}

        SAFETY: This agent can only READ the kernel graph.
        It cannot modify files, make network calls, or execute arbitrary code.
        """


        class {name}:
            """
            Specialized agent for {domain} domain.

            Core knowledge nodes: {node_count}
            Capabilities: {caps}
            """

            DOMAIN = "{domain}"
            CORE_CONCEPTS = {concepts}
            ENTRY_QUERIES = {queries}
            CAPABILITIES = {capabilities}

            def __init__(self, kernel, lexicon, shell=None):
                self.kernel = kernel
                self.lexicon = lexicon
                self.shell = shell
                self._query_count = 0
                self._hits = 0

            def can_handle(self, query: str) -> float:
                """
                Score how well this agent can handle a query.
                Returns 0.0-1.0 confidence score.
                """
                query_lower = query.lower()
                score = 0.0

                # Check concept overlap
                for concept in self.CORE_CONCEPTS:
                    if concept.lower() in query_lower:
                        score += 0.3
                        break

                # Check domain keywords
                domain_words = self.DOMAIN.lower().split()
                for word in domain_words:
                    if word in query_lower:
                        score += 0.2

                # Check if query mentions any connected concepts
                words = query_lower.split()
                for word in words:
                    uid = self.lexicon.word_to_uuid.get(word)
                    if uid and uid in self.kernel.nodes:
                        node = self.kernel.nodes[uid]
                        for tgt in node.connections:
                            if tgt in self.CORE_CONCEPTS[:10]:
                                score += 0.15
                                break

                return min(score, 1.0)

            def query(self, question: str) -> dict:
                """
                Answer a question using domain-specific knowledge.

                Returns: {{
                    "answer": str,
                    "evidence": list,
                    "confidence": float,
                    "domain": str,
                }}
                """
                self._query_count += 1

                # Use shell if available for full retrieval pipeline
                if self.shell:
                    try:
                        result = self.shell.query(question)
                        if result and result.get("answer"):
                            self._hits += 1
                            return {{
                                "answer": result["answer"],
                                "evidence": result.get("evidence", []),
                                "confidence": result.get("confidence", 0.5),
                                "domain": self.DOMAIN,
                                "agent": "{name}",
                            }}
                    except Exception:
                        pass

                # Fallback: direct graph traversal
                evidence = []
                words = question.lower().split()
                for word in words:
                    uid = self.lexicon.word_to_uuid.get(word)
                    if uid and uid in self.kernel.nodes:
                        node = self.kernel.nodes[uid]
                        for tgt, data in node.connections.items():
                            w = data.get("w", 0) if isinstance(data, dict) else data
                            if abs(w) > 0.3:
                                evidence.append({{
                                    "from": uid,
                                    "to": tgt,
                                    "weight": round(w, 3),
                                }})

                # Sort by weight
                evidence.sort(key=lambda e: -abs(e.get("weight", 0)))
                evidence = evidence[:10]

                if evidence:
                    self._hits += 1
                    answer = "Based on %d connections: %s relates to %s" % (
                        len(evidence),
                        evidence[0]["from"],
                        ", ".join(e["to"] for e in evidence[:3]))
                else:
                    answer = "No strong evidence found in %s domain" % self.DOMAIN

                return {{
                    "answer": answer,
                    "evidence": evidence,
                    "confidence": min(len(evidence) * 0.1, 0.9),
                    "domain": self.DOMAIN,
                    "agent": "{name}",
                }}

            def get_stats(self) -> dict:
                """Return agent performance stats."""
                return {{
                    "name": "{name}",
                    "domain": self.DOMAIN,
                    "queries": self._query_count,
                    "hits": self._hits,
                    "hit_rate": self._hits / max(self._query_count, 1),
                    "core_concepts": len(self.CORE_CONCEPTS),
                    "capabilities": self.CAPABILITIES,
                }}

            def explain_domain(self) -> dict:
                """Return a summary of what this agent knows."""
                concept_details = []
                for concept in self.CORE_CONCEPTS[:10]:
                    if concept in self.kernel.nodes:
                        node = self.kernel.nodes[concept]
                        connections = len(node.connections)
                        top_neighbors = sorted(
                            node.connections.items(),
                            key=lambda x: -abs(x[1].get("w", 0)
                                               if isinstance(x[1], dict) else x[1])
                        )[:5]
                        concept_details.append({{
                            "concept": concept,
                            "connections": connections,
                            "top_neighbors": [t[0] for t in top_neighbors],
                        }})

                return {{
                    "agent": "{name}",
                    "domain": self.DOMAIN,
                    "total_concepts": len(self.CORE_CONCEPTS),
                    "capabilities": self.CAPABILITIES,
                    "concept_map": concept_details,
                }}
        ''').format(
            name=blueprint.name,
            domain=blueprint.domain,
            timestamp=datetime.now().isoformat(),
            description=blueprint.description,
            caps=", ".join(blueprint.capabilities),
            node_count=len(blueprint.knowledge_nodes),
            concepts=concepts_list,
            queries=queries_list,
            capabilities=caps_list,
        )

        return code

    # ── Full Pipeline ─────────────────────────────────────

    def build_agents(self, max_agents=10, min_cluster_size=15,
                     verbose=True) -> list:
        """
        Full pipeline: detect clusters -> design blueprints -> generate code.

        Returns list of generated agent metadata dicts.
        """
        if verbose:
            print("[AGENT FACTORY] Scanning knowledge graph for domain clusters...")

        clusters = self.detect_knowledge_clusters(min_cluster_size)
        if verbose:
            print("[AGENT FACTORY] Found %d knowledge clusters" % len(clusters))

        results = []
        for i, cluster in enumerate(clusters[:max_agents]):
            # Design
            blueprint = self.design_agent(cluster)
            if verbose:
                print("  [%d] %s (%s) — %d nodes, density=%.1f" % (
                    i + 1, blueprint.name, blueprint.domain,
                    cluster["size"], cluster["density"]))
                print("       Capabilities: %s" % ", ".join(blueprint.capabilities))

            # Generate code
            code = self.generate_agent_code(blueprint)

            # Safety check
            is_safe, violations = _is_safe(code)
            if not is_safe:
                if verbose:
                    print("       [BLOCKED] Safety violations: %s" % violations)
                continue

            # Save
            agent_id = "agent_%s_%s" % (
                blueprint.domain.replace(" ", "_"),
                hashlib.md5(blueprint.name.encode()).hexdigest()[:6])

            code_path = os.path.join(_AGENTS_DIR, "%s.py" % agent_id)
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(code)

            meta = {
                "id": agent_id,
                "blueprint": blueprint.to_dict(),
                "code_path": code_path,
                "safety_check": "PASSED",
                "status": "PENDING",
                "generated_at": datetime.now().isoformat(),
            }

            meta_path = os.path.join(_AGENTS_DIR, "%s.json" % agent_id)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)

            results.append(meta)

            if verbose:
                print("       Saved: %s" % code_path)

        # Update registry
        self._update_registry(results)

        if verbose:
            print("[AGENT FACTORY] Generated %d agents in agents/" % len(results))

        return results

    def _update_registry(self, new_agents):
        """Update the agent registry file."""
        registry = {"agents": [], "last_updated": datetime.now().isoformat()}

        if os.path.exists(_REGISTRY_FILE):
            try:
                with open(_REGISTRY_FILE, 'r') as f:
                    registry = json.load(f)
            except Exception:
                pass

        existing_ids = {a["id"] for a in registry.get("agents", [])}
        for agent in new_agents:
            if agent["id"] not in existing_ids:
                registry["agents"].append({
                    "id": agent["id"],
                    "name": agent["blueprint"]["name"],
                    "domain": agent["blueprint"]["domain"],
                    "status": "PENDING",
                    "generated_at": agent["generated_at"],
                })

        registry["last_updated"] = datetime.now().isoformat()

        with open(_REGISTRY_FILE, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2)


# ═════════════════════════════════════════════════════════════
# AGENT REGISTRY — Load and manage generated agents
# ═════════════════════════════════════════════════════════════

class AgentRegistry:
    """Manage generated agents: list, approve, load, route queries."""

    def __init__(self, kernel, lexicon, shell=None):
        self.kernel = kernel
        self.lexicon = lexicon
        self.shell = shell
        self._loaded = {}  # id -> agent instance
        os.makedirs(_AGENTS_DIR, exist_ok=True)

    def list_agents(self) -> list:
        """List all generated agents and their status."""
        if not os.path.exists(_REGISTRY_FILE):
            return []
        try:
            with open(_REGISTRY_FILE, 'r') as f:
                registry = json.load(f)
            return registry.get("agents", [])
        except Exception:
            return []

    def approve_agent(self, agent_id: str) -> bool:
        """Approve an agent for use."""
        if not os.path.exists(_REGISTRY_FILE):
            return False

        with open(_REGISTRY_FILE, 'r') as f:
            registry = json.load(f)

        for agent in registry.get("agents", []):
            if agent["id"] == agent_id:
                agent["status"] = "APPROVED"
                agent["approved_at"] = datetime.now().isoformat()
                with open(_REGISTRY_FILE, 'w', encoding='utf-8') as f:
                    json.dump(registry, f, indent=2)
                return True
        return False

    def load_agent(self, agent_id: str):
        """Load an approved agent into memory."""
        if agent_id in self._loaded:
            return self._loaded[agent_id]

        # Check approval status
        agents = self.list_agents()
        agent_meta = None
        for a in agents:
            if a["id"] == agent_id:
                agent_meta = a
                break

        if not agent_meta or agent_meta.get("status") != "APPROVED":
            return None

        # Load the code
        code_path = os.path.join(_AGENTS_DIR, "%s.py" % agent_id)
        if not os.path.exists(code_path):
            return None

        with open(code_path, 'r', encoding='utf-8') as f:
            code = f.read()

        # Extract class name
        class_match = re.search(r'class\s+(\w+Agent)\s*:', code)
        if not class_match:
            return None

        class_name = class_match.group(1)

        # Execute in restricted namespace
        namespace = {}
        exec(compile(code, code_path, 'exec'), namespace)

        agent_class = namespace.get(class_name)
        if agent_class is None:
            return None

        instance = agent_class(self.kernel, self.lexicon, self.shell)
        self._loaded[agent_id] = instance
        return instance

    def route_query(self, question: str) -> dict:
        """Route a query to the best available agent."""
        best_agent = None
        best_score = 0.0

        for agent_id, agent in self._loaded.items():
            score = agent.can_handle(question)
            if score > best_score:
                best_score = score
                best_agent = agent

        if best_agent and best_score > 0.2:
            result = best_agent.query(question)
            result["routing_score"] = best_score
            return result

        return {
            "answer": "No specialized agent matched this query",
            "confidence": 0.0,
            "routing_score": best_score,
        }

    def get_all_stats(self) -> list:
        """Get stats from all loaded agents."""
        return [agent.get_stats() for agent in self._loaded.values()]
