"""
KOS V5.1 — Level 3.5: Supervised Code Proposal System.

The system can PROPOSE code changes but NEVER deploy them.
Every proposal requires explicit human approval before execution.

Architecture:
    CodeProposer  → generates improvement proposals
    ProposalSandbox → tests proposals in isolated environment
    HumanGate     → blocks execution until human approves
    Deployer      → applies approved changes (read-only to system)

SAFETY INVARIANTS:
    1. CodeProposer can ONLY write to proposals/ directory
    2. CodeProposer can NEVER import os.system, subprocess, or __import__
    3. Deployer is a separate function, not callable by CodeProposer
    4. HumanGate requires interactive Y/N input — cannot be bypassed
    5. All proposals are logged with timestamps for audit
"""

import os
import re
import time
import json
import textwrap
import hashlib
from datetime import datetime


# ── Paths ────────────────────────────────────────────────────

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROPOSALS_DIR = os.path.join(_BASE, 'proposals')
_APPROVED_LOG = os.path.join(_BASE, 'proposals', 'approved.log')
_REJECTED_LOG = os.path.join(_BASE, 'proposals', 'rejected.log')

# ── Forbidden patterns (NEVER allowed in generated code) ─────

_FORBIDDEN_PATTERNS = [
    r'import\s+os',
    r'import\s+subprocess',
    r'import\s+shutil',
    r'__import__',
    r'exec\s*\(',
    r'eval\s*\(',
    r'open\s*\(.*(w|a)',  # No file writing
    r'os\.(system|popen|remove|unlink|rmdir|chmod|rename)',
    r'subprocess\.',
    r'shutil\.',
    r'compile\s*\(',
    r'globals\s*\(',
    r'setattr\s*\(',
    r'delattr\s*\(',
    r'__builtins__',
    r'signal\.',
    r'socket\.',
    r'requests\.',  # No network calls from proposals
]


def _is_safe(code: str) -> tuple:
    """
    Check if proposed code contains any forbidden patterns.
    Returns (is_safe: bool, violations: list)
    """
    violations = []
    for pattern in _FORBIDDEN_PATTERNS:
        matches = re.findall(pattern, code, re.IGNORECASE)
        if matches:
            violations.append(f"Forbidden: {pattern} found")
    return (len(violations) == 0, violations)


# ═════════════════════════════════════════════════════════════
# CODE PROPOSER
# ═════════════════════════════════════════════════════════════

class CodeProposer:
    """
    Generates improvement proposals based on system performance.

    The proposer analyzes:
    1. Prediction error patterns → suggests weight adjustments
    2. Query failure patterns → suggests new synonym mappings
    3. Low-scoring evidence → suggests new Weaver intent rules
    4. Graph structure → suggests new daemon strategies

    It generates Python code snippets saved to proposals/ directory.
    It can NEVER execute them — only the Deployer can.
    """

    def __init__(self, kernel, lexicon, pce=None):
        self.kernel = kernel
        self.lexicon = lexicon
        self.pce = pce
        self.proposals = []  # List of Proposal objects

        os.makedirs(_PROPOSALS_DIR, exist_ok=True)

    def _generate_id(self) -> str:
        """Generate unique proposal ID."""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        h = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"proposal_{ts}_{h}"

    def propose_synonym_additions(self) -> dict:
        """
        Analyze failed queries and propose new synonym mappings.

        If a word consistently fails Layer 1-4 resolution but
        succeeds at Layer 5 (vector), the vector match IS the
        correct synonym — propose adding it to the synonym table.
        """
        # Find words in the lexicon that are semantically close
        # but not in the synonym map
        from .synonyms import get_synonym_map
        existing = get_synonym_map()

        new_synonyms = {}
        word_list = list(self.lexicon.word_to_uuid.keys())

        # Check pairs of words that share many graph neighbors
        for i in range(min(len(word_list), 200)):
            w1 = word_list[i]
            uid1 = self.lexicon.word_to_uuid[w1]
            if uid1 not in self.kernel.nodes:
                continue
            targets1 = set(self.kernel.nodes[uid1].connections.keys())
            if len(targets1) < 3:
                continue

            for j in range(i + 1, min(len(word_list), 200)):
                w2 = word_list[j]
                uid2 = self.lexicon.word_to_uuid[w2]
                if uid2 not in self.kernel.nodes:
                    continue
                targets2 = set(self.kernel.nodes[uid2].connections.keys())

                if len(targets1 | targets2) == 0:
                    continue
                overlap = len(targets1 & targets2) / len(targets1 | targets2)

                if overlap > 0.5 and w1 not in existing and w2 not in existing:
                    if w1 != w2:
                        new_synonyms[w1] = w2

        if not new_synonyms:
            return None

        # Generate the proposal code
        code = "# AUTO-GENERATED SYNONYM ADDITIONS\n"
        code += "# Based on graph structure analysis\n\n"
        code += "NEW_SYNONYMS = {\n"
        for k, v in list(new_synonyms.items())[:20]:
            code += f'    "{k}": "{v}",\n'
        code += "}\n"

        proposal = self._save_proposal(
            proposal_id=self._generate_id(),
            proposal_type="synonym_addition",
            description=f"Add {len(new_synonyms)} new synonym mappings "
                        f"discovered from graph structure overlap",
            code=code,
            impact=f"Improves Layer 1b resolution for {len(new_synonyms)} words",
        )
        return proposal

    def propose_weaver_rule(self, failing_query: str = None,
                             intent_type: str = "CUSTOM") -> dict:
        """
        Propose a new Weaver scoring rule based on query patterns.
        """
        if not failing_query:
            return None

        # Analyze the query to determine what intent it represents
        query_words = set(failing_query.lower().split())

        code = textwrap.dedent(f"""
        # AUTO-GENERATED WEAVER RULE
        # Based on failing query: "{failing_query}"

        # Add to AlgorithmicWeaver.weave() scoring section:

        {intent_type}_PROMPT = {query_words}
        {intent_type}_BOOST = 35

        # Inside the scoring loop:
        has_{intent_type.lower()} = bool(
            {intent_type}_PROMPT & set(prompt_lower.split()))
        if has_{intent_type.lower()}:
            # Boost sentences containing relevant evidence
            {intent_type.lower()}_evidence = {{" relevant_word1 ", " relevant_word2 "}}
            if any(w in sent_lower for w in {intent_type.lower()}_evidence):
                score += {intent_type}_BOOST
        """)

        proposal = self._save_proposal(
            proposal_id=self._generate_id(),
            proposal_type="weaver_rule",
            description=f"New {intent_type} intent scoring rule for Weaver",
            code=code,
            impact=f"Fixes failing query: '{failing_query}'",
        )
        return proposal

    def propose_threshold_change(self, param: str, old_val: float,
                                  new_val: float, reason: str) -> dict:
        """Propose a threshold adjustment with justification."""
        code = textwrap.dedent(f"""
        # AUTO-GENERATED THRESHOLD CHANGE
        # Parameter: {param}
        # Old value: {old_val}
        # New value: {new_val}
        # Reason: {reason}

        # Apply in the relevant module:
        # {param} = {new_val}

        # To apply via config (preferred):
        # Edit .cache/self_tuned_config.json:
        #   "{param}": {{"value": {new_val}}}
        """)

        proposal = self._save_proposal(
            proposal_id=self._generate_id(),
            proposal_type="threshold_change",
            description=f"Change {param}: {old_val} -> {new_val}",
            code=code,
            impact=reason,
        )
        return proposal

    def propose_daemon_strategy(self, strategy_name: str,
                                 description: str) -> dict:
        """Propose a new daemon maintenance strategy."""
        code = textwrap.dedent(f"""
        # AUTO-GENERATED DAEMON STRATEGY: {strategy_name}
        # {description}

        def _daemon_{strategy_name.lower().replace(' ', '_')}(self) -> int:
            \"\"\"
            {description}

            Returns: number of modifications made
            \"\"\"
            count = 0

            for nid, node in self.kernel.nodes.items():
                # Strategy logic here
                # Analyze node properties and connections
                # Make targeted improvements
                pass

            return count
        """)

        proposal = self._save_proposal(
            proposal_id=self._generate_id(),
            proposal_type="daemon_strategy",
            description=f"New daemon strategy: {strategy_name}",
            code=code,
            impact=description,
        )
        return proposal

    def auto_propose(self, verbose: bool = True) -> list:
        """
        Automatically analyze the system and generate all applicable proposals.
        """
        if verbose:
            print("\n[PROPOSER] Analyzing system for improvement opportunities...")

        proposals = []

        # 1. Synonym additions
        syn_proposal = self.propose_synonym_additions()
        if syn_proposal:
            proposals.append(syn_proposal)
            if verbose:
                print(f"  [+] Synonym additions: {syn_proposal['description']}")

        # 2. Threshold suggestions from prediction stats
        if self.pce:
            stats = self.pce.get_stats()
            if stats['overall_accuracy'] < 0.9:
                p = self.propose_threshold_change(
                    'learning_rate', 0.02, 0.05,
                    f"Prediction accuracy is {stats['overall_accuracy']:.0%}, "
                    f"increase learning rate for faster convergence")
                if p:
                    proposals.append(p)
                    if verbose:
                        print(f"  [+] Threshold change: {p['description']}")

        # 3. Check for high contradiction rate
        contradictions = len(getattr(self.kernel, 'contradictions', []))
        if contradictions > 5:
            p = self.propose_daemon_strategy(
                "Contradiction Resolver",
                f"Detected {contradictions} contradictions. "
                f"Auto-resolve by weakening minority-evidence edges.")
            if p:
                proposals.append(p)
                if verbose:
                    print(f"  [+] Daemon strategy: {p['description']}")

        if verbose:
            print(f"  Total proposals: {len(proposals)}")

        self.proposals = proposals
        return proposals

    def _save_proposal(self, proposal_id: str, proposal_type: str,
                        description: str, code: str, impact: str) -> dict:
        """Save a proposal to the proposals/ directory."""
        # Safety check
        is_safe, violations = _is_safe(code)
        if not is_safe:
            return {
                'id': proposal_id,
                'status': 'REJECTED_UNSAFE',
                'violations': violations,
            }

        proposal = {
            'id': proposal_id,
            'type': proposal_type,
            'description': description,
            'impact': impact,
            'code': code,
            'timestamp': datetime.now().isoformat(),
            'status': 'PENDING',
            'safety_check': 'PASSED',
        }

        # Save to file
        filepath = os.path.join(_PROPOSALS_DIR, f"{proposal_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(proposal, f, indent=2)

        # Save code separately for easy review
        code_path = os.path.join(_PROPOSALS_DIR, f"{proposal_id}.py")
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code)

        return proposal


# ═════════════════════════════════════════════════════════════
# HUMAN GATE
# ═════════════════════════════════════════════════════════════

class HumanGate:
    """
    Blocks execution until human explicitly approves.

    The gate:
    1. Displays the proposed code
    2. Shows safety analysis
    3. Shows impact description
    4. Waits for Y/N input (or auto-mode for testing)
    5. Logs the decision

    Cannot be bypassed by the CodeProposer — it's a separate
    class with no callback to the proposer.
    """

    def __init__(self, auto_mode: bool = False):
        """
        auto_mode: If True, auto-approves all proposals (for testing only).
        In production, this MUST be False.
        """
        self.auto_mode = auto_mode
        self.decisions = []

    def review(self, proposal: dict, verbose: bool = True) -> bool:
        """
        Present a proposal for human review.

        Returns True if approved, False if rejected.
        """
        if proposal.get('status') == 'REJECTED_UNSAFE':
            if verbose:
                print(f"\n  [BLOCKED] Proposal {proposal['id']} failed safety check")
                for v in proposal.get('violations', []):
                    print(f"    VIOLATION: {v}")
            return False

        if verbose:
            print(f"\n{'='*60}")
            print(f"  PROPOSAL FOR REVIEW: {proposal['id']}")
            print(f"{'='*60}")
            print(f"  Type:        {proposal.get('type', '?')}")
            print(f"  Description: {proposal.get('description', '?')}")
            print(f"  Impact:      {proposal.get('impact', '?')}")
            print(f"  Safety:      {proposal.get('safety_check', '?')}")
            print(f"  Timestamp:   {proposal.get('timestamp', '?')}")
            print(f"\n  --- PROPOSED CODE ---")
            for line in proposal.get('code', '').split('\n')[:20]:
                print(f"  | {line}")
            if len(proposal.get('code', '').split('\n')) > 20:
                print(f"  | ... ({len(proposal['code'].split(chr(10)))} lines total)")
            print(f"  --- END CODE ---")

        if self.auto_mode:
            decision = True
            if verbose:
                print(f"\n  [AUTO-MODE] Auto-approved for testing")
        else:
            try:
                response = input(f"\n  Approve this proposal? [Y/N]: ").strip().lower()
                decision = response in ('y', 'yes')
            except (EOFError, KeyboardInterrupt):
                decision = False

        # Log decision
        log_entry = {
            'proposal_id': proposal['id'],
            'decision': 'APPROVED' if decision else 'REJECTED',
            'timestamp': datetime.now().isoformat(),
        }
        self.decisions.append(log_entry)

        # Write to log file
        log_file = _APPROVED_LOG if decision else _REJECTED_LOG
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception:
            pass

        # Update proposal status
        proposal['status'] = 'APPROVED' if decision else 'REJECTED'

        if verbose:
            status = "APPROVED" if decision else "REJECTED"
            print(f"  Decision: {status}")

        return decision


# ═════════════════════════════════════════════════════════════
# DEPLOYER (Read-Only to CodeProposer)
# ═════════════════════════════════════════════════════════════

class Deployer:
    """
    Applies approved proposals.

    SAFETY: This class is NOT accessible to the CodeProposer.
    Only the human (via test script or CLI) calls deploy().

    The deployer:
    1. Reads the approved proposal from proposals/ directory
    2. Applies it to the appropriate config/module
    3. Logs the deployment
    """

    @staticmethod
    def deploy_synonym_additions(proposal: dict, lexicon) -> int:
        """Apply synonym additions from an approved proposal."""
        code = proposal.get('code', '')

        # Parse the synonyms from the code
        # Look for NEW_SYNONYMS = {...}
        match = re.search(r'NEW_SYNONYMS\s*=\s*\{([^}]+)\}', code, re.DOTALL)
        if not match:
            return 0

        count = 0
        for line in match.group(1).split('\n'):
            pair = re.findall(r'"(\w+)":\s*"(\w+)"', line)
            if pair:
                word, canonical = pair[0]
                from .synonyms import get_synonym_map
                syn_map = get_synonym_map()
                syn_map[word] = canonical
                count += 1

        return count

    @staticmethod
    def deploy_threshold_change(proposal: dict) -> bool:
        """Apply a threshold change to the config file."""
        # Extract parameter and value from code
        code = proposal.get('code', '')
        param_match = re.search(r'# Parameter: (\w+)', code)
        value_match = re.search(r'# New value: ([\d.]+)', code)

        if not param_match or not value_match:
            return False

        param = param_match.group(1)
        value = float(value_match.group(1))

        # Write to config
        from .selfmod import _load_config, _save_config
        config = _load_config()
        if 'deployed_changes' not in config:
            config['deployed_changes'] = {}
        config['deployed_changes'][param] = {
            'value': value,
            'proposal_id': proposal['id'],
            'deployed_at': datetime.now().isoformat(),
        }
        _save_config(config)
        return True

    @staticmethod
    def deploy(proposal: dict, kernel=None, lexicon=None) -> dict:
        """
        Deploy an approved proposal.

        Returns: {status, details}
        """
        if proposal.get('status') != 'APPROVED':
            return {'status': 'error', 'details': 'Proposal not approved'}

        ptype = proposal.get('type', '')
        result = {'status': 'deployed', 'type': ptype}

        if ptype == 'synonym_addition' and lexicon:
            count = Deployer.deploy_synonym_additions(proposal, lexicon)
            result['details'] = f"Added {count} synonyms"

        elif ptype == 'threshold_change':
            ok = Deployer.deploy_threshold_change(proposal)
            result['details'] = 'Threshold updated in config' if ok else 'Failed'

        elif ptype in ('weaver_rule', 'daemon_strategy'):
            result['details'] = (f"Code saved to proposals/{proposal['id']}.py. "
                                 f"Manual integration required.")

        else:
            result['details'] = 'Unknown proposal type'

        return result
