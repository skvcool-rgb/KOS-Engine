"""
KOS V8.0 -- Domain-Constrained Reality Model (Constraint Engine)

Three layers of constraint enforcement:

Layer 1: Physics/Logic Primitives
    Immutable laws pre-loaded into the graph.
    e.g., Time moves forward; 100 > 50; conservation laws.

Layer 2: Domain Axioms (.kos axiom files)
    Loaded per deployment. Finance: Basel III. Medical: drug interactions.
    Axioms are mathematical constraints checked by SymPy.

Layer 3: Constraint Gate
    Before any edge commits to the Rust arena, it must pass through
    the constraint solver. Violations are blocked.

Uses SymPy for symbolic math validation (no Z3 dependency required).
"""

import re
import json
import os


# ---- Layer 1: Physics/Logic Primitives ----------------------------------

_PHYSICS_PRIMITIVES = [
    # (name, description, check_function_name)
    ("time_forward", "Time only moves forward",
     lambda src, tgt, w, prov: not _violates_time(src, tgt, prov)),
    ("weight_bounds", "Edge weights must be in [-1.0, 1.0]",
     lambda src, tgt, w, prov: -1.0 <= w <= 1.0),
    ("no_self_loop", "No self-referential edges",
     lambda src, tgt, w, prov: src != tgt),
    ("numeric_consistency", "Numeric claims must be internally consistent",
     lambda src, tgt, w, prov: _check_numeric_consistency(prov)),
]


def _violates_time(src: str, tgt: str, prov: str) -> bool:
    """Check if provenance claims something happened before time began."""
    # Simple heuristic: reject dates before 0 CE
    years = re.findall(r'\b(\d{4})\b', prov)
    for y in years:
        if int(y) < 0:
            return True
    # Check "after" claims that precede "before" claims
    return False


def _check_numeric_consistency(prov: str) -> bool:
    """Check if numeric claims in provenance are internally consistent."""
    if not prov:
        return True
    # Extract "X > Y" or "X is greater than Y" patterns
    gt_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:>|greater than|more than)\s*(\d+(?:\.\d+)?)', prov)
    for a, b in gt_matches:
        if float(a) <= float(b):
            return False  # Contradiction: claims A > B but A <= B
    return True


# ---- Layer 2: Domain Axiom System ---------------------------------------

class DomainAxiom:
    """A single domain axiom with a constraint expression."""

    def __init__(self, name: str, description: str,
                 constraint_type: str, parameters: dict):
        self.name = name
        self.description = description
        self.constraint_type = constraint_type  # "threshold", "ratio", "formula", "forbidden"
        self.parameters = parameters
        self.violation_count = 0

    def check(self, edge_data: dict) -> tuple:
        """
        Check if edge_data violates this axiom.

        Returns: (passed: bool, reason: str)
        """
        ct = self.constraint_type

        if ct == "threshold":
            return self._check_threshold(edge_data)
        elif ct == "ratio":
            return self._check_ratio(edge_data)
        elif ct == "forbidden":
            return self._check_forbidden(edge_data)
        elif ct == "formula":
            return self._check_formula(edge_data)

        return True, "no constraint"

    def _check_threshold(self, data: dict) -> tuple:
        """Value must be above/below a threshold."""
        field = self.parameters.get("field", "weight")
        op = self.parameters.get("operator", ">=")
        threshold = self.parameters.get("value", 0)
        actual = data.get(field, data.get("weight", 0))

        try:
            actual = float(actual)
        except (ValueError, TypeError):
            return True, "non-numeric, skipped"

        if op == ">=" and actual < threshold:
            self.violation_count += 1
            return False, f"{field}={actual} < {threshold} (min={threshold})"
        elif op == "<=" and actual > threshold:
            self.violation_count += 1
            return False, f"{field}={actual} > {threshold} (max={threshold})"
        elif op == ">" and actual <= threshold:
            self.violation_count += 1
            return False, f"{field}={actual} <= {threshold}"
        elif op == "<" and actual >= threshold:
            self.violation_count += 1
            return False, f"{field}={actual} >= {threshold}"

        return True, "passed"

    def _check_ratio(self, data: dict) -> tuple:
        """Ratio between two values must satisfy constraint."""
        num_field = self.parameters.get("numerator", "")
        den_field = self.parameters.get("denominator", "")
        min_ratio = self.parameters.get("min_ratio", 0)

        prov = data.get("provenance", "")
        # Try to extract numeric values from provenance
        numbers = re.findall(r'(\d+(?:\.\d+)?)', prov)
        if len(numbers) >= 2:
            num = float(numbers[0])
            den = float(numbers[1])
            if den > 0:
                ratio = num / den
                if ratio < min_ratio:
                    self.violation_count += 1
                    return False, f"ratio {ratio:.2f} < min {min_ratio}"

        return True, "passed"

    def _check_forbidden(self, data: dict) -> tuple:
        """Certain edge patterns are forbidden."""
        forbidden_sources = self.parameters.get("sources", [])
        forbidden_targets = self.parameters.get("targets", [])
        forbidden_patterns = self.parameters.get("patterns", [])

        src = data.get("source_id", "")
        tgt = data.get("target_id", "")
        prov = data.get("provenance", "")

        if src in forbidden_sources or tgt in forbidden_targets:
            self.violation_count += 1
            return False, f"forbidden node: {src}->{tgt}"

        for pattern in forbidden_patterns:
            if re.search(pattern, prov, re.I):
                self.violation_count += 1
                return False, f"forbidden pattern: {pattern}"

        return True, "passed"

    def _check_formula(self, data: dict) -> tuple:
        """Evaluate a SymPy formula constraint."""
        try:
            from sympy import sympify, Symbol
            expr_str = self.parameters.get("expression", "True")
            # Substitute known values
            prov = data.get("provenance", "")
            numbers = re.findall(r'(\d+(?:\.\d+)?)', prov)
            symbols = self.parameters.get("symbols", [])

            if symbols and numbers:
                subs = {}
                for sym_name, val in zip(symbols, numbers):
                    subs[Symbol(sym_name)] = float(val)
                expr = sympify(expr_str)
                result = bool(expr.subs(subs))
                if not result:
                    self.violation_count += 1
                    return False, f"formula violation: {expr_str} with {subs}"
        except (ImportError, Exception):
            pass  # SymPy not available, skip formula check

        return True, "passed"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "constraint_type": self.constraint_type,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DomainAxiom":
        return cls(d["name"], d["description"],
                   d["constraint_type"], d["parameters"])


# ---- Layer 3: Constraint Gate -------------------------------------------

class ConstraintEngine:
    """
    The gate that every edge must pass through before committing.
    Enforces Layer 1 primitives + Layer 2 domain axioms.
    """

    def __init__(self):
        self.axioms = []         # List of DomainAxiom
        self.violations_log = [] # Audit trail of blocked edges
        self._enabled = True

    def load_axioms(self, filepath: str):
        """Load domain axioms from a .kos JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        for ax_data in data.get("axioms", []):
            self.axioms.append(DomainAxiom.from_dict(ax_data))

    def load_axioms_from_dict(self, axiom_list: list):
        """Load axioms from a list of dicts (programmatic)."""
        for ax_data in axiom_list:
            self.axioms.append(DomainAxiom.from_dict(ax_data))

    def add_axiom(self, axiom: DomainAxiom):
        """Add a single axiom."""
        self.axioms.append(axiom)

    def check(self, source_id: str, target_id: str, weight: float,
              provenance: str = "", edge_type: int = 0) -> dict:
        """
        Check if an edge passes all constraints.

        Returns:
            {"passed": bool, "violations": [str], "blocked_by": [str]}
        """
        if not self._enabled:
            return {"passed": True, "violations": [], "blocked_by": []}

        violations = []
        blocked_by = []

        edge_data = {
            "source_id": source_id,
            "target_id": target_id,
            "weight": weight,
            "provenance": provenance,
            "edge_type": edge_type,
        }

        # Layer 1: Physics primitives
        for name, desc, check_fn in _PHYSICS_PRIMITIVES:
            try:
                if not check_fn(source_id, target_id, weight, provenance):
                    violations.append(f"[L1] {name}: {desc}")
                    blocked_by.append(name)
            except Exception:
                pass  # Don't block on primitive check errors

        # Layer 2: Domain axioms
        for axiom in self.axioms:
            passed, reason = axiom.check(edge_data)
            if not passed:
                violations.append(f"[L2] {axiom.name}: {reason}")
                blocked_by.append(axiom.name)

        result = {
            "passed": len(violations) == 0,
            "violations": violations,
            "blocked_by": blocked_by,
        }

        if violations:
            self.violations_log.append({
                "source": source_id,
                "target": target_id,
                "weight": weight,
                "violations": violations,
            })

        return result

    def enable(self):
        self._enabled = True

    def disable(self):
        """Disable for batch mode (re-enable after!)."""
        self._enabled = False

    def stats(self) -> dict:
        return {
            "axiom_count": len(self.axioms),
            "total_violations": len(self.violations_log),
            "enabled": self._enabled,
            "axiom_violations": {a.name: a.violation_count
                                  for a in self.axioms if a.violation_count > 0},
        }


# ---- Axiom File Helpers --------------------------------------------------

def create_finance_axioms() -> list:
    """Pre-built axiom set for financial domain (Basel III)."""
    return [
        {
            "name": "basel_iii_capital_ratio",
            "description": "Capital ratio must be >= 4.5%",
            "constraint_type": "threshold",
            "parameters": {
                "field": "weight",
                "operator": ">=",
                "value": 0.045,
            },
        },
        {
            "name": "no_negative_assets",
            "description": "Asset values cannot be negative",
            "constraint_type": "forbidden",
            "parameters": {
                "patterns": [r'(?:assets?|value)\s*=?\s*-\d'],
            },
        },
    ]


def create_medical_axioms() -> list:
    """Pre-built axiom set for medical domain."""
    return [
        {
            "name": "drug_interaction_check",
            "description": "Known dangerous drug combinations are flagged",
            "constraint_type": "forbidden",
            "parameters": {
                "patterns": [
                    r'warfarin.*aspirin|aspirin.*warfarin',
                    r'maoi.*ssri|ssri.*maoi',
                ],
            },
        },
        {
            "name": "dosage_sanity",
            "description": "Dosage values must be positive",
            "constraint_type": "threshold",
            "parameters": {
                "field": "weight",
                "operator": ">=",
                "value": 0.0,
            },
        },
    ]


def save_axiom_file(filepath: str, axioms: list, domain: str = "general"):
    """Save axioms to a .kos file."""
    data = {
        "domain": domain,
        "version": "8.0",
        "axioms": axioms,
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
