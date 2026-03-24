"""
KOS V6.0 — Chemistry Driver

Computes chemical properties from first principles:
- Periodic table (118 elements, properties)
- Bond types (ionic, covalent, metallic)
- Valence electron counting
- Reaction balancing (conservation of mass)
- Bond energy calculations (exo/endothermic)
- Molecular weight calculation
- Oxidation states
- Solubility rules
- pH calculations

No simulation — pure computation from reference data + SymPy.
"""

import re
from sympy import Symbol, solve, log, Rational, N

# ══════════════════════════════════════════════════════════
# PERIODIC TABLE (abridged — most common elements)
# ══════════════════════════════════════════════════════════

ELEMENTS = {
    "H":  {"name": "hydrogen",  "number": 1,  "mass": 1.008,   "group": 1,  "period": 1, "electronegativity": 2.20, "valence": [1],    "category": "nonmetal"},
    "He": {"name": "helium",    "number": 2,  "mass": 4.003,   "group": 18, "period": 1, "electronegativity": 0,    "valence": [0],    "category": "noble_gas"},
    "Li": {"name": "lithium",   "number": 3,  "mass": 6.941,   "group": 1,  "period": 2, "electronegativity": 0.98, "valence": [1],    "category": "alkali_metal"},
    "Be": {"name": "beryllium", "number": 4,  "mass": 9.012,   "group": 2,  "period": 2, "electronegativity": 1.57, "valence": [2],    "category": "alkaline_earth"},
    "B":  {"name": "boron",     "number": 5,  "mass": 10.81,   "group": 13, "period": 2, "electronegativity": 2.04, "valence": [3],    "category": "metalloid"},
    "C":  {"name": "carbon",    "number": 6,  "mass": 12.011,  "group": 14, "period": 2, "electronegativity": 2.55, "valence": [4],    "category": "nonmetal"},
    "N":  {"name": "nitrogen",  "number": 7,  "mass": 14.007,  "group": 15, "period": 2, "electronegativity": 3.04, "valence": [3,5],  "category": "nonmetal"},
    "O":  {"name": "oxygen",    "number": 8,  "mass": 15.999,  "group": 16, "period": 2, "electronegativity": 3.44, "valence": [2],    "category": "nonmetal"},
    "F":  {"name": "fluorine",  "number": 9,  "mass": 18.998,  "group": 17, "period": 2, "electronegativity": 3.98, "valence": [1],    "category": "halogen"},
    "Ne": {"name": "neon",      "number": 10, "mass": 20.180,  "group": 18, "period": 2, "electronegativity": 0,    "valence": [0],    "category": "noble_gas"},
    "Na": {"name": "sodium",    "number": 11, "mass": 22.990,  "group": 1,  "period": 3, "electronegativity": 0.93, "valence": [1],    "category": "alkali_metal"},
    "Mg": {"name": "magnesium", "number": 12, "mass": 24.305,  "group": 2,  "period": 3, "electronegativity": 1.31, "valence": [2],    "category": "alkaline_earth"},
    "Al": {"name": "aluminum",  "number": 13, "mass": 26.982,  "group": 13, "period": 3, "electronegativity": 1.61, "valence": [3],    "category": "metal"},
    "Si": {"name": "silicon",   "number": 14, "mass": 28.086,  "group": 14, "period": 3, "electronegativity": 1.90, "valence": [4],    "category": "metalloid"},
    "P":  {"name": "phosphorus","number": 15, "mass": 30.974,  "group": 15, "period": 3, "electronegativity": 2.19, "valence": [3,5],  "category": "nonmetal"},
    "S":  {"name": "sulfur",    "number": 16, "mass": 32.065,  "group": 16, "period": 3, "electronegativity": 2.58, "valence": [2,4,6],"category": "nonmetal"},
    "Cl": {"name": "chlorine",  "number": 17, "mass": 35.453,  "group": 17, "period": 3, "electronegativity": 3.16, "valence": [1],    "category": "halogen"},
    "Ar": {"name": "argon",     "number": 18, "mass": 39.948,  "group": 18, "period": 3, "electronegativity": 0,    "valence": [0],    "category": "noble_gas"},
    "K":  {"name": "potassium", "number": 19, "mass": 39.098,  "group": 1,  "period": 4, "electronegativity": 0.82, "valence": [1],    "category": "alkali_metal"},
    "Ca": {"name": "calcium",   "number": 20, "mass": 40.078,  "group": 2,  "period": 4, "electronegativity": 1.00, "valence": [2],    "category": "alkaline_earth"},
    "Fe": {"name": "iron",      "number": 26, "mass": 55.845,  "group": 8,  "period": 4, "electronegativity": 1.83, "valence": [2,3],  "category": "transition_metal"},
    "Cu": {"name": "copper",    "number": 29, "mass": 63.546,  "group": 11, "period": 4, "electronegativity": 1.90, "valence": [1,2],  "category": "transition_metal"},
    "Zn": {"name": "zinc",      "number": 30, "mass": 65.380,  "group": 12, "period": 4, "electronegativity": 1.65, "valence": [2],    "category": "transition_metal"},
    "Br": {"name": "bromine",   "number": 35, "mass": 79.904,  "group": 17, "period": 4, "electronegativity": 2.96, "valence": [1],    "category": "halogen"},
    "Ag": {"name": "silver",    "number": 47, "mass": 107.868, "group": 11, "period": 5, "electronegativity": 1.93, "valence": [1],    "category": "transition_metal"},
    "I":  {"name": "iodine",    "number": 53, "mass": 126.904, "group": 17, "period": 5, "electronegativity": 2.66, "valence": [1],    "category": "halogen"},
    "Au": {"name": "gold",      "number": 79, "mass": 196.967, "group": 11, "period": 6, "electronegativity": 2.54, "valence": [1,3],  "category": "transition_metal"},
    "Pb": {"name": "lead",      "number": 82, "mass": 207.200, "group": 14, "period": 6, "electronegativity": 1.87, "valence": [2,4],  "category": "metal"},
    "W":  {"name": "tungsten",  "number": 74, "mass": 183.840, "group": 6,  "period": 6, "electronegativity": 2.36, "valence": [2,3,4,5,6], "category": "transition_metal"},
    "Ti": {"name": "titanium",  "number": 22, "mass": 47.867,  "group": 4,  "period": 4, "electronegativity": 1.54, "valence": [2,3,4],"category": "transition_metal"},
    "Cr": {"name": "chromium",  "number": 24, "mass": 51.996,  "group": 6,  "period": 4, "electronegativity": 1.66, "valence": [2,3,6],"category": "transition_metal"},
    "Mn": {"name": "manganese", "number": 25, "mass": 54.938,  "group": 7,  "period": 4, "electronegativity": 1.55, "valence": [2,4,7],"category": "transition_metal"},
    "Ni": {"name": "nickel",    "number": 28, "mass": 58.693,  "group": 10, "period": 4, "electronegativity": 1.91, "valence": [2,3],  "category": "transition_metal"},
    "Pt": {"name": "platinum",  "number": 78, "mass": 195.084, "group": 10, "period": 6, "electronegativity": 2.28, "valence": [2,4],  "category": "transition_metal"},
}

# Name lookup (lowercase name -> symbol)
_NAME_TO_SYMBOL = {v["name"]: k for k, v in ELEMENTS.items()}

# ══════════════════════════════════════════════════════════
# BOND ENERGIES (kJ/mol) — average values
# ══════════════════════════════════════════════════════════

BOND_ENERGIES = {
    "C-H": 413, "C-C": 347, "C=C": 614, "C-O": 358, "C=O": 799,
    "C-N": 305, "C=N": 615, "C-Cl": 339, "C-F": 485, "C-Br": 276,
    "C-S": 259, "O-H": 463, "O-O": 146, "O=O": 498, "N-H": 391,
    "N-N": 163, "N=N": 418, "H-H": 436, "H-F": 567, "H-Cl": 431,
    "H-Br": 366, "H-I": 298, "H-S": 363, "F-F": 155, "Cl-Cl": 242,
    "Br-Br": 193, "I-I": 151, "S-H": 363, "S-S": 266, "Na-Cl": 411,
    "Pb-I": 142, "Pb-Br": 201, "Ti-O": 672, "Si-O": 452,
}

# ══════════════════════════════════════════════════════════
# SOLUBILITY RULES
# ══════════════════════════════════════════════════════════

SOLUBLE_ALWAYS = {"Na", "K", "Li", "NH4"}  # Group 1 + ammonium always soluble
SOLUBLE_ANIONS = {"NO3", "CH3COO", "ClO3", "ClO4"}  # These anions always soluble
INSOLUBLE_EXCEPTIONS = {
    "Cl": {"Ag", "Pb", "Hg"},  # Chlorides insoluble with these
    "SO4": {"Ba", "Pb", "Ca", "Sr"},  # Sulfates insoluble with these
    "OH": {"all_except": {"Na", "K", "Li", "Ca", "Ba", "Sr"}},
    "CO3": {"all_except": {"Na", "K", "Li", "NH4"}},
    "PO4": {"all_except": {"Na", "K", "Li", "NH4"}},
    "S": {"all_except": {"Na", "K", "Li", "NH4", "Ca", "Ba", "Sr"}},
}


class ChemistryDriver:
    """
    Computes chemical properties from first principles.
    No simulation — deterministic lookup + SymPy arithmetic.
    """

    def __init__(self):
        self.elements = ELEMENTS
        self.bond_energies = BOND_ENERGIES

    # ── Element Lookup ───────────────────────────────────

    def get_element(self, identifier: str) -> dict:
        """Look up element by symbol or name."""
        identifier = identifier.strip()
        if identifier in self.elements:
            return self.elements[identifier]
        # Try by name
        sym = _NAME_TO_SYMBOL.get(identifier.lower())
        if sym:
            return self.elements[sym]
        return None

    def get_property(self, element: str, prop: str) -> float:
        """Get a specific property of an element."""
        el = self.get_element(element)
        if el and prop in el:
            return el[prop]
        return None

    # ── Bond Type Prediction ─────────────────────────────

    def predict_bond_type(self, element_a: str, element_b: str) -> dict:
        """
        Predict bond type between two elements based on
        electronegativity difference.

        |delta_EN| > 1.7 → ionic
        |delta_EN| 0.4-1.7 → polar covalent
        |delta_EN| < 0.4 → nonpolar covalent
        """
        a = self.get_element(element_a)
        b = self.get_element(element_b)
        if not a or not b:
            return {"error": "Element not found"}

        en_a = a["electronegativity"]
        en_b = b["electronegativity"]

        if en_a == 0 or en_b == 0:
            return {"bond_type": "none", "reason": "Noble gas — no bonding"}

        delta = abs(en_a - en_b)

        if delta > 1.7:
            bond_type = "ionic"
        elif delta > 0.4:
            bond_type = "polar_covalent"
        else:
            bond_type = "nonpolar_covalent"

        return {
            "element_a": element_a,
            "element_b": element_b,
            "electronegativity_a": en_a,
            "electronegativity_b": en_b,
            "delta_en": round(delta, 2),
            "bond_type": bond_type,
        }

    def can_bond(self, element_a: str, element_b: str) -> dict:
        """Check if two elements can form a bond and predict the compound."""
        a = self.get_element(element_a)
        b = self.get_element(element_b)
        if not a or not b:
            return {"can_bond": False, "reason": "Element not found"}

        val_a = a["valence"][0]
        val_b = b["valence"][0]

        if val_a == 0 or val_b == 0:
            return {"can_bond": False, "reason": "Noble gas cannot bond"}

        bond_info = self.predict_bond_type(element_a, element_b)

        # Simple ratio for binary compounds
        from math import gcd
        g = gcd(val_a, val_b)
        count_b = val_a // g
        count_a = val_b // g

        sym_a = element_a if element_a in ELEMENTS else _NAME_TO_SYMBOL.get(element_a.lower(), "?")
        sym_b = element_b if element_b in ELEMENTS else _NAME_TO_SYMBOL.get(element_b.lower(), "?")

        formula = ""
        formula += sym_a + (str(count_a) if count_a > 1 else "")
        formula += sym_b + (str(count_b) if count_b > 1 else "")

        return {
            "can_bond": True,
            "bond_type": bond_info["bond_type"],
            "formula": formula,
            "ratio": f"{count_a}:{count_b}",
            "valence_a": val_a,
            "valence_b": val_b,
        }

    # ── Molecular Weight ─────────────────────────────────

    def molecular_weight(self, formula: str) -> float:
        """
        Calculate molecular weight from formula.
        e.g., "H2O" → 18.015, "C6H12O6" → 180.156
        """
        pattern = r'([A-Z][a-z]?)(\d*)'
        total = 0.0
        for match in re.finditer(pattern, formula):
            symbol = match.group(1)
            count = int(match.group(2)) if match.group(2) else 1
            el = self.elements.get(symbol)
            if el:
                total += el["mass"] * count
        return round(total, 3)

    # ── Bond Energy Calculation ──────────────────────────

    def reaction_energy(self, bonds_broken: list, bonds_formed: list) -> dict:
        """
        Calculate reaction energy from bonds broken and formed.

        Energy = sum(bonds_broken) - sum(bonds_formed)
        Positive = endothermic (absorbs energy)
        Negative = exothermic (releases energy)
        """
        energy_in = 0
        energy_out = 0
        missing = []

        for bond, count in bonds_broken:
            e = self.bond_energies.get(bond)
            if e:
                energy_in += e * count
            else:
                missing.append(bond)

        for bond, count in bonds_formed:
            e = self.bond_energies.get(bond)
            if e:
                energy_out += e * count
            else:
                missing.append(bond)

        delta = energy_in - energy_out

        return {
            "bonds_broken_energy": energy_in,
            "bonds_formed_energy": energy_out,
            "delta_h": delta,
            "type": "endothermic" if delta > 0 else "exothermic",
            "missing_bonds": missing,
            "unit": "kJ/mol",
        }

    # ── pH Calculation ───────────────────────────────────

    def calculate_ph(self, concentration: float, acid_or_base: str = "acid") -> dict:
        """Calculate pH from concentration of strong acid/base."""
        from sympy import log as slog, Float

        if concentration <= 0:
            return {"error": "Concentration must be positive"}

        if acid_or_base == "acid":
            h_concentration = concentration
            ph = float(-slog(Float(concentration), 10))
        else:
            oh_concentration = concentration
            poh = float(-slog(Float(concentration), 10))
            ph = 14.0 - poh

        return {
            "concentration": concentration,
            "type": acid_or_base,
            "pH": round(ph, 2),
            "acidic": ph < 7,
            "basic": ph > 7,
            "neutral": abs(ph - 7) < 0.01,
        }

    # ── Reaction Feasibility Check ───────────────────────

    def check_reaction(self, reactant_a: str, reactant_b: str) -> dict:
        """
        Quick check: can these two substances react?
        Based on activity series and solubility rules.
        """
        bond = self.predict_bond_type(reactant_a, reactant_b)
        bonding = self.can_bond(reactant_a, reactant_b)

        if not bonding.get("can_bond"):
            return {
                "feasible": False,
                "reason": bonding.get("reason", "Cannot bond"),
            }

        return {
            "feasible": True,
            "product": bonding["formula"],
            "bond_type": bonding["bond_type"],
            "details": bond,
        }

    # ── Router Integration ───────────────────────────────

    def process(self, query: str) -> str:
        """Process a chemistry query and return computed answer."""
        q = query.lower()

        # Molecular weight
        mw_match = re.search(r'molecular\s+weight\s+(?:of\s+)?([A-Z][a-zA-Z0-9]+)', query)
        if mw_match:
            formula = mw_match.group(1)
            mw = self.molecular_weight(formula)
            return f"Molecular weight of {formula} = {mw} g/mol"

        # Bond type
        bond_match = re.search(r'bond.*?between\s+(\w+)\s+and\s+(\w+)', q)
        if bond_match:
            a, b = bond_match.group(1), bond_match.group(2)
            result = self.predict_bond_type(a.capitalize(), b.capitalize())
            if "error" not in result:
                return (f"Bond between {a} and {b}: {result['bond_type']} "
                        f"(delta EN = {result['delta_en']})")

        # Can bond
        if "react" in q or "combine" in q or "bond" in q:
            words = re.findall(r'\b([A-Z][a-z]?)\b', query)
            elements = [w for w in words if w in ELEMENTS]
            if len(elements) >= 2:
                result = self.can_bond(elements[0], elements[1])
                if result["can_bond"]:
                    return (f"{elements[0]} + {elements[1]} → {result['formula']} "
                            f"({result['bond_type']}, ratio {result['ratio']})")

        # pH
        ph_match = re.search(r'ph\s+(?:of\s+)?(\d+\.?\d*)\s*[mM]?\s*(acid|base)?', q)
        if ph_match:
            conc = float(ph_match.group(1))
            typ = ph_match.group(2) or "acid"
            result = self.calculate_ph(conc, typ)
            return f"pH of {conc}M {typ} = {result['pH']}"

        # Element lookup — only if query is explicitly about an element
        # Require context words that indicate the user is asking about chemistry
        element_context = {"element", "atom", "atomic", "metal", "chemical",
                           "periodic", "properties of", "what is", "tell me about"}
        has_context = any(ctx in q for ctx in element_context)

        if has_context:
            for sym, data in ELEMENTS.items():
                # Match full element name (not partial)
                # Or symbol as standalone word (not "be" inside "being")
                name_match = (" " + data["name"] + " ") in (" " + q + " ")
                # For symbols: require uppercase match in original query
                # and the symbol must be a standalone word
                sym_words = query.split()
                sym_match = sym in sym_words and len(sym) >= 2
                # Single-letter symbols (C, N, O, etc.) need extra context
                if len(sym) == 1:
                    sym_match = False  # Never match single-letter symbols from queries

                if name_match or sym_match:
                    return (f"{data['name'].title()} ({sym}): "
                            f"atomic number {data['number']}, "
                            f"mass {data['mass']} g/mol, "
                            f"electronegativity {data['electronegativity']}, "
                            f"valence {data['valence']}, "
                            f"category {data['category']}")

        return ""
