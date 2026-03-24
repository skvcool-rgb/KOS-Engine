"""
KOS V6.0 — Biology Driver

Computes biology from first principles:
- Amino acids (20 standard, properties)
- Codon table (DNA codon -> amino acid, all 64)
- Enzyme kinetics (Michaelis-Menten, inhibition)
- Pharmacology (half-life, therapeutic window, drug interaction, dosage)
- Genetics (translation, Hardy-Weinberg, mutation rate)
- Cell biology (Nernst potential, osmotic pressure, ATP yield)
- Ecology (logistic growth, SIR model, Lotka-Volterra)

No simulation — pure computation from reference data + SymPy.
"""

import re
from sympy import (
    Symbol, symbols, solve, sqrt, pi, Rational, N,
    log, exp, Float, ln
)

# ══════════════════════════════════════════════════════════
# BIOLOGICAL CONSTANTS
# ══════════════════════════════════════════════════════════

CONSTANTS = {
    "R": {"value": 8.314, "unit": "J/(mol*K)", "name": "gas constant"},
    "F": {"value": 96485, "unit": "C/mol", "name": "Faraday constant"},
    "body_temp": {"value": 310, "unit": "K", "name": "human body temperature (37C)"},
    "k_B": {"value": 1.381e-23, "unit": "J/K", "name": "Boltzmann constant"},
    "N_A": {"value": 6.022e23, "unit": "1/mol", "name": "Avogadro number"},
    "atm": {"value": 101325, "unit": "Pa", "name": "standard atmosphere"},
    "R_L_atm": {"value": 0.08206, "unit": "L*atm/(mol*K)", "name": "gas constant (L*atm)"},
}

# ══════════════════════════════════════════════════════════
# AMINO ACIDS (20 standard)
# ══════════════════════════════════════════════════════════

AMINO_ACIDS = {
    "Ala": {"name": "alanine",        "single_letter": "A", "molecular_weight": 89.09,  "charge_at_pH7": 0,  "hydrophobicity": 1.8},
    "Arg": {"name": "arginine",       "single_letter": "R", "molecular_weight": 174.20, "charge_at_pH7": 1,  "hydrophobicity": -4.5},
    "Asn": {"name": "asparagine",     "single_letter": "N", "molecular_weight": 132.12, "charge_at_pH7": 0,  "hydrophobicity": -3.5},
    "Asp": {"name": "aspartic acid",  "single_letter": "D", "molecular_weight": 133.10, "charge_at_pH7": -1, "hydrophobicity": -3.5},
    "Cys": {"name": "cysteine",       "single_letter": "C", "molecular_weight": 121.16, "charge_at_pH7": 0,  "hydrophobicity": 2.5},
    "Gln": {"name": "glutamine",      "single_letter": "Q", "molecular_weight": 146.15, "charge_at_pH7": 0,  "hydrophobicity": -3.5},
    "Glu": {"name": "glutamic acid",  "single_letter": "E", "molecular_weight": 147.13, "charge_at_pH7": -1, "hydrophobicity": -3.5},
    "Gly": {"name": "glycine",        "single_letter": "G", "molecular_weight": 75.03,  "charge_at_pH7": 0,  "hydrophobicity": -0.4},
    "His": {"name": "histidine",      "single_letter": "H", "molecular_weight": 155.16, "charge_at_pH7": 0,  "hydrophobicity": -3.2},
    "Ile": {"name": "isoleucine",     "single_letter": "I", "molecular_weight": 131.17, "charge_at_pH7": 0,  "hydrophobicity": 4.5},
    "Leu": {"name": "leucine",        "single_letter": "L", "molecular_weight": 131.17, "charge_at_pH7": 0,  "hydrophobicity": 3.8},
    "Lys": {"name": "lysine",         "single_letter": "K", "molecular_weight": 146.19, "charge_at_pH7": 1,  "hydrophobicity": -3.9},
    "Met": {"name": "methionine",     "single_letter": "M", "molecular_weight": 149.21, "charge_at_pH7": 0,  "hydrophobicity": 1.9},
    "Phe": {"name": "phenylalanine",  "single_letter": "F", "molecular_weight": 165.19, "charge_at_pH7": 0,  "hydrophobicity": 2.8},
    "Pro": {"name": "proline",        "single_letter": "P", "molecular_weight": 115.13, "charge_at_pH7": 0,  "hydrophobicity": -1.6},
    "Ser": {"name": "serine",         "single_letter": "S", "molecular_weight": 105.09, "charge_at_pH7": 0,  "hydrophobicity": -0.8},
    "Thr": {"name": "threonine",      "single_letter": "T", "molecular_weight": 119.12, "charge_at_pH7": 0,  "hydrophobicity": -0.7},
    "Trp": {"name": "tryptophan",     "single_letter": "W", "molecular_weight": 204.23, "charge_at_pH7": 0,  "hydrophobicity": -0.9},
    "Tyr": {"name": "tyrosine",       "single_letter": "Y", "molecular_weight": 181.19, "charge_at_pH7": 0,  "hydrophobicity": -1.3},
    "Val": {"name": "valine",         "single_letter": "V", "molecular_weight": 117.15, "charge_at_pH7": 0,  "hydrophobicity": 4.2},
}

# Name / single-letter lookups
_AA_NAME_TO_CODE = {v["name"]: k for k, v in AMINO_ACIDS.items()}
_AA_SINGLE_TO_CODE = {v["single_letter"]: k for k, v in AMINO_ACIDS.items()}

# ══════════════════════════════════════════════════════════
# CODON TABLE (DNA codon -> amino acid, all 64 codons)
# ══════════════════════════════════════════════════════════

CODON_TABLE = {
    # Phenylalanine
    "TTT": "Phe", "TTC": "Phe",
    # Leucine
    "TTA": "Leu", "TTG": "Leu", "CTT": "Leu", "CTC": "Leu", "CTA": "Leu", "CTG": "Leu",
    # Isoleucine
    "ATT": "Ile", "ATC": "Ile", "ATA": "Ile",
    # Methionine (start)
    "ATG": "Met",
    # Valine
    "GTT": "Val", "GTC": "Val", "GTA": "Val", "GTG": "Val",
    # Serine
    "TCT": "Ser", "TCC": "Ser", "TCA": "Ser", "TCG": "Ser", "AGT": "Ser", "AGC": "Ser",
    # Proline
    "CCT": "Pro", "CCC": "Pro", "CCA": "Pro", "CCG": "Pro",
    # Threonine
    "ACT": "Thr", "ACC": "Thr", "ACA": "Thr", "ACG": "Thr",
    # Alanine
    "GCT": "Ala", "GCC": "Ala", "GCA": "Ala", "GCG": "Ala",
    # Tyrosine
    "TAT": "Tyr", "TAC": "Tyr",
    # Stop codons
    "TAA": "Stop", "TAG": "Stop", "TGA": "Stop",
    # Histidine
    "CAT": "His", "CAC": "His",
    # Glutamine
    "CAA": "Gln", "CAG": "Gln",
    # Asparagine
    "AAT": "Asn", "AAC": "Asn",
    # Lysine
    "AAA": "Lys", "AAG": "Lys",
    # Aspartic acid
    "GAT": "Asp", "GAC": "Asp",
    # Glutamic acid
    "GAA": "Glu", "GAG": "Glu",
    # Cysteine
    "TGT": "Cys", "TGC": "Cys",
    # Tryptophan
    "TGG": "Trp",
    # Arginine
    "CGT": "Arg", "CGC": "Arg", "CGA": "Arg", "CGG": "Arg", "AGA": "Arg", "AGG": "Arg",
    # Glycine
    "GGT": "Gly", "GGC": "Gly", "GGA": "Gly", "GGG": "Gly",
}


class BiologyDriver:
    """
    Computes biological properties from first principles.
    No simulation — deterministic lookup + SymPy arithmetic.
    """

    def __init__(self):
        self.amino_acids = AMINO_ACIDS
        self.codon_table = CODON_TABLE
        self.constants = CONSTANTS

    # ══════════════════════════════════════════════════════
    # AMINO ACID LOOKUP
    # ══════════════════════════════════════════════════════

    def get_amino_acid(self, identifier: str) -> dict:
        """
        Look up amino acid by three-letter code, single letter, or name.

        Args:
            identifier: Three-letter code (e.g. 'Ala'), single letter ('A'),
                        or full name ('alanine').

        Returns:
            Dict with amino acid properties, or None if not found.
        """
        identifier = identifier.strip()
        # Three-letter code
        if identifier in self.amino_acids:
            return {"code": identifier, **self.amino_acids[identifier]}
        # Capitalize first letter
        cap = identifier.capitalize()
        if cap in self.amino_acids:
            return {"code": cap, **self.amino_acids[cap]}
        # Single letter
        if len(identifier) == 1:
            code = _AA_SINGLE_TO_CODE.get(identifier.upper())
            if code:
                return {"code": code, **self.amino_acids[code]}
        # Full name
        code = _AA_NAME_TO_CODE.get(identifier.lower())
        if code:
            return {"code": code, **self.amino_acids[code]}
        return None

    # ══════════════════════════════════════════════════════
    # ENZYME KINETICS
    # ══════════════════════════════════════════════════════

    def michaelis_menten(self, vmax: float, km: float, substrate_conc: float) -> dict:
        """
        Michaelis-Menten enzyme kinetics.

        v = Vmax * [S] / (Km + [S])

        Args:
            vmax: Maximum reaction rate (mol/s or arbitrary units).
            km: Michaelis constant (same concentration units as substrate).
            substrate_conc: Substrate concentration [S].

        Returns:
            Dict with reaction rate, fraction of Vmax, and saturation.
        """
        if km + substrate_conc == 0:
            return {"error": "Km + [S] cannot be zero"}

        v = vmax * substrate_conc / (km + substrate_conc)
        fraction = substrate_conc / (km + substrate_conc)

        return {
            "reaction_rate": round(float(v), 6),
            "vmax": vmax,
            "km": km,
            "substrate_conc": substrate_conc,
            "fraction_of_vmax": round(float(fraction), 4),
            "saturation_percent": round(float(fraction * 100), 2),
            "formula": "v = Vmax * [S] / (Km + [S])",
        }

    def enzyme_inhibition(self, vmax: float, km: float, substrate: float,
                          inhibitor: float, ki: float,
                          inhibition_type: str = "competitive") -> dict:
        """
        Enzyme inhibition kinetics.

        Competitive:     v = Vmax*[S] / (Km*(1 + [I]/Ki) + [S])
        Uncompetitive:   v = Vmax*[S] / (Km + [S]*(1 + [I]/Ki))
        Noncompetitive:  v = Vmax*[S] / ((Km + [S]) * (1 + [I]/Ki))

        Args:
            vmax: Maximum reaction rate.
            km: Michaelis constant.
            substrate: Substrate concentration [S].
            inhibitor: Inhibitor concentration [I].
            ki: Inhibition constant Ki.
            inhibition_type: 'competitive', 'uncompetitive', or 'noncompetitive'.

        Returns:
            Dict with inhibited rate, uninhibited rate, and percent inhibition.
        """
        if ki <= 0:
            return {"error": "Ki must be positive"}

        alpha = 1 + inhibitor / ki
        itype = inhibition_type.lower().strip()

        if itype == "competitive":
            v = vmax * substrate / (km * alpha + substrate)
            apparent_km = km * alpha
            apparent_vmax = vmax
        elif itype == "uncompetitive":
            v = vmax * substrate / (km + substrate * alpha)
            apparent_km = km / alpha
            apparent_vmax = vmax / alpha
        elif itype == "noncompetitive":
            v = vmax * substrate / ((km + substrate) * alpha)
            apparent_km = km
            apparent_vmax = vmax / alpha
        else:
            return {"error": f"Unknown inhibition type: {inhibition_type}"}

        v_uninhibited = vmax * substrate / (km + substrate)
        pct_inhibition = (1 - v / v_uninhibited) * 100 if v_uninhibited > 0 else 0

        return {
            "inhibited_rate": round(float(v), 6),
            "uninhibited_rate": round(float(v_uninhibited), 6),
            "percent_inhibition": round(float(pct_inhibition), 2),
            "apparent_km": round(float(apparent_km), 4),
            "apparent_vmax": round(float(apparent_vmax), 6),
            "inhibition_type": itype,
            "alpha": round(float(alpha), 4),
        }

    # ══════════════════════════════════════════════════════
    # PHARMACOLOGY
    # ══════════════════════════════════════════════════════

    def drug_half_life(self, initial_conc: float, half_life: float,
                       time: float) -> dict:
        """
        First-order drug elimination.

        C(t) = C0 * e^(-0.693 * t / t_half)

        The elimination rate constant k = ln(2) / t_half = 0.693 / t_half.

        Args:
            initial_conc: Initial drug concentration C0.
            half_life: Elimination half-life (same time units as 'time').
            time: Elapsed time.

        Returns:
            Dict with remaining concentration, fraction remaining, and
            number of half-lives elapsed.
        """
        if half_life <= 0:
            return {"error": "Half-life must be positive"}

        k = 0.693147 / half_life  # ln(2) / t_half
        conc = initial_conc * float(exp(-k * time))
        fraction = conc / initial_conc if initial_conc > 0 else 0
        n_halves = time / half_life

        return {
            "remaining_conc": round(conc, 6),
            "initial_conc": initial_conc,
            "half_life": half_life,
            "time": time,
            "elimination_constant_k": round(k, 6),
            "fraction_remaining": round(fraction, 6),
            "half_lives_elapsed": round(n_halves, 2),
            "formula": "C(t) = C0 * exp(-0.693 * t / t_half)",
        }

    def therapeutic_window(self, drug_conc: float, ec50: float,
                           ld50: float) -> dict:
        """
        Determine if a drug concentration falls within the therapeutic window.

        The therapeutic index (TI) = LD50 / EC50.
        A drug is in the safe range if EC50 <= drug_conc < LD50.

        Args:
            drug_conc: Current drug concentration.
            ec50: Half-maximal effective concentration.
            ld50: Median lethal dose concentration.

        Returns:
            Dict with therapeutic index, safety assessment, and efficacy estimate.
        """
        if ec50 <= 0 or ld50 <= 0:
            return {"error": "EC50 and LD50 must be positive"}

        ti = ld50 / ec50
        in_window = ec50 <= drug_conc < ld50
        sub_therapeutic = drug_conc < ec50
        toxic = drug_conc >= ld50

        # Hill equation efficacy estimate (Hill coefficient = 1)
        efficacy = drug_conc / (ec50 + drug_conc)

        if sub_therapeutic:
            status = "sub_therapeutic"
        elif toxic:
            status = "toxic"
        else:
            status = "therapeutic"

        return {
            "drug_conc": drug_conc,
            "ec50": ec50,
            "ld50": ld50,
            "therapeutic_index": round(float(ti), 2),
            "in_therapeutic_window": in_window,
            "status": status,
            "efficacy_fraction": round(float(efficacy), 4),
        }

    def drug_interaction(self, drug_a_effect: float,
                         drug_b_effect: float) -> dict:
        """
        Bliss independence model for drug interaction.

        E_combined = Ea + Eb - Ea * Eb

        Effects should be expressed as fractions (0 to 1).
        Result > Ea + Eb suggests synergy when compared to observed.
        Result < Ea + Eb suggests antagonism when compared to observed.

        Args:
            drug_a_effect: Fractional effect of drug A (0-1).
            drug_b_effect: Fractional effect of drug B (0-1).

        Returns:
            Dict with expected combined effect and individual contributions.
        """
        ea = float(drug_a_effect)
        eb = float(drug_b_effect)
        combined = ea + eb - ea * eb

        return {
            "drug_a_effect": round(ea, 4),
            "drug_b_effect": round(eb, 4),
            "expected_combined_effect": round(combined, 6),
            "additive_sum": round(ea + eb, 4),
            "interaction_term": round(-ea * eb, 6),
            "formula": "E_combined = Ea + Eb - Ea * Eb",
        }

    def dosage_by_weight(self, dose_per_kg: float,
                         patient_weight: float) -> dict:
        """
        Calculate total drug dose based on patient weight.

        Total dose = dose_per_kg * patient_weight

        Args:
            dose_per_kg: Prescribed dose in mg/kg.
            patient_weight: Patient body weight in kg.

        Returns:
            Dict with total dose in mg and g.
        """
        total_mg = dose_per_kg * patient_weight
        total_g = total_mg / 1000.0

        return {
            "dose_per_kg_mg": dose_per_kg,
            "patient_weight_kg": patient_weight,
            "total_dose_mg": round(total_mg, 4),
            "total_dose_g": round(total_g, 6),
        }

    # ══════════════════════════════════════════════════════
    # GENETICS
    # ══════════════════════════════════════════════════════

    def translate_dna(self, dna_sequence: str) -> dict:
        """
        Translate a DNA sequence into a protein sequence using the codon table.

        Reads in-frame from position 0 in triplets. Stops at a stop codon
        or end of sequence.

        Args:
            dna_sequence: DNA string using A, T, G, C (case-insensitive).

        Returns:
            Dict with protein sequence (single-letter), codons used, and length.
        """
        seq = dna_sequence.upper().replace(" ", "").replace("\n", "")
        # Validate
        if not all(c in "ATGC" for c in seq):
            return {"error": "Invalid DNA sequence (use only A, T, G, C)"}

        protein = []
        codons_used = []
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i+3]
            aa = self.codon_table.get(codon)
            if aa is None:
                return {"error": f"Unknown codon: {codon}"}
            if aa == "Stop":
                codons_used.append((codon, "Stop"))
                break
            single = AMINO_ACIDS[aa]["single_letter"]
            protein.append(single)
            codons_used.append((codon, aa))

        protein_str = "".join(protein)

        return {
            "dna_sequence": seq,
            "protein_sequence": protein_str,
            "protein_length": len(protein_str),
            "codons": codons_used,
            "num_codons_read": len(codons_used),
        }

    def hardy_weinberg(self, p: float) -> dict:
        """
        Hardy-Weinberg equilibrium genotype frequencies.

        Given allele frequency p for dominant allele:
            q = 1 - p
            Genotype frequencies: AA = p^2, Aa = 2pq, aa = q^2

        Args:
            p: Frequency of the dominant allele (0-1).

        Returns:
            Dict with genotype and allele frequencies.
        """
        if not 0 <= p <= 1:
            return {"error": "Allele frequency p must be between 0 and 1"}

        q = 1.0 - p
        aa_homo = p ** 2       # AA
        hetero = 2 * p * q     # Aa
        bb_homo = q ** 2       # aa

        return {
            "p": round(p, 6),
            "q": round(q, 6),
            "freq_AA": round(aa_homo, 6),
            "freq_Aa": round(hetero, 6),
            "freq_aa": round(bb_homo, 6),
            "check_sum": round(aa_homo + hetero + bb_homo, 6),
            "formula": "p^2 + 2pq + q^2 = 1",
        }

    def mutation_rate(self, genome_size: int, mutation_per_base: float,
                      generations: int) -> dict:
        """
        Expected number of mutations accumulated over generations.

        Expected mutations per generation = genome_size * mutation_per_base
        Total expected mutations = genome_size * mutation_per_base * generations

        Args:
            genome_size: Number of base pairs in the genome.
            mutation_per_base: Mutation rate per base pair per generation.
            generations: Number of generations.

        Returns:
            Dict with mutations per generation, total expected, and Poisson lambda.
        """
        per_gen = genome_size * mutation_per_base
        total = per_gen * generations

        return {
            "genome_size_bp": genome_size,
            "mutation_per_base_per_gen": mutation_per_base,
            "generations": generations,
            "mutations_per_generation": round(per_gen, 4),
            "total_expected_mutations": round(total, 4),
            "poisson_lambda": round(per_gen, 6),
            "note": "Actual count follows Poisson distribution with lambda = mutations_per_generation",
        }

    # ══════════════════════════════════════════════════════
    # CELL BIOLOGY
    # ══════════════════════════════════════════════════════

    def nernst_potential(self, z: int, conc_out: float, conc_in: float,
                         temp: float = 310) -> dict:
        """
        Nernst equation for equilibrium membrane potential.

        E = (R * T) / (z * F) * ln([out] / [in])

        At body temperature (310 K) this simplifies to approximately:
            E = (26.7 mV / z) * ln([out]/[in])

        Args:
            z: Ion valence (charge number, e.g. +1 for Na+, -1 for Cl-).
            conc_out: Extracellular ion concentration (mM).
            conc_in: Intracellular ion concentration (mM).
            temp: Temperature in Kelvin (default 310 K = 37 C).

        Returns:
            Dict with equilibrium potential in volts and millivolts.
        """
        if z == 0:
            return {"error": "Ion valence z cannot be zero"}
        if conc_out <= 0 or conc_in <= 0:
            return {"error": "Concentrations must be positive"}

        R = 8.314   # J/(mol*K)
        F = 96485   # C/mol

        e_volts = (R * temp) / (z * F) * float(ln(Float(conc_out) / Float(conc_in)))
        e_mv = e_volts * 1000

        return {
            "nernst_potential_V": round(e_volts, 6),
            "nernst_potential_mV": round(e_mv, 2),
            "z": z,
            "conc_out": conc_out,
            "conc_in": conc_in,
            "temp_K": temp,
            "formula": "E = (RT / zF) * ln([out] / [in])",
        }

    def osmotic_pressure(self, molarity: float, temp: float = 310) -> dict:
        """
        Van't Hoff equation for osmotic pressure.

        pi = M * R * T

        where M is molar concentration of solute, R is gas constant,
        T is absolute temperature.

        Args:
            molarity: Molar concentration of solute (mol/L).
            temp: Temperature in Kelvin (default 310 K).

        Returns:
            Dict with osmotic pressure in Pascals, kPa, and atm.
        """
        R = 8.314  # J/(mol*K) = Pa*L/(mol*K) when M in mol/m^3... but:
        # pi = M * R * T with M in mol/L gives pressure in L*J/(mol*L*K)*K = not right
        # Use R = 0.08206 L*atm/(mol*K) for direct atm result, then convert.
        R_L_atm = 0.08206

        pressure_atm = molarity * R_L_atm * temp
        pressure_pa = pressure_atm * 101325
        pressure_kpa = pressure_pa / 1000

        return {
            "osmotic_pressure_atm": round(pressure_atm, 4),
            "osmotic_pressure_Pa": round(pressure_pa, 2),
            "osmotic_pressure_kPa": round(pressure_kpa, 4),
            "molarity_mol_L": molarity,
            "temp_K": temp,
            "formula": "pi = MRT",
        }

    def atp_yield(self, glucose_molecules: float) -> dict:
        """
        Theoretical ATP yield from aerobic cellular respiration.

        Per glucose molecule (complete oxidation):
            Glycolysis:                2 ATP + 2 NADH (-> ~5 ATP via ETC)
            Pyruvate dehydrogenase:    2 NADH (-> ~5 ATP)
            Krebs cycle:               2 ATP + 6 NADH (-> ~15 ATP) + 2 FADH2 (-> ~3 ATP)
            Total:                     ~30-32 ATP (using ~2.5 ATP/NADH, ~1.5 ATP/FADH2)
            Traditional estimate:      36 ATP per glucose

        Args:
            glucose_molecules: Number of glucose molecules (or moles).

        Returns:
            Dict with ATP yield breakdown.
        """
        atp_per_glucose = 36  # traditional textbook value

        glycolysis_atp = 2
        glycolysis_nadh = 2
        pyruvate_dehydrogenase_nadh = 2
        krebs_atp = 2
        krebs_nadh = 6
        krebs_fadh2 = 2
        etc_from_nadh = (glycolysis_nadh + pyruvate_dehydrogenase_nadh + krebs_nadh) * 2.5
        etc_from_fadh2 = krebs_fadh2 * 1.5

        modern_total = glycolysis_atp + krebs_atp + etc_from_nadh + etc_from_fadh2

        return {
            "glucose_input": glucose_molecules,
            "atp_per_glucose_traditional": atp_per_glucose,
            "atp_per_glucose_modern": round(modern_total, 1),
            "total_atp_traditional": round(glucose_molecules * atp_per_glucose, 2),
            "total_atp_modern": round(glucose_molecules * modern_total, 2),
            "breakdown": {
                "glycolysis_atp": glycolysis_atp,
                "glycolysis_nadh": glycolysis_nadh,
                "pyruvate_dehydrogenase_nadh": pyruvate_dehydrogenase_nadh,
                "krebs_cycle_atp": krebs_atp,
                "krebs_cycle_nadh": krebs_nadh,
                "krebs_cycle_fadh2": krebs_fadh2,
                "etc_atp_from_nadh": etc_from_nadh,
                "etc_atp_from_fadh2": etc_from_fadh2,
            },
            "formula": "C6H12O6 + 6O2 -> 6CO2 + 6H2O + ~30-36 ATP",
        }

    # ══════════════════════════════════════════════════════
    # ECOLOGY
    # ══════════════════════════════════════════════════════

    def logistic_growth(self, n: float, r: float, k: float) -> dict:
        """
        Logistic population growth model.

        dN/dt = r * N * (1 - N/K)

        Args:
            n: Current population size N.
            r: Intrinsic growth rate.
            k: Carrying capacity K.

        Returns:
            Dict with growth rate dN/dt, per-capita rate, and status.
        """
        if k == 0:
            return {"error": "Carrying capacity K cannot be zero"}

        dndt = r * n * (1 - n / k)
        per_capita = r * (1 - n / k)

        if n < k:
            status = "growing"
        elif n == k:
            status = "at_equilibrium"
        else:
            status = "declining"

        return {
            "dN_dt": round(float(dndt), 4),
            "per_capita_rate": round(float(per_capita), 6),
            "N": n,
            "r": r,
            "K": k,
            "N_over_K": round(n / k, 4),
            "status": status,
            "formula": "dN/dt = rN(1 - N/K)",
        }

    def sir_model(self, s: float, i: float, r: float,
                  beta: float, gamma: float) -> dict:
        """
        SIR compartmental model for epidemic spread.

        dS/dt = -beta * S * I
        dI/dt =  beta * S * I - gamma * I
        dR/dt =  gamma * I

        R0 (basic reproduction number) = beta / gamma

        Args:
            s: Susceptible fraction (0-1).
            i: Infected fraction (0-1).
            r: Recovered fraction (0-1).
            beta: Transmission rate.
            gamma: Recovery rate (1/gamma = average infectious period).

        Returns:
            Dict with derivatives dS/dt, dI/dt, dR/dt, and R0.
        """
        ds_dt = -beta * s * i
        di_dt = beta * s * i - gamma * i
        dr_dt = gamma * i
        r0 = beta / gamma if gamma > 0 else float('inf')

        if di_dt > 0:
            phase = "epidemic_growing"
        elif di_dt < 0:
            phase = "epidemic_declining"
        else:
            phase = "equilibrium"

        return {
            "dS_dt": round(float(ds_dt), 6),
            "dI_dt": round(float(di_dt), 6),
            "dR_dt": round(float(dr_dt), 6),
            "R0": round(float(r0), 4),
            "effective_R": round(float(r0 * s), 4),
            "S": s, "I": i, "R": r,
            "beta": beta, "gamma": gamma,
            "phase": phase,
            "herd_immunity_threshold": round(1 - 1/r0, 4) if r0 > 1 else 0,
            "formulas": {
                "dS_dt": "-beta * S * I",
                "dI_dt": "beta * S * I - gamma * I",
                "dR_dt": "gamma * I",
                "R0": "beta / gamma",
            },
        }

    def predator_prey(self, prey: float, predator: float,
                      alpha: float, beta: float,
                      delta: float, gamma: float) -> dict:
        """
        Lotka-Volterra predator-prey model.

        dx/dt = alpha*x - beta*x*y    (prey growth - predation)
        dy/dt = delta*x*y - gamma*y   (predator growth - predator death)

        Args:
            prey: Prey population x.
            predator: Predator population y.
            alpha: Prey natural growth rate.
            beta: Predation rate coefficient.
            delta: Predator growth rate per prey consumed.
            gamma: Predator natural death rate.

        Returns:
            Dict with population derivatives and equilibrium values.
        """
        x = prey
        y = predator

        dx_dt = alpha * x - beta * x * y
        dy_dt = delta * x * y - gamma * y

        # Equilibrium: dx/dt = 0 and dy/dt = 0 (non-trivial)
        # x* = gamma / delta,  y* = alpha / beta
        eq_prey = gamma / delta if delta > 0 else float('inf')
        eq_predator = alpha / beta if beta > 0 else float('inf')

        return {
            "dx_dt": round(float(dx_dt), 4),
            "dy_dt": round(float(dy_dt), 4),
            "prey": prey,
            "predator": predator,
            "alpha": alpha,
            "beta": beta,
            "delta": delta,
            "gamma": gamma,
            "equilibrium_prey": round(float(eq_prey), 4),
            "equilibrium_predator": round(float(eq_predator), 4),
            "formulas": {
                "dx_dt": "alpha*x - beta*x*y",
                "dy_dt": "delta*x*y - gamma*y",
            },
        }

    # ══════════════════════════════════════════════════════
    # ROUTER INTEGRATION
    # ══════════════════════════════════════════════════════

    def process(self, query: str) -> str:
        """Process a biology query and return computed answer."""
        q = query.lower()

        # Michaelis-Menten
        mm_match = re.search(
            r'michaelis.*?vmax\s*[=:]\s*(\d+\.?\d*).*?km\s*[=:]\s*(\d+\.?\d*).*?(?:\[?s\]?|substrate)\s*[=:]\s*(\d+\.?\d*)',
            q
        )
        if mm_match:
            vmax = float(mm_match.group(1))
            km = float(mm_match.group(2))
            s = float(mm_match.group(3))
            r = self.michaelis_menten(vmax, km, s)
            return (f"Michaelis-Menten: v = {r['reaction_rate']} "
                    f"({r['saturation_percent']}% of Vmax)")

        # Drug half-life
        hl_match = re.search(
            r'half[- ]?life.*?(\d+\.?\d*)\s*(?:mg|ng|ug)?.*?half[- ]?life\s*[=:]\s*(\d+\.?\d*).*?(?:time|at|after)\s*[=:]\s*(\d+\.?\d*)',
            q
        )
        if hl_match:
            c0 = float(hl_match.group(1))
            hl = float(hl_match.group(2))
            t = float(hl_match.group(3))
            r = self.drug_half_life(c0, hl, t)
            return (f"Drug concentration after {t}h: {r['remaining_conc']} "
                    f"({r['half_lives_elapsed']} half-lives elapsed)")

        # Hardy-Weinberg
        hw_match = re.search(r'hardy.*?weinberg.*?p\s*[=:]\s*(\d+\.?\d*)', q)
        if hw_match:
            p = float(hw_match.group(1))
            r = self.hardy_weinberg(p)
            if "error" not in r:
                return (f"Hardy-Weinberg (p={p}): AA={r['freq_AA']}, "
                        f"Aa={r['freq_Aa']}, aa={r['freq_aa']}")

        # DNA translation
        trans_match = re.search(r'translat.*?([ATGC]{6,})', query)
        if trans_match:
            seq = trans_match.group(1)
            r = self.translate_dna(seq)
            if "error" not in r:
                return (f"Protein: {r['protein_sequence']} "
                        f"({r['protein_length']} amino acids)")

        # Nernst potential
        nernst_match = re.search(
            r'nernst.*?z\s*[=:]\s*([+-]?\d+).*?out\s*[=:]\s*(\d+\.?\d*).*?in\s*[=:]\s*(\d+\.?\d*)',
            q
        )
        if nernst_match:
            z = int(nernst_match.group(1))
            cout = float(nernst_match.group(2))
            cin = float(nernst_match.group(3))
            r = self.nernst_potential(z, cout, cin)
            if "error" not in r:
                return (f"Nernst potential: {r['nernst_potential_mV']} mV "
                        f"(z={z}, [out]={cout}, [in]={cin})")

        # Logistic growth
        log_match = re.search(
            r'logistic.*?n\s*[=:]\s*(\d+\.?\d*).*?r\s*[=:]\s*(\d+\.?\d*).*?k\s*[=:]\s*(\d+\.?\d*)',
            q
        )
        if log_match:
            n = float(log_match.group(1))
            r_val = float(log_match.group(2))
            k = float(log_match.group(3))
            r = self.logistic_growth(n, r_val, k)
            return (f"Logistic growth: dN/dt = {r['dN_dt']} "
                    f"(N={n}, r={r_val}, K={k}, {r['status']})")

        # SIR model
        sir_match = re.search(
            r'sir.*?beta\s*[=:]\s*(\d+\.?\d*).*?gamma\s*[=:]\s*(\d+\.?\d*)',
            q
        )
        if sir_match:
            beta = float(sir_match.group(1))
            gamma = float(sir_match.group(2))
            r0 = beta / gamma if gamma > 0 else float('inf')
            return f"SIR R0 = {round(r0, 2)} (beta={beta}, gamma={gamma})"

        # ATP yield
        atp_match = re.search(r'atp.*?(\d+\.?\d*)\s*(?:glucose|molecules?)', q)
        if atp_match:
            gluc = float(atp_match.group(1))
            r = self.atp_yield(gluc)
            return (f"ATP from {gluc} glucose: {r['total_atp_modern']} ATP "
                    f"(modern) / {r['total_atp_traditional']} ATP (traditional)")

        # Osmotic pressure
        osm_match = re.search(r'osmotic.*?(\d+\.?\d*)\s*[mM]', q)
        if osm_match:
            m = float(osm_match.group(1))
            r = self.osmotic_pressure(m)
            return (f"Osmotic pressure at {m} M: {r['osmotic_pressure_atm']} atm "
                    f"= {r['osmotic_pressure_kPa']} kPa")

        # Dosage
        dose_match = re.search(
            r'dos(?:age|e).*?(\d+\.?\d*)\s*mg/kg.*?(\d+\.?\d*)\s*kg',
            q
        )
        if dose_match:
            dpk = float(dose_match.group(1))
            wt = float(dose_match.group(2))
            r = self.dosage_by_weight(dpk, wt)
            return f"Total dose: {r['total_dose_mg']} mg for {wt} kg patient"

        # Amino acid lookup
        for code, data in AMINO_ACIDS.items():
            if data["name"] in q or code.lower() in q.split():
                return (f"{data['name'].title()} ({code}/{data['single_letter']}): "
                        f"MW {data['molecular_weight']} Da, "
                        f"charge at pH7 {data['charge_at_pH7']}, "
                        f"hydrophobicity {data['hydrophobicity']}")

        return ""
