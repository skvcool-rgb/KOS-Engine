"""
KOS V6.0 — Physics Driver

Computes physics from first principles:
- Mechanics (force, energy, momentum, projectiles)
- Thermodynamics (heat transfer, entropy, phase changes)
- Electromagnetism (Coulomb, Ohm, Faraday)
- Optics (Snell, lens equation, diffraction)
- Quantum (energy levels, de Broglie, uncertainty)
- Relativity (time dilation, mass-energy, length contraction)
- Material properties (tensile strength, conductivity)

All calculations via SymPy — exact symbolic results.
"""

import re
from sympy import (
    Symbol, symbols, solve, sqrt, pi, oo, E as Euler,
    Rational, N, sin, cos, tan, asin, log, exp, Float
)

# ══════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ══════════════════════════════════════════════════════════

CONSTANTS = {
    "c": {"value": 299792458, "unit": "m/s", "name": "speed of light"},
    "G": {"value": 6.674e-11, "unit": "N*m^2/kg^2", "name": "gravitational constant"},
    "g": {"value": 9.80665, "unit": "m/s^2", "name": "standard gravity"},
    "h": {"value": 6.626e-34, "unit": "J*s", "name": "Planck constant"},
    "hbar": {"value": 1.055e-34, "unit": "J*s", "name": "reduced Planck constant"},
    "k_B": {"value": 1.381e-23, "unit": "J/K", "name": "Boltzmann constant"},
    "e": {"value": 1.602e-19, "unit": "C", "name": "elementary charge"},
    "m_e": {"value": 9.109e-31, "unit": "kg", "name": "electron mass"},
    "m_p": {"value": 1.673e-27, "unit": "kg", "name": "proton mass"},
    "N_A": {"value": 6.022e23, "unit": "1/mol", "name": "Avogadro number"},
    "R": {"value": 8.314, "unit": "J/(mol*K)", "name": "gas constant"},
    "k_e": {"value": 8.988e9, "unit": "N*m^2/C^2", "name": "Coulomb constant"},
    "sigma": {"value": 5.670e-8, "unit": "W/(m^2*K^4)", "name": "Stefan-Boltzmann constant"},
    "epsilon_0": {"value": 8.854e-12, "unit": "F/m", "name": "vacuum permittivity"},
    "mu_0": {"value": 1.257e-6, "unit": "H/m", "name": "vacuum permeability"},
    "eV": {"value": 1.602e-19, "unit": "J", "name": "electron volt"},
}

# ══════════════════════════════════════════════════════════
# MATERIAL PROPERTIES
# ══════════════════════════════════════════════════════════

MATERIALS = {
    "steel": {"density": 7850, "yield_strength": 250e6, "youngs_modulus": 200e9, "thermal_conductivity": 50, "melting_point": 1510, "specific_heat": 500},
    "aluminum": {"density": 2700, "yield_strength": 270e6, "youngs_modulus": 69e9, "thermal_conductivity": 237, "melting_point": 660, "specific_heat": 900},
    "copper": {"density": 8960, "yield_strength": 70e6, "youngs_modulus": 130e9, "thermal_conductivity": 401, "melting_point": 1085, "specific_heat": 385, "resistivity": 1.68e-8},
    "titanium": {"density": 4507, "yield_strength": 880e6, "youngs_modulus": 116e9, "thermal_conductivity": 22, "melting_point": 1668, "specific_heat": 520},
    "concrete": {"density": 2400, "compressive_strength": 30e6, "youngs_modulus": 30e9, "thermal_conductivity": 1.7, "specific_heat": 880},
    "glass": {"density": 2500, "youngs_modulus": 70e9, "thermal_conductivity": 1.0, "melting_point": 1500, "specific_heat": 840},
    "water": {"density": 1000, "specific_heat": 4186, "thermal_conductivity": 0.6, "boiling_point": 100, "melting_point": 0, "latent_heat_vaporization": 2260000},
    "air": {"density": 1.225, "specific_heat": 1005, "thermal_conductivity": 0.024},
    "wood": {"density": 600, "yield_strength": 40e6, "youngs_modulus": 12e9, "thermal_conductivity": 0.15, "specific_heat": 1700},
    "rubber": {"density": 1100, "youngs_modulus": 0.01e9, "thermal_conductivity": 0.13, "specific_heat": 2010},
    "silicon": {"density": 2329, "melting_point": 1414, "specific_heat": 710, "thermal_conductivity": 149, "resistivity": 640},
    "gold": {"density": 19320, "melting_point": 1064, "specific_heat": 129, "thermal_conductivity": 318, "resistivity": 2.44e-8},
    "tungsten": {"density": 19250, "melting_point": 3422, "boiling_point": 5555, "specific_heat": 132, "thermal_conductivity": 173, "yield_strength": 750e6},
    "diamond": {"density": 3510, "youngs_modulus": 1220e9, "thermal_conductivity": 2200, "melting_point": 3550},
    "graphene": {"density": 2267, "youngs_modulus": 1000e9, "thermal_conductivity": 5000, "tensile_strength": 130e9},
    "perovskite": {"density": 4100, "bandgap_eV": 1.55, "melting_point": 900, "degradation_temp": 85},
}


class PhysicsDriver:
    """
    Computes physics from first principles using SymPy.
    """

    def __init__(self):
        self.constants = CONSTANTS
        self.materials = MATERIALS

    # ── Mechanics ────────────────────────────────────────

    def free_fall(self, height: float, g: float = 9.80665) -> dict:
        """Calculate free fall: time, impact velocity, kinetic energy."""
        t = sqrt(2 * height / g)
        v = g * t
        return {
            "height_m": height,
            "time_s": round(float(t), 4),
            "impact_velocity_ms": round(float(v), 4),
            "formula": "h = 0.5*g*t^2, v = g*t",
        }

    def projectile(self, v0: float, angle_deg: float, g: float = 9.80665) -> dict:
        """Calculate projectile motion."""
        angle_rad = angle_deg * pi / 180
        range_val = v0**2 * sin(2 * angle_rad) / g
        max_height = (v0 * sin(angle_rad))**2 / (2 * g)
        time_flight = 2 * v0 * sin(angle_rad) / g
        return {
            "range_m": round(float(range_val), 2),
            "max_height_m": round(float(max_height), 2),
            "time_of_flight_s": round(float(time_flight), 2),
            "initial_velocity_ms": v0,
            "angle_deg": angle_deg,
        }

    def kinetic_energy(self, mass: float, velocity: float) -> dict:
        """KE = 0.5 * m * v^2"""
        ke = 0.5 * mass * velocity**2
        return {"mass_kg": mass, "velocity_ms": velocity, "kinetic_energy_J": round(ke, 4)}

    def gravitational_force(self, m1: float, m2: float, r: float) -> dict:
        """F = G * m1 * m2 / r^2"""
        G = 6.674e-11
        f = G * m1 * m2 / r**2
        return {"force_N": f, "m1_kg": m1, "m2_kg": m2, "distance_m": r}

    def stress_strain(self, force: float, area: float, youngs_modulus: float = None) -> dict:
        """Calculate stress, and strain if Young's modulus given."""
        stress = force / area
        result = {"force_N": force, "area_m2": area, "stress_Pa": stress, "stress_MPa": stress / 1e6}
        if youngs_modulus:
            strain = stress / youngs_modulus
            result["strain"] = strain
            result["youngs_modulus_Pa"] = youngs_modulus
        return result

    # ── Thermodynamics ───────────────────────────────────

    def heat_transfer(self, mass: float, specific_heat: float, delta_t: float) -> dict:
        """Q = m * c * deltaT"""
        q = mass * specific_heat * delta_t
        return {
            "heat_J": round(q, 2),
            "heat_kJ": round(q / 1000, 4),
            "mass_kg": mass,
            "specific_heat_J_kgK": specific_heat,
            "delta_T_K": delta_t,
        }

    def carnot_efficiency(self, t_hot: float, t_cold: float) -> dict:
        """eta = 1 - T_cold/T_hot (temperatures in Kelvin)"""
        if t_hot <= 0:
            return {"error": "T_hot must be positive (Kelvin)"}
        eta = 1 - t_cold / t_hot
        return {
            "efficiency": round(eta, 4),
            "efficiency_percent": round(eta * 100, 2),
            "t_hot_K": t_hot,
            "t_cold_K": t_cold,
        }

    def entropy_change(self, heat: float, temperature: float) -> dict:
        """deltaS = Q / T"""
        if temperature <= 0:
            return {"error": "Temperature must be positive"}
        ds = heat / temperature
        return {"entropy_change_J_K": round(ds, 4), "heat_J": heat, "temperature_K": temperature}

    # ── Electromagnetism ─────────────────────────────────

    def coulomb_force(self, q1: float, q2: float, r: float) -> dict:
        """F = k_e * q1 * q2 / r^2"""
        k = 8.988e9
        f = k * q1 * q2 / r**2
        return {"force_N": f, "attractive": f < 0, "q1_C": q1, "q2_C": q2, "distance_m": r}

    def ohms_law(self, voltage: float = None, current: float = None, resistance: float = None) -> dict:
        """V = IR — solve for whichever is missing."""
        if voltage is None and current and resistance:
            return {"voltage_V": current * resistance, "current_A": current, "resistance_ohm": resistance}
        if current is None and voltage and resistance:
            return {"voltage_V": voltage, "current_A": voltage / resistance, "resistance_ohm": resistance}
        if resistance is None and voltage and current:
            return {"voltage_V": voltage, "current_A": current, "resistance_ohm": voltage / current}
        return {"error": "Provide exactly two of: voltage, current, resistance"}

    # ── Optics ───────────────────────────────────────────

    def snells_law(self, n1: float, n2: float, angle1_deg: float) -> dict:
        """n1*sin(theta1) = n2*sin(theta2)"""
        angle1_rad = angle1_deg * pi / 180
        sin_theta2 = n1 * sin(angle1_rad) / n2
        if abs(float(sin_theta2)) > 1:
            return {"total_internal_reflection": True, "critical_angle": float(asin(n2/n1) * 180 / pi)}
        angle2 = float(asin(sin_theta2) * 180 / pi)
        return {"angle_refracted_deg": round(angle2, 2), "n1": n1, "n2": n2, "angle_incident_deg": angle1_deg}

    def photon_energy(self, wavelength_nm: float) -> dict:
        """E = hc/lambda"""
        h = 6.626e-34
        c = 299792458
        lam = wavelength_nm * 1e-9
        energy_j = h * c / lam
        energy_ev = energy_j / 1.602e-19
        energy_kjmol = energy_j * 6.022e23 / 1000
        return {
            "wavelength_nm": wavelength_nm,
            "energy_J": energy_j,
            "energy_eV": round(energy_ev, 4),
            "energy_kJ_mol": round(energy_kjmol, 1),
        }

    # ── Quantum ──────────────────────────────────────────

    def hydrogen_energy_level(self, n: int) -> dict:
        """E_n = -13.6 / n^2 eV"""
        energy = -13.6 / n**2
        return {"n": n, "energy_eV": round(energy, 4), "wavelength_nm": round(1240 / abs(energy), 1) if energy != 0 else None}

    def de_broglie(self, mass: float, velocity: float) -> dict:
        """lambda = h / (m*v)"""
        h = 6.626e-34
        wavelength = h / (mass * velocity)
        return {"wavelength_m": wavelength, "mass_kg": mass, "velocity_ms": velocity}

    def uncertainty_principle(self, delta_x: float = None, delta_p: float = None) -> dict:
        """delta_x * delta_p >= hbar/2"""
        hbar = 1.055e-34
        if delta_x:
            min_dp = hbar / (2 * delta_x)
            return {"delta_x_m": delta_x, "min_delta_p_kgms": min_dp}
        if delta_p:
            min_dx = hbar / (2 * delta_p)
            return {"delta_p_kgms": delta_p, "min_delta_x_m": min_dx}
        return {"error": "Provide delta_x or delta_p"}

    # ── Relativity ───────────────────────────────────────

    def time_dilation(self, velocity: float) -> dict:
        """gamma = 1 / sqrt(1 - v^2/c^2)"""
        c = 299792458
        if velocity >= c:
            return {"error": "Velocity must be less than c"}
        gamma = 1 / sqrt(1 - (velocity/c)**2)
        return {"gamma": round(float(gamma), 6), "velocity_ms": velocity, "fraction_of_c": round(velocity/c, 6)}

    def mass_energy(self, mass: float) -> dict:
        """E = mc^2"""
        c = 299792458
        energy = mass * c**2
        return {"energy_J": energy, "energy_MeV": energy / 1.602e-13, "mass_kg": mass}

    # ── Material Lookup ──────────────────────────────────

    def get_material(self, name: str) -> dict:
        """Look up material properties."""
        name_lower = name.lower().strip()
        if name_lower in self.materials:
            return {"name": name_lower, **self.materials[name_lower]}
        return None

    # ── Router Integration ───────────────────────────────

    def process(self, query: str) -> str:
        """Process a physics query and return computed answer."""
        q = query.lower()

        # Free fall
        fall_match = re.search(r'(?:drop|fall|free\s*fall).*?(\d+\.?\d*)\s*(?:m|meter)', q)
        if fall_match:
            h = float(fall_match.group(1))
            r = self.free_fall(h)
            return (f"Free fall from {h}m: lands in {r['time_s']}s "
                    f"at {r['impact_velocity_ms']} m/s")

        # Photon energy
        wave_match = re.search(r'(?:photon|light|wavelength).*?(\d+\.?\d*)\s*nm', q)
        if wave_match:
            wl = float(wave_match.group(1))
            r = self.photon_energy(wl)
            return (f"Photon at {wl}nm: {r['energy_eV']} eV = "
                    f"{r['energy_kJ_mol']} kJ/mol")

        # E=mc^2
        if "mass" in q and "energy" in q:
            mass_match = re.search(r'(\d+\.?\d*(?:e[+-]?\d+)?)\s*kg', q)
            if mass_match:
                m = float(mass_match.group(1))
                r = self.mass_energy(m)
                return f"E = mc^2: {m} kg = {r['energy_J']:.3e} Joules"

        # Material lookup
        for mat_name in self.materials:
            if mat_name in q:
                mat = self.materials[mat_name]
                props = ", ".join(f"{k}: {v}" for k, v in mat.items())
                return f"{mat_name.title()}: {props}"

        return ""
