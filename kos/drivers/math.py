"""
KOS V2.0 — Math Coprocessor (Symbolic CAS Driver).

Translates text strings into strict Abstract Syntax Trees (ASTs)
for mathematical evaluation via SymPy. Zero hallucination —
all results are computed symbolically.

Supports: Integration, Differentiation, Algebra, Arithmetic.
"""
import re
import sympy as sp


class MathDriver:
    def __init__(self):
        self.math_keywords = {
            "integrate", "derive", "derivative",
            "solve", "calculate", "limit",
        }

    def is_math_query(self, prompt: str) -> bool:
        """Detects if the user is asking a calculus/algebra question."""
        lower_prompt = prompt.lower()
        # Trigger if it contains math keywords and math symbols
        if any(kw in lower_prompt for kw in self.math_keywords):
            return True
        if re.search(r'[\+\-\*/\^=]', prompt):
            return True
        return False

    def solve(self, prompt: str) -> dict:
        """Executes symbolic mathematics with 0% hallucination."""
        # Clean the prompt (convert ^ to ** for Python syntax)
        clean_prompt = prompt.lower().replace("^", "**")

        try:
            # Basic NLP Triage for Calculus
            if "integrate" in clean_prompt:
                # E.g., "integrate x**2 * cos(x)"
                expr_str = clean_prompt.split("integrate")[1].strip()
                expr = sp.sympify(expr_str)
                var = (list(expr.free_symbols)[0]
                       if expr.free_symbols else sp.Symbol('x'))
                result = sp.integrate(expr, var)
                return {
                    "status": "success",
                    "operation": "Integration",
                    "result": str(result),
                    "equation": str(expr),
                }

            elif "derive" in clean_prompt or "derivative" in clean_prompt:
                # E.g., "derivative of sin(x)*exp(x)"
                expr_str = re.split(
                    r'derive|derivative of', clean_prompt)[-1].strip()
                expr = sp.sympify(expr_str)
                var = (list(expr.free_symbols)[0]
                       if expr.free_symbols else sp.Symbol('x'))
                result = sp.diff(expr, var)
                return {
                    "status": "success",
                    "operation": "Differentiation",
                    "result": str(result),
                    "equation": str(expr),
                }

            else:
                # Standard Algebra / Arithmetic
                # E.g., "calculate 24591 * 13492"
                expr_str = clean_prompt.replace(
                    "calculate", "").replace("solve", "").strip()
                result = sp.sympify(expr_str).evalf()
                return {
                    "status": "success",
                    "operation": "Calculation",
                    "result": str(result),
                    "equation": expr_str,
                }

        except Exception as e:
            return {"status": "error", "message": str(e)}
