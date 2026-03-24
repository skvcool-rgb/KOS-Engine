"""
KOS V2.1 — Math Coprocessor (Symbolic CAS Driver).

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
            "integrate", "integral", "derive", "derivative", "differentiate",
            "solve", "calculate", "compute", "evaluate", "limit",
            # Agent Fix 2: add math function names
            "sqrt", "log", "ln", "sin", "cos", "tan", "exp",
            "abs", "factorial", "summation", "product",
        }
        # Regex: matches expressions like "345000000 * 0.0825" or "2+3*4"
        self._bare_math_re = re.compile(
            r'^\s*[\d\.\(\)]+\s*[\+\-\*/\*\*\^%]+\s*[\d\.\(\)]+')
        # Agent Fix 2: matches function(number) patterns like sqrt(144), log(100)
        self._func_math_re = re.compile(
            r'^\s*(?:sqrt|log|ln|sin|cos|tan|exp|abs|factorial)\s*\(\s*[\d\.\s\+\-\*/\^]+\s*\)\s*\??$',
            re.IGNORECASE)
        # Regex: matches "what is <math>" or "how much is <math>"
        self._question_math_re = re.compile(
            r'(?:what\s+is|how\s+much\s+is|compute|evaluate|calculate)\s+'
            r'([\d\.\s\+\-\*/\^\(\)]+[\d\.\)])\s*\??',
            re.IGNORECASE)

    def is_math_query(self, prompt: str) -> bool:
        """Detects if the user is asking a calculus/algebra/arithmetic question."""
        lower = prompt.lower()
        # 1. Explicit math keywords (whole word match only!)
        # Prevents "entanglement" matching "tan", "integral" matching "integrate"
        prompt_words = set(re.findall(r'\b[a-z]+\b', lower))
        if self.math_keywords & prompt_words:
            return True
        # 2. Bare arithmetic expression (e.g. "345000000 * 0.0825")
        if self._bare_math_re.search(prompt):
            return True
        # 3. Agent Fix 2: function(number) like sqrt(144), log(100)
        if self._func_math_re.search(prompt):
            return True
        # 4. "What is 2+2?" style questions with numbers and operators
        if self._question_math_re.search(prompt):
            return True
        # 5. Contains math operators between numbers
        if re.search(r'\d+\s*[\+\-\*/\^]\s*\d+', prompt):
            return True
        return False

    def _clean_expr(self, text: str) -> str:
        """Strip natural language wrappers and normalize math syntax."""
        s = text.strip()
        # Strip question marks
        s = s.rstrip('?')
        # Strip common question prefixes
        s = re.sub(r'^(?:what\s+is|how\s+much\s+is|compute|evaluate|calculate|solve)\s+',
                   '', s, flags=re.IGNORECASE).strip()
        # Convert ^ to ** for Python/SymPy
        s = s.replace('^', '**')
        # Convert ln() to log() for SymPy (SymPy log = natural log)
        s = re.sub(r'\bln\b', 'log', s)
        return s

    def solve(self, prompt: str) -> dict:
        """Executes symbolic mathematics with 0% hallucination."""
        lower = prompt.lower()

        try:
            # --- INTEGRATION ---
            if 'integrat' in lower or 'integral' in lower:
                # Extract expression after integration keyword
                expr_str = re.split(
                    r'integrate|integral\s+of|integral', lower, maxsplit=1)[-1].strip()
                expr_str = self._clean_expr(expr_str)
                # Remove trailing ", x" or "dx" variable spec
                expr_str = re.sub(r',\s*[a-z]\s*$', '', expr_str)
                expr_str = re.sub(r'\s+d[a-z]\s*$', '', expr_str)
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

            # --- DIFFERENTIATION ---
            elif 'deriv' in lower or 'differentiat' in lower:
                expr_str = re.split(
                    r'derivative\s+of|derive|differentiate', lower, maxsplit=1)[-1].strip()
                expr_str = self._clean_expr(expr_str)
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

            # --- ARITHMETIC / ALGEBRA ---
            else:
                expr_str = self._clean_expr(lower)
                # Agent Fix 2: preserve math function names before stripping
                # Replace function names with SymPy equivalents
                _math_funcs = {
                    'sqrt': 'sqrt', 'log': 'log', 'ln': 'log',
                    'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
                    'exp': 'exp', 'abs': 'Abs', 'factorial': 'factorial',
                }
                for fname, sympy_name in _math_funcs.items():
                    expr_str = re.sub(r'\b' + fname + r'\b', sympy_name, expr_str)
                # Remove remaining non-math words (keep math tokens + function names)
                # Only strip words that are NOT known math functions/variables
                tokens = expr_str.split()
                clean_tokens = []
                for tok in tokens:
                    # Keep if it contains digits, operators, parens, or is a math func
                    if (re.search(r'[\d\+\-\*/\^\(\)\.]', tok) or
                        tok in _math_funcs.values() or
                        tok in {'x', 'y', 'z', 'e', 'pi', 'n', 'i'} or
                        '(' in tok):
                        clean_tokens.append(tok)
                expr_str = ' '.join(clean_tokens).strip()
                if not expr_str:
                    expr_str = self._clean_expr(lower)
                    expr_str = re.sub(r'[a-df-hj-mo-qs-wyz]+', '', expr_str).strip()
                if not expr_str:
                    return {"status": "error", "message": "No expression found"}
                result = sp.sympify(expr_str).evalf()
                return {
                    "status": "success",
                    "operation": "Calculation",
                    "result": str(result),
                    "equation": expr_str,
                }

        except Exception as e:
            return {"status": "error", "message": str(e)}
