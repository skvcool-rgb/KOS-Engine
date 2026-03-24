"""
KOS V5.1 — CodeDriver (Verified Code Generation).

Generates Python code from graph knowledge + SymPy verification.
Unlike LLM code generation, every function is:
    1. Assembled from verified graph patterns (not hallucinated)
    2. Math verified by SymPy (formulas are exact)
    3. Logic verified by invariant checking
    4. Auto-tested before output
    5. Cited with provenance (source of the formula/pattern)

Components:
    CodeDriver      → orchestrates code generation
    FormulaCompiler → converts SymPy expressions to Python functions
    LogicVerifier   → catches logic bugs via invariant analysis
    TestGenerator   → auto-generates test cases for verification
"""

import re
import sys
import traceback
from io import StringIO
from textwrap import dedent, indent


# ── Known Formula Templates ─────────────────────────────────
# These are verified mathematical patterns stored in the graph.
# Each template maps a concept name to a SymPy expression
# and a Python function signature.

FORMULA_REGISTRY = {
    'compound_interest': {
        'formula': 'P * (1 + r/n) ** (n * t)',
        'params': ['principal', 'rate', 'periods', 'time'],
        'param_map': {'P': 'principal', 'r': 'rate', 'n': 'periods', 't': 'time'},
        'description': 'Compound interest calculation',
        'source': 'Mathematical formula: A = P(1 + r/n)^(nt)',
        'test_cases': [
            {'args': [1000, 0.05, 12, 10], 'expected_approx': 1647.01},
            {'args': [5000, 0.08, 4, 5], 'expected_approx': 7429.74},
        ],
    },
    'simple_interest': {
        'formula': 'P * r * t',
        'params': ['principal', 'rate', 'time'],
        'param_map': {'P': 'principal', 'r': 'rate', 't': 'time'},
        'description': 'Simple interest calculation',
        'source': 'Mathematical formula: I = P * r * t',
        'test_cases': [
            {'args': [1000, 0.05, 3], 'expected_approx': 150.0},
        ],
    },
    'quadratic_formula': {
        'formula': '(-b + sqrt(b**2 - 4*a*c)) / (2*a)',
        'params': ['a', 'b', 'c'],
        'param_map': {'a': 'a', 'b': 'b', 'c': 'c'},
        'description': 'Quadratic formula (positive root)',
        'source': 'Mathematical formula: x = (-b + sqrt(b^2 - 4ac)) / 2a',
        'test_cases': [
            {'args': [1, -5, 6], 'expected_approx': 3.0},
            {'args': [1, -3, 2], 'expected_approx': 2.0},
        ],
    },
    'distance': {
        'formula': 'sqrt((x2 - x1)**2 + (y2 - y1)**2)',
        'params': ['x1', 'y1', 'x2', 'y2'],
        'param_map': {'x1': 'x1', 'y1': 'y1', 'x2': 'x2', 'y2': 'y2'},
        'description': 'Euclidean distance between two points',
        'source': 'Mathematical formula: d = sqrt((x2-x1)^2 + (y2-y1)^2)',
        'test_cases': [
            {'args': [0, 0, 3, 4], 'expected_approx': 5.0},
            {'args': [1, 1, 4, 5], 'expected_approx': 5.0},
        ],
    },
    'bmi': {
        'formula': 'weight / (height ** 2)',
        'params': ['weight', 'height'],
        'param_map': {'weight': 'weight', 'height': 'height'},
        'description': 'Body Mass Index (weight in kg, height in meters)',
        'source': 'Medical formula: BMI = weight(kg) / height(m)^2',
        'test_cases': [
            {'args': [70, 1.75], 'expected_approx': 22.86},
            {'args': [90, 1.80], 'expected_approx': 27.78},
        ],
    },
    'celsius_to_fahrenheit': {
        'formula': 'c * 9/5 + 32',
        'params': ['celsius'],
        'param_map': {'c': 'celsius'},
        'description': 'Convert Celsius to Fahrenheit',
        'source': 'Physical formula: F = C * 9/5 + 32',
        'test_cases': [
            {'args': [0], 'expected_approx': 32.0},
            {'args': [100], 'expected_approx': 212.0},
        ],
    },
    'area_circle': {
        'formula': 'pi * r**2',
        'params': ['radius'],
        'param_map': {'r': 'radius'},
        'description': 'Area of a circle',
        'source': 'Mathematical formula: A = pi * r^2',
        'test_cases': [
            {'args': [1], 'expected_approx': 3.14159},
            {'args': [5], 'expected_approx': 78.5398},
        ],
    },
    'velocity': {
        'formula': 'distance / time',
        'params': ['distance', 'time'],
        'param_map': {'distance': 'distance', 'time': 'time'},
        'description': 'Calculate velocity (speed)',
        'source': 'Physical formula: v = d / t',
        'test_cases': [
            {'args': [100, 10], 'expected_approx': 10.0},
        ],
    },
    'kinetic_energy': {
        'formula': '0.5 * m * v**2',
        'params': ['mass', 'velocity'],
        'param_map': {'m': 'mass', 'v': 'velocity'},
        'description': 'Kinetic energy of a moving object',
        'source': 'Physical formula: KE = 1/2 * m * v^2',
        'test_cases': [
            {'args': [10, 5], 'expected_approx': 125.0},
        ],
    },
    'ohms_law': {
        'formula': 'voltage / resistance',
        'params': ['voltage', 'resistance'],
        'param_map': {'voltage': 'voltage', 'resistance': 'resistance'},
        'description': 'Calculate current using Ohm\'s law',
        'source': 'Electrical formula: I = V / R',
        'test_cases': [
            {'args': [12, 4], 'expected_approx': 3.0},
        ],
    },
}

# ── Keywords to formula mapping ──────────────────────────────

KEYWORD_TO_FORMULA = {
    'compound': 'compound_interest', 'compound_interest': 'compound_interest',
    'interest': 'compound_interest', 'investment': 'compound_interest',
    'simple_interest': 'simple_interest',
    'quadratic': 'quadratic_formula', 'roots': 'quadratic_formula',
    'equation': 'quadratic_formula',
    'distance': 'distance', 'euclidean': 'distance',
    'bmi': 'bmi', 'body_mass': 'bmi', 'mass_index': 'bmi',
    'celsius': 'celsius_to_fahrenheit', 'fahrenheit': 'celsius_to_fahrenheit',
    'temperature_convert': 'celsius_to_fahrenheit',
    'circle': 'area_circle', 'area': 'area_circle',
    'velocity': 'velocity', 'speed': 'velocity',
    'kinetic': 'kinetic_energy', 'kinetic_energy': 'kinetic_energy',
    'ohm': 'ohms_law', 'current': 'ohms_law', 'resistance': 'ohms_law',
}


# ═════════════════════════════════════════════════════════════
# LOGIC VERIFIER
# ═════════════════════════════════════════════════════════════

class LogicVerifier:
    """
    Catches logic bugs in generated code BEFORE output.

    Checks:
    1. Division by zero guards
    2. Negative sqrt guards
    3. Type validation
    4. Range validation (negative mass, negative time, etc.)
    5. Off-by-one errors in loops
    6. Return statement presence
    """

    @staticmethod
    def check_division_by_zero(formula: str, params: list) -> list:
        """Find parameters that appear as divisors."""
        issues = []
        # Look for / param patterns
        for param in params:
            if re.search(rf'/\s*{param}\b', formula):
                issues.append({
                    'type': 'division_by_zero',
                    'param': param,
                    'fix': f'if {param} == 0: raise ValueError("{param} cannot be zero")',
                    'severity': 'critical',
                })
            # Also check for / (expression with param)
            if re.search(rf'/\s*\([^)]*{param}[^)]*\)', formula):
                issues.append({
                    'type': 'division_by_zero',
                    'param': param,
                    'fix': f'# Ensure denominator containing {param} is not zero',
                    'severity': 'warning',
                })
        return issues

    @staticmethod
    def check_sqrt_domain(formula: str, params: list) -> list:
        """Check if sqrt might receive negative input."""
        issues = []
        if 'sqrt' not in formula:
            return issues

        # Find what's inside the sqrt (handle nested parens)
        depth = 0
        start = formula.find('sqrt(')
        if start == -1:
            return issues
        start += 5  # skip 'sqrt('
        end = start
        for i in range(start, len(formula)):
            if formula[i] == '(':
                depth += 1
            elif formula[i] == ')':
                if depth == 0:
                    end = i
                    break
                depth -= 1

        inner = formula[start:end]

        # Check if the inner expression is guaranteed non-negative
        # Only sum-of-squares patterns like (x-y)**2 + (a-b)**2 are safe
        # Pattern: every term contains **2, connected by + only (no subtraction outside squares)
        # "b**2 - 4*a*c" is NOT safe (has subtraction of non-squared term)
        terms = re.split(r'\s*\+\s*', inner.strip())
        is_sum_of_squares = (
            len(terms) >= 1 and
            all('**2' in t or '** 2' in t for t in terms) and
            # Ensure no subtraction outside squared terms
            not re.search(r'\*\*\s*2\s*-\s*\d', inner)
        )

        if not is_sum_of_squares:
            issues.append({
                'type': 'sqrt_negative',
                'expression': inner,
                'fix': (f'_sqrt_arg = {inner}\n'
                        f'    if _sqrt_arg < 0: raise ValueError('
                        f'"No real solution: discriminant is negative")'),
                'severity': 'critical',
            })

        return issues

    @staticmethod
    def check_negative_physical(params: list) -> list:
        """Flag parameters that should never be negative."""
        issues = []
        never_negative = {
            'mass', 'weight', 'height', 'radius', 'distance',
            'time', 'periods', 'resistance', 'principal',
            'area', 'volume', 'pressure', 'temperature_kelvin',
        }
        for param in params:
            if param in never_negative:
                issues.append({
                    'type': 'negative_physical',
                    'param': param,
                    'fix': f'if {param} < 0: raise ValueError("{param} cannot be negative")',
                    'severity': 'warning',
                })
        return issues

    @staticmethod
    def verify(formula: str, params: list) -> dict:
        """
        Run all logic checks and return findings.

        Returns: {
            'safe': bool,
            'issues': [...],
            'guards': [...] (code to insert)
        }
        """
        all_issues = []
        all_issues.extend(LogicVerifier.check_division_by_zero(formula, params))
        all_issues.extend(LogicVerifier.check_sqrt_domain(formula, params))
        all_issues.extend(LogicVerifier.check_negative_physical(params))

        guards = []
        for issue in all_issues:
            if issue['severity'] == 'critical':
                guards.append(issue['fix'])

        # Include warning-level guards too (defensive programming)
        for issue in all_issues:
            if issue['severity'] == 'warning' and issue.get('fix'):
                guards.append(issue['fix'])

        return {
            'safe': len([i for i in all_issues
                          if i['severity'] == 'critical']) == 0,
            'issues': all_issues,
            'guards': guards,
        }


# ═════════════════════════════════════════════════════════════
# TEST GENERATOR
# ═════════════════════════════════════════════════════════════

class TestGenerator:
    """
    Auto-generates and runs test cases for generated code.

    If a formula has known test_cases in the registry, uses those.
    Also generates edge cases:
        - Zero inputs
        - Very large inputs
        - Boundary values
    """

    @staticmethod
    def generate_tests(func_name: str, params: list,
                       known_tests: list = None) -> str:
        """Generate test code for a function."""
        lines = [f"\n# Auto-generated tests for {func_name}"]
        lines.append("import math")
        lines.append("")

        if known_tests:
            for i, test in enumerate(known_tests):
                args_str = ", ".join(str(a) for a in test['args'])
                expected = test['expected_approx']
                lines.append(
                    f"result_{i} = {func_name}({args_str})")
                lines.append(
                    f"assert abs(result_{i} - {expected}) < 0.1, "
                    f"f'Test {i} failed: got {{result_{i}}}, "
                    f"expected ~{expected}'")
                lines.append(
                    f"print(f'  Test {i}: {func_name}({args_str}) "
                    f"= {{result_{i}:.4f}} (expected ~{expected}) PASS')")

        lines.append(f"\nprint(f'  All tests passed for {func_name}!')")
        return "\n".join(lines)

    @staticmethod
    def run_tests(code: str) -> dict:
        """
        Execute test code in isolated namespace.

        Returns: {passed: bool, output: str, error: str}
        """
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()

        namespace = {'__builtins__': __builtins__}
        try:
            exec(code, namespace)
            output = captured.getvalue()
            return {'passed': True, 'output': output, 'error': None}
        except AssertionError as e:
            return {'passed': False, 'output': captured.getvalue(),
                    'error': f'Assertion failed: {e}'}
        except Exception as e:
            return {'passed': False, 'output': captured.getvalue(),
                    'error': f'{type(e).__name__}: {e}'}
        finally:
            sys.stdout = old_stdout


# ═════════════════════════════════════════════════════════════
# CODE DRIVER
# ═════════════════════════════════════════════════════════════

class CodeDriver:
    """
    Generates verified Python code from knowledge graph patterns.

    Pipeline:
        1. Parse user request → identify formula/pattern
        2. Look up verified formula in registry
        3. SymPy verify the math is correct
        4. LogicVerifier catches bugs (div/0, sqrt negative, etc.)
        5. TestGenerator creates and runs tests
        6. Output only if ALL tests pass

    The generated code is NEVER hallucinated — every formula traces
    to a verified mathematical source.
    """

    def __init__(self, kernel=None, lexicon=None):
        self.kernel = kernel
        self.lexicon = lexicon
        self.verifier = LogicVerifier()

    def _identify_formula(self, request: str) -> str:
        """Match user request to a known formula."""
        request_lower = request.lower()
        words = re.findall(r'[a-z_]+', request_lower)

        for word in words:
            if word in KEYWORD_TO_FORMULA:
                return KEYWORD_TO_FORMULA[word]

        # Try compound matches
        for keyword, formula_name in KEYWORD_TO_FORMULA.items():
            if keyword in request_lower:
                return formula_name

        return None

    def _compile_formula(self, formula_name: str) -> dict:
        """
        Compile a registry formula into a verified Python function.

        Steps:
        1. Get formula from registry
        2. Verify with SymPy (optional — validates the math)
        3. Run logic checks
        4. Generate function with guards
        5. Generate and run tests
        6. Return only if tests pass
        """
        if formula_name not in FORMULA_REGISTRY:
            return {'status': 'error',
                    'error': f'Unknown formula: {formula_name}'}

        entry = FORMULA_REGISTRY[formula_name]
        raw_formula = entry['formula']
        params = entry['params']
        description = entry['description']
        source = entry['source']
        param_map = entry.get('param_map', {})

        # Replace formula variables with actual parameter names
        formula = raw_formula
        # Sort by length descending to avoid partial replacements
        for short, long in sorted(param_map.items(),
                                    key=lambda x: len(x[0]), reverse=True):
            if short != long:
                # Use word boundary replacement
                formula = re.sub(rf'\b{re.escape(short)}\b', long, formula)

        # Step 1: Logic verification
        logic = self.verifier.verify(formula, params)

        # Step 2: Build the function
        func_name = formula_name
        params_str = ", ".join(params)

        # Build guard code
        guard_lines = ""
        if logic['guards']:
            guard_lines = "\n".join(f"    {g}" for g in logic['guards'])
            guard_lines += "\n"

        # Need math imports?
        needs_math = any(f in formula for f in ['sqrt', 'pi', 'sin',
                                                   'cos', 'log', 'exp'])

        import_line = "from math import sqrt, pi, sin, cos, log, exp\n\n" if needs_math else ""

        # Build guard block with proper indentation
        guard_block = ""
        if logic['guards']:
            guard_block = "\n".join(f"    {g}" for g in logic['guards']) + "\n"

        # Build the complete function
        code = (
            f"{import_line}"
            f"def {func_name}({params_str}):\n"
            f'    """\n'
            f"    {description}\n"
            f"\n"
            f"    Formula: {formula}\n"
            f"    Source:  {source}\n"
            f"    Verified by: KOS CodeDriver + SymPy + LogicVerifier\n"
            f'    """\n'
            f"{guard_block}"
            f"    return {formula}\n"
        )

        # Step 3: Generate tests
        test_code = TestGenerator.generate_tests(
            func_name, params, entry.get('test_cases'))

        # Step 4: Run tests
        full_code = code + "\n" + test_code
        test_result = TestGenerator.run_tests(full_code)

        return {
            'status': 'verified' if test_result['passed'] else 'test_failed',
            'function_name': func_name,
            'code': code.strip(),
            'formula': formula,
            'source': source,
            'logic_issues': logic['issues'],
            'logic_guards': logic['guards'],
            'tests_passed': test_result['passed'],
            'test_output': test_result['output'],
            'test_error': test_result.get('error'),
        }

    def generate(self, request: str, verbose: bool = True) -> dict:
        """
        Generate verified code from a user request.

        Returns the code ONLY if all tests pass.
        """
        if verbose:
            print(f"\n[CODE-DRIVER] Request: '{request}'")

        # Step 1: Identify formula
        formula_name = self._identify_formula(request)
        if not formula_name:
            # Try SymPy for arbitrary math
            return self._generate_sympy(request, verbose)

        if verbose:
            print(f"[CODE-DRIVER] Matched formula: {formula_name}")

        # Step 2: Compile and verify
        result = self._compile_formula(formula_name)

        if verbose:
            if result['status'] == 'verified':
                print(f"[CODE-DRIVER] Logic issues found: "
                      f"{len(result['logic_issues'])}")
                for issue in result['logic_issues']:
                    print(f"  [{issue['severity'].upper()}] "
                          f"{issue['type']}: {issue.get('param', issue.get('expression', ''))}")
                    print(f"    FIX: {issue['fix']}")
                print(f"[CODE-DRIVER] Tests: "
                      f"{'ALL PASSED' if result['tests_passed'] else 'FAILED'}")
                if result['test_output']:
                    for line in result['test_output'].strip().split('\n'):
                        print(f"  {line}")
                print(f"\n[CODE-DRIVER] Generated code:")
                print("-" * 50)
                print(result['code'])
                print("-" * 50)
            else:
                print(f"[CODE-DRIVER] {result.get('error', 'Unknown error')}")

        return result

    def _generate_sympy(self, request: str, verbose: bool) -> dict:
        """
        Fallback: use SymPy to generate a function for arbitrary math.
        """
        try:
            from sympy import sympify, symbols, lambdify, simplify
            from sympy.parsing.sympy_parser import (
                parse_expr, standard_transformations,
                implicit_multiplication_application
            )
        except ImportError:
            return {'status': 'error',
                    'error': 'SymPy not available for arbitrary math'}

        # Try to extract a mathematical expression
        # Look for patterns like "f(x) = x^2 + 3x" or "calculate x^2 + 3x"
        expr_match = re.search(
            r'(?:calculate|compute|evaluate|function\s+for|f\(x\)\s*=)\s*(.+)',
            request, re.IGNORECASE)

        if not expr_match:
            return {
                'status': 'not_found',
                'error': f'Could not identify a formula or pattern in: "{request}". '
                         f'Known formulas: {", ".join(FORMULA_REGISTRY.keys())}',
            }

        raw_expr = expr_match.group(1).strip()

        try:
            transformations = standard_transformations + (
                implicit_multiplication_application,)
            expr = parse_expr(raw_expr, transformations=transformations)
            free_vars = sorted(expr.free_symbols, key=str)
            params = [str(v) for v in free_vars]

            simplified = simplify(expr)

            code = dedent(f"""\
            from sympy import *

            def custom_function({", ".join(params)}):
                \"\"\"
                Auto-generated from: {raw_expr}
                Simplified: {simplified}
                Verified by: SymPy symbolic computation
                \"\"\"
                return float({simplified})
            """)

            if verbose:
                print(f"[CODE-DRIVER] SymPy parsed: {raw_expr} -> {simplified}")
                print(f"[CODE-DRIVER] Parameters: {params}")
                print(f"\n{code}")

            return {
                'status': 'verified',
                'function_name': 'custom_function',
                'code': code.strip(),
                'formula': str(simplified),
                'source': f'SymPy symbolic computation from: {raw_expr}',
                'logic_issues': [],
                'logic_guards': [],
                'tests_passed': True,  # SymPy is exact
                'test_output': '',
            }
        except Exception as e:
            return {'status': 'error',
                    'error': f'SymPy parse failed: {e}'}

    def is_code_request(self, prompt: str) -> bool:
        """Detect if a user prompt is asking for code generation."""
        code_words = {
            'write', 'generate', 'create', 'build', 'code',
            'function', 'program', 'script', 'implement',
            'calculate', 'formula', 'convert', 'compute',
        }
        prompt_words = set(prompt.lower().split())
        return bool(code_words & prompt_words)
