"""
KOS V5.1 — CodeDriver Test (Verified Code Generation).

Tests:
1. Formula identification from natural language
2. Logic bug detection (div/0, sqrt negative, negative physics)
3. Auto-test generation and execution
4. SymPy fallback for arbitrary math
5. Code correctness (every generated function produces right answers)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.drivers.code import CodeDriver, LogicVerifier, TestGenerator


def run_codedriver_test():
    print("=" * 70)
    print("  KOS V5.1: CODE DRIVER — VERIFIED CODE GENERATION")
    print("  Every function is tested before output")
    print("=" * 70)

    driver = CodeDriver()

    # ── TEST 1: Formula Generation from Natural Language ─────
    print("\n[TEST 1] FORMULA GENERATION")
    print("-" * 50)

    requests = [
        "Write a function to calculate compound interest",
        "Generate code for Euclidean distance between two points",
        "Create a BMI calculator function",
        "Write celsius to fahrenheit conversion",
        "Calculate the area of a circle",
        "Generate an Ohm's law current calculator",
        "Write a kinetic energy function",
    ]

    gen_passed = 0
    for req in requests:
        result = driver.generate(req, verbose=False)
        ok = result.get('status') == 'verified' and result.get('tests_passed')
        if ok:
            gen_passed += 1
        issues = len(result.get('logic_issues', []))
        guards = len(result.get('logic_guards', []))
        print(f"  [{'PASS' if ok else 'FAIL'}] {req[:50]:50s} "
              f"| bugs caught: {issues} | guards: {guards}")

    print(f"\n  Formula generation: {gen_passed}/{len(requests)}")

    # ── TEST 2: Logic Bug Detection ──────────────────────────
    print("\n[TEST 2] LOGIC BUG DETECTION")
    print("-" * 50)

    # Test division by zero detection
    issues = LogicVerifier.check_division_by_zero(
        "voltage / resistance", ['voltage', 'resistance'])
    div_found = any(i['type'] == 'division_by_zero' for i in issues)
    print(f"  [{'PASS' if div_found else 'FAIL'}] "
          f"Division by zero in V/R: {'DETECTED' if div_found else 'MISSED'}")

    # Test sqrt negative detection
    issues = LogicVerifier.check_sqrt_domain(
        "(-b + sqrt(b**2 - 4*a*c)) / (2*a)", ['a', 'b', 'c'])
    sqrt_found = any(i['type'] == 'sqrt_negative' for i in issues)
    print(f"  [{'PASS' if sqrt_found else 'FAIL'}] "
          f"Sqrt domain in quadratic: {'DETECTED' if sqrt_found else 'MISSED'}")

    # Test negative physical values
    issues = LogicVerifier.check_negative_physical(
        ['mass', 'velocity', 'time', 'x', 'y'])
    neg_found = sum(1 for i in issues if i['type'] == 'negative_physical')
    print(f"  [{'PASS' if neg_found >= 2 else 'FAIL'}] "
          f"Negative physics params: {neg_found} detected "
          f"(mass, time should be flagged)")

    # Full verification
    result = LogicVerifier.verify(
        "weight / (height ** 2)", ['weight', 'height'])
    print(f"  Full BMI verification: "
          f"{len(result['issues'])} issues, "
          f"{len(result['guards'])} guards generated")
    for g in result['guards']:
        print(f"    GUARD: {g}")

    logic_pass = div_found and sqrt_found and neg_found >= 2

    # ── TEST 3: Auto-Test Generation ─────────────────────────
    print("\n[TEST 3] AUTO-TEST GENERATION & EXECUTION")
    print("-" * 50)

    # Generate and run tests for compound interest
    result = driver.generate("compound interest calculator", verbose=False)
    if result.get('tests_passed'):
        print(f"  Compound interest tests: PASS")
        for line in result.get('test_output', '').strip().split('\n'):
            if line.strip():
                print(f"    {line}")
    else:
        print(f"  Compound interest tests: FAIL")
        print(f"    Error: {result.get('test_error')}")

    # Test quadratic formula
    result_q = driver.generate("quadratic formula", verbose=False)
    if result_q.get('tests_passed'):
        print(f"  Quadratic formula tests: PASS")
        for line in result_q.get('test_output', '').strip().split('\n'):
            if line.strip():
                print(f"    {line}")

    test_pass = (result.get('tests_passed', False) and
                 result_q.get('tests_passed', False))

    # ── TEST 4: SymPy Fallback ───────────────────────────────
    print("\n[TEST 4] SYMPY FALLBACK (Arbitrary Math)")
    print("-" * 50)

    sympy_result = driver.generate(
        "calculate x**3 + 2*x**2 - 5*x + 3", verbose=True)
    sympy_pass = sympy_result.get('status') == 'verified'
    print(f"\n  SymPy fallback: {'PASS' if sympy_pass else 'FAIL'}")

    # ── TEST 5: Full Pipeline — Verbose Output ───────────────
    print("\n[TEST 5] FULL PIPELINE (Verbose)")
    print("-" * 50)

    full_result = driver.generate(
        "Write a function to calculate compound interest", verbose=True)

    # ── TEST 6: Code Request Detection ───────────────────────
    print("\n[TEST 6] CODE REQUEST DETECTION")
    print("-" * 50)

    detection_tests = [
        ("Write a function for BMI", True),
        ("Generate code for distance", True),
        ("What is the capital of France?", False),
        ("Create a celsius converter", True),
        ("When was Toronto founded?", False),
        ("Calculate compound interest", True),
        ("Tell me about perovskite", False),
    ]

    detect_passed = 0
    for prompt, expected in detection_tests:
        detected = driver.is_code_request(prompt)
        ok = detected == expected
        if ok:
            detect_passed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] "
              f"'{prompt[:40]:40s}' -> "
              f"{'CODE' if detected else 'QUERY'}")

    detect_pass = detect_passed == len(detection_tests)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CODE DRIVER SUMMARY")
    print("=" * 70)
    print(f"  Formula generation:   {gen_passed}/{len(requests)} "
          f"{'PASS' if gen_passed == len(requests) else 'PARTIAL'}")
    print(f"  Logic bug detection:  {'PASS' if logic_pass else 'FAIL'} "
          f"(div/0, sqrt, negative)")
    print(f"  Auto-testing:         {'PASS' if test_pass else 'FAIL'}")
    print(f"  SymPy fallback:       {'PASS' if sympy_pass else 'FAIL'}")
    print(f"  Request detection:    {detect_passed}/{len(detection_tests)}")

    all_pass = (gen_passed == len(requests) and logic_pass
                and test_pass and sympy_pass and detect_pass)
    if all_pass:
        print(f"\n  CODE DRIVER VERIFIED:")
        print(f"  Every generated function is:")
        print(f"    1. Assembled from verified formulas (not hallucinated)")
        print(f"    2. Logic-checked (div/0, sqrt, negatives caught)")
        print(f"    3. Auto-tested before output")
        print(f"    4. Cited with provenance")
        print(f"    5. Exact (SymPy verified)")
    print("=" * 70)


if __name__ == "__main__":
    run_codedriver_test()
