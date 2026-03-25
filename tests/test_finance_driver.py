"""
Test suite for KOS FinanceDriver — Banking-Grade Risk Assessment.

Tests all 12 computation modules against known regulatory values.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kos.drivers.finance import FinanceDriver

fd = FinanceDriver()
passed = 0
failed = 0


def test(name, query, must_contain):
    """Run a single test."""
    global passed, failed
    try:
        result = fd.process(query)
        missing = [s for s in must_contain if s.lower() not in result.lower()]
        if missing:
            print(f"  FAIL  {name}")
            print(f"         Missing: {missing}")
            print(f"         Got: {result[:200]}...")
            failed += 1
        else:
            print(f"  PASS  {name}")
            passed += 1
    except Exception as e:
        print(f"  FAIL  {name} -- Exception: {e}")
        failed += 1


print("=" * 60)
print("KOS FinanceDriver Test Suite")
print("=" * 60)

# 1. VaR
test("VaR - Parametric 99%",
     "Calculate value at risk for 10000000 portfolio with 2% daily volatility at 99% confidence",
     ["value at risk", "10-day var", "465"])

# 2. Capital Ratio
test("Basel III Capital Ratio",
     "What is the CET1 capital ratio for 50000000000 capital and 400000000000 RWA",
     ["capital", "ratio", "%"])

# 3. Capital Requirements Overview
test("Basel III Overview",
     "What are the Basel III capital requirements?",
     ["4.5%", "6.0%", "8.0%", "conservation buffer"])

# 4. RWA
test("Risk-Weighted Assets - Mortgage",
     "Calculate risk weighted assets for 1000000 mortgage exposure",
     ["risk weight", "35%"])

# 5. Credit Risk
test("Credit Risk - EL/UL",
     "Calculate expected loss for PD 2%, LGD 45%, EAD 1000000",
     ["expected loss", "unexpected loss", "irb"])

# 6. Liquidity
test("LCR Calculation",
     "Liquidity coverage ratio for 500000000 HQLA and 400000000 outflows",
     ["lcr", "125", "compliant"])

# 7. Leverage Ratio
test("Leverage Ratio",
     "Calculate leverage ratio for 30000000000 tier 1 capital and 600000000000 exposure",
     ["leverage ratio", "5.0", "compliant"])

# 8. Black-Scholes
test("Black-Scholes Call Option",
     "Black-Scholes option pricing S=100 K=105 r=5 T=1 sigma=20",
     ["call price", "put price", "delta", "gamma"])

# 9. Sharpe Ratio
test("Sharpe Ratio",
     "Calculate sharpe ratio for portfolio return 12% and volatility 15%",
     ["sharpe ratio"])

# 10. EMI
test("EMI Calculation",
     "EMI for 500000 loan at 8.5% for 20 years",
     ["monthly emi", "$"])

# 11. Compound Interest
test("Compound Interest",
     "Future value of 100000 at 7% compound interest for 10 years",
     ["future value", "compound", "$"])

# 12. DTI
test("Debt-to-Income",
     "Debt to income ratio for 2000 monthly debt and 6000 income",
     ["dti", "33"])

# 13. LTV
test("Loan-to-Value",
     "Loan to value for 400000 loan on 500000 property",
     ["ltv", "80"])

# 14. Stress Test
test("Stress Testing",
     "Run stress test on 1000000000 portfolio with 2% default rate",
     ["baseline", "adverse", "severely adverse", "expected loss"])

# 15. Is-Finance Detection
print("\n--- Detection Tests ---")
detect_pass = 0
detect_fail = 0

finance_queries = [
    "What is the value at risk of my portfolio?",
    "Calculate credit risk for a corporate loan",
    "Basel III capital requirements",
    "Black-Scholes option pricing for AAPL",
    "What is the EMI for a home loan?",
    "Run a stress test on our lending book",
]

non_finance_queries = [
    "What is the population of Toronto?",
    "How does photosynthesis work?",
    "Tell me about DNA replication",
]

for q in finance_queries:
    if fd.is_finance_query(q):
        print(f"  PASS  Detected: '{q[:50]}...'")
        detect_pass += 1
    else:
        print(f"  FAIL  Missed:   '{q[:50]}...'")
        detect_fail += 1

for q in non_finance_queries:
    if not fd.is_finance_query(q):
        print(f"  PASS  Rejected: '{q[:50]}...'")
        detect_pass += 1
    else:
        print(f"  FAIL  False+:   '{q[:50]}...'")
        detect_fail += 1

print("\n" + "=" * 60)
total_pass = passed + detect_pass
total_fail = failed + detect_fail
total = total_pass + total_fail
print(f"RESULTS: {total_pass}/{total} passed "
      f"({total_pass/total*100:.0f}%)")
if total_fail > 0:
    print(f"  {total_fail} test(s) FAILED")
else:
    print("  ALL TESTS PASSED")
print("=" * 60)
