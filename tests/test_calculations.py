import pytest
from src.calculations import (
    qualifying_rate, monthly_payment, calc_gds, calc_tds,
    passes_stress_test, assess,
)

# (name, principal, contract_rate, years, income, tax, heat, condo, debts,
#  expected: qual_rate, payment, gds, tds, passes)
CASES = [
    ("A_normal",   400000, 4.0, 25, 120000, 300, 100, 0,   500, 6.0,  2559.23, 29.59, 34.59, True),
    ("B_floor",    400000, 3.0, 25, 120000, 300, 100, 0,   500, 5.25, 2383.67, 27.84, 32.84, True),
    ("C_fails",    500000, 4.5, 25, 100000, 350, 120, 400, 800, 6.5,  3349.12, 48.23, 57.83, False),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_assess_known_cases(case):
    (_, principal, rate, years, income, tax, heat, condo, debts,
     exp_rate, exp_pmt, exp_gds, exp_tds, exp_pass) = case
    result = assess(principal, rate, years, income, tax, heat, debts, condo)
    assert result["qualifying_rate"] == pytest.approx(exp_rate)
    assert result["monthly_payment"] == pytest.approx(exp_pmt, abs=0.01)
    assert result["gds"] == pytest.approx(exp_gds, abs=0.01)
    assert result["tds"] == pytest.approx(exp_tds, abs=0.01)
    assert result["passes"] is exp_pass


def test_qualifying_rate_adds_two_percent():
    assert qualifying_rate(4.0) == 6.0


def test_qualifying_rate_uses_floor():
    assert qualifying_rate(3.0) == 5.25


def test_qualifying_rate_at_floor_boundary():
    assert qualifying_rate(3.25) == 5.25


def test_monthly_payment_case_a():
    assert monthly_payment(400000, 6.0, 25) == pytest.approx(2559.23, abs=0.01)


def test_gds_condo_counts_half():
    with_condo = calc_gds(120000, 2000, 0, 0, condo_fees=400)
    without = calc_gds(120000, 2000, 0, 0)
    assert with_condo - without == pytest.approx(2.0)  # 200 of 10,000 monthly


def test_tds_is_gds_plus_debts():
    gds = calc_gds(120000, 2000, 300, 100)
    tds = calc_tds(120000, 2000, 300, 100, other_debts=1000)
    assert tds - gds == pytest.approx(10.0)  # 1,000 of 10,000 monthly


# Boundaries
@pytest.mark.parametrize("gds,tds,expected", [
    (39.0, 44.0, True),     # exactly at both limits passes
    (39.01, 40.0, False),   # just over GDS
    (30.0, 44.01, False),   # just over TDS
    (0.0, 0.0, True),
])
def test_stress_test_boundaries(gds, tds, expected):
    assert passes_stress_test(gds, tds) is expected


# Invalid input
def test_zero_income_raises():
    with pytest.raises(ValueError):
        calc_gds(0, 2000, 300, 100)


def test_negative_loan_raises():
    with pytest.raises(ValueError):
        monthly_payment(-400000, 6.0, 25)


def test_zero_years_raises():
    with pytest.raises(ValueError):
        monthly_payment(400000, 6.0, 0)


def test_negative_debts_raises():
    with pytest.raises(ValueError):
        calc_tds(120000, 2000, 300, 100, other_debts=-500)


def test_none_value_raises():
    with pytest.raises(ValueError):
        calc_gds(None, 2000, 300, 100)