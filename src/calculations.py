GDS_LIMIT = 39.0
TDS_LIMIT = 44.0
MQR_FLOOR = 5.25


def _check_positive(**values):
    for name, value in values.items():
        if value is None or value <= 0:
            raise ValueError(f"{name} must be greater than 0, got {value}")


def _check_not_negative(**values):
    for name, value in values.items():
        if value is None or value < 0:
            raise ValueError(f"{name} cannot be negative, got {value}")


def qualifying_rate(contract_rate):
    _check_positive(contract_rate=contract_rate)
    return max(contract_rate + 2, MQR_FLOOR)


def monthly_payment(principal, annual_rate, years):
    _check_positive(principal=principal, annual_rate=annual_rate, years=years)
    r = (1 + annual_rate / 100 / 2) ** (1 / 6) - 1
    n = years * 12
    return principal * r / (1 - (1 + r) ** -n)


def calc_gds(annual_income, payment, property_tax, heating, condo_fees=0):
    _check_positive(annual_income=annual_income)
    _check_not_negative(payment=payment, property_tax=property_tax,
                        heating=heating, condo_fees=condo_fees)
    housing = payment + property_tax + heating + condo_fees / 2
    return housing / (annual_income / 12) * 100


def calc_tds(annual_income, payment, property_tax, heating, other_debts, condo_fees=0):
    _check_not_negative(other_debts=other_debts)
    gds = calc_gds(annual_income, payment, property_tax, heating, condo_fees)
    return gds + other_debts / (annual_income / 12) * 100


def passes_stress_test(gds, tds):
    return gds <= GDS_LIMIT and tds <= TDS_LIMIT


def assess(principal, contract_rate, years, annual_income,
           property_tax, heating, other_debts, condo_fees=0):
    """Full stress-test assessment. This is the function the LLM will call as a tool."""
    rate = qualifying_rate(contract_rate)
    payment = monthly_payment(principal, rate, years)
    gds = calc_gds(annual_income, payment, property_tax, heating, condo_fees)
    tds = calc_tds(annual_income, payment, property_tax, heating, other_debts, condo_fees)
    return {
        "qualifying_rate": round(rate, 2),
        "monthly_payment": round(payment, 2),
        "gds": round(gds, 2),
        "tds": round(tds, 2),
        "gds_limit": GDS_LIMIT,
        "tds_limit": TDS_LIMIT,
        "passes": passes_stress_test(gds, tds),
    }