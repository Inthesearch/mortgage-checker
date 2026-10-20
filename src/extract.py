import ollama
from pydantic import BaseModel, Field, ValidationError

MODEL = "llama3.2"


class MortgageApplication(BaseModel):
    """The fields we need from an application. None = not stated in the text."""
    annual_income: float | None = Field(default=None, gt=0)
    property_price: float | None = Field(default=None, gt=0)
    down_payment: float | None = Field(default=None, ge=0)
    contract_rate: float | None = Field(default=None, gt=0, lt=20)
    amortization_years: int | None = Field(default=None, gt=0, le=30)
    property_tax_monthly: float | None = Field(default=None, ge=0)
    heating_monthly: float | None = Field(default=None, ge=0)
    condo_fees_monthly: float | None = Field(default=None, ge=0)
    other_debts_monthly: float | None = Field(default=None, ge=0)

    def missing_fields(self):
        required = ["annual_income", "property_price", "down_payment",
                    "contract_rate", "amortization_years",
                    "property_tax_monthly", "heating_monthly", "other_debts_monthly"]
        return [f for f in required if getattr(self, f) is None]


SYSTEM_PROMPT = """You extract data from Canadian mortgage applications and return JSON.

Rules:
- Only use numbers stated in the application. Never guess.
- If a value is not stated, use null.
- Convert yearly amounts to monthly where the field name says monthly.
- Rates are percentages: "4.5%" -> 4.5.

Example 1 application: "Income $90,000/year, buying for $400,000 with $80,000 down, 5% rate, 25 years, tax $2400/year, heat $80/month, $200/month car loan."
Example 1 output: {"annual_income": 90000, "property_price": 400000, "down_payment": 80000, "contract_rate": 5.0, "amortization_years": 25, "property_tax_monthly": 200, "heating_monthly": 80, "condo_fees_monthly": null, "other_debts_monthly": 200}

Example 2 application: "Applicant earns $100,000 per year. House listed at $450,000, bringing $90,000 down. 4.5% fixed, 30-year amortization. Property taxes $3,000 annually, heating $120 monthly, $300 monthly student loan."
Example 2 output: {"annual_income": 100000, "property_price": 450000, "down_payment": 90000, "contract_rate": 4.5, "amortization_years": 30, "property_tax_monthly": 250, "heating_monthly": 120, "condo_fees_monthly": null, "other_debts_monthly": 300}"""


def extract(application_text: str) -> MortgageApplication:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":  f"Extract fields from this application and return JSON:\n\n{application_text}"},
        ],
        format=MortgageApplication.model_json_schema(),
        options={"temperature": 0},
    )
    return MortgageApplication.model_validate_json(response.message.content)


if __name__ == "__main__":
    sample = """
    Applicant: Jane Doe (fictional). Gross salary $120,000 per year.
    Buying a house listed at $500,000 with $100,000 down.
    Lender offered 4% fixed, 25-year amortization.
    Property tax is $3,600 a year, heating about $100/month.
    Car loan payment of $500 a month. No condo fees.
    """
    try:
        app = extract(sample)
        print(app.model_dump_json(indent=2))
        print("Missing:", app.missing_fields())
    except ValidationError as e:
        print("LLM output failed validation:\n", e)