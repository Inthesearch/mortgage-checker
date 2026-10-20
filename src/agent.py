import json
import ollama
from src.calculations import assess
from src.extract import MODEL, extract
from pydantic import BaseModel, Field, ValidationError

TOOLS = [{
    "type": "function",
    "function": {
        "name": "assess",
        "description": "Run the Canadian mortgage stress test. Returns qualifying rate, monthly payment, GDS, TDS and pass/fail.",
        "parameters": {
            "type": "object",
            "required": ["principal", "contract_rate", "years", "annual_income",
                         "property_tax", "heating", "other_debts"],
            "properties": {
                "principal": {"type": "number", "description": "Loan amount: property price minus down payment"},
                "contract_rate": {"type": "number", "description": "Contract interest rate as a percent, e.g. 4.0"},
                "years": {"type": "integer", "description": "Amortization in years"},
                "annual_income": {"type": "number"},
                "property_tax": {"type": "number", "description": "Monthly"},
                "heating": {"type": "number", "description": "Monthly"},
                "other_debts": {"type": "number", "description": "Monthly non-mortgage debts"},
                "condo_fees": {"type": "number", "description": "Monthly, 0 if none"},
            },
        },
    },
}]

SYSTEM = """You are a mortgage pre-screening assistant.
When given an application, you MUST call the `assess` tool once with the values provided.
Never calculate the ratios yourself. Then explain the result in 3-4 plain sentences:
whether it passes, the GDS and TDS, and (if it fails) which ratio is over the limit."""

class AssessArgs(BaseModel):
    """Gate for the model's tool-call arguments: converts types and rejects bad values."""
    principal: float = Field(gt=0)
    contract_rate: float = Field(gt=0, lt=20)
    years: int = Field(gt=0, le=30)
    annual_income: float = Field(gt=0)
    property_tax: float = Field(ge=0)
    heating: float = Field(ge=0)
    other_debts: float = Field(ge=0)
    condo_fees: float = Field(default=0, ge=0)


def run(application_text: str) -> str:
    app = extract(application_text)
    missing = app.missing_fields()
    if missing:
        return f"Cannot assess: missing {missing}. Please provide these values."

    principal = app.property_price - app.down_payment
    user_msg = (
        f"Assess this application:\n"
        f"principal={principal}, contract_rate={app.contract_rate}, years={app.amortization_years}, "
        f"annual_income={app.annual_income}, property_tax={app.property_tax_monthly}, "
        f"heating={app.heating_monthly}, other_debts={app.other_debts_monthly}, "
        f"condo_fees={app.condo_fees_monthly or 0}"
    )

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_msg}]

    response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS,
                           options={"temperature": 0})
    msg = response.message

    if not msg.tool_calls:
        return "ERROR: model did not call the tool."

    call = msg.tool_calls[0]
    args = call.function.arguments
    print("Tool called with:", json.dumps(args, indent=2))
    try:
        clean = AssessArgs(**args)
    except ValidationError as e:
        return f"ERROR: model sent invalid tool arguments: {e}"
    result = assess(**clean.model_dump())
    print("Tool returned:", json.dumps(result, indent=2))

    messages.append(msg)
    messages.append({"role": "tool", "name": "assess",
                     "content": json.dumps(result)})

    final = ollama.chat(model=MODEL, messages=messages, options={"temperature": 0})
    return final.message.content


if __name__ == "__main__":
    sample = """
    Applicant: Jane Doe (fictional). Gross salary $120,000 per year.
    Buying a house listed at $500,000 with $100,000 down.
    Lender offered 4% fixed, 25-year amortization.
    Property tax is $3,600 a year, heating about $100/month.
    Car loan payment of $500 a month. No condo fees.
    """
    print("\n--- Final answer ---\n")
    print(run(sample))