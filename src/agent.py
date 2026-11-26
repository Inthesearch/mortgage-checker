import json
import ollama
from pydantic import BaseModel, Field, ValidationError
from src.calculations import assess
from src.extract import MODEL, extract
from src.retrieval import search_rules, RELEVANCE_THRESHOLD

K = 3

SITES = {
    "osfi_mqr": "osfi-bsif.gc.ca",
    "cmhc_requirements": "cmhc-schl.gc.ca",
}

def _refusal(hits: list[dict]) -> str:
    """The 'documents don't cover this' answer, naming the sites of the chunks we did look at."""
    base = "The saved OSFI and CMHC documents don't cover this."
    if not hits:
        return base
    sites = sorted({SITES[h["source"]] for h in hits if h["source"] in SITES})
    pointer = f" Check {' or '.join(sites)}." if sites else ""
    ids = ", ".join(h["section_id"] for h in hits)
    return f"{base}{pointer}\n\nSources: {ids}"



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

class SearchArgs(BaseModel):
    """Gate for search_rules arguments."""
    query: str = Field(min_length=3)

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
},
{
    "type": "function",
    "function": {
        "name": "search_rules",
        "description": (
            "Look up the saved OSFI and CMHC mortgage rule documents and return "
            "the most relevant passages with their chunk ids. Use this for questions "
            "about rules, definitions, eligibility or policy wording "
            "(for example the stress-test floor, CMHC price limits, minimum down payment rules). "
            "Do NOT use it to calculate anything for a specific application - use `assess` for that."
        ),
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's rule question, in their own words",
                },
            },
        },
    },
}

]

SYSTEM = """You are a mortgage pre-screening assistant.
When given an application, you MUST call the `assess` tool once with the values provided.
Never calculate the ratios yourself. Then explain the result in 3-4 plain sentences:
whether it passes, the GDS and TDS, and (if it fails) which ratio is over the limit."""

ASK_SYSTEM = """You are a mortgage rules assistant.
For any question about rules, definitions or eligibility, you MUST call the
`search_rules` tool once with the user's question.
Answer ONLY from the passages the tool returns. Never add numbers or conditions
that are not in them. Keep it to 2-4 plain sentences.
End your answer with a final line: Sources: <the chunk ids you used>"""

def _run_tool(name: str, args: dict):
    """Run the tool the model asked for. Returns (result, citations)."""
    if name == "assess":
        clean = AssessArgs(**args)
        return assess(**clean.model_dump()), []
    if name == "search_rules":
        clean = SearchArgs(**args)
        hits = search_rules(clean.query, K)
        return hits, [h["section_id"] for h in hits]
    raise ValueError(f"unknown tool: {name}")


def citations_valid(answer_ids: list[str], retrieved_ids: list[str]) -> bool:
    """True only if the answer cites at least one id and every one was retrieved."""
    if not answer_ids:
        return False
    return set(answer_ids) <= set(retrieved_ids)

def run_detailed(application_text: str) -> dict:
    """Runs the agent and returns everything the eval suite needs to check."""
    app = extract(application_text)
    missing = app.missing_fields()
    if missing:
        return {"refused": True, "reason": f"missing {missing}", "result": None,
                "answer": f"Cannot assess: missing {missing}. Please provide these values."}

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
        return {"refused": True, "reason": "model did not call the tool",
                "result": None, "answer": msg.content}

    args = msg.tool_calls[0].function.arguments
    try:
        clean = AssessArgs(**args)
    except ValidationError as e:
        return {"refused": True, "reason": f"invalid tool arguments: {e}",
                "result": None, "answer": "ERROR: model sent invalid tool arguments."}
    result = assess(**clean.model_dump())

    messages.append(msg)
    messages.append({"role": "tool", "name": "assess", "content": json.dumps(result)})
    final = ollama.chat(model=MODEL, messages=messages, options={"temperature": 0})
    return {"refused": False, "reason": None, "result": result,
            "answer": final.message.content}


def run(application_text: str) -> str:
    return run_detailed(application_text)["answer"]

def cited_ids(answer: str) -> list[str]:
    """The chunk ids the model listed after its last `Sources:` marker."""
    _, sep, tail = answer.rpartition("Sources:")
    if not sep:
        return []
    parts = tail.replace("\n", ",").split(",")
    return [p.strip(" .`[]") for p in parts if p.strip(" .`[]")]


def ask_detailed(question: str) -> dict:
    """Answers a rule question from the documents, with citations."""
    messages = [{"role": "system", "content": ASK_SYSTEM},
                {"role": "user", "content": question}]

    response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS,
                           options={"temperature": 0})
    msg = response.message
    if not msg.tool_calls:
        return {"tool": None, "hits": [], "citations": [], "refused": True,
                "answer": _refusal([])}

    call = msg.tool_calls[0].function
    try:
        result, citations = _run_tool(call.name, call.arguments)
    except (ValidationError, ValueError) as e:
        return {"tool": call.name, "hits": [], "citations": [],
                "answer": f"ERROR: bad tool call: {e}"}

    if call.name == "search_rules" and (
            not result or result[0]["score"] < RELEVANCE_THRESHOLD):
        return {"tool": call.name, "hits": result, "citations": citations,
                "refused": True, "answer": _refusal(result)}

    messages.append(msg)
    messages.append({"role": "tool", "name": call.name,
                     "content": json.dumps(result)})
    final = ollama.chat(model=MODEL, messages=messages, options={"temperature": 0})
    answer = final.message.content
    answer_ids = cited_ids(answer)
    return {"tool": call.name, "hits": result, "citations": citations,
            "refused": False, "cited_ids": answer_ids,
            "citations_ok": citations_valid(answer_ids, citations),
            "answer": answer}


def ask(question: str) -> str:
    return ask_detailed(question)["answer"]


if __name__ == "__main__":
    sample = """
    Applicant: Jane Doe (fictional). Gross salary $120,000 per year.
    Buying a house listed at $500,000 with $100,000 down.
    Lender offered 4% fixed, 25-year amortization.
    Property tax is $3,600 a year, heating about $100/month.
    Car loan payment of $500 a month. No condo fees.
    """
    out = run_detailed(sample)
    print("Tool result:", json.dumps(out["result"], indent=2))
    print("\n--- Final answer ---\n")
    print(out["answer"])

