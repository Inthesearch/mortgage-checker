"""Run the project's evals. One file, two modes.

    python run_evals.py                      # agent evals (the original 20 cases), as before
    python run_evals.py agent                # same as above
    python run_evals.py retrieval            # retrieval-only evals (does search_rules find the right section?)
    python run_evals.py all                  # both modes, one after the other
    python run_evals.py 07_casual_wording    # one case by id (mode is found automatically)
    python run_evals.py retrieval R01_xxx    # one case, mode named explicitly

Each mode has its own cases file and its own history file, so results never get mixed up.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from src.agent import run_detailed

HERE = Path(__file__).parent
TOLERANCE = 0.01  # agent mode: GDS/TDS may differ by at most 0.01
K = 3  # retrieval mode: hit@3
RELEVANCE_THRESHOLD = 0.0  # retrieval mode: TODO set from your data in the Nov 16-22 week.
# At 0.0 the "not covered" cases fail on purpose, until you tune this number.


# ---------- agent mode: the whole agent, end to end ----------

def check_agent(case, out):
    """Compare the agent's output with the expected values. Returns a list of problems."""
    expect = case["expect"]
    problems = []

    if "refuses" in expect:
        if out["refused"] != expect["refuses"]:
            problems.append(f"refused={out['refused']}, expected {expect['refuses']} ({out['reason']})")
        return problems

    if out["refused"]:
        return [f"agent refused unexpectedly: {out['reason']}"]

    result = out["result"]
    if "passes" in expect and result["passes"] != expect["passes"]:
        problems.append(f"passes={result['passes']}, expected {expect['passes']}")
    for key in ("gds", "tds"):
        if key in expect and abs(result[key] - expect[key]) > TOLERANCE:
            problems.append(f"{key}={result[key]}, expected {expect[key]}")
    return problems


def run_agent_case(case):
    return check_agent(case, run_detailed(case["input"]))


# ---------- retrieval mode: search_rules only, no answer generation ----------
# Expected interface, built Nov 9-15 in src/retrieval.py:
#   search_rules(query: str, k: int) -> list[dict]   (best first)
#   each hit: {"source": "osfi_mqr", "section_id": "...", "text": "...", "score": 0.0-1.0}

def check_retrieval(case, hits):
    """Compare the retrieved chunks with what the case expects. Returns a list of problems."""
    expect = case["expect"]

    if "refuses" in expect:
        top_score = hits[0]["score"] if hits else 0.0
        would_refuse = top_score < RELEVANCE_THRESHOLD
        if would_refuse != expect["refuses"]:
            return [f"top score {top_score:.2f}, threshold {RELEVANCE_THRESHOLD}: "
                    f"would_refuse={would_refuse}, expected {expect['refuses']}"]
        return []

    top = hits[:K]
    found = any(
        h["source"] == expect["source"] and expect.get("contains", "") in h["text"]
        for h in top
    )
    if not found:
        got = [f"{h['source']}/{h.get('section_id', '?')}" for h in top]
        return [f"expected source={expect['source']} containing {expect.get('contains')!r} "
                f"in top {K}; got {got}"]
    return []


def run_retrieval_case(case):
    from src.retrieval import search_rules  # imported here so agent mode works before retrieval exists
    return check_retrieval(case, search_rules(case["input"], K))


# ---------- shared runner ----------

MODES = {
    "agent": {"cases": HERE / "cases.json", "history": HERE / "history.jsonl", "run": run_agent_case},
    "retrieval": {"cases": HERE / "retrieval_cases.json", "history": HERE / "retrieval_history.jsonl",
                  "run": run_retrieval_case},
}


def load_cases(mode_name):
    return json.loads(MODES[mode_name]["cases"].read_text(encoding="utf-8"))


def run_mode(mode_name, only=None):
    mode = MODES[mode_name]
    cases = load_cases(mode_name)
    if only:
        cases = [c for c in cases if c["id"] == only]

    print(f"=== {mode_name} ===")
    passed = 0
    for case in cases:
        try:
            problems = mode["run"](case)
        except Exception as e:  # a crash is a failure, not a stop
            problems = [f"CRASH: {type(e).__name__}: {e}"]

        if problems:
            print(f"FAIL  {case['id']}")
            for p in problems:
                print(f"      - {p}")
        else:
            passed += 1
            print(f"PASS  {case['id']}")

    total = len(cases)
    rate = round(100 * passed / total, 1) if total else 0
    print(f"\n{passed}/{total} passed ({rate}%)\n")

    with mode["history"].open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                            "passed": passed, "total": total, "rate": rate}) + "\n")


def main():
    args = sys.argv[1:]
    explicit = bool(args) and (args[0] in MODES or args[0] == "all")
    if explicit:
        first = args.pop(0)
        names = list(MODES) if first == "all" else [first]
    else:
        names = ["agent"]  # default, same as the original script
    only = args[0] if args else None

    if only and not explicit:  # a bare case id: find which mode it belongs to
        names = [n for n in MODES if any(c["id"] == only for c in load_cases(n))]
        if not names:
            print(f"No case with id {only!r} in any mode.")
            return

    for name in names:
        run_mode(name, only)


if __name__ == "__main__":
    main()