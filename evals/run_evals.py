import json
import sys
from datetime import datetime
from pathlib import Path

from src.agent import run_detailed

CASES_FILE = Path(__file__).parent / "cases.json"
HISTORY_FILE = Path(__file__).parent / "history.jsonl"
TOLERANCE = 0.01  # GDS/TDS may differ by at most 0.01


def check(case, out):
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


def main():
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    only = sys.argv[1] if len(sys.argv) > 1 else None  # optional: run one case by id
    if only:
        cases = [c for c in cases if c["id"] == only]

    passed = 0
    for case in cases:
        try:
            out = run_detailed(case["input"])
            problems = check(case, out)
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
    print(f"\n{passed}/{total} passed ({rate}%)")

    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                            "passed": passed, "total": total, "rate": rate}) + "\n")


if __name__ == "__main__":
    main()