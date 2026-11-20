import json
from pathlib import Path

HISTORY = Path("evals/retrieval_history.jsonl")


def read_runs(path=HISTORY):
    runs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def print_summary(runs, last=5):
    print(f"{'time':<22}{'passed':>7}{'total':>7}{'rate':>8}")
    for r in runs[-last:]:
        print(f"{r['time']:<22}{r['passed']:>7}{r['total']:>7}{r['rate']:>8}")
    if runs:
        rates = [r["rate"] for r in runs]
        print(f"\nbest rate:  {max(rates)}")
        print(f"worst rate: {min(rates)}")


if __name__ == "__main__":
    retrieval_runs = [r for r in read_runs() if r["total"] == 15]
    print_summary(retrieval_runs)