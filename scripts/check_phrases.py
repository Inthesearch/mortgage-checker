import json
import pathlib

docs = {p.stem: p.read_text(encoding="utf-8") for p in pathlib.Path("docs").glob("*.txt")}
print("Docs found:", sorted(docs))

cases = json.loads(pathlib.Path("evals/retrieval_cases.json").read_text(encoding="utf-8"))
problems = 0
for c in cases:
    exp = c["expect"]
    if "contains" not in exp:
        continue
    src = exp["source"]
    if src not in docs:
        print(f"NO SOURCE  {c['id']}: '{src}' has no docs/{src}.txt")
        problems += 1
    elif exp["contains"] not in docs[src]:
        print(f"MISSING    {c['id']}: '{exp['contains']}' not in {src}")
        problems += 1

print("All expected phrases found." if problems == 0 else f"{problems} problem(s).")