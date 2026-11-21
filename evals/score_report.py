"""Print the top score for every retrieval case, plus the rank of the expected chunk."""
import json

from src.retrieval import search_rules

CASES = "evals/retrieval_cases.json"
K = 18  # the whole index: we want the rank even when it is terrible


def load_cases(path=CASES):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def expected_chunk_rank(hits, source, contains):
    """Position (1-based) and score of the first hit matching the expected chunk."""
    for position, hit in enumerate(hits, start=1):
        if hit["source"] != source:
            continue
        text = hit["text"].lower()
        if all(term.lower() in text for term in contains):
            return position, hit["score"]
    return None, None


def report(cases):
    print(f"{'case':<28} {'kind':<12} {'top':>6} {'exp.rank':>9} {'exp.score':>10}")
    for case in cases:
        hits = search_rules(case["input"], K)
        top = hits[0]["score"]
        kind = case.get("category", "?")
        expect = case.get("expect", {})

        if expect.get("refuses"):
            rank_text, score_text = "-", "-"
        else:
            source = expect.get("source") or expect.get("expected_source")
            contains = expect.get("contains") or expect.get("contains_all") or []
            if isinstance(contains, str):
                contains = [contains]
            if source is None:
                rank_text, score_text = "?", "?"
            else:
                rank, score = expected_chunk_rank(hits, source, contains)
                rank_text = str(rank) if rank else "MISS"
                score_text = f"{score:.3f}" if rank else "-"

        print(f"{case['id']:<28} {kind:<12} {top:>6.3f} {rank_text:>9} {score_text:>10}")


if __name__ == "__main__":
    report(load_cases())