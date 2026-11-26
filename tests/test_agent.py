# tests/test_agent.py
from src.agent import citations_valid, cited_ids, _refusal


def _hit(source, section_id):
    """Only the two keys _refusal reads; the real hits also carry text and score."""
    return {"source": source, "section_id": section_id}


# --- citations_valid (CP-T3.6b) ---

def test_all_cited_ids_were_retrieved():
    assert citations_valid(["osfi_mqr-004"], ["osfi_mqr-004", "osfi_mqr-002"]) is True


def test_invented_id_fails():
    assert citations_valid(["osfi_mqr-999"], ["osfi_mqr-004"]) is False


def test_one_real_one_invented_fails():
    assert citations_valid(["osfi_mqr-004", "cmhc_requirements-099"],
                          ["osfi_mqr-004"]) is False


def test_no_citations_fails():
    assert citations_valid([], ["osfi_mqr-004"]) is False


# --- _refusal (T3.7) ---

def test_refusal_no_hits_has_no_sources_line():
    assert _refusal([]) == "The saved OSFI and CMHC documents don't cover this."


def test_refusal_one_source_names_one_site():
    answer = _refusal([_hit("osfi_mqr", "osfi_mqr-007"),
                       _hit("osfi_mqr", "osfi_mqr-002")])
    assert "Check osfi-bsif.gc.ca." in answer
    assert "Sources: osfi_mqr-007, osfi_mqr-002" in answer


def test_refusal_both_sources_names_both_sites():
    answer = _refusal([_hit("cmhc_requirements", "cmhc_requirements-003"),
                       _hit("osfi_mqr", "osfi_mqr-002")])
    assert "Check cmhc-schl.gc.ca or osfi-bsif.gc.ca." in answer


# --- cited_ids (T3.7) ---

def test_cited_ids_at_end_of_sentence():
    assert cited_ids("The limit is $1.5M. Sources: cmhc_requirements-003") == \
        ["cmhc_requirements-003"]


def test_cited_ids_on_own_line():
    assert cited_ids("The limit is $1.5M.\n\nSources: osfi_mqr-004, osfi_mqr-002") == \
        ["osfi_mqr-004", "osfi_mqr-002"]


def test_cited_ids_no_marker_is_empty():
    assert cited_ids("I don't know.") == []


def test_cited_ids_takes_the_last_marker():
    assert cited_ids("Sources: see below.\n\nSources: osfi_mqr-004") == ["osfi_mqr-004"]