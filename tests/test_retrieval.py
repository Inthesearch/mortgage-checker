import pytest

from src.retrieval import cosine_similarity


def test_cosine_identical():
    assert cosine_similarity([3,4], [3,4]) == pytest.approx(1.0)

def test_cosine_perpendicular():
    assert cosine_similarity([3,0], [0,4]) == pytest.approx(0.0)

def test_cosine_opposite():
    assert cosine_similarity([3,4], [-3,-4]) == pytest.approx(-1.0)