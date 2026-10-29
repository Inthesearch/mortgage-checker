import pytest

from scratch_evals import contains_all, exact_match, agreement

def test_function():
    assert contains_all(["floor"], "The floor is 5.25.")

def test_function2():
    assert contains_all(["mortgage rate"], "the mortgage        rate is 5%")

def test_function3():
    assert not contains_all(["floor"], "The rate is 5.25 percent.")

def test_exact_match():
    assert exact_match("floor   ", "Floor")

def test_exact_match2():
    assert exact_match("morTgage rate", "Mortgage Rate")

def test_exact_match3():
    assert not exact_match("5.2.5", "5.25")

def test_contains_all():
    assert contains_all([], "anything")

def test_agreement_all():
    human= ["pass", "pass", "fail"]
    judge = ["pass", "pass", "fail"]
    assert agreement(human, judge) == (100, [])

def test_agreement_none():
    human= ["pass", "pass", "pass"]
    judge = ["fail", "fail", "fail"]
    agree_on, disagreed_positions = agreement(human, judge)
    print("agree_on:", agree_on)
    print("disagreed_positions:", disagreed_positions)
    assert agreement(human, judge) == (0,[0,1,2])

def test_agreement_one():
    human= ["fail", "pass", "fail"]
    judge = ["pass", "pass", "fail"]
    agree_on, disagreed_positions = agreement(human, judge)
    print("agree_on:", agree_on)
    print("disagreed_positions:", disagreed_positions)
    assert agreement(human, judge) == (66.67, [0])

def test_agreement_error():
    with pytest.raises(ValueError):
        human= []
        judge = ["pass", "pass", "fail"]
        agree_on, disagreed_positions = agreement(human, judge)
        print("agree_on:", agree_on)
        print("disagreed_positions:", disagreed_positions)
        assert agreement(human, judge)