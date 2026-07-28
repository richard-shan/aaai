from reasoncontrol.data.grading import (answers_equal, extract_answer,
                                        extract_boxed, extract_forced_answer,
                                        grade)


def test_extract_boxed_nested():
    assert extract_boxed(r"the answer \boxed{\frac{3}{4}} done") == r"\frac{3}{4}"
    assert extract_boxed(r"\boxed{\sqrt{2}}") == r"\sqrt{2}"
    assert extract_boxed(r"first \boxed{1} then \boxed{{2, 3}}") == "{2, 3}"
    assert extract_boxed("no box") is None


def test_forced_answer_nested_braces():
    # suffix ended with '\\boxed{'; continuation balances braces
    assert extract_forced_answer(r"\frac{3}{4}}. So") == r"\frac{3}{4}"
    assert extract_forced_answer("42}") == "42"
    assert extract_forced_answer(r"\sqrt{2}} and more") == r"\sqrt{2}"
    # never truncate at the FIRST '}' inside nesting
    assert extract_forced_answer(r"{2, 3}}") == "{2, 3}"


def test_forced_answer_unbalanced_fallback():
    assert extract_forced_answer("42\nmore text") == "42"
    assert extract_forced_answer("") is None


def test_forced_answer_mcq():
    assert extract_forced_answer("B) because", style="mcq") == "B"
    assert extract_forced_answer("  C", style="mcq") == "C"


def test_math_verify_equivalence():
    assert answers_equal("0.5", r"\frac{1}{2}")
    assert answers_equal("1/2", r"\frac{1}{2}")
    assert not answers_equal("0.5", "0.6")
    assert not answers_equal(None, "1")


def test_grade_and_extract_answer():
    text = r"reasoning ... </think> The final answer is \boxed{\frac{1}{2}}."
    assert extract_answer(text) == r"\frac{1}{2}"
    assert grade(extract_answer(text), "0.5")
    assert grade("B", "b", style="mcq")
