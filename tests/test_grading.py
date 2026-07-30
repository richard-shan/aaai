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


def test_extract_answer_marker_math():
    # R1-distill frequently answers with a bold marker and no \boxed{}
    t = "think</think>\n\n**Answer:** The sixth term is 486."
    assert grade(extract_answer(t), "486")
    t2 = r"think</think>\n\n**Answer:** \(3\sqrt{13}\)"
    assert grade(extract_answer(t2), r"3\sqrt{13}")
    # boxed still wins over a marker appearing earlier
    t3 = r"</think> Answer: 7 ... final \boxed{9}"
    assert extract_answer(t3) == "9"


def test_extract_answer_marker_mcq():
    assert extract_answer("</think>\n\nAnswer: C", style="mcq") == "C"
    assert extract_answer("</think>\n\n**Answer: D**", style="mcq") == "D"
    assert extract_answer("</think>\n\nANSWER: A", style="mcq") == "A"
    assert extract_answer("</think>\n\n**Option B** is correct",
                          style="mcq") == "B"


def test_permissive_tier_is_opt_in():
    # truncated before committing: strict finds nothing, permissive recovers
    t = "</think>\n\nSumming: \\[100 + 72 + 64 = 236\\]\n\nThus"
    assert extract_answer(t) is None
    assert grade(extract_answer(t, permissive=True), "236")
    # permissive must not credit an unrelated trailing number
    assert not grade(extract_answer(t, permissive=True), "72")


def test_extraction_does_not_regress_boxed_or_answer_is():
    assert extract_answer(r"</think> the answer is \boxed{42}") == "42"
    assert extract_answer("</think> the answer is 42") == "42"
    assert extract_answer("</think> no answer here at all") is None
