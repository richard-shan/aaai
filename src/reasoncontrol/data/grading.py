"""Answer extraction and grading.

Two hard-won rules (see plan risk register):
- Never stop generation on '}' and never compare answers with string equality.
- \boxed{...} contents are extracted by balanced-brace matching, and equivalence
  is judged by math-verify (so 0.5 == \frac{1}{2}).
"""
from __future__ import annotations

import re

from math_verify import parse, verify

_BOXED = re.compile(r"\\boxed\s*\{")
_MCQ = re.compile(r"\(?([A-D])\)?")


def extract_boxed(text: str) -> str | None:
    """Contents of the LAST \\boxed{...} in text, balanced-brace matched."""
    last = None
    for m in _BOXED.finditer(text):
        depth = 1
        i = m.end()
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = text[m.end():i - 1]
    return last


def extract_answer(text: str, style: str = "math") -> str | None:
    """Final answer from a completed generation (post-</think> region if present)."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if style == "mcq":
        m = None
        for m in _MCQ.finditer(text):
            pass
        return m.group(1) if m else None
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed.strip()
    # fallback: last "answer is X" pattern
    m = None
    for m in re.finditer(r"answer is[:\s]*([^\n.]+)", text, flags=re.I):
        pass
    return m.group(1).strip() if m else None


def extract_forced_answer(generated: str, style: str = "math") -> str | None:
    """Answer from a forced-suffix continuation.

    The suffix ends in '\\boxed{' (math) or '(' (mcq); `generated` is the raw
    continuation. Math: read up to the balancing close brace (nested braces ok).
    """
    if style == "mcq":
        m = re.match(r"\s*([A-D])", generated)
        return m.group(1) if m else None
    depth = 1
    for i, ch in enumerate(generated):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                out = generated[:i].strip()
                return out if out else None
    # ran out of tokens before balancing: take up to first newline as best effort
    out = generated.split("\n", 1)[0].strip().rstrip("}").strip()
    return out or None


def grade(pred: str | None, gold: str, style: str = "math") -> bool:
    """math-verify equivalence for math; letter match for MCQ."""
    if pred is None:
        return False
    if style == "mcq":
        return pred.strip().upper() == gold.strip().upper()
    return answers_equal(pred, gold)


def answers_equal(a: str | None, b: str | None) -> bool:
    """Symmetric math-verify equivalence between two answer strings."""
    if a is None or b is None:
        return False
    a, b = a.strip(), b.strip()
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        pa, pb = parse(f"${a}$"), parse(f"${b}$")
        return bool(verify(pb, pa) or verify(pa, pb))
    except Exception:
        return False
