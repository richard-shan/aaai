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


# "Answer: X", "**Answer:** X", "ANSWER - X", "final answer: X" — an explicit
# answer marker, which R1-distill emits instead of \boxed{} in ~20-25% of
# completions. Kept separate from the permissive fallback below so the paper can
# report which extraction tier a number came from.
_ANS_MARKER = re.compile(
    r"(?:final\s+)?answer\s*\**\s*[:\-—]\s*\**\s*(.+)", re.I)
_ANS_MARKER_MCQ = re.compile(
    r"(?:final\s+)?(?:answer|option)\s*(?:is)?\s*\**\s*[:\-—]?\s*"
    r"\**\s*\(?([A-D])\b", re.I)


def _last_line(text: str) -> str | None:
    """Last non-empty line that could carry an answer (has a digit or latex)."""
    for ln in reversed(text.splitlines()):
        if ln.strip() and re.search(r"\d|\\", ln):
            return ln
    return None


def _reduce_to_expression(seg: str) -> str | None:
    """Reduce a prose segment to its trailing math expression.

    'The sixth term is 486.' -> '486'. Returns None when nothing parses, so a
    marker followed by pure prose stays unparsed rather than guessing. Done here
    (not in answers_equal, which also grades convergence labels).
    """
    seg = seg.strip()
    if not seg:
        return None
    try:
        p = parse(seg)
    except Exception:
        return None
    if not p:
        return None
    for item in reversed(p):
        if isinstance(item, str) and item.strip():
            return item.strip()
    return str(p[0])


def extract_answer(text: str, style: str = "math",
                   permissive: bool = False) -> str | None:
    """Final answer from a completed generation (post-</think> region if present).

    Strict tier (default): \\boxed{...}, "answer is X", or an explicit answer
    marker ("Answer: X"). Permissive tier additionally falls back to the last
    mathematical expression on the final line (math-verify picks the trailing
    expression), which recovers completions that were truncated by the answer
    budget before committing. Report the tier alongside any accuracy number.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if style == "mcq":
        # last parenthesized letter wins; bare capitals in prose are too noisy
        m = None
        for m in re.finditer(r"\(([A-D])\)", text):
            pass
        if m:
            return m.group(1)
        m = None
        for m in re.finditer(r"answer is[:\s]*\(?([A-D])\b", text, flags=re.I):
            pass
        if m:
            return m.group(1)
        m = None
        for m in _ANS_MARKER_MCQ.finditer(text):
            pass
        if m:
            return m.group(1)
        m = re.match(r"\s*\(?([A-D])\)?\s*$", text.strip())
        if m:
            return m.group(1)
        if permissive:
            last = _last_line(text)
            m = None
            for m in re.finditer(r"\b([A-D])\b", last or ""):
                pass
            if m:
                return m.group(1)
        return None
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed.strip()
    # "answer is X"
    m = None
    for m in re.finditer(r"answer is[:\s]*([^\n]+)", text, flags=re.I):
        pass
    if m:
        return m.group(1).strip().rstrip(".")
    # explicit answer marker; latex delimiters are preserved so math-verify can
    # parse "\(3\sqrt{13}\)" rather than degrading to "3"
    m = None
    for m in _ANS_MARKER.finditer(text):
        pass
    if m:
        return _reduce_to_expression(m.group(1))
    if permissive:
        last = _last_line(text)
        return _reduce_to_expression(last) if last else None
    return None


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
