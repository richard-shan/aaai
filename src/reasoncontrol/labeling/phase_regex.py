"""Keyword taxonomy for cognitive phases (Venhoff et al.-style, adapted to
R1-distill trace idiom). A judge pass on a stratified subset audits these
labels; kappa is reported and low-kappa triggers judge-only probe training.
"""
from __future__ import annotations

import re

PHASES = ("exploration", "deduction", "verification", "backtracking", "other")

_PATTERNS: dict[str, list[re.Pattern]] = {
    "backtracking": [
        re.compile(p, re.I) for p in [
            r"\bwait\b", r"\bhold on\b", r"\bactually,?\s", r"\bon second thought\b",
            r"\bhmm\b", r"\bno[,.]?\s+that('s| is)? (not|wrong)", r"\bi (made|think i made) (a|an) (mistake|error)",
            r"\blet me (re-?read|reconsider|rethink|start over|go back)",
            r"\bscratch that\b", r"\bthat( is|'s) (incorrect|wrong)",
        ]
    ],
    "verification": [
        re.compile(p, re.I) for p in [
            r"\blet me (check|verify|double-?check|confirm|make sure)",
            r"\b(to )?(check|verify|confirm)(ing)?\b", r"\bdouble-?check", r"\bsanity check",
            r"\bplug(ging)? (it|this|that|in|back)", r"\bsubstitut(e|ing) back",
            r"\bindeed\b", r"\bwhich (checks|matches|agrees)", r"\bconsistent with\b",
            r"\bso that('s| is) correct\b", r"\bthat looks right\b",
        ]
    ],
    "exploration": [
        re.compile(p, re.I) for p in [
            r"\balternatively\b", r"\banother (way|approach|method|idea)",
            r"\bwhat if\b", r"\bsuppose\b", r"\blet('s| us| me) try\b", r"\bmaybe\b",
            r"\bperhaps\b", r"\bconsider\b", r"\bone (approach|option|way)\b",
            r"\bi could\b", r"\bfor example\b", r"\bfor instance\b",
        ]
    ],
    "deduction": [
        re.compile(p, re.I) for p in [
            r"\btherefore\b", r"\bso\b", r"\bthus\b", r"\bhence\b", r"\bit follows\b",
            r"\bwhich (means|gives|implies|yields)\b", r"\bsolving\b", r"\bwe (get|have|find|obtain)\b",
            r"\bthis (gives|means|implies)\b", r"=",
        ]
    ],
}

# priority: distinctive phases win over the (very common) deduction cues
_PRIORITY = ("backtracking", "verification", "exploration", "deduction")


def label_phase(text: str) -> str:
    scores = {ph: sum(1 for p in pats if p.search(text)) for ph, pats in _PATTERNS.items()}
    for ph in _PRIORITY[:3]:
        if scores[ph] > 0:
            return ph
    if scores["deduction"] > 0:
        return "deduction"
    return "other"


VERIFICATION_CUE_COUNT = _PATTERNS["verification"]
BACKTRACK_CUE_COUNT = _PATTERNS["backtracking"]


def cue_counts(text: str) -> dict[str, int]:
    """Shallow features for the probe-vs-shallow-baseline comparison."""
    return {f"cues_{ph}": sum(1 for p in pats if p.search(text))
            for ph, pats in _PATTERNS.items()}
