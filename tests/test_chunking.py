import pytest

from reasoncontrol.chunking import BoundaryDetector, chunk_trace, find_boundaries

from util import ToyTokenizer


@pytest.fixture(scope="module")
def tok():
    return ToyTokenizer()


def test_roundtrip_and_alignment(tok):
    text = "Let me check x = 1.\n\nSo the answer is 2.\n\nWait, verify x.\n\nTherefore the answer is 2."
    ids = tok.encode(text)
    prompt_len = 7
    chunks = chunk_trace(ids, tok, prompt_len, min_chunk_tokens=2)
    assert "".join(c.text for c in chunks) == tok.decode(ids)
    for c in chunks:
        seg = ids[c.tok_start - prompt_len:c.tok_end - prompt_len]
        assert tok.decode(seg) == c.text


def test_merged_period_newline_token(tok):
    # ".\n\n" is a single token; boundary must still be detected after it
    ids = tok.encode("So the answer is 2") + [tok.vocab.index(".\n\n")] + tok.encode("Wait")
    bounds = find_boundaries(ids, tok)
    assert ids.index(tok.vocab.index(".\n\n")) in bounds


def test_split_newlines_across_tokens(tok):
    nl = tok.vocab.index("\n")
    ids = tok.encode("Suppose x") + [nl, nl] + tok.encode("Therefore")
    bounds = find_boundaries(ids, tok)
    # boundary after the SECOND "\n" token (tail "\n"+"\n" == "\n\n")
    assert ids.index(nl) + 1 in bounds


def test_triple_newline_run(tok):
    ids = tok.encode("Wait") + [tok.vocab.index("\n\n\n")] + tok.encode("So")
    bounds = find_boundaries(ids, tok)
    assert len(bounds) == 0 or all(b < len(ids) for b in bounds)
    # "\n\n\n" ends with "\n\n" -> boundary right after it
    assert ids.index(tok.vocab.index("\n\n\n")) in bounds


def test_min_chunk_merge(tok):
    nn = tok.vocab.index("\n\n")
    # many tiny segments: all merged into >= min_chunk_tokens chunks (except possibly one)
    ids = (tok.encode("Wait") + [nn]) * 10
    chunks = chunk_trace(ids, tok, 0, min_chunk_tokens=6)
    assert all((c.tok_end - c.tok_start) >= 6 for c in chunks[:-1])
    assert "".join(c.text for c in chunks) == tok.decode(ids)


def test_cap_and_subsample(tok):
    nn = tok.vocab.index("\n\n")
    seg = tok.encode("So the answer is 2 and x and y") + [nn]
    ids = seg * 100
    chunks = chunk_trace(ids, tok, 0, min_chunk_tokens=2, max_chunks=20,
                         keep_first=5, keep_last=3)
    assert len(chunks) <= 20
    assert chunks[0].tok_start == 0


def test_streaming_matches_offline(tok):
    text = "Let me solve.\n\nWait, verify.\n\nSo the answer is 5."
    ids = tok.encode(text)
    det = BoundaryDetector(tok)
    online = [i for i, t in enumerate(ids) if det.push(t)]
    assert online == find_boundaries(ids, tok)
