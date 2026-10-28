"""Task 1d: pipeline correctness checks."""
import numpy as np
import pytest
import torch

from tinygpt.data import get_batch, make_example, run_sanity_checks


@pytest.fixture(scope="module")
def enc():
    from tinygpt.data import get_tokenizer

    try:
        return get_tokenizer()
    except Exception as e:  # tiktoken downloads its vocabulary on first use
        pytest.skip(f"GPT-2 tokenizer files unavailable ({type(e).__name__}); run this test with internet access")


# ---- tokenizer (needs the real GPT-2 files) --------------------------------
@pytest.mark.parametrize("text", [
    "The cat sat.", "Addis Ababa is the capital of Ethiopia.", "  leading spaces\nand newlines",
    "Numbers 12345 and symbols #@!", "Unbelievably long rarely-seen words",
])
def test_round_trip(enc, text):
    assert enc.decode(enc.encode_ordinary(text)) == text


def test_vocab_size_and_id_range(enc):
    assert enc.n_vocab == 50257
    ids = enc.encode_ordinary("Some ordinary text, with punctuation!")
    assert all(0 <= i < enc.n_vocab for i in ids)


def test_full_sanity_suite_on_real_text(enc):
    text = "The quick brown fox jumps over the lazy dog. " * 200
    ids = np.asarray(enc.encode_ordinary(text), dtype=np.uint16)
    run_sanity_checks(enc, ids, batch_size=4, context_length=16)


# ---- batching (pure numpy/torch: always runs) -------------------------------
DATA = np.arange(1000, dtype=np.uint16)  # token i has value i, so shifts are easy to see


def test_make_example_matches_spec_example():
    x, y = make_example(np.array([10, 20, 30, 40, 50]), 0, 4)
    assert list(x) == [10, 20, 30, 40] and list(y) == [20, 30, 40, 50]


def test_batch_shapes_and_dtype():
    x, y = get_batch(DATA, batch_size=8, context_length=64, rng=np.random.default_rng(0))
    assert x.shape == y.shape == (8, 64)
    assert x.dtype == y.dtype == torch.int64


def test_targets_are_inputs_shifted_by_one():
    x, y = get_batch(DATA, 16, 32, rng=np.random.default_rng(1))
    assert torch.equal(y, x + 1)  # on this data, next token == current token + 1
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_never_runs_past_the_end_of_the_data():
    T = 10
    for seed in range(200):
        x, y = get_batch(DATA, 64, T, rng=np.random.default_rng(seed))
        assert int(y.max()) <= len(DATA) - 1 and int(x.min()) >= 0


def test_minimum_length_data_gives_exactly_one_window():
    data = np.arange(11, dtype=np.uint16)  # T+1 tokens -> exactly one valid start (0)
    x, y = get_batch(data, 5, 10, rng=np.random.default_rng(0))
    assert (x[:, 0] == 0).all() and (y[:, -1] == 10).all()


def test_too_short_data_raises():
    with pytest.raises(ValueError):
        get_batch(np.arange(10, dtype=np.uint16), 2, 10)


def test_same_seed_same_batch():
    a = get_batch(DATA, 4, 8, rng=np.random.default_rng(7))
    b = get_batch(DATA, 4, 8, rng=np.random.default_rng(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
