"""Task 2 + 3 checks: shapes, causality, configurability, training, checkpoint round trip, generation."""
import math

import numpy as np
import pytest
import torch

from tinygpt.generate import generate_ids, generate_text, load_model
from tinygpt.model import GPT, CausalSelfAttention, Embeddings, GPTConfig, TransformerBlock
from tinygpt.train import estimate_loss, get_lr, train
from tests.helpers import ByteEncoder

SMALL = GPTConfig(vocab_size=256, context_length=16, n_embd=32, n_head=4, n_layer=2)


def test_embedding_shape_matches_spec_example():
    cfg = GPTConfig(vocab_size=1000, context_length=64, n_embd=64, n_head=4)
    out = Embeddings(cfg)(torch.randint(0, 1000, (8, 64)))
    assert out.shape == (8, 64, 64)


def test_block_preserves_shape():
    cfg = GPTConfig(vocab_size=1000, context_length=64, n_embd=64, n_head=4)
    x = torch.randn(8, 64, 64)
    assert TransformerBlock(cfg)(x).shape == (8, 64, 64)


def test_logits_shape_is_batch_context_vocab():
    cfg = GPTConfig(vocab_size=500, context_length=64, n_embd=64, n_head=4, n_layer=1)
    logits, loss = GPT(cfg)(torch.randint(0, 500, (8, 64)))
    assert logits.shape == (8, 64, 500) and loss is None


def test_shorter_sequences_than_context_are_allowed():
    logits, _ = GPT(SMALL)(torch.randint(0, 256, (3, 5)))
    assert logits.shape == (3, 5, 256)


def test_too_long_sequence_is_rejected():
    with pytest.raises(ValueError):
        GPT(SMALL)(torch.randint(0, 256, (1, SMALL.context_length + 1)))


@pytest.mark.parametrize("ctx,emb,heads,layers", [(8, 16, 2, 1), (32, 48, 6, 3), (64, 64, 4, 1), (128, 128, 8, 2)])
def test_configurable_without_code_changes(ctx, emb, heads, layers):
    cfg = GPTConfig(vocab_size=300, context_length=ctx, n_embd=emb, n_head=heads, n_layer=layers)
    logits, _ = GPT(cfg)(torch.randint(0, 300, (2, ctx)))
    assert logits.shape == (2, ctx, 300)


def test_heads_must_divide_embedding_dim():
    with pytest.raises(ValueError):
        GPTConfig(n_embd=64, n_head=5)


def test_causal_mask_blocks_the_future():
    """Changing token t must not change the logits at any position BEFORE t."""
    torch.manual_seed(0)
    model = GPT(SMALL).eval()
    a = torch.randint(0, 256, (1, 12))
    b = a.clone()
    b[0, 7] = (b[0, 7] + 1) % 256
    la, lb = model(a)[0], model(b)[0]
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-6)  # past unaffected
    assert not torch.allclose(la[:, 7:], lb[:, 7:], atol=1e-6)  # present and later positions do change


def test_attention_mask_is_strictly_upper_triangular():
    m = CausalSelfAttention(SMALL).causal_mask
    assert m.dtype == torch.bool and m.shape == (16, 16)
    assert not m.diagonal().any()  # a token may attend to itself
    assert torch.equal(m, torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1))


def test_initial_loss_is_close_to_uniform_guess():
    cfg = GPTConfig(vocab_size=1000, context_length=16, n_embd=32, n_head=4)
    x = torch.randint(0, 1000, (16, 16))
    _, loss = GPT(cfg)(x, x)
    assert abs(loss.item() - math.log(1000)) < 0.3  # untrained model ~ ln(V)


def test_weight_tying_shares_one_matrix():
    m = GPT(SMALL)
    assert m.lm_head.weight is m.embed.tok.weight


def _pattern_data(n=4000):
    return (np.arange(n) % 7 + 10).astype(np.uint16)  # 10,11,...,16,10,11,... (perfectly learnable)


def test_training_reduces_loss_and_writes_outputs(tmp_path):
    data = _pattern_data()
    res = train(SMALL, data, data[:1000], steps=150, batch_size=16, lr=3e-3, warmup=10, eval_interval=50,
                eval_iters=4, log_interval=25, out_dir=tmp_path, verbose=False)
    first, last = res["evals"][0], res["evals"][-1]
    assert abs(first["train_loss"] - math.log(256)) < 0.5
    assert last["train_loss"] < first["train_loss"] - 2.0
    assert last["val_loss"] < first["val_loss"] - 2.0
    for f in ("checkpoint.pt", "results.json", "metrics.csv"):
        assert (tmp_path / f).exists()


def test_validation_does_not_change_weights_or_leave_eval_mode_on():
    model = GPT(SMALL)
    model.train()
    before = [p.clone() for p in model.parameters()]
    estimate_loss(model, {"val": _pattern_data(500)}, 4, 16, 3, "cpu")
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert model.training  # restored


def test_validation_uses_same_batches_each_call():
    model = GPT(SMALL)
    d = {"val": _pattern_data(500)}
    assert estimate_loss(model, d, 4, 16, 3, "cpu") == estimate_loss(model, d, 4, 16, 3, "cpu")


def test_lr_schedule_warms_up_then_decays():
    assert get_lr(0, 100, 1e-3, 10) < get_lr(9, 100, 1e-3, 10) <= 1e-3
    assert get_lr(99, 100, 1e-3, 10) < get_lr(20, 100, 1e-3, 10)


def test_checkpoint_round_trip_gives_identical_logits(tmp_path):
    data = _pattern_data()
    train(SMALL, data, data[:1000], steps=20, batch_size=8, warmup=2, eval_interval=10, eval_iters=2,
          out_dir=tmp_path, verbose=False)
    model = load_model(str(tmp_path / "checkpoint.pt"))
    assert not model.training and model.cfg == SMALL
    ckpt = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    ref = GPT(SMALL).eval()
    ref.load_state_dict(ckpt["model_state"])
    x = torch.randint(0, 256, (2, 16))
    assert torch.equal(model(x)[0], ref(x)[0])


def test_greedy_generation_is_deterministic_and_respects_context():
    model = GPT(SMALL).eval()
    start = torch.randint(0, 256, (1, 5))
    a = generate_ids(model, start, 40)  # 5 + 40 tokens >> context_length 16: must not crash
    b = generate_ids(model, start, 40)
    assert a.shape == (1, 45) and torch.equal(a, b)
    assert torch.equal(a[:, :5], start)  # prompt preserved


def test_sampling_options_run_and_stay_in_vocab():
    model = GPT(SMALL).eval()
    out = generate_ids(model, torch.zeros(1, 1, dtype=torch.long), 20, temperature=0.8, top_k=10)
    assert out.shape == (1, 21) and 0 <= int(out.min()) and int(out.max()) < 256


def test_generation_restores_training_mode():
    model = GPT(SMALL).train()
    generate_ids(model, torch.zeros(1, 1, dtype=torch.long), 3)
    assert model.training


def test_trained_pattern_model_continues_the_pattern(tmp_path):
    data = _pattern_data()
    train(SMALL, data, data[:1000], steps=300, batch_size=16, lr=3e-3, warmup=10, eval_interval=100,
          eval_iters=4, out_dir=tmp_path, verbose=False)
    model = load_model(str(tmp_path / "checkpoint.pt"))
    out = generate_ids(model, torch.tensor([[10, 11, 12, 13]]), 6)[0].tolist()
    assert out == [10, 11, 12, 13, 14, 15, 16, 10, 11, 12]


def test_generate_text_with_stand_in_tokenizer():
    model = GPT(SMALL).eval()
    enc = ByteEncoder()
    full = generate_text(model, enc, "hello", max_new_tokens=8)
    new_only = generate_text(model, enc, "hello", max_new_tokens=8, include_prompt=False)
    assert full.startswith("hello") and isinstance(new_only, str)
    assert isinstance(generate_text(model, enc, "", max_new_tokens=3), str)  # empty prompt works
