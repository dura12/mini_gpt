"""Bonus: API smoke tests, using a tiny random checkpoint and the stand-in tokenizer (no downloads)."""
import dataclasses

import pytest
import torch
from fastapi.testclient import TestClient

from tests.helpers import ByteEncoder
from tinygpt.model import GPT, GPTConfig


@pytest.fixture()
def client(tmp_path, monkeypatch):
    cfg = GPTConfig(vocab_size=256, context_length=16, n_embd=32, n_head=4, n_layer=1)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state": GPT(cfg).state_dict(), "config": dataclasses.asdict(cfg)}, path)
    monkeypatch.setenv("CHECKPOINT", str(path))
    import server.main as srv

    monkeypatch.setattr(srv, "get_tokenizer", lambda: ByteEncoder())
    with TestClient(srv.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["context_length"] == 16


def test_generate_returns_text_and_is_deterministic_when_greedy(client):
    body = {"prompt": "hello", "max_new_tokens": 10}
    a, b = client.post("/generate", json=body), client.post("/generate", json=body)
    assert a.status_code == 200 and a.json()["prompt"] == "hello"
    assert isinstance(a.json()["generated"], str) and a.json() == b.json()


def test_generate_validates_input(client):
    assert client.post("/generate", json={"prompt": "x", "max_new_tokens": 100000}).status_code == 422
    assert client.post("/generate", json={"prompt": "x", "temperature": -1}).status_code == 422


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Tiny GPT" in r.text
