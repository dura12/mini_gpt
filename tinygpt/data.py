from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

DEFAULT_DATASET = "Salesforce/wikitext"
DEFAULT_CONFIG = "wikitext-2-raw-v1"

def get_tokenizer():
    import tiktoken
    return tiktoken.get_encoding("gpt2")

def _encode_in_chunks(enc, lines: list[str], lines_per_chunk: int = 2000) -> np.ndarray:
    out: list[int] = []
    for i in range(0, len(lines), lines_per_chunk):
        out.extend(enc.encode_ordinary("".join(lines[i : i + lines_per_chunk])))
    return np.asarray(out, dtype=np.uint16)

def load_token_data(dataset_name: str = DEFAULT_DATASET, dataset_config: str | None = DEFAULT_CONFIG, cache_dir: str = "data"):
    tag = f"{dataset_name}_{dataset_config}".replace("/", "-")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    paths = {s: cache / f"{tag}.{s}.bin" for s in ("train", "validation")}

    if all(p.exists() for p in paths.values()):
        return tuple(np.fromfile(paths[s], dtype=np.uint16) for s in ("train", "validation"))

    from datasets import load_dataset
    ds = load_dataset(dataset_name, dataset_config) if dataset_config else load_dataset(dataset_name)
    enc = get_tokenizer()
    streams = {}
    for split in ("train", "validation"):
        ids = _encode_in_chunks(enc, ds[split]["text"])
        ids.tofile(paths[split])
        streams[split] = ids
    return streams["train"], streams["validation"]

def make_example(data, start: int, context_length: int):
    return data[start : start + context_length], data[start + 1 : start + context_length + 1]

def get_batch(data, batch_size: int, context_length: int, device="cpu", rng=None):
    if len(data) < context_length + 1:
        raise ValueError(f"need at least context_length+1={context_length + 1} tokens, got {len(data)}")
    if rng is None:
        rng = np.random.default_rng()
    starts = rng.integers(0, len(data) - context_length, size=batch_size)
    pairs = [make_example(data, int(s), context_length) for s in starts]
    x = torch.from_numpy(np.stack([p[0] for p in pairs]).astype(np.int64))
    y = torch.from_numpy(np.stack([p[1] for p in pairs]).astype(np.int64))
    return x.to(device), y.to(device)

def run_sanity_checks(enc, train_ids, batch_size: int = 8, context_length: int = 64) -> None:
    examples = [
        "The cat sat on the mat.",
        "Addis Ababa is the capital of Ethiopia.",
        "Language models predict the next token.",
        "Numbers like 12345 and symbols like #@! are tokenised too.",
        "  Leading spaces and\nnewlines matter.",
    ]
    for s in examples:
        assert enc.decode(enc.encode_ordinary(s)) == s
    x, y = get_batch(train_ids, batch_size, context_length, rng=np.random.default_rng(0))
    assert x.shape == y.shape == (batch_size, context_length)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert 0 <= int(x.min()) and int(x.max()) < enc.n_vocab
    assert 0 <= int(y.min()) and int(y.max()) < enc.n_vocab
