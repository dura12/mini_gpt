from __future__ import annotations
import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .data import get_batch
from .model import GPT, GPTConfig

DEFAULT_PROMPTS = ["The history of", "In 1990 , the", "The city is located"]

def get_lr(step: int, steps: int, lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

@torch.no_grad()
def estimate_loss(model, splits: dict, batch_size, context_length, eval_iters, device, seed=1234) -> dict:
    was_training = model.training
    model.eval()
    out = {}
    for name, data in splits.items():
        rng = np.random.default_rng(seed)
        losses = [model(*get_batch(data, batch_size, context_length, device, rng))[1].item() for _ in range(eval_iters)]
        out[name] = float(np.mean(losses))
    model.train(was_training)
    return out

def train(cfg: GPTConfig, train_data, val_data, *, steps=2000, batch_size=32, lr=3e-3, warmup=100,
          eval_interval=250, eval_iters=20, log_interval=50, out_dir="outputs", device="cpu",
          seed=1337, grad_clip=1.0, sample_fn=None, verbose=True) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "checkpoint.pt"

    model = GPT(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hparams = dict(steps=steps, batch_size=batch_size, lr=lr, warmup=warmup, eval_interval=eval_interval,
                   eval_iters=eval_iters, grad_clip=grad_clip, seed=seed, device=str(device))
    if verbose:
        print(f"model parameters: {model.num_parameters():,} | config: {asdict(cfg)}")

    samples_before = sample_fn(model) if sample_fn else []
    log, evals = [], []
    best_val = float("inf")
    t0 = time.time()

    def run_eval(step: int):
        nonlocal best_val
        m = estimate_loss(model, {"train": train_data, "val": val_data}, batch_size, cfg.context_length, eval_iters, device)
        evals.append({"step": step, "train_loss": m["train"], "val_loss": m["val"]})
        saved = ""
        if m["val"] < best_val:
            best_val = m["val"]
            torch.save({"model_state": model.state_dict(), "config": asdict(cfg), "step": step,
                        "val_loss": best_val, "hparams": hparams, "tokenizer": "tiktoken-gpt2"}, ckpt_path)
            saved = "  (saved checkpoint)"
        if verbose:
            print(f"[eval] step {step:5d} | train {m['train']:.4f} | val {m['val']:.4f}{saved}")

    run_eval(0)
    model.train()
    for step in range(1, steps + 1):
        for g in optimizer.param_groups:
            g["lr"] = get_lr(step - 1, steps, lr, warmup)
        x, y = get_batch(train_data, batch_size, cfg.context_length, device, rng)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % log_interval == 0:
            log.append({"step": step, "batch_loss": loss.item()})
            if verbose:
                print(f"step {step:5d} | batch loss {loss.item():.4f} | lr {optimizer.param_groups[0]['lr']:.2e} | {time.time() - t0:.0f}s")
        if step % eval_interval == 0 or step == steps:
            run_eval(step)

    seconds = time.time() - t0
    best = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])
    samples_after = sample_fn(model) if sample_fn else []

    results = {
        "config": asdict(cfg), "hparams": hparams, "num_parameters": model.num_parameters(),
        "train_tokens": int(len(train_data)), "val_tokens": int(len(val_data)),
        "seconds": round(seconds, 1), "best_step": best["step"], "best_val_loss": best["val_loss"],
        "log": log, "evals": evals, "samples_before": samples_before, "samples_after": samples_after,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "val_loss"])
        for e in evals:
            w.writerow([e["step"], f"{e['train_loss']:.6f}", f"{e['val_loss']:.6f}"])
    if verbose:
        print(f"done in {seconds:.0f}s | best val {best['val_loss']:.4f} at step {best['step']} -> {ckpt_path}")
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--context_length", type=int, default=64)
    p.add_argument("--n_embd", type=int, default=64)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_layer", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--cache_dir", default="data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    from .data import get_tokenizer, load_token_data, run_sanity_checks
    from .generate import generate_text

    enc = get_tokenizer()
    train_ids, val_ids = load_token_data(a.dataset, a.dataset_config or None, a.cache_dir)
    print(f"train tokens: {len(train_ids):,} | validation tokens: {len(val_ids):,}")
    run_sanity_checks(enc, train_ids, a.batch_size, a.context_length)

    cfg = GPTConfig(vocab_size=enc.n_vocab, context_length=a.context_length, n_embd=a.n_embd,
                    n_head=a.n_head, n_layer=a.n_layer, dropout=a.dropout)

    def sample_fn(model):
        return [{"prompt": pr, "text": generate_text(model, enc, pr, max_new_tokens=40)} for pr in DEFAULT_PROMPTS]

    train(cfg, train_ids, val_ids, steps=a.steps, batch_size=a.batch_size, lr=a.lr, warmup=a.warmup,
          eval_interval=a.eval_interval, eval_iters=a.eval_iters, log_interval=a.log_interval,
          out_dir=a.out_dir, device=a.device, seed=a.seed, sample_fn=sample_fn)

if __name__ == "__main__":
    main()
