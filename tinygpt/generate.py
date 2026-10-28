from __future__ import annotations
import argparse
import torch
from .model import GPT, GPTConfig

@torch.no_grad()
def generate_ids(model: GPT, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.0, top_k: int | None = None):
    was_training = model.training
    model.eval()
    ctx_len = model.cfg.context_length
    for _ in range(max_new_tokens):
        context = idx[:, -ctx_len:]
        logits, _ = model(context)
        last = logits[:, -1, :]
        if temperature <= 0:
            nxt = last.argmax(dim=-1, keepdim=True)
        else:
            last = last / temperature
            if top_k is not None:
                kth = torch.topk(last, min(top_k, last.size(-1))).values[:, [-1]]
                last = last.masked_fill(last < kth, float("-inf"))
            nxt = torch.multinomial(torch.softmax(last, dim=-1), num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
    model.train(was_training)
    return idx

def generate_text(model: GPT, enc, prompt: str, max_new_tokens: int = 60, temperature: float = 0.0, top_k: int | None = None, include_prompt: bool = True) -> str:
    device = next(model.parameters()).device
    ids = enc.encode_ordinary(prompt) or [enc.eot_token]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate_ids(model, idx, max_new_tokens, temperature, top_k)[0].tolist()
    return enc.decode(out if include_prompt else out[len(ids):])

def load_model(checkpoint_path: str, device: str = "cpu") -> GPT:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/checkpoint.pt")
    p.add_argument("--prompt", default="The")
    p.add_argument("--max_new_tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    from .data import get_tokenizer
    model = load_model(a.checkpoint, a.device)
    enc = get_tokenizer()
    if model.cfg.vocab_size != enc.n_vocab:
        raise SystemExit(f"checkpoint vocab {model.cfg.vocab_size} != tokenizer vocab {enc.n_vocab}")
    print(generate_text(model, enc, a.prompt, a.max_new_tokens, a.temperature, a.top_k))

if __name__ == "__main__":
    main()
