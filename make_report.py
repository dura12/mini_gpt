import argparse
import json
import math
import os
from pathlib import Path

def validation_verdict(evals: list[dict]) -> str:
    val = [e["val_loss"] for e in evals]
    steps = [e["step"] for e in evals]
    first, final = val[0], val[-1]
    best = min(val)
    best_step = steps[val.index(best)]
    if best_step != steps[-1] and final > best * 1.02:
        return (f"Validation loss improved from {first:.3f} to a minimum of {best:.3f} at step {best_step}, then rose "
                f"to {final:.3f} by step {steps[-1]}. The model started to overfit the training text, so the saved checkpoint is the step-{best_step} one.")
    if final < first * 0.98:
        return (f"Validation loss improved from {first:.3f} (before training) to {final:.3f} at step {steps[-1]} and was still best at the end, so training had not yet converged.")
    return f"Validation loss stayed roughly flat ({first:.3f} before training, {final:.3f} at the end)."

def plot_losses(evals, log, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(7, 4))
    if log:
        ax.plot([r["step"] for r in log], [r["batch_loss"] for r in log], alpha=0.35, label="batch loss (noisy)")
    ax.plot([e["step"] for e in evals], [e["train_loss"] for e in evals], marker="o", label="train loss (eval)")
    ax.plot([e["step"] for e in evals], [e["val_loss"] for e in evals], marker="o", label="validation loss")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return True

def sample_block(samples):
    return "\n\n".join(f"**Prompt:** `{s['prompt']}`\n\n```text\n{s['text']}\n```" for s in samples) or "_none recorded_"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/results.json")
    ap.add_argument("--out", default="REPORT.md")
    a = ap.parse_args()

    r = json.loads(Path(a.results).read_text())
    cfg, hp, evals = r["config"], r["hparams"], r["evals"]
    img = Path(a.results).with_name("loss_curve.png")
    has_img = plot_losses(evals, r.get("log", []), img)
    rel_img = Path(os.path.relpath(img, Path(a.out).resolve().parent)).as_posix()
    last = evals[-1]
    best = min(evals, key=lambda e: e["val_loss"])

    rows = "\n".join(f"| {e['step']} | {e['train_loss']:.4f} | {e['val_loss']:.4f} |" for e in evals)
    md = f"""# Training Report

## Configuration

| setting | value |
|---|---|
| context length | {cfg['context_length']} |
| embedding dimension | {cfg['n_embd']} |
| attention heads | {cfg['n_head']} |
| transformer blocks | {cfg['n_layer']} |
| dropout | {cfg['dropout']} |
| tied embedding/output weights | {cfg['tie_weights']} |
| vocabulary size | {cfg['vocab_size']} |
| trainable parameters | {r['num_parameters']:,} |
| training steps | {hp['steps']} |
| batch size | {hp['batch_size']} |
| optimizer | Adam, peak lr {hp['lr']}, {hp['warmup']} warm-up steps then cosine decay, grad-clip {hp['grad_clip']} |
| seed | {hp['seed']} |
| device | {hp['device']} |
| training tokens / validation tokens | {r['train_tokens']:,} / {r['val_tokens']:,} |
| wall-clock training time | {r['seconds']} s |

## Loss results

Losses are cross-entropy in nats, averaged over {hp['eval_iters']} fixed batches per split (same batches at every evaluation). Step 0 is the untrained model; a model guessing uniformly would score ln({cfg['vocab_size']}) = {math.log(cfg['vocab_size']):.3f}.

| step | train loss | validation loss |
|---|---|---|
{rows}

* Best validation loss: **{best['val_loss']:.4f}** at step {best['step']} (this is the saved checkpoint).
* Final train / validation loss: {last['train_loss']:.4f} / {last['val_loss']:.4f} (gap {last['val_loss'] - last['train_loss']:+.4f}).
* {validation_verdict(evals)}

{f'![loss curves]({rel_img})' if has_img else ''}

## Generated samples (greedy decoding, 40 new tokens, from the saved checkpoint)

### Before training (random weights)

{sample_block(r['samples_before'])}

### After training

{sample_block(r['samples_after'])}
"""
    Path(a.out).write_text(md)
    print(f"wrote {a.out}" + (f" and {img}" if has_img else ""))

if __name__ == "__main__":
    main()
