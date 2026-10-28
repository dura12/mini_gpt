# Tiny GPT: a small language model trained from scratch

A decoder-only Transformer (GPT-style) built from PyTorch components, trained from random initialisation on
Wikipedia text (WikiText-2) with the GPT-2 tokenizer. Built for the iCog Labs "From ChatGPT to a Tiny GPT" project.

```
tinygpt/data.py       Task 1  tokenizer, dataset loading, (x, y) batches, sanity checks
tinygpt/model.py      Task 2  Embeddings, CausalSelfAttention, TransformerBlock, GPT
tinygpt/train.py      Task 3  training loop, validation, checkpointing, results.json
tinygpt/generate.py   Task 3  autoregressive generation + inference command (no training)
tokenizer_demo.py     Task 1a prints IDs and decoded text for 5 sentences
make_report.py        Task 3e builds REPORT.md + loss curve from outputs/results.json
tests/                Task 1d correctness tests (data, model, training, checkpoint, API)
server/main.py        Task 4  FastAPI backend (loads the checkpoint at start-up)
frontend/index.html   Task 4  prompt box that calls the backend
EXPLAINER.md          study notes: the concepts and how they map to this code
```

## 1. Setup

Python 3.10+.

Create the virtual environment:

```powershell
python -m venv .venv
```

Activate it using the command for your shell:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.venv\Scripts\activate.bat
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

If `pip install torch` pulls a huge CUDA build you do not need, install the CPU build first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

The first run downloads the GPT-2 tokenizer files (tiktoken) and the dataset (Hugging Face), so it needs internet.

## 2. Run everything

```bash
python tokenizer_demo.py                  # Task 1a: IDs <-> text for 5 sentences
python -m pytest -q                       # Task 1d + more: all tests (tokenizer tests need internet once)

# quick pipeline check: proves every step works end to end before the long run
python -m tinygpt.train --steps 200 --eval_interval 100 --out_dir outputs_quick

# the real run (defaults: context 64, embedding 64, 4 heads, 1 block, 3000 steps, batch 32)
python -m tinygpt.train --out_dir outputs

python make_report.py --results outputs/results.json --out REPORT.md      # training report

# generate from the saved checkpoint in a fresh process, without training
python -m tinygpt.generate --checkpoint outputs/checkpoint.pt --prompt "The history of"
python -m tinygpt.generate --checkpoint outputs/checkpoint.pt --prompt "The" --temperature 0.8 --top_k 40
```

Training prints the batch loss every 50 steps and the train/validation loss every 250 steps. `outputs/` will contain:

| file | contents |
|---|---|
| `checkpoint.pt` | weights + model config of the step with the **lowest validation loss** |
| `results.json` | config, hyper-parameters, loss history, before/after generated samples |
| `metrics.csv` | step, train loss, validation loss |
| `loss_curve.png` | created by `make_report.py` |

All model dimensions are command-line flags (`--context_length --n_embd --n_head --n_layer`), e.g. a bigger run
for a Colab GPU (a suggestion, not a tuned setting):

```bash
python -m tinygpt.train --context_length 128 --n_embd 256 --n_head 8 --n_layer 4 --batch_size 64 --steps 5000 --out_dir outputs
```

### Google Colab (GPU)

Runtime -> Change runtime type -> GPU, then in cells:

```bash
!git clone <your-repo-url> && cd tiny-gpt && pip install -q -r requirements.txt
!cd tiny-gpt && python -m tinygpt.train --out_dir outputs
```

Download `outputs/` and commit it, or `git push` from Colab.

## 3. Bonus: backend + frontend

```bash
pip install -r requirements.txt
uvicorn server.main:app --port 8000          # backend: loads outputs/checkpoint.pt once at start-up
# open http://127.0.0.1:8000                 # frontend is served by the same server
```

* `GET /health` returns the loaded model's configuration.
* `POST /generate` with JSON `{"prompt": "The history of", "max_new_tokens": 60, "temperature": 0, "top_k": null}`
  returns `{"prompt": ..., "generated": ...}`.
* Use another checkpoint with `CHECKPOINT=path/to/checkpoint.pt uvicorn server.main:app --port 8000`.
* Backend and frontend are one process on purpose (no CORS or second server to run). The frontend is a plain HTML
  file, so it also works opened separately while the backend is running on port 8000.

## 4. Design decisions

* **Tokenizer:** tiktoken GPT-2 BPE, 50,257 IDs. IDs are stored as `uint16` (max id 50,256 fits).
* **Dataset:** `Salesforce/wikitext` / `wikitext-2-raw-v1`, text from Wikipedia with its own train/validation split.
  Small enough to train on a laptop. Tokenised once and cached in `data/`.
* **Batches:** random windows. Valid start positions are `0 .. len-T-1` so the shifted target never runs past the end.
* **Block layout:** pre-LayerNorm (`x = x + attn(ln(x))`, `x = x + ffn(ln(x))`) like GPT-2; it trains more stably than
  post-LN for small models. FFN is `Linear(C,4C) -> GELU -> Linear(4C,C)`.
* **Attention:** `nn.MultiheadAttention(batch_first=True)` with a boolean upper-triangular mask (True = blocked).
* **Weight tying:** the output head shares the token-embedding matrix. The vocabulary table (50,257 x C) dominates the
  parameter count of a tiny model, so sharing it halves that cost. Set `tie_weights=False` in `GPTConfig` to disable.
* **Optimiser:** Adam, linear warm-up then cosine decay, gradient clipping at 1.0 (all in `train.py`).
* **Honest validation:** every evaluation uses the same fixed batches (`seed=1234`) in `eval()` mode under
  `torch.no_grad()`, so numbers are comparable between steps. Step 0 (untrained) is logged: it should be close to
  ln(50257) = 10.825.
* **Checkpoint = best validation step**, not the last step; the report states which step that was.
* **Generation:** greedy by default (deterministic); temperature and top-k are optional. The context is cropped to the last
  `context_length` tokens at every step.

## 5. Tests

`python -m pytest -q` covers: encode/decode round trips, vocabulary range, input/target shift by exactly one token,
batches never running past the data, the spec's shape examples (`(8,64) -> (8,64,64)`, block shape preserved, logits
`(B,T,V)`), the causal mask (changing a future token does not change earlier logits), configurable dimensions, initial
loss near ln(V), training reducing loss on a learnable pattern, validation not changing weights, checkpoint
save/load giving identical logits, deterministic greedy generation beyond the context length, and the API.

Tests that need the real GPT-2 tokenizer files are **skipped** (shown as `s`) when there is no internet; run them
once on a connected machine and confirm none are skipped before you submit.

## 6. Results

See `REPORT.md` (generated from your own run by `make_report.py`): configuration, train/validation loss table, loss
curve, before/after samples, limitations and improvements. Edit the two closing paragraphs so they describe what
you actually observed.

## 7. Submission checklist

- [ ] `python -m pytest -q` shows no skipped tests
- [ ] trained with `python -m tinygpt.train` (from random init); `outputs/checkpoint.pt` committed
- [ ] `python make_report.py` run; `REPORT.md` and `outputs/loss_curve.png` committed
- [ ] fresh-terminal check: `python -m tinygpt.generate --checkpoint outputs/checkpoint.pt --prompt "The"` works
- [ ] (bonus) `uvicorn server.main:app` works and the page generates text
- [ ] read `EXPLAINER.md` and be able to answer every question in it without looking
