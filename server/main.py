import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from tinygpt.data import get_tokenizer
from tinygpt.generate import generate_text, load_model

ROOT = Path(__file__).resolve().parent.parent
resources: dict = {}
lock = threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.environ.get("CHECKPOINT", str(ROOT / "outputs" / "checkpoint.pt"))
    model = load_model(path, device="cpu")
    enc = get_tokenizer()
    if model.cfg.vocab_size != enc.n_vocab:
        raise RuntimeError(f"checkpoint vocab {model.cfg.vocab_size} != tokenizer vocab {enc.n_vocab}")
    resources.update(model=model, enc=enc, checkpoint=path)
    yield
    resources.clear()

app = FastAPI(title="Tiny GPT", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class GenerateRequest(BaseModel):
    prompt: str = Field("", max_length=2000)
    max_new_tokens: int = Field(60, ge=1, le=300)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(None, ge=1)

@app.get("/health")
def health():
    cfg = resources["model"].cfg
    return {"status": "ok", "checkpoint": resources["checkpoint"], "context_length": cfg.context_length,
            "n_embd": cfg.n_embd, "n_head": cfg.n_head, "n_layer": cfg.n_layer}

@app.post("/generate")
def generate(req: GenerateRequest):
    with lock:
        text = generate_text(resources["model"], resources["enc"], req.prompt, req.max_new_tokens,
                             req.temperature, req.top_k, include_prompt=False)
    return {"prompt": req.prompt, "generated": text}

app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
