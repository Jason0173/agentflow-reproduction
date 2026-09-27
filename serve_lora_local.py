"""Local FastAPI planner that mimics the AgentFlow Modal endpoint contract.

Runs on Explorer H200 (or any single CUDA GPU). Serves a POST /chat endpoint
that speaks the same JSON shape as ``agentflow/engine/modal_engine.py``::

    request:  {"messages": [...], "temperature": 0.0}
    response: {"choices": [{"message": {"role": "assistant", "content": "..."}}]}

This replaces the Modal deployment for Step 5 LoRA benchmarking. The five
benchmark runners (bamboogle/2wiki/hotpotqa/musique/gaia) keep using the
``modal-...`` engine prefix from factory.py, but we point
``MODAL_PLANNER_URL`` at this local server instead of modal.com.

Two loading modes:

* ``MERGED`` (recommended, fastest): set ``MERGED_MODEL`` to a HF id or local
  path that already contains merged weights, e.g.
  ``Skypioneer/qwen35-0.8b-agentflow-lora``. transformers 4.47+ natively
  supports ``Qwen3_5ForCausalLM``, so no peft wiring is needed.
* ``ADAPTER``: set ``LORA_DIR`` to a PEFT adapter directory (containing
  ``adapter_config.json``) and ``BASE_MODEL`` to the base checkpoint. We load
  the base in bf16 and attach the adapter via peft.

If ``MERGED_MODEL`` is set it wins; otherwise we fall back to adapter mode.

Environment variables (all optional):
    MERGED_MODEL     HF id or local path of a merged checkpoint. When set,
                     BASE_MODEL and LORA_DIR are ignored.
    BASE_MODEL       HF id or local path for the base model (adapter mode).
                     Default: "Qwen/Qwen3.5-0.8B"
    LORA_DIR         Directory containing adapter_config.json + tokenizer.
                     Default: "results/final_qwen35_lora"
    HOST / PORT      Bind address. Default 127.0.0.1:8765
    MAX_NEW_TOKENS   Generation cap. Default 2048
    DTYPE            "bfloat16" (H200 native) or "float16". Default bfloat16
    ENABLE_THINKING  "true"/"false". Forwarded to Qwen3.5 chat template.
                     Default "false" (matches legacy modal_serve.py:111).
    MERGE_ADAPTER    "true"/"false". Adapter mode only. If true, merge LoRA
                     weights for faster inference. Default "true".
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
log = logging.getLogger("serve_lora_local")

MERGED_MODEL = os.environ.get("MERGED_MODEL", "").strip()
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-0.8B")
LORA_DIR = os.environ.get("LORA_DIR", "results/final_qwen35_lora")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8765"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))
DTYPE = os.environ.get("DTYPE", "bfloat16").lower()
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "false").lower() == "true"
MERGE_ADAPTER = os.environ.get("MERGE_ADAPTER", "true").lower() == "true"

STATE: Dict[str, Any] = {}
# torch.generate is not thread-safe for a single model; serialize concurrent
# HTTP requests so the benchmark can still use parallel=N on the client side.
GEN_LOCK = threading.Lock()


def _resolve_dtype() -> torch.dtype:
    if DTYPE == "bfloat16":
        return torch.bfloat16
    if DTYPE == "float16":
        return torch.float16
    raise ValueError(f"Unsupported DTYPE={DTYPE!r}, use bfloat16 or float16")


def _load_merged(dtype: torch.dtype) -> tuple:
    log.info("MERGED mode: loading %s in %s", MERGED_MODEL, dtype)
    tok = AutoTokenizer.from_pretrained(MERGED_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL,
        torch_dtype=dtype,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    return tok, model


def _load_adapter(dtype: torch.dtype) -> tuple:
    # Imported here so MERGED-mode users don't need peft installed.
    from peft import PeftModel

    log.info("ADAPTER mode: tokenizer from %s", LORA_DIR)
    try:
        tok = AutoTokenizer.from_pretrained(LORA_DIR, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("Falling back to base tokenizer at %s: %s", BASE_MODEL, exc)
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    log.info("ADAPTER mode: base %s in %s", BASE_MODEL, dtype)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="cuda",
        trust_remote_code=True,
    )

    log.info("ADAPTER mode: applying LoRA from %s", LORA_DIR)
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()

    if MERGE_ADAPTER:
        try:
            model = model.merge_and_unload()
            log.info("Merged LoRA weights into base (faster inference)")
        except Exception as exc:  # noqa: BLE001
            log.warning("merge_and_unload failed, keeping adapter mode: %s", exc)

    return tok, model


def load_model() -> None:
    dtype = _resolve_dtype()

    if MERGED_MODEL:
        tok, model = _load_merged(dtype)
        src_label = f"merged={MERGED_MODEL}"
    else:
        tok, model = _load_adapter(dtype)
        src_label = f"base={BASE_MODEL} + lora={LORA_DIR}"

    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    STATE["tok"] = tok
    STATE["model"] = model
    STATE["device"] = next(model.parameters()).device
    log.info(
        "Server ready: %s | device=%s | dtype=%s | thinking=%s",
        src_label, STATE["device"], dtype, ENABLE_THINKING,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    load_model()
    yield


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    temperature: float = 0.0
    max_tokens: Optional[int] = None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": "model" in STATE,
        "mode": "merged" if MERGED_MODEL else "adapter",
        "merged_model": MERGED_MODEL or None,
        "base_model": None if MERGED_MODEL else BASE_MODEL,
        "lora_dir": None if MERGED_MODEL else LORA_DIR,
    }


@app.post("/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    tok = STATE["tok"]
    model = STATE["model"]
    device = STATE["device"]

    messages = list(req.messages)
    # ModalEngine.generate() guarantees at least one user message; replicate
    # that defensive normalization here so we stay wire-compatible.
    if not any(m.get("role") == "user" for m in messages):
        fallback = messages[-1].get("content", "") if messages else ""
        messages = [{"role": "user", "content": fallback}]

    template_kwargs: Dict[str, Any] = {}
    if not ENABLE_THINKING:
        template_kwargs["enable_thinking"] = False

    encoded = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        **template_kwargs,
    )
    # In transformers 5.x, apply_chat_template with return_tensors="pt"
    # returns a BatchEncoding (dict-like) rather than a plain tensor.
    # .generate() still expects a Tensor or will call .shape directly and
    # trip BatchEncoding.__getattr__ -> AttributeError. Normalize to tensor.
    if hasattr(encoded, "input_ids"):
        prompt_ids = encoded["input_ids"].to(device)
    else:
        prompt_ids = encoded.to(device)

    max_new = req.max_tokens or MAX_NEW_TOKENS
    do_sample = req.temperature > 0.0
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new,
        "do_sample": do_sample,
        "pad_token_id": tok.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = req.temperature

    with GEN_LOCK, torch.inference_mode():
        out = model.generate(prompt_ids, **gen_kwargs)

    new_tokens = out[0][prompt_ids.shape[-1] :]
    text = tok.decode(new_tokens, skip_special_tokens=True)

    return {
        "choices": [
            {"message": {"role": "assistant", "content": text}}
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
