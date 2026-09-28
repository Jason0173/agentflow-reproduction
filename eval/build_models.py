"""Build the two checkpoints the evaluation serves.

    python build_models.py <base checkpoint> <LoRA adapter dir> <out: base> <out: LoRA>

Both are loaded in float32 with the same code and saved in bfloat16 with the
base model's tokenizer; the second one has the LoRA update merged in (in
float32) first. The two served models therefore differ only by the LoRA
update, and both go through the same MERGED_MODEL path of serve_lora_local.py.

The merge is done here instead of in the server because serve_lora_local.py's
adapter mode fails with the pinned torch 2.6 + peft 0.19: peft looks up a float8
dtype that only exists from torch 2.7. The team served a merged checkpoint too.
"""

import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(path):
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, trust_remote_code=True)


def main(src, adapter, out_base, out_lora):
    with safe_open(str(Path(adapter) / "adapter_model.safetensors"), "pt") as f:
        expected = sum(1 for k in f.keys() if "lora_B" in k)

    base = load(src)
    base.to(torch.bfloat16).save_pretrained(out_base)
    AutoTokenizer.from_pretrained(src, trust_remote_code=True).save_pretrained(out_base)
    del base

    # autocast_adapter_dtype=False avoids the peft/torch float8 incompatibility;
    # the base is float32 here, so the adapter is merged in float32 anyway.
    model = PeftModel.from_pretrained(load(src), adapter, autocast_adapter_dtype=False)
    layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
    loaded = sum(1 for m in layers if m.lora_B["default"].weight.abs().sum().item() > 0)
    if len(layers) != expected or loaded != expected:
        sys.exit(f"LoRA adapter not applied: {loaded} of {len(layers)} layers got weights, file has {expected}")
    merged = model.merge_and_unload()
    merged.to(torch.bfloat16).save_pretrained(out_lora)
    AutoTokenizer.from_pretrained(src, trust_remote_code=True).save_pretrained(out_lora)

    info = {"lora_layers": expected, "lora_layers_loaded": loaded}
    print(json.dumps(info))
    (Path(out_lora) / "merge_info.json").write_text(json.dumps(info))


if __name__ == "__main__":
    main(*sys.argv[1:5])
