# modal_serve.py
import os
import modal

# Read defaults at module-import time so `export MODAL_MODEL_ID=...` in
# deploy_model.sh is actually picked up by modal.parameter(default=...).
_DEFAULT_MODEL = os.environ.get("MODAL_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
_DEFAULT_IS_QWEN35 = os.environ.get("MODAL_IS_QWEN35", "false").lower() == "true"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .run_commands(
        "pip install uv",
        # vLLM for non-Qwen3.5 models
        "uv pip install vllm --torch-backend=auto "
        "--extra-index-url https://download.pytorch.org/whl/cu124",
        # SGLang for Qwen3.5 — vLLM has a weight-prefix mismatch with Qwen3.5
        "uv pip install 'sglang[all]' --find-links "
        "https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/",
    )
    .pip_install("huggingface_hub", "transformers")
)

app = modal.App("agentflow-planner")

@app.cls(
    gpu=modal.gpu.A100(memory=80),
    image=image,
    timeout=7200,
    keep_warm=1,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class PlannerServer:
    model_id: str = modal.parameter(default=_DEFAULT_MODEL)
    is_qwen35: bool = modal.parameter(default=_DEFAULT_IS_QWEN35)

    @modal.enter()
    def load_model(self):
        if self.is_qwen35:
            self._load_sglang()
        else:
            self._load_vllm()

    def _load_vllm(self):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        self._backend = "vllm"
        self.tok = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.llm = LLM(
            model=self.model_id,
            dtype="bfloat16",
            max_model_len=32768,
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
        )
        self.sp = SamplingParams(temperature=0.0, max_tokens=2048)

    def _load_sglang(self):
        import subprocess, time, requests as req
        self._backend = "sglang"
        self._sglang_proc = subprocess.Popen([
            "python", "-m", "sglang.launch_server",
            "--model-path", self.model_id,
            "--host", "127.0.0.1", "--port", "30000",
            "--dtype", "bfloat16",
            "--chat-template", "qwen",
            "--disable-radix-cache",
        ])
        # Poll until the SGLang HTTP server is ready (up to 120 s)
        for _ in range(60):
            try:
                if req.get("http://127.0.0.1:30000/health", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise RuntimeError("SGLang server did not become healthy within 120 s")

    @modal.web_endpoint(method="POST", label="chat")
    def chat(self, request: dict) -> dict:
        if self._backend == "sglang":
            return self._chat_sglang(request)
        return self._chat_vllm(request)

    def _chat_vllm(self, request: dict) -> dict:
        prompt = self.tok.apply_chat_template(
            request["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        out = self.llm.generate([prompt], self.sp)
        text = out[0].outputs[0].text
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    def _chat_sglang(self, request: dict) -> dict:
        import requests as req
        resp = req.post(
            "http://127.0.0.1:30000/v1/chat/completions",
            json={
                "model": self.model_id,
                "messages": request["messages"],
                "max_tokens": 2048,
                "temperature": 0.0,
                # disable thinking mode so Planner JSON parsing stays intact
                # SGLang expects chat_template_kwargs at the top level, not inside extra_body
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=180,
        )
        resp.raise_for_status()
        return {"choices": resp.json()["choices"]}
