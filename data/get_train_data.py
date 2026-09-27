import os
import yaml
import sys
import modal
import re

# -----------------------------------------------------------------------------
# 1. Define Modal App and Comprehensive Cloud Image
# -----------------------------------------------------------------------------
app = modal.App("agentflow-train-0.8b")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "peft", "trl", "accelerate", "datasets",
        "pyyaml", "pydantic", "numpy", "pandas", "requests", "aiohttp",
        "httpx", "fastapi", "uvicorn", "flask", "starlette", "agentops",
        "opentelemetry-api", "opentelemetry-sdk", "opentelemetry-semantic-conventions",
        "opentelemetry-exporter-otlp-proto-http", "opentelemetry-instrumentation",
        "openai", "tiktoken", "wandb", "jinja2", "setproctitle",
        "omegaconf", "hydra-core", "ray", "vllm", "codetiming", "deepspeed",
        "verl==0.6.0", "pyarrow", "sentencepiece"
    )
    .add_local_dir("./agentflow", remote_path="/root/agentflow")
    .add_local_dir("./data", remote_path="/data")
)


# -----------------------------------------------------------------------------
# 2. Cloud Execution Logic
# -----------------------------------------------------------------------------
@app.function(image=image, gpu="L40S", timeout=1200)
def run_cloud_training(env_vars, python_args):
    print("Cloud execution started on L40S GPU.")

    # Final path and env setup
    os.environ["BASE_DATA_DIR"] = "/data"
    for key, value in env_vars.items():
        os.environ[key] = str(value)

    try:
        # Construct arguments
        sys.argv = ["agentflow.verl"] + python_args

        import runpy
        runpy.run_module("agentflow.verl", run_name="__main__", alter_sys=True)
        print("Training successfully finished.")
    except Exception as e:
        print(f"Cloud execution failure: {e}")
        raise e


# -----------------------------------------------------------------------------
# 3. Local Script Entrypoint
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    print("Resolving configurations and scaling GPU requirements...")
    config_path = "train/config.yaml"

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        return

    env_vars = {str(k): str(v) for k, v in config.get('env', {}).items()}
    python_args = []

    for key, value in config.get('python_args', {}).items():
        val_str = str(value)
        # Force translate all relative data paths to cloud absolute paths
        if "${BASE_DATA_DIR}" in val_str:
            val_str = val_str.replace("${BASE_DATA_DIR}", "/data")
        elif val_str.startswith("data/"):
            val_str = "/" + val_str

        # Resolve other placeholders
        if "${" in val_str:
            matches = re.findall(r"\$\{(.+?)\}", val_str)
            for m in matches:
                resolved_val = env_vars.get(m, os.environ.get(m, "1"))
                val_str = val_str.replace("${" + m + "}", str(resolved_val))
        python_args.append(f"{key}={val_str}")

    # Mandatory overrides for 0.8B model and Single GPU setup
    # These overrides take priority over config.yaml
    python_args.append("+model_name=Qwen/Qwen3.5-0.8B")
    python_args.append("+use_lora=True")

    # CRITICAL: Force trainer to use exactly 1 GPU provided by Modal
    python_args.append("trainer.n_gpus_per_node=1")
    python_args.append("actor_rollout_ref.rollout.n=1")
    python_args.append("actor_rollout_ref.actor.ppo_mini_batch_size=32")

    # Update Ray to use the 1 available GPU
    python_args.append("+ray_init.num_cpus=2")
    python_args.append("+ray_init.num_gpus=1")

    # Remove problematic/deprecated keys
    python_args = [arg for arg in python_args if "num_examine" not in arg]

    print(f"Dispatching to Modal L40S Cluster (Single GPU Mode)...")
    run_cloud_training.remote(env_vars, python_args)