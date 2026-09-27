import os
import modal
import yaml
import re

# 1. Define persistent storage (This creates a "cloud hard drive")
# This ensures checkpoints are saved even if the training is interrupted.
volume = modal.Volume.from_name("agentflow-checkpoints", create_if_missing=True)

app = modal.App("agentflow-step5-persistent")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "torchaudio",
        "transformers>=4.48.0", "datasets", "accelerate",
        "peft", "trl>=0.15.0", "wandb", "pandas", "numpy"
    )
    # Note: We keep add_local_dir for code/data access,
    # but we will use the Volume for SAVING results.
    .add_local_dir("./data", remote_path="/data")
)


# 2. Mount the volume in the function decorator
@app.function(
    image=image,
    gpu="L40S",
    timeout=43200,
    volumes={"/results": volume}  # Mount persistent volume to /results
)
def run_trl_training(hf_token: str):
    import os
    import torch
    import re
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    print("🚀 Starting Step 5: Qwen3.5-0.8B GRPO - Persistent Run (RESUMING)")

    model_id = "Qwen/Qwen3.5-0.8B"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )

    def reward_logic_function(completions, result, **kwargs):
        rewards = []
        for text, ground_truth in zip(completions, result):
            score = 0.0
            if "<think>" in text: score += 0.2
            if "</think>" in text: score += 0.2
            if "<answer>" in text: score += 0.2
            if "</answer>" in text: score += 0.2

            if ground_truth:
                gt_text = str(ground_truth).strip().lower()
                match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
                if match:
                    predicted = match.group(1).strip().lower()
                    if predicted == gt_text:
                        score += 2.0
                elif gt_text in text.lower():
                    score += 0.5
            rewards.append(score)
        return rewards

    print("Loading full dataset...")
    # Make sure this path is exactly what you had before
    dataset = load_dataset("parquet", data_files="/data/train/combined_train.parquet")["train"]
    if "question" in dataset.column_names:
        dataset = dataset.rename_column("question", "prompt")

    # 3. IMPORTANT: Change output_dir to the volume mount path (/results)
    training_args = GRPOConfig(
        output_dir="/results/qwen35_training",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=3000,
        logging_steps=10,
        save_steps=200,  # More frequent saves for safety
        save_total_limit=5,
        bf16=True,
        report_to="none"
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_logic_function,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print(f"🔥 TRAINING RESUMING FROM STEP 1800. Target Max steps: {training_args.max_steps}.")

    # 🌟 CRITICAL CHANGE HERE: Resuming from the Volume path
    trainer.train(resume_from_checkpoint="/results/qwen35_training/checkpoint-1800")

    # 4. Save final model to the Volume (Changed name slightly to avoid confusion)
    trainer.save_model("/results/final_qwen35_lora")

    # 5. Force a commit to ensure data is written to the cloud storage
    volume.commit()
    print("✅ Training Completed. Model safely stored in persistent volume.")


@app.local_entrypoint()
def main():
    import yaml
    config_path = "train/config.yaml"
    hf_token = ""
    try:
        with open(config_path, 'r') as f:
            raw_conf = yaml.safe_load(f)
            hf_token = raw_conf.get('env', {}).get("HF_TOKEN", "")
    except:
        hf_token = os.environ.get("HF_TOKEN", "")

    run_trl_training.remote(hf_token)