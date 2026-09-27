> This is the report our team submitted, kept unchanged. Paths are relative to the team repo's `week08_group_work_finalVersion/` folder. In this repository, `AgentFlow/` is the root, and `test/text2sql/agentflow_spider.py` was replaced by `test/text2sql/spider_eval.py`.

# Week 08 Group Work - AgentFlow Reproduction

This README documents our Week 08 assignment implementation based on the code and data in this directory.

## Team Members

- Chien-Cheng Wang
- Zhenyu Dai
- Ke Wang
- Yan Zhao

## 1) Repository Structure Used for This Assignment

- Main implementation: `AgentFlow/`
- Alternative/backup implementation: `AgentFlow_/`
- We use `AgentFlow/` as the primary codebase for all steps below.

Key paths in `AgentFlow/`:
- Inference quick test: `quick_start.py`
- Benchmark runner scripts: `test/*/run.sh`
- Qwen3.5 (9B/27B) benchmark helper: `test/run_step3.sh`
- New benchmark starter (Text2SQL): `test/text2sql/agentflow_spider.py`
- Flow-GRPO training config: `train/config.yaml`
- Modal + TRL + LoRA training script: `train/modal_train_agent.py`
- Saved LoRA adapter (0.8B): `results/final_qwen35_lora/`

## 2) Assignment Requirement Mapping

### Requirement A - Reimplement AgentFlow paper

- Status: Completed (code-level reproduction).
- Base framework and scripts are implemented in `AgentFlow/`.
- Core dependencies include modern HF ecosystem usage in project scripts:
  - `transformers` (see `pyproject.toml`)
  - `peft` and `trl` (see `train/modal_train_agent.py`)

### Requirement B - Reproduce AgentFlow without Flow-GRPO training

- Target setting: "second-to-last row" style evaluation (no Flow-GRPO training).
- Status: Completed for submission scope (`Qwen3.5-0.8B/2B/4B/9B`; `27B` excluded by scope note).
- Implemented benchmark pipeline exists under `test/` for:
  - `bamboogle`, `2wiki`, `hotpotqa`, `musique`, `gaia`
- Filled no-training scores are reported in Section `7.1`.
- Qwen3.5 Modal serving/evaluation scripts are in `test/run_step3.sh`.

### Requirement C - New benchmark not used in paper

- Teacher suggested options: WebShop, Text2SQL, Code Generation, Security Vulnerabilities.
- Status: Completed with Text2SQL (Spider) reportable results.
- Implemented files:
  - `test/text2sql/agentflow_spider.py`
  - `test/text2sql/finalscore_spider.json`
- Notes:
  - Additional new benchmarks (WebShop/Code/Security) are not included in this submission scope.

### Requirement D - Flow-GRPO + LoRA on Qwen3.5-0.8B (compare with non-trained system)

- Status: Completed.
- Training script: `train/modal_train_agent.py`
  - Uses `trl.GRPOTrainer`
  - Uses `peft.LoraConfig`
  - Base model: `Qwen/Qwen3.5-0.8B`
- Saved adapter: `results/final_qwen35_lora/`
- Merged adapter used for evaluation: `Skypioneer/qwen35-0.8b-agentflow-lora`
- Benchmark evaluation: the merged adapter was served on a single H200 GPU on the Northeastern Explorer cluster (`serve_lora_local.py`, SLURM job `5980831`, node `d4055`) and run through the full AgentFlow loop on 50 samples per benchmark; per-task scores and deltas are reported in Section `7.2`.

### Requirement E - Flow-GRPO + LoRA on new benchmark

- Status: Optional based on instructor note; can be skipped.
- Instructor recommendation accepted: prioritize Hugging Face TRL + LoRA stack over heavier systems.

## 3) Environment and Setup

From `AgentFlow/`:

```bash
bash setup.sh
source .venv/bin/activate
cp agentflow/.env.template agentflow/.env
```

Set required keys in `agentflow/.env` (at least as needed by your selected tools/models):
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `DASHSCOPE_API_KEY` or `TOGETHER_API_KEY`

## 4) How to Run (Repro Commands)

### 4.1 Quick inference sanity check

```bash
cd AgentFlow
python quick_start.py
```

### 4.2 Benchmark without Flow-GRPO training

Run one benchmark (example):

```bash
cd AgentFlow/test/bamboogle
bash run.sh
```

Run prepared Qwen3.5 Modal script (9B or 27B across 5 tasks):

```bash
cd AgentFlow/test
bash run_step3.sh 9B
# or
bash run_step3.sh 27B
```

### 4.3 Flow-GRPO + LoRA training (Qwen3.5-0.8B)

```bash
cd AgentFlow
modal run train/modal_train_agent.py
```

After training, check:
- adapter output path: `AgentFlow/results/final_qwen35_lora/`

## 5) Data Used

- Training data scripts:
  - `data/get_train_data.py`
  - `data/aime24_data.py`
- Benchmark data files are organized per task under:
  - `test/<task>/data/`
- Text2SQL resources expected by script:
  - `data/spider/database` (used by `test/text2sql/agentflow_spider.py`)

## 6) Current Deliverables in This Submission

- AgentFlow codebase integrated and runnable
- Non-Flow-GRPO benchmark scripts for AgentFlow tasks
- Qwen3.5 Modal inference scripts for larger models (9B/27B)
- TRL + LoRA Flow-GRPO training script on Qwen3.5-0.8B
- Trained LoRA adapter artifacts in `results/final_qwen35_lora/`

## 7) Final Results

Minimal submission format (only required rows).

Scoring convention:
- `No-Training`: AgentFlow inference without Flow-GRPO fine-tuning
- `LoRA-Training`: AgentFlow with Flow-GRPO + LoRA
- `Delta`: `LoRA-Training - No-Training`

### 7.1 No-Training Results (all required Qwen3.5 models)

Scope note: `Qwen3.5-27B` is excluded from this submission scope.

| Model | Benchmark | No-Training Score | Notes |
|---|---|---:|---|
| Qwen3.5-0.8B | 2wiki | 15.0 | from `final_scores_direct_output.json` |
| Qwen3.5-0.8B | bamboogle | 10.4 | from `final_scores_direct_output.json` |
| Qwen3.5-0.8B | hotpotqa | 8.0 | from `final_scores_direct_output.json` |
| Qwen3.5-0.8B | musique | 3.0 | from `final_scores_direct_output.json` |
| Qwen3.5-0.8B | gaia | 0.0 | from `final_scores_direct_output.json` |
| Qwen3.5-2B | 2wiki | 19.0 | from `final_scores_direct_output.json` |
| Qwen3.5-2B | bamboogle | 32.8 | from `final_scores_direct_output.json` |
| Qwen3.5-2B | hotpotqa | 28.12 | from `final_scores_direct_output.json` |
| Qwen3.5-2B | musique | 4.0 | from `final_scores_direct_output.json` |
| Qwen3.5-2B | gaia | 6.3 | from `final_scores_direct_output.json` |
| Qwen3.5-4B | 2wiki | 42.0 | from `final_scores_direct_output.json` |
| Qwen3.5-4B | bamboogle | 54.4 | from `final_scores_direct_output.json` |
| Qwen3.5-4B | hotpotqa | 47.0 | from `final_scores_direct_output.json` |
| Qwen3.5-4B | musique | 8.0 | from `final_scores_direct_output.json` |
| Qwen3.5-4B | gaia | 14.17 | from `final_scores_direct_output.json` |
| Qwen3.5-9B | 2wiki | 29.0 | label in files: `Qwen3.5-9B-Modal` |
| Qwen3.5-9B | bamboogle | 70.34 | label in files: `Qwen3.5-9B-Modal` |
| Qwen3.5-9B | hotpotqa | 65.31 | label in files: `Qwen3.5-9B-Modal` |
| Qwen3.5-9B | musique | 3.0 | label in files: `Qwen3.5-9B-Modal` |
| Qwen3.5-9B | gaia | 7.09 | label in files: `Qwen3.5-9B-Modal` |
### 7.2 LoRA Comparison (required training comparison on 0.8B)

| Model | Benchmark | No-Training Score | LoRA-Training Score | Delta |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B | 2wiki | 15.0 | 28.0 | +13.0 |
| Qwen3.5-0.8B | bamboogle | 10.4 | 18.0 | +7.6 |
| Qwen3.5-0.8B | hotpotqa | 8.0 | 30.0 | +22.0 |
| Qwen3.5-0.8B | musique | 3.0 | 6.0 | +3.0 |
| Qwen3.5-0.8B | gaia | 0.0 | 6.0 | +6.0 |

Note: LoRA-Training scores are from the merged adapter `Skypioneer/qwen35-0.8b-agentflow-lora` served locally on a single H200 GPU (Northeastern Explorer cluster, job `5980831`, node `d4055`), labeled `Qwen3.5-0.8B-LoRA-20260416` in `AgentFlow/test/*/results/`. Each LoRA benchmark was evaluated on the first 50 samples of the corresponding test set (No-Training scores use the full sample counts reported in §7.1); all five deltas are positive, with the largest gains on multi-hop QA (hotpotqa +22.0, 2wiki +13.0). Scores are read directly from each benchmark's `final_scores_direct_output.json`.

### 7.3 New Benchmark (required, different from paper tasks)

| Benchmark Type | Dataset/Task | Model | No-Training Score | LoRA-Training Score | Notes |
|---|---|---|---:|---:|---|
| Text2SQL | Spider (local script) | Qwen3.5-0.8B | 0.50 (10/20) | Optional / Not Available | from `test/text2sql/finalscore_spider.json` |
| Text2SQL | Spider (reference run) | Qwen2.5-7B-Instruct | 0.35 (7/20) | Optional / Not Available | from `test/text2sql/finalscore_spider.json` |

### 7.4 Final Conclusion

This submission uses a reduced scope by excluding `Qwen3.5-27B`, and now includes complete no-training scores for `0.8B / 2B / 4B / 9B` across five main benchmarks. Reported values prioritize valid entries from `final_scores_direct_output.json`. `GAIA 0.8B = 0.0%` means the run completed with `0/127` correct rather than being skipped. Text2SQL now includes a `Qwen3.5-0.8B` Spider score of `0.50 (10/20)`, and the same file also contains a `Qwen2.5-7B` reference score of `0.35 (7/20)`. The `0.8B no-training vs LoRA-training` comparison is now complete: the merged LoRA adapter was served on a single H200 GPU (Northeastern Explorer cluster) and evaluated on 50 samples per benchmark, yielding positive deltas on all five tasks (2wiki +13.0, bamboogle +7.6, hotpotqa +22.0, musique +3.0, gaia +6.0) — see §7.2 for details.

### 7.5 Final Conclusion

This submission uses a reduced scope: `Qwen3.5-27B` is excluded, and no-training results for `0.8B / 2B / 4B / 9B` across five main benchmarks are filled. We prioritize valid values from `final_scores_direct_output.json`. `GAIA 0.8B = 0.0%` means the run completed with `0/127` correct, not a skipped run. Text2SQL includes a `Qwen3.5-0.8B` Spider score of `0.50 (10/20)` with a `Qwen2.5-7B` reference score of `0.35 (7/20)`. The required `0.8B no-training vs LoRA-training` comparison is complete: the merged LoRA adapter was served on a single H200 GPU on the Northeastern Explorer cluster and evaluated on 50 samples per benchmark, yielding positive deltas on all five tasks (2wiki +13.0, bamboogle +7.6, hotpotqa +22.0, musique +3.0, gaia +6.0); see Section `7.2`.

## 8) Environment Versions

### Inference / LoRA serving stack (Northeastern Explorer H200, `~/AgentFlow/.venv`)

This is the environment used to serve the merged LoRA adapter and run the Section `7.2` benchmark sweep.

| Package | Version |
|---|---|
| Python | 3.11.14 |
| torch | 2.6.0 |
| transformers | 5.5.4 |
| peft | 0.19.0 |
| accelerate | 1.13.0 |
| sentencepiece | 0.2.1 |
| fastapi | 0.128.0 |
| uvicorn | 0.31.1 |

Command used to produce the table above:

```bash
source ~/AgentFlow/.venv/bin/activate
for p in transformers peft torch accelerate sentencepiece fastapi uvicorn; do
  python3 -c "import importlib.metadata as m;print('$p', m.version('$p'))"
done
python3 --version
```

### Training stack (Modal, `train/modal_train_agent.py`)

Flow-GRPO + LoRA training ran on Modal (not on H200); the image is pinned in `train/modal_train_agent.py`:

| Package | Version constraint |
|---|---|
| transformers | `>=4.48.0` |
| peft | (unpinned) |
| trl | `>=0.15.0` |
| accelerate | (unpinned) |
| datasets | (unpinned) |

The resulting adapter is saved to `AgentFlow/results/final_qwen35_lora/` and the merged variant used for evaluation is published at `Skypioneer/qwen35-0.8b-agentflow-lora`.
