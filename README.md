# AgentFlow Reproduction: Qwen3.5 Planners, Flow-GRPO LoRA and a Text-to-SQL Benchmark

[![tests](https://github.com/Jason0173/agentflow-reproduction/actions/workflows/tests.yml/badge.svg)](https://github.com/Jason0173/agentflow-reproduction/actions/workflows/tests.yml)

This was a four-person team project for INFO 7375 Self-Improving AI at Northeastern University (Spring 2026). We reproduced **AgentFlow** ([paper](https://arxiv.org/abs/2510.05592), [code](https://github.com/lupantech/AgentFlow)), an agentic framework whose planner is trained inside the agent loop with Flow-GRPO. We ran it with Qwen3.5 models on five of the paper's benchmarks, trained the 0.8B planner with Flow-GRPO and LoRA, and added Spider Text-to-SQL as a benchmark the paper did not use.

This repository is my (Ke Wang's) organized copy of the team's final code and results. The original is our team repo, [Melody-coder923/neu-self-improve-ai](https://github.com/Melody-coder923/neu-self-improve-ai) (`week08_group_work_finalVersion/`). [Who did what](#who-did-what) credits each part to the person who wrote it.

## My contribution

**Controlled re-evaluation of the training result.** The team report credits Flow-GRPO + LoRA with large gains, up to +22 points on HotpotQA. That comparison was confounded:

- the two models answered different sets of questions;
- in the untrained runs, Google Search had run out of quota;
- the two models were served by different code.

I built [`eval/`](eval/), a Modal pipeline that runs both models on the same 250 questions. Everything except the LoRA weights is held fixed: base-weight snapshot, serving code, tools, judge and GPU. Questions that hit API errors are retried. Before any GPU time is spent, a preflight checks every tool and API key, and that the served LoRA weights really differ from the base.

Building it also surfaced three problems that would have skewed any comparison:

- the scorer's judge prompt is written for multiple-choice questions and marked correct free-form answers wrong;
- Wikipedia rejected the tools' generic user agent;
- the pinned torch/peft versions cannot load the adapter.

**Result:** no accuracy gain (23.6% vs 23.2%, McNemar p = 1.00). What training did change is the planner's behaviour: the trained model answers after about one step instead of five, and takes a fifth of the time per question. See [Results](#with-flow-grpo--lora-qwen35-08b-planner).

**Spider Text-to-SQL benchmark (new-benchmark requirement).** I added Spider 1.0 as the project's new benchmark. The evaluation puts the database schema in the prompt, runs the full AgentFlow loop, extracts the SQL from the final answer, and scores execution accuracy against the gold query. With Qwen3.5-0.8B as the planner, it got **0.50 (10/20)**, compared with **0.35 (7/20)** for an earlier Qwen2.5-7B-Instruct reference run. Details and caveats: [`test/text2sql/`](test/text2sql/).

**First attempt at the training step.** Before the team moved training to Modal, I tried to run AgentFlow's own Flow-GRPO training stack (verl, vLLM and LoRA) locally on WSL2 with an RTX 4080 SUPER (16 GB). I got stuck in a chain of version conflicts between flash-attn, transformers, tokenizers, vLLM and CUDA inside the project environment. The team then trained on Modal with TRL's `GRPOTrainer` and PEFT LoRA instead, which is the run reported below. The lesson I took away: when a research stack's pinned dependencies fight your hardware, a smaller, well-supported training library on rented GPUs can be the faster path to a result.

**This cleaned-up copy.** I restored the lowercase `agentflow/` package folder (see [Changes from the team repo](#changes-from-the-team-repo)), made the Spider script runnable with command-line options, and added unit tests.

## Results

### Without Flow-GRPO training

Accuracy (%) of the full AgentFlow loop, with each Qwen3.5 model as the planner:

| Planner | Bamboogle | 2Wiki | HotpotQA | Musique | GAIA |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 10.4 | 15.0 | 8.0 | 3.0 | 0.0 |
| Qwen3.5-2B | 32.8 | 19.0 | 28.1 | 4.0 | 6.3 |
| Qwen3.5-4B | 54.4 | 42.0 | 47.0 | 8.0 | 14.2 |
| Qwen3.5-9B | 70.3 | 29.0 | 65.3 | 3.0 | 7.1 |

Each score comes from `test/<benchmark>/results/<model>/final_scores_direct_output.json`. There were 96–127 questions per benchmark. GAIA answers for 0.8B–4B were judged by an LLM (`test/score_gaia_llm.py`).

These are the team's runs, and two problems affect them. Scores from `test/calculate_score_unified.py` used its multiple-choice judge prompt, described below. And 25–39% of the 0.8B runs' answers report Google Search quota errors; the 9B runs show none, and the 2B and 4B per-question answers are not in the repo. In the controlled re-run below, the 0.8B planner scores 40.0 on the first 50 HotpotQA questions, where the committed run above has 8.0.

### With Flow-GRPO + LoRA (Qwen3.5-0.8B planner)

The team report found large gains from training. A controlled re-run on the same questions does not reproduce them:

| Benchmark | Team report: no training → LoRA | Controlled re-run: no training → LoRA | McNemar p |
|---|---:|---:|---:|
| HotpotQA | 8.0 → 30.0 (+22.0) | 40.0 → 32.0 (−8.0) | 0.45 |
| 2Wiki | 15.0 → 28.0 (+13.0) | 30.0 → 34.0 (+4.0) | 0.81 |
| Bamboogle | 10.4 → 18.0 (+7.6) | 24.0 → 24.0 (0.0) | 1.00 |
| GAIA | 0.0 → 6.0 (+6.0) | 20.0 → 16.0 (−4.0) | 0.77 |
| Musique | 3.0 → 6.0 (+3.0) | 4.0 → 10.0 (+6.0) | 0.38 |
| All five (250 questions) | | 23.6 → 23.2 (−0.4) | 1.00 |

**Team report.** The trained model answered the first 50 questions of each benchmark; the no-training scores cover the full sets. The two columns also differ in two other ways:

- **Search failures.** In the no-training runs, 25–39% of the final answers on 2Wiki, HotpotQA, Musique and GAIA say that Google Search failed with a quota error.
- **Serving.** The two models were served by different code on different hardware.

**Controlled re-run** ([`eval/`](eval/)).

- **Held fixed:**
  - the first 50 questions of each benchmark;
  - the agent settings of the team's LoRA run;
  - one snapshot of the base weights, with the adapter merged in for the trained model;
  - the same serving code, tools and gpt-4o judge.
- **Retries:** questions that hit API errors are retried.
- **Judge prompt:** the judge uses an open-QA prompt, because the scorer's default prompt is written for multiple-choice questions. Absolute scores are therefore not comparable across the two columns.
- **Significance:** none of the per-benchmark differences is statistically significant. The two models agree on most questions, and each gets about 35 right that the other misses.

![Accuracy and steps per question, base vs LoRA](eval/plots/fair50.png)

What training did change is how the planner works. Averages over all 250 questions:

| | No training | Flow-GRPO + LoRA |
|---|---:|---:|
| Steps per question | 5.5 | 1.1 |
| Questions answered after one step | 30% | 98% |
| Questions that ran to the 10-step limit | 38% | 1% |
| Tool selections AgentFlow could not use | 49% | 9% |
| Seconds per question (NVIDIA L4) | 185 | 39 |

Half of the untrained model's tool choices are wasted: it wraps the tool name in backticks (96% of the failures), AgentFlow's parser rejects it, and the model often keeps planning until the step limit. The trained model usually names the tool plainly, searches once and answers. It reaches the same accuracy at about a fifth of the cost. Per-benchmark numbers, the per-question verdicts and every trajectory are in [`eval/RESULTS.md`](eval/RESULTS.md) and `test/<benchmark>/results/Qwen3.5-0.8B-{base,LoRA}-fair50/`.

The LoRA adapter is in `results/final_qwen35_lora/`, and the team's merged model is published as [`Skypioneer/qwen35-0.8b-agentflow-lora`](https://huggingface.co/Skypioneer/qwen35-0.8b-agentflow-lora).

### New benchmark: Spider Text-to-SQL

| Planner | Execution accuracy (20 dev questions) |
|---|---:|
| Qwen3.5-0.8B | 0.50 |
| Qwen2.5-7B-Instruct (reference) | 0.35 |

## Who did what

This is based on the commit history of the team repo.

| Part | Who | Where |
|---|---|---|
| AgentFlow framework, benchmark harness, verl training code | Upstream [AgentFlow](https://github.com/lupantech/AgentFlow) authors (MIT License) | `agentflow/`, `test/solve.py`, `test/calculate_score_unified.py`, `test/*/run.sh`, `train/` (except below), `util/`, `scripts/`, `data/` |
| Project setup, Modal model engine, no-training runs for 0.8B/2B/4B, GAIA LLM judging, Flow-GRPO + LoRA training on Modal | Chien-Cheng Wang | `agentflow/agentflow/engine/modal_engine.py`, `modal_serve.py`, `test/score_gaia_llm.py`, `train/modal_train_agent.py`, `results/final_qwen35_lora/` |
| Qwen3.5-9B/27B runs on Modal vLLM | Zhenyu Dai | `test/run_step3.sh`, `test/*/run_modal_*.sh` |
| Serving and evaluating the LoRA model on an H200 (Northeastern Explorer cluster), final report | Yan Zhao | `serve_lora_local.py`, `run_lora_h200.sbatch`, `smoke_test_lora.py`, `cluster_check.sh`, `test/run_lora_bench.sh`, `docs/team_report.md` |
| Spider Text-to-SQL benchmark; local Flow-GRPO training attempt; controlled base-vs-LoRA re-evaluation | Ke Wang | `test/text2sql/`, `eval/`, `tests/` |

## Repository layout

```
agentflow/            AgentFlow framework (planner, executor, verifier, tools, engines)
test/                 benchmark runners, scoring and results
  <benchmark>/results/<model>/   aggregated scores for each run
  text2sql/           Spider Text-to-SQL benchmark
train/                Flow-GRPO training: modal_train_agent.py (TRL + LoRA) and the upstream verl setup
results/              trained LoRA adapter for Qwen3.5-0.8B
serve_lora_local.py   serve the LoRA model with an OpenAI-compatible API (used on the H200)
modal_serve.py        serve a model on Modal
eval/                 controlled base-vs-LoRA re-evaluation on Modal
docs/                 the team's submitted report and the upstream how-to guides
tests/                unit tests (Spider scoring, evaluation runner)
```

## Running

Setup and the full set of commands are in the team report ([`docs/team_report.md`](docs/team_report.md), sections 3–4). The main entry points are:

```bash
bash setup.sh && source .venv/bin/activate      # environment
cp agentflow/.env.template agentflow/.env        # then add API keys

python quick_start.py                            # one question end to end
cd test/bamboogle && bash run.sh                 # one benchmark, no training
modal run train/modal_train_agent.py             # Flow-GRPO + LoRA training on Modal
python test/text2sql/spider_eval.py --limit 20   # Spider (see test/text2sql/README.md)
modal run --detach eval/modal_fair_eval.py       # base vs LoRA on the same questions (see eval/README.md)
pytest tests                                     # unit tests, no GPU or API key
```

## Changes from the team repo

- **Package folder name.** The team repo stored the framework in `AgentFlow/`, but `setup.sh`, `pyproject.toml` and all imports expect `agentflow/`. This only works on case-insensitive file systems such as macOS. Here it is `agentflow/` again, as upstream.
- **Spider script.** The script that produced the Spider numbers had hard-coded paths. It is now `test/text2sql/spider_eval.py` with command-line options, and the evaluation logic is unchanged. An unfinished duplicate script and an unused `execute_sql_tool.py` with a syntax error were removed.
- **Google Search tool.** The search model can be set with `GOOGLE_SEARCH_MODEL`, and retries now wait 1, 2, 4 and 8 seconds instead of firing five times within a second.
- **Judge prompt.** `test/calculate_score_unified.py` has a new `--judge_prompt open_qa` option. The default prompt, kept as it was, is worded for multiple-choice questions; with it, gpt-4o judged a correct free-form answer wrong in our tests. The re-evaluation in [`eval/`](eval/) uses the new option.
- **Wikipedia tools.** Requests to Wikipedia now send a User-Agent that identifies this project, as [Wikimedia's policy](https://meta.wikimedia.org/wiki/User-Agent_policy) asks. The generic default was being refused.
- **Trimmed.** The ~2,200 per-question trajectory files (about 440 MB), the upstream README images and the `.DS_Store` files were not copied. The trajectory files remain in the team repo.

## Citation and license

The AgentFlow framework is © the AgentFlow Team and released under the MIT License ([`LICENSE`](LICENSE)).

```bibtex
@inproceedings{li2026flow,
  title     = {In-the-Flow Agentic System Optimization for Effective Planning and Tool Use},
  author    = {Li, Zhuofeng and Zhang, Haoxiang and Han, Seungju and Liu, Sheng and Xie, Jianwen and Zhang, Yu and Choi, Yejin and Zou, James and Lu, Pan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
