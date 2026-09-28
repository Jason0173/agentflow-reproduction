# Fair base-vs-LoRA evaluation

This folder re-runs the untrained Qwen3.5-0.8B planner and the Flow-GRPO + LoRA planner under identical conditions, so the difference between them can be attributed to the training.

## Result

Over the first 50 questions of each of the five benchmarks (250 in total), the two planners are equally accurate: 23.6% without training, 23.2% with Flow-GRPO + LoRA, McNemar p = 1.00. Training did change how the planner works. Averages over the 250 questions:

| | No training | Flow-GRPO + LoRA |
|---|---:|---:|
| Steps per question | 5.5 | 1.1 |
| Tool selections AgentFlow could not use | 49% | 9% |
| Seconds per question | 185 | 39 |

Full tables and the figure are in [`RESULTS.md`](RESULTS.md). `python eval/summarize.py` rebuilds them from the committed results.

## Why a re-run

The comparison in the team report differs in more than the training:

1. **Different questions.** The base model was scored on 100–127 questions per benchmark, the LoRA model on the first 50.
2. **Search failures in the base runs.** In the committed base results, 25–39% of the final answers on 2Wiki, HotpotQA, Musique and GAIA say that Google Search failed with a quota error (`429 RESOURCE_EXHAUSTED`). The LoRA run's per-question outputs were not committed, so they cannot be checked the same way.
3. **Different serving.** The base model went through the team's Modal deployment; the LoRA model was served with `serve_lora_local.py` on an H200.

## What is held fixed

| | Both models |
|---|---|
| Questions | First 50 of Bamboogle, 2Wiki, HotpotQA, Musique and GAIA, from upstream AgentFlow at commit `b940064` (SHA-256 checked) |
| Agent settings | Exactly those of `test/run_lora_bench.sh`, which produced the team's LoRA numbers: planner, verifier and executor all use the 0.8B model, 10 steps, 300 s, temperature 0. `tests/test_fair_eval.py` fails if the two drift apart |
| Serving | `serve_lora_local.py` with a merged checkpoint (`MERGED_MODEL`) and the base model's tokenizer, bf16, greedy decoding, thinking off, 2,048 new tokens |
| Weights | One snapshot of `Qwen/Qwen3.5-0.8B`, baked into the image with its revision recorded. `build_models.py` writes both served checkpoints with the same code, merging `results/final_qwen35_lora` into the second one in float32. The preflight checks that the two differ only in the matrices the adapter targets |
| Tools | Base_Generator (gpt-4o-mini), Google Search with grounding, Wikipedia search |
| Judge | gpt-4o through `test/calculate_score_unified.py`, with its open-QA prompt (`--judge_prompt open_qa`, see below) |
| Hardware | NVIDIA L4 on Modal |

Base and LoRA work on the same questions at the same time (the scheduler interleaves them), so a temporary API problem affects both alike.

**Changed from the team's setup, for both models:**

- **Search model.** Google Search uses `gemini-3.5-flash-lite` instead of `gemini-2.5-flash`, because new Gemini API projects may not get access to the 2.5 models. It can be set with `--search-model`.
- **Merged checkpoint.** The team served its published merged model (`Skypioneer/qwen35-0.8b-agentflow-lora`). Here the merge is rebuilt from the committed adapter on the same base snapshot, so nothing but the LoRA update differs. (The server's own adapter mode does not work with the pinned torch 2.6 and peft 0.19.)
- **Retry backoff.** The search tool now waits 1, 2, 4 and 8 seconds between retries. Before, a burst of rate-limit errors used up all five retries within a second.
- **Judge prompt.** The scorer's default single-stage prompt, unchanged since upstream AgentFlow's first commit, is written for multiple-choice questions: a prediction counts only if it matches "the correct choice letter". All five benchmarks here are free-form. In the preflight, gpt-4o therefore judged `<answer>Paris</answer>` against the answer "Paris" as wrong because "Paris" is not a letter. The evaluation uses a new `--judge_prompt open_qa` option instead, which asks whether the two answers name the same thing. The default is unchanged, so absolute accuracies here are not comparable with the team's table.
- **Wikipedia user agent.** The Wikipedia tools now identify the project in their User-Agent, as [Wikimedia's policy](https://meta.wikimedia.org/wiki/User-Agent_policy) asks. With the generic default, Wikipedia returned no search results from Modal.

**Kept as in the team's setup, for both models:** the Wikipedia tool never reads page text. Upstream AgentFlow's `Wikipedia_Search_Tool` does not store its `model_string`, so creating its page reader fails (`Error creating Web RAG tool` in the logs), and the tool returns search titles only. Fixing it would change the tools the team's LoRA run used.

## API failures

An answer should count as wrong because of the planner, not because an API ran out of quota. After each question, `fair_eval_core.infra_errors` scans the tool results and model calls for error strings, for example quota errors, failed searches, OpenAI error objects, timeouts and planner-server errors.

- **Retries.** A question that hit such an error is set aside and re-run, up to 3 attempts. The last attempt is kept either way and counted in the results. A question that never produced an answer gets an empty one, which the judge marks wrong, so no question silently drops out.
- **Rounds.** The coordinator repeats rounds, 2 minutes apart, until every question is finished. A round that neither finishes a question nor makes an attempt ends the run.
- **Stopping early.** A chunk stops after 3 such questions in a row, and the whole run stops after 3 failed chunks, so a broken key does not burn GPU hours.
- **Preflight.** Before any GPU starts, `preflight.py` makes one real call to each tool and to the judge, and checks that the tools load.

## Running

Requirements: a Modal account with a payment method, an OpenAI API key, and a Gemini API key on a billed project (Google Search grounding is not in the free tier).

```bash
pip install modal && modal setup                     # once
modal secret create agentflow-keys OPENAI_API_KEY=... GOOGLE_API_KEY=...

modal run eval/modal_fair_eval.py --preflight-only   # no GPU; builds the image the first time
modal run eval/modal_fair_eval.py --run smoke --limit 2 --tasks bamboogle
modal run --detach eval/modal_fair_eval.py           # full run: 2 models x 5 benchmarks x 50 questions
modal run eval/modal_fair_eval.py --download-only    # fetch results, e.g. after the terminal closed
```

Progress is stored in the Modal volume `agentflow-fair-eval`. In the reported run, Modal preempted the coordinator container twice; each time it restarted, cancelled the chunks in flight and continued from the stored progress. Running the same command again continues where it stopped. A run keeps the settings it started with: resuming with a different `--limit` or `--search-model` is refused. The smoke run prints the average time per question and the projected cost of the full run.

## Output

- **Summary.** `RESULTS.md`, `results.csv` and `plots/fair50.png` (written by `summarize.py`) hold per-benchmark accuracy for both models on the same questions, the questions only one model got right, and an exact McNemar test.
- **Per run folder.** Each `test/<benchmark>/results/Qwen3.5-0.8B-{base,LoRA}-fair50/` contains:
  - `final_scores_direct_output.json` and `finalresults_direct_output.json`: the scorer's output.
  - `questions.csv`: attempts, remaining API errors and time for each question.
  - `trajectories.jsonl.gz`: every question's full AgentFlow trajectory.

## Files

| File | Purpose |
|---|---|
| `modal_fair_eval.py` | Modal app: image, preflight, GPU chunks, coordinator, scoring, download |
| `chunk_runner.py` | Runs `solve.py` for a list of questions against one server, detects API failures, scores |
| `fair_eval_core.py` | Settings, error patterns, scheduling and statistics (unit-tested) |
| `build_models.py` | Image build step: writes the base and base + LoRA checkpoints |
| `summarize.py` | Builds `RESULTS.md`, `results.csv` and `plots/` from the committed results |
| `preflight.py` | Checks data, tools, keys, judge, the two checkpoints and tokenizers before any GPU time |
