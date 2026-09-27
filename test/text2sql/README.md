# Spider Text-to-SQL Benchmark

The course asked us to evaluate AgentFlow on a benchmark the paper did not use. This folder adds [Spider 1.0](https://yale-lily.github.io/spider), a cross-database Text-to-SQL benchmark. I (Ke Wang) wrote this part of the project.

## How it works

For each Spider dev question, `spider_eval.py`:

1. Reads the schema of the question's SQLite database and puts it in the prompt with the question.
2. Runs the full AgentFlow loop (planner, executor, verifier, generator) and takes the SQL from the final answer. It uses the ```` ```sql ```` block if there is one, otherwise the last line that starts with `SELECT`.
3. Scores **execution accuracy**: it runs the predicted and gold SQL on the same database and counts the prediction as correct when both return the same set of rows.

## Results

From `finalscore_spider.json`, first 20 questions of the dev set, no training:

| Planner model | Correct | Execution accuracy |
|---|---|---|
| Qwen3.5-0.8B (served on Modal) | 10 / 20 | 0.50 |
| Qwen2.5-7B-Instruct (earlier reference run) | 7 / 20 | 0.35 |

Things to keep in mind when reading these numbers:

- **20 questions is a small sample.** One question changes accuracy by 5 points.
- **Only the main planner runs on the model in the table.** The script calls `construct_solver(llm_engine_name="modal")` with AgentFlow's default `model_engine`, so the fixed planner, verifier and executor run on `gpt-4o`. This matches AgentFlow's design, where only the planner is trained.
- **No SQL tool is registered.** The prompt mentions an `execute_sql` tool, but AgentFlow's planner can only call its registered tools (search, Python coder, base generator). Adding a real SQL execution tool is the obvious next step.

## Run

1. Download Spider 1.0 and unzip it so that `data/spider_data/dev.json` and `data/spider_data/database/` exist.
2. Set up AgentFlow with `setup.sh` from the repository root.
3. Serve the planner model, for example with `modal_serve.py`, then set `MODAL_PLANNER_URL` to the endpoint. Also set `OPENAI_API_KEY` for the fixed modules.
4. Run:

```bash
python test/text2sql/spider_eval.py --limit 20 --output test/text2sql/run.json
```

The scoring helpers have unit tests that need no model or API key:

```bash
pytest tests/test_spider_eval.py
```

## Files

| File | Contents |
|---|---|
| `spider_eval.py` | Evaluation script |
| `finalscore_spider.json` | Scores reported above |
| `../../tests/test_spider_eval.py` | Unit tests for schema extraction, SQL extraction and execution matching |
