"""Spider Text-to-SQL evaluation for AgentFlow (the project's new benchmark).

Runs the full AgentFlow loop (planner -> executor -> verifier -> generator) on
questions from the Spider dev set and reports execution accuracy: a prediction
counts as correct when the rows it returns equal the rows returned by the gold
query on the same SQLite database. Row order and duplicate rows are ignored.

This is a cleaned-up version of the script that produced finalscore_spider.json.
The evaluation logic is unchanged; the hard-coded paths became command-line
options. See README.md in this folder for setup.

Example:
    export MODAL_PLANNER_URL=https://<your-modal-endpoint>/v1/chat/completions
    export OPENAI_API_KEY=...        # used by the fixed modules, see README
    python test/text2sql/spider_eval.py --spider-dir data/spider_data --limit 20
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PROMPT_TEMPLATE = (
    "Database schema:\n{schema}\n\nQuestion: {question}\n\n"
    "Use execute_sql tool to find the answer. Output final SQL in ```sql``` block."
)


def get_schema(db_path):
    """Describe every table as 'Table t: col (TYPE), ...', one line per table."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    lines = []
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = [f"{r[1]} ({r[2]})" for r in cur.fetchall()]
        lines.append(f"Table {t}: {', '.join(cols)}")
    conn.close()
    return "\n".join(lines)


def execute_sql(sql, db_path):
    """Run a query and return all rows, or [("ERROR", message)] if it fails."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        return [("ERROR", str(e))]


def extract_sql(text):
    """Take the SQL from a ```sql``` block, else the last line starting with SELECT."""
    m = re.search(r"```sql\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    lines = [l for l in text.split("\n") if l.strip().upper().startswith("SELECT")]
    return lines[-1].strip() if lines else text.strip()


def results_match(pred_sql, gold_sql, db_path):
    """Execution match: same set of result rows (order and duplicates ignored)."""
    pred_rows = execute_sql(pred_sql, db_path)
    gold_rows = execute_sql(gold_sql, db_path)
    return set(map(tuple, pred_rows)) == set(map(tuple, gold_rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spider-dir", default=os.environ.get("SPIDER_DIR", REPO_ROOT / "data" / "spider_data"),
                        help="Spider 1.0 folder containing dev.json and database/ (default: data/spider_data)")
    parser.add_argument("--limit", type=int, default=20, help="number of dev questions to run (default: 20)")
    parser.add_argument("--engine", default="modal",
                        help="llm_engine_name passed to construct_solver (default: modal, which reads MODAL_PLANNER_URL)")
    parser.add_argument("--output", default=None, help="optional path for a JSON file with per-question results")
    args = parser.parse_args()

    spider_dir = Path(args.spider_dir)
    db_dir = spider_dir / "database"
    with open(spider_dir / "dev.json") as f:
        dataset = json.load(f)[: args.limit]

    sys.path.insert(0, str(REPO_ROOT))
    from agentflow.agentflow.solver import construct_solver  # heavy import, only needed for a real run

    solver = construct_solver(llm_engine_name=args.engine)

    records, correct = [], 0
    for i, sample in enumerate(dataset, 1):
        db_path = str(db_dir / sample["db_id"] / f"{sample['db_id']}.sqlite")
        os.environ["SPIDER_DB_PATH"] = db_path
        prompt = PROMPT_TEMPLATE.format(schema=get_schema(db_path), question=sample["question"])
        output = solver.solve(prompt)
        pred_sql = extract_sql(output["direct_output"])
        ok = results_match(pred_sql, sample["query"], db_path)
        correct += ok
        records.append({"db_id": sample["db_id"], "question": sample["question"],
                        "gold_sql": sample["query"], "pred_sql": pred_sql, "correct": ok})
        print(f"[{i}/{len(dataset)}] Acc: {correct / i:.3f} | {sample['question'][:50]}")

    acc = correct / len(dataset) if dataset else 0.0
    print(f"\nFinal: {acc:.3f} ({correct}/{len(dataset)})  engine={args.engine}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"engine": args.engine, "execution_accuracy": acc, "correct": correct,
                       "total": len(dataset), "results": records}, f, indent=2)


if __name__ == "__main__":
    main()
