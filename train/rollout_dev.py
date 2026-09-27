import os
from agentflow import Trainer, DevTaskLoader, LLM
from rollout import Rollout

import pandas as pd

def dev_task_loader() -> DevTaskLoader:
    parquet_path = "data/val/aime24.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    df = df.head(10)

    if 'question' not in df.columns or 'result' not in df.columns:
        raise ValueError(f"Parquet file must have 'question' and 'result' columns. Found: {list(df.columns)}")

    ground_truth_col = 'ground_truth' if 'ground_truth' in df.columns else 'result'

    tasks = []
    for idx, row in df.iterrows():
        task = {
            "question": str(row["question"]),
            "result": str(row["result"]),
            "extra_info": {
                "ground_truth": str(row[ground_truth_col]),
                "idx": int(idx),
            }
        }
        tasks.append(task)

    return DevTaskLoader(
        tasks=tasks,
        resources={
            "main_llm": LLM(
                endpoint="https://api.openai.com/v1",
                model="gpt-4o-mini",
                sampling_parameters={"temperature": 0.7}
            ),
        },
    )

def dev_one_sample_loader() -> DevTaskLoader:
    question = "A point $(x,y)$ is randomly and uniformly chosen inside the square with vertices (0,0), (0,2), (2,2), and (2,0).  What is the probability that $x+y < 3$? When ready, output the final answer enclosed in <answer> and </answer> tags. Do not generate any content after the </answer>"
    task = {
            "question": question,
            "result": "0.75",
            "extra_info": {
                "ground_truth": "0.75",
                "idx": 0,
            }
        }

    tasks = [task]
    return DevTaskLoader(
        tasks=tasks,
        resources={
            "main_llm": LLM(
                endpoint="https://api.openai.com/v1",
                model="gpt-4o",
                sampling_parameters={"temperature": 0.7}
            ),
        },
    )

if __name__ == "__main__":
    Trainer(n_workers=1, dev=True, max_tasks=3).fit(Rollout(), "http://localhost:9991/", dev_task_loader())
