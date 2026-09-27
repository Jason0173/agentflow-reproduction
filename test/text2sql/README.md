# Step 4: Spider Text-to-SQL Benchmark

## Results
| Model | Execution Accuracy |
|-------|-------------------|
| Qwen2.5-7B (no training) | 0.350 |

## Setup
1. Download Spider 1.0 from https://yale-nlp.github.io/spider/
2. Place database files at data/spider/database/
3. Place dev.json at data/spider_data/dev.json

## Run
python test/text2sql/agentflow_spider.py
