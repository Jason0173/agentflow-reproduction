"""Helpers for the fair base-vs-LoRA evaluation.

This module has no Modal or AgentFlow imports, so the logic that decides what
is run, what counts as an infrastructure failure and how results are summarized
can be unit-tested (tests/test_fair_eval.py) without a GPU or API keys.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

TASKS = ["bamboogle", "2wiki", "hotpotqa", "musique", "gaia"]
MODELS = ["base", "lora"]
BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B"
LORA_DIR = "results/final_qwen35_lora"

# Benchmark questions come from the upstream AgentFlow repository, pinned to one
# commit and checked against these SHA-256 hashes when the image is built.
UPSTREAM_REPO = "lupantech/AgentFlow"
UPSTREAM_COMMIT = "b94006436b8712ab8682846fb0d886a5f174f2d4"
DATA_SHA256 = {
    "bamboogle": "647c37dda2f27341e3f28e5b2a9625f6abcff44639f26a08dcbe1667d831d5e2",
    "2wiki": "eb4b045ebf8e3441bb14a7420e5d84bafcaea58c776cfec75c3bb98ee1653122",
    "hotpotqa": "81f09495a18c6bc1cd1c48483babc6c59dfb9381e524850be53ffb62a9c28fd7",
    "musique": "195b9e75a45c58a5a0398c753a1d8fb7a8adf87b4bfde3f12d20ce22517e4759",
    "gaia": "fa714861d661c54c8999f3b9de0155c6da3d2d6496e4c95cd12f4dcf59210b76",
}


# Number of questions in each file, to validate --limit.
DATA_SIZES = {"bamboogle": 125, "2wiki": 200, "hotpotqa": 100, "musique": 200, "gaia": 127}


def data_url(task: str) -> str:
    return (f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/"
            f"{UPSTREAM_COMMIT}/test/{task}/data/data.json")


# Client-side configuration of test/run_lora_bench.sh, the script that produced
# the team's LoRA numbers. Both models are run with exactly these settings;
# tests/test_fair_eval.py fails if they drift from that script.
PLANNER_ENGINE = "modal-Qwen/Qwen3.5-0.8B-LoRA"  # only routes calls to the local server
ENABLED_TOOLS = "Base_Generator_Tool,Google_Search_Tool,Wikipedia_Search_Tool"
TOOL_ENGINE = "Default,Default"
MODEL_ENGINE = "trainable,trainable,trainable,trainable"
MAX_TIME = "300"
MAX_STEPS = "10"
TEMPERATURE = "0.0"
OUTPUT_TYPES = "direct"

# What the AgentFlow initializer reports when every enabled tool loaded
# (the Wikipedia tool brings in the web RAG tool).
EXPECTED_TOOLS = [
    "Generalist_Solution_Generator_Tool",
    "Ground_Google_Search_Tool",
    "Web_RAG_Search_Tool",
    "Wikipedia_RAG_Search_Tool",
]

MAX_ATTEMPTS = 3  # tries per question when a run hits an infrastructure error
MAX_ROUNDS = 8    # scheduling rounds; a round that finishes no question also ends the run
MAX_CHUNK_SIZE = 12  # keeps a chunk well inside the GPU function's time limit


def validate_run_name(run: str) -> str:
    # The scorer derives the log folder with result_dir.replace("results", "logs").
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", run) or "results" in run or "logs" in run:
        raise ValueError(f"run name {run!r}: use letters, digits, '-' or '_', without 'results' or 'logs'")
    return run


def label(model: str, run: str) -> str:
    """Result folder name, e.g. Qwen3.5-0.8B-LoRA-fair50."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    return f"Qwen3.5-0.8B-{'LoRA' if model == 'lora' else 'base'}-{run}"


def solve_command(task: str, index: int, data_file: str, cache_dir: str, out_dir: str) -> List[str]:
    """The solve.py call made by run_lora_bench.sh for one question."""
    return [
        "python", "solve.py",
        "--index", str(index),
        "--task", task,
        "--data_file", data_file,
        "--llm_engine_name", PLANNER_ENGINE,
        "--root_cache_dir", cache_dir,
        "--output_json_dir", out_dir,
        "--output_types", OUTPUT_TYPES,
        "--enabled_tools", ENABLED_TOOLS,
        "--tool_engine", TOOL_ENGINE,
        "--model_engine", MODEL_ENGINE,
        "--max_time", MAX_TIME,
        "--max_steps", MAX_STEPS,
        "--temperature", TEMPERATURE,
    ]


def score_command(task: str, data_file: str, result_dir: str, max_workers: int = 4) -> List[str]:
    """The scoring call made by run_lora_bench.sh (fewer parallel judge calls,
    which only changes how fast the judge runs, not its verdicts)."""
    return [
        "python", "calculate_score_unified.py",
        "--task_name", task,
        "--data_file", data_file,
        "--result_dir", result_dir,
        "--response_type", "direct_output",
        "--output_file", "finalresults_direct_output.json",
        "--max_workers", str(max_workers),
    ]


# ---------------------------------------------------------------------------
# Infrastructure failures
#
# A question should only count as wrong because of the planner, not because a
# search API ran out of quota or the model server went down. These patterns
# match the error strings that AgentFlow's tools and engines return instead of
# raising.
# ---------------------------------------------------------------------------

TOOL_ERROR_PATTERNS: List[Tuple[str, str]] = [
    ("google_search_failed", r"Google Search tried \d+ times but failed|Google Search failed to get a valid response"),
    ("quota_or_rate_limit", r"RESOURCE_EXHAUSTED|Error code: 429|RateLimitError|insufficient_quota|exceeded your current quota"),
    ("api_key", r"API key not valid|API_KEY_INVALID|Incorrect API key|PERMISSION_DENIED"),
    ("openai_error", r"""['"]error['"]\s*:\s*['"](?:rate_limit|AuthenticationError|APIConnectionError|APITimeoutError|InternalServerError|PermissionDeniedError|NotFoundError)['"]"""),
    ("tool_exception", r"Error generating response:|Error searching Wikipedia:"),
    ("tool_timeout", r"Execution timed out after \d+ seconds"),
]
PLANNER_ERROR_PATTERN = r"Error calling Modal endpoint"


def _llm_texts(output: dict) -> str:
    keys = [k for k in output if k.endswith("_response")] + ["query_analysis", "direct_output"]
    return "\n".join(str(output.get(k, "")) for k in keys)


def infra_errors(output: dict) -> List[str]:
    """Names of the infrastructure errors visible in one solve.py output file."""
    found = set()
    memory = output.get("memory") or {}
    steps = memory.values() if isinstance(memory, dict) else memory
    tool_text = "\n".join(repr(step.get("result")) if isinstance(step, dict) else repr(step) for step in steps)
    for name, pattern in TOOL_ERROR_PATTERNS:
        if re.search(pattern, tool_text):
            found.add(name)
    if re.search(PLANNER_ERROR_PATTERN, _llm_texts(output)):
        found.add("planner_endpoint")
    return sorted(found)


def tools_from_log(log_text: str) -> Optional[List[str]]:
    """The tool list solve.py printed after loading its tools, or None."""
    match = re.search(r"Final available tools: (\[.*?\])", log_text)
    if not match:
        return None
    try:
        return list(ast.literal_eval(match.group(1)))
    except (ValueError, SyntaxError):
        return None


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def plan_chunks(todo: Dict[Tuple[str, str], Sequence[int]], chunk_size: int) -> List[Tuple[str, str, List[int]]]:
    """Split each (model, task) to-do list into chunks and interleave them.

    Chunks for the same questions are placed next to each other for the two
    models, and tasks alternate, so a temporary API problem hits both models
    and all benchmarks about equally instead of one block of work.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    split = {}
    for key, indices in todo.items():
        ordered = sorted(indices)
        split[key] = [ordered[i:i + chunk_size] for i in range(0, len(ordered), chunk_size)]
    longest = max((len(c) for c in split.values()), default=0)
    plan = []
    for k in range(longest):
        for task in TASKS:
            for model in MODELS:
                chunks = split.get((model, task), [])
                if k < len(chunks):
                    plan.append((model, task, chunks[k]))
    return plan


def classify_poll_error(exc: BaseException) -> str:
    """How to read an exception from Modal's FunctionCall.get(timeout=0).

    'pending'   the call is still running (Modal raises the builtin TimeoutError;
                older versions raised modal.exception.TimeoutError itself)
    'failed'    the call is over and failed (e.g. FunctionTimeoutError, a subclass)
    'transient' anything else, e.g. a network hiccup of the local client
    """
    kind = type(exc)
    if kind is TimeoutError or (kind.__name__ == "TimeoutError" and kind.__module__.startswith("modal")):
        return "pending"
    if kind.__name__ in {"FunctionTimeoutError", "OutputExpiredError", "InputCancellation",
                         "RemoteError", "ExecutionError", "InvalidError"}:
        return "failed"
    return "transient"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def mcnemar_exact_p(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar test (binomial test on the discordant pairs)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_counts(base: Dict[str, bool], lora: Dict[str, bool]) -> Dict[str, int]:
    """Per-question agreement between the two models on the questions both answered."""
    common = sorted(set(base) & set(lora), key=lambda p: int(p) if str(p).isdigit() else str(p))
    counts = {"n": len(common), "both": 0, "only_base": 0, "only_lora": 0, "neither": 0}
    for pid in common:
        b, l = bool(base[pid]), bool(lora[pid])
        key = "both" if b and l else "only_base" if b else "only_lora" if l else "neither"
        counts[key] += 1
    return counts


def load_verdicts(finalresults_path: Path) -> Dict[str, bool]:
    data = json.loads(Path(finalresults_path).read_text())
    return {str(pid): bool(item.get("true_false")) for pid, item in data.items()}


def pct(correct: int, total: int) -> str:
    return f"{100 * correct / total:.1f}" if total else "n/a"


def summarize(verdicts: Dict[Tuple[str, str], Dict[str, bool]],
              infra: Dict[Tuple[str, str], int],
              limit: int) -> Tuple[str, str]:
    """Markdown table and CSV comparing base and LoRA on the same questions.

    verdicts[(model, task)] maps question id -> judged correct.
    infra[(model, task)] is the number of questions whose final run still hit an
    infrastructure error after all retries.
    """
    lines = [
        "| Benchmark | Questions | Base | LoRA | Change | LoRA-only correct | Base-only correct | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    pooled = {"n": 0, "both": 0, "only_base": 0, "only_lora": 0, "neither": 0}
    for task in TASKS:
        base = verdicts.get(("base", task))
        lora = verdicts.get(("lora", task))
        if base is None and lora is None:
            continue
        if base is None or lora is None:
            missing = "base" if base is None else "LoRA"
            lines.append(f"| {task} | {missing} not finished | | | | | | |")
            continue
        c = paired_counts(base, lora)
        for k in pooled:
            pooled[k] += c[k]
        b_ok = c["both"] + c["only_base"]
        l_ok = c["both"] + c["only_lora"]
        p = mcnemar_exact_p(c["only_base"], c["only_lora"])
        change = 100 * (l_ok - b_ok) / c["n"] if c["n"] else 0.0
        lines.append(f"| {task} | {c['n']} | {pct(b_ok, c['n'])} | {pct(l_ok, c['n'])} | "
                     f"{change:+.1f} | {c['only_lora']} | {c['only_base']} | {p:.2f} |")
        rows.append({
            "benchmark": task, "questions": c["n"],
            "base_correct": b_ok, "lora_correct": l_ok,
            "base_accuracy": pct(b_ok, c["n"]), "lora_accuracy": pct(l_ok, c["n"]),
            "only_lora": c["only_lora"], "only_base": c["only_base"], "mcnemar_p": f"{p:.4f}",
            "base_infra_errors": infra.get(("base", task), 0),
            "lora_infra_errors": infra.get(("lora", task), 0),
        })
    if pooled["n"]:
        b_ok = pooled["both"] + pooled["only_base"]
        l_ok = pooled["both"] + pooled["only_lora"]
        p = mcnemar_exact_p(pooled["only_base"], pooled["only_lora"])
        change = 100 * (l_ok - b_ok) / pooled["n"]
        lines.append(f"| **all** | {pooled['n']} | {pct(b_ok, pooled['n'])} | {pct(l_ok, pooled['n'])} | "
                     f"{change:+.1f} | {pooled['only_lora']} | {pooled['only_base']} | {p:.2f} |")
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return "\n".join(lines), buf.getvalue()


def cost_estimate(gpu_seconds: float, gpu_usd_per_hour: float = 0.80, overhead_usd_per_hour: float = 0.16) -> float:
    """Rough Modal cost: L4 plus the requested CPU and memory."""
    return gpu_seconds / 3600 * (gpu_usd_per_hour + overhead_usd_per_hour)


def parse_list(value: str, allowed: Iterable[str]) -> List[str]:
    allowed = list(allowed)
    items = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in items if v not in allowed]
    if bad or not items:
        raise ValueError(f"expected a comma-separated subset of {allowed}, got {value!r}")
    return items
