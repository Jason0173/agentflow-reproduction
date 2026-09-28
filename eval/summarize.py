"""Summarize a finished base-vs-LoRA run into eval/RESULTS.md, eval/results.csv and a figure.

Reads only files that are committed with the results, so it can be re-run from
a fresh clone:

    python eval/summarize.py                # run "fair50", 50 questions
    python eval/summarize.py --run smoke --limit 2 --tasks bamboogle

Per model and benchmark it uses test/<task>/results/Qwen3.5-0.8B-{base,LoRA}-<run>/:
finalresults_direct_output.json (judge verdicts), questions.csv (attempts, errors,
time) and trajectories.jsonl.gz (the agent's steps and tool choices).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import fair_eval_core as core  # noqa: E402

NAMES = {"base": "Base", "lora": "LoRA"}
DISPLAY = {"bamboogle": "Bamboogle", "2wiki": "2Wiki", "hotpotqa": "HotpotQA", "musique": "Musique", "gaia": "GAIA"}
COLORS = {"base": "#2a78d6", "lora": "#eb6834"}  # categorical slots 1 and 2 (validated, light surface)
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"


def result_dir(repo: Path, task: str, model: str, run: str) -> Path:
    return repo / "test" / task / "results" / core.label(model, run)


def invalid_tool(step: dict) -> bool:
    """The planner named no tool, or one AgentFlow could not match (for example a
    tool name wrapped in backticks)."""
    name = step.get("tool_name")
    return name is None or str(name).startswith("No matched tool")


def behavior(folder: Path) -> Optional[dict]:
    path = folder / "trajectories.jsonl.gz"
    if not path.exists():
        return None
    steps, selections, invalid = [], 0, 0
    with gzip.open(path, "rt") as f:
        for line in f:
            out = json.loads(line)
            if out.get("stub"):
                continue
            steps.append(out.get("step_count") or 0)
            for step in (out.get("memory") or {}).values():
                selections += 1
                invalid += invalid_tool(step)
    times = []
    qcsv = folder / "questions.csv"
    if qcsv.exists():
        times = [float(r["wall_s"]) for r in csv.DictReader(qcsv.open()) if r["wall_s"]]
    return {
        "questions": len(steps),
        "mean_steps": statistics.mean(steps) if steps else 0.0,
        "one_step": sum(1 for s in steps if s == 1),
        "selections": selections,
        "invalid": invalid,
        "mean_time": statistics.mean(times) if times else 0.0,
        "time_total": sum(times),
    }


def remaining_errors(folder: Path) -> List[Tuple[str, str]]:
    qcsv = folder / "questions.csv"
    if not qcsv.exists():
        return []
    return [(r["index"], r["final_errors"]) for r in csv.DictReader(qcsv.open()) if r["final_errors"]]


def behavior_table(stats: Dict[Tuple[str, str], dict], tasks: List[str]) -> str:
    lines = [
        "| Benchmark | Steps per question (base / LoRA) | Answered after one step | "
        "Tool selections AgentFlow could not use | Seconds per question |",
        "|---|---:|---:|---:|---:|",
    ]
    pooled = {m: {"q": 0, "steps": 0.0, "one": 0, "sel": 0, "bad": 0, "time": 0.0} for m in core.MODELS}
    for task in tasks:
        b, l = stats.get(("base", task)), stats.get(("lora", task))
        if not b or not l:
            continue
        for m, s in (("base", b), ("lora", l)):
            p = pooled[m]
            p["q"] += s["questions"]
            p["steps"] += s["mean_steps"] * s["questions"]
            p["one"] += s["one_step"]
            p["sel"] += s["selections"]
            p["bad"] += s["invalid"]
            p["time"] += s["mean_time"] * s["questions"]
        lines.append(_behavior_row(task, b["questions"], b["mean_steps"], l["mean_steps"], b["one_step"], l["one_step"],
                                   b["invalid"], b["selections"], l["invalid"], l["selections"],
                                   b["mean_time"], l["mean_time"]))
    b, l = pooled["base"], pooled["lora"]
    if b["q"] and l["q"]:
        lines.append(_behavior_row("**all**", b["q"], b["steps"] / b["q"], l["steps"] / l["q"], b["one"], l["one"],
                                   b["bad"], b["sel"], l["bad"], l["sel"], b["time"] / b["q"], l["time"] / l["q"]))
    return "\n".join(lines)


def _behavior_row(name, n, b_steps, l_steps, b_one, l_one, b_bad, b_sel, l_bad, l_sel, b_time, l_time) -> str:
    def share(x, total):
        return f"{100 * x / total:.0f}%" if total else "n/a"
    return (f"| {name} | {b_steps:.1f} / {l_steps:.1f} | {share(b_one, n)} / {share(l_one, n)} | "
            f"{share(b_bad, b_sel)} / {share(l_bad, l_sel)} | {b_time:.0f} / {l_time:.0f} |")


def plot(verdicts, stats, tasks: List[str], out: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    shown = [t for t in tasks if ("base", t) in verdicts and ("lora", t) in verdicts and ("base", t) in stats]
    if not shown:
        return False
    acc = {m: [] for m in core.MODELS}
    steps = {m: [] for m in core.MODELS}
    for t in shown:
        common = set(verdicts[("base", t)]) & set(verdicts[("lora", t)])
        for m in core.MODELS:
            acc[m].append(100 * sum(verdicts[(m, t)][q] for q in common) / len(common))
            steps[m].append(stats[(m, t)]["mean_steps"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    x = range(len(shown))
    width = 0.36
    panels = [(axes[0], acc, "Accuracy on the same questions (%)", 50),
              (axes[1], steps, "Agent steps per question", 10)]
    for ax, data, title, top in panels:
        ax.set_facecolor(SURFACE)
        for k, m in enumerate(core.MODELS):
            ax.bar([i + (k - 0.5) * width for i in x], data[m], width=width, color=COLORS[m],
                   edgecolor=SURFACE, linewidth=2, label=NAMES[m], zorder=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels([DISPLAY.get(t, t) for t in shown], color=INK_2)
        ax.set_ylim(0, top)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(axis="both", colors=INK_2, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["Base (no training)", "Flow-GRPO + LoRA"], loc="lower center", ncol=2, frameon=False,
               labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return True


def write_summary(run: str = "fair50", limit: int = 50, tasks: Optional[List[str]] = None,
                  models: Optional[List[str]] = None, repo: Path = REPO) -> str:
    tasks = tasks or list(core.TASKS)
    models = models or list(core.MODELS)
    verdicts, infra, stats, errors = {}, {}, {}, []
    for m in models:
        for t in tasks:
            folder = result_dir(repo, t, m, run)
            fr = folder / "finalresults_direct_output.json"
            if fr.exists():
                verdicts[(m, t)] = core.load_verdicts(fr)
            left = remaining_errors(folder)
            infra[(m, t)] = len(left)
            errors += [f"{NAMES[m]} {t} #{i} ({e})" for i, e in left]
            s = behavior(folder)
            if s:
                stats[(m, t)] = s
    table, csv_text = core.summarize(verdicts, infra, limit)

    out_dir = repo / "eval" if run == "fair50" else repo / "eval" / "runs" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    figure = out_dir / "plots" / f"{run}.png"
    has_plot = plot(verdicts, stats, tasks, figure)

    report_file = repo / "eval" / "runs" / run / "REPORT.json"
    report = json.loads(report_file.read_text()) if report_file.exists() else {}
    pre_file = repo / "eval" / "runs" / run / "PREFLIGHT.json"
    pre = json.loads(pre_file.read_text()) if pre_file.exists() else {}
    adapter_tok = pre.get("checks", {}).get("adapter_tokenizer", {})
    gpu_s = sum(s["time_total"] for s in stats.values())

    lines = [
        f"# Base vs Flow-GRPO LoRA on the same questions (run `{run}`)",
        "",
        f"Qwen3.5-0.8B as the AgentFlow planner, on the first {limit} questions of each benchmark. "
        "Both models ran with the agent settings of `test/run_lora_bench.sh`, the same serving code "
        "(`serve_lora_local.py`, bf16, greedy decoding, thinking off), the same tools and the same judge. "
        "How the run was set up and why: [`README.md`](README.md).",
        "",
        "## Accuracy",
        "",
        table,
        "",
        "*Change* is LoRA minus base, in percentage points. *LoRA-only correct* and *Base-only correct* count the "
        "questions only one of the two models got right. The exact McNemar test gives the probability of a split "
        "at least that uneven if both models were equally good.",
        "",
    ]
    if has_plot:
        lines += [f"![Accuracy and steps per question](plots/{run}.png)", ""]
    if stats:
        lines += [
            "## How the planners behave",
            "",
            behavior_table(stats, tasks),
            "",
            "Each cell shows base / LoRA.",
            "",
            "- *Steps per question*: tool calls the agent made, with a limit of 10.",
            "- *Tool selections AgentFlow could not use*: the planner named no tool, or a name the framework could not "
            "match. This includes a valid name wrapped in backticks, which AgentFlow's parser rejects. The step is "
            "then wasted.",
            "- *Seconds per question*: wall time on one NVIDIA L4, including tool calls.",
            "",
        ]
    lines += [
        "## Setup",
        "",
        f"- **Weights:** `{core.BASE_MODEL_ID}` at revision `{report.get('base_revision', '?')}`. The LoRA "
        f"`{core.LORA_DIR}` is merged into the same weights in float32. Both checkpoints are served in bf16 and "
        "written by `eval/build_models.py`.",
        "- **Tokenizer:** the base model's, for both. The tokenizer saved with the adapter "
        f"{'renders the same prompts' if adapter_tok.get('ok') else adapter_tok.get('detail', 'was not checked')}.",
        f"- **Questions:** upstream AgentFlow `{core.UPSTREAM_COMMIT[:7]}`, SHA-256 checked.",
        "- **Tools:**",
        "  - Base_Generator (gpt-4o-mini).",
        f"  - Google Search (`{report.get('search_model', '?')}` with Google Search grounding).",
        "  - Wikipedia search: see the known issue below.",
        "- **Judge:** gpt-4o via `test/calculate_score_unified.py --judge_prompt open_qa`.",
        f"- **Hardware:** {', '.join(report.get('gpus', [])) or '?'} on Modal. The run finished on "
        f"{report.get('finished', '?')[:10]}. Planner time was {gpu_s / 3600:.1f} GPU-hours, not counting "
        "container start-up or interrupted work.",
        "",
        "## Notes",
        "",
    ]
    if errors:
        lines.append(f"- **Questions still showing a tool error after {core.MAX_ATTEMPTS} attempts:** "
                     + ", ".join(errors) + ". They are scored like every other question.")
    lines += [
        "- **Known issue: the Wikipedia tool.** It never reads page text. Upstream AgentFlow's "
        "`Wikipedia_Search_Tool` does not store its `model_string`, so creating its page reader fails "
        "(`Error creating Web RAG tool` in the logs) and the tool returns search titles only. This is the same "
        "for both models and in the team's runs, and was kept for comparability.",
        "- **Sample size:** 50 questions per benchmark is small. With the 5–18 discordant questions seen here, a "
        "benchmark needs a difference of roughly 15–20 points before the McNemar test puts it below p = 0.05.",
    ]
    text = "\n".join(lines) + "\n"
    (out_dir / "RESULTS.md").write_text(text)
    (out_dir / "results.csv").write_text(csv_text)
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="fair50")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--tasks", default=",".join(core.TASKS))
    args = parser.parse_args(argv)
    text = write_summary(core.validate_run_name(args.run), args.limit, core.parse_list(args.tasks, core.TASKS))
    print(text)


if __name__ == "__main__":
    main()
