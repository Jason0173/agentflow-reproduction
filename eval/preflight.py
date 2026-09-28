"""Checks that run before any GPU time is spent.

Run from test/ (like solve.py) inside the evaluation image:
    cd test && python ../eval/preflight.py

It verifies the benchmark files, that every AgentFlow tool loads, that each
API key works through the same code path the benchmark uses (one real call per
tool and one judge call), and whether the base and LoRA tokenizers render the
same chat prompt. The last line of output is `PREFLIGHT {json report}`.
"""

import hashlib
import json
import os
import sys
import traceback

sys.path.insert(0, os.getcwd())  # test/, for calculate_score_unified
import fair_eval_core as core  # noqa: E402

REPORT = {"checks": {}, "ok": True}


def check(name, required=True):
    def wrap(fn):
        try:
            detail = fn()
            REPORT["checks"][name] = {"ok": True, "detail": detail}
        except Exception as exc:  # noqa: BLE001
            REPORT["checks"][name] = {
                "ok": False,
                "required": required,
                "detail": f"{type(exc).__name__}: {exc}"[:1500],
                "trace": traceback.format_exc()[-1500:],
            }
            if required:
                REPORT["ok"] = False
        return fn
    return wrap


@check("benchmark_files")
def _data():
    for task in core.TASKS:
        digest = hashlib.sha256(open(f"{task}/data/data.json", "rb").read()).hexdigest()
        assert digest == core.DATA_SHA256[task], f"{task}: sha256 {digest} does not match"
    return "all five match the pinned upstream files"


TOOLS = {}


@check("tools_load")
def _tools():
    from agentflow.agentflow.models.initializer import Initializer

    enabled = core.ENABLED_TOOLS.split(",")
    init = Initializer(enabled_tools=enabled, tool_engine=["Default"] * len(enabled),
                       model_string=core.PLANNER_ENGINE, verbose=False, vllm_config_path=None,
                       base_url=None, check_model=True)
    missing = set(core.EXPECTED_TOOLS) - set(init.available_tools)
    assert not missing, f"tools failed to load: {sorted(missing)} (loaded {init.available_tools})"
    TOOLS["cache"] = init.tool_instances_cache
    return sorted(init.available_tools)


def run_tool(tool_name, command):
    from agentflow.agentflow.models.executor import Executor

    executor = Executor(llm_engine_name="gpt-4o-mini", root_cache_dir="/tmp/preflight_cache",
                        verbose=False, tool_instances_cache=TOOLS.get("cache"))
    executor.set_query_cache_dir("/tmp/preflight_cache/0")
    result = executor.execute_tool_command(tool_name, command)
    errors = core.infra_errors({"memory": {"Action Step 1": {"result": result}}})
    assert not errors, f"{tool_name} returned an error ({errors}): {str(result)[:800]}"
    return result


@check("google_search")
def _google():
    model = os.environ.get("GOOGLE_SEARCH_MODEL", "gemini-2.5-flash")
    result = run_tool("Ground_Google_Search_Tool", 'execution = tool.execute(query="What is the capital of France?")')
    assert "paris" in str(result).lower(), f"unexpected answer: {str(result)[:500]}"
    return f"{model}: ok"


@check("generalist_gpt4o_mini")
def _generator():
    result = run_tool("Generalist_Solution_Generator_Tool",
                      'execution = tool.execute(query="What is 2 + 3? Reply with the number only.")')
    assert "5" in str(result), f"unexpected answer: {str(result)[:500]}"
    return "ok"


def wikipedia_diagnosis():
    """HTTP status and start of the body of one raw Wikipedia API call, with the
    User-Agent the wikipedia package now sends, to explain a failure."""
    import requests
    import wikipedia

    try:
        r = requests.get("https://en.wikipedia.org/w/api.php",
                         params={"action": "query", "list": "search", "srsearch": "Eiffel Tower", "format": "json"},
                         headers={"User-Agent": wikipedia.wikipedia.USER_AGENT}, timeout=20)
        return (f"raw API call: HTTP {r.status_code}, {r.headers.get('content-type')}, "
                f"user agent {wikipedia.wikipedia.USER_AGENT!r}, body starts {r.text[:200]!r}")
    except Exception as exc:  # noqa: BLE001
        return f"raw API call failed: {exc}"


@check("wikipedia_search")
def _wikipedia():
    try:
        result = run_tool("Wikipedia_RAG_Search_Tool", 'execution = tool.execute(query="Eiffel Tower height")')
        assert "eiffel" in str(result).lower(), f"unexpected result: {str(result)[:500]}"
    except AssertionError as exc:
        raise AssertionError(f"{exc}\n    {wikipedia_diagnosis()}") from None
    return "ok"


@check("judge_gpt4o")
def _judge():
    """The judge answers through the same code and prompt as the benchmark scoring.
    The agent puts its final answer in <answer> tags, which the scorer extracts first."""
    from calculate_score_unified import ResultScorer

    scorer = ResultScorer(judge_prompt="open_qa")
    cases = [  # (question, response, correct answer, expected verdict)
        ("What is the capital of France?", "It is Paris. <answer>Paris</answer>", "Paris", True),
        ("What is the capital of France?", "It is Lyon. <answer>Lyon</answer>", "Paris", False),
        ("What rocket launched Voyager 2?", "<answer>A Titan IIIE/Centaur rocket</answer>", "['Titan IIIE']", True),
        ("Who was US president in 1812?", "<answer>Thomas Jefferson</answer>", "['james madison']", False),
    ]
    for question, response, gold, expected in cases:
        why, verdict = scorer.answer_verification(question, response, gold)
        assert verdict is expected, f"judge said {verdict} for {response!r} vs {gold!r}: {why!r}"
    return f"open-QA judge prompt: {len(cases)} of {len(cases)} test verdicts as expected"


@check("lora_applies")
def _lora():
    """The served LoRA checkpoint must differ from the served base checkpoint only
    in weight matrices the adapter targets, so a silently ignored adapter (or any
    other difference between the two models) is caught before any GPU time."""
    from pathlib import Path

    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    def weights(folder):
        out = {}
        for f in sorted(Path(folder).glob("*.safetensors")):
            out.update(load_file(str(f)))
        return out

    base, lora = weights(os.environ["SERVED_BASE"]), weights(os.environ["SERVED_LORA"])
    assert set(base) == set(lora), "the two checkpoints contain different tensors"
    with safe_open(os.path.join("..", core.LORA_DIR, "adapter_model.safetensors"), "pt") as f:
        expected = sum(1 for k in f.keys() if "lora_B" in k)
    targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    changed = [n for n in base if not torch.equal(base[n], lora[n])]
    outside = [n for n in changed if n.split(".")[-2] not in targets]
    assert not outside, f"weights outside the LoRA target modules differ: {outside[:5]}"
    assert changed, "the LoRA checkpoint is identical to the base checkpoint"
    rel = sum(((lora[n].float() - base[n].float()).norm() / base[n].float().norm()).item()
              for n in changed) / len(changed)
    return (f"{len(changed)} of the adapter's {expected} target matrices differ from the base "
            f"(mean relative change {rel:.1e}); nothing else differs")


MESSAGES = [{"role": "system", "content": "You are a planner."},
            {"role": "user", "content": "Which tool should answer: who wrote Hamlet?"}]


def same_prompt(path_a, path_b):
    from transformers import AutoTokenizer

    a = AutoTokenizer.from_pretrained(path_a, trust_remote_code=True)
    b = AutoTokenizer.from_pretrained(path_b, trust_remote_code=True)
    render = lambda tok: tok.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
    ra, rb = render(a), render(b)
    return ra == rb and a.encode(ra) == b.encode(rb), ra, rb


@check("tokenizers")
def _tokenizers():
    same, b, l = same_prompt(os.environ["SERVED_BASE"], os.environ["SERVED_LORA"])
    assert same, f"the served base and LoRA checkpoints render different prompts:\n{b!r}\n{l!r}"
    return "both served checkpoints render identical chat prompts"


@check("adapter_tokenizer", required=False)
def _adapter_tokenizer():
    """Information only: whether the tokenizer saved with the adapter (not used
    here) renders the same prompt as the base model's tokenizer."""
    same, b, l = same_prompt(os.environ["SERVED_BASE"], os.path.join("..", core.LORA_DIR))
    assert same, f"the adapter's own tokenizer renders a different prompt:\n{b!r}\n{l!r}"
    return "the tokenizer saved with the adapter renders the same prompt as the base tokenizer"


if __name__ == "__main__":
    for name, result in REPORT["checks"].items():
        print(f"{'ok  ' if result['ok'] else 'FAIL'} {name}: {result['detail']}")
    print("PREFLIGHT " + json.dumps(REPORT))
    sys.exit(0 if REPORT["ok"] else 1)
