"""Unit tests for the fair base-vs-LoRA evaluation (eval/). No GPU, Modal or API key needed."""

import json
import re
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import chunk_runner as cr  # noqa: E402
import fair_eval_core as core  # noqa: E402

BENCH_SCRIPT = (ROOT / "test" / "run_lora_bench.sh").read_text()


def script_var(name):
    match = re.search(rf'^{name}="([^"]*)"', BENCH_SCRIPT, re.M)
    assert match, f"{name} not found in run_lora_bench.sh"
    return match.group(1)


def script_call(program):
    """The arguments of one `python <program> ...` call in run_lora_bench.sh, as {flag: value}."""
    match = re.search(rf"python {re.escape(program)} (.*?)(?:\| tee|\n\s*\n)", BENCH_SCRIPT, re.S)
    assert match, f"call to {program} not found"
    tokens = match.group(1).replace("\\\n", " ").split()
    flags = {}
    for i, tok in enumerate(tokens):
        if tok.startswith("--"):
            flags[tok] = tokens[i + 1].strip('"') if i + 1 < len(tokens) else None
    return flags


def as_flags(cmd):
    return {tok: cmd[i + 1] for i, tok in enumerate(cmd) if tok.startswith("--")}


# --- the runner uses exactly the team's LoRA benchmark configuration ---------------------------

def test_client_settings_match_run_lora_bench():
    assert core.PLANNER_ENGINE == script_var("LLM")
    assert core.ENABLED_TOOLS == script_var("ENABLED_TOOLS")
    assert core.TOOL_ENGINE == script_var("TOOL_ENGINE")
    assert core.MODEL_ENGINE == script_var("MODEL_ENGINE")


def test_solve_command_matches_run_lora_bench():
    ours = as_flags(core.solve_command("hotpotqa", 7, "hotpotqa/data/data.json", "c", "o"))
    theirs = script_call("solve.py")
    assert set(ours) == set(theirs)
    for flag in ("--output_types", "--max_time", "--max_steps", "--temperature"):
        assert ours[flag] == theirs[flag], flag
    assert ours["--llm_engine_name"] == script_var("LLM")


def test_score_command_matches_run_lora_bench():
    ours = as_flags(core.score_command("gaia", "gaia/data/data.json", "gaia/results/x"))
    theirs = script_call("calculate_score_unified.py")
    assert set(theirs) <= set(ours)
    assert set(ours) - set(theirs) == {"--max_workers"}  # only how many judge calls run at once
    for flag in ("--response_type", "--output_file"):
        assert ours[flag] == theirs[flag]


def test_every_task_has_a_pinned_data_file():
    assert set(core.DATA_SHA256) == set(core.TASKS)
    assert all(len(h) == 64 for h in core.DATA_SHA256.values())
    assert core.UPSTREAM_COMMIT in core.data_url("gaia")


def test_labels():
    assert core.label("base", "fair50") == "Qwen3.5-0.8B-base-fair50"
    assert core.label("lora", "smoke") == "Qwen3.5-0.8B-LoRA-smoke"
    with pytest.raises(ValueError):
        core.label("other", "x")


def test_server_env_differs_only_in_checkpoint():
    base = cr.server_env("/models/served/base", 8765)
    lora = cr.server_env("/models/served/lora", 8765)
    assert base["MERGED_MODEL"] == "/models/served/base" and lora["MERGED_MODEL"] == "/models/served/lora"
    assert {k: v for k, v in base.items() if k != "MERGED_MODEL"} == \
           {k: v for k, v in lora.items() if k != "MERGED_MODEL"}
    assert "LORA_DIR" not in base and "BASE_MODEL" not in lora


# --- infrastructure-error detection ---------------------------------------------------------------

def output(result, direct="<answer>Paris</answer>"):
    return {"memory": {"Action Step 1": {"tool_name": "Ground_Google_Search_Tool", "result": result}},
            "direct_output": direct}


@pytest.mark.parametrize("result, expected", [
    (["Paris is the capital of France."], []),
    (["The 1998 census counted 429 residents."], []),  # a number is not an error
    ("Not result is generated due to the tool not found.", []),  # a planner mistake, not infrastructure
    (["Google Search tried 5 times but failed. Last error: 503 UNAVAILABLE"], ["google_search_failed"]),
    (["429 RESOURCE_EXHAUSTED. Your prepayment credits are depleted."], ["quota_or_rate_limit"]),
    ([{"error": "rate_limit", "message": "Error code: 429 - rate limit"}], ["openai_error", "quota_or_rate_limit"]),
    ("Error generating response: Connection reset", ["tool_exception"]),
    ("Execution timed out after 120 seconds", ["tool_timeout"]),
    ("400 API key not valid. Please pass a valid API key.", ["api_key"]),
])
def test_infra_errors_in_tool_results(result, expected):
    assert core.infra_errors(output(result)) == sorted(expected)


def test_planner_endpoint_errors_are_detected():
    out = output(["ok"], direct="Error calling Modal endpoint: 500 Server Error")
    assert core.infra_errors(out) == ["planner_endpoint"]


def test_tools_from_log():
    log = "noise\n✅ Final available tools: ['A_Tool', 'B_Tool']\nmore"
    assert core.tools_from_log(log) == ["A_Tool", "B_Tool"]
    assert core.tools_from_log("no tools line") is None


# --- scheduling ---------------------------------------------------------------------------------

def test_plan_chunks_covers_everything_and_interleaves_models():
    todo = {(m, t): list(range(25)) for m in core.MODELS for t in ("hotpotqa", "gaia")}
    plan = core.plan_chunks(todo, 10)
    covered = {}
    for model, task, idx in plan:
        covered.setdefault((model, task), []).extend(idx)
    assert {k: sorted(v) for k, v in covered.items()} == todo
    assert [(m, t) for m, t, _ in plan[:4]] == [("base", "hotpotqa"), ("lora", "hotpotqa"),
                                               ("base", "gaia"), ("lora", "gaia")]
    assert plan[0][2] == plan[1][2]  # both models get the same questions side by side


# --- summary statistics -------------------------------------------------------------------------

def test_mcnemar_exact():
    assert core.mcnemar_exact_p(0, 0) == 1.0
    assert core.mcnemar_exact_p(3, 3) == 1.0
    assert core.mcnemar_exact_p(0, 5) == pytest.approx(2 / 32)
    assert core.mcnemar_exact_p(1, 9) == pytest.approx(2 * 11 / 1024)


def test_summarize_pairs_questions():
    base = {"0": True, "1": False, "2": False, "3": True}
    lora = {"0": True, "1": True, "2": True, "3": False, "4": True}  # "4" has no base answer: left out
    table, csv_text = core.summarize({("base", "hotpotqa"): base, ("lora", "hotpotqa"): lora},
                                     {("lora", "hotpotqa"): 1}, limit=5)
    row = [l for l in table.splitlines() if l.startswith("| hotpotqa")][0]
    assert row == "| hotpotqa | 4 | 50.0 | 75.0 | +25.0 | 2 | 1 | 1.00 |"
    assert "lora_infra_errors" in csv_text.splitlines()[0]
    assert csv_text.splitlines()[1].endswith(",0,1")


# --- the question runner, with a stand-in for solve.py ------------------------------------------

FAKE_SOLVE = textwrap.dedent("""
    import json, os, sys
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    tools = ["Generalist_Solution_Generator_Tool", "Ground_Google_Search_Tool",
             "Web_RAG_Search_Tool", "Wikipedia_RAG_Search_Tool"]
    if os.environ.get("FAKE_TOOLS_MISSING"):
        tools = tools[:1]
    print("Final available tools:", tools)
    if os.environ.get("FAKE_NO_OUTPUT"):
        sys.exit(1)
    result = os.environ.get("FAKE_RESULT", "Paris")
    out = {"pid": args["--index"], "memory": {"Action Step 1": {"result": result}}, "direct_output": "x"}
    path = os.path.join(args["--output_json_dir"], "output_%s.json" % args["--index"])
    json.dump(out, open(path, "w"))
""")


@pytest.fixture
def fake_solve(tmp_path, monkeypatch):
    script = tmp_path / "fake_solve.py"
    script.write_text(FAKE_SOLVE)
    real = core.solve_command

    def command(task, index, data_file, cache_dir, out_dir):
        return [sys.executable, str(script)] + real(task, index, data_file, cache_dir, out_dir)[2:]

    monkeypatch.setattr(core, "solve_command", command)
    return tmp_path


def run(tmp, indices, **env):
    import os
    return cr.run_questions("bamboogle", indices, tmp / "store", tmp, dict(os.environ, **env),
                            work_dir=tmp / "work")


def test_clean_questions_are_accepted(fake_solve):
    summary = run(fake_solve, [0, 1])
    assert summary["accepted"] == [0, 1]
    assert cr.todo(fake_solve / "store", 3) == [2]
    assert (fake_solve / "store" / "logs" / "0.log").exists()


def test_errored_question_is_retried_then_kept(fake_solve):
    bad = {"FAKE_RESULT": "Google Search tried 5 times but failed. Last error: 429"}
    for attempt in range(1, core.MAX_ATTEMPTS):
        assert run(fake_solve, [0], **bad)["set_aside"] == [0]
        assert cr.todo(fake_solve / "store", 1) == [0]
        assert (fake_solve / "store" / "errored" / f"output_0.attempt{attempt}.json").exists()
    assert run(fake_solve, [0], **bad)["accepted"] == [0]  # last attempt is kept, and reported
    assert cr.todo(fake_solve / "store", 1) == []
    rows = cr.question_rows(fake_solve / "store", 1)
    assert rows[0]["attempts"] == core.MAX_ATTEMPTS and "google_search_failed" in rows[0]["final_errors"]


def test_repeated_infrastructure_errors_stop_the_chunk(fake_solve):
    with pytest.raises(cr.InfraFailure, match="in a row"):
        run(fake_solve, [0, 1, 2, 3], FAKE_RESULT="429 RESOURCE_EXHAUSTED")
    assert cr.load_meta(fake_solve / "store", 3)["attempts"] == []  # never started


def test_question_without_output_is_scored_not_dropped(fake_solve):
    for _ in range(core.MAX_ATTEMPTS):
        run(fake_solve, [0], FAKE_NO_OUTPUT="1")
    out = json.loads((fake_solve / "store" / "results" / "output_0.json").read_text())
    assert out["direct_output"] == "" and out["pid"] == "0" and "no_output" in out["stub"]
    assert cr.todo(fake_solve / "store", 1) == []


def test_missing_tools_stop_immediately_without_using_an_attempt(fake_solve):
    with pytest.raises(cr.InfraFailure, match="tools did not load"):
        run(fake_solve, [0, 1], FAKE_TOOLS_MISSING="1")
    assert cr.load_meta(fake_solve / "store", 0)["attempts"] == []
    assert (fake_solve / "store" / "errored" / "0.tools_missing.log").exists()


# --- coordinator polling ------------------------------------------------------------------------

class FunctionTimeoutError(Exception):  # stand-ins for modal.exception classes
    pass


FunctionTimeoutError.__module__ = "modal.exception"


def test_classify_poll_error():
    assert core.classify_poll_error(TimeoutError()) == "pending"  # what Modal 1.5 raises while running
    assert core.classify_poll_error(FunctionTimeoutError()) == "failed"
    assert core.classify_poll_error(ConnectionError("reset")) == "transient"


class FakeCall:
    def __init__(self, polls_until_done, result):
        self.polls, self.result, self.cancelled = polls_until_done, result, False

    def get(self):
        if self.polls > 0:
            self.polls -= 1
            raise TimeoutError()
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def round_with(calls):
    plan = [("base", "gaia", [i]) for i in range(len(calls))]
    it = iter(calls)
    return cr.run_round(plan, spawn=lambda item: next(it), get=lambda c: c.get(),
                        cancel=lambda c: setattr(c, "cancelled", True), sleep=lambda s: None)


def test_run_round_waits_for_running_chunks():
    calls = [FakeCall(3, {"accepted": [0]}), FakeCall(0, {"accepted": [1]})]
    out = round_with(calls)
    assert len(out["results"]) == 2 and not out["failures"] and out["stopped"] is None
    assert not any(c.cancelled for c in calls)


def test_run_round_stops_and_cancels_when_tools_are_missing():
    calls = [FakeCall(0, {"error": "InfraFailure: AgentFlow tools did not load (got [])"}),
             FakeCall(100, {"accepted": [1]})]
    out = round_with(calls)
    assert out["stopped"] == "the AgentFlow tools did not load"
    assert calls[1].cancelled


def test_run_round_counts_timed_out_chunks_as_failures():
    calls = [FakeCall(0, FunctionTimeoutError("6h")) for _ in range(3)] + [FakeCall(100, {})]
    out = round_with(calls)
    assert len(out["failures"]) == 3 and out["stopped"] == "3 chunks failed"
    assert calls[3].cancelled


def test_run_names_are_validated():
    assert core.validate_run_name("fair50") == "fair50"
    for bad in ("my results", "results2", "a/b", ""):
        with pytest.raises(ValueError):
            core.validate_run_name(bad)


def test_summary_shows_benchmarks_that_did_not_finish():
    table, _ = core.summarize({("base", "gaia"): {"0": True}}, {}, limit=1)
    assert "| gaia | LoRA not finished |" in table
