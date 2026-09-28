"""Run benchmark questions against one planner server and store the results.

Used inside the Modal containers of eval/modal_fair_eval.py, but has no Modal
dependency, so it can also be run on any machine with a GPU (or tested against
a fake planner server).

Storage layout for one (model, task) under `store`:
    results/output_<i>.json   accepted solve.py output (read by the scorer)
    logs/<i>.log              solve.py log for the accepted output
    errored/                  outputs and logs of attempts that hit an infrastructure error
    meta/<i>.json             one record per attempt: wall time, return code, errors
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import fair_eval_core as core


class InfraFailure(RuntimeError):
    """The environment, not the model, is failing; stop spending GPU time."""


def write_json(path: Path, obj) -> None:
    """Write atomically, so an interrupted container never leaves a truncated file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def stub_output(index: int, reason: str) -> dict:
    """Stand-in for a question that never produced an output: scored as wrong."""
    return {"pid": str(index), "query": None, "answer": None, "memory": {}, "direct_output": "",
            "stub": reason}


def server_env(model_path: str, port: int) -> Dict[str, str]:
    """Environment for serve_lora_local.py. Both models are served from a merged
    checkpoint through the same code with the same generation settings."""
    env = dict(os.environ)
    for key in ("MERGED_MODEL", "BASE_MODEL", "LORA_DIR", "MERGE_ADAPTER"):
        env.pop(key, None)
    env.update({
        "MERGED_MODEL": model_path,
        "HOST": "127.0.0.1",
        "PORT": str(port),
        "MAX_NEW_TOKENS": "2048",
        "DTYPE": "bfloat16",
        "ENABLE_THINKING": "false",
    })
    return env


def start_server(model_path: str, repo: Path, port: int, log_path: Path,
                 command: Optional[List[str]] = None) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = command or [sys.executable, str(repo / "serve_lora_local.py")]
    log = open(log_path, "w")
    return subprocess.Popen(cmd, cwd=str(repo), env=server_env(model_path, port),
                            stdout=log, stderr=subprocess.STDOUT)


def tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def wait_ready(proc: subprocess.Popen, port: int, log_path: Path, timeout: float = 900) -> dict:
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise InfraFailure(f"planner server exited with code {proc.returncode}:\n{tail(log_path)}")
        try:
            health = requests.get(f"http://127.0.0.1:{port}/health", timeout=5).json()
            if health.get("model_loaded", True):
                return health
        except Exception:  # noqa: BLE001 - server still starting
            pass
        time.sleep(3)
    raise InfraFailure(f"planner server not ready after {timeout:.0f} s:\n{tail(log_path)}")


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def load_meta(store: Path, index: int) -> dict:
    path = store / "meta" / f"{index}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"index": index, "attempts": []}


def is_finished(store: Path, index: int) -> bool:
    """Accepted, or out of attempts."""
    if (store / "results" / f"output_{index}.json").exists():
        return True
    return len(load_meta(store, index)["attempts"]) >= core.MAX_ATTEMPTS


def todo(store: Path, limit: int) -> List[int]:
    return [i for i in range(limit) if not is_finished(store, i)]


def attempts_made(store: Path, limit: int) -> int:
    return sum(len(load_meta(store, i)["attempts"]) for i in range(limit)) if store.exists() else 0


def run_questions(
    task: str,
    indices: List[int],
    store: Path,
    test_dir: Path,
    env: Dict[str, str],
    server: Optional[subprocess.Popen] = None,
    server_log: Optional[Path] = None,
    commit: Callable[[], None] = lambda: None,
    question_timeout: float = 1500,
    max_consecutive_errors: int = 3,
    work_dir: Path = Path("/tmp/fair_eval_work"),
) -> dict:
    """Run solve.py for each index, exactly as test/run_lora_bench.sh does.

    An attempt that shows an infrastructure error is set aside in errored/
    and retried later, until it has had core.MAX_ATTEMPTS tries; the last
    attempt is then kept as it is (and reported). Raises InfraFailure when
    the tools did not load, the server died, or several questions in a row
    failed for infrastructure reasons.
    """
    for sub in ("results", "logs", "errored", "meta"):
        (store / sub).mkdir(parents=True, exist_ok=True)
    out_dir = work_dir / "out"
    cache_dir = work_dir / "cache"
    summary = {"accepted": [], "set_aside": [], "wall_s": {}}
    consecutive = 0

    for index in indices:
        if is_finished(store, index):
            continue
        if server is not None and server.poll() is not None:
            raise InfraFailure(f"planner server died (code {server.returncode}):\n{tail(server_log)}")

        meta = load_meta(store, index)
        attempt = len(meta["attempts"]) + 1
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True)
        log_path = work_dir / f"{index}.log"
        cmd = core.solve_command(task, index, f"{task}/data/data.json", str(cache_dir), str(out_dir))

        started = time.time()
        timed_out = False
        with open(log_path, "w") as log:
            try:
                proc = subprocess.run(cmd, cwd=str(test_dir), env=env, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=question_timeout)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                returncode, timed_out = None, True
        wall = round(time.time() - started, 1)

        log_text = log_path.read_text(errors="replace")
        errors: List[str] = []
        tools = core.tools_from_log(log_text)
        tools_ok = tools is not None and not (set(core.EXPECTED_TOOLS) - set(tools))
        if not tools_ok:
            errors.append("tools_missing")
        out_file = out_dir / f"output_{index}.json"
        if out_file.exists():
            errors += core.infra_errors(json.loads(out_file.read_text()))
        else:
            errors.append("timeout" if timed_out else "no_output")

        if not tools_ok:  # the environment is broken; this does not use up one of the question's attempts
            shutil.copy(log_path, store / "errored" / f"{index}.tools_missing.log")
            commit()
            raise InfraFailure(f"AgentFlow tools did not load (got {tools}):\n{tail(log_path)}")

        last_try = attempt >= core.MAX_ATTEMPTS
        if last_try and not out_file.exists():
            # Keep the most recent earlier output if there is one; otherwise a stub
            # that the judge will mark wrong, so the question is not silently dropped.
            earlier = sorted((store / "errored").glob(f"output_{index}.attempt*.json"))
            if earlier:
                shutil.copy(earlier[-1], out_file)
            else:
                write_json(out_file, stub_output(index, ", ".join(errors)))
        accept = out_file.exists() and (not errors or last_try)
        if accept:
            shutil.copy(out_file, store / "results" / f"output_{index}.json")
            shutil.copy(log_path, store / "logs" / f"{index}.log")
            summary["accepted"].append(index)
        else:
            if out_file.exists():
                shutil.copy(out_file, store / "errored" / f"output_{index}.attempt{attempt}.json")
            shutil.copy(log_path, store / "errored" / f"{index}.attempt{attempt}.log")
            summary["set_aside"].append(index)
        meta["attempts"].append({
            "attempt": attempt, "wall_s": wall, "returncode": returncode,
            "errors": errors, "accepted": accept, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        write_json(store / "meta" / f"{index}.json", meta)
        summary["wall_s"][index] = wall
        commit()
        note = "" if accept else " (set aside, will retry)"
        print(f"[{task} #{index}] attempt {attempt}: {wall:.0f}s, {'ok' if not errors else ', '.join(errors)}{note}",
              flush=True)

        consecutive = consecutive + 1 if errors else 0
        if consecutive >= max_consecutive_errors:
            raise InfraFailure(f"{consecutive} questions in a row hit infrastructure errors "
                               f"(last: {errors}); stopping this chunk.\n{tail(log_path, 25)}")
    return summary


def score(store: Path, task: str, label: str, limit: int, test_dir: Path,
          commit: Callable[[], None] = lambda: None, env: Optional[Dict[str, str]] = None,
          retry_wait: float = 60) -> dict:
    """Score the accepted outputs with test/calculate_score_unified.py, the same
    scorer and gpt-4o judge as run_lora_bench.sh. Results go to store/final/."""
    final = store / "final"
    outputs = [p for p in (store / "results").glob("output_*.json") if int(p.stem.split("_")[1]) < limit]
    if not outputs:
        return {"error": "no accepted outputs to score"}
    scores_file = final / "final_scores_direct_output.json"
    if scores_file.exists() and json.loads(scores_file.read_text()).get("total") == len(outputs):
        return json.loads(scores_file.read_text())  # already scored
    shutil.rmtree(final, ignore_errors=True)  # never export scores of an older set of outputs
    result_dir, log_dir = test_dir / task / "results" / label, test_dir / task / "logs" / label
    for d in (result_dir, log_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for p in outputs:
        shutil.copy(p, result_dir / p.name)
        log = store / "logs" / f"{p.stem.split('_')[1]}.log"
        if log.exists():
            shutil.copy(log, log_dir / log.name)
    cmd = core.score_command(task, f"{task}/data/data.json", f"{task}/results/{label}", max_workers=2)
    proc = None
    for attempt in range(3):  # the judge caches its verdicts, so a retry only redoes the failed calls
        proc = subprocess.run(cmd, cwd=str(test_dir), env=env, capture_output=True, text=True)
        if proc.returncode == 0 and (result_dir / "final_scores_direct_output.json").exists():
            break
        print(f"scoring {label} {task} failed (attempt {attempt + 1}):\n{proc.stderr[-1500:]}", flush=True)
        if attempt < 2:
            time.sleep(retry_wait)
    else:
        return {"error": (proc.stdout[-2000:] + "\n" + proc.stderr[-2000:]) if proc else "not run"}
    final.mkdir(parents=True, exist_ok=True)
    for name in ("final_scores_direct_output.json", "finalresults_direct_output.json"):
        shutil.copy(result_dir / name, final / name)
    (final / "finalscore_direct_output.log").write_text(proc.stdout)
    commit()
    return json.loads(scores_file.read_text())


def question_rows(store: Path, limit: int) -> List[dict]:
    """One row per question for questions.csv: attempts, final errors, wall time."""
    rows = []
    for index in range(limit):
        meta = load_meta(store, index)
        attempts = meta["attempts"]
        last = attempts[-1] if attempts else {}
        rows.append({
            "index": index,
            "attempts": len(attempts),
            "accepted": (store / "results" / f"output_{index}.json").exists(),
            "final_errors": ";".join(last.get("errors", [])),
            "wall_s": last.get("wall_s", ""),
            "total_wall_s": round(sum(a.get("wall_s", 0) for a in attempts), 1),
        })
    return rows


def run_round(plan: List[tuple], spawn: Callable, get: Callable, cancel: Callable,
              tick: Callable[[], None] = lambda: None, sleep: Callable[[float], None] = time.sleep,
              poll_every: float = 20, max_failures: int = 3, max_transient: int = 5) -> dict:
    """Start one call per chunk and wait for all of them.

    get(handle) returns the chunk's result dict or raises; exceptions are read
    with core.classify_poll_error. A result with an "error" key is a failed
    chunk. The round stops early (cancelling every call still running) when
    the AgentFlow tools did not load or `max_failures` chunks failed; calls
    are also cancelled if anything else goes wrong here.
    """
    outcome = {"results": [], "failures": [], "stopped": None}
    active: List[list] = []
    try:
        for item in plan:
            active.append([item, spawn(item), 0])
        while active:
            still = []
            for item, handle, hiccups in active:
                try:
                    res = get(handle)
                except Exception as exc:  # noqa: BLE001
                    kind = core.classify_poll_error(exc)
                    if kind == "pending":
                        still.append([item, handle, 0])
                        continue
                    if kind == "transient" and hiccups + 1 < max_transient:
                        still.append([item, handle, hiccups + 1])
                        continue
                    try:
                        cancel(handle)
                    except Exception:  # noqa: BLE001
                        pass
                    res = {"error": f"{type(exc).__name__}: {exc}"}
                name = f"{item[0]} {item[1]} {item[2][0]}-{item[2][-1]}"
                if res.get("error"):
                    outcome["failures"].append(f"{name}: {str(res['error'])[:1500]}")
                    print(f"CHUNK FAILED {name}: {str(res['error'])[:500]}", flush=True)
                else:
                    outcome["results"].append((item, res))
                    print(f"chunk done {name}: {len(res.get('accepted', []))} accepted, "
                          f"{len(res.get('set_aside', []))} set aside", flush=True)
            active = still
            if any("tools did not load" in f for f in outcome["failures"]):
                outcome["stopped"] = "the AgentFlow tools did not load"
            elif len(outcome["failures"]) >= max_failures:
                outcome["stopped"] = f"{len(outcome['failures'])} chunks failed"
            if outcome["stopped"]:
                break
            tick()
            if active:
                sleep(poll_every)
    finally:
        for _, handle, _ in active:
            try:
                cancel(handle)
            except Exception:  # noqa: BLE001
                pass
    return outcome
