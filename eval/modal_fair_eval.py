"""Fair base-vs-LoRA comparison of the Qwen3.5-0.8B AgentFlow planner, on Modal.

The base model and the Flow-GRPO LoRA model answer the same questions with the
same client settings as test/run_lora_bench.sh, the same serving code
(serve_lora_local.py), the same tools and the same judge. See eval/README.md.

    # once: API keys for the tools and the judge
    modal secret create agentflow-keys OPENAI_API_KEY=... GOOGLE_API_KEY=...

    modal run eval/modal_fair_eval.py --preflight-only                    # checks keys and tools, no GPU
    modal run eval/modal_fair_eval.py --run smoke --limit 2 --tasks bamboogle
    modal run --detach eval/modal_fair_eval.py                            # 50 questions x 5 benchmarks x 2 models
    modal run eval/modal_fair_eval.py --download-only                     # fetch results again later

Progress is kept in the Modal volume "agentflow-fair-eval", so an interrupted
run continues where it stopped when the same command is run again.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
for _path in (HERE, Path("/repo/eval")):
    if (_path / "fair_eval_core.py").exists():
        sys.path.insert(0, str(_path))
        break
import fair_eval_core as core  # noqa: E402

REPO = HERE.parent
APP_NAME = "agentflow-fair-eval"
VOLUME_NAME = "agentflow-fair-eval"
SECRET_NAME = "agentflow-keys"
BASE_MODEL_PATH = "/models/Qwen3.5-0.8B"          # the download from Hugging Face
ADAPTER_PATH = "/models/lora-adapter"              # results/final_qwen35_lora
SERVED = {"base": "/models/served/base", "lora": "/models/served/lora"}  # built by eval/build_models.py
PORT = 8765
VOL = Path("/vol")
# New Gemini API projects may not get gemini-2.5-flash (AgentFlow's default),
# so the search tool uses a current Flash-Lite model that supports Google
# Search grounding. Both planners get the same search model.
DEFAULT_SEARCH_MODEL = "gemini-3.5-flash-lite"

# Serving stack of the team's LoRA evaluation (docs/team_report.md, section 8).
SERVE_PINS = [
    "transformers==5.5.4", "peft==0.19.0", "accelerate==1.13.0",
    "sentencepiece==0.2.1", "fastapi==0.128.0", "uvicorn==0.31.1",
]
# AgentFlow client: versions from agentflow/requirements.txt and uv.lock.
AGENT_PINS = [
    "openai==1.75.0", "google-genai==2.25.0", "wikipedia==1.4.0", "diskcache==5.6.3",
    "tenacity==9.0.0", "python-dotenv==1.0.1", "pillow==11.1.0", "platformdirs==4.3.6",
    "agentops==0.4.18", "aiohttp==3.13.3", "flask==3.1.2", "graphviz==0.21", "numpy==2.4.1",
    "omegaconf==2.3.0", "psutil==7.0.0", "pydantic==2.12.5", "requests==2.32.5",
    "setproctitle==1.3.7", "tqdm==4.67.1",
]
DOWNLOAD_BASE = (
    "from huggingface_hub import HfApi, snapshot_download; "
    f"sha = HfApi().model_info('{core.BASE_MODEL_ID}').sha; "
    f"snapshot_download('{core.BASE_MODEL_ID}', revision=sha, local_dir='{BASE_MODEL_PATH}'); "
    f"open('{BASE_MODEL_PATH}.revision', 'w').write(sha); "
    "print('base model revision', sha)"
)
IGNORE = [
    ".git", ".venv", "**/__pycache__", "**/.DS_Store", "**/.pytest_cache",
    "test/*/results", "test/*/logs", "test/*/cache", "test/*/data", "eval/runs",
]


def _build_served_models():
    """Image build step: write the base and base+LoRA checkpoints (eval/build_models.py)."""
    subprocess.run([sys.executable, "/opt/build_models.py", BASE_MODEL_PATH, ADAPTER_PATH,
                    SERVED["base"], SERVED["lora"]], check=True)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(*SERVE_PINS, *AGENT_PINS)
    # Base weights are baked into the image, so every container (and both
    # models) uses the same snapshot. Its revision is recorded in the results.
    .run_commands(f'python -c "{DOWNLOAD_BASE}"')
    .run_commands(*[
        f"mkdir -p /data/{t} && curl -fsSL {core.data_url(t)} -o /data/{t}/data.json && "
        f"echo '{core.DATA_SHA256[t]}  /data/{t}/data.json' | sha256sum -c -"
        for t in core.TASKS
    ])
    # The two served checkpoints: base, and base with the LoRA update merged in,
    # both written by the same script (see eval/build_models.py).
    .add_local_dir(REPO / core.LORA_DIR, ADAPTER_PATH, copy=True)
    .add_local_file(HERE / "build_models.py", "/opt/build_models.py", copy=True)
    .add_local_file(HERE / "fair_eval_core.py", "/root/fair_eval_core.py", copy=True)  # imported by this file
    .run_function(_build_served_models, memory=16384, timeout=3600)
    .env({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
          "PYTHONPATH": "/repo/eval"})
    .add_local_dir(REPO, "/repo", copy=True, ignore=IGNORE)
    .run_commands(
        "cd /repo && pip install --no-deps -e ./agentflow -e .",
        *[f"mkdir -p /repo/test/{t}/data && cp /data/{t}/data.json /repo/test/{t}/data/data.json"
          for t in core.TASKS],
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
keys = modal.Secret.from_name(SECRET_NAME, required_keys=["OPENAI_API_KEY", "GOOGLE_API_KEY"])


def _store(run: str, model: str, task: str) -> Path:
    return VOL / run / core.label(model, run) / task


def _agent_env(search_model: str) -> dict:
    env = dict(os.environ)
    env["MODAL_PLANNER_URL"] = f"http://127.0.0.1:{PORT}/chat"
    env["GOOGLE_SEARCH_MODEL"] = search_model
    return env


# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------

@app.function(cpu=2.0, memory=8192, timeout=30 * 60, secrets=[keys], volumes={str(VOL): volume})
def preflight(run: str, search_model: str) -> dict:
    """Check data, tools, API keys and tokenizers without a GPU."""
    env = _agent_env(search_model)
    env["SERVED_BASE"], env["SERVED_LORA"] = SERVED["base"], SERVED["lora"]
    proc = subprocess.run([sys.executable, "/repo/eval/preflight.py"], cwd="/repo/test", env=env,
                          capture_output=True, text=True, timeout=25 * 60)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("PREFLIGHT ")]
    report = json.loads(lines[-1][len("PREFLIGHT "):]) if lines else {"ok": False, "checks": {}}
    report["search_model"] = search_model
    report["base_revision"] = Path(BASE_MODEL_PATH + ".revision").read_text().strip()
    if not lines:
        report["error"] = (proc.stdout[-3000:] + proc.stderr[-3000:])
    (VOL / run).mkdir(parents=True, exist_ok=True)
    (VOL / run / "PREFLIGHT.json").write_text(json.dumps(report, indent=1))
    volume.commit()
    return report


@app.function(gpu="L4", cpu=2.0, memory=8192, timeout=6 * 3600, secrets=[keys],
              volumes={str(VOL): volume}, max_containers=10)
def solve_chunk(run: str, model: str, task: str, indices: list, search_model: str) -> dict:
    """Serve one model and run a list of questions of one benchmark.

    Never raises: failures come back as {"error": ...}, so the coordinator can
    tell a failed chunk from one that is still running."""
    import chunk_runner as cr

    result = {"model": model, "task": task, "indices": indices, "accepted": [], "set_aside": []}
    server = None
    server_log = Path("/tmp/server.log")
    store = _store(run, model, task)
    try:
        volume.reload()
        pending = [i for i in indices if not cr.is_finished(store, i)]
        if not pending:
            return result
        result["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                       capture_output=True, text=True).stdout.strip()
        server = cr.start_server(SERVED[model], Path("/repo"), PORT, server_log)
        started = time.time()
        cr.wait_ready(server, PORT, server_log)
        result["load_s"] = round(time.time() - started, 1)
        print(f"{core.label(model, run)} {task}: model ready in {result['load_s']} s on {result['gpu']}",
              flush=True)
        result.update(cr.run_questions(task, pending, store, Path("/repo/test"), _agent_env(search_model),
                                       server=server, server_log=server_log, commit=volume.commit))
    except Exception as exc:  # noqa: BLE001 - reported to the coordinator as data
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if server is not None:
            cr.stop_server(server)
            try:
                (store / "server_logs").mkdir(parents=True, exist_ok=True)
                shutil.copy(server_log, store / "server_logs" / f"{indices[0]}-{int(time.time())}.log")
                volume.commit()
            except Exception:  # noqa: BLE001
                pass
    return result


def _score(run: str, model: str, task: str, limit: int, commit) -> dict:
    import chunk_runner as cr

    return cr.score(_store(run, model, task), task, core.label(model, run), limit, Path("/repo/test"), commit)


@app.function(cpu=1.0, memory=4096, timeout=24 * 3600, secrets=[keys], volumes={str(VOL): volume})
def coordinate(run: str, limit: int, tasks: list, models: list, chunk_size: int, search_model: str,
               code_version: str = "unknown") -> dict:
    """Schedule chunks until every question is finished, then score. Runs remotely,
    so it keeps going with `modal run --detach` after the laptop disconnects."""
    import chunk_runner as cr

    run_dir = VOL / run
    lock_file = run_dir / "LOCK.json"
    last_beat = [0.0]

    def heartbeat(force: bool = False):
        if force or time.time() - last_beat[0] > 60:
            run_dir.mkdir(parents=True, exist_ok=True)
            lock_file.write_text(json.dumps({"heartbeat": time.time()}))
            volume.commit()
            last_beat[0] = time.time()

    def wait(seconds: float):
        end = time.time() + seconds
        while time.time() < end:
            heartbeat()
            time.sleep(min(20, max(0, end - time.time())))

    base_revision = Path(BASE_MODEL_PATH + ".revision").read_text().strip()
    report = {"run": run, "limit": limit, "search_model": search_model, "base_revision": base_revision,
              "chunk_failures": [], "gpus": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        volume.reload()
        heartbeat(force=True)
        # A run keeps the settings it started with; resuming with others would mix them.
        config = {"limit": limit, "search_model": search_model, "base_revision": base_revision}
        run_file = run_dir / "RUN.json"
        if run_file.exists():
            saved = json.loads(run_file.read_text())
            diff = {k: (saved.get(k), v) for k, v in config.items() if saved.get(k) != v}
            if diff:
                raise RuntimeError(f"run '{run}' was started with other settings {diff} (saved, now). "
                                   "Use the original settings or a new --run name.")
            saved.setdefault("code_versions", [])
            if code_version not in saved["code_versions"]:
                saved["code_versions"].append(code_version)
            run_file.write_text(json.dumps(saved, indent=1))
        else:
            run_file.write_text(json.dumps({**config, "code_versions": [code_version]}, indent=1))
        volume.commit()

        previous = None
        for rnd in range(1, core.MAX_ROUNDS + 1):
            volume.reload()
            todo = {(m, t): cr.todo(_store(run, m, t), limit) for m in models for t in tasks}
            todo = {k: v for k, v in todo.items() if v}
            remaining = sum(len(v) for v in todo.values())
            if not remaining:
                break
            # Progress = questions finished or attempts made; a round without either ends the run.
            state = (remaining, sum(cr.attempts_made(_store(run, m, t), limit) for m in models for t in tasks))
            if previous is not None and state[0] >= previous[0] and state[1] <= previous[1]:
                print(f"round {rnd - 1} made no progress; stopping", flush=True)
                break
            if previous is not None:
                wait(120)  # give rate limits a moment before retrying
            previous = state
            plan = core.plan_chunks(todo, chunk_size)
            print(f"round {rnd}: {remaining} questions in {len(plan)} chunks", flush=True)
            outcome = cr.run_round(
                plan,
                spawn=lambda item: solve_chunk.spawn(run, item[0], item[1], item[2], search_model),
                get=lambda call: call.get(timeout=0),
                cancel=lambda call: call.cancel(),
                tick=heartbeat,
            )
            report["chunk_failures"] += outcome["failures"]
            for _, res in outcome["results"]:
                if res.get("gpu") and res["gpu"] not in report["gpus"]:
                    report["gpus"].append(res["gpu"])
            if outcome["stopped"]:
                raise RuntimeError(f"stopped early: {outcome['stopped']}.\n" + "\n".join(outcome["failures"][-3:]))

        volume.reload()
        unfinished = {f"{m}/{t}": cr.todo(_store(run, m, t), limit) for m in models for t in tasks}
        report["unfinished"] = {k: v for k, v in unfinished.items() if v}
        report["scores"] = {}
        for m in models:
            for t in tasks:
                if f"{m}/{t}" not in report["unfinished"]:
                    heartbeat(force=True)
                    report["scores"][f"{m}/{t}"] = _score(run, m, t, limit, volume.commit)
                    print(f"scored {core.label(m, run)} {t}: {report['scores'][f'{m}/{t}']}", flush=True)
        report["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                              capture_output=True, text=True).stdout
        return report
    finally:
        report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "REPORT.json").write_text(json.dumps(report, indent=1))
            lock_file.unlink(missing_ok=True)
            volume.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"could not write REPORT.json: {exc}", flush=True)


@app.function(cpu=0.5, memory=1024, timeout=10 * 60, volumes={str(VOL): volume})
def status(run: str, limit: int, tasks: list, models: list) -> dict:
    import chunk_runner as cr

    volume.reload()
    lock = VOL / run / "LOCK.json"
    age = None
    if lock.exists():
        age = time.time() - json.loads(lock.read_text()).get("heartbeat", 0)
    progress = {}
    for m in models:
        for t in tasks:
            store = _store(run, m, t)
            progress[f"{m}/{t}"] = limit - len(cr.todo(store, limit)) if store.exists() else 0
    return {"lock_age_s": age, "progress": progress}


@app.function(cpu=1.0, memory=2048, timeout=15 * 60, volumes={str(VOL): volume})
def export(run: str, limit: int, model: str, task: str) -> dict:
    """Files for one (model, task), keyed by their path in the repository."""
    import chunk_runner as cr

    volume.reload()
    store = _store(run, model, task)
    lbl = core.label(model, run)
    prefix = f"test/{task}/results/{lbl}/"
    files = {}
    for p in sorted((store / "final").glob("*")):
        files[prefix + p.name] = p.read_bytes()
    outputs = sorted((store / "results").glob("output_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        for p in outputs:
            files[prefix + p.name] = p.read_bytes()
            gz.write(json.dumps(json.loads(p.read_text()), ensure_ascii=False).encode() + b"\n")
    if outputs:
        files[prefix + "trajectories.jsonl.gz"] = buf.getvalue()
    rows = cr.question_rows(store, limit)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    files[prefix + "questions.csv"] = out.getvalue().encode()
    for p in sorted((store / "logs").glob("*.log")):
        files[f"test/{task}/logs/{lbl}/{p.name}"] = p.read_bytes()
    for name in ("REPORT.json", "PREFLIGHT.json"):
        if (VOL / run / name).exists():
            files[f"eval/runs/{run}/{name}"] = (VOL / run / name).read_bytes()
    return files


# ---------------------------------------------------------------------------
# Local side
# ---------------------------------------------------------------------------

def _print_preflight(report: dict) -> None:
    for name, check in report.get("checks", {}).items():
        mark = "ok  " if check["ok"] else ("FAIL" if check.get("required", True) else "warn")
        print(f"  {mark} {name}: {check['detail']}")
    if report.get("error"):
        print(report["error"])


def _code_version() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return (head or "unknown") + ("-modified" if dirty else "")
    except OSError:
        return "unknown"


@app.local_entrypoint()
def main(run: str = "fair50", limit: int = 50, tasks: str = ",".join(core.TASKS), models: str = "base,lora",
         chunk_size: int = 10, search_model: str = DEFAULT_SEARCH_MODEL, preflight_only: bool = False,
         skip_preflight: bool = False, download_only: bool = False):
    core.validate_run_name(run)
    task_list = core.parse_list(tasks, core.TASKS)
    model_list = core.parse_list(models, core.MODELS)
    largest = min(core.DATA_SIZES[t] for t in task_list)
    if not 1 <= limit <= largest:
        raise SystemExit(f"--limit must be between 1 and {largest} for these benchmarks")
    if not 1 <= chunk_size <= core.MAX_CHUNK_SIZE:
        raise SystemExit(f"--chunk-size must be between 1 and {core.MAX_CHUNK_SIZE}")

    if preflight_only:
        print(f"Preflight (search model {search_model}) ...")
        report = preflight.remote(run, search_model)
        _print_preflight(report)
        print("\nPreflight passed." if report.get("ok") else "\nPreflight failed: fix the item marked FAIL.")
        return

    st = status.remote(run, limit, task_list, model_list)
    print(f"Progress for run '{run}' (questions finished of {limit}): {st['progress']}")
    busy = st["lock_age_s"] is not None and st["lock_age_s"] < 900

    if not download_only:
        if busy:
            print("Another session is still running this evaluation (its heartbeat is recent). Wait for it "
                  "to finish, or check `modal app list`. Use --download-only to fetch what is done so far.")
            return
        if not skip_preflight:
            print(f"Preflight (search model {search_model}) ...")
            report = preflight.remote(run, search_model)
            _print_preflight(report)
            if not report.get("ok"):
                print("\nPreflight failed; nothing was run on a GPU. Fix the item marked FAIL and run again.")
                return
        try:
            coordinate.remote(run, limit, task_list, model_list, chunk_size, search_model, _code_version())
        except Exception as exc:  # noqa: BLE001
            print(f"\nThe evaluation stopped: {exc}\nFinished questions are kept; run the same command "
                  "again after fixing the problem to continue.")
            return

    written = 0
    for m in model_list:
        for t in task_list:
            for rel, data in export.remote(run, limit, m, t).items():
                dest = REPO / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                written += 1
    print(f"Downloaded {written} files into the repository.")
    import summarize

    text = summarize.write_summary(run, limit, task_list, model_list, repo=REPO)
    out = "eval/RESULTS.md" if run == "fair50" else f"eval/runs/{run}/RESULTS.md"
    print("\n" + text + f"\nWritten: {out}")
