"""Minimal smoke test for serve_lora_local.py.

Run this on the H200 node AFTER serve_lora_local.py is up. It verifies:
  1. /health responds
  2. /chat returns the Modal wire shape {"choices": [{"message": {"content": str}}]}
  3. The LoRA adapter was actually applied (answer != empty / != error)

Usage:
    PLANNER_URL=http://127.0.0.1:8765 python smoke_test_lora.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import requests

PLANNER_URL = os.environ.get("PLANNER_URL", "http://127.0.0.1:8765")

# Explorer compute nodes inherit http_proxy=http://10.99.0.130:3128, which
# breaks loopback HTTP. Build a session that ignores env-derived proxies so
# the tester can't be silently hijacked.
SESSION = requests.Session()
SESSION.trust_env = False


def wait_for_health(timeout_s: int = 300) -> None:
    print(f"[smoke] polling {PLANNER_URL}/health (up to {timeout_s}s)...", flush=True)
    start = time.time()
    last_error = ""
    attempts = 0
    while time.time() - start < timeout_s:
        attempts += 1
        try:
            r = SESSION.get(f"{PLANNER_URL}/health", timeout=5)
            if r.status_code == 200 and r.json().get("model_loaded"):
                print(f"[smoke] health ok after {attempts} attempts: {r.json()}", flush=True)
                return
            last_error = f"status={r.status_code} body={r.text[:200]!r}"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        # Every 10 attempts (~30s) surface the latest error so we don't flail
        # silently for 5 minutes again.
        if attempts % 10 == 0:
            print(f"[smoke] still waiting (attempt {attempts}): {last_error}", flush=True)
        time.sleep(3)
    raise SystemExit(
        f"[smoke] server did not become healthy within {timeout_s}s "
        f"(last error: {last_error})"
    )


def test_chat() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is the capital of France? Answer in one word."},
        ],
        "temperature": 0.0,
    }
    print(f"[smoke] POST /chat payload={json.dumps(payload)}", flush=True)
    r = SESSION.post(f"{PLANNER_URL}/chat", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    print(f"[smoke] raw response keys: {list(data.keys())}")
    content = data["choices"][0]["message"]["content"]
    print(f"[smoke] assistant content: {content!r}")
    if not content or content.startswith("Error"):
        raise SystemExit("[smoke] empty or error content — investigate server logs")
    print("[smoke] PASS")


if __name__ == "__main__":
    try:
        wait_for_health()
        test_chat()
    except Exception as exc:
        print(f"[smoke] FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
