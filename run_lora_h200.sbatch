#!/bin/bash
#SBATCH --job-name=lora_h200_20260416
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Explorer H200 launcher for Step 5 LoRA benchmark evaluation.
# Tuned for cluster layout confirmed 2026-04-16:
#   - AgentFlow checkout lives at /home/zhao.y4/AgentFlow
#   - venv with transformers>=4.47 / peft / fastapi at ~/AgentFlow/.venv
#   - merged model already downloaded to /home/zhao.y4/AgentFlow/models/qwen35-0.8b-merged
#   - isolated working subdir ~/AgentFlow/lora_h200_20260416/ (parallel to teammate's files)
#
# To submit:
#   cd ~/AgentFlow/lora_h200_20260416
#   sbatch run_lora_h200.sbatch

set -uo pipefail
# NOTE: deliberately NOT using `-e`. We want to survive individual step failures
# so the H200 allocation is preserved for interactive debugging via
#   srun --jobid=$SLURM_JOB_ID --overlap --pty bash
# instead of the job exiting and forcing a fresh queue wait.

FAILED=0
SMOKE_OK=0

# ============ EDIT ME ============
# Absolute path to the AgentFlow checkout on the cluster (contains test/run_lora_bench.sh).
AGENTFLOW_DIR="${AGENTFLOW_DIR:-/home/zhao.y4/AgentFlow}"
# Absolute path to a pre-created virtualenv with transformers/peft/fastapi installed.
VENV_DIR="${VENV_DIR:-$HOME/AgentFlow/.venv}"
# Working directory where serve_lora_local.py / smoke_test_lora.py were copied.
LORA_WORKDIR="${LORA_WORKDIR:-$HOME/AgentFlow/lora_h200_20260416}"
# Which benchmarks to run (space-separated; order matters for logs only).
BENCHES="${BENCHES:-bamboogle 2wiki hotpotqa musique gaia}"
# =================================

# Pick a high random port to avoid clashes if two jobs land on the same node.
PORT="${PORT:-$((20000 + RANDOM % 20000))}"
export MODAL_PLANNER_URL="http://127.0.0.1:${PORT}/chat"

# CRITICAL: Explorer compute nodes inherit http_proxy=http://10.99.0.130:3128
# from the shell env. Python's `requests` library honors that proxy for ALL
# outbound HTTP — including our 127.0.0.1 planner. The proxy cannot route
# back to the loopback interface of a compute node, so every call hangs
# silently. We whitelist localhost here so the planner is reachable while
# external calls (OpenAI judge, web search tools) still go via the proxy.
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="127.0.0.1,localhost,::1"

echo "=========================================="
echo "SLURM job: $SLURM_JOB_ID on $(hostname)"
echo "AGENTFLOW_DIR: $AGENTFLOW_DIR"
echo "VENV_DIR:      $VENV_DIR"
echo "PORT:          $PORT"
echo "MODAL_PLANNER_URL: $MODAL_PLANNER_URL"
echo "BENCHES:       $BENCHES"
nvidia-smi
echo "=========================================="

source "$VENV_DIR/bin/activate"
cd "$LORA_WORKDIR"
mkdir -p logs

# Load .env (OPENAI_API_KEY, GOOGLE_API_KEY, DASHSCOPE_API_KEY, ...) if present.
# The benchmark uses these for search tools even when the planner is local.
if [ -f "$AGENTFLOW_DIR/agentflow/.env" ]; then
    set -a; source "$AGENTFLOW_DIR/agentflow/.env"; set +a
fi

# ---- 1. Launch the local planner server in the background ----
# MERGED_MODEL defaults to the pre-downloaded local merged checkpoint on this
# cluster so we never hit HF during serve startup. Override with an env var if
# you want to try the HF repo (Skypioneer/qwen35-0.8b-agentflow-lora) or an
# adapter-mode setup (leave MERGED_MODEL empty + set BASE_MODEL/LORA_DIR).
echo "[launcher] starting serve_lora_local.py on port $PORT ..."
PORT="$PORT" \
MERGED_MODEL="${MERGED_MODEL:-/home/zhao.y4/AgentFlow/models/qwen35-0.8b-merged}" \
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-0.8B}" \
LORA_DIR="${LORA_DIR:-$AGENTFLOW_DIR/results/final_qwen35_lora}" \
python serve_lora_local.py > "logs/serve_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
echo "[launcher] server PID=$SERVER_PID"

cleanup() {
    echo "[launcher] cleanup: killing server PID=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ---- 2. Wait for health (failure tolerant) ----
if PLANNER_URL="http://127.0.0.1:${PORT}" python smoke_test_lora.py; then
    SMOKE_OK=1
    echo "[launcher] smoke test PASSED"
else
    echo "[launcher] !!! SMOKE TEST FAILED — skipping benchmarks, will hold GPU for debug"
    FAILED=1
fi

# ---- 3. Run benchmarks sequentially via the dedicated launcher ----
# run_lora_bench.sh lives inside AGENTFLOW_DIR/test because it cd's into its
# own directory and expects <bench>/data/data.json + solve.py alongside.
if [ "$SMOKE_OK" = "1" ]; then
    cd "$AGENTFLOW_DIR/test"
    for bench in $BENCHES; do
        echo ""
        echo "=== [launcher] running benchmark: $bench ==="
        if bash run_lora_bench.sh "$bench"; then
            echo "=== [launcher] benchmark $bench: OK ==="
        else
            echo "=== [launcher] benchmark $bench: FAILED (continuing to next) ==="
            FAILED=1
        fi
    done
fi

# ---- 4. Hold GPU if anything went wrong (or forced via HOLD_GPU=1) ----
if [ "$FAILED" = "1" ] || [ "${HOLD_GPU:-0}" = "1" ]; then
    echo "=========================================================="
    echo "[launcher] HOLDING GPU until time limit so debug is possible."
    echo "[launcher] Attach from login node with:"
    echo "    srun --jobid=$SLURM_JOB_ID --overlap --pty bash"
    echo "[launcher] Then inside that shell:"
    echo "    source $VENV_DIR/bin/activate"
    echo "    export MODAL_PLANNER_URL=$MODAL_PLANNER_URL"
    echo "    curl -s http://127.0.0.1:$PORT/health"
    echo "[launcher] Server log: $LORA_WORKDIR/logs/serve_${SLURM_JOB_ID}.log"
    echo "[launcher] Server PID: $SERVER_PID (may be dead; ps $SERVER_PID to check)"
    echo "=========================================================="
    # sleep infinity blocks until SIGTERM from SLURM when time limit hits;
    # that is when the EXIT trap finally kills the server.
    sleep infinity
fi

echo ""
echo "[launcher] all benchmarks complete successfully."
