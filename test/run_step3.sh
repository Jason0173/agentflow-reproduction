#!/bin/bash
# ============================================================
#  Step 3: Run AgentFlow on Qwen3.5-9B and Qwen3.5-27B
#
#  Prerequisites:
#    - Step 2 completed (AgentFlow repo + env working)
#    - modal CLI installed and logged in
#    - API keys configured (GOOGLE_API_KEY, OPENAI_API_KEY)
#
#  This script runs ONE model on ALL 5 benchmarks.
#  Usage:
#    bash run_step3.sh 9B     # Run Qwen3.5-9B
#    bash run_step3.sh 27B    # Run Qwen3.5-27B
# ============================================================

set -e

MODEL_SIZE="${1:-9B}"
MODEL_SIZE=$(echo "$MODEL_SIZE" | tr '[:lower:]' '[:upper:]')

if [[ "$MODEL_SIZE" != "9B" && "$MODEL_SIZE" != "27B" ]]; then
    echo "Usage: bash run_step3.sh [9B|27B]"
    exit 1
fi

LABEL="Qwen3.5-${MODEL_SIZE}-Modal"
LLM="modal-Qwen/Qwen3.5-${MODEL_SIZE}"
HF_MODEL="Qwen/Qwen3.5-${MODEL_SIZE}"

echo "=========================================="
echo "  Step 3: AgentFlow + $HF_MODEL"
echo "  Time: $(date)"
echo "=========================================="

# --- Check API keys ---
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "ERROR: GOOGLE_API_KEY not set"
    exit 1
fi
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi
if [ -z "$MODAL_VLLM_URL" ]; then
    echo "ERROR: MODAL_VLLM_URL not set"
    echo "Deploy first: MODEL_SIZE=$MODEL_SIZE modal deploy modal_serve_qwen35.py"
    exit 1
fi

echo "MODAL_VLLM_URL=$MODAL_VLLM_URL"
echo "Model: $HF_MODEL"
echo "Label: $LABEL"
echo ""

# --- Test Modal connection ---
echo "Testing Modal vLLM connection..."
python test_modal_qwen35.py
echo ""

# --- Activate environment ---
cd ~/AgentFlow
source .venv/bin/activate 2>/dev/null || true

# --- Run 5 benchmarks ---
BENCHMARKS=("bamboogle" "2wiki" "hotpotqa" "musique" "gaia")
cd ~/AgentFlow/test

for bench in "${BENCHMARKS[@]}"; do
    SCRIPT="$bench/run_modal_${LABEL}.sh"
    if [ -f "$SCRIPT" ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Running: $bench with $LABEL"
        echo "  Time:    $(date '+%H:%M:%S')"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        cd "$bench"
        bash "run_modal_${LABEL}.sh"
        cd ..
    else
        echo "[WARN] Script not found: $SCRIPT"
        echo "  Run generate_run_scripts.sh first"
    fi
done

# --- Summary ---
echo ""
echo "=========================================="
echo "  Step 3 Results: $LABEL"
echo "=========================================="
for bench in "${BENCHMARKS[@]}"; do
    SCORE_FILE="$bench/results/$LABEL/finalscore_direct_output.log"
    if [ -f "$SCORE_FILE" ]; then
        ACCURACY=$(grep -oP 'Accuracy: \K[0-9.]+%' "$SCORE_FILE" || echo "N/A")
        echo "  $bench: $ACCURACY"
    else
        echo "  $bench: (not yet scored)"
    fi
done
echo "=========================================="
