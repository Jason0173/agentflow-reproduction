#!/bin/bash
# Run one AgentFlow benchmark against the local LoRA planner server.
#
# Usage (from AgentFlow/test/):
#   export MODAL_PLANNER_URL=http://127.0.0.1:8765/chat
#   bash run_lora_bench.sh bamboogle
#   bash run_lora_bench.sh 2wiki
#   bash run_lora_bench.sh hotpotqa
#   bash run_lora_bench.sh musique
#   bash run_lora_bench.sh gaia
#
# This is modeled on run_modal_Qwen3.5-9B-Modal.sh. The only runtime
# differences are:
#   - LABEL is "Qwen3.5-0.8B-LoRA" so results land in a new folder.
#   - LLM engine string still uses the "modal-" prefix so factory.py routes
#     it to ModalEngine, which in turn reads MODAL_PLANNER_URL.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <benchmark>  (bamboogle|2wiki|hotpotqa|musique|gaia)" >&2
    exit 1
fi
TASK="$1"

if [ -z "${MODAL_PLANNER_URL:-}" ]; then
    echo "ERROR: MODAL_PLANNER_URL is not set. Start serve_lora_local.py and export it." >&2
    exit 1
fi

# Per-task sample counts required for README section 7.2 LoRA comparison.
# Source: user spec 2026-04-16 — 552 total across five benchmarks.
case "$TASK" in
    bamboogle)  MAX_IDX=124 ;;
    2wiki)      MAX_IDX=99  ;;
    hotpotqa)   MAX_IDX=99  ;;
    musique)    MAX_IDX=99  ;;
    gaia)       MAX_IDX=126 ;;
    *) echo "Unknown task: $TASK" >&2; exit 1 ;;
esac

# Optional smoke-test cap: SMOKE_LIMIT=5 only runs indices 0..4.
if [ -n "${SMOKE_LIMIT:-}" ]; then
    MAX_IDX=$((SMOKE_LIMIT - 1))
    echo "[run_lora_bench] SMOKE mode: capping $TASK to indices 0..$MAX_IDX"
fi

DATA_FILE_NAME="data.json"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"   # we're already inside test/
cd "$PROJECT_DIR"

LLM="modal-Qwen/Qwen3.5-0.8B-LoRA"
# Label carries today's date so old Qwen3.5-0.8B-LoRA/ attempts are preserved.
LABEL="${LORA_LABEL:-Qwen3.5-0.8B-LoRA-20260416}"
ENABLED_TOOLS="Base_Generator_Tool,Google_Search_Tool,Wikipedia_Search_Tool"
TOOL_ENGINE="Default,Default"
MODEL_ENGINE="trainable,trainable,trainable,trainable"

DATA_FILE="$TASK/data/$DATA_FILE_NAME"
LOG_DIR="$TASK/logs/$LABEL"
OUT_DIR="$TASK/results/$LABEL"
CACHE_DIR="$TASK/cache"

mkdir -p "$LOG_DIR" "$OUT_DIR"

INDICES=($(seq 0 $MAX_IDX))

new_indices=()
for i in "${INDICES[@]}"; do
    if [ ! -f "$OUT_DIR/output_$i.json" ]; then
        new_indices+=($i)
    fi
done
indices=("${new_indices[@]}")

echo "[run_lora_bench] TASK=$TASK LABEL=$LABEL remaining=${#indices[@]} planner=$MODAL_PLANNER_URL"

if [ ${#indices[@]} -eq 0 ]; then
    echo "[run_lora_bench] all tasks already completed, skipping to scoring."
else
    for i in "${indices[@]}"; do
        echo "--- $LABEL | $TASK | Task $i ---"
        python solve.py --index "$i" --task "$TASK" --data_file "$DATA_FILE" \
            --llm_engine_name "$LLM" \
            --root_cache_dir "$CACHE_DIR" \
            --output_json_dir "$OUT_DIR" \
            --output_types direct \
            --enabled_tools "$ENABLED_TOOLS" \
            --tool_engine "$TOOL_ENGINE" \
            --model_engine "$MODEL_ENGINE" \
            --max_time 300 --max_steps 10 --temperature 0.0 \
            2>&1 | tee "$LOG_DIR/$i.log"
    done
fi

echo ""
echo "=== Scoring $LABEL on $TASK ==="
python calculate_score_unified.py --task_name "$TASK" --data_file "$DATA_FILE" \
    --result_dir "$OUT_DIR" --response_type "direct_output" \
    --output_file "finalresults_direct_output.json" \
    | tee "$OUT_DIR/finalscore_direct_output.log"
