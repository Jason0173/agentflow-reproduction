#!/bin/bash
# Auto-generated run script for Step 3
# Model: Qwen3.5-9B-Modal
# Benchmark: hotpotqa

TASK="hotpotqa"
DATA_FILE_NAME="data.json"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd $PROJECT_DIR

LLM="modal-Qwen/Qwen3.5-9B"
LABEL="Qwen3.5-9B-Modal"
ENABLED_TOOLS="Base_Generator_Tool,Google_Search_Tool,Wikipedia_Search_Tool"
TOOL_ENGINE="Default,Default"
MODEL_ENGINE="trainable,trainable,trainable,trainable"

DATA_FILE="$TASK/data/$DATA_FILE_NAME"
LOG_DIR="$TASK/logs/$LABEL"
OUT_DIR="$TASK/results/$LABEL"
CACHE_DIR="$TASK/cache"

mkdir -p "$LOG_DIR" "$OUT_DIR"

INDICES=($(seq 0 99))

# Skip already completed
new_indices=()
for i in "${INDICES[@]}"; do
    if [ ! -f "$OUT_DIR/output_$i.json" ]; then
        new_indices+=($i)
    fi
done
indices=("${new_indices[@]}")

if [ ${#indices[@]} -eq 0 ]; then
    echo "All subtasks completed for $LABEL on $TASK."
else
    echo "Running ${#indices[@]} tasks for $LABEL on $TASK..."
    for i in "${indices[@]}"; do
        echo "--- $LABEL | $TASK | Task $i ---"
        python solve.py --index $i --task "$TASK" --data_file "$DATA_FILE" \
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
