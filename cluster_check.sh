#!/bin/bash
# One-shot diagnostic to run on Explorer LOGIN node (NOT the GPU node).
# Usage:
#   cd ~/neu-self-improve-ai/week08_group_work_finalVersion/AgentFlow
#   bash cluster_check.sh 2>&1 | tee cluster_check.log
# Paste the contents of cluster_check.log back so we can decide between
# merged-model mode and adapter mode.

echo "===== 1. basic host info ====="
hostname
uname -a
date
echo ""

echo "===== 2. python & pip availability ====="
which python python3 pip pip3 2>&1
python3 --version 2>&1 || true
echo ""

echo "===== 3. HuggingFace cache (merged model / base model) ====="
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
echo "HF_HOME=${HF_HOME:-(unset)} -> looking in $HF_CACHE"
if [ -d "$HF_CACHE" ]; then
    ls -la "$HF_CACHE" 2>/dev/null | head -30
    echo "-- Qwen-related dirs --"
    ls -la "$HF_CACHE" | grep -Ei 'qwen|skypioneer' || echo "(no Qwen/Skypioneer dirs found)"
else
    echo "HF cache directory does not exist"
fi
echo ""

echo "===== 4. broader filesystem search ====="
echo "(this may take 10-30s; only scans your HOME and common scratch roots)"
for root in "$HOME" /scratch /work /projects; do
    [ -d "$root" ] || continue
    find "$root" -maxdepth 6 \
        \( -iname "*qwen35-0.8b*" -o -iname "*qwen3.5-0.8b*" -o -iname "*agentflow-lora*" -o -iname "adapter_config.json" -o -iname "model.safetensors" \) \
        -print 2>/dev/null | head -40
done
echo ""

echo "===== 5. adapter dir check (project-local) ====="
ADAPTER_DIR="results/final_qwen35_lora"
if [ -d "$ADAPTER_DIR" ]; then
    echo "Found $ADAPTER_DIR:"
    ls -la "$ADAPTER_DIR"
else
    echo "$ADAPTER_DIR missing (did you git pull the latest week08?)"
fi
echo ""

echo "===== 6. GPU visibility on login node ====="
nvidia-smi 2>&1 | head -20 || echo "(no nvidia-smi on login node — expected)"
echo ""

echo "===== 7. existing virtualenvs ====="
ls -d "$HOME/venvs/"*/ 2>/dev/null || echo "(no $HOME/venvs)"
ls -d "$HOME/.venv" "$HOME/miniconda3" "$HOME/anaconda3" 2>/dev/null || true
echo ""

echo "===== 8. relevant pip packages in current python ====="
python3 -c "import importlib.metadata as m; \
for p in ['transformers','peft','torch','fastapi','uvicorn','accelerate']: \
    try: print(p, m.version(p))
    except Exception as e: print(p, 'MISSING')" 2>&1 || true
echo ""

echo "===== 9. HF token availability ====="
if [ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "HF_TOKEN / HUGGING_FACE_HUB_TOKEN is set (value hidden)"
elif [ -f "$HOME/.cache/huggingface/token" ]; then
    echo "huggingface-cli login token file exists"
else
    echo "No HF token detected. If Skypioneer/qwen35-0.8b-agentflow-lora is private, you'll need one."
fi
echo ""

echo "===== 10. network reachability to HF ====="
curl -sS -o /dev/null -w "huggingface.co HTTP %{http_code}\n" https://huggingface.co 2>&1 || echo "curl failed"
echo ""

echo "===== DONE ====="
