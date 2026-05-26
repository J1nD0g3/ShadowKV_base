#!/bin/bash
# LongBench-v2 evaluation with ShadowKV (CPU mode)
#
# Usage:
#   bash run_longbenchv2_shadowkv.sh [MODEL_PATH] [DATALEN] [SPARSE_BUDGET_RATIO]
#
# Examples:
#   bash run_longbenchv2_shadowkv.sh
#   bash run_longbenchv2_shadowkv.sh Qwen/Qwen3-8B 32768 0.27

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHADOWKV_DIR="$(dirname "$SCRIPT_DIR")"

# ===================== Config =====================
MODEL_PATH="${1:-Qwen/Qwen3-8B}"
DATALEN="${2:-32768}"
SPARSE_BUDGET_RATIO="${3:-0.27}"
RANK="${4:-160}"
CHUNK_SIZE="${5:-8}"
MODEL_SHORT="$(basename "$MODEL_PATH")"
# ==================================================

# Activate conda (works with anaconda, miniconda, miniforge)
CONDA_SH="${CONDA_PREFIX:-$HOME/anaconda3}/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] || CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate shadowkv

cd "${SHADOWKV_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SHADOWKV_DIR}/logs/${MODEL_SHORT}_longbenchv2_shadowkv_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=== LongBench-v2 ShadowKV ===" | tee "$LOG_DIR/run.log"
echo "Start: $(date)" | tee -a "$LOG_DIR/run.log"
echo "Model: $MODEL_PATH" | tee -a "$LOG_DIR/run.log"
echo "Dataset: THUDM/LongBench-v2 (503 samples)" | tee -a "$LOG_DIR/run.log"
echo "Max context: $DATALEN" | tee -a "$LOG_DIR/run.log"
echo "sparse_budget_ratio: $SPARSE_BUDGET_RATIO" | tee -a "$LOG_DIR/run.log"
echo "Log dir: $LOG_DIR" | tee -a "$LOG_DIR/run.log"

python test/eval_acc.py \
    --model_name "$MODEL_PATH" \
    --dataset_name "longbenchv2" \
    --datalen $DATALEN \
    --method shadowKV_cpu \
    --batch_size 1 \
    --sparse_budget_ratio $SPARSE_BUDGET_RATIO \
    --rank $RANK \
    --chunk_size $CHUNK_SIZE \
    2>&1 | tee -a "$LOG_DIR/run.log"

echo "End: $(date)" | tee -a "$LOG_DIR/run.log"
echo "Results saved to: $LOG_DIR"
