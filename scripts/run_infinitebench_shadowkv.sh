#!/bin/bash
# InfiniteBench evaluation with Qwen3-8B using ShadowKV
# 128k context, 10 tasks (code_run/math_calc excluded)
# sparse_budget_ratio=0.27: total KV = 27% of input (outlier+local subtracted automatically)

# ===================== Config =====================
MODEL_PATH="/home/jheo/models/Qwen3-8B-128k"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHADOWKV_DIR="$(dirname "$SCRIPT_DIR")"
# ==================================================

source ~/anaconda3/etc/profile.d/conda.sh
conda activate shadowkv
cd "${SHADOWKV_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SHADOWKV_DIR}/logs/Qwen3-8B_infinitebench_shadowkv_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=== InfiniteBench ShadowKV ===" | tee "$LOG_DIR/run.log"
echo "Start: $(date)" | tee -a "$LOG_DIR/run.log"
echo "Model: Qwen/Qwen3-8B" | tee -a "$LOG_DIR/run.log"
echo "Dataset: InfiniteBench (10 tasks)" | tee -a "$LOG_DIR/run.log"
echo "Max context: 131072" | tee -a "$LOG_DIR/run.log"
echo "sparse_budget_ratio: 0.27" | tee -a "$LOG_DIR/run.log"
echo "Log dir: $LOG_DIR" | tee -a "$LOG_DIR/run.log"

python test/eval_acc.py \
    --model_name "${MODEL_PATH}" \
    --dataset_name "infinitebench/passkey,infinitebench/number_string,infinitebench/kv_retrieval,infinitebench/longbook_qa_eng,infinitebench/longbook_choice_eng,infinitebench/longbook_sum_eng,infinitebench/longdialogue_qa_eng,infinitebench/longbook_qa_chn,infinitebench/code_debug,infinitebench/math_find" \
    --datalen 131072 \
    --method shadowKV_cpu \
    --batch_size 1 \
    --sparse_budget_ratio 0.27 \
    --rank 160 \
    --chunk_size 8 \
    2>&1 | tee -a "$LOG_DIR/run.log"

echo "End: $(date)" | tee -a "$LOG_DIR/run.log"
echo "Results saved to: $LOG_DIR"
