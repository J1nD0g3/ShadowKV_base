#!/bin/bash
# RULER ShadowKV 3% test (3 samples per task)
# Quick estimation run before full evaluation

# ===================== Config =====================
MODEL_PATH="/home/jheo/models/Qwen3-8B-128k"
MODEL_TEMPLATE="qwen3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHADOWKV_DIR="$(dirname "$SCRIPT_DIR")"
RULER_DATA_DIR="${SHADOWKV_DIR}/data/ruler/data/${MODEL_TEMPLATE}/102400"
RULER_TASKS="ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multivalue,ruler/niah_multiquery,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2"
DATALEN=102400
SPARSE_BUDGET_RATIO=0.27
RANK=160
CHUNK_SIZE=8
NUM_SAMPLES=3
# ==================================================

source ~/anaconda3/etc/profile.d/conda.sh
conda activate shadowkv
cd "${SHADOWKV_DIR}"
export PYTHONPATH=""

# Generate RULER data if not exists
TASK_LIST=(niah_single_1 niah_single_2 niah_multikey_1 niah_multikey_2 niah_multivalue niah_multiquery vt fwe qa_1 qa_2)
MISSING=0
for t in "${TASK_LIST[@]}"; do
    if [ ! -f "${RULER_DATA_DIR}/${t}/validation.jsonl" ]; then
        echo "[INFO] Missing RULER data: ${t}"
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo "[INFO] Generating RULER data..."
    cd "${SHADOWKV_DIR}/data/ruler"
    bash create_dataset.sh "$MODEL_PATH" "$MODEL_TEMPLATE"
    cd "${SHADOWKV_DIR}"
    echo "[INFO] RULER data generation complete."
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SHADOWKV_DIR}/logs/Qwen3-8B-128k_ruler_shadowkv_3pct_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=== RULER ShadowKV 3% test ===" | tee "$LOG_DIR/run.log"
echo "Start: $(date)" | tee -a "$LOG_DIR/run.log"
echo "Model: $MODEL_PATH" | tee -a "$LOG_DIR/run.log"
echo "Samples per task: $NUM_SAMPLES" | tee -a "$LOG_DIR/run.log"
echo "sparse_budget_ratio: $SPARSE_BUDGET_RATIO" | tee -a "$LOG_DIR/run.log"
echo "Log dir: $LOG_DIR" | tee -a "$LOG_DIR/run.log"

python test/eval_acc.py \
    --model_name "$MODEL_PATH" \
    --dataset_name "$RULER_TASKS" \
    --datalen $DATALEN \
    --method shadowKV_cpu \
    --batch_size 1 \
    --sparse_budget_ratio $SPARSE_BUDGET_RATIO \
    --rank $RANK \
    --chunk_size $CHUNK_SIZE \
    --num_samples $NUM_SAMPLES \
    2>&1 | tee -a "$LOG_DIR/run.log"

echo "End: $(date)" | tee -a "$LOG_DIR/run.log"
echo "Results saved to: $LOG_DIR"
