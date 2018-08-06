
declare -r JOB_ID="pilotnet_$(date +%Y%m%d%H%M%S)"

# training variables
declare -r MODEL_DIR="models/$JOB_ID"
# declare -r MODEL_DIR="models/pilotnet_20180802041904"
declare -r DATA_DIR="data/augmented"
declare -r MODE="train"

echo MODEL_DIR: $MODEL_DIR

python -m pilotnet.train \
    --job-dir $MODEL_DIR \
    --data-dir $DATA_DIR \
    --mode $MODE

echo MODEL_DIR: $MODEL_DIR