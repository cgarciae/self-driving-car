
JOB_ID="pilotnet_$(date +%Y%m%d%H%M%S)"

# training variables
MODEL_DIR="models/$JOB_ID"
# MODEL_DIR="models/pilotnet_20180802041904"
DATA_DIR="data/raw"
MODE="train"

# train params


echo MODEL_DIR: $MODEL_DIR

python train.py \
    --job-dir $MODEL_DIR \
    --data-dir $DATA_DIR \
    --mode $MODE

echo MODEL_DIR: $MODEL_DIR