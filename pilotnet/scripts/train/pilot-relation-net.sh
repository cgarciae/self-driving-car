
JOB_ID="pilotnet_$(date +%Y%m%d%H%M%S)"


# training variables
MODEL_DIR="models/$JOB_ID"
# MODEL_DIR="models/pilotnet_20180802041904"
DATA_DIR="data/raw"
MODE="train"

# train params
NETWORK="relation"

echo MODEL_DIR: $MODEL_DIR

python -m pilotnet.train \
    --data-dir $DATA_DIR \
    --job-dir $MODEL_DIR \
    --mode $MODE \
    --network $NETWORK

echo MODEL_DIR: $MODEL_DIR