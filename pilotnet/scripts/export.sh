
JOB_ID="pilotnet_$(date +%Y%m%d%H%M%S)"

# training variables
MODEL_DIR="models/pilotnet_20180802041904"
DATA_DIR="data/raw"
MODE="export"

python -m pilotnet.main \
    --job-dir $MODEL_DIR \
    --data-dir $DATA_DIR \
    --mode $MODE