
declare -r JOB_ID="pilotnet_$(date +%Y%m%d%H%M%S)"

# training variables
declare -r MODEL_DIR="models/pilotnet_20180731190340"
declare -r DATA_DIR="data/raw"
declare -r MODE="export"

python -m pilotnet.main \
    --job-dir $MODEL_DIR \
    --data-dir $DATA_DIR \
    --mode $MODE