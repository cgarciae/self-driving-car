
declare -r JOB_ID="pilotnet_$(date +%Y%m%d%H%M%S)"

# training variables
# declare -r MODEL_DIR="gs://vision-198622-development-models/auto_pilot/0.0.1/$JOB_ID"
declare -r MODEL_DIR="models/$JOB_ID"
declare -r DATA_DIR="data/raw"


python -m pilotnet.main \
    --job-dir $MODEL_DIR \
    --data-dir $DATA_DIR