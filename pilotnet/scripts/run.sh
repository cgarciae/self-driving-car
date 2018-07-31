declare -r EXPORT_DIR="models/pilotnet_20180731190340/export/pilotnet/1533071440"

python -m pilotnet.run \
    --export-dir $EXPORT_DIR \
    $@