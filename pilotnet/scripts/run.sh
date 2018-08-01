declare -r EXPORT_DIR="models/pilotnet_20180731220338/export/pilotnet/1533075183"

python -m pilotnet.run \
    --export-dir $EXPORT_DIR \
    $@