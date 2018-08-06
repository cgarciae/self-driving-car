# declare -r EXPORT_DIR="models/pilotnet_20180802023014/export/pilotnet/1533177407" #=> Good
declare -r EXPORT_DIR="models/pilotnet_20180806135731/export/pilotnet/1533567083" #=> Good


python -m pilotnet.run \
    --export-dir $EXPORT_DIR \
    $@