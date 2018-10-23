# EXPORT_DIR="models/pilotnet_20180807230428/export/pilotnet/1533686269" #=> good - 51
# EXPORT_DIR="models/pilotnet_20180808055900/export/pilotnet/1533711551" #=> BEST - bins=51, dropout=0.15, data augmentation=10
EXPORT_DIR="models/pilotnet_20181022195417/export/pilotnet/1540258811" #=> relation


python -m pilotnet.run \
    --export-dir $EXPORT_DIR \
    $@