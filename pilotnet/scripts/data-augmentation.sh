

RAW_DIR="data/raw"
AUGMENTED_DIR="data/augmented"


python -m pilotnet.data_augmentation \
    --raw-dir $RAW_DIR \
    --augmented-dir $AUGMENTED_DIR \
    --rm
