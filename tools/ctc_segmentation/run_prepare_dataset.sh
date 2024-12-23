#!/bin/bash


SCRIPTS_DIR="scripts" # /<PATH TO>/NeMo/tools/ctc_segmentation/tools/scripts/ directory
INPUT_AUDIO_DIR="" # Path to original directory with audio files
MANIFEST=""
OUTPUT_DIR=""
DATASET_DIR=""
CUT_PREFIX=0
MIN_SCORE=-2
OFFSET=0
SAMPLE_RATE=44100 # Target sample rate for high quality data
MAX_DURATION=20 # Maximum audio segment duration, in seconds. Samples that are longer will be dropped.
MODE="" # "ganjoor"

for ARG in "$@"
do
    key=$(echo $ARG | cut -f1 -d=)
    value=$(echo $ARG | cut -f2 -d=)

    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        declare $v="${value}"
    fi
done

echo "SCRIPTS_DIR = $SCRIPTS_DIR"
echo "INPUT_AUDIO_DIR = $INPUT_AUDIO_DIR"
echo "MANIFEST = $MANIFEST"
echo "OUTPUT_DIR = $OUTPUT_DIR"
echo "DATASET_DIR = $DATASET_DIR"
echo "CUT_PREFIX = $CUT_PREFIX"
echo "MIN_SCORE = $MIN_SCORE"
echo "OFFSET = $OFFSET"
echo "SAMPLE_RATE = $SAMPLE_RATE"
echo "MAX_DURATION = $MAX_DURATION"
echo "MODE = $MODE"

echo "AUDIO PREPROCESSING..."
rm -rf $OUTPUT_DIR/processed
python $SCRIPTS_DIR/prepare_data.py \
--audio_dir=$INPUT_AUDIO_DIR \
--output_dir=$OUTPUT_DIR/processed/ \
--cut_prefix=$CUT_PREFIX \
--sample_rate=$SAMPLE_RATE || exit

# above the MIN_SCORE value will be saved to $OUTPUT_DIR/manifests/manifest.json
echo "CUTTING AUDIO..."
python $SCRIPTS_DIR/cut_audio.py \
--output_dir=$OUTPUT_DIR \
--alignment=$OUTPUT_DIR/verified_segments \
--threshold=$MIN_SCORE \
--offset=$OFFSET \
--sample_rate=$SAMPLE_RATE \
--max_duration=$MAX_DURATION || exit

echo "PREPARING DATASET..."
python $SCRIPTS_DIR/preparing_dataset.py \
--manifest=$MANIFEST \
--output_dir=$OUTPUT_DIR/ \
--mode=$MODE \
--clips_dir=$OUTPUT_DIR/clips_44k/ || exit

echo ""COPYING DATASET TO $DATASET_DIR"..."
mkdir -p $DATASET_DIR/audio
cp -r $OUTPUT_DIR/clips_44k/* $DATASET_DIR/audio/ || exit
cp $OUTPUT_DIR/metadata.csv $DATASET_DIR || exit

echo ""ZIPPING DATASET TO $DATASET_DIR"..."
zip -rq $DATASET_DIR.zip $DATASET_DIR || exit




