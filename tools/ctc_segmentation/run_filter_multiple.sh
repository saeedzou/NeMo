#!/bin/bash
[ ! -f transcribe_speech.py ] && wget -O transcribe_speech.py https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/asr/transcribe_speech.py

SCRIPTS_DIR="scripts" # /<PATH TO>/NeMo/tools/ctc_segmentation/tools/scripts/ directory
INPUT_AUDIO_DIR="" # Path to original directory with audio files
MANIFEST=""
BATCH_SIZE=4 # batch size for ASR transcribe
NUM_JOBS=-2 # The maximum number of concurrently running jobs, `-2` - all CPUs but one are used

# Thresholds for filtering
CER_THRESHOLD=30
WER_THRESHOLD=75
CER_EDGE_THRESHOLD=60
LEN_DIFF_RATIO_THRESHOLD=0.3
MIN_DURATION=1 # in seconds
MAX_DURATION=20 # in seconds
EDGE_LEN=5 # number of characters for calculating edge cer

# Parse arguments
for ARG in "$@"; do
    key=$(echo "$ARG" | cut -f1 -d=)
    value=$(echo "$ARG" | cut -f2 -d=)

    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        declare $v="${value}"
    fi
done

# Validate required arguments
if [[ -z $INPUT_AUDIO_DIR ]] || [[ -z $MANIFEST ]] || [[ -z $MODEL_NAME_OR_PATH ]]; then
  echo "Usage: $(basename "$0")
  --MODEL_NAME_OR_PATH=[space-separated paths to .nemo models or pre-trained model names]
  --INPUT_AUDIO_DIR=[path to original directory with audio files used for segmentation (for retention rate estimate)]
  --MANIFEST=[path to manifest file generated during segmentation]"
  exit 1
fi

# Fix dirname usage and define output directory
OUT_MANIFEST_DIR=$(dirname "$MANIFEST")
MANIFESTS_TO_PROCESS=()

# Split MODEL_NAME_OR_PATH into an array
IFS=' ' read -r -a MODEL_PATHS <<< "$MODEL_NAME_OR_PATH"

for MODEL in "${MODEL_PATHS[@]}"; do
  MODEL_LOWERCASE="${MODEL,,}" # Convert model name to lowercase for easier checks
  
  if [[ $MODEL_LOWERCASE == *".nemo" ]] || [[ $MODEL_LOWERCASE == *"nvidia"* ]] || [[ $MODEL_LOWERCASE == *"nemo"* ]]; then
    if [[ ${MODEL_NAME_OR_PATH,,} == *".nemo" ]]; then
        ARG_MODEL="model_path";
    else
        ARG_MODEL="pretrained_name";
    fi
    # Use transcribe_speech.py for .nemo models or NVIDIA models
    OUT_MANIFEST="${OUT_MANIFEST_DIR}/manifest_transcribed_${MODEL_LOWERCASE//[^a-zA-Z0-9]/_}.json"
    echo "--- Adding transcripts to ${OUT_MANIFEST} using ${MODEL} ---"
    python ./transcribe_speech.py \
    $ARG_MODEL="$MODEL" \
    dataset_manifest="$MANIFEST" \
    output_filename="$OUT_MANIFEST" \
    use_cer=True \
    batch_size="${BATCH_SIZE}" \
    num_workers=0 || exit
  else
    # Use custom_transcribe_speech.py for other models
    OUT_MANIFEST="${OUT_MANIFEST_DIR}/manifest_transcribed_${MODEL_LOWERCASE//[^a-zA-Z0-9]/_}.json"
    echo "--- Adding transcripts to ${OUT_MANIFEST} using ${MODEL} ---"
    python "${SCRIPTS_DIR}/custom_transcribe_speech.py" \
    model_name="$MODEL" \
    dataset_manifest="$MANIFEST" \
    output_filename="$OUT_MANIFEST" \
    batch_size=1 || exit
  fi

  MANIFESTS_TO_PROCESS+=("$OUT_MANIFEST")
done

# Handle manifest array properly for metrics and filtering
echo "--- Calculating metrics and filtering out samples based on thresholds ---"
echo "CER_THRESHOLD = ${CER_THRESHOLD}"
echo "WER_THRESHOLD = ${WER_THRESHOLD}"
echo "CER_EDGE_THRESHOLD = ${CER_EDGE_THRESHOLD}"
echo "LEN_DIFF_RATIO_THRESHOLD = ${LEN_DIFF_RATIO_THRESHOLD}"

MANIFESTS_ARG=$(IFS=" " ; echo "${MANIFESTS_TO_PROCESS[*]}")
python ${SCRIPTS_DIR}/get_metrics_and_filter_multiple.py \
--manifests="$MANIFESTS_ARG" \
--audio_dir="${INPUT_AUDIO_DIR}" \
--max_cer="${CER_THRESHOLD}" \
--max_wer="${WER_THRESHOLD}" \
--max_len_diff_ratio="${LEN_DIFF_RATIO_THRESHOLD}" \
--max_edge_cer="${CER_EDGE_THRESHOLD}" \
--min_duration="${MIN_DURATION}" \
--max_duration="${MAX_DURATION}" \
--edge_len="${EDGE_LEN}"

