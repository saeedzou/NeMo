import argparse
import json
import os

import editdistance
from joblib import Parallel, delayed
from tqdm import tqdm
from glob import glob

from nemo.utils import logging

from mutagen import File  # Importing mutagen for audio file duration calculation


parser = argparse.ArgumentParser("Calculate metrics and filters out samples based on thresholds")
parser.add_argument(
    "--manifests", required=True, type=str, help="Space-separated list of .json manifest file paths with ASR predictions"
)
parser.add_argument(
    "--edge_len", type=int, help="Number of characters to use for CER calculation at the edges", default=5
)
parser.add_argument("--audio_dir", type=str, help="Path to original .wav files", default=None)
parser.add_argument("--max_cer", type=int, help="Threshold CER value, %", default=30)
parser.add_argument("--max_wer", type=int, help="Threshold WER value, %", default=75)
parser.add_argument(
    "--max_len_diff_ratio",
    type=float,
    help="Threshold for len diff ratio between reference text "
    "length and predicted text length with respect to "
    "the reference text length (length measured "
    "in number of characters)",
    default=0.3,
)
parser.add_argument("--max_edge_cer", type=int, help="Threshold edge CER value, %", default=60)
parser.add_argument("--max_duration", type=int, help="Max duration of a segment, seconds", default=-1)
parser.add_argument("--min_duration", type=int, help="Min duration of a segment, seconds", default=1)
parser.add_argument(
    "--num_jobs",
    default=-2,
    type=int,
    help="The maximum number of concurrently running jobs, `-2` - all CPUs but one are used",
)
parser.add_argument(
    "--only_filter",
    action="store_true",
    help="Set to True to perform only filtering (when transcripts" "are already available)",
)


def _calculate(line: dict, edge_len: int):
    """
    Calculates metrics for every entry on manifest.json.

    Args:
        line - line of manifest.json (dict)
        edge_len - number of characters for edge Character Error Rate (CER) calculations

    Returns:
        line - line of manifest.json (dict) with the following metrics added:
        WER - word error rate
        CER - character error rate
        start_CER - CER at the beginning of the audio sample considering first 'edge_len' characters
        end_CER - CER at the end of the audio sample considering last 'edge_len' characters
        len_diff_ratio - ratio between reference text length and predicted text length with respect to
            the reference text length (length measured in number of characters)
    """
    logging.debug(f"Calculating metrics for line with audio file: {line.get('audio_filepath')}")
    eps = 1e-9

    text = line["text"].split()
    pred_text = line["pred_text"].split()

    num_words = max(len(text), eps)
    word_dist = editdistance.eval(text, pred_text)
    line["WER"] = word_dist / num_words * 100.0
    num_chars = max(len(line["text"]), eps)
    char_dist = editdistance.eval(line["text"], line["pred_text"])
    line["CER"] = char_dist / num_chars * 100.0

    line["start_CER"] = editdistance.eval(line["text"][:edge_len], line["pred_text"][:edge_len]) / edge_len * 100
    line["end_CER"] = editdistance.eval(line["text"][-edge_len:], line["pred_text"][-edge_len:]) / edge_len * 100
    line["len_diff_ratio"] = 1.0 * abs(len(text) - len(pred_text)) / max(len(text), eps)
    return line


def get_metrics(manifest, manifest_out):
    """Calculate metrics for sample in manifest and saves the results to manifest_out"""
    logging.info(f"Calculating metrics for manifest: {manifest}")
    with open(manifest, "r") as f:
        lines = f.readlines()

    lines = Parallel(n_jobs=args.num_jobs)(
        delayed(_calculate)(json.loads(line), edge_len=args.edge_len) for line in tqdm(lines)
    )
    with open(manifest_out, "w") as f_out:
        for line in lines:
            f_out.write(json.dumps(line) + "\n")
    logging.info(f"Metrics save at {manifest_out}")


def filter_manifests(manifests, manifest_with_metrics_filtered, original_duration):
    """
    Filters samples based on criteria across multiple manifests.
    Retains only samples that pass the thresholds in ALL manifests.
    Uses the manifest with the lowest CER as the representative for each sample.

    Args:
        manifests: List of manifest file paths.
    """
    logging.info(f"Filtering manifests: {manifests}")
    combined_data = {}
    segmented_duration = 0

    # Load and combine data from all manifests
    for manifest_idx, manifest_path in enumerate(manifests):
        logging.info(f"Processing manifest: {manifest_path}")
        with open(manifest_path, "r") as f:
            for line in f:
                item = json.loads(line)
                filepath = item["audio_filepath"]
                if manifest_idx == 0:
                    duration = item["duration"]
                    segmented_duration += duration

                if filepath not in combined_data:
                    combined_data[filepath] = []

                # Add model name to the entry
                item["model_name"] = os.path.basename(manifest_path).split("manifest_transcribed_")[-1].split("_metrics.json")[0]
                combined_data[filepath].append(item)

    filtered_data = []
    retained_duration = 0
    for filepath, entries in combined_data.items():
        # Check if the sample passes thresholds in all manifests
        passes_any = any(
            entry["CER"] <= args.max_cer
            and entry["WER"] <= args.max_wer
            and entry["len_diff_ratio"] <= args.max_len_diff_ratio
            and entry["end_CER"] <= args.max_edge_cer
            and entry["start_CER"] <= args.max_edge_cer
            and (args.max_duration == -1 or entry["duration"] < args.max_duration)
            and entry["duration"] > args.min_duration
            for entry in entries
        )
        if passes_any:
            # Use the entry with the lowest CER as the representative
            best_entry = min(entries, key=lambda x: x["CER"])
            retained_duration += best_entry["duration"]
            filtered_data.append(best_entry)

    logging.info("-" * 50)
    logging.info("Threshold values:")
    logging.info(f"max WER, %: {args.max_wer}")
    logging.info(f"max CER, %: {args.max_cer}")
    logging.info(f"max edge CER, %: {args.max_edge_cer}")
    logging.info(f"max Word len diff: {args.max_len_diff_ratio}")
    logging.info(f"max Duration, s: {args.max_duration}")
    logging.info("-" * 50)

    retained_duration /= 60  # Convert to minutes
    segmented_duration /= 60
    original_duration /= 60
    logging.info(f"Original audio duration: {round(original_duration, 2)} min")
    logging.info(
        f"Segmented duration: {round(segmented_duration, 2)} min "
        f"({round(100 * segmented_duration / original_duration, 2)}% of original audio)"
    )
    logging.info(
        f"Retained {round(retained_duration, 2)} min "
        f"({round(100 * retained_duration / original_duration, 2)}% of original or "
        f"{round(100 * retained_duration / segmented_duration, 2)}% of segmented audio)."
    )
    logging.info(f"Filtered data saved to {manifest_with_metrics_filtered}")
    with open(manifest_with_metrics_filtered, "w") as f_out:
        for item in filtered_data:
            f_out.write(json.dumps(item) + "\n")

    logging.info(f"Filtered data saved to {manifest_with_metrics_filtered}")
    
def calculate_original_duration(audio_dir):
    logging.info(f"Calculating original audio duration from directory: {audio_dir}")
    total_duration = 0
    if audio_dir:
        audio_files = glob(f"{os.path.abspath(audio_dir)}/*")
        for audio in audio_files:
            try:
                audio_file = File(audio)
                if audio_file is not None and audio_file.info.length:
                    total_duration += audio_file.info.length
            except Exception as e:
                logging.warning(f"Skipping {audio} due to error: {e}")
    return total_duration


if __name__ == "__main__":
    args = parser.parse_args()
    manifests = args.manifests.split()
    logging.info(f"Arguments: {args}")
    metric_manifests = []

    original_duration = calculate_original_duration(args.audio_dir)

    if not args.only_filter:
        for manifest in manifests:
            manifest_with_metrics = manifest.replace(".json", "_metrics.json")
            logging.info(f"Generating metrics for {manifest}")
            get_metrics(manifest, manifest_with_metrics)
            metric_manifests.append(manifest_with_metrics)
    else:
        metric_manifests = manifests

    manifest_with_metrics_filtered = os.path.join(
        os.path.dirname(manifests[0]), "manifest_transcribed_metrics_filtered.json"
    )
    filter_manifests(metric_manifests, manifest_with_metrics_filtered, original_duration)