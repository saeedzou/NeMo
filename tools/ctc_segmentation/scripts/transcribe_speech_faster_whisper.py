import argparse
import json
import torch
import os
from faster_whisper import WhisperModel
from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer
from nemo.utils import logging
from tqdm import tqdm


def write_transcriptions_to_disk(output_filename, transcriptions, filepaths):
    output_data = [{
        'audio_filepath': filepaths[idx]['audio_filepath'],
        'duration': filepaths[idx]['duration'],
        'text': filepaths[idx]['text'],
        'text_no_preprocessing': filepaths[idx]['text_no_preprocessing'],
        'text_normalized': filepaths[idx]['text_normalized'],
        'score': filepaths[idx]['score'],
        'start_abs': filepaths[idx]['start_abs'],
        'end_abs': filepaths[idx]['end_abs'],
        'pred_text': transcription,
    } for idx, transcription in enumerate(transcriptions)]
    
    # Write the entire batch to the output file at once
    with open(output_filename, 'w') as output_file:
        output_file.writelines([json.dumps(entry) + '\n' for entry in output_data])

# Load faster_whisper model
def load_model(model_path, device, compute_type):
    """Loads the faster_whisper model."""
    logging.info(f"Loading faster_whisper model: {model_path} with compute_type: {compute_type} on device: {device}")
    model = WhisperModel(model_path, device=device, compute_type=compute_type)
    return model

# faster_whisper handles audio loading and processing internally, so process_audio_files_batch is not needed.


def transcribe_audio(model, device, dataset_manifest, dataset_manifest_transcribed, batch_size=4, language='fa'): # Added language parameter
    # Load all input filepaths
    with open(dataset_manifest, 'r') as f:
        all_filepaths = [json.loads(line) for line in f]

    # Load already transcribed filepaths (if output exists)
    already_transcribed = {}
    if os.path.exists(dataset_manifest_transcribed):
        with open(dataset_manifest_transcribed, 'r') as f:
            for line in f:
                entry = json.loads(line)
                already_transcribed[entry['audio_filepath']] = entry

    # Filter out already processed entries
    filepaths_to_process = [entry for entry in all_filepaths if entry['audio_filepath'] not in already_transcribed]

    if not filepaths_to_process:
        logging.info("All files already transcribed.")
        return

    logging.info(f"{len(filepaths_to_process)} files remaining to transcribe.")

    # Keep appending to the output manifest instead of overwriting
    with open(dataset_manifest_transcribed, 'a') as output_file:
        for i in tqdm(range(0, len(filepaths_to_process), batch_size)):
            batch_entries = filepaths_to_process[i:i + batch_size]
            batch_files = [x['audio_filepath'] for x in batch_entries]

            # faster_whisper's transcribe method processes files individually or in batches internally
            # It takes file paths directly.
            # The transcribe method yields segments, so we need to concatenate them.
            
            transcriptions_batch = []
            for audio_file in batch_files:
                 # beam_size=5 is a common default, language='fa' for Persian
                segments, info = model.transcribe(audio_file, beam_size=5, language=language) 
                full_transcription = "".join([segment.text for segment in segments])
                transcriptions_batch.append(full_transcription)

            # Write results for the batch
            for idx, transcription in enumerate(transcriptions_batch):
                entry = batch_entries[idx]
                entry['pred_text'] = transcription.strip() # Add strip() for potential leading/trailing spaces
                output_file.write(json.dumps(entry) + '\n')
                output_file.flush()  # ensure progress is saved in case of interruption

    # WER calculation (optional, can be done post-hoc if partial run)
    _, total_res, _ = cal_write_wer(
        pred_manifest=dataset_manifest_transcribed,
        gt_text_attr_name="text",
        pred_text_attr_name="pred_text",
        clean_groundtruth_text=False,
        langid='fa',
        use_cer=True,
        output_filename=None,
    )

    logging.info(f"Transcriptions written to {dataset_manifest_transcribed}")
    logging.info(f"{total_res}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe speech using faster-whisper model.")
    parser.add_argument("--model_path", required=True, type=str, help="Path or name of the faster-whisper model (e.g., 'large-v2', 'distil-large-v2').")
    parser.add_argument("--dataset_manifest", required=True, type=str, help="Path to the dataset manifest file.")
    parser.add_argument("--dataset_manifest_transcribed", required=True, type=str, help="Path to save the transcribed manifest file.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing manifest entries (faster-whisper handles internal batching).") # Increased default batch size
    parser.add_argument("--device", type=str, default="auto", help="Device for inference ('cuda', 'cpu', 'auto').")
    parser.add_argument("--compute_type", type=str, default="float16", help="Compute type for faster-whisper ('float16', 'int8_float16', 'int8', 'float32').")
    parser.add_argument("--language", type=str, default="fa", help="Language code for transcription (e.g., 'en', 'fa').")


    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load model
    model = load_model(args.model_path, device, args.compute_type)
    # processor is removed
    # model = model.to(device) # faster-whisper handles device placement during init

    logging.info(f"Inference will be done on device: {device} with compute_type: {args.compute_type}")
    # Pass language to transcribe_audio
    transcribe_audio(model, device, args.dataset_manifest, args.dataset_manifest_transcribed, batch_size=args.batch_size, language=args.language)


if __name__ == '__main__':
    main()
