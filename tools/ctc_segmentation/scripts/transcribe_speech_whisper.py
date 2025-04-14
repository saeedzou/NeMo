import argparse
import json
import torch
import os
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
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

# Load Whisper model and processor
def load_model_and_processor(model_path, processor_path):
    model = WhisperForConditionalGeneration.from_pretrained(model_path, use_cache=False)
    processor = WhisperProcessor.from_pretrained(processor_path, language='fa', task='transcribe')
    return model, processor

# This function processes a batch of audio files in parallel, improving efficiency
def process_audio_files_batch(batch_files, processor):
    input_features_list = []
    attention_mask_list = []

    for audio_file in batch_files:
        audio_data, sampling_rate = torchaudio.load(audio_file, normalize=True)
        audio_data = audio_data.mean(dim=0)  # Convert stereo to mono if necessary
        inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt")

        input_features_list.append(inputs.input_features)

        # If attention_mask exists, use it; otherwise, create one
        if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None:
            attention_mask_list.append(inputs.attention_mask)
        else:
            # Create a dummy attention mask of ones
            dummy_mask = torch.ones(inputs.input_features.shape[:-1], dtype=torch.long)
            attention_mask_list.append(dummy_mask)

    input_features_batch = torch.cat(input_features_list, dim=0)
    attention_mask_batch = torch.cat(attention_mask_list, dim=0)

    return input_features_batch, attention_mask_batch



def transcribe_audio(model, processor, device, dataset_manifest, dataset_manifest_transcribed, batch_size=4):
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

            audio_input, attention_mask = process_audio_files_batch(batch_files, processor)
            audio_input = audio_input.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                generated_ids = model.generate(audio_input, attention_mask=attention_mask)
                transcriptions_batch = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for idx, transcription in enumerate(transcriptions_batch):
                entry = batch_entries[idx]
                entry['pred_text'] = transcription
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
    parser = argparse.ArgumentParser(description="Transcribe speech using Whisper model.")
    parser.add_argument("--model_path", required=True, type=str, help="Path to the pretrained Whisper model.")
    parser.add_argument("--processor_path", required=True, type=str, help="Path to the Whisper processor.")
    parser.add_argument("--dataset_manifest", required=True, type=str, help="Path to the dataset manifest file.")
    parser.add_argument("--dataset_manifest_transcribed", required=True, type=str, help="Path to save the transcribed manifest file.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_path, args.processor_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    logging.info(f"Inference will be done on device: {device}")
    transcribe_audio(model, processor, device, args.dataset_manifest, args.dataset_manifest_transcribed, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
