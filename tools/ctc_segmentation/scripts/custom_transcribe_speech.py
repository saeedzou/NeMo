import json
import os
import subprocess
from typing import Optional
from dataclasses import dataclass, is_dataclass
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from hezar.models import Model
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer
from nemo.utils import logging
from nemo.core.config import hydra_runner
from omegaconf import OmegaConf
from tqdm import tqdm

# Similar to the previous script, we define the transcription configuration
@dataclass
class TranscriptionConfig:
    """
    Transcription Configuration for audio to text transcription using Whisper.
    """

    model_name: str = "masoudmzb/wav2vec2-xlsr-multilingual-53-fa"  # Pretrained model name
    model_type: str = 'wav2vec2'
    audio_dir: Optional[str] = None  # Path to a directory containing audio files
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest
    output_filename: Optional[str] = None  # Path to output file for transcriptions
    batch_size: int = 1
    compute_dtype: str = "float32"
    overwrite_transcripts: bool = True  # Recompute transcription even if output exists
    audio_type: str = "mp3"  # Supported audio formats (wav, mp3, flac)
    timestamps: bool = False  # Whether to include timestamps or not
    cuda: Optional[int] = 0  # Device ID to use for inference, None for CPU
    append_pred: bool = False  # Whether to append to existing predictions in the output file
    gt_text_attr_name: str = "text"
    pred_text_attr_name: str = "pred_text"
    clean_groundtruth_text: bool = False
    langid: str = 'fa'
    use_cer: bool = True
    vosk_cli_path: Optional[str] = "vosk-transcriber"  # Path to Vosk CLI executable
    temp_output_file: str = "temp_transcription.txt"

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
def load_model_and_processor(cfg: TranscriptionConfig):
    if cfg.model_type == "hezar":
        model = Model.load(cfg.model_name)
        processor = None
    elif cfg.model_type == "whisper":
        model = WhisperForConditionalGeneration.from_pretrained(cfg.model_name)
        processor = WhisperProcessor.from_pretrained(cfg.model_name)
    elif cfg.model_type == "wav2vec2":
        model = Wav2Vec2ForCTC.from_pretrained(cfg.model_name)
        processor = Wav2Vec2Processor.from_pretrained(cfg.model_name)
    elif cfg.model_type == "vosk":
        model, processor = None, None
    else:
        raise ValueError(f"Unsupported model type: {cfg.model_type}")
    
    return model, processor

def process_with_vosk(audio_file, cfg: TranscriptionConfig):
    """Call Vosk CLI to transcribe the given audio file and save output to a temp file."""
    command = [cfg.vosk_cli_path, 
               '-l', cfg.langid, 
               '-i', audio_file,
               '--model-name', cfg.model_name,
               '-o', cfg.temp_output_file,
               '--log-level', 'error'] 
    
    try:
        # Run the Vosk CLI command
        subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Read the transcription from the temporary output file
        if os.path.exists(cfg.temp_output_file):
            with open(cfg.temp_output_file, 'r') as file:
                output_json = file.read()
                return output_json  # Extract transcribed text
        else:
            logging.error(f"Temporary output file {cfg.temp_output_file} not found.")
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Vosk CLI failed for {audio_file}: {e}")
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON output in {cfg.temp_output_file}")
    finally:
        # Clean up the temporary file
        if os.path.exists(cfg.temp_output_file):
            os.remove(cfg.temp_output_file)
    
    return ""

# This function processes a batch of audio files in parallel, improving efficiency
def process_audio_files_batch(batch_files, processor, model_type):
    if model_type == 'hezar' or 'vosk':
        return None
    audio_data_list = []
    for audio_file in batch_files:
        audio_data, sampling_rate = torchaudio.load(audio_file, normalize=True)
        audio_data = audio_data.mean(dim=0)  # Convert stereo to mono if necessary
        inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt")
        if model_type == "whisper":
            # Whisper-specific processing
            audio_data_list.append(inputs.input_features)
        elif model_type == "wav2vec2":
            # Wav2Vec2-specific processing
            audio_data_list.append(inputs.input_values)

    # Stack all the processed audio features in one batch
    return torch.cat(audio_data_list, dim=0)  # Combine into one tensor


def transcribe_audio(cfg: TranscriptionConfig, model, processor, device, model_type):
    # Prepare manifest (list of audio files)
    filepaths = []
    if cfg.dataset_manifest:
        with open(cfg.dataset_manifest, 'r') as f:
            filepaths = [json.loads(line) for line in f]

    elif cfg.audio_dir:
        filepaths = [os.path.join(cfg.audio_dir, fname) for fname in os.listdir(cfg.audio_dir) if fname.endswith(cfg.audio_type)]

    if not filepaths:
        raise ValueError("No audio files found to transcribe")
    
    # Assert that batch size is 1 for wav2vec2 model
    if model_type == "wav2vec2" and cfg.batch_size != 1:
        raise ValueError(f"Batch size for wav2vec2 must be 1, but got {cfg.batch_size}.")
    # Get the forced_decoder_ids if the model is Whisper
    forced_decoder_ids = None
    if cfg.model_type == "whisper" and cfg.langid=='fa':
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="fa", task="transcribe")  # Adjust language if needed
    
    transcriptions = []
    for i in tqdm(range(0, len(filepaths), cfg.batch_size)):
        batch_files = [x['audio_filepath'] for x in filepaths[i:i + cfg.batch_size]]
        print(batch_files)
        
        # Process the batch of audio files
        audio_input = process_audio_files_batch(batch_files, processor, model_type).to(device) if model_type not in ['hezar', 'vosk'] else None
        
        # Start transcription
        with torch.no_grad():
            if model_type == 'hezar':
                transcriptions_batch = [x['text'] for x in model.predict(batch_files)]
            elif model_type == 'vosk':
                transcriptions_batch = [process_with_vosk(batch_files[0], cfg)]
            elif model_type == "whisper":
                # Whisper-specific transcription logic
                generated_ids = model.generate(audio_input, forced_decoder_ids=forced_decoder_ids)
                transcriptions_batch = processor.batch_decode(generated_ids, skip_special_tokens=True)
            elif model_type == "wav2vec2":
                # Wav2Vec2-specific transcription logic
                logits = model(audio_input).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcriptions_batch = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        
        transcriptions.extend(transcriptions_batch)

    # Write the transcriptions to the output file in manifest format
    write_transcriptions_to_disk(cfg.output_filename, transcriptions, filepaths)
    if os.path.exists(cfg.output_filename) and not cfg.overwrite_transcripts:
        raise ValueError(f"Output file {cfg.output_filename} already exists. Set `overwrite_transcripts=True` to overwrite.")
    
    _, total_res, _ = cal_write_wer(
        pred_manifest=cfg.output_filename,
        gt_text_attr_name=cfg.gt_text_attr_name,
        pred_text_attr_name=cfg.pred_text_attr_name,
        clean_groundtruth_text=cfg.clean_groundtruth_text,
        langid=cfg.langid,
        use_cer=cfg.use_cer,
        output_filename=None,
    )

    logging.info(f"Transcriptions written to {cfg.output_filename}")
    logging.info(f"{total_res}")

# Main entry point
@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if 'wav2vec2' in cfg.model_name.lower():
        cfg.model_type = 'wav2vec2'
    elif 'hezar' in cfg.model_name.lower():
        cfg.model_type = 'hezar'
    elif 'whisper' in cfg.model_name.lower():
        cfg.model_type = 'whisper'
    elif 'vosk' in cfg.model_name.lower():
        cfg.model_type = 'vosk'
    else:
        raise ValueError(f'model should be whisper/wav2vec2/hezar/whisper/vosk')

    model, processor = load_model_and_processor(cfg)
    
    # Setup device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda is None else "cpu")
    if cfg.cuda is not None:
        device = torch.device(f"cuda:{cfg.cuda}")
    logging.info(f"Inference will be done on device: {device}")
    if model:
        model.to(device)
    
    transcribe_audio(cfg, model, processor, device, cfg.model_type)


if __name__ == '__main__':
    main()
