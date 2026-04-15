import os
import json
import logging
import argparse
import whisperx
import torch

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def process_audio(audio_path, device="cpu", model_size="base"):
    try:
        # 1. Transcribe
        logger.info(f"Loading Whisper model: {model_size}")
        model = whisperx.load_model(model_size, device, compute_type="int8")
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16)

        # 2. Align
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device)

        # 3. Diarize (with safety fallback)
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                logger.info("Starting Diarization...")
                diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as e:
                logger.warning(f"Diarization failed: {e}. Outputting transcription only.")
        else:
            logger.warning("No HF_TOKEN found. Skipping speaker identification.")

        # 4. Format Output
        output = {
            "metadata": {
                "model": model_size,
                "device": device,
                "language": result.get("language", "unknown")
            },
            "segments": [
                {
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "text": s.get("text"),
                    "speaker": s.get("speaker", "Unknown")
                } for s in result["segments"]
            ]
        }
        return output

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to audio file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = process_audio(args.input, device=device)

    with open("output.json", "w") as f:
        json.dump(data, f, indent=4)
    logger.info("Done. Check output.json")

if __name__ == "__main__":
    main()
