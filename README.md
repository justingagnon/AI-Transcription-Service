# Audio Transcription & Diarization Service (POC)

## Overview
This is a proof-of-concept solution for transcribing audio files and identifying speakers (diarization). It leverages **WhisperX**, which builds upon OpenAI's Whisper by adding forced alignment and speaker diarization via `pyannote.audio`.

## Technical Choices & Trade-offs
- **WhisperX vs. Standard Whisper:** Chosen for its ability to provide word-level timestamps and integrated diarization. Standard Whisper often suffers from "hallucination" in timestamps during long silences; WhisperX mitigates this via phoneme alignment.
- **Compute Type:** The implementation uses `int8` quantization for CPU execution. This allows for a reasonable processing speed without requiring a high-end GPU.
- **CLI Approach:** A CLI was chosen for this POC to minimize boilerplate and focus on core AI logic, though the structure is modular enough to be wrapped in a FastAPI endpoint easily.

## Setup Instructions
1. **System Dependencies:** Ensure `ffmpeg` is installed on your system.
2. **Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

## Test Examples
To demonstrate the system's performance, I have included an `examples/` directory containing two audio samples and their corresponding JSON outputs:
- **Example 1:** A short voice segment reciting a poem showing timestamp accuracy.
- **Example 2:** A longer audio file capturing a business meeting with several people.
