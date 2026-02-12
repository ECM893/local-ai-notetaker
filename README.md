# LAIN - Local AI Notetaker

LAIN is an offline CLI tool that generates meeting notes from Zoom recordings using NVIDIA Parakeet ASR for transcription and Ollama thinking models for summarization. All processing runs locally on your hardware.

The current version works with Zoom meetings where each participant has their own audio file.

## Workflow

1. Select a Zoom recording folder
2. Convert audio files to WAV (ffmpeg)
3. Transcribe individual audio streams (Parakeet-TDT)
4. Interleave transcripts by timestamp
5. Generate meeting notes (Ollama thinking models)
6. Save as Markdown

## Prerequisites

- **Python 3.12** (`python --version`)
- **NVIDIA GPU** with at least 12GB VRAM and compatible CUDA drivers
- **ffmpeg** installed and in your `PATH`
- **Ollama** server running with a thinking model pulled

### System Dependencies (Ubuntu/Debian)

```bash
sudo apt update && sudo apt install ffmpeg pandoc
```

## Installation

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.

2. From the project root:
   ```bash
   uv sync
   ```

## Usage

```bash
# Basic usage - process a Zoom recording folder
lain -f <zoom_recording_folder>

# Full options
lain -f <zoom_recording_folder> \
     -l <ollama_model> \
     -o <output_folder> \
     -s "YYYY-MM-DD HH:MM:SS" \
     --overwrite

# Show all options
lain --help
```

If you haven't activated the virtualenv, prefix with `uv run`:
```bash
uv run lain -f <zoom_recording_folder>
```

## Zoom Settings

Set local recording storage for your Zoom recordings.

From the Zoom app, go to **Settings > Recording > Advanced** and enable:
- **Record separate participant audio files**
- **Optimize for 3rd party video editors**

### Expected Folder Structure

The `Audio Record` folder must be named exactly that.

```
<meeting_folder>/
    <master_recording>.m4a
    Audio Record/
        <audio1>.m4a
        <audio2>.m4a
        ...
```

## Notes

### LLM Compatibility

Currently only thinking models from Ollama are supported. Ensure your Ollama server is running and the model is pulled before running LAIN.

### GPU Support

A compatible NVIDIA GPU with CUDA support is required. At least 12GB VRAM is recommended for the ASR model.

## TODO

- Support for simpler (non-thinking) LLMs
- Single audio file with diarization (pyannote)
- Non-Ollama model backends
- Cross-platform testing (currently validated on Ubuntu/Arch)
