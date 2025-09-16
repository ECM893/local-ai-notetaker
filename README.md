# Offline Meeting Notes

Offline Meeting Notes is a CLI tool that generates concise meeting minutes from an audio file using, WhisperX (OpenAI Whisper, faster-whisper, phoneme-model) for transcription/Alignment, and local transformer-based summarization (via Ollama).
The current version works with Zoom meetings where each participand has their own audio file.


## Basic Workflow
1. Zoom recording folder selection
2. Audio file conversion to WAV (ffmpeg)
3. Audio file offset Calculation (librosa)(currently not implemented)
4. Individual Audio stream transcription (Whisper/Whisperx)
5. Transcript Interleaving
6. Meeting notetaking (Ollama python API using Thinking models)
7. Markdown meeting notes to Word doc conversion (pandoc/pypandoc)

## Prerequisites

**Python:**
- Requires Python 3.11 (You can verify with `python --version`).

**System dependencies:**
- **ffmpeg** must be installed and available in your `PATH`.
  - On Ubuntu/Debian:
    ```bash
    sudo apt update; sudo apt install ffmpeg
    ```

## Installation with UV

1. Install 'ffmpeg':
   ```bash
   sudo apt update; sudo apt install ffmpeg
   ```

2. Install Pandoc (if you haven’t already):
   ```bash
   sudo apt update; sudo apt install pandoc
   ```

3. Install UV (if you haven’t already):
   ```bash
   pip3 install --user uv
   ```

4. From the project root (where `pyproject.toml` is):
   - Install runtime dependencies:
     ```bash
     uv sync
     ```
    > **(Optional) If you see a warning about failed hardlinks:**
    > - Set the environment variable:
    >   ```bash
    >   export UV_LINK_MODE=copy
    >   ```
    > - Or use the `--link-mode=copy` flag:
    >   ```bash
    >   uv sync --link-mode=copy
    >   ```
     ```

5. It is possible to have an error like: 'Could not load library libcudnn_ops_infer.so.8. Error: libcudnn_ops_infer.so.8: cannot open shared object file: No such file or directory
Aborted (core dumped)' if you do, check the bottom notes for a solution

## Usage

To run the notetaker without activating the venv:
```bash
uv run offline-meeting-notes -f <zoom_recording_folder>
```

Or activate manually in Bash and run:
```bash
source .venv/bin/activate
offline-meeting-notes -h
offline-meeting-notes -f <zoom_recording_folder> -l <llm model> -o <output_folder> --start-time "YYYY-MM-DD HH:MM:SS" --over-write
```

## Notes:
# Zoom Settings
Set Local recording storage for your zoom recordings.

It's recommended (required) to adjust some settings for your zoom recordings:
From zoom app -> Settings -> Recording -> Advanced:
- Record separate participant audio files
- Optimize for 3rd party video editors

# LLM Compatibility
Currently Only 'thinking' models from Ollama are supported. Ensure your Ollama server is running and accessible.
# GPU Support
Ensure you have a compatible NVIDIA GPU and the correct CUDA version installed for PyTorch. (recommended at least 12GB VRAM)
# Zoom Recording Structure
The folder structure should be similar to Zoom's default recording layout
The 'Audio Record' folder must be named exactly that.
Structure must look like:
```
<meeting_folder>/
    <master_recording>.mp4
    Audio Record/
        <audio1>.mp4
        <audio2>.mp4
        ...
```
# Whisperx known issue
If using whisperx and you get a 'Could not load library libcudnn_ops_infer.so.8. Error: libcudnn_ops_infer.so.8: cannot open shared object file: No such file or directory
Aborted (core dumped)' error, try: (This fix is for Ubuntu distros)

```
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-cuda-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/nvidia-cuda.list
sudo apt update
sudo apt install libcudnn8 libcudnn8-dev
echo "/usr/lib/x86_64-linux-gnu" | sudo tee /etc/ld.so.conf.d/cudnn.conf
sudo ldconfig
```
and check 

```
ldconfig -p | grep libcudnn_ops_infer
```

should print 
```
(whisperx) ➜  ~ ldconfig -p | grep libcudnn_ops_infer
        libcudnn_ops_infer.so.8 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8
        libcudnn_ops_infer.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so
```
# TODO:
- Add support for simpler LLMs (Non-thinking models)
- Add support for single audio file with diarization (pyannote)
- Add support for non-Ollama based Models
- Test fresh installation
- Add updated tests
- Check OS support, shouldn't matter but still (Currently working on Ubuntu 24)
- Remake Pytests to actually work again
