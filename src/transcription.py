"""
Transcription utilities using NVIDIA Parakeet-TDT for multiple single-speaker audio files.
Uses Silero VAD for silence detection to skip empty/silent audio.
Multi-speaker audio files (diarization) not yet implemented.
"""

import pickle
from datetime import datetime, timedelta

import torch


def _load_vad_model():
    """Load Silero VAD model for speech activity detection."""
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    read_audio = utils[2]
    return vad_model, get_speech_timestamps, read_audio


def _has_speech(file_path, vad_model, get_speech_timestamps, read_audio, threshold=0.5):
    """
    Check if an audio file contains speech using Silero VAD.

    Parameters
    ----------
    file_path : str
        Path to a WAV audio file.
    vad_model : torch.nn.Module
        Loaded Silero VAD model.
    get_speech_timestamps : callable
        Silero utility function to extract speech timestamps.
    read_audio : callable
        Silero utility function to read audio files.
    threshold : float, optional
        VAD confidence threshold. Default is 0.5.

    Returns
    -------
    bool
        True if speech is detected, False otherwise.
    """
    audio = read_audio(file_path, sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(
        audio,
        vad_model,
        sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=250,
    )
    return len(speech_timestamps) > 0


def transcribe_audio_multi(
    wav_files,
    meeting_start_time: datetime | None = None,
    model_size: str = "parakeet-tdt-0.6b-v2",
):
    """
    Transcribe multiple speaker audio files using NVIDIA Parakeet-TDT.

    Parameters
    ----------
    wav_files : dict of str to str
        Mapping from speaker labels to WAV file paths.
    meeting_start_time : datetime or None, optional
        Start time of the meeting to offset segment timestamps. Default is None.
    model_size : str, optional
        Parakeet model name to load. Default is 'parakeet-tdt-0.6b-v2'.

    Returns
    -------
    dict of str to list of dict
        Mapping from speaker labels to lists of segment dictionaries.
    """
    import nemo.collections.asr as nemo_asr

    if meeting_start_time is None:
        meeting_start_time = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    transcriptions = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Silero VAD for silence detection
    print("Loading Silero VAD model...")
    vad_model, get_speech_timestamps, read_audio = _load_vad_model()

    # Load Parakeet-TDT model
    print(f"Loading Parakeet-TDT model: {model_size}...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=f"nvidia/{model_size}"
    )

    # Enable local attention for long audio support (up to 3+ hours)
    asr_model.change_attention_model("rel_pos_local_attn", [128, 128])
    asr_model.change_subsampling_conv_chunking_factor(1)

    print(f"Model loaded on device: {device}")
    print("Starting Parakeet-TDT transcription...")

    for speaker, file in wav_files.items():
        print(f"Checking audio for {speaker}...")

        # Silence detection: skip files with no speech
        if not _has_speech(file, vad_model, get_speech_timestamps, read_audio):
            print(f"  No speech detected for {speaker}, skipping.")
            transcriptions[speaker] = []
            continue

        print(f"  Transcribing audio for {speaker}...")
        output = asr_model.transcribe([file], timestamps=True)

        segments = []
        for seg in output[0].timestamp.get("segment", []):
            start_seconds = seg["start"]
            end_seconds = seg["end"]

            if meeting_start_time:
                seg_start = meeting_start_time + timedelta(seconds=start_seconds)
                seg_end = meeting_start_time + timedelta(seconds=end_seconds)
            else:
                seg_start = start_seconds
                seg_end = end_seconds

            segments.append(
                {
                    "start": seg_start,
                    "end": seg_end,
                    "text": seg["segment"].strip(),
                    "speaker": speaker,
                }
            )
        transcriptions[speaker] = segments

    # Clean up models
    del asr_model
    del vad_model
    torch.cuda.empty_cache()
    return transcriptions


def interleave_transcripts(transcriptions: dict[str, list[dict]]) -> list[dict]:
    """
    Interleave transcriptions from different speakers based on chronological order.

    Parameters
    ----------
    transcriptions : dict of str to list of dict
        Mapping of speaker labels to their transcript segment lists.

    Returns
    -------
    list of dict
        A single list of all segments from all speakers, sorted by start time.
        Each segment dict contains keys 'start', 'end', 'text', 'speaker'.
    """
    all_segments = []
    for speaker, segments in transcriptions.items():
        for seg in segments:
            all_segments.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "speaker": speaker,
                }
            )
    # Sort all segments by start time
    all_segments.sort(key=lambda x: x["start"])
    return all_segments


def save_transcript_to_file(
    segments: list[dict],
    output_file: str,
    pickle_bool: bool = False,
    start_time: datetime | None = None,
):
    """
    Save transcript segments to a text file and optionally as a pickle.

    Parameters
    ----------
    segments : list of dict
        List of segment dictionaries with 'start', 'end', 'text', 'speaker'.
    output_file : str
        File path to write the text transcript ('.txt').
    pickle_bool : bool, optional
        If True, also save segments as a pickle file '.pkl'. Default is False.

    Returns
    -------
    None
    """
    if pickle_bool:
        # Save as python object by replacing the .txt extension with .pkl
        output_file_pickle = output_file.replace(".txt", ".pkl")
        with open(output_file_pickle, "wb") as f:
            pickle.dump(segments, f)
        print(f"Pickled transcript saved to {output_file_pickle}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(
            f"Meeting Start Date and Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        for seg in segments:
            start_time = (
                seg["start"].strftime("%H:%M:%S")
                if isinstance(seg["start"], datetime)
                else str(timedelta(seconds=seg["start"]))
            )
            end_time = (
                seg["end"].strftime("%H:%M:%S")
                if isinstance(seg["end"], datetime)
                else str(timedelta(seconds=seg["end"]))
            )
            f.write(
                f"[{start_time} - {end_time}] ({seg['speaker']}) {seg['text']}\n"
            )
    print(f"Text transcript saved to {output_file}")
