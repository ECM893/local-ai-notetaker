"""
Transcription utilities using WhisperX for multiple single speaker audio files.
Multi-speaker audio files (diarization) not yet implemented.
"""

import pickle
from datetime import datetime, timedelta

import torch
import whisperx


def transcribe_audio_multi(
    wav_files,
    meeting_start_time: datetime | None = None,
    model_size: str = "turbo",
):
    """
    Transcribe multiple speaker audio files, optionally using WhisperX.

    Parameters
    ----------
    wav_files : dict of str to str
        Mapping from speaker labels to WAV file paths.
    meeting_start_time : datetime or None, optional
        Start time of the meeting to offset segment timestamps. Default is None.
    model_size : str, optional
        Whisper model size to load. Default is 'turbo'.
    whisperx : bool, optional
        If True, use whisperx instead of standard Whisper. Default is False.

    Returns
    -------
    dict of str to list of dict
        Mapping from speaker labels to lists of segment dictionaries.
    """

    if meeting_start_time is None:
        # set to midnight today
        meeting_start_time = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    transcriptions = {}
    batch_size = 16
    compute_type = "float16"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading WhisperX model...")
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    model_a, metadata = whisperx.load_align_model(
        language_code="en", device=device
    )

    print(f"Loading model to device: {device}")

    print("Starting WhisperX Transcribe model...")
    for speaker, file in wav_files.items():
        print(f"Transcribing audio for {speaker}...")
        audio = whisperx.load_audio(file)
        result = model.transcribe(audio, batch_size=batch_size)

        if result["language"] != "en":
            raise ValueError(
                "Currently only English language is supported for this pipeline, literally just change the whisperx.load_align_model(language_code='en') above this code to fix."
            )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        segments = []
        for seg in result.get("segments", []):
            # meeting starttime is a datetime object, and i want this to be displayed in the hour:minute format
            if meeting_start_time:
                seg["start"] = meeting_start_time + timedelta(
                    seconds=seg["start"]
                )
                seg["end"] = meeting_start_time + timedelta(seconds=seg["end"])

            segments.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                    "speaker": speaker,
                }
            )
        transcriptions[speaker] = segments

    # Clean up and delete model
    del model
    del model_a
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
