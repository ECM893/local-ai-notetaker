"""
Offline Meeting Notes Pipeline
-----------------------------
This module provides a pipeline for processing multi-speaker meeting audio recordings (e.g., Zoom),
transcribing them, and generating professional meeting notes using LLMs (Ollama API).
Called via cli.py

Steps:
    1. Convert and gather audio files
    2. Transcribe audio to text
    3. Interleave and save transcript
    4. Generate meeting notes (Markdown)
    5. Save notes and convert to DOCX

Requirements:
    - Audio files organized by speaker
    - Meeting start time
    - Output folder for results
    - Parakeet-TDT and Ollama models available
"""

import os
from argparse import Namespace

import pypandoc

from lain.tools.log import log

from lain.convert_audio_files import (
    align_audio_file_offsets,
    combine_audio_files,
    convert_audio_files,
    gather_wave_files,
)
from lain.ollama_notes import ollama_api_notes
from lain.transcription import (
    interleave_transcripts,
    save_transcript_to_file,
    transcribe_audio_multi,
)

_STAGE = "Pipeline"


def note_taker_pipeline(args: Namespace):
    """
    This Pipeline is currently designed for Zoom Recordings with multiple audio files
    The structure of the folder at minimum should be:
    meeting_folder/
        master_recording.mp4
        audio_record/
            audio1.mp4
            audio2.mp4
            ...

    Args:
        args (Namespace): Parsed arguments from argparse
            - meeting_folder (str): Path to the meeting folder
            - output_folder (str): Path to save the output files
            - start_time (datetime, optional): Start time of the meeting
            - asr_model (str): ASR model name to use (e.g., parakeet-tdt-0.6b-v2)
            - language_model (str): Language model to use for notes generation
            - ollama_api (bool): Whether to use Ollama API for notes generation
            - overwrite (bool): Whether to overwrite existing transcript files

    Returns:
        None
    """
    # === 1. Get meeting start time ===
    if args.start_time:
        meeting_start_time = args.start_time
        log(_STAGE, f"Using provided start time: {meeting_start_time}")
    else:
        import datetime

        # Set meeting start time to midnight on a standard day (e.g., Jan 1, 2020)
        meeting_start_time = datetime.datetime(2020, 1, 1, 0, 0, 0)
        log(_STAGE, f"No start time provided, using default: {meeting_start_time}")

    # === 2. Convert and process audio files ===
    log(_STAGE, "Processing folder of audio files...")
    convert_audio_files(args.meeting_folder)

    # === 3. Gather and combine WAV files ===
    wav_list = gather_wave_files(args.meeting_folder)
    wav_files = combine_audio_files(wav_list)

    # NOTE: Zoom Audio Files are already padded for the full length of the recording
    # Align get Audio file offsets (optional)
    adjust_offsets = False
    if adjust_offsets:
        # FIXME: Currnetly No need for this when using Zoom Recordings, expanded functionality can use this in the future
        raise NotImplementedError(
            "Offset adjustment not implemented for Zoom recordings."
        )
        master_audio_wav = None
        offsets = align_audio_file_offsets(wav_files, master_audio_wav)
        log(_STAGE, f"Calculated audio offsets: {offsets}")

    log(_STAGE, "Detected audio files:")
    for wav_file in wav_files:
        log(_STAGE, f"  {wav_file}")

    # === 4. Prepare output filenames ===
    output_transcript_filename = os.path.join(
        args.output_folder,
        f"transcript_{meeting_start_time.strftime('%Y%m%d_%H%M')}.txt",
    )
    output_notes_filename = os.path.join(
        args.output_folder,
        f"notes_{meeting_start_time.strftime('%Y%m%d_%H%M')}.md",
    )

    # === 5. Transcribe audio files (Parakeet-TDT) ===
    if not os.path.exists(output_transcript_filename) or args.overwrite:
        transcriptions = transcribe_audio_multi(
            wav_files=wav_files,
            meeting_start_time=meeting_start_time,
            model_size=args.asr_model,
        )
        interleaved_transcript = interleave_transcripts(transcriptions)
        # transcripts are lists of dicts with keys: start, end, text, speaker
        save_transcript_to_file(
            interleaved_transcript,
            output_transcript_filename,
            start_time=meeting_start_time,
        )
    else:
        log(_STAGE, f"Transcript already exists at {output_transcript_filename}, skipping transcription")
        log(_STAGE, "Continuing to generate notes")

    # === 6. Generate meeting notes (Ollama API) ===
    if args.ollama_api:
        notes = ollama_api_notes(
            transcript_path=output_transcript_filename,
            model=args.language_model,
        )
    else:
        # FIXME: Implement non-ollama local model inference here.
        raise NotImplementedError(
            "Only ollama Server method currently validated"
        )

    log(_STAGE, "Generated meeting notes")

    # === 7. Save notes to Markdown file ===
    with open(output_notes_filename, "w", encoding="utf-8") as f:
        f.write(notes)
    log(_STAGE, f"Meeting notes saved to {output_notes_filename}")

    # # === 8. Convert Markdown notes to DOCX format ===
    # print("Converting Markdown notes to DOCX format...")
    # docx_path = output_notes_filename.replace(".md", ".docx")
    # pypandoc.convert_file(output_notes_filename, "docx", outputfile=docx_path)
    # print(f"DOCX meeting notes saved to {docx_path}")
