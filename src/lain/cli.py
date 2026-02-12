import argparse
import shutil

from lain.tools.validate_inputs import validate_args


def main():
    epilog = (
        "Examples:\n"
        "  # Process a folder of per-speaker audio files and save transcript:\n"
        "  offline-meeting-notes -f ./audio_folder -o notes.txt -\n\n"
        "Notes:\n"
        "  - Use --audio-folder for a folder (one audio file per speaker); files will be converted to WAV if needed.\n"
        "  - Start time can be supplied with --start-time in ISO format (YYYY-MM-DD HH:MM:SS or YYYY-MM-DDTHH:MM:SS).\n"
        "  - --over-write allows overwriting existing output files.\n"
        """
        Structure of the folder should Zoom-like:
        <meeting_folder>/
            <master_recording>.mp4
            Audio Record/
                <audio1>.mp4
                <audio2>.mp4
                ...
        """
    )
    parser = argparse.ArgumentParser(
        description="Offline Meeting Notetaker",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--meeting-folder",
        help="Path to a folder of meeting audio files (one per speaker, will be converted to WAV if needed).",
    )
    parser.add_argument(
        "-p",
        "--pyannotate-hf-token",
        help="Hugging Face token for pyannote (required for diarization).",
    )
    parser.add_argument(
        "-m",
        "--asr-model",
        default="parakeet-tdt-0.6b-v2",
        help="ASR model name (e.g., parakeet-tdt-0.6b-v2, parakeet-tdt-1.1b).",
    )
    parser.add_argument(
        "-l",
        "--language-model",
        default="gpt-oss:20b",
        help="Language model ID for summarization (e.g., gpt-oss:20b).",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        help="Output folder to save transcript and notes (default: Transcripts/<meeting-folder-name>/).",
        default=None,
    )
    parser.add_argument(
        "-s",
        "--start-time",
        help="Wall Clock start time in ISO 8601 format (e.g. '2023-01-01 10:00:00' or '2023-01-01T10:00:00'). default: None, will use the time the recording was made",
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if present.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="offline_meeting_notes 0.1.0",
        help="Show program version and exit.",
    )
    parser.add_argument(
        "--ollama-api",
        default="True",
        help="Use Ollama API for note generation.",
    )

    args = parser.parse_args()
    args = validate_args(args, parser)

    # Ensure ffmpeg is installed for audio processing
    if shutil.which("ffmpeg") is None:
        parser.error(
            "ffmpeg is required but not found in PATH. Please install ffmpeg."
        )

    if args:
        from lain.note_taker_pipeline import note_taker_pipeline

        note_taker_pipeline(args)


if __name__ == "__main__":
    main()
