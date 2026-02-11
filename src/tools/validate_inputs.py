import argparse
import os
import re
from datetime import datetime


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser):
    args.start_time = (
        datetime.fromisoformat(args.start_time) if args.start_time else None
    )
    if args.start_time:
        print(f"Using start time: {args.start_time}")

    if args.meeting_folder:
        args.meeting_folder = os.path.normpath(args.meeting_folder)

        if not os.path.isdir(args.meeting_folder):
            parser.error(
                f"Meeting folder does not exist: {args.meeting_folder}"
            )

        # If no Start time given, try to extract from folder name
        if not args.start_time:
            args.start_time = get_meeting_start_time_from_folder_name(
                args.meeting_folder
            )
            if args.start_time:
                print(f"Using extracted start time: {args.start_time}")

    if args.output_folder:
        args.output_folder = os.path.normpath(args.output_folder)
        os.makedirs(args.output_folder, exist_ok=True)

    if args.pyannotate_hf_token and args.audio_folder:
        print(
            "Warning: pyannotate_hf_token is set and audio_folder is provided"
        )
        print(
            "Warning: pyannotate is only used for single audio file and will be skipped"
        )

    asr_models = (
        "parakeet-tdt-0.6b-v2",
        "parakeet-tdt-1.1b",
    )
    if args.asr_model not in asr_models:
        parser.error(
            f"Invalid ASR model: {args.asr_model}. Choose from {asr_models}"
        )

    if args.ollama_api == "True":
        args.ollama_api = True
    elif args.ollama_api == "False":
        args.ollama_api = False
    else:
        parser.error("ollama_api must be a boolean value")

    return args


def get_meeting_start_time_from_folder_name(folder_name: str) -> datetime:
    """Zoom Folder based meeting start time extraction"""
    base_folder_name = os.path.basename(folder_name)
    match = re.match(
        r"(\d{4}-\d{2}-\d{2})\s+(\d{2}\.\d{2}\.\d{2})\s+(.*)", base_folder_name
    )
    if match:
        date_str = match.group(1)
        time_str = match.group(2)

        try:
            meeting_start_time = datetime.strptime(
                f"{date_str} {time_str}", "%Y-%m-%d %H.%M.%S"
            )
            print(
                f"Extracted meeting start from filename: {meeting_start_time}"
            )
        except ValueError:
            print(
                "Failed to parse meeting start time.\n Expected Zoom layout like:\n 'YYYY-MM-DD HH.MM.SS <Speaker's Name>'s Zoom Meeting' \n falling back to file creation time"
            )
            meeting_start_time = None
    else:
        print(
            "Failed to extract meeting start time from folder name. Expected Zoom layout like:\n 'YYYY-MM-DD HH.MM.SS <Speaker's Name>'s Zoom Meeting' \n falling back to file creation time"
        )
        meeting_start_time = None

    return meeting_start_time
