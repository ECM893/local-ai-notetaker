import argparse
import os
import re
from datetime import datetime

from lain.tools.log import log

_STAGE = "Setup"


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser):
    args.start_time = (
        datetime.fromisoformat(args.start_time) if args.start_time else None
    )
    if args.start_time:
        log(_STAGE, f"Using start time: {args.start_time}")

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
                log(_STAGE, f"Extracted start time: {args.start_time}")

    # Build output folder: default is Transcripts/<meeting-folder-name>/
    # __file__ is src/lain/tools/validate_inputs.py → 4x dirname to reach project root
    if args.output_folder is None and args.meeting_folder:
        meeting_name = os.path.basename(args.meeting_folder)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        args.output_folder = os.path.join(project_root, "Transcripts", meeting_name)
    elif args.output_folder is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        args.output_folder = os.path.join(project_root, "Transcripts")

    args.output_folder = os.path.normpath(args.output_folder)
    os.makedirs(args.output_folder, exist_ok=True)

    if args.pyannotate_hf_token and args.audio_folder:
        log(_STAGE, "Warning: pyannotate_hf_token is set and audio_folder is provided")
        log(_STAGE, "Warning: pyannotate is only used for single audio file and will be skipped")

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
        except ValueError:
            log(
                _STAGE,
                "Failed to parse meeting start time. Expected Zoom layout like: "
                "'YYYY-MM-DD HH.MM.SS <Name>'s Zoom Meeting'. Falling back to file creation time",
            )
            meeting_start_time = None
    else:
        log(
            _STAGE,
            "Failed to extract start time from folder name. Expected Zoom layout like: "
            "'YYYY-MM-DD HH.MM.SS <Name>'s Zoom Meeting'. Falling back to file creation time",
        )
        meeting_start_time = None

    return meeting_start_time
