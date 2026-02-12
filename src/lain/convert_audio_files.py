import os
import re
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from lain.tools.log import log

_STAGE = "Audio"


def convert_audio_files(meeting_folder_path: str) -> None:

    log(_STAGE, f"Analyzing folder: {meeting_folder_path}")
    if not os.path.isdir(meeting_folder_path):
        raise NotADirectoryError(
            f"{meeting_folder_path} is not a valid directory."
        )

    # Step into first folder and collect audio files that end in .m4a
    master_audio_m4a = [
        os.path.join(meeting_folder_path, f)
        for f in os.listdir(meeting_folder_path)
        if f.lower().endswith((".m4a"))
    ]

    master_audio_wav = [f.replace(".m4a", ".wav") for f in master_audio_m4a]
    missing_wav_files = [f for f in master_audio_wav if not os.path.exists(f)]
    if missing_wav_files:
        log(_STAGE, "Converting master audio files to .wav")
        for f in missing_wav_files:
            log(_STAGE, f"  Converting: {f}")
            subprocess.run(
                [
                    "ffmpeg", "-i", f.replace(".wav", ".m4a"),
                    "-ac", "1", "-ar", "16000", f,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
    else:
        log(_STAGE, "All master audio files already converted to .wav")

    # Step into sub directory called "Audio Record" and collect audio files that end in .m4a
    audio_record_path = os.path.join(meeting_folder_path, "Audio Record")
    if not os.path.isdir(audio_record_path):
        raise NotADirectoryError(
            f"{audio_record_path} is not a valid directory. Currently must have an audio record"
        )

    audio_files = [
        os.path.join(audio_record_path, f)
        for f in os.listdir(audio_record_path)
        if f.lower().endswith((".m4a"))
    ]
    if not audio_files:
        raise FileNotFoundError(
            "No audio files found in the Audio Record folder."
        )

    missing_wav_files = [
        f for f in audio_files if not os.path.exists(f.replace(".m4a", ".wav"))
    ]

    if missing_wav_files:
        log(_STAGE, "Converting speaker audio files to .wav")
        for f in missing_wav_files:
            log(_STAGE, f"  Converting: {f.replace('.m4a', '.wav')}")
            subprocess.run(
                [
                    "ffmpeg", "-i", f,
                    "-ac", "1", "-ar", "16000", f.replace(".m4a", ".wav"),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
    else:
        log(_STAGE, "All speaker audio files already converted to .wav")


def get_creation_time(file_path: str) -> datetime | None:
    """
    Extract the creation time of an audio file using ffmpeg.

    Parameters
    ----------
    file_path : str
        Path to the audio file.

    Returns
    -------
    datetime or None
        The creation time in ET, or None if not found.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format_tags=creation_time",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,  # Ensure subprocess raises an error on failure
    )
    creation_time_str = result.stdout.strip()
    if not creation_time_str:
        log(_STAGE, f"Warning: Creation time not found for file: {file_path}")
        return None

    # Convert to datetime and adjust to ET
    creation_time = datetime.fromisoformat(
        creation_time_str.replace("Z", "+00:00")
    )
    return creation_time - timedelta(hours=4)  # Convert UTC to ET


def convert_to_wav(file_path: str, output_folder: str) -> str:
    """
    Convert an audio file to .wav format using ffmpeg.

    Parameters
    ----------
    file_path : str
        Path to the audio file.
    output_folder : str
        Path to the folder where the .wav file will be saved.

    Returns
    -------
    str
        Path to the created .wav file.
    """
    wav_file = (
        os.path.basename(file_path)
        .replace(".m4a", ".wav")
        .replace(".mp3", ".wav")
    )
    output_path = os.path.join(output_folder, wav_file)
    subprocess.run(
        ["ffmpeg", "-i", file_path, output_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    log(_STAGE, f"Converted {file_path} to {output_path}")
    return output_path


def extract_speaker_name(file_name: str) -> str:
    """
    Extract the speaker name from the file name based on the Zoom naming pattern.

    Parameters
    ----------
    file_name : str
        Name of the audio file.

    Returns
    -------
    str
        Extracted speaker name.

    Raises
    ------
    ValueError
        If the file name does not match the expected pattern.
    """
    base_name = os.path.basename(file_name)
    # Work on the filename without extension
    name_no_ext = os.path.splitext(base_name)[0]

    # Pattern: the literal "audio" then the speaker name, then 11 trailing digits (no underscores between sections)
    # Example: "audiospeaker120230101"
    m = re.match(
        r"(?i)audio(?P<name>.+?)(?P<recording>\d)(?P<duplicate>\d)(?P<magic>\d{9})$",
        name_no_ext,
    )
    if m:
        speaker = m.group("name").strip()
        return speaker

    else:
        # NOTE: If this becomes annoying consider making other naming patterns, or a fallback of some sort.
        raise ValueError(
            f"Could not extract speaker name from file: {file_name} \n Expected pattern like 'audiospeaker01234567891.m4a'"
        )


def check_converted_files(expected_files: list) -> list[str]:
    """
    Check if all expected converted audio files exist.

    Parameters
    ----------
    expected_files : list of str
        List of expected .wav file paths.

    Returns
    -------
    list of str
        List of missing .wav files.
    """
    missing_files = [f for f in expected_files if not os.path.exists(f)]
    if missing_files:
        log(_STAGE, "Missing converted files:")
        for f in missing_files:
            log(_STAGE, f"  {f}")
    return missing_files


def get_unconverted_audio_files(
    audio_files: list, converted_folder: str
) -> list[str]:
    """
    Get a list of audio files that have not been converted.

    Parameters
    ----------
    audio_files : list of str
        List of audio file paths.
    converted_folder : str
        Path to the folder containing the converted audio files.

    Returns
    -------
    list of str
        List of unconverted audio file paths.
    """
    # Create a set of all converted file names (without paths)
    converted_files = {
        os.path.basename(f)
        for f in os.listdir(converted_folder)
        if f.lower().endswith(".wav")
    }

    # Find all audio files that are not in the converted set
    unconverted_files = [
        f
        for f in audio_files
        if os.path.basename(f).replace(".m4a", ".wav").replace(".mp3", ".wav")
        not in converted_files
    ]

    if unconverted_files:
        log(_STAGE, f"Unconverted audio files in {os.path.dirname(audio_files[0])}:")
        for f in unconverted_files:
            log(_STAGE, f"  {os.path.basename(f)}")
    else:
        log(_STAGE, "No unconverted audio files found")

    return unconverted_files


def align_audio_file_offsets(
    wav_files: list[str], master_audio_wav: str
) -> dict[str, float]:
    """
    Compute time offsets for each speaker file relative to a master audio track.

    Parameters
    ----------
    wav_files : dict of str to str
        Mapping from speaker labels to their WAV file paths.
    master_audio_wav : str
        Path to the master audio WAV file.

    Returns
    -------
    dict of str to float
        Mapping from speaker labels to offset times (in seconds) relative to master audio.
    """
    import librosa
    from scipy.signal import correlate

    log(_STAGE, "Calculating audio file offsets...")
    offsets = {}
    for speaker, file in wav_files.items():
        y1, sr1 = librosa.load(master_audio_wav, sr=None)
        y2, sr2 = librosa.load(file, sr=None)

        # Resample if needed
        if sr1 != sr2:
            y2 = librosa.resample(y2, sr2, sr1)
            sr2 = sr1

        # Cross-correlation
        corr = correlate(y1, y2, mode="full")
        lag = np.argmax(corr) - (len(y2) - 1)
        offset_seconds = lag / sr1
        offsets[speaker] = offset_seconds

    return offsets


def gather_wave_files(meeting_folder_path: str) -> list[str]:
    """Collect all WAV files in the meeting folder subdirectory "Audio Record" only"""
    wave_list = []
    audio_record_folder = os.path.join(meeting_folder_path, "Audio Record")
    for file in os.listdir(audio_record_folder):
        if file.lower().endswith(".wav"):
            wave_list.append(os.path.join(audio_record_folder, file))

    m4a_list = []
    for file in os.listdir(audio_record_folder):
        if file.lower().endswith(".m4a"):
            m4a_list.append(os.path.join(audio_record_folder, file))

    if len(m4a_list) != len(wave_list):
        log(
            _STAGE,
            "Warning: .m4a count does not match .wav count in Audio Record. Something may have gone wrong",
        )

    return wave_list


def get_recordings_dict(wave_files: list) -> bool:
    """Check folder for split recordings"""
    # Pattern is the literal 'audio' followed by '<name>' '<recording number(single digit)> '<duplicate number(single digit)>' '<9 digit magic number>'
    # The name can be numbers letters and dots
    pattern = r"(?i)audio(?P<name>.+?)(?P<recording>\d)(?P<duplicate>\d)(?P<magic>\d{9})\.wav"
    pattern = re.compile(pattern)

    wav_dict = {}

    # This might work, but sorting by the duplicate number
    wave_files.sort()  # Sort to ensure consistent order

    for file in wave_files:
        base_name = os.path.basename(file)
        match = pattern.match(base_name)
        if match:
            name = match.group("name")
            duplicate = int(match.group("duplicate"))

            if name not in wav_dict:
                wav_dict[name] = {duplicate: file}
            else:
                wav_dict[name][duplicate] = file

        else:
            raise ValueError(
                f"Could not extract speaker name and/or duplicate from file: {file} \n Expected pattern like 'audiospeaker01234567891.wav'"
            )

    return wav_dict


def combine_audio_files(wav_list: list) -> dict:
    # Check if split recordings exist
    wav_dict = get_recordings_dict(wav_list)

    # If wav_dict has only single entries per name and dpulcate, return original list
    if all(len(duplicates) == 1 for duplicates in wav_dict.values()):
        log(_STAGE, "No split recordings detected, using original audio files")
        return {
            name: files[next(iter(files))] for name, files in wav_dict.items()
        }

    else:
        log(_STAGE, "Split recordings detected, combining audio files...")
        combine_folder = os.path.join(os.path.dirname(wav_list[0]), "Combined")
        os.makedirs(combine_folder, exist_ok=True)

        wav_dict_new = {}

        for name, files in wav_dict.items():
            # Combine the audio files for this speaker
            combined_file_path = os.path.join(
                combine_folder, f"audio{name}_combined.wav"
            )
            wav_dict_new[name] = combined_file_path
            if not os.path.exists(combined_file_path):
                log(_STAGE, f"Combining audio files for speaker: {name}")
                files_list = [
                    files[k] for k in sorted(files.keys())
                ]  # Sort by duplicate number
                concat_wavs_copy(files_list, combined_file_path)
            else:
                log(_STAGE, f"Combined file already exists, skipping: {combined_file_path}")

        return wav_dict_new


def concat_wavs_copy(wavs: list[str | Path], out_path: str | Path) -> Path:
    wavs = [Path(p).resolve() for p in wavs]
    out_path = Path(out_path)

    if len(wavs) == 1:
        # simply copy the file
        subprocess.run(["cp", str(wavs[0]), str(out_path)], check=True)

    # write the concat list in order
    tf = tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8", newline="\n"
    )
    try:
        for p in wavs:
            tf.write(f"file '{p.as_posix()}'\n")
        tf.flush()
        tf.close()

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                tf.name,
                "-c",
                "copy",
                str(out_path),
            ],
            check=True,
        )
    finally:
        try:
            os.unlink(tf.name)
        except OSError:
            pass
