import argparse
import sys

import torch


def load_whisper(model_size: str) -> str:
    try:
        import whisper
    except ImportError as e:
        return f"Whisper not installed: {e}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(model_size)

    model.to(device)

    return f"Whisper '{model_size}' loaded on device: {device}"


def load_whisperx(model_size: str) -> str:
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = (
        "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
    )

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    return f"WhisperX '{model_size}' loaded on device: {device} with compute_type: {compute_type}"


def load_text_gen(
    model_id: str, test_prompt: str = "Say hello."
) -> tuple[str, str]:
    try:
        from transformers import pipeline
    except Exception as e:
        return (f"Transformers not available: {e}", "")

    try:
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        out = pipe(test_prompt, max_new_tokens=24)
        text = out[0].get("generated_text", "")
        return (f"Text model '{model_id}' loaded (device_map=auto)", text)
    except Exception as e:
        return (f"Failed to load text model '{model_id}': {e}", "")


def load_diarizer(hf_token: str | None) -> str:
    if not hf_token:
        return "Skipping diarizer (no HF token provided)"
    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        return f"pyannote.audio not available: {e}"
    try:
        _ = Pipeline.from_pretrained(
            "pyannote/speaker-diarization", use_auth_token=hf_token
        )
        return "Diarization pipeline loaded"
    except Exception as e:
        return f"Failed to load diarization pipeline: {e}"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Model smoke loader: load local models and print a hello"
    )
    parser.add_argument(
        "-g",
        "--gpt-model",
        default=None,
        help="Transformers model id for text-generation (e.g., openai/gpt-oss-20b)",
    )
    parser.add_argument(
        "-w",
        "--whisper-model",
        default=None,
        help="Whisper model size (e.g., tiny, base, small, medium, large, turbo)",
    )
    parser.add_argument(
        "-x",
        "--whisperx-model",
        default=None,
        help="WhisperX model size (e.g., tiny, base, small, medium, large, turbo)",
    )
    parser.add_argument(
        "-t",
        "--pyannote-hf-token",
        default=None,
        help="Optional HF token to attempt loading pyannote diarization",
    )
    args = parser.parse_args(argv)

    # Require at least one model to be specified
    if not (args.gpt_model or args.whisper_model or args.pyannote_hf_token):
        parser.error(
            "Specify at least one of --gpt-model, --whisper-model, or --pyannote-hf-token to load."
        )

    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("CUDA device:", torch.cuda.get_device_name(0))
        except Exception:
            pass

    # Load Whisper if requested
    if args.whisper_model:
        print(load_whisper(args.whisper_model))

    # Load WhisperX if requested
    if args.whisperx_model:
        print(load_whisperx(args.whisperx_model))

    # Load text-generation model and generate a short hello
    if args.gpt_model:
        status, hello = load_text_gen(
            args.gpt_model, test_prompt="Say 'hello' briefly."
        )
        print(status)
        if hello:
            print("Model output:")
            print(hello)

    # Optionally try diarizer
    if args.pyannote_hf_token:
        print(load_diarizer(args.pyannote_hf_token))


if __name__ == "__main__":
    sys.exit(main())
