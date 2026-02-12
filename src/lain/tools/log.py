"""Simple stage-based logging for the pipeline."""


def log(stage: str, message: str) -> None:
    """Print a formatted log message with a stage tag."""
    print(f"[{stage}] {message}")
