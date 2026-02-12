import json
import os
import re
from typing import Literal

from ollama import generate

from lain.tools.log import log

_STAGE = "Notes"
_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def _read_prompt(filename: str) -> str:
    """Read a prompt template from the prompts/ directory."""
    path = os.path.join(_PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def notes_json_to_markdown(data: dict) -> str:
    """Render structured notes JSON into the required Markdown layout.

    Expected keys: header, topics, action_items, metanotes.
    Missing keys are handled gracefully.
    """
    header = data.get("header", {}) or {}
    topics = data.get("topics", []) or []
    action_items = data.get("action_items", []) or []
    metanotes = data.get("metanotes", []) or []

    lines: list[str] = []
    lines.append("# Meeting Notes")
    if header:
        if header.get("date"):
            lines.append(f"**Date:** {header['date']}")
        if header.get("time"):
            lines.append(f"**Time:** {header['time']}")
        if header.get("attendees"):
            lines.append("**Attendees:**")
            for a in header.get("attendees", []) or []:
                lines.append(f"- {a}")
        if header.get("subject"):
            lines.append("")
            lines.append(f"**Subject:** {header['subject']}")

    if topics:
        lines.append("")
        lines.append("---")
        for idx, t in enumerate(topics, 1):
            title = t.get("title") or f"Topic {idx}"
            tr = t.get("time_range")
            heading = f"## {idx}. {title}" + (f" ({tr})" if tr else "")
            lines.append("")
            lines.append(heading)
            for b in t.get("bullets", []) or []:
                lines.append(f"- {b}")
            concl = t.get("conclusion")
            if concl:
                lines.append("")
                lines.append(f"**Conclusion:** {concl}")

    # Action items grouped by owner
    if action_items:
        lines.append("")
        lines.append("## Action Items")
        for grp in action_items:
            owner = grp.get("owner") or "Unassigned"
            items = grp.get("items", []) or []
            lines.append(f"- **{owner}**")
            for it in items:
                desc = it.get("description") or ""
                deadline = it.get("deadline")
                if deadline:
                    lines.append(f"  - {desc} (due {deadline})")
                else:
                    lines.append(f"  - {desc}")

    if metanotes:
        lines.append("")
        lines.append("## Metanotes")
        for m in metanotes:
            lines.append(f"- {m}")

    return "\n".join(lines)


def ollama_api_notes(
    transcript_path: str,
    model: str,
    think: Literal["low", "medium", "high"] | bool = "high",
    save_thought_process: bool = True,
) -> str:

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    system_prompt = _read_prompt("system_prompt.txt")
    user_prompt_template = _read_prompt("user_prompt.txt")
    prompt = user_prompt_template.format(transcript=transcript)

    approx_tokens = int(
        (len(prompt) + len(system_prompt)) // 2.5
    )  # rough over-estimate of tokens
    log(_STAGE, f"Approximate tokens: {approx_tokens}")
    if approx_tokens > 128000:
        raise ValueError(
            f"Transcript is too long ({approx_tokens} tokens). Please shorten the transcript."
        )

    # TODO: This is setup only for thinking models, should generalize inputs for other smaller models too
    # NOTE: Think parameter hasn't been give updated type hints in ollama package as of 2025-09-18
    # num_ctx must cover input + thinking + output; use 4x input as a safe minimum
    num_ctx = max(approx_tokens * 4, 8192)
    response = generate(
        model=model,
        prompt=prompt,
        system=system_prompt,
        think=think,  # type: ignore
        options={"num_ctx": num_ctx},
    )  # type: ignore

    # If save thought process is enabled, print out to file for debugging
    if save_thought_process:
        with open(
            transcript_path.replace(".txt", "_thought_process.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(response.thinking or "No thought process returned.")

    if response.total_duration:
        log(_STAGE, f"Response time: {response.total_duration / 1e9 / 60:.2f} minutes")
    if response.prompt_eval_count:
        log(_STAGE, f"Actual input tokens: {response.prompt_eval_count}")
        if approx_tokens <= response.prompt_eval_count:
            log(_STAGE, "Warning: Approximate tokens was less than or equal to actual input tokens")
            log(_STAGE, "Consider adjusting the approximate tokens calculation")
    if response.eval_count:
        log(_STAGE, f"Output tokens: {response.eval_count}")

    resp_raw = response["response"]

    # Try to extract JSON from the response, handling common LLM output quirks
    resp_json = _extract_json(resp_raw)

    # If the response was empty, the model may have put JSON in the thinking block
    if resp_json is None and response.thinking:
        resp_json = _extract_json(response.thinking)

    if resp_json is None:
        raise ValueError(
            f"Could not parse JSON from Ollama response. Raw response:\n{resp_raw!r}"
        )

    resp_md = notes_json_to_markdown(resp_json)

    # This should always return a string in Markdown format
    return resp_md


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from text, handling markdown fences and surrounding prose."""
    if not text or not text.strip():
        return None

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try parsing the full text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object ({...}) in the text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None
