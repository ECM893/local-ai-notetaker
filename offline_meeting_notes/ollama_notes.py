from typing import Literal
import json
from ollama import generate

SYS_PROMPT_JSON = """
You are a meeting notes structurer. Read the transcript and return ONLY JSON that matches this schema (no extra text):
{
    "header": {
        "date": "YYYY-MM-DD",
        "time": "HH:MM - HH:MM ET",
        "attendees": ["Name", "Name"],
        "subject": "Short subject"
    },
    "topics": [
        {
            "title": "Topic title",
            "time_range": "HH:MM:SS - HH:MM:SS",
            "bullets": ["Who: point", "Group: point"],
            "conclusion": "One-sentence conclusion"
        }
    ],
    "action_items": [
        {
            "owner": "Speaker name",
            "items": [
                {"description": "What to do", "deadline": "YYYY-MM-DD or null"}
            ]
        }
    ],
    "metanotes": ["Optional note", "..."]
}

Rules:
- Use only information present in the transcript; do not invent details
- Exclude side/personal conversations and off-topic content
- If a field is unknown, use null or [] appropriately
- Return valid JSON only (no Markdown, no prose)
"""

USER_PROMPT_JSON = """Summarize the meeting transcript into the specified JSON schema.
Return JSON only (no Markdown, no extra text).

Transcript:

{transcript}
"""


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

    prompt = USER_PROMPT_JSON.format(transcript=transcript)

    approx_tokens = int((len(prompt) + len(SYS_PROMPT_JSON)) // 2.5)  # rough over-estimate of tokens
    print(f"Approximate tokens: {approx_tokens}")
    if approx_tokens > 128000:
        raise ValueError(
            f"Transcript is too long ({approx_tokens} tokens). Please shorten the transcript."
        )

    # TODO: This is setup only for thinking models, should generalize inputs for other smaller models too
    # NOTE: Think parameter hasn't been give updated type hints in ollama package as of 2025-09-18
    response = generate(
        model=model,
        prompt=prompt,
        system=SYS_PROMPT_JSON,
        think=think, # type: ignore
        options={"num_ctx": approx_tokens},
    ) # type: ignore

    # If save thought process is enabled, print out to file for debugging
    if save_thought_process:
        with open(transcript_path.replace(".txt", "_thought_process.txt"), "w", encoding="utf-8") as f:
            f.write(response.thinking or "No thought process returned.")

    print(f"Response time: {(response.total_duration)/1e9/60:.2f} minutes")
    print(f"Actual Input tokens: {response.prompt_eval_count}")
    if approx_tokens <= response.prompt_eval_count:
        print("Warning: Approximate tokens was less than or equal to actual input tokens.")
        print("Consider Adjusting the Approximate tokens calculation.")
    print(f"Output tokens: {response.eval_count}")

    resp_raw = response['response']
    resp_json = json.loads(resp_raw)

    resp_md = notes_json_to_markdown(resp_json)

    # This should always return a string in Markdown format
    return resp_md
