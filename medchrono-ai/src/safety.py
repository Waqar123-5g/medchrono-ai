from __future__ import annotations

SAFETY_NOTE = (
    "This explanation is for educational purposes only. It is not a diagnosis, "
    "treatment plan, or medical advice. Please review these results with a licensed clinician."
)

BLOCKED_PHRASES = [
    "you have ",
    "you definitely have",
    "start taking",
    "stop taking",
    "increase your dose",
    "decrease your dose",
    "you should take",
]


def safety_filter(text: str) -> str:
    safe_text = text
    for phrase in BLOCKED_PHRASES:
        safe_text = safe_text.replace(phrase, "you may want to ask your doctor about ")
    if SAFETY_NOTE not in safe_text:
        safe_text = f"{safe_text.strip()}\n\n**Safety note:** {SAFETY_NOTE}"
    return safe_text
