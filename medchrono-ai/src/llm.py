from __future__ import annotations

import os
from typing import Any


def get_llm():
    """Return an OpenAI-compatible chat model if endpoint env vars exist.

    This works with AMD Developer Cloud when you run vLLM as an OpenAI-compatible
    endpoint. If not configured, the app uses a rule-based fallback.
    """
    api_base = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    if not api_base:
        return None

    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=api_base,
            temperature=0.2,
        )
    except Exception:
        return None


def fallback_explanation(lab_values: list[dict[str, Any]], trends: list[dict[str, Any]]) -> str:
    if not lab_values:
        return (
            "I could not confidently extract lab values from the uploaded file. "
            "Try uploading a clearer PDF/image or paste the report text directly."
        )

    abnormal = [v for v in lab_values if v.get("status") in {"low", "high"}]
    lines = ["## Health Trend Summary", ""]

    if trends:
        lines.append("### Important changes over time")
        for trend in trends:
            direction = "increased" if trend["change"] > 0 else "decreased"
            lines.append(
                f"- **{trend['test_name']}** {direction} from {trend['first_value']} to "
                f"{trend['last_value']} {trend.get('unit','')}."
            )
        lines.append("")

    lines.append("### Values to review")
    if abnormal:
        for value in abnormal[:10]:
            lines.append(
                f"- **{value['test_name']}** is **{value['status']}**: {value['value']} "
                f"{value.get('unit','')} compared with reference range "
                f"{value.get('ref_low')} - {value.get('ref_high')}."
            )
    else:
        lines.append("- No clearly abnormal values were detected from the extracted data.")

    lines.extend(
        [
            "",
            "### Questions to ask your doctor",
            "- Which of these changes are clinically important in my situation?",
            "- Should any of these tests be repeated?",
            "- Are additional tests needed to understand the cause of abnormal values?",
            "- Do medications, diet, hydration, or recent illness affect these results?",
            "- What should I monitor before my next appointment?",
        ]
    )
    return "\n".join(lines)
