from __future__ import annotations

from typing import Any, TypedDict
from collections import defaultdict

import pandas as pd
from .extraction import extract_text_from_pdf_or_image  # Import the updated hybrid extraction method
from .llm import get_llm, fallback_explanation
from .safety import safety_filter


class AgentState(TypedDict, total=False):
    reports: list[dict[str, Any]]
    lab_values: list[dict[str, Any]]
    trends: list[dict[str, Any]]
    table_markdown: str
    explanation: str
    final_answer: str


def extractor_node(state: AgentState) -> AgentState:
    """
    Extract lab values from the reports using hybrid text and OCR extraction.
    """
    reports = state.get("reports", [])
    lab_values = []

    # Process each report to extract text (hybrid method)
    for report in reports:
        # Assuming each report contains a 'file_path' key for the uploaded file
        report_text = extract_text_from_pdf_or_image(report.get("file_path", ""))
        extracted_values = extract_lab_values(report_text)  # Extract lab values from the text
        lab_values.extend(extracted_values)

    state["lab_values"] = lab_values
    return state


def trend_node(state: AgentState) -> AgentState:
    lab_values = state.get("lab_values", [])
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in lab_values:
        grouped[row["test_name"].lower()].append(row)

    trends: list[dict[str, Any]] = []
    for _, rows in grouped.items():
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda x: str(x.get("report_date", "")))
        first = rows[0]
        last = rows[-1]
        change = float(last["value"]) - float(first["value"])
        trends.append(
            {
                "test_name": last["test_name"],
                "first_date": first["report_date"],
                "last_date": last["report_date"],
                "first_value": first["value"],
                "last_value": last["value"],
                "change": round(change, 3),
                "unit": last.get("unit", ""),
                "latest_status": last.get("status", "unknown"),
            }
        )
    state["trends"] = trends
    return state


def table_node(state: AgentState) -> AgentState:
    lab_values = state.get("lab_values", [])
    if not lab_values:
        state["table_markdown"] = "No lab values extracted."
        return state
    df = pd.DataFrame(lab_values)
    columns = ["report_date", "test_name", "value", "unit", "ref_low", "ref_high", "status"]
    df = df[[col for col in columns if col in df.columns]]
    state["table_markdown"] = df.to_markdown(index=False)
    return state


def explanation_node(state: AgentState) -> AgentState:
    llm = get_llm()
    lab_values = state.get("lab_values", [])
    trends = state.get("trends", [])

    if llm:
        prompt = f"""
You are MedChrono AI, a safe medical report explanation assistant.
Explain the following lab values and trends in simple language.
Do not diagnose, prescribe, or tell the user to start/stop medication.
Focus on report understanding and doctor-visit preparation.

Lab values:
{lab_values}

Trends:
{trends}

Return sections:
1. Overall summary
2. Important changes over time
3. Values to review
4. Questions to ask a doctor
5. Safety note
"""
        try:
            response = llm.invoke(prompt)
            state["explanation"] = getattr(response, "content", str(response))
        except Exception:
            state["explanation"] = fallback_explanation(lab_values, trends)
    else:
        state["explanation"] = fallback_explanation(lab_values, trends)
    return state


def safety_node(state: AgentState) -> AgentState:
    explanation = safety_filter(state.get("explanation", ""))
    state["final_answer"] = f"{explanation}\n\n## Extracted Values\n\n{state.get('table_markdown', '')}"
    return state


def run_agent_pipeline(reports: list[dict[str, Any]]) -> AgentState:
    """Run with LangGraph if available; otherwise use deterministic sequence."""
    initial_state: AgentState = {"reports": reports}

    try:
        from langgraph.graph import StateGraph, END

        graph = StateGraph(AgentState)
        graph.add_node("extractor", extractor_node)
        graph.add_node("trend", trend_node)
        graph.add_node("table", table_node)
        graph.add_node("explain", explanation_node)
        graph.add_node("safety", safety_node)

        graph.set_entry_point("extractor")
        graph.add_edge("extractor", "trend")
        graph.add_edge("trend", "table")
        graph.add_edge("table", "explain")
        graph.add_edge("explain", "safety")
        graph.add_edge("safety", END)

        app = graph.compile()
        return app.invoke(initial_state)
    except Exception:
        state = extractor_node(initial_state)
        state = trend_node(state)
        state = table_node(state)
        state = explanation_node(state)
        state = safety_node(state)
        return state