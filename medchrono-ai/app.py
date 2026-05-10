from __future__ import annotations

from typing import Any, TypedDict
from collections import defaultdict

import pandas as pd
from src.extraction import extract_text_from_pdf_or_image
from src.llm import get_llm, fallback_explanation
from src.parsing import extract_lab_values
from src.safety import safety_filter


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

#
# import gradio as gr
#
#
# def analyze_report(files):
#     if not files:
#         return "Please upload at least one report."
#
#     reports = [{"file_path": file.name} for file in files]
#     result = run_agent_pipeline(reports)
#
#     return result.get("final_answer", "No result generated.")
#
#
# custom_css = """
# :root {
#   --radius-lg: 22px;
# }
#
# .gradio-container {
#   background:
#     radial-gradient(circle at top left, rgba(59, 130, 246, 0.18), transparent 35%),
#     radial-gradient(circle at top right, rgba(16, 185, 129, 0.14), transparent 35%),
#     #0b1120 !important;
#   font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
# }
#
# #main-wrapper {
#   max-width: 1180px;
#   margin: 0 auto;
#   padding: 40px 20px;
# }
#
# #hero {
#   text-align: center;
#   margin-bottom: 34px;
# }
#
# #hero h1 {
#   font-size: 44px;
#   line-height: 1.1;
#   font-weight: 850;
#   letter-spacing: -0.04em;
#   margin-bottom: 12px;
#   background: linear-gradient(90deg, #ffffff, #93c5fd, #6ee7b7);
#   -webkit-background-clip: text;
#   -webkit-text-fill-color: transparent;
# }
#
# #hero p {
#   color: #cbd5e1;
#   font-size: 17px;
#   max-width: 680px;
#   margin: 0 auto;
# }
#
# .panel {
#   background: rgba(15, 23, 42, 0.78) !important;
#   border: 1px solid rgba(148, 163, 184, 0.22) !important;
#   border-radius: var(--radius-lg) !important;
#   box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35) !important;
#   backdrop-filter: blur(18px);
#   padding: 22px !important;
# }
#
# .panel-title {
#   font-size: 18px;
#   font-weight: 750;
#   color: #f8fafc;
#   margin-bottom: 6px;
# }
#
# .panel-subtitle {
#   color: #94a3b8;
#   font-size: 14px;
#   margin-bottom: 16px;
# }
#
# button.primary {
#   background: linear-gradient(135deg, #2563eb, #06b6d4) !important;
#   border: none !important;
#   border-radius: 14px !important;
#   min-height: 48px !important;
#   font-weight: 750 !important;
#   box-shadow: 0 14px 35px rgba(37, 99, 235, 0.35) !important;
# }
#
# button.secondary {
#   border-radius: 14px !important;
#   min-height: 48px !important;
#   background: rgba(148, 163, 184, 0.14) !important;
#   border: 1px solid rgba(148, 163, 184, 0.22) !important;
# }
#
# textarea {
#   border-radius: 16px !important;
#   background: rgba(2, 6, 23, 0.45) !important;
#   border: 1px solid rgba(148, 163, 184, 0.24) !important;
#   color: #e5e7eb !important;
#   font-size: 14px !important;
#   line-height: 1.6 !important;
# }
#
# label {
#   color: #e5e7eb !important;
#   font-weight: 650 !important;
# }
#
# #footer-note {
#   text-align: center;
#   color: #64748b;
#   font-size: 13px;
#   margin-top: 24px;
# }
#
# .info-card {
#   background: rgba(2, 6, 23, 0.35);
#   border: 1px solid rgba(148, 163, 184, 0.18);
#   border-radius: 18px;
#   padding: 16px;
#   color: #cbd5e1;
#   font-size: 14px;
#   line-height: 1.55;
# }
# """
#
#
# with gr.Blocks(
#     title="MedChrono AI",
#     theme=gr.themes.Soft(
#         primary_hue="blue",
#         secondary_hue="cyan",
#         neutral_hue="slate",
#     ),
#     css=custom_css,
# ) as demo:
#     with gr.Column(elem_id="main-wrapper"):
#         gr.HTML(
#             """
#             <div id="hero">
#                 <h1>MedChrono AI</h1>
#                 <p>
#                     Upload medical lab reports and get a clean, safe, easy-to-understand
#                     health trend summary with extracted values and doctor-visit questions.
#                 </p>
#             </div>
#             """
#         )
#
#         with gr.Row(equal_height=True):
#             with gr.Column(scale=1, elem_classes=["panel"]):
#                 gr.HTML(
#                     """
#                     <div class="panel-title">Upload Reports</div>
#                     <div class="panel-subtitle">
#                         Supports PDF, PNG, JPG and TXT reports.
#                     </div>
#                     """
#                 )
#
#                 file_input = gr.File(
#                     label="Choose medical reports",
#                     file_count="multiple",
#                     file_types=[".pdf", ".png", ".jpg", ".jpeg", ".txt"],
#                 )
#
#                 with gr.Row():
#                     clear_btn = gr.Button("Clear", variant="secondary")
#                     submit_btn = gr.Button("Analyze Report", variant="primary")
#
#                 gr.HTML(
#                     """
#                     <div class="info-card">
#                         <strong>Tip:</strong> For best results, upload clear reports with visible
#                         test names, values, units and reference ranges.
#                     </div>
#                     """
#                 )
#
#             with gr.Column(scale=1, elem_classes=["panel"]):
#                 gr.HTML(
#                     """
#                     <div class="panel-title">Analysis Result</div>
#                     <div class="panel-subtitle">
#                         Your extracted lab values and safe explanation will appear here.
#                     </div>
#                     """
#                 )
#
#                 result_output = gr.Markdown(
#                     label="",
#                     value="Upload a report and click **Analyze Report** to begin.",
#                 )
#
#         gr.HTML(
#             """
#             <div id="footer-note">
#                 Educational use only. Not a medical diagnosis, treatment plan, or clinical advice.
#             </div>
#             """
#         )
#
#     submit_btn.click(
#         fn=analyze_report,
#         inputs=file_input,
#         outputs=result_output,
#     )
#
#     clear_btn.click(
#         fn=lambda: (None, "Upload a report and click **Analyze Report** to begin."),
#         inputs=None,
#         outputs=[file_input, result_output],
#     )
#
#
# if __name__ == "__main__":
#     demo.launch(
#         server_name="127.0.0.1",
#         server_port=7860,
#         share=True,
#         inbrowser=True,
#     )

import gradio as gr


def analyze_report(files):
    if not files:
        return (
            "⚠️ Please upload at least one medical report.",
            "No report uploaded yet.",
            "Waiting",
        )

    reports = [{"file_path": file.name} for file in files]
    result = run_agent_pipeline(reports)

    final_answer = result.get("final_answer", "No result generated.")
    extracted_count = len(result.get("lab_values", []))
    trend_count = len(result.get("trends", []))

    status = f"✅ Analysis complete | Extracted {extracted_count} values | Found {trend_count} trends"

    return final_answer, status, "Completed"


def clear_all():
    return None, "Upload a report and click **Analyze Report** to begin.", "No report uploaded yet.", "Waiting"


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
  --bg: #020617;
  --card: rgba(15, 23, 42, 0.78);
  --card-border: rgba(148, 163, 184, 0.22);
  --text: #f8fafc;
  --muted: #94a3b8;
  --blue: #2563eb;
  --cyan: #06b6d4;
  --green: #10b981;
  --orange: #f97316;
  --red: #ef4444;
}

.gradio-container {
  background:
    radial-gradient(circle at 12% 8%, rgba(37, 99, 235, 0.22), transparent 28%),
    radial-gradient(circle at 90% 12%, rgba(6, 182, 212, 0.16), transparent 28%),
    radial-gradient(circle at 50% 100%, rgba(16, 185, 129, 0.10), transparent 35%),
    var(--bg) !important;
  font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  color: var(--text) !important;
}

#app-shell {
  max-width: 1320px;
  margin: 0 auto;
  padding: 34px 22px 42px;
}

#topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 28px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo {
  width: 46px;
  height: 46px;
  border-radius: 16px;
  display: grid;
  place-items: center;
  background: linear-gradient(135deg, var(--blue), var(--cyan));
  box-shadow: 0 16px 40px rgba(37, 99, 235, 0.38);
  font-size: 22px;
}

.brand h1 {
  font-size: 26px;
  margin: 0;
  font-weight: 900;
  letter-spacing: -0.04em;
}

.brand p {
  margin: 2px 0 0;
  color: var(--muted);
  font-size: 13px;
}

.badge {
  border: 1px solid rgba(148, 163, 184, 0.22);
  background: rgba(15, 23, 42, 0.7);
  color: #cbd5e1;
  padding: 9px 14px;
  border-radius: 999px;
  font-size: 13px;
}

#hero-card {
  position: relative;
  overflow: hidden;
  border-radius: 30px;
  padding: 34px;
  margin-bottom: 22px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  background:
    linear-gradient(135deg, rgba(37, 99, 235, 0.18), rgba(6, 182, 212, 0.08)),
    rgba(15, 23, 42, 0.65);
  box-shadow: 0 26px 90px rgba(0, 0, 0, 0.36);
}

#hero-card::after {
  content: "";
  position: absolute;
  inset: auto -80px -130px auto;
  width: 340px;
  height: 340px;
  border-radius: 50%;
  background: rgba(6, 182, 212, 0.18);
  filter: blur(20px);
}

.hero-grid {
  display: grid;
  grid-template-columns: 1.45fr 0.8fr;
  gap: 28px;
  position: relative;
  z-index: 2;
}

.hero-title {
  font-size: 48px;
  line-height: 1.04;
  font-weight: 900;
  letter-spacing: -0.055em;
  margin: 0 0 14px;
  background: linear-gradient(90deg, #fff, #bfdbfe, #67e8f9);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.hero-desc {
  max-width: 760px;
  color: #cbd5e1;
  font-size: 16px;
  line-height: 1.75;
  margin: 0;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
  margin-top: 24px;
}

.metric {
  padding: 16px;
  border-radius: 20px;
  background: rgba(2, 6, 23, 0.38);
  border: 1px solid rgba(148, 163, 184, 0.16);
}

.metric strong {
  display: block;
  font-size: 22px;
  margin-bottom: 4px;
}

.metric span {
  color: var(--muted);
  font-size: 13px;
}

.warning-box {
  border-radius: 22px;
  padding: 18px;
  background: rgba(249, 115, 22, 0.1);
  border: 1px solid rgba(249, 115, 22, 0.28);
  color: #fed7aa;
  font-size: 14px;
  line-height: 1.6;
}

.main-grid {
  display: grid;
  grid-template-columns: 0.78fr 1.22fr;
  gap: 22px;
  align-items: stretch;
}

.panel {
  background: var(--card) !important;
  border: 1px solid var(--card-border) !important;
  border-radius: 28px !important;
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.34) !important;
  backdrop-filter: blur(18px);
  padding: 22px !important;
}

.panel-head {
  display: flex;
  align-items: start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 18px;
}

.panel-title {
  font-size: 19px;
  font-weight: 850;
  color: #f8fafc;
  margin-bottom: 5px;
}

.panel-subtitle {
  color: var(--muted);
  font-size: 13px;
  line-height: 1.55;
}

.step-list {
  display: grid;
  gap: 12px;
  margin-top: 18px;
}

.step {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  padding: 14px;
  border-radius: 18px;
  background: rgba(2, 6, 23, 0.36);
  border: 1px solid rgba(148, 163, 184, 0.14);
}

.step-number {
  width: 28px;
  height: 28px;
  flex: 0 0 28px;
  border-radius: 10px;
  display: grid;
  place-items: center;
  background: rgba(37, 99, 235, 0.18);
  color: #93c5fd;
  font-weight: 800;
  font-size: 13px;
}

.step strong {
  color: #e2e8f0;
  font-size: 14px;
}

.step p {
  margin: 3px 0 0;
  color: var(--muted);
  font-size: 12.5px;
  line-height: 1.45;
}

.status-card {
  margin-top: 14px;
  border-radius: 18px;
  padding: 14px;
  background: rgba(16, 185, 129, 0.09);
  border: 1px solid rgba(16, 185, 129, 0.22);
  color: #bbf7d0;
  font-size: 13px;
}

button.primary {
  min-height: 50px !important;
  border-radius: 16px !important;
  border: none !important;
  background: linear-gradient(135deg, var(--blue), var(--cyan)) !important;
  color: white !important;
  font-weight: 850 !important;
  box-shadow: 0 18px 45px rgba(37, 99, 235, 0.35) !important;
}

button.secondary {
  min-height: 50px !important;
  border-radius: 16px !important;
  background: rgba(148, 163, 184, 0.12) !important;
  border: 1px solid rgba(148, 163, 184, 0.22) !important;
  color: #e2e8f0 !important;
  font-weight: 750 !important;
}

button:hover {
  transform: translateY(-1px);
  transition: 0.18s ease;
}

label {
  color: #e5e7eb !important;
  font-weight: 700 !important;
}

.markdown-result {
  max-height: 660px;
  overflow: auto;
  padding: 20px !important;
  border-radius: 22px !important;
  background: rgba(2, 6, 23, 0.42) !important;
  border: 1px solid rgba(148, 163, 184, 0.20) !important;
}

.markdown-result h1,
.markdown-result h2,
.markdown-result h3 {
  color: #f8fafc !important;
  letter-spacing: -0.025em;
}

.markdown-result p,
.markdown-result li {
  color: #dbeafe !important;
  line-height: 1.7;
}

.markdown-result table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 16px;
  font-size: 13px;
  overflow: hidden;
  border-radius: 14px;
}

.markdown-result th {
  background: rgba(37, 99, 235, 0.22);
  color: #bfdbfe;
  padding: 10px;
  border: 1px solid rgba(148, 163, 184, 0.18);
}

.markdown-result td {
  padding: 10px;
  color: #e5e7eb;
  border: 1px solid rgba(148, 163, 184, 0.14);
}

.footer {
  text-align: center;
  color: #64748b;
  font-size: 12.5px;
  margin-top: 24px;
  line-height: 1.6;
}

.upload-box {
  border-radius: 20px !important;
}

@media (max-width: 900px) {
  .hero-grid,
  .main-grid,
  .metric-grid {
    grid-template-columns: 1fr;
  }

  .hero-title {
    font-size: 36px;
  }

  #topbar {
    align-items: flex-start;
    gap: 14px;
    flex-direction: column;
  }
}
"""


with gr.Blocks(
    title="MedChrono AI",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
    ),
    css=custom_css,
) as demo:
    with gr.Column(elem_id="app-shell"):
        gr.HTML(
            """
            <div id="topbar">
              <div class="brand">
                <div class="logo">🩺</div>
                <div>
                  <h1>MedChrono AI</h1>
                  <p>Medical report extraction, trend review and safe patient-friendly summaries</p>
                </div>
              </div>
              <div class="badge">Educational Assistant • Not Medical Advice</div>
            </div>

            <section id="hero-card">
              <div class="hero-grid">
                <div>
                  <h2 class="hero-title">Understand your lab reports with clarity.</h2>
                  <p class="hero-desc">
                    Upload PDF, image, or text-based medical reports. MedChrono AI extracts lab values,
                    compares them with reference ranges, highlights values to review, and prepares
                    safe questions for your doctor.
                  </p>

                  <div class="metric-grid">
                    <div class="metric">
                      <strong>PDF</strong>
                      <span>Report support</span>
                    </div>
                    <div class="metric">
                      <strong>OCR</strong>
                      <span>Image reading</span>
                    </div>
                    <div class="metric">
                      <strong>Safe</strong>
                      <span>No diagnosis claims</span>
                    </div>
                  </div>
                </div>

                <div class="warning-box">
                  <strong>Important safety note</strong><br>
                  This tool helps explain report values in simple language. It should not be used
                  as a diagnosis, prescription, treatment plan, or replacement for a licensed clinician.
                </div>
              </div>
            </section>
            """
        )

        with gr.Row(elem_classes=["main-grid"]):
            with gr.Column(elem_classes=["panel"]):
                gr.HTML(
                    """
                    <div class="panel-head">
                      <div>
                        <div class="panel-title">Upload Medical Reports</div>
                        <div class="panel-subtitle">
                          Add one or multiple files. For trend analysis, upload reports from different dates.
                        </div>
                      </div>
                    </div>
                    """
                )

                file_input = gr.File(
                    label="Choose PDF, image, or text reports",
                    file_count="multiple",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".txt"],
                    elem_classes=["upload-box"],
                )

                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    submit_btn = gr.Button("Analyze Reports", variant="primary")

                status_output = gr.Textbox(
                    label="Analysis Status",
                    value="No report uploaded yet.",
                    interactive=False,
                    lines=2,
                )

                pipeline_state = gr.Textbox(
                    label="Pipeline State",
                    value="Waiting",
                    interactive=False,
                    lines=1,
                )

                gr.HTML(
                    """
                    <div class="step-list">
                      <div class="step">
                        <div class="step-number">1</div>
                        <div>
                          <strong>Upload</strong>
                          <p>Upload clear lab reports with values, units, and reference ranges.</p>
                        </div>
                      </div>

                      <div class="step">
                        <div class="step-number">2</div>
                        <div>
                          <strong>Extract</strong>
                          <p>The app extracts text from PDFs/images and identifies lab values.</p>
                        </div>
                      </div>

                      <div class="step">
                        <div class="step-number">3</div>
                        <div>
                          <strong>Review</strong>
                          <p>Results are summarized with abnormal values and doctor questions.</p>
                        </div>
                      </div>
                    </div>
                    """
                )

            with gr.Column(elem_classes=["panel"]):
                gr.HTML(
                    """
                    <div class="panel-head">
                      <div>
                        <div class="panel-title">Analysis Dashboard</div>
                        <div class="panel-subtitle">
                          Extracted values, trend summary, abnormal results, and safe explanation.
                        </div>
                      </div>
                    </div>
                    """
                )

                result_output = gr.Markdown(
                    value="Upload a report and click **Analyze Reports** to begin.",
                    elem_classes=["markdown-result"],
                )

        gr.HTML(
            """
            <div class="footer">
              MedChrono AI is designed for educational report understanding only.
              Always consult a licensed clinician for diagnosis, treatment, or urgent concerns.
            </div>
            """
        )

    submit_btn.click(
        fn=analyze_report,
        inputs=file_input,
        outputs=[result_output, status_output, pipeline_state],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[file_input, result_output, status_output, pipeline_state],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        inbrowser=True,
    )