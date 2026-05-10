# MedChrono AI

**MedChrono AI** is a quick AMD Developer Hackathon prototype for a multi-report health trend explanation agent.

Instead of only summarizing one medical report, MedChrono AI compares multiple lab reports over time, extracts key values, detects trends, explains them in simple language, and generates doctor-visit questions.

> Safety: This project is for educational/report-understanding purposes only. It does not diagnose, treat, prescribe, or replace a licensed clinician.

## Why this fits the AMD Developer Hackathon

This prototype is designed around the hackathon-friendly stack:

- **AI Agents / Agentic Workflows**: LangGraph-style multi-step agent pipeline
- **Vision & Multimodal AI**: PDF/image/text medical report ingestion
- **Hugging Face / Qwen / Open-source models**: Works with OpenAI-compatible vLLM endpoints, including AMD Developer Cloud deployments
- **AMD ROCm / MI300X-ready**: Can connect to a vLLM server running on AMD Developer Cloud
- **Gradio**: Fast deployable demo UI for Hugging Face Spaces

## Core user flow

1. Upload one or more medical reports as PDF, image, or text.
2. The app extracts text.
3. The extraction agent finds lab test values, units, and reference ranges.
4. The trend agent compares values over time.
5. The explanation agent produces a plain-English summary.
6. The safety agent keeps the output educational and non-diagnostic.
7. The question agent generates doctor-visit questions.

## Demo idea

Upload two or three reports with values like:

```text
Report Date: 2025-01-10
Hemoglobin: 10.5 g/dL Reference Range: 13.5 - 17.5
Vitamin D: 14 ng/mL Reference Range: 30 - 100
HbA1c: 5.6 % Reference Range: 4.0 - 5.6
```

Then upload another report:

```text
Report Date: 2026-05-10
Hemoglobin: 12.1 g/dL Reference Range: 13.5 - 17.5
Vitamin D: 28 ng/mL Reference Range: 30 - 100
HbA1c: 6.1 % Reference Range: 4.0 - 5.6
```

MedChrono AI will explain what improved, what worsened, and what questions the patient should ask a doctor.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL in your browser.

## Optional: Connect to AMD Developer Cloud vLLM endpoint

If you deploy Qwen/Llama/Mistral on AMD Developer Cloud using vLLM, expose an OpenAI-compatible API endpoint and set:

```bash
export OPENAI_API_BASE="http://YOUR_AMD_SERVER:8000/v1"
export OPENAI_API_KEY="EMPTY"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
```

The app will use that endpoint for better explanations. Without it, the prototype still works using a local rule-based fallback.

## Hugging Face Spaces deployment

Create a new Gradio Space and upload:

```text
app.py
requirements.txt
src/
sample_data/
README.md
```

For CPU demo, keep the fallback mode. For stronger demo, point environment variables to your AMD Developer Cloud vLLM endpoint.

## Repo structure

```text
medchrono-ai/
  app.py
  requirements.txt
  README.md
  src/
    agents.py
    extraction.py
    llm.py
    parsing.py
    safety.py
  sample_data/
    report_jan_2025.txt
    report_may_2026.txt
  docs/
    pitch.md
```

## Suggested hackathon pitch

MedChrono AI helps patients understand how their health metrics change over time by comparing multiple medical reports, detecting meaningful trends, and generating doctor-ready questions using an agentic multimodal AI workflow.

## Limitations

- This is not a medical device.
- OCR quality depends on the uploaded image/PDF quality.
- Reference ranges vary by lab, gender, age, and clinical context.
- The app should always encourage review by a licensed clinician.
