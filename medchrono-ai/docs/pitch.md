# MedChrono AI Pitch

## One-line pitch

MedChrono AI helps patients understand how their health metrics change over time by comparing multiple medical reports and generating doctor-ready questions using an agentic multimodal AI workflow.

## Problem

Patients often receive lab reports filled with technical terms, abnormal flags, and reference ranges they do not understand. Most tools summarize one report at a time, but patients and doctors care about how values change over time.

## Solution

MedChrono AI ingests multiple reports, extracts lab values, compares trends, explains changes in plain language, and prepares questions for a doctor visit.

## What makes it different

- Multi-report health timeline instead of single-report summarization
- Agentic pipeline with extraction, trend analysis, explanation, safety, and question generation
- Can use vision/OCR for scanned reports and PDFs
- AMD Developer Cloud compatible through OpenAI-compatible vLLM endpoint

## Agent workflow

1. Document Reader Agent
2. Lab Value Extraction Agent
3. Trend Analysis Agent
4. Explanation Agent
5. Safety Agent
6. Doctor Question Agent

## Hackathon track fit

- AI Agents & Agentic Workflows
- Vision & Multimodal AI
- Hugging Face Spaces demo
- AMD Developer Cloud / ROCm / MI300X compatible inference

## Demo script

1. Upload two sample reports.
2. Show extracted values table.
3. Show trend summary.
4. Show abnormal/latest values to review.
5. Show doctor questions.
6. Explain AMD deployment option using vLLM on AMD Developer Cloud.

## Future roadmap

- Better OCR using multimodal models
- Fine-tuned medical entity extraction model
- Timeline charts
- Patient-friendly and doctor-facing export PDF
- Multi-language explanations
