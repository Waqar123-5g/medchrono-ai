from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
from PIL import Image
import pytesseract


def extract_text_from_file(file_path: str | Path) -> str:
    """Extract text from txt, pdf, or image files.

    This keeps the hackathon prototype simple. For production, replace this
    with a stronger document parser such as LlamaParse, Unstructured, or a
    multimodal model.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md", ".csv"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        text_parts: list[str] = []
        with fitz.open(path) as doc:
            for page in doc:
                text_parts.append(page.get_text("text"))
        return "\n".join(text_parts).strip()

    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        image = Image.open(path)
        return pytesseract.image_to_string(image)

    return ""


def guess_report_date(text: str, fallback_index: int) -> str:
    patterns = [
        r"Report Date\s*:\s*(\d{4}-\d{2}-\d{2})",
        r"Date\s*:\s*(\d{4}-\d{2}-\d{2})",
        r"(\d{4}-\d{2}-\d{2})",
        r"(\d{1,2}/\d{1,2}/\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return f"Report {fallback_index}"


def combine_uploaded_texts(files: Iterable[str | Path]) -> list[dict]:
    reports = []
    for index, file_path in enumerate(files, start=1):
        text = extract_text_from_file(file_path)
        reports.append(
            {
                "file_name": Path(file_path).name,
                "report_date": guess_report_date(text, index),
                "raw_text": text,
            }
        )
    return reports

import re

def extract_lab_values(report_text: str) -> list[dict]:
    """
    Extract lab values from the report text.
    This function assumes the text is in a format like:
    'Hemoglobin: 12.1 g/dL Reference Range: 13.5 - 17.5'
    """
    lab_values = []
    
    # Regex pattern to match test names and values
    pattern = r"(?P<test_name>[\w\s]+):\s*(?P<value>[\d.]+)\s*(?P<unit>[a-zA-Z/]+)\s*Reference Range:\s*(?P<ref_low>[\d.]+)\s*-\s*(?P<ref_high>[\d.]+)"
    
    # Find all matches in the text
    matches = re.finditer(pattern, report_text)

    for match in matches:
        lab_values.append({
            "test_name": match.group("test_name"),
            "value": match.group("value"),
            "unit": match.group("unit"),
            "ref_low": match.group("ref_low"),
            "ref_high": match.group("ref_high"),
            "status": "low" if float(match.group("value")) < float(match.group("ref_low")) else "normal"
        })
    
    return lab_values