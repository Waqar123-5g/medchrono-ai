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


def extract_lab_values(text: str) -> list[dict]:
    results = []

    report_date_match = re.search(
        r"Report Date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
        text,
        re.IGNORECASE,
    )
    report_date = report_date_match.group(1) if report_date_match else ""

    pattern = re.compile(
        r"Test Name:\s*(?P<test_name>.+?)\s*"
        r"Value:\s*(?P<value>[0-9.]+)\s*"
        r"Unit:\s*(?P<unit>[^\n]+)\s*"
        r"Reference Range:\s*(?P<ref_low>[0-9.]+)\s*-\s*(?P<ref_high>[0-9.]+)",
        re.IGNORECASE | re.DOTALL,
    )

    for match in pattern.finditer(text):
        value = float(match.group("value"))
        ref_low = float(match.group("ref_low"))
        ref_high = float(match.group("ref_high"))

        if value < ref_low:
            status = "low"
        elif value > ref_high:
            status = "high"
        else:
            status = "normal"

        results.append(
            {
                "report_date": report_date,
                "test_name": match.group("test_name").strip(),
                "value": value,
                "unit": match.group("unit").strip(),
                "ref_low": ref_low,
                "ref_high": ref_high,
                "status": status,
            }
        )

    return results