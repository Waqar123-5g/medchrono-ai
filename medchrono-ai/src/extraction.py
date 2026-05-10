from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import pytesseract


def extract_text_from_pdf_or_image(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)

    if suffix in [".png", ".jpg", ".jpeg"]:
        return pytesseract.image_to_string(Image.open(file_path))

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    return ""


def extract_text_from_pdf(file_path: str) -> str:
    text_parts = []

    doc = fitz.open(file_path)

    for page in doc:
        text = page.get_text("text")
        if text.strip():
            text_parts.append(text)

    doc.close()

    return "\n".join(text_parts).strip()