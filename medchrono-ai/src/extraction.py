import pytesseract
from PIL import Image, ImageEnhance
import pdfplumber
import cv2
import numpy as np
from pathlib import Path


# Function for extracting text from image-based or text-based PDFs
def extract_text_from_pdf_or_image(file_path: str) -> str:
    # First, attempt to extract text from the PDF (if it's text-based)
    text = extract_text_from_pdf(file_path)
    if not text:
        # If no text is extracted, fallback to OCR for image-based PDFs
        text = extract_text_from_image(file_path)
    return text


# Extract text from a text-based PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text if text.strip() else ""
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return ""


# OCR processing for image-based PDFs or other images
def extract_text_from_image(image_path: str) -> str:
    try:
        # Open the image using PIL
        img = Image.open(image_path)

        # Convert image to grayscale
        img_gray = img.convert("L")

        # Enhance contrast for better text extraction
        enhancer = ImageEnhance.Contrast(img_gray)
        img_enhanced = enhancer.enhance(2.0)

        # Convert image back to OpenCV format for thresholding
        img_cv = np.array(img_enhanced)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # Thresholding (binary image) to enhance text extraction
        _, img_threshold = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY)

        # Run OCR to extract text
        text = pytesseract.image_to_string(img_threshold)
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
    return ""