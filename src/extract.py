"""PDF text extraction with an OCR fallback for scanned pages.

We use PyMuPDF (fitz) for digital text. If a page yields almost no text it is
probably a scan (an image), so we render it to an image and run Tesseract OCR.
This keeps the scanned fraud-dispute policy in the corpus instead of silently
dropping it.
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# Below this many characters a page is treated as "no real text" -> OCR it.
_MIN_CHARS_FOR_TEXT_PAGE = 40
# Render scale for OCR. 300 DPI / 72 default = ~4.16; 3x is a good speed/quality balance.
_OCR_ZOOM = 3.0


def _ocr_page(page: "fitz.Page") -> str:
    pix = page.get_pixmap(matrix=fitz.Matrix(_OCR_ZOOM, _OCR_ZOOM))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """Return one record per page: {page: int, text: str, ocr: bool}."""
    pdf_path = Path(pdf_path)
    pages: list[dict] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            used_ocr = False
            if len(text) < _MIN_CHARS_FOR_TEXT_PAGE:
                ocr_text = _ocr_page(page).strip()
                if len(ocr_text) > len(text):
                    text, used_ocr = ocr_text, True
            if text:
                pages.append({"page": i, "text": text, "ocr": used_ocr})
    return pages
