import os
from typing import List
import PyPDF2
from docx import Document
import pandas as pd
try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None
try:
    from utils.layout_parse import layout_aware_parse
except Exception:
    layout_aware_parse = None

def parse_pdf(pdf_path: str) -> str:
    if layout_aware_parse is not None:
        try:
            sorted_text, _ = layout_aware_parse(pdf_path)
            if sorted_text:
                return sorted_text
        except Exception:
            pass
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t
            return text
    except Exception:
        return ""

def parse_word(docx_path: str) -> str:
    try:
        doc = Document(docx_path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

def parse_image(img_path: str) -> str:
    if PaddleOCR is None:
        return ""
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang="ch")
        res = ocr.ocr(img_path, cls=True)
        lines: List[str] = []
        for page in res or []:
            for item in page or []:
                txt = item[1][0]
                if txt:
                    lines.append(str(txt))
        if not lines and PaddleOCR is not None:
            try:
                ocr_en = PaddleOCR(use_angle_cls=True, lang="en")
                res2 = ocr_en.ocr(img_path, cls=True)
                for page in res2 or []:
                    for item in page or []:
                        txt = item[1][0]
                        if txt:
                            lines.append(str(txt))
            except Exception:
                pass
        return "\n".join(lines)
    except Exception:
        return ""

def parse_excel(xlsx_path: str) -> str:
    try:
        sheets = pd.read_excel(xlsx_path, sheet_name=None, dtype=str)
        lines: List[str] = []
        for _, df in sheets.items():
            for row in df.fillna("").astype(str).values.tolist():
                lines.append(" ".join(row))
        return "\n".join(lines)
    except Exception:
        return ""