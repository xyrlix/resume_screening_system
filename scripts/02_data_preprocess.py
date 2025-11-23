import os
import re
import json
import PyPDF2
from docx import Document
import sys
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
from utils.clean_tools import clean_text as _clean_text_util
from modules.debias import mask_sensitive, collect_fairness_tags
from utils.logger import get_logger
from utils.lang_tools import detect_language

RAW_RESUMES_DIR = "data/raw_resumes"
PROCESSED_DATA_DIR = "data/processed"

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def parse_pdf(pdf_path):
    try:
        # 使用utils.parse_tools中的parse_resume函数
        from utils.parse_tools import parse_resume
        result = parse_resume(pdf_path)
        return result["text"]
    except Exception as e:
        print(f"解析PDF文件失败: {pdf_path}, 原因: {e}")
        return ""

def parse_word(docx_path):
    try:
        # 使用utils.parse_tools中的parse_resume函数
        from utils.parse_tools import parse_resume
        result = parse_resume(docx_path)
        return result["text"]
    except Exception as e:
        print(f"解析Word文件失败: {docx_path}, 原因: {e}")
        return ""

def parse_image(image_path):
    try:
        # 使用utils.parse_tools中的parse_resume函数
        from utils.parse_tools import parse_resume
        result = parse_resume(image_path)
        return result["text"]
    except Exception as e:
        print(f"解析图片失败: {image_path}, 原因: {e}")
        return ""

def parse_excel(xlsx_path):
    try:
        # 使用utils.parse_tools中的parse_resume函数
        from utils.parse_tools import parse_resume
        result = parse_resume(xlsx_path)
        return result["text"]
    except Exception as e:
        print(f"解析Excel失败: {xlsx_path}, 原因: {e}")
        return ""

def clean_resume_text(text):
    return _clean_text_util(text)

def process_resumes():
    logger = get_logger("preprocess")
    processed_resumes = []
    for filename in os.listdir(RAW_RESUMES_DIR):
        file_path = os.path.join(RAW_RESUMES_DIR, filename)
        low = filename.lower()
        text = ""
        if low.endswith(".pdf"):
            text = parse_pdf(file_path)
        elif low.endswith(".docx") or low.endswith(".doc"):
            text = parse_word(file_path)
        elif low.endswith(('.txt', '.md')):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            except Exception:
                text = ""
        elif low.endswith(('.jpg', '.jpeg', '.png')):
            text = parse_image(file_path)
        elif low.endswith('.xlsx'):
            text = parse_excel(file_path)
        if text:
            clean_text = clean_resume_text(text)
            lang = detect_language(clean_text)
            
            # 只处理中文简历
            if lang != "zh":
                logger.info(f"skipped {filename} lang={lang} (not Chinese)")
                continue
                
            masked = mask_sensitive(clean_text)
            fairness = collect_fairness_tags(clean_text)
            processed_data = {
                "text": clean_text,
                "language": lang,
                "masked_text": masked,
                "fairness_tags": fairness,
                "entities": []
            }
            processed_resumes.append(processed_data)
            logger.info(f"processed {filename} len={len(clean_text)} lang={lang}")
    output_path = os.path.join(PROCESSED_DATA_DIR, "resumes_for_annotation.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_resumes, f, ensure_ascii=False, indent=4)
    logger.info(f"count={len(processed_resumes)} output={output_path}")

if __name__ == "__main__":
    process_resumes()