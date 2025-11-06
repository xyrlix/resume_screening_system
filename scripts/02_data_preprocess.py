import os
import re
import json
import PyPDF2
from docx import Document

RAW_RESUMES_DIR = "data/raw_resumes"
PROCESSED_DATA_DIR = "data/processed"

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def parse_pdf(pdf_path):
    """解析PDF文件，提取文本内容"""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        return text
    except Exception as e:
        print(f"解析PDF文件失败: {pdf_path}, 原因: {e}")
        return ""

def parse_word(docx_path):
    """解析Word文件，提取文本内容"""
    try:
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"解析Word文件失败: {docx_path}, 原因: {e}")
        return ""

def clean_resume_text(text):
    """清洗简历文本，去除噪音"""
    # 去除特殊符号和多余的空格，保留中文、英文、数字和基本标点
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s,.!?;:()，。！？；：（）]', '', text)
    # 将多个换行符替换为单个
    text = re.sub(r'\n+', '\n', text)
    # 去除首尾的空白字符
    text = text.strip()
    return text

def process_resumes():
    """批量处理所有原始简历"""
    processed_resumes = []
    for filename in os.listdir(RAW_RESUMES_DIR):
        file_path = os.path.join(RAW_RESUMES_DIR, filename)
        text = ""
        if filename.endswith(".pdf"):
            text = parse_pdf(file_path)
        elif filename.endswith(".docx") or filename.endswith(".doc"):
            text = parse_word(file_path)
        
        if text:
            clean_text = clean_resume_text(text)
            # 准备用于实体标注的格式
            processed_data = {
                "text": clean_text,
                "entities": [] # 初始为空，待标注
            }
            processed_resumes.append(processed_data)

    # 将处理后的数据保存为JSON文件，用于后续的标注
    output_path = os.path.join(PROCESSED_DATA_DIR, "resumes_for_annotation.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_resumes, f, ensure_ascii=False, indent=4)
    
    print(f"成功处理 {len(processed_resumes)} 份简历。")
    print(f"预处理后的数据已保存到: {output_path}")

if __name__ == "__main__":
    process_resumes()