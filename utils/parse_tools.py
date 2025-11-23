import os
from typing import List, Dict, Any
import PyPDF2
import docx
import pdfplumber
from PIL import Image
import pytesseract
import camelot
import pandas as pd
from .layout_parse import layout_aware_parse

def parse_editable_pdf(file_path: str) -> str:
    """解析可编辑PDF文件"""
    text = ""
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"解析PDF时出错: {e}")
        text = ""
    return text

def parse_scanned_pdf(file_path: str) -> str:
    """解析扫描版PDF文件"""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"解析扫描版PDF时出错: {e}")
        text = ""
    return text

def parse_word(file_path: str) -> str:
    """解析Word文件"""
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"解析Word文件时出错: {e}")
        text = ""
    return text

def parse_image(file_path: str) -> str:
    """解析图片文件"""
    text = ""
    try:
        image = Image.open(file_path)
        # 只使用中文识别
        text = pytesseract.image_to_string(image, lang='chi_sim')
    except Exception as e:
        print(f"解析图片时出错: {e}")
        text = ""
    return text

def parse_excel(file_path: str) -> str:
    """解析Excel文件"""
    text = ""
    try:
        # 使用camelot提取表格
        tables = camelot.read_pdf(file_path, pages='all')
        for table in tables:
            df = table.df
            # 将表格转换为结构化文本
            for index, row in df.iterrows():
                for col in df.columns:
                    cell_value = str(row[col])
                    if cell_value and cell_value != 'nan':
                        text += f"{df.columns[col]}: {cell_value}\n"
    except Exception as e:
        print(f"解析Excel文件时出错: {e}")
        # 尝试使用pandas读取
        try:
            df = pd.read_excel(file_path)
            text = df.to_string()
        except Exception as e2:
            print(f"使用pandas解析Excel时也出错: {e2}")
            text = ""
    return text

def parse_layout_elements(file_path: str) -> Dict[str, Any]:
    """提取布局元素"""
    features = {}
    try:
        if file_path.lower().endswith('.pdf'):
            sorted_text, text_blocks = layout_aware_parse(file_path)
            features["layout_text"] = sorted_text
            features["text_blocks"] = text_blocks
    except Exception as e:
        print(f"提取布局特征时出错: {e}")
    return features

def parse_resume(file_path: str) -> Dict[str, Any]:
    """根据文件类型选择合适的解析方法"""
    if not os.path.exists(file_path):
        return {"text": "", "features": {}}
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        # 尝试判断是可编辑PDF还是扫描版PDF
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                # 如果能提取到文本，则认为是可编辑PDF
                if pdf_reader.pages[0].extract_text().strip():
                    text = parse_editable_pdf(file_path)
                else:
                    text = parse_scanned_pdf(file_path)
        except:
            text = parse_scanned_pdf(file_path)
    elif file_ext == '.docx':
        text = parse_word(file_path)
    elif file_ext in ['.jpg', '.jpeg', '.png']:
        text = parse_image(file_path)
    elif file_ext == '.xlsx':
        text = parse_excel(file_path)
    else:
        # 默认按文本文件处理
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except:
            text = ""
    
    # 提取布局特征
    features = parse_layout_elements(file_path)
    
    return {
        "text": text,
        "features": features
    }