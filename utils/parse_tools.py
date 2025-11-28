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
        # 尝试使用pytesseract进行OCR
        try:
            image = Image.open(file_path)
            
            # 1. 首先尝试从配置文件读取tesseract路径
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'tesseract.json')
            tesseract_path = None
            
            if os.path.exists(config_path):
                try:
                    import json
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        tesseract_path = config.get('tesseract_path')
                    if tesseract_path and os.path.exists(tesseract_path):
                        pytesseract.pytesseract.tesseract_cmd = tesseract_path
                        print(f"从配置文件使用tesseract路径: {tesseract_path}")
                except Exception as config_e:
                    print(f"读取tesseract配置文件失败: {config_e}")
            
            # 2. 其次尝试从环境变量获取tesseract路径
            if not tesseract_path:
                tesseract_path = os.environ.get('TESSERACT_PATH')
                if tesseract_path and os.path.exists(tesseract_path):
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
                    print(f"从环境变量使用tesseract路径: {tesseract_path}")
            
            # 3. 尝试直接使用tesseract
            try:
                text = pytesseract.image_to_string(image, lang='chi_sim')
            except pytesseract.TesseractNotFoundError:
                # 4. 如果找不到tesseract，尝试检查常见安装路径（Windows系统）
                possible_paths = [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
                ]
                
                found_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        found_path = path
                        break
                
                if found_path:
                    # 设置tesseract路径并再次尝试
                    pytesseract.pytesseract.tesseract_cmd = found_path
                    try:
                        text = pytesseract.image_to_string(image, lang='chi_sim')
                        print(f"成功使用tesseract路径: {found_path}")
                    except Exception as inner_e:
                        print(f"tesseract已找到但执行失败: {inner_e}")
                        raise
                else:
                    # 如果仍未找到，提供详细的错误信息和解决方案
                    print(f"解析图片时注意: tesseract已安装但无法找到可执行文件")
                    print(f"解决方案:")
                    print(f"1. 确保已正确安装tesseract (https://github.com/UB-Mannheim/tesseract/wiki)")
                    print(f"2. 将tesseract添加到系统环境变量PATH")
                    print(f"3. 设置环境变量 TESSERACT_PATH 指向tesseract.exe的完整路径")
                    print(f"4. 创建配置文件 config/tesseract.json，内容格式: {{\"tesseract_path\": \"C:\\path\\to\\tesseract.exe\"}}")
                    image = Image.open(file_path)
                    text = f"[图片文件 - 尺寸: {image.size[0]}x{image.size[1]} 像素]"
        except ImportError:
            print(f"解析图片时注意: pytesseract Python包未安装")
            image = Image.open(file_path)
            text = f"[图片文件 - 尺寸: {image.size[0]}x{image.size[1]} 像素]"
    except Exception as e:
        print(f"解析图片时出错: {e}")
        text = "[图片文件 - 解析失败]"
    return text

def parse_excel(file_path: str) -> str:
    """解析Excel文件"""
    text = ""
    try:
        # 直接使用pandas读取Excel文件，这是正确的方法
        df = pd.read_excel(file_path)
        # 转换为结构化文本，包含表头信息
        text += "Excel表格内容:\n"
        # 先添加列名信息
        text += "列名: " + ", ".join(df.columns.tolist()) + "\n\n"
        # 添加表格数据，限制行数避免文本过长
        max_rows = 50  # 最多显示50行
        display_df = df.head(max_rows)
        
        # 逐行转换为结构化文本
        for index, row in display_df.iterrows():
            text += f"--- 第{index+1}行 ---\n"
            for col in df.columns:
                cell_value = str(row[col])
                if cell_value and cell_value != 'nan' and cell_value.strip():
                    text += f"{col}: {cell_value}\n"
        
        # 如果行数超过限制，提示用户
        if len(df) > max_rows:
            text += f"... 还有{len(df) - max_rows}行未显示"
            
    except Exception as e:
        print(f"解析Excel文件时出错: {e}")
        # 提供基本信息
        try:
            # 即使解析失败，也尝试获取文件名等基本信息
            file_name = os.path.basename(file_path)
            text = f"[Excel文件: {file_name}]"
        except:
            text = "[Excel文件 - 解析失败]"
    
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
    
    return {"text": text, "features": features}

def parse_job_description(text: str) -> Dict[str, Any]:
    """
    解析岗位描述文本，提取结构化信息
    
    Args:
        text: 岗位描述文本
    
    Returns:
        包含结构化信息的字典
    """
    # 使用现有函数提取基本信息
    basic_info = job_profile_from_text(text)

    # 使用实体提取器提取更多结构化信息
    entities = extract_entities(text)
    structured_info = extract_structured_info(text, info_type="jd")

    # 按实体类型分组
    entities_by_type = {}
    for entity in entities:
        if entity["type"] not in entities_by_type:
            entities_by_type[entity["type"]] = []
        entities_by_type[entity["type"]].append(entity["text"])

    # 提取岗位职责和要求
    duties = []
    requirements = []
    qualifications = []
    benefits = []

    # 按段落分析内容
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # 关键词标识不同部分
    duty_keywords = ["负责", "参与", "搭建", "维护", "优化", "推进", "开发", "设计", "管理"]
    req_keywords = [
        "熟悉", "精通", "掌握", "具备", "了解", "至少", "本科", "硕士", "博士", "年经验"
    ]
    qual_keywords = ["资质", "经验", "要求", "条件", "背景", "能力"]
    benefit_keywords = ["福利", "奖金", "待遇", "薪资", "工作地点", "工作时间", "绩效"]

    # 遍历段落，根据关键词分类
    for para in paragraphs:
        # 检查是否是标题
        if para.endswith('：') or para.endswith(':'):
            title = para[:-1].strip()
            # 根据标题分类
            if any(keyword in title.lower()
                   for keyword in ["职责", "工作内容", "工作描述"]):
                continue  # 跳过标题，处理内容
            elif any(keyword in title.lower()
                     for keyword in ["要求", "条件", "资质", "任职"]):
                continue  # 跳过标题，处理内容
            elif any(keyword in title.lower()
                     for keyword in ["福利", "待遇", "薪资", "地点"]):
                benefits.append(para)
                continue

        # 根据内容关键词分类
        if any(keyword in para for keyword in duty_keywords):
            duties.append(para)
        elif any(keyword in para for keyword in req_keywords):
            requirements.append(para)
        elif any(keyword in para for keyword in qual_keywords):
            qualifications.append(para)
        elif any(keyword in para for keyword in benefit_keywords):
            benefits.append(para)

    # 提取岗位名称
    job_title = ""
    title_matches = []
    for line in text.split('\n'):
        line = line.strip()
        if line and len(line) < 50:  # 岗位名称通常较短
            if "招聘" in line or "职位" in line or "岗位" in line:
                # 提取"招聘职位："后的内容
                if "：" in line:
                    title_matches.append(line.split("：")[1].strip())
                elif ":" in line:
                    title_matches.append(line.split(":")[1].strip())
            else:
                # 可能直接是岗位名称
                title_matches.append(line)

    if title_matches:
        job_title = title_matches[0]

    # 提取工作地点
    work_location = ""
    location_matches = []
    for line in text.split('\n'):
        line = line.strip()
        if "工作地点" in line or "地点" in line:
            if "：" in line:
                location_matches.append(line.split("：")[1].strip())
            elif ":" in line:
                location_matches.append(line.split(":")[1].strip())

    if location_matches:
        work_location = location_matches[0]

    # 提取薪资信息
    salary = ""
    salary_matches = []
    for line in text.split('\n'):
        line = line.strip()
        if "薪资" in line or "待遇" in line or "奖金" in line:
            salary_matches.append(line)

    if salary_matches:
        salary = salary_matches[0]

    # 整合所有信息
    result = {
        "基本信息": {
            "岗位名称": job_title,
            "工作地点": work_location,
            "薪资待遇": salary
        },
        "岗位职责": duties,
        "岗位要求": requirements,
        "任职资格": qualifications,
        "福利待遇": benefits,
        "技能要求": basic_info.get("skills", []),
        "领域关键词": basic_info.get("keywords", []),
        "证书要求": basic_info.get("certs", []),
        "语言要求": basic_info.get("languages", []),
        "经验要求": {
            "工作年限": f"{basic_info.get('years', 0)}年",
            "学历要求": basic_info.get('degree', '')
        },
        "提取的实体": entities_by_type,
        "原始文本": text
    }

    return result
