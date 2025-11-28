import re


def clean_text(text: str) -> str:
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s,.!?;:()，。！？；：（）]', '', text
                  or '')
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def remove_special_chars(text: str) -> str:
    """移除文本中的特殊字符，保留中文、英文、数字和基本标点符号"""
    return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s,.!?;:()，。！？；：（）]', '', text
                  or '')


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', (text or '')).strip()
